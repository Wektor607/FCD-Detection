import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meld_graph")))

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from meld_graph.meld_cohort import MeldCohort
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import languidemedseg_meld.utils.config as config
from languidemedseg_meld.engine.wrapper import LanGuideMedSegWrapper
from languidemedseg_meld.utils.data import SingleEpilepSample
from languidemedseg_meld.utils.utils import (convert_preds_to_nifti,
                                             get_device, move_to_device,
                                             summarize_clusters,
                                             worker_init_fn)

torch.multiprocessing.set_sharing_strategy("file_system")


SEED = 42
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def get_cfg(argv=[]):
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument(
        "--config", default="./meld_graph/languidemedseg_meld/config/training.yaml", type=str, help="config file"
    )

    cli = parser.parse_args(argv)

    cfg = config.load_cfg_from_cfg_file(cli.config)
    return cfg


def create_inference_loader(subject_data: dict, description: str, tokenizer, cohort: MeldCohort) -> DataLoader:
    """
    Prepare the Dataset and DataLoader for inference.

    Returns (dataset, dataloader).
    """
    ds_inference = SingleEpilepSample(
        data=subject_data,
        description=description,
        tokenizer=tokenizer,
        cohort=cohort,
        max_length=256,
    )

    dl_inference = DataLoader(
        ds_inference,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    return dl_inference


def load_ensemble_models(ckpt_prefix: str, args, eva, exp_flags, device: torch.device) -> List[torch.nn.Module]:
    save_dir = Path("meld_graph") / "languidemedseg_meld" / "save_model"
    ckpt_paths = [save_dir / f"{ckpt_prefix}_fold{i+1}.ckpt" for i in range(0, 5)]
    # ckpt_paths = [save_dir / f"{ckpt_prefix}_fold{i+1}.ckpt" for i in range(0, 3)]
    att_mechanism = False
    text_emb = False
    for exp, flags in exp_flags.items():
        if exp in (ckpt_prefix or ""):
            att_mechanism = flags.get("self_att_mechanism", False)
            text_emb = flags.get("text_emb", False)
            print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
            break
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True) if text_emb else None
    print(f"[INFO] Using ensemble of {len(ckpt_paths)} models:", ckpt_paths)
    models = []
    for i, ckpt_path in enumerate(ckpt_paths):
        model = LanGuideMedSegWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            args=args,
            eva=eva,
            fold_number=i,
            att_mechanism=att_mechanism,
            text_emb=text_emb,
            mode="inference",
        )
        model.eval()
        model.to(device)
        models.append(model)

    return models, tokenizer

def run_ensemble_inference(dl_inference: DataLoader, models: List[torch.nn.Module], device: torch.device, cohort: MeldCohort = None):
    cortex_mask = torch.from_numpy(cohort.cortex_mask).to(device)

    all_subject_ids = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dl_inference, desc="Ensemble inference"):
            subject_ids = batch["subject_id"]
            batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            text = batch.get("text", batch_on_device.get("text"))

            # collect model predictions for this batch
            B = len(subject_ids)
            H = 2
            model_probs = []
            for model in models:
                text_on_device = move_to_device(text, device)

                outputs = model([subject_ids, text_on_device])  # [B * H * V7, 2]
                # move logits to device early and reshape safely (reshape handles non-contiguous tensors)
                logp = outputs["log_softmax"].to(device)

                # infer the V7 dimension automatically using -1 and reorder to (B, 2, H, V7)
                logp = logp.reshape(B, H, -1, 2).permute(0, 3, 1, 2)

                # select cortex vertices and collapse dims to (B, 2, H * n_cortex)
                logp = logp[..., cortex_mask].reshape(B, 2, -1)

                # get positive-class probabilities and restore (B, H, n_cortex)
                probs = logp[:, 1, :].exp()
                pprobs = probs.view(B, H, -1).contiguous().detach()
                model_probs.append(pprobs)

            probs_stack = torch.stack(model_probs, dim=0)
            probs_mean = probs_stack.mean(dim=0)

            all_subject_ids.extend(subject_ids)
            all_probs.append(probs_mean.detach())

    all_probs = torch.cat(all_probs, dim=0)

    return all_subject_ids, all_probs

# TODO: adopt for list outputs
def postprocess_and_save(all_subject_ids, all_probs, eva, cohort: MeldCohort, model_type: str):
    H = 2 # number of hemispheres
    for i in range(all_probs.shape[0]):
        sid = all_subject_ids[i]
        pred = all_probs[i]

        pred_np = torch.cat([pred[0], pred[1]], dim=0).detach().cpu().numpy().astype("float32")
        mini = {sid: {"result": pred_np}}
        out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
        probs_flat = out[sid]["cluster_thresholded"]

        # Собираем информацию о кластерах
        probs_2h = probs_flat.reshape(H, -1)
        lh_probs = probs_2h[0]
        rh_probs = probs_2h[1]
        clusters_summary = summarize_clusters([lh_probs, rh_probs])

        if clusters_summary:
            hemis = sorted(set(cl['hemi'] for cl in clusters_summary))
            hemi_str = " and ".join(hemis)
            result_text = f"Suspicious lesion detected in {hemi_str} hemisphere(s)."
            epilepsy_flag = 1
        else:
            result_text = "No suspicious lesion detected."
            epilepsy_flag = 0

        final_nii = convert_preds_to_nifti(
            model_type, [sid], [[lh_probs, rh_probs]], cohort, 'inference'
        )

    return str(final_nii), {"clusters": clusters_summary,
                            "report": result_text,
                            "epilepsy": epilepsy_flag}


def process_meld_mode(dl_inference: DataLoader, subject_data: dict, cohort: MeldCohort, model_type: str):
    """
    Process subjects when model_type == 'MELD'.

    Returns (final_nii, epi_dict)
    """
    H = 2  # number of hemispheres
    final_nii = None
    clusters_summary = None
    result_text = "No suspicious lesion detected."
    epilepsy_flag = 0

    for batch in tqdm(dl_inference):
        subject_ids = batch["subject_id"]  # list[str]
        for b, sid in enumerate(subject_ids):
            preds = subject_data[sid]["result"].astype("float32")
            mini = {sid: {"result": preds.copy()}}
            out = cohort.eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False) if hasattr(cohort, 'eva') else None
            # If eva isn't attached to cohort, fallback to using passed thresholds via utils (original behavior expects eva variable)
            if out is None:
                # try using summarize_clusters directly on preds
                probs_flat = preds
            else:
                probs_flat = out[sid]["cluster_thresholded"]

            # Собираем информацию о кластерах
            probs_2h = probs_flat.reshape(H, -1)
            lh_probs = probs_2h[0]
            rh_probs = probs_2h[1]
            clusters_summary = summarize_clusters([lh_probs, rh_probs])

            if clusters_summary:
                hemis = sorted(set(cl['hemi'] for cl in clusters_summary))
                hemi_str = " and ".join(hemis)
                result_text = f"Suspicious lesion detected in {hemi_str} hemisphere(s)."
                epilepsy_flag = 1
            else:
                result_text = "No suspicious lesion detected."
                epilepsy_flag = 0

            final_nii = convert_preds_to_nifti(
                model_type, [sid], [[lh_probs, rh_probs]], cohort, 'inference'
            )

    epi_dict = {"clusters": clusters_summary, "report": result_text, "epilepsy": epilepsy_flag}
    return str(final_nii), epi_dict


def inference(subject_data, description, model_type):
    args = get_cfg()
    eva, cohort, exp_flags = config.inference_config()

    att_mechanism = False
    text_emb = False
    for exp, flags in exp_flags.items():
        if exp in (model_type or ""):
            att_mechanism = flags.get("self_att_mechanism", False)
            text_emb = flags.get("text_emb", False)
            print(f"Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
            break

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True) if text_emb else None

    # create dataset and dataloader
    dl_inference = create_inference_loader(subject_data, description, tokenizer, cohort, batch_size=1)

    # MELD has a simpler postprocessing flow
    if model_type == "MELD":
        return process_meld_mode(dl_inference, subject_data, cohort, model_type)

    # ensemble inference for other model types
    device = get_device()
    models, tokenizer = load_ensemble_models(model_type, args, eva, exp_flags, device)
    all_subject_ids, all_probs = run_ensemble_inference(dl_inference, models, device, cohort)
    final_predictions, epi_dict = postprocess_and_save(all_subject_ids, all_probs, eva, cohort, model_type)

    return final_predictions, epi_dict