import os
import sys
from typing import List, Tuple

from tqdm import tqdm
import torch
import argparse
import random
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import pytorch_lightning as pl

import torch.multiprocessing
from utils.utils import summarize_ci, get_device, worker_init_fn, move_to_device

from meld_graph.meld_cohort import MeldCohort
from transformers import AutoTokenizer

from utils.data import EpilepDataset
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
from engine.wrapper import LanGuideMedSegWrapper
import utils.config as config

# Keep reproducibility settings at top
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

# Multiprocessing strategy for dataloaders
torch.multiprocessing.set_sharing_strategy("file_system")

def get_cfg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument("--config", default="./config/training.yaml", type=str, help="config file")
    parser.add_argument("--meld_check", action="store_true", help="enable MELD test check mode")
    parser.add_argument("--ckpt_prefix", default=None, type=str, help="optional checkpoint prefix to load")
    parser.add_argument("--ckpt_path", default=None, type=str, help="comma-separated list of checkpoints for ensemble")

    cli = parser.parse_args()
    if cli.config is None:
        parser.error("--config is required")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.meld_check = cli.meld_check
    cfg.ckpt_path = cli.ckpt_path
    cfg.ckpt_prefix = cli.ckpt_prefix
    return cfg

def prepare_dataloader(args, tokenizer, cohort) -> DataLoader:
    df = pd.read_csv(args.split_path, sep=",")
    test_ids = df[(df["split"] == "test") & (df["subject_id"].str.contains("FCD"))]["subject_id"].tolist()

    ds_test = EpilepDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        feature_path=args.feature_path,
        subject_ids=test_ids,
        cohort=cohort,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )
    return dl_test


def load_ensemble_models(ckpt_prefix: str, args, eva, exp_flags, device: torch.device) -> List[torch.nn.Module]:
    ckpt_paths = [f"./save_model/{ckpt_prefix}_fold{i+1}.ckpt" for i in range(0, 5)]
    att_mechanism = False
    text_emb = False
    for exp, flags in exp_flags.items():
        if exp in (args.ckpt_prefix or ""):
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
        )
        model.eval()
        model.to(device)
        models.append(model)

    return models, tokenizer


def run_ensemble_inference(dl_test: DataLoader, models: List[torch.nn.Module], eva, device: torch.device, cohort: MeldCohort = None) -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    cortex_mask = torch.from_numpy(cohort.cortex_mask).to(device)

    all_subject_ids = []
    all_labels = []
    all_probs = []
    all_dist_maps = []

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="Ensemble inference"):
            subject_ids = batch["subject_id"]
            batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            y = batch_on_device["roi"]
            text = batch.get("text", batch_on_device.get("text"))

            B, H, V7 = y.shape
            y_mask = y[:, :, cortex_mask]
            target = y_mask.view(B, -1).long().view(B, H, -1)

            dist_maps = batch_on_device["dist_maps"].reshape(B, H, V7)
            dist_maps_cortex = dist_maps[:, :, cortex_mask].view(B, -1)

            # collect model predictions for this batch
            model_probs = []
            for model in models:
                text_on_device = move_to_device(text, device)

                outputs = model([subject_ids, text_on_device])  # [B * H * V7, 2]
                # move logits to device early and reshape safely (reshape handles non-contiguous tensors)
                logp = outputs["log_softmax"].to(device)

                # infer the V7 dimension automatically using -1 and reorder to (B, 2, H, V7)
                logp = logp.reshape(B, H, V7, 2).permute(0, 3, 1, 2)

                # select cortex vertices and collapse dims to (B, 2, H * n_cortex)
                logp = logp[..., cortex_mask].reshape(B, 2, -1)

                # get positive-class probabilities and restore (B, H, n_cortex)
                probs = logp[:, 1, :].exp()
                pprobs = probs.view(B, H, -1).contiguous().detach()
                model_probs.append(pprobs)

            probs_stack = torch.stack(model_probs, dim=0)
            probs_mean = probs_stack.mean(dim=0)

            all_subject_ids.extend(subject_ids)
            all_labels.append(target.cpu())
            all_dist_maps.append(dist_maps_cortex.cpu())
            all_probs.append(probs_mean.detach())

    all_labels = torch.cat(all_labels, dim=0)
    all_dist_maps = torch.cat(all_dist_maps, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    return all_subject_ids, all_labels, all_dist_maps, all_probs


def postprocess_and_save(all_subject_ids, all_labels, all_dist_maps, all_probs, eva, ckpt_prefix: str):
    dice_scores, iou_scores, ppv_scores = [], [], []
    results = []

    for i in range(all_labels.shape[0]):
        sid = all_subject_ids[i]
        pred = all_probs[i]
        tgt = all_labels[i]
        dist_map_subj = all_dist_maps[i]

        pred_np = torch.cat([pred[0], pred[1]], dim=0).detach().cpu().numpy().astype("float32")
        mini = {sid: {"result": pred_np}}
        out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
        probs_flat = out[sid]["cluster_thresholded"]
        boundary_zone = dist_map_subj < 20

        if isinstance(probs_flat, torch.Tensor):
            probs_flat_cpu = probs_flat.detach().cpu().numpy()
        else:
            probs_flat_cpu = probs_flat

        boundary_zone_cpu = boundary_zone.detach().cpu().numpy() if isinstance(boundary_zone, torch.Tensor) else np.array(boundary_zone)

        difference = np.setdiff1d(np.unique(probs_flat_cpu), np.unique(probs_flat_cpu[boundary_zone_cpu]))
        difference = difference[difference > 0]
        n_fp_clusters = len(difference)
        correct_values = np.unique(probs_flat_cpu[boundary_zone_cpu])
        correct_values = correct_values[correct_values > 0]
        n_tp_clusters = len(correct_values)

        gt_flat = tgt.reshape(-1)
        mask_np = (probs_flat_cpu > 0).astype(int)
        mask = torch.from_numpy(mask_np).long().to(gt_flat.device)
        labels = gt_flat.bool().long()

        dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)
        tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
        iou = tp / (tp + fp + fn + 1e-8)
        ppv = tp / (tp + fp + 1e-8)

        dice_scores.append(float(dices[1].detach().cpu()))
        ppv_scores.append(float(ppv))
        iou_scores.append(float(iou))

        print(f"[{sid}] Dice lesional={dices[1]:.3f}, IoU={iou:.3f}, PPV={ppv:.3f}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        results.append({
            "subject_id": sid,
            "number FP clusters": n_fp_clusters,
            "number TP clusters": n_tp_clusters,
            "dice": float(dices[1]),
            "iou": float(iou),
            "ppv_voxel": float(ppv),
        })

    d_med, d_lo, d_hi = summarize_ci(dice_scores)
    p_med, p_lo, p_hi = summarize_ci(ppv_scores)
    i_med, i_lo, i_hi = summarize_ci(iou_scores)
    n_tp_clusters = sum(r["number TP clusters"] for r in results)
    n_fp_clusters = sum(r["number FP clusters"] for r in results)
    ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters + 1e-8)

    tp_clusters_list = [r["number TP clusters"] for r in results]
    total = len(tp_clusters_list)
    found = sum(1 for t in tp_clusters_list if t > 0)
    pct = found / total if total > 0 else 0.0

    print("\n=== ENSEMBLE OVERALL TEST METRICS ===")
    print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
    print(f"PPV_pixels  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
    print(f"PPV_clusters  : {ppv_clusters:.3f}")
    print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
    print(f"Detected {found} / {total} FCDs ({pct:.1%})")

    df = pd.DataFrame(results)
    df.to_csv(f"{ckpt_prefix}_results.csv", index=False)


def main():
    args = get_cfg()
    eva, cohort, exp_flags = config.inference_config()

    device = get_device()
    print("start testing on device:", device)

    models, tokenizer = load_ensemble_models(args.ckpt_prefix, args, eva, exp_flags, device)
    dl_test = prepare_dataloader(args, tokenizer, cohort)

    all_subject_ids, all_labels, all_dist_maps, all_probs = run_ensemble_inference(dl_test, models, eva, device, cohort)

    postprocess_and_save(all_subject_ids, all_labels, all_dist_maps, all_probs, eva, args.ckpt_prefix)


if __name__ == "__main__":
    main()
