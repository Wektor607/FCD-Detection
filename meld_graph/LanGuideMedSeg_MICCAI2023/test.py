import os
import sys
# Ensure repository root is on sys.path so imports like `meld_graph` resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tqdm import tqdm
import torch
import argparse
import random
import numpy as np
import pandas as pd

import torch.multiprocessing
from utils.utils import summarize_ci
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from meld_graph.meld_cohort import MeldCohort
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from utils.data import EpilepDataset
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
from engine.wrapper import LanGuideMedSegWrapper
from meld_graph.paths import MELD_DATA_PATH
import utils.config as config
from utils.utils import convert_preds_to_nifti

# теперь можно вызвать
torch.multiprocessing.set_sharing_strategy("file_system")


SEED = 42
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Полная детерминированность и отключение TF32 (даёт меньший дрейф на Ampere/Ada)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# (опц.) для единообразной матричной точности
# torch.set_float32_matmul_precision("high")


# Если вы используете DataLoader с несколькими воркерами:
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def set_seed(seed: int = SEED):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def make_dl_test(args, tokenizer, cohort):
    ds_test = EpilepDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        feature_path=args.feature_path,
        # subject_ids=[sid for sid in pd.read_csv(args.split_path).query("split=='test'")["subject_id"].tolist() if "FCD" in sid],
        subject_ids=[sid for sid in pd.read_csv(args.split_path).query("split=='test'")["subject_id"].tolist()],
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


def run_meld_check(dl_test, eva, cohort):
    # This preserves the existing MELD check logic but is placed into a helper
    dice_metric = []
    ppv_metric = []
    iou_metric = []
    fp_control_clusters = []
    ppv_clusters_metric = []
    results = []
    all_preds = []

    for batch in tqdm(dl_test):
        subject_ids = batch["subject_id"]  # list[str]

        ######################################################### 
        # target_ids = {
        #     # "MELD_H3_3T_FCD_0018",
        #     # "MELD_H4_15T_FCD_0021",
        #     # "MELD_H6_3T_FCD_0017"
        #     "MELD_H2_3T_FCD_0002"
        # }

        # # если пересечение пустое → пропускаем batch
        # if not (set(subject_ids) & target_ids):
        #     continue
        ######################################################### 
        y = batch["roi"]  # torch.Tensor
        B, H, N = y.shape

        dist_maps = batch["dist_maps"]  # torch.Tensor
        dist_maps = dist_maps.view(B, H, -1)
        dist_maps_cortex = dist_maps[:, :, cohort.cortex_mask]
        dist_maps_cortex = dist_maps_cortex.view(B, -1)

        for b, sid in enumerate(subject_ids):
            gt = y[b]  # [2, N]
            dist_maps_cortex_subj = dist_maps_cortex[b]
            gt_cortex = gt[:, cohort.cortex_mask]  # [2, N_cortex]
            gt_flat = gt_cortex.reshape(-1)  # [2*N_cortex]

            is_control = ("FCD" not in sid) or (gt.sum() == 0)

            nii_path = os.path.join(
                MELD_DATA_PATH,
                f"preprocessed/meld_files/{sid}/features",
                "result.npz",
            )
            with np.load(nii_path, allow_pickle=False) as npz:
                arr = npz["result"].astype("float32")

            mini = {sid: {"result": arr.copy()}}
            out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
            probs_flat = out[sid]["cluster_thresholded"]           # (2*N_cortex,)
            boundary_zone = dist_maps_cortex_subj < 20
            # boundary_zone_np = boundary_zone.cpu().numpy() if hasattr(boundary_zone, "cpu") else np.array(boundary_zone)
            #probs_flat = np.asarray(probs_flat, dtype=np.float32).ravel()
            
            probs = probs_flat.reshape(2, -1)
            final_nii = convert_preds_to_nifti("meld", [sid], [probs], cohort)

            # cluster sizes and distances
            all_ids = np.unique(probs_flat)
            all_ids = all_ids[all_ids > 0]
            # full_cluster_sizes = {int(cid): int((probs_flat == cid).sum()) for cid in all_ids}
            # full_cluster_distance = {int(cid): float(dist_maps_cortex_subj[probs_flat == cid].sum()) for cid in all_ids}

            tp_ids = np.unique(probs_flat[boundary_zone])
            tp_ids = tp_ids[tp_ids > 0]

            difference = np.setdiff1d(np.unique(probs_flat), np.unique(probs_flat[boundary_zone]))
            difference = difference[difference > 0]
            n_fp_clusters = len(difference)
            correct_values = np.unique(probs_flat[boundary_zone])
            correct_values = correct_values[correct_values > 0]
            n_tp_clusters = len(correct_values)

            all_preds.append(torch.from_numpy(probs_flat).view(H, -1).contiguous())

            mask = torch.as_tensor(np.array(probs_flat > 0)).long()
            labels = torch.as_tensor(np.array(gt_flat).astype(bool)).long()
            dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)

            tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
            iou = tp / (tp + fp + fn + 1e-8)
            ppv = tp / (tp + fp + 1e-8)
            ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters + 1e-8)

            if not is_control:
                dice_metric.append(dices[1])
                ppv_metric.append(ppv)
                iou_metric.append(iou)
                if (n_tp_clusters + n_fp_clusters) > 0:
                    ppv_clusters_metric.append(float(ppv_clusters))

                print(f"[{sid}] Dice lesional={dices[1]:.3f}, IoU={iou:.3f}, PPV={ppv:.3f}, PPV_clusters={ppv_clusters:.3f}, "
                    f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

                results.append({
                    "subject_id": sid,
                    "number FP clusters": n_fp_clusters,
                    "number TP clusters": n_tp_clusters,
                    "dice": float(dices[1]),
                    "iou": float(iou),
                    "ppv_voxel": float(ppv),
                    "ppv_clusters": float(ppv_clusters),
                })
            else:
                fp_control_clusters.append(n_fp_clusters)
                results.append({
                    "subject_id": sid,
                    "number FP clusters": n_fp_clusters,
                    "number TP clusters": 0,
                    "dice": None,
                    "iou": None,
                    "ppv_voxel": None,
                    "ppv_clusters": None,
                })
    # save and report
    df = pd.DataFrame(results)
    df.to_csv("meld_results.csv", index=False)
    
    d_med, d_lo, d_hi = summarize_ci(dice_metric)
    p_med, p_lo, p_hi = summarize_ci(ppv_metric)
    i_med, i_lo, i_hi = summarize_ci(iou_metric)
    ppv_clusters_med, ppv_clusters_lo, ppv_clusters_hi = summarize_ci(ppv_clusters_metric)
    
    n_tp_clusters = df["number TP clusters"].sum() if len(df) > 0 else 0
    n_fp_clusters = df["number FP clusters"].sum() if len(df) > 0 else 0
    ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters) if (n_tp_clusters + n_fp_clusters) > 0 else 0.0
    
    # Sensitivity
    tp_clusters_list = [r["number TP clusters"] for r in results if "FCD" in r["subject_id"]]
    total_tp = len(tp_clusters_list)
    found_tp = sum(1 for t in tp_clusters_list if t > 0)
    pct_tp = found_tp / total_tp if total_tp > 0 else 0.0

    # Specificity
    total_ctrl = len(fp_control_clusters)
    no_fp_ctrl = sum(1 for t in fp_control_clusters if t == 0)
    pct_spec = no_fp_ctrl / total_ctrl if total_ctrl > 0 else 0.0

    print("\n=== ENSEMBLE OVERALL TEST METRICS ===")
    print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
    print(f"PPV_pixels  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
    print(f"PPV_clusters_mean  : {ppv_clusters:.3f}")
    print(f"PPV_clusters_median  : {ppv_clusters_med:.3f} (95% CI {ppv_clusters_lo:.3f}-{ppv_clusters_hi:.3f})")
    print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
    print(f"Sensitivity (patients only): {found_tp} / {total_tp} FCDs ({pct_tp:.1%})")
    print(f"Specificity (controls only): {no_fp_ctrl} / {total_ctrl} scans with no FP ({pct_spec:.1%})")


def run_trainer_test(model, dl_test, args, accelerator, devices, ckpt_path=None):
    model.eval()
    if ckpt_path is None:
        filename = args.model_save_filename
    else:
        filename = os.path.splitext(os.path.basename(ckpt_path))[0]

    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=filename,
        monitor="val_dice",
        save_top_k=1,
        mode="max",
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor="val_dice", patience=args.patience, mode="max")

    trainer = pl.Trainer(
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[model_ckpt, early_stopping],
        enable_progress_bar=True,
    )

    test_results = trainer.test(model, dataloaders=dl_test, ckpt_path=ckpt_path, verbose=True)
    print("=== TEST metrics ===")
    print(test_results)


def get_cfg():
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument(
        "--config", default="./config/training.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--meld_check", action="store_true", help="enable MELD test check mode"
    )
    parser.add_argument(
        "--ckpt_prefix", default=None, type=str, help="optional checkpoint prefix to load"
    )
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="comma-separated list of checkpoints for ensemble"
    )

    cli = parser.parse_args()

    if cli.config is None:
        parser.error("--config is required")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.meld_check = cli.meld_check
    cfg.ckpt_path = cli.ckpt_path
    cfg.ckpt_prefix = cli.ckpt_prefix
    return cfg

if __name__ == "__main__":
    # Build config and evaluation/cohort objects
    args = get_cfg()
    eva, cohort, _ = config.inference_config()
    # wandb_logger = WandbLogger(project=args.project_name, log_model=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)

    # Prepare test dataloader once and reuse
    dl_test = make_dl_test(args, tokenizer, cohort)

    # Trainer device setup
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        strategy = "ddp_sharded"
    else:
        accelerator = "cpu"
        args.device = "cpu"
        devices = 1
        strategy = None

    # Branch: MELD check (dataset-level clustering evaluation) or usual model test
    if args.meld_check:
        run_meld_check(dl_test, eva, cohort)
    else:
        ckpt_path = args.ckpt_path
        print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
        if ckpt_path is not None:
            model = LanGuideMedSegWrapper.load_from_checkpoint(checkpoint_path=ckpt_path, args=args, eva=eva, fold_number=0)
        else:
            model = LanGuideMedSegWrapper(args, eva=eva, fold_number=0)

        run_trainer_test(model, dl_test, args, accelerator, devices, ckpt_path=ckpt_path)