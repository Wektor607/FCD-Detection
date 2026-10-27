import os
import sys

# Ensure repository root is on sys.path so imports like `meld_graph` resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
from engine.wrapper import LanGuideMedSegWrapper
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
# from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

import utils.config as config
from compute_meld_distance_maps import DEFAULT_OUT_DIR as DEFAULT_MELD_DIST_DIR
from compute_meld_distance_maps import predictions_file
from meld_graph.paths import MELD_DATA_PATH
from utils.data import EpilepDataset
from utils.utils import convert_preds_to_nifti, move_to_device, summarize_ci

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


def make_dl_test(args, tokenizer, cohort, text_emb=False):
    test_ids = [sid for sid in pd.read_csv(args.split_path).query("split=='test'")["subject_id"].tolist()]
    if getattr(args, "dry_run", 0) > 0:
        test_ids = test_ids[:args.dry_run]
        print(f"[DRY RUN] Using first {args.dry_run} subjects: {test_ids}")
    ds_test = EpilepDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        feature_path=args.feature_path,
        subject_ids=test_ids,
        cohort=cohort,
        max_length=args.max_len,
        text_emb=text_emb,
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


DIST_SOURCES = ("meld", "vlm")


def load_meld_pred_dist(meld_dist_h5, sid):
    """MELD Graph's own predicted distance (non_lesion_logits, dist/300) for the cortex of both hemis."""
    if sid not in meld_dist_h5 or "rh/distance_map" not in meld_dist_h5[sid]:
        raise KeyError(
            f"No MELD distance_map for {sid} in {meld_dist_h5.filename}. "
            f"Run languidemedseg_meld/compute_meld_distance_maps.py for this split first."
        )
    return np.concatenate([meld_dist_h5[sid][h]["distance_map"][:] for h in ("lh", "rh")])


def predict_vlm_dist(models, batch, cortex_mask, device):
    """Ensemble-mean predicted distance of the Vision+LM models, shape [B, H * n_cortex]."""
    subject_ids = batch["subject_id"]
    B, H, N = batch["roi"].shape
    model_pred_dists = []
    for model in models:
        outputs = model([subject_ids, move_to_device(batch["text"], device)])
        if "non_lesion_logits" not in outputs:
            raise RuntimeError("Vision+LM model has no distance head output 'non_lesion_logits'")
        pred_dist = outputs["non_lesion_logits"].reshape(B, H, N)[:, :, cortex_mask].reshape(B, -1)
        model_pred_dists.append(pred_dist.detach())
    return torch.stack(model_pred_dists, 0).mean(0).cpu().numpy()


def collect_meld_data(dl_test, eva, cohort, dist_source, models=None, device=None, meld_dist_h5=None):
    """Run inference + clustering once, return per-subject data for reuse across gating variants.

    dist_source selects the *predicted* distance used for gating; there is no ground-truth fallback:
      "meld" - MELD Graph's own distance head, read from meld_dist_h5 (compute_meld_distance_maps.py)
      "vlm"  - Vision+LM ensemble distance head (models must be given)
    The GT distance maps are only used for the evaluation borderzone (< 20 mm).
    """
    if dist_source not in DIST_SOURCES:
        raise ValueError(f"dist_source must be one of {DIST_SOURCES}, got {dist_source!r}")
    if dist_source == "meld" and meld_dist_h5 is None:
        raise ValueError("dist_source='meld' requires meld_dist_h5")
    if dist_source == "vlm" and (not models or device is None):
        raise ValueError("dist_source='vlm' requires loaded models (--ckpt_prefix) and a device")

    cortex_mask = torch.from_numpy(cohort.cortex_mask)
    if device is not None:
        cortex_mask = cortex_mask.to(device)

    subject_data = []  # list of dicts

    with torch.no_grad():
        for batch in tqdm(dl_test, desc="MELD inference"):
            subject_ids = batch["subject_id"]
            y = batch["roi"]
            B, H, N = y.shape

            dist_maps = batch["dist_maps"].view(B, H, -1)
            dist_maps_cortex = dist_maps[:, :, cohort.cortex_mask].view(B, -1)

            if dist_source == "vlm":
                batch_pred_dists = predict_vlm_dist(models, batch, cortex_mask, device)
            else:
                batch_pred_dists = np.stack([load_meld_pred_dist(meld_dist_h5, sid) for sid in subject_ids])

            gt_dists = (dist_maps_cortex / 300.0).numpy()
            if np.allclose(batch_pred_dists, gt_dists):
                raise RuntimeError("Predicted distances equal the ground-truth distance maps: label leakage")

            for b, sid in enumerate(subject_ids):
                gt = y[b]
                dist_maps_cortex_subj = dist_maps_cortex[b]
                gt_cortex = gt[:, cohort.cortex_mask]
                gt_flat = gt_cortex.reshape(-1)
                is_control = ("FCD" not in sid) or (gt.sum() == 0)

                nii_path = os.path.join(
                    MELD_DATA_PATH,
                    f"input/preprocessed/meld_files/{sid}/features",
                    "result.npz",
                )
                with np.load(nii_path, allow_pickle=False) as npz:
                    arr = npz["result"].astype("float32")

                mini = {sid: {"result": arr.copy()}}
                out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
                probs_flat = out[sid]["cluster_thresholded"]
                if isinstance(probs_flat, torch.Tensor):
                    probs_flat = probs_flat.detach().cpu().numpy()

                boundary_zone = (dist_maps_cortex_subj < 20).numpy() if isinstance(dist_maps_cortex_subj, torch.Tensor) else np.array(dist_maps_cortex_subj < 20)

                subject_data.append({
                    "sid": sid,
                    "H": H,
                    "probs_flat": probs_flat.copy(),
                    "gt_flat": gt_flat,
                    "pred_dist": batch_pred_dists[b],
                    "boundary_zone": boundary_zone,
                    "is_control": is_control,
                })

    return subject_data


def meld_postprocess(subject_data, val_threshold=None, val_aggregation="mean", label="", dist_source=""):
    """Compute metrics over collected subject data, optionally applying distance-based gating."""
    agg_fn = {"mean": np.mean, "median": np.median, "min": np.min}.get(val_aggregation, np.mean)

    dice_metric, ppv_metric, iou_metric, ppv_clusters_metric = [], [], [], []
    fp_control_clusters = []
    results = []

    for s in subject_data:
        sid = s["sid"]
        probs_flat = s["probs_flat"].copy()

        # Distance-based cluster filtering
        if val_threshold is not None:
            cluster_ids = np.unique(probs_flat)
            cluster_ids = cluster_ids[cluster_ids > 0]
            for cid in cluster_ids:
                cmask = probs_flat == cid
                if agg_fn(s["pred_dist"][cmask]) >= val_threshold:
                    probs_flat[cmask] = 0

        boundary_zone = s["boundary_zone"]
        gt_flat = s["gt_flat"]
        is_control = s["is_control"]

        tp_ids = np.unique(probs_flat[boundary_zone])
        tp_ids = tp_ids[tp_ids > 0]
        difference = np.setdiff1d(np.unique(probs_flat), np.unique(probs_flat[boundary_zone]))
        difference = difference[difference > 0]
        n_fp_clusters = len(difference)
        correct_values = np.unique(probs_flat[boundary_zone])
        correct_values = correct_values[correct_values > 0]
        n_tp_clusters = len(correct_values)

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
                "dice": None, "iou": None, "ppv_voxel": None, "ppv_clusters": None,
            })

    df = pd.DataFrame(results)
    suffix = f"{dist_source}dist_{val_aggregation}" if val_threshold is not None else "no_gating"
    df.to_csv(f"meld_results_{suffix}.csv", index=False)

    d_med, d_lo, d_hi = summarize_ci(dice_metric)
    p_med, p_lo, p_hi = summarize_ci(ppv_metric)
    i_med, i_lo, i_hi = summarize_ci(iou_metric)
    ppv_clusters_med, ppv_clusters_lo, ppv_clusters_hi = summarize_ci(ppv_clusters_metric)

    n_tp = df["number TP clusters"].sum() if len(df) > 0 else 0
    n_fp = df["number FP clusters"].sum() if len(df) > 0 else 0
    ppv_clusters_mean = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0

    tp_clusters_list = [r["number TP clusters"] for r in results if "FCD" in r["subject_id"]]
    total_tp = len(tp_clusters_list)
    found_tp = sum(1 for t in tp_clusters_list if t > 0)
    pct_tp = found_tp / total_tp if total_tp > 0 else 0.0

    total_ctrl = len(fp_control_clusters)
    no_fp_ctrl = sum(1 for t in fp_control_clusters if t == 0)
    pct_spec = no_fp_ctrl / total_ctrl if total_ctrl > 0 else 0.0

    fp_fcd_clusters = [r["number FP clusters"] for r in results if "FCD" in r["subject_id"]]
    avg_fp_ctrl = float(np.mean(fp_control_clusters)) if fp_control_clusters else 0.0
    avg_fp_fcd  = float(np.mean(fp_fcd_clusters))     if fp_fcd_clusters     else 0.0

    if val_threshold is not None:
        label = f"{label}, distance source={dist_source}"
    header = f"=== MELD TEST METRICS{' [' + label + ']' if label else ''} ==="
    print(f"\n{header}")
    print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
    print(f"PPV_pixels  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
    print(f"PPV_clusters_mean  : {ppv_clusters_mean:.3f}")
    print(f"PPV_clusters_median  : {ppv_clusters_med:.3f} (95% CI {ppv_clusters_lo:.3f}-{ppv_clusters_hi:.3f})")
    print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
    print(f"Sensitivity (patients only): {found_tp} / {total_tp} FCDs ({pct_tp:.1%})")
    print(f"Specificity (controls only): {no_fp_ctrl} / {total_ctrl} scans with no FP ({pct_spec:.1%})")
    print(f"Avg FP clusters per control : {avg_fp_ctrl:.2f}")
    print(f"Avg FP clusters per FCD     : {avg_fp_fcd:.2f}")


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
    parser.add_argument("--config", default="./config/training.yaml", type=str, help="config file")
    parser.add_argument("--meld_check", action="store_true", help="enable MELD test check mode")
    parser.add_argument("--ckpt_prefix", default=None, type=str, help="optional checkpoint prefix to load")
    parser.add_argument("--ckpt_path", default=None, type=str, help="comma-separated list of checkpoints for ensemble")
    parser.add_argument("--dry_run", default=0, type=int, help="limit to first N subjects for quick testing (0 = disabled)")
    parser.add_argument("--dist_source", default=None, choices=DIST_SOURCES,
                        help="--meld_check: predicted distance used for gating MELD clusters "
                             "(meld = MELD Graph's own distance head, vlm = Vision+LM ensemble from --ckpt_prefix)")
    parser.add_argument("--meld_dist_file", default=None, type=str,
                        help="hdf5 written by compute_meld_distance_maps.py (default: its output path)")

    cli = parser.parse_args()

    if cli.config is None:
        parser.error("--config is required")
    if cli.meld_check and cli.dist_source is None:
        parser.error("--meld_check requires --dist_source {meld,vlm}")
    if cli.dist_source == "vlm" and not cli.ckpt_prefix:
        parser.error("--dist_source vlm requires --ckpt_prefix")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.meld_check = cli.meld_check
    cfg.ckpt_path = cli.ckpt_path
    cfg.ckpt_prefix = cli.ckpt_prefix
    cfg.dry_run = cli.dry_run
    cfg.dist_source = cli.dist_source
    cfg.meld_dist_file = cli.meld_dist_file
    return cfg

if __name__ == "__main__":
    args = get_cfg()
    eva, cohort, exp_flags = config.inference_config()

    # Vision+LM experiments with text_emb need the text both in the dataset and in the model
    att_mechanism, text_emb = False, False
    for exp, flags in exp_flags.items():
        if exp in Path(args.ckpt_prefix or "").name:
            att_mechanism = flags.get("self_att_mechanism", False)
            text_emb = flags.get("text_emb", False)
            print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
            break

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
    print(f"[INFO] atlas_type={args.atlas_type}")

    dl_test = make_dl_test(args, tokenizer, cohort, text_emb=text_emb)

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        strategy = "ddp_sharded"
    else:
        accelerator = "cpu"
        args.device = "cpu"
        devices = 1
        strategy = None

    if args.meld_check:
        models, meld_dist_h5 = None, None
        device = get_device()
        print(f"[INFO] Gating MELD clusters with predicted distance from: {args.dist_source}")
        if args.dist_source == "vlm":
            ckpt_prefix = Path(args.ckpt_prefix)
            ckpt_paths = [ckpt_prefix.parent / f"{ckpt_prefix.name}_fold{i+1}.ckpt" for i in range(5)]
            models = []
            for i, cp in enumerate(ckpt_paths):
                m = LanGuideMedSegWrapper.load_from_checkpoint(
                    checkpoint_path=str(cp), args=args, eva=eva, fold_number=i,
                    att_mechanism=att_mechanism, text_emb=text_emb,
                    weights_only=False,
                )
                m.eval()
                m.to(device)
                models.append(m)
            print(f"[INFO] Loaded {len(models)} models for distance filtering from {args.ckpt_prefix}")
        else:
            meld_dist_path = args.meld_dist_file or predictions_file(DEFAULT_MELD_DIST_DIR)
            if not os.path.isfile(meld_dist_path):
                raise FileNotFoundError(
                    f"{meld_dist_path} not found; run languidemedseg_meld/compute_meld_distance_maps.py first"
                )
            meld_dist_h5 = h5py.File(meld_dist_path, "r")
            print(f"[INFO] MELD distance maps: {meld_dist_path}")

        # Collect all data once (inference + clustering)
        subject_data = collect_meld_data(
            dl_test, eva, cohort, args.dist_source, models=models, device=device, meld_dist_h5=meld_dist_h5,
        )

        # Optimal thresholds found on validation (Youden's J):
        # mean=0.48 (Sens=70.0%, Spec=95.5%), median=0.48 (Sens=69.6%, Spec=95.5%), min=0.16 (Sens=67.1%, Spec=94.4%)
        AGGREGATIONS = ("mean", "median", "min")
        val_results = {
            "mean":   0.48,
            "median": 0.48,
            "min":    0.16,
        }

        # Without gating
        meld_postprocess(subject_data, val_threshold=None, val_aggregation="no_gating", label="NO GATING")

        # With gating: one run per aggregation
        for agg in AGGREGATIONS:
            thresh = val_results.get(agg)
            meld_postprocess(
                subject_data, val_threshold=thresh, val_aggregation=agg,
                label=f"GATING {agg} threshold={thresh:.2f}" if thresh is not None else f"GATING {agg} (no threshold)",
                dist_source=args.dist_source,
            )
    else:
        ckpt_path = args.ckpt_path
        print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
        if ckpt_path is not None:
            model = LanGuideMedSegWrapper.load_from_checkpoint(checkpoint_path=ckpt_path, args=args, eva=eva, fold_number=0)
        else:
            model = LanGuideMedSegWrapper(args, eva=eva, fold_number=0)

        run_trainer_test(model, dl_test, args, accelerator, devices, ckpt_path=ckpt_path)
