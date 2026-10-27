import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
import pytorch_lightning as pl
import torch
import torch.multiprocessing
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
from engine.wrapper import LanGuideMedSegWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

import utils.config as config
from meld_graph.meld_cohort import MeldCohort
from utils.data import EpilepDataset
from utils.utils import (get_device, move_to_device, summarize_ci,
                         worker_init_fn)
from utils.utils import convert_preds_to_nifti, summarize_ci

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
    parser.add_argument("--dry_run", default=0, type=int, help="limit to first N subjects for quick testing (0 = disabled)")
    parser.add_argument("--find_thresholds", action="store_true", help="run threshold search on trainval before testing")

    cli = parser.parse_args()
    if cli.config is None:
        parser.error("--config is required")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.meld_check = cli.meld_check
    cfg.ckpt_path = cli.ckpt_path
    cfg.ckpt_prefix = cli.ckpt_prefix
    cfg.dry_run = cli.dry_run
    cfg.find_thresholds = cli.find_thresholds
    return cfg

def prepare_dataloader(args, tokenizer, cohort, text_emb) -> DataLoader:
    df = pd.read_csv(args.split_path, sep=",")
    # test_ids = df[(df["split"] == "test") & (df["subject_id"].str.contains("FCD"))]["subject_id"].tolist()
    test_ids = df[(df["split"] == "test")]["subject_id"].tolist()
    if getattr(args, "dry_run", 0) > 0:
        test_ids = test_ids[:args.dry_run]
        print(f"[DRY RUN] Using first {args.dry_run} subjects: {test_ids}")
    print("CSV and SPLIT path: ", args.csv_path, args.split_path)
    ds_test = EpilepDataset(
        csv_path=args.csv_path,
        tokenizer=tokenizer,
        feature_path=args.feature_path,
        subject_ids=test_ids,
        cohort=cohort,
        max_length=args.max_len,
        text_emb=text_emb,
        atlas_type=args.atlas_type,
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
    ckpt_prefix = Path(ckpt_prefix)

    ckpt_paths = [
        ckpt_prefix.parent / f"{ckpt_prefix.name}_fold{i+1}.ckpt"
        for i in range(5)
    ]
    att_mechanism = False
    text_emb = False
    for exp, flags in exp_flags.items():
        if exp in (args.ckpt_prefix or ""):
            att_mechanism = flags.get("self_att_mechanism", False)
            text_emb = flags.get("text_emb", False)
            print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
            break
    # ######################## DELETE AFTER EXPERIMENT WITHOUT TEXT EMBEDDING
    # text_emb = False
    # print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
    ######################## DELETE AFTER EXPERIMENT WITHOUT TEXT EMBEDDING
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
            weights_only=False,
        )
        model.eval()
        model.to(device)
        models.append(model)

    return models, tokenizer


def run_ensemble_inference(dl_test: DataLoader, models: List[torch.nn.Module], eva, device: torch.device, cohort: MeldCohort = None, prefix: str = "") -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cortex_mask = torch.from_numpy(cohort.cortex_mask).to(device)
    all_subject_ids = []
    all_labels = []
    all_probs = []
    all_dist_maps = []
    all_pred_dists = []

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
            model_pred_dists = []
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

                # distance head predictions
                if "non_lesion_logits" in outputs:
                    pred_dist = outputs["non_lesion_logits"].to(device)
                    pred_dist = pred_dist.reshape(B, H, V7)[:, :, cortex_mask].reshape(B, -1).detach()
                    model_pred_dists.append(pred_dist)

            probs_stack = torch.stack(model_probs, dim=0)
            probs_mean = probs_stack.mean(dim=0)

            if model_pred_dists:
                pred_dist_mean = torch.stack(model_pred_dists, dim=0).mean(dim=0)
            else:
                pred_dist_mean = torch.zeros(B, dist_maps_cortex.shape[-1], device=device)

            all_subject_ids.extend(subject_ids)
            all_labels.append(target.cpu())
            all_dist_maps.append(dist_maps_cortex.cpu())
            all_probs.append(probs_mean.detach())
            all_pred_dists.append(pred_dist_mean.cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_dist_maps = torch.cat(all_dist_maps, dim=0)
    all_preds = torch.cat(all_probs, dim=0)
    all_pred_dists = torch.cat(all_pred_dists, dim=0)

    return all_subject_ids, all_labels, all_dist_maps, all_preds, all_pred_dists


def find_optimal_threshold_on_val(
    args, models, eva, device, cohort, tokenizer, text_emb,
    aggregations=("mean", "median", "min"),
    sensitivity_weight: float = 1.5,
    min_sensitivity: float = 0.70,
):
    """Ensemble inference on trainval to find optimal distance filter threshold.

    Two strategies are printed for each aggregation:
      - Weighted J:   argmax(sensitivity_weight * sens + spec - 1)  — prioritises sensitivity
      - Constrained:  argmax(spec) subject to sens >= min_sensitivity

    Returns dict: {aggregation: threshold} using the Weighted J strategy.
    """
    df = pd.read_csv(args.split_path, sep=",")
    trainval_ids = df[df.split == "trainval"]["subject_id"].tolist()

    if len(trainval_ids) == 0:
        print("[WARN] No trainval subjects found in split CSV. Skipping threshold finding.")
        return {}

    print(f"\n[VAL] Running ensemble on {len(trainval_ids)} trainval subjects for threshold finding...")

    ds_val = EpilepDataset(
        csv_path=args.csv_path, tokenizer=tokenizer,
        feature_path=args.feature_path, subject_ids=trainval_ids,
        cohort=cohort, max_length=args.max_len,
        text_emb=text_emb, atlas_type=args.atlas_type,
    )
    dl_val = DataLoader(
        ds_val, batch_size=args.valid_batch_size, shuffle=False,
        num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    val_sids, val_labels, val_dist_maps, val_probs, val_pred_dists = \
        run_ensemble_inference(dl_val, models, eva, device, cohort, prefix="val")

    # Collect per-subject cluster stats
    subject_clusters = []  # list of (sid, is_control, [{label, agg_values}])
    for i in range(len(val_sids)):
        sid = val_sids[i]
        pred = val_probs[i]
        tgt = val_labels[i]
        dist_map_subj = val_dist_maps[i]
        is_control = ("FCD" not in sid) or (tgt.sum() == 0)

        pred_left = pred[0].detach().cpu().numpy().astype("float32")
        pred_right = pred[1].detach().cpu().numpy().astype("float32")
        pred_np = np.concatenate([pred_left, pred_right], axis=-1)

        mini = {sid: {"result": pred_np}}
        out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
        probs_flat = out[sid]["cluster_thresholded"]
        if isinstance(probs_flat, torch.Tensor):
            probs_flat = probs_flat.detach().cpu().numpy()

        boundary_zone = dist_map_subj.cpu().numpy() if isinstance(dist_map_subj, torch.Tensor) else np.array(dist_map_subj)
        boundary_zone = boundary_zone < 20

        cluster_ids = np.unique(probs_flat)
        cluster_ids = cluster_ids[cluster_ids > 0]
        tp_set = set(np.unique(probs_flat[boundary_zone])[np.unique(probs_flat[boundary_zone]) > 0].tolist())

        pred_dist_subj = val_pred_dists[i].numpy()
        clusters = []
        for cid in cluster_ids:
            cmask = probs_flat == cid
            cdists = pred_dist_subj[cmask]
            label = "TP" if cid in tp_set else "FP"
            clusters.append({
                "label": label,
                "pred_dist_min": float(cdists.min()),
                "pred_dist_mean": float(cdists.mean()),
                "pred_dist_median": float(np.median(cdists)),
            })
        subject_clusters.append({"sid": sid, "is_control": is_control, "clusters": clusters})

    total_patients = sum(1 for s in subject_clusters if not s["is_control"])
    total_ctrl = sum(1 for s in subject_clusters if s["is_control"])
    print(f"[VAL] Patients: {total_patients}, Controls: {total_ctrl}")

    # Per-patient sweep: for each threshold, compute sensitivity & specificity
    thresholds = np.arange(-0.2, 1.01, 0.01)
    results = {}

    W = sensitivity_weight
    col_w  = f"{'Weighted J':>10} (w={W:.1f})"
    col_c  = f"{'Constrained':>11} (sens>={min_sensitivity:.0%})"
    hdr_w  = f"{'Agg':>6} | {'Thresh':>6} | {'Sens':>7} | {'Spec':>7} | {'wJ':>7}"
    hdr_c  = f"{'Agg':>6} | {'Thresh':>6} | {'Sens':>7} | {'Spec':>7}"
    sep    = "-" * 40

    print(f"\n{'='*85}")
    print(f"OPTIMAL THRESHOLDS ON VALIDATION")
    print(f"  Left : Weighted J (sensitivity_weight={W:.1f})")
    print(f"  Right: Constrained (max specificity s.t. sensitivity >= {min_sensitivity:.0%})")
    print(f"{'='*85}")
    print(f"{hdr_w}    ||    {hdr_c}")
    print(f"{sep}    ||    {sep}")

    for agg in aggregations:
        col = f"pred_dist_{agg}"

        # Pre-compute sens/spec for every threshold
        sweep = []
        for thresh in thresholds:
            patients_with_tp = 0
            ctrl_with_no_fp = 0
            for s in subject_clusters:
                surviving = [c for c in s["clusters"] if c[col] < thresh]
                has_tp = any(c["label"] == "TP" for c in surviving)
                has_fp = any(c["label"] == "FP" for c in surviving)
                if not s["is_control"]:
                    if has_tp:
                        patients_with_tp += 1
                else:
                    if not has_fp:
                        ctrl_with_no_fp += 1
            sens = patients_with_tp / total_patients if total_patients > 0 else 0.0
            spec = ctrl_with_no_fp / total_ctrl if total_ctrl > 0 else 0.0
            sweep.append((thresh, sens, spec))

        # Strategy 1: Weighted J
        best_wj, best_wj_thresh, best_wj_sens, best_wj_spec = -9, 0.0, 0.0, 0.0
        for thresh, sens, spec in sweep:
            wj = W * sens + spec - 1.0
            if wj > best_wj:
                best_wj, best_wj_thresh, best_wj_sens, best_wj_spec = wj, thresh, sens, spec

        # Strategy 2: Constrained — best specificity with sens >= min_sensitivity
        candidates = [(thresh, sens, spec) for thresh, sens, spec in sweep if sens >= min_sensitivity]
        if candidates:
            best_c_thresh, best_c_sens, best_c_spec = max(candidates, key=lambda x: x[2])
        else:
            best_c_thresh, best_c_sens, best_c_spec = best_wj_thresh, best_wj_sens, best_wj_spec

        results[agg] = best_wj_thresh  # default: use weighted J
        print(
            f" {agg:>6} | {best_wj_thresh:>6.2f} | {best_wj_sens:>6.1%} | {best_wj_spec:>6.1%} | {best_wj:>7.3f}"
            f"    ||"
            f"  {agg:>6} | {best_c_thresh:>6.2f} | {best_c_sens:>6.1%} | {best_c_spec:>6.1%}"
        )

    print(f"{'='*85}")
    print(f"[INFO] val_results uses Weighted J thresholds by default. Update manually if preferred.")
    return results


def postprocess_and_save(args, all_subject_ids, all_labels, all_dist_maps, all_preds, eva, ckpt_prefix: str, cohort: MeldCohort, dataset_type: str, dataset_name: str, text_emb: bool, all_pred_dists: torch.Tensor = None, val_threshold: float = None, val_aggregation: str = "mean", label_method: str = ""):
    dice_scores, iou_scores, ppv_scores, ppv_clusters_scores = [], [], [], []
    results = []
    cluster_dist_analysis = []  # for distance-based filtering analysis

    fp_control_clusters = []
    for i in range(all_labels.shape[0]):
        sid = all_subject_ids[i]
        pred = all_preds[i]
        tgt = all_labels[i]
        dist_map_subj = all_dist_maps[i]

        is_control = ("FCD" not in sid) or (tgt.sum() == 0)

        # pred_np = torch.cat([pred[0], pred[1]], dim=0).detach().cpu().numpy().astype("float32")
        # mini = {sid: {"result": pred_np}}
        pred_left = pred[0].detach().cpu().numpy().astype("float32")
        pred_right = pred[1].detach().cpu().numpy().astype("float32")

        # Убедимся, что мы объединяем по оси вершин, не по каналам
        pred_np = np.concatenate([pred_left, pred_right], axis=-1)  # <-- правильно
        mini = {sid: {"result": pred_np}}

        out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
        probs_flat = out[sid]["cluster_thresholded"]

        # --- Distance-based cluster filtering (threshold from validation) ---
        if val_threshold is not None and all_pred_dists is not None:
            agg_fn = {"mean": np.mean, "median": np.median, "min": np.min}[val_aggregation]
            pred_dist_subj_filter = all_pred_dists[i].numpy()
            probs_flat_np = probs_flat.detach().cpu().numpy() if isinstance(probs_flat, torch.Tensor) else probs_flat.copy()
            cluster_ids_filter = np.unique(probs_flat_np)
            cluster_ids_filter = cluster_ids_filter[cluster_ids_filter > 0]
            for cid in cluster_ids_filter:
                cmask = probs_flat_np == cid
                if agg_fn(pred_dist_subj_filter[cmask]) >= val_threshold:
                    probs_flat_np[cmask] = 0
            probs_flat = probs_flat_np

        boundary_zone = dist_map_subj < 20

        if isinstance(probs_flat, torch.Tensor):
            probs_flat_cpu = probs_flat.detach().cpu().numpy()
        else:
            probs_flat_cpu = probs_flat

        ###############################
        cluster_ids = probs_flat_cpu[probs_flat_cpu > 0].astype(np.int32)
        ids, counts = np.unique(cluster_ids, return_counts=True)
        # отсортируем по размеру
        order = np.argsort(-counts)
        ids, counts = ids[order], counts[order]

        max_cluster = int(counts[0]) if len(counts) else 0
        # можно залогировать для контролей
        if False:  # per-sample cluster debug disabled
            print(sid, "n_clusters=", len(counts), "max_cluster=", max_cluster, "top_sizes=", counts[:5])
        ###############################
        # if "_C_" in sid:
        #     convert_preds_to_nifti(ckpt_prefix, [sid], [probs_flat_cpu.reshape(2, -1)], [tgt], cohort)

        boundary_zone_cpu = boundary_zone.detach().cpu().numpy() if isinstance(boundary_zone, torch.Tensor) else np.array(boundary_zone)


        difference = np.setdiff1d(np.unique(probs_flat_cpu), np.unique(probs_flat_cpu[boundary_zone_cpu]))
        difference = difference[difference > 0]
        n_fp_clusters = len(difference)
        correct_values = np.unique(probs_flat_cpu[boundary_zone_cpu])
        correct_values = correct_values[correct_values > 0]
        n_tp_clusters = len(correct_values)

        # --- Distance head analysis per cluster ---
        if all_pred_dists is not None:
            pred_dist_subj = all_pred_dists[i].numpy()  # [2*N_cortex], normalized (/ 300)
            all_cluster_ids = np.unique(probs_flat_cpu)
            all_cluster_ids = all_cluster_ids[all_cluster_ids > 0]
            tp_set = set(correct_values.tolist())
            fp_set = set(difference.tolist())

            for cid in all_cluster_ids:
                cmask = probs_flat_cpu == cid
                cluster_pred_dists = pred_dist_subj[cmask]
                label = "TP" if cid in tp_set else "FP"
                cluster_dist_analysis.append({
                    "subject_id": sid,
                    "cluster_id": int(cid),
                    "label": label,
                    "is_control": is_control,
                    "n_vertices": int(cmask.sum()),
                    "pred_dist_min": float(cluster_pred_dists.min()),
                    "pred_dist_mean": float(cluster_pred_dists.mean()),
                    "pred_dist_median": float(np.median(cluster_pred_dists)),
                    "pred_dist_max": float(cluster_pred_dists.max()),
                })

        gt_flat = tgt.reshape(-1)
        mask_np = (probs_flat_cpu > 0).astype(int)
        mask = torch.from_numpy(mask_np).long().to(gt_flat.device)
        labels = gt_flat.bool().long()

        dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)
        tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
        iou = tp / (tp + fp + fn + 1e-8)
        ppv = tp / (tp + fp + 1e-8)
        ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters + 1e-8)

        if not is_control:
            dice_scores.append(float(dices[1].detach().cpu()))
            ppv_scores.append(float(ppv))
            iou_scores.append(float(iou))
            ppv_clusters_scores.append(float(ppv_clusters))

            # per-sample print disabled
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

    d_med, d_lo, d_hi = summarize_ci(dice_scores)
    p_med, p_lo, p_hi = summarize_ci(ppv_scores)
    i_med, i_lo, i_hi = summarize_ci(iou_scores)
    ppv_clusters_med, ppv_clusters_lo, ppv_clusters_hi = summarize_ci(ppv_clusters_scores)
    n_tp_clusters = sum(r["number TP clusters"] for r in results)
    n_fp_clusters = sum(r["number FP clusters"] for r in results)
    ppv_clusters = n_tp_clusters / (n_tp_clusters + n_fp_clusters + 1e-8)

    # Sensitivity
    tp_clusters_list = [r["number TP clusters"] for r in results if "FCD" in r["subject_id"]]
    total_tp = len(tp_clusters_list)
    found_tp = sum(1 for t in tp_clusters_list if t > 0)
    pct_tp = found_tp / total_tp if total_tp > 0 else 0.0

    # Specificity
    total_ctrl = len(fp_control_clusters)
    no_fp_ctrl = sum(1 for t in fp_control_clusters if t == 0)
    pct_spec = no_fp_ctrl / total_ctrl if total_ctrl > 0 else 0.0


    fp_fcd_clusters = [r["number FP clusters"] for r in results if "FCD" in r["subject_id"]]
    avg_fp_ctrl = float(np.mean(fp_control_clusters)) if fp_control_clusters else 0.0
    avg_fp_fcd  = float(np.mean(fp_fcd_clusters))     if fp_fcd_clusters     else 0.0

    header = f"=== ENSEMBLE OVERALL TEST METRICS{' [' + label_method + ']' if label_method else ''} ==="
    print(f"\n{header}")
    print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
    print(f"PPV_pixels  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
    print(f"PPV_clusters_mean  : {ppv_clusters:.3f}")
    print(f"PPV_clusters_median  : {ppv_clusters_med:.3f} (95% CI {ppv_clusters_lo:.3f}-{ppv_clusters_hi:.3f})")
    print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
    print(f"Sensitivity (patients only): {found_tp} / {total_tp} FCDs ({pct_tp:.1%})")
    print(f"Specificity (controls only): {no_fp_ctrl} / {total_ctrl} scans with no FP ({pct_spec:.1%})")
    print(f"Avg FP clusters per control : {avg_fp_ctrl:.2f}")
    print(f"Avg FP clusters per FCD     : {avg_fp_fcd:.2f}")

    data_type = dataset_type.split("/")[-1].split("_")[0]
    data_name = dataset_name.split("/")[-1].split(".")[0]
    df = pd.DataFrame(results)
    if not text_emb:
        data_name = "no_text"

    df.to_csv(f"{ckpt_prefix}_{data_type}_{args.atlas_type}_{data_name}_{val_aggregation}_results.csv", index=False)

    # --- Distance head cluster analysis ---
    if cluster_dist_analysis:
        df_dist = pd.DataFrame(cluster_dist_analysis)
        dist_csv = f"{ckpt_prefix}_{data_type}_{data_name}_cluster_dist_analysis_{val_aggregation}.csv"
        df_dist.to_csv(dist_csv, index=False)
        print(f"\n[INFO] Saved cluster distance analysis to {dist_csv}")

        tp_rows = df_dist[df_dist["label"] == "TP"]
        fp_rows = df_dist[df_dist["label"] == "FP"]
        fp_ctrl = df_dist[(df_dist["label"] == "FP") & (df_dist["is_control"])]

        print("\n=== DISTANCE HEAD CLUSTER ANALYSIS ===")
        print(f"Total clusters: {len(df_dist)} (TP={len(tp_rows)}, FP={len(fp_rows)}, FP on controls={len(fp_ctrl)})")
        if len(tp_rows) > 0:
            print(f"  TP pred_dist_min:  mean={tp_rows['pred_dist_min'].mean():.4f}, median={tp_rows['pred_dist_min'].median():.4f}, "
                  f"std={tp_rows['pred_dist_min'].std():.4f}, range=[{tp_rows['pred_dist_min'].min():.4f}, {tp_rows['pred_dist_min'].max():.4f}]")
        if len(fp_rows) > 0:
            print(f"  FP pred_dist_min:  mean={fp_rows['pred_dist_min'].mean():.4f}, median={fp_rows['pred_dist_min'].median():.4f}, "
                  f"std={fp_rows['pred_dist_min'].std():.4f}, range=[{fp_rows['pred_dist_min'].min():.4f}, {fp_rows['pred_dist_min'].max():.4f}]")
        if len(fp_ctrl) > 0:
            print(f"  FP(ctrl) pred_dist_min: mean={fp_ctrl['pred_dist_min'].mean():.4f}, median={fp_ctrl['pred_dist_min'].median():.4f}")

        # --- Sweep distance filter thresholds ---
        print("\n=== DISTANCE FILTER SWEEP ===")
        print(f"{'threshold':>10} | {'TP_kept':>7} / {'TP_total':>8} | {'FP_kept':>7} / {'FP_total':>8} | {'FP_ctrl_kept':>12} / {'FP_ctrl_total':>13} | {'Sensitivity':>11} | {'Specificity':>11}")
        print("-" * 110)

        for thresh in [-0.1, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
            tp_kept = len(tp_rows[tp_rows["pred_dist_min"] < thresh]) if len(tp_rows) > 0 else 0
            fp_kept = len(fp_rows[fp_rows["pred_dist_min"] < thresh]) if len(fp_rows) > 0 else 0
            fp_ctrl_kept = len(fp_ctrl[fp_ctrl["pred_dist_min"] < thresh]) if len(fp_ctrl) > 0 else 0

            # Recompute sensitivity: how many FCD patients still have at least one TP cluster
            tp_surviving = tp_rows[tp_rows["pred_dist_min"] < thresh]
            patients_with_tp = tp_surviving["subject_id"].nunique() if len(tp_surviving) > 0 else 0
            sens = patients_with_tp / total_tp if total_tp > 0 else 0.0

            # Recompute specificity: controls with zero surviving FP clusters
            fp_ctrl_surviving = fp_ctrl[fp_ctrl["pred_dist_min"] < thresh]
            ctrl_with_fp = fp_ctrl_surviving["subject_id"].nunique() if len(fp_ctrl_surviving) > 0 else 0
            spec = (total_ctrl - ctrl_with_fp) / total_ctrl if total_ctrl > 0 else 0.0

            print(f"{thresh:>10.2f} | {tp_kept:>7} / {len(tp_rows):>8} | {fp_kept:>7} / {len(fp_rows):>8} | {fp_ctrl_kept:>12} / {len(fp_ctrl):>13} | {sens:>10.1%} | {spec:>10.1%}")

def main():
    args = get_cfg()
    eva, cohort, exp_flags = config.inference_config()

    device = get_device()
    print("start testing on device:", device)

    models, tokenizer = load_ensemble_models(args.ckpt_prefix, args, eva, exp_flags, device)
    text_emb = True     # DONT FORGET TO CHANGE ON TRUE
    print(f"[INFO] Text_emb={text_emb}, atlas_type={args.atlas_type}")

    AGGREGATIONS = ("mean", "median", "min")

    if args.find_thresholds:
        val_results = find_optimal_threshold_on_val(
            args, models, eva, device, cohort, tokenizer, text_emb,
            aggregations=AGGREGATIONS,
        )
    else:
        # Last found thresholds (Weighted J, sensitivity_weight=1.5):
        # mean=0.48 (Sens=70.0%, Spec=95.5%), median=0.48 (Sens=69.6%, Spec=95.5%), min=0.16 (Sens=67.1%, Spec=94.4%)
        val_results = {
            "mean":   0.58,#0.55,
            "median": 0.66,#0.55,
            "min":    0.26,#0.20,
        }
        print(f"[INFO] Using hardcoded thresholds: {val_results}  (re-run with --find_thresholds to update)")

    # --- Step 2: Run test ---
    dl_test = prepare_dataloader(args, tokenizer, cohort, text_emb)
    all_subject_ids, all_labels, all_dist_maps, all_probs, all_pred_dists = run_ensemble_inference(dl_test, models, eva, device, cohort, prefix=args.ckpt_prefix)

    # --- Without gating ---
    postprocess_and_save(
        args, all_subject_ids, all_labels, all_dist_maps, all_probs, eva,
        args.ckpt_prefix, cohort, args.split_path, args.csv_path, text_emb,
        all_pred_dists=all_pred_dists, val_threshold=None, val_aggregation="no_gating",
        label_method="NO GATING",
    )

    # --- With gating: one run per aggregation ---
    for agg in AGGREGATIONS:
        thresh = val_results.get(agg)
        postprocess_and_save(
            args, all_subject_ids, all_labels, all_dist_maps, all_probs, eva,
            args.ckpt_prefix, cohort, args.split_path, args.csv_path, text_emb,
            all_pred_dists=all_pred_dists, val_threshold=thresh, val_aggregation=agg,
            label_method=f"GATING {agg} threshold={thresh:.2f}" if thresh is not None else f"GATING {agg} (no threshold)",
        )

if __name__ == "__main__":
    main()
