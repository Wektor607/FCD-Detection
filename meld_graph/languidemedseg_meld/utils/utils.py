import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Tuple

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Sampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import meld_graph.mesh_tools as mt
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import (DEFAULT_HDF5_FILE_ROOT, MELD_DATA_PATH,
                              MELD_PARAMS_PATH, SURFACE_PARTIAL)
from scripts.manage_results.plot_prediction_report import create_surface_plots
from utils.config import SUBJECTS_DIR
from utils.converter_mgh_to_nifti import (convert_prediction_mgh_to_nii,
                                          get_combat_feature_path, save_mgh)

SEED = 42

class LesionOversampleSampler(Sampler):
    """
    Сэмплер, который берёт ВСЕ healthy-примеры ровно по одному разу,
    а lesion-примеры — с replacement, чтобы заполнить всю эпоху.
    """

    def __init__(self, labels, seed=42):
        self.labels = labels
        random.seed(seed)
        # индексы здоровых и lesion
        self.hc_idx = [i for i, label in enumerate(labels) if label == 0]
        self.les_idx = [i for i, label in enumerate(labels) if label == 1]
        # хотим ровно len(labels) выборок за эпoху
        self.epoch_size = len(labels)

    def __iter__(self):
        # начинаем с всех hc-индексов
        idxs = self.hc_idx.copy()
        # сколько нужно докинуть lesion'ов
        n_les_to_sample = self.epoch_size - len(idxs)
        # добавляем lesion с replacement
        idxs += random.choices(self.les_idx, k=n_les_to_sample)
        # перемешиваем всю эпоху
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return self.epoch_size


def summarize_ci(scores, B=10_000, alpha=0.05, seed=42):
    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    N = x.size
    if N == 0:
        return np.nan, np.nan, np.nan
    if N == 1:
        return float(x[0]), float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(B, N))  # 10k resamples
    boot_meds = np.median(x[idx], axis=1)  # median in each resample
    lo, hi = np.percentile(boot_meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return float(np.median(x)), float(lo), float(hi)

def threshold_surface_prediction(pred, percentile=99.5):
    pred = pred.copy()
    nz = pred[pred > 0]
    if nz.size == 0:
        return np.zeros_like(pred)
    thr = np.percentile(nz, percentile)
    pred[pred < thr] = 0
    return pred


def convert_preds_to_nifti(ckpt_path, subject_ids, probs_bin, c, mode="test"):
    subjects_fs_dir = Path(MELD_DATA_PATH) / "input" # / "data4sharing" <- later return back for training or change the folder name, which contains data
    predictions_output_root = Path(MELD_DATA_PATH) / "output" / "predictions_reports" / ckpt_path
    os.makedirs(predictions_output_root, exist_ok=True)

    results = {}

    for (sid, pred) in zip(subject_ids, probs_bin):

        # Skip-list
        # if sid in ["MELD_H3_3T_FCD_0018", "MELD_H4_15T_FCD_0021", "MELD_H6_3T_FCD_0017"]:
        #     print(f"Skipping {sid} as per request")
        #     continue   

        # Convert prediction tensor → numpy
        predictions = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)

        classifier_dir = subjects_fs_dir / sid / "xhemi" / "classifier"
        predictions_dir = predictions_output_root / sid / "predictions"
        os.makedirs(classifier_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)

        combat_path = subjects_fs_dir / "meld_combats"     

        # ============================================================
        # (1) Save each hemisphere prediction as MGH → NIfTI
        # ============================================================
        for idx, hemi in enumerate(["lh", "rh"]):

            overlay = np.zeros_like(c.cortex_mask, dtype=np.float32)
            overlay[c.cortex_mask] = predictions[idx]

            if predictions[idx].shape[0] != np.sum(c.cortex_mask):
                print(f"[WARN] {sid}: mismatch cortex_mask vs prediction size")
                continue

            combat_file = get_combat_feature_path(combat_path, sid)

            with h5py.File(combat_file, "r") as f:
                key = ".combat.on_lh.thickness.sm3.mgh"
                if key not in f[hemi]:
                    raise KeyError(f"No dataset {key} in HDF5 for {hemi}")
                base_arr = f[hemi][key][:]

            # MGH template
            affine = nib.load(
                SUBJECTS_DIR / "fsaverage_sym" / "mri" / "T1.mgz"
                # "/app/data/input/sub-00170_acq-T2sel_FLAIR_likeT1.nii.gz"
            ).affine

            mgh_img = nib.MGHImage(base_arr[np.newaxis, :, np.newaxis], affine)

            out_mgh_pred = classifier_dir / f"{hemi}.prediction.mgh"
            save_mgh(out_mgh_pred, overlay, mgh_img)

            convert_prediction_mgh_to_nii(
                subjects_fs_dir,
                out_mgh_pred,
                hemi,
                predictions_dir,
                verbose=True,
            )

            surf_vis_path = predictions_dir / f"{hemi}_surface_visualisation.png"
            # volume_3d_visualisation(
            #     prediction_surf=overlay, # maybe add np.squeeze
            #     hemi_name=hemi,
            #     save_path=surf_vis_path
            # )
            surf_pred = threshold_surface_prediction(overlay, percentile=99.5)

            volume_3d_visualisation(
                prediction_surf=surf_pred,
                hemi_name=hemi,
                save_path=surf_vis_path
            )
            print(f"✓ Saved surface visualisation: {surf_vis_path}")


        # ============================================================
        # (2) Combine LH + RH into final_nii
        # ============================================================
        lh_nii = predictions_dir / "lh.prediction.nii.gz"
        rh_nii = predictions_dir / "rh.prediction.nii.gz"
        final_nii = predictions_dir / f"prediction_{sid}.nii.gz"

        # def _binarize_nii(path):
        #     if not path.exists():
        #         return None
        #     img = nib.load(str(path))
        #     arr = img.get_fdata()
        #     sys.stderr.write(
        #         f"{path}\n"
        #         f"  min/max: {np.nanmin(arr)} / {np.nanmax(arr)}\n"
        #         f"  unique~: {np.unique(arr[~np.isnan(arr)])[:10]}\n"
        #         f"  fraction > 0: {(arr > 0).mean():.4f}\n"
        #         f"  fraction > 0.5: {(arr > 0.5).mean():.4f}\n"
        #     )

        #     arr_bin = (arr > 0).astype(np.uint8)
        #     nib.save(nib.Nifti1Image(arr_bin, img.affine, img.header), str(path))
        #     return path

        # lh_p = _binarize_nii(lh_nii)
        # rh_p = _binarize_nii(rh_nii)
        lh_p = lh_nii if lh_nii.exists() else None
        rh_p = rh_nii if rh_nii.exists() else None
        if lh_p and rh_p:
            lh_img = nib.load(str(lh_p))
            rh_img = nib.load(str(rh_p))
            combined = np.maximum(lh_img.get_fdata(), rh_img.get_fdata())
            # combined = (combined > 0).astype(np.uint8)
            nib.save(nib.Nifti1Image(combined, lh_img.affine, lh_img.header), str(final_nii))
            print(f"🎉 Final combined PRED NIfTI: {final_nii}")
        else:
            src = lh_p or rh_p
            if src:
                shutil.copy2(str(src), str(final_nii))
            else:
                raise FileNotFoundError("No hemi predictions found")

        # =============================================
        # (3) Combine LH + RH visualisations
        # =============================================

        lh_png = predictions_dir / "lh_surface_visualisation.png"
        rh_png = predictions_dir / "rh_surface_visualisation.png"
        combined_png = predictions_dir / f"{sid}_surface_combined.png"

        if lh_png.exists() and rh_png.exists():
            concat_side_by_side(lh_png, rh_png, combined_png)
        else:
            print(f"[WARN] Missing hemisphere PNGs for subject {sid}")

        results[sid] = final_nii   # <---- save result for this subject

    # ============================================================
    # RETURN ONLY AFTER PROCESSING ALL SUBJECTS
    # ============================================================
    return results

def volume_3d_visualisation(prediction_surf, hemi_name, save_path):
    """
    Creates MELD-style lateral + medial hemisphere render and saves as PNG.
    """
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "bool"):
        np.bool = bool

    c = MeldCohort(hdf5_file_root=DEFAULT_HDF5_FILE_ROOT, dataset=None)
    surf = mt.load_mesh_geometry(os.path.join(MELD_PARAMS_PATH, SURFACE_PARTIAL))

    # Use MELD's native renderer (from plot_prediction_report)
    im_lat, im_med = create_surface_plots(
        surf,
        prediction=prediction_surf,
        c=c
    )

    fig = plt.figure(figsize=(10, 4))
    plt.suptitle(f"{hemi_name.upper()} hemisphere", fontsize=16)

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(im_lat)
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(im_med)
    ax2.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def concat_side_by_side(img1_path, img2_path, save_path):
    im1 = Image.open(img1_path)
    im2 = Image.open(img2_path)

    # выравниваем по высоте
    h = im1.height + im2.height
    w = max(im1.width, im2.width)

    new_im = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    new_im.paste(im1, (0, 0))
    new_im.paste(im2, (0, im1.height))

    new_im.save(save_path)
    print(f"✓ Saved combined: {save_path}")

def summarize_clusters(cluster_mask, hemi_names=["left", "right"]):
    summary = []
    for h, hemi in enumerate(hemi_names):
        hemi_mask = cluster_mask[h]  # бинарная маска для полушария
        labels, num = ndimage.label(hemi_mask)  # находим кластеры
        for cluster_id in range(1, num + 1):
            coords = np.argwhere(labels == cluster_id)
            volume = coords.shape[0]  # количество вокселей
            center = coords.mean(axis=0).astype(int).tolist()  # центр масс
            summary.append({
                "hemi": hemi,
                "volume_voxels": int(volume),
                "center": center
            })
    return summary

def get_device() -> Tuple[torch.device, bool]:
    # Probe CUDA availability but be defensive: some builds report CUDA available
    # but initializing CUDA fails if drivers are missing. Try a lightweight probe
    # and fall back to CPU on any exception.
    try:
        if torch.cuda.is_available():
            try:
                # This may raise if CUDA driver isn't present or not initialized
                _ = torch.cuda.current_device()
                device = torch.device("cuda")
            except Exception:
                # CUDA not usable at runtime — fall back to CPU
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
    except Exception:
        device = torch.device("cpu")

    return device

def worker_init_fn(worker_id: int) -> None:
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

def move_to_device(obj, device: torch.device):
    """Recursively move tensors in nested structures to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [move_to_device(v, device) for v in obj]
        return type(obj)(seq)
    return obj

def random_from_distribution(dist: dict):
    keys = list(dist.keys())
    probs = list(dist.values())
    return random.choices(keys, weights=probs, k=1)[0]

def generate_random_text(text_probs: json):
    """
    Generate: <hemisphere> + <lobe>
    Example: "Left Hemisphere; Temporal lobe"
    """
    # if not isinstance(text_probs, dict):
    #     return "full brain"

    hemi = random_from_distribution(text_probs.get("hemisphere_text", {}))
    lobe = random_from_distribution(text_probs.get("lobe_text", {}))

    return f"{hemi}; {lobe}"