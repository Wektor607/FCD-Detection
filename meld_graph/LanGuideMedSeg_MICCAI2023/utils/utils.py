import sys, os
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import Sampler
import numpy as np
import random
from pathlib import Path
from scipy import ndimage
from LanGuideMedSeg_MICCAI2023.engine.converter_mgh_to_nifti import *
from meld_graph.paths import MELD_DATA_PATH

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

def convert_preds_to_nifti(ckpt_path, subject_ids, probs_bin, c, mode="test"):
    subjects_fs_dir = Path(MELD_DATA_PATH) / "input" / "data4sharing"
    predictions_output_root = Path(MELD_DATA_PATH) / "output" / "predictions_reports" / ckpt_path
    os.makedirs(predictions_output_root, exist_ok=True)
    for (sid, pred) in zip(subject_ids, probs_bin):
        if hasattr(pred, "detach"):
            predictions = pred.detach().cpu().numpy()
        else:
            predictions = np.asarray(pred)

        classifier_dir = subjects_fs_dir / sid / "xhemi" / "classifier"
        predictions_dir = predictions_output_root / sid / "predictions"
        os.makedirs(classifier_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)

        combat_path = subjects_fs_dir / "meld_combats"
        if "_C_" in sid:
            group = "control"
        else:
            group = "patient"

        h5_path = combat_path / f"{sid}_{group}_featurematrix_combat.hdf5"
        for idx, hemi in enumerate(["lh", "rh"]):
            # Prediction overlay
            overlay = np.zeros_like(c.cortex_mask, dtype=np.float32)
            overlay[c.cortex_mask] = predictions[idx]

            # Read template thickness to get shape/affine
            combat_file = get_combat_feature_path(combat_path, sid)
            with h5py.File(combat_file, "r") as f:
                key = ".combat.on_lh.thickness.sm3.mgh"
                if key not in f[hemi]:
                    raise KeyError(f"No dataset {key!r} in group {hemi}")
                base_arr = f[hemi][key][:]

            mgh_img = nb.MGHImage(
                base_arr[np.newaxis, :, np.newaxis],
                affine=nb.load(subjects_fs_dir / "fsaverage_sym" / "mri" / "T1.mgz").affine,
            )

            # Save prediction MGH → NIfTI
            out_mgh_pred = classifier_dir / f"{hemi}.prediction.mgh"
            save_mgh(out_mgh_pred, overlay, mgh_img)
            print(f"Saved PRED MGH: {out_mgh_pred}")
            convert_prediction_mgh_to_nii(
                subjects_fs_dir,
                out_mgh_pred,
                hemi,
                predictions_dir,
                verbose=True,
            )
            # Check NIfTI after conversion
            nii_path = predictions_dir / f"{hemi}.prediction.nii.gz"
            if nii_path.exists():
                import nibabel as nib
                arr = nib.load(str(nii_path)).get_fdata()
                print(f"DEBUG {sid} {hemi} NIfTI unique:", np.unique(arr))

            if mode == "test":
                mgh_gt = save_gt_as_mgh(
                    h5_path, hemi, predictions_dir, subjects_fs_dir
                )
                if mgh_gt:
                    convert_gt_to_nii(subjects_fs_dir, mgh_gt, hemi, verbose=True)

        # Combine both hemispheres for prediction
        # First, ensure hemi NIfTIs contain only binary values (0/1).
        lh_nii = predictions_dir / "lh.prediction.nii.gz"
        rh_nii = predictions_dir / "rh.prediction.nii.gz"
        final_nii = predictions_dir / f"prediction_{sid}.nii.gz"

        try:
            import nibabel as nib

            def _binarize_nii(path):
                if not path.exists():
                    return None
                img = nib.load(str(path))
                arr = img.get_fdata()
                # if there are unexpected values (>1), binarize them (any positive -> 1)
                if np.any(arr > 1) or not np.array_equal(np.unique(arr), np.array([0])) and np.any(arr != 0):
                    arr_bin = (arr > 0).astype(np.uint8)
                    new_img = nib.Nifti1Image(arr_bin, img.affine, img.header)
                    nib.save(new_img, str(path))
                    print(f"🔧 Binarized NIfTI: {path} (unique: {np.unique(arr_bin)})")
                else:
                    # still make sure dtype is reasonable
                    if arr.dtype != np.uint8 and arr.dtype != np.int8 and arr.dtype != np.int16:
                        arr_bin = (arr > 0).astype(np.uint8)
                        new_img = nib.Nifti1Image(arr_bin, img.affine, img.header)
                        nib.save(new_img, str(path))
                        print(f"🔧 Normalized dtype and binarized NIfTI: {path}")
                return path

            lh_p = _binarize_nii(lh_nii)
            rh_p = _binarize_nii(rh_nii)

            # If both hemispheres exist, combine voxel-wise using numpy.maximum (keepmax behaviour)
            if lh_p and rh_p:
                lh_img = nib.load(str(lh_p))
                rh_img = nib.load(str(rh_p))
                lh_arr = lh_img.get_fdata()
                rh_arr = rh_img.get_fdata()

                # Ensure shapes match; if not, try to broadcast sensibly or raise
                if lh_arr.shape != rh_arr.shape:
                    raise RuntimeError(f"Shape mismatch between LH and RH NIfTIs: {lh_arr.shape} vs {rh_arr.shape}")

                combined = np.maximum(lh_arr, rh_arr)
                combined = (combined > 0).astype(np.uint8)
                combined_img = nib.Nifti1Image(combined, lh_img.affine, lh_img.header)
                nib.save(combined_img, str(final_nii))
                print(f"🎉 Final combined PRED NIfTI (max): {final_nii}")
            else:
                # fallback: if only one hemisphere exists, copy it
                src = lh_p or rh_p
                if src:
                    import shutil

                    shutil.copy2(str(src), str(final_nii))
                    print(f"⚠️ Only one hemisphere NIfTI found, copied to {final_nii}")
                else:
                    raise FileNotFoundError(f"No hemisphere prediction NIfTIs found for {sid}")

        except Exception as e:
            # If nibabel approach fails for any reason, fall back to previous external command
            print(f"⚠️ Python combine failed ({e}), falling back to mri_concat command")
            cmd = f"mri_concat --i {lh_nii} --i {rh_nii} --o {final_nii} --combine --keepmax"
            run_command(cmd, verbose=True)
            print(f"🎉 Final combined PRED NIfTI: {final_nii}")
    
        if mode == "test":
            # Combine both hemispheres for ground‐truth using the same python approach
            gt_lh_nii = predictions_dir / "lh.gt.nii.gz"
            gt_rh_nii = predictions_dir / "rh.gt.nii.gz"
            gt_final = predictions_dir / f"ground_truth_{sid}.nii.gz"
            try:
                import nibabel as nib

                def _combine_max(lhs, rhs, outp):
                    if not lhs.exists() and not rhs.exists():
                        raise FileNotFoundError("No GT hemisphere NIfTIs found")
                    if lhs.exists() and rhs.exists():
                        l_img = nib.load(str(lhs))
                        r_img = nib.load(str(rhs))
                        l_arr = l_img.get_fdata()
                        r_arr = r_img.get_fdata()
                        if l_arr.shape != r_arr.shape:
                            raise RuntimeError("Shape mismatch between GT LH and RH NIfTIs")
                        combined = np.maximum(l_arr, r_arr)
                        combined = (combined > 0).astype(np.uint8)
                        nib.save(nib.Nifti1Image(combined, l_img.affine, l_img.header), str(outp))
                    else:
                        src = lhs if lhs.exists() else rhs
                        import shutil

                        shutil.copy2(str(src), str(outp))

                _combine_max(gt_lh_nii, gt_rh_nii, gt_final)
                print(f"🎉 Final combined GT   NIfTI: {gt_final}")
            except Exception as e:
                print(f"⚠️ Python GT combine failed ({e}), falling back to mri_concat")
                cmd_gt = f"mri_concat --i {gt_lh_nii} --i {gt_rh_nii} --o {gt_final} --combine"
                run_command(cmd_gt, verbose=False)
                print(f"🎉 Final combined GT   NIfTI: {gt_final}")

        return final_nii

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