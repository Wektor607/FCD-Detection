import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from torch.utils.data import Sampler
import numpy as np
import random
from pathlib import Path
from engine.converter_mgh_to_nifti import *
from meld_graph.paths import MELD_DATA_PATH

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

def convert_preds_to_nifti(ckpt_path, subject_ids, probs_bin, c):
    subjects_fs_dir = Path(MELD_DATA_PATH) / "input" / "data4sharing"
    predictions_output_root = Path(MELD_DATA_PATH) / "output" / "predictions_reports" / ckpt_path
    os.makedirs(predictions_output_root, exist_ok=True)
    for (sid, pred) in zip(subject_ids, probs_bin):
        predictions = pred.detach().cpu().numpy()

        classifier_dir = subjects_fs_dir / sid / "xhemi" / "classifier"
        predictions_dir = predictions_output_root / sid / "predictions"
        os.makedirs(classifier_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)

        combat_path = subjects_fs_dir / "meld_combats"
        h5_path = combat_path / f"{sid}_patient_featurematrix_combat.hdf5"
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

            mgh_gt = save_gt_as_mgh(
                h5_path, hemi, predictions_dir, subjects_fs_dir
            )
            if mgh_gt:
                convert_gt_to_nii(subjects_fs_dir, mgh_gt, hemi, verbose=True)

        # Combine both hemispheres for prediction
        lh_nii = predictions_dir / "lh.prediction.nii.gz"
        rh_nii = predictions_dir / "rh.prediction.nii.gz"
        final_nii = predictions_dir / f"prediction_{sid}.nii.gz"
        cmd = f"mri_concat --i {lh_nii} --i {rh_nii} --o {final_nii} --combine"
        run_command(cmd, verbose=True)
        print(f"🎉 Final combined PRED NIfTI: {final_nii}")

        # Combine both hemispheres for ground‐truth
        gt_lh_nii = predictions_dir / "lh.gt.nii.gz"
        gt_rh_nii = predictions_dir / "rh.gt.nii.gz"
        gt_final = predictions_dir / f"ground_truth_{sid}.nii.gz"
        cmd_gt = f"mri_concat --i {gt_lh_nii} --i {gt_rh_nii} --o {gt_final} --combine"
        run_command(cmd_gt, verbose=False)
        print(f"🎉 Final combined GT   NIfTI: {gt_final}")

        sys.exit(0)