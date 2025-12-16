from subprocess import Popen, PIPE
import h5py
import numpy as np
import os
import nibabel as nb
from pathlib import Path
from typing import Union


def run_command(command: str, verbose: bool):
    proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, encoding="utf-8")
    stdout, stderr = proc.communicate()
    if verbose and stdout:
        print(stdout)
    if proc.returncode != 0:
        print(f"❌ COMMAND FAILED: {command}\nERROR: {stderr}")
        raise RuntimeError(f"Command failed: {command}")


def save_gt_as_mgh(h5_path: str, hemi: str, out_dir: str, subjects_fs_dir: str):
    """
    Extracts a GT array from HDF5 and saves it to an MGH file <hemi>.gt.mgh
    """
    key = ".on_lh.lesion.mgh"
    thick_key = ".combat.on_lh.thickness.sm3.mgh"

    # читаем данные
    with h5py.File(h5_path, "r") as f:
        grp = f[hemi]
        if key in grp:
            print(f"✅ Found GT in HDF5: {key!r}")
            arr1d = grp[key][:]
        else:
            print(
                f"⚠️ GT not found in {hemi}, generating empty using template {thick_key!r}"
            )
            if thick_key not in grp:
                raise KeyError(
                    f"There is no GT or thickness template {thick_key!r} in group {hemi}"
                )
            template = grp[thick_key][:]
            arr1d = np.zeros_like(template, dtype=np.float32)

    # shape (n_vertices, 1, 1)
    data = arr1d[:, np.newaxis, np.newaxis].astype(np.float32)

    # # make MGHImage with the same affinity as T1.mgz
    # t1_mgz = subjects_fs_dir / "fsaverage_sym" / "mri" / "T1.mgz"
    # affine = nb.load(t1_mgz).affine

    # make MGHImage with the same affinity as FLAIR
    flair_path = subjects_fs_dir / "fsaverage_sym" / "sub-00170_acq-T2sel_FLAIR_likeT1.nii.gz"
    affine = nb.load(flair_path).affine

    img = nb.MGHImage(data, affine)

    os.makedirs(out_dir, exist_ok=True)
    out_mgh = out_dir / f"{hemi}.gt.mgh"
    nb.save(img, out_mgh)
    print(f"✅ Save GT MGH: {out_mgh}")
    return out_mgh

def convert_gt_to_nii(
    subjects_dir: str, mgh_path: str, hemi: str, verbose: bool = True
):
    """
    Converts <hemi>.gt.mgh → <hemi>.gt.nii.gz via fsaverage_sym template
    """
    base = os.path.dirname(mgh_path)
    mgz_path = mgh_path.with_suffix(".mgz") #.replace(".mgh", ".mgz")
    nii_path = Path(base) / f"{hemi}.gt.nii.gz"

    fsavg = subjects_dir / "fsaverage_sym" / "mri"
    # T1 = fsavg / "T1.mgz"
    T1 = subjects_dir / "fsaverage_sym" / "sub-00170_acq-T2sel_FLAIR_likeT1.nii.gz"
    orig = fsavg / "orig.mgz"

    # 1) surface → volume
    cmd1 = (
        f"SUBJECTS_DIR={subjects_dir} "
        f"mri_surf2vol --identity fsaverage_sym "
        f"--template {T1} --o {mgz_path} "
        f"--hemi {hemi} --surfval {mgh_path} --fillribbon"
    )
    run_command(cmd1, verbose)

    # 2) optional reprojection via orig
    if os.path.isfile(orig):
        cmd2 = (
            f"SUBJECTS_DIR={subjects_dir} "
            f"mri_vol2vol --mov {mgz_path} --targ {orig} "
            f"--regheader --o {mgz_path} --nearest"
        )
        run_command(cmd2, verbose)

    # 3) MGZ → NIfTI
    cmd3 = f"SUBJECTS_DIR={subjects_dir} mri_convert {mgz_path} {nii_path} -rt nearest"
    run_command(cmd3, verbose)
    print(f"✅ Save GT NIfTI: {nii_path}")
    return nii_path


def convert_prediction_mgh_to_nii(
    subjects_dir: str,
    out_mgh: str,
    hemi: str,
    predictions_dir: str,
    verbose: bool = True,
):
    """
    subjects_dir — the root of SUBJECTS_DIR (should be fsaverage_sym/)
    out_mgh — the full path to the just saved prediction.mgh
    hemi — 'lh' or 'rh'
    predictions_dir — where to put prediction_{sid}.nii.gz
    """
    # 1) Surface→Volume
    vol_mgz = out_mgh.with_suffix(".mgz") #.replace(".mgh", ".mgz")
    mri_path = subjects_dir / "fsaverage_sym" / "mri"
    # T1_path = mri_path / "T1.mgz"
    T1_path = subjects_dir / "fsaverage_sym" / "sub-00170_acq-T2sel_FLAIR_likeT1.nii.gz"
    cmd1 = (
        f"SUBJECTS_DIR={subjects_dir} "
        f"mri_surf2vol --identity fsaverage_sym "
        f"--template {T1_path} "
        f"--o {vol_mgz} "
        f"--hemi {hemi} "
        f"--surfval {out_mgh} "
        f"--fillribbon"
    )
    run_command(cmd1, verbose)

    # 2) (optional) reprojection to orig - can be skipped if there is no orig
    orig_path = mri_path / "orig.mgz"
    if os.path.isfile(orig_path):
        cmd2 = (
            f"SUBJECTS_DIR={subjects_dir} "
            f"mri_vol2vol --mov {vol_mgz} "
            f"--targ {orig_path} "
            f"--regheader "
            f"--o {vol_mgz} "
            f"--nearest"
        )
        run_command(cmd2, verbose)

    # 3) MGZ→NIfTI
    vol_nii = predictions_dir / os.path.basename(out_mgh).replace(".mgh", ".nii.gz")
    cmd3 = f"SUBJECTS_DIR={subjects_dir} mri_convert {vol_mgz} {vol_nii} -rt nearest"
    run_command(cmd3, verbose)

    if not os.path.isfile(vol_nii):
        raise FileNotFoundError(f"Conversion failed, no file: {vol_nii}")
    print(f"✅ NIfTI saved at: {vol_nii}")
    return vol_nii


def save_mgh(filename, vertex_values, demo_img):
    """save mgh file using nibabel and imported demo mgh file"""
    shape = demo_img.header.get_data_shape()
    data = np.zeros(shape, dtype=np.float32)
    data.flat[:] = vertex_values
    # Save result
    new_img = nb.MGHImage(data, demo_img.affine, demo_img.header)
    nb.save(new_img, filename)

def get_combat_feature_path(combat_hdf5_path: Union[str, Path], sid: str) -> Path:
    base = Path(combat_hdf5_path)
    candidates = [
        base / f"{sid}_patient_featurematrix_combat.hdf5",
        base / f"{sid}_control_featurematrix_combat.hdf5",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"File don't find for {sid}: {', '.join(map(str, candidates))}")

