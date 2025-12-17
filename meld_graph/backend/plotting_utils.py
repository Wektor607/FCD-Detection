from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting


def plot_and_save(img_nii, epi_dict, file_name: str, out_dir: Path, t1_file: str) -> None:
    """
    Normalize/save NIfTI and create PNG overlay using nilearn.plot_roi.
    Returns paths (png_path, nii_out_path).
    """
    # Force non-interactive backend to avoid GUI issues in headless envs
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    # Ensure nibabel image and normalize to binary 0/1
    try:
        nii_img = nib.load(str(img_nii)) if not hasattr(img_nii, "get_fdata") else img_nii
        arr = nii_img.get_fdata().astype(float)
        arr = (arr > 0).astype(float)
        img_float = nib.Nifti1Image(arr, nii_img.affine, nii_img.header)
    except Exception:
        # fallback: assume img_nii path works directly
        img_float = img_nii

    center = None
    try:
        data = img_float.get_fdata()
        coords = np.argwhere(data > 0)
        if coords.size:
            center_vox = coords.mean(axis=0)
            center = nib.affines.apply_affine(img_float.affine, center_vox)
    except Exception:
        center = None

    plotting.plot_roi(
        img_float,
        bg_img=t1_file,
        display_mode="ortho",
        title=f"Prediction {file_name}",
        cmap="autumn",
        annotate=True,
        cut_coords=center,
        colorbar=True,
        draw_cross=False,
        vmin=0.0,
        vmax=1.0,
    )

    png_path = out_dir / f"{file_name}.png"
    plt.savefig(png_path, bbox_inches="tight")

    nii_out_path = out_dir / f"{file_name}.nii.gz"
    try:
        if hasattr(img_float, "get_fdata"):
            nii_out_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(img_float, str(nii_out_path))
        else:
            # img_nii might be a path-like
            try:
                import shutil

                shutil.copy(str(img_nii), str(nii_out_path))
            except Exception:
                pass
    except Exception:
        pass
