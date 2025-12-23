from matplotlib import gridspec
from nilearn import plotting
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
import shutil


def plot_and_save(img_nii, epi_dict, file_name: str, out_dir: Path, t1_file: str):

    plt.switch_backend("Agg")

    # load nifti
    nii_img = nib.load(str(img_nii)) if not hasattr(img_nii, "get_fdata") else img_nii
    arr = np.nan_to_num(nii_img.get_fdata()).astype(np.float32)
    img_float = nib.Nifti1Image(arr, nii_img.affine, nii_img.header)

    # adaptive vmax
    nonzero = arr[arr > 0]
    vmax = np.percentile(nonzero, 99.5) if nonzero.size else 1.0

    # figure + grid
    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(1, 1, figure=fig)

    ax = fig.add_subplot(gs[0, 0])

    plotting.plot_roi(
        img_float,
        bg_img=t1_file,
        axes=ax,
        display_mode="ortho",
        title=f"Prediction {file_name}",
        cmap="autumn",
        annotate=False,
        colorbar=False,
        vmin=0.0,
        vmax=vmax,
        draw_cross=False,
    )

    png_path = out_dir / f"{file_name}.png"
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # save nifti
    nii_out_path = out_dir / f"{file_name}.nii.gz"
    nib.save(img_float, nii_out_path)

    # copy surface PNG (если он реально обновляется!)
    src = Path("data/output/predictions_reports/MELD") / file_name / "predictions" / f"{file_name}_surface_combined.png"
    dst = out_dir / f"{file_name}_surface_combined.png"

    if src.exists():
        shutil.copy(src, dst)
    else:
        print(f"[WARN] 3D surface PNG not found: {src}")
