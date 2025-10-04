import re
from pathlib import Path
import nibabel as nib
import numpy as np


def parse_id(filename: str) -> str:
    """
    Оставляет только MELD_ID до суффиксов _control/_patient
    """
    match = re.match(r"^(MELD_[^_]+_[^_]+_[^_]+_[^_]+)", filename)
    if match:
        return match.group(1)
    return filename


def get_center_of_mass(mask):
    """
    Accepts either a path (str/Path) to a NIfTI/MGZ file or a nibabel image object.
    Returns center of mass in mm (x,y,z) or None if no nonzero voxels.
    """
    # If user passed a nibabel image, use it directly
    if hasattr(mask, "get_fdata"):
        nii = mask
    else:
        # assume it's a path-like object
        nii = nib.load(str(mask))

    data = nii.get_fdata()
    coords = np.argwhere(data > 0)
    if coords.size == 0:
        return None
    center_vox = coords.mean(axis=0)
    center_mm = nib.affines.apply_affine(nii.affine, center_vox)
    return tuple(center_mm)
