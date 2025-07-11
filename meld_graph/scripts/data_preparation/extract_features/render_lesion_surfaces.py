#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
from nilearn import plotting, surface
import matplotlib.pyplot as plt

def main(subject_id, subjects_dir):
    # Prepare directories
    surf_dir   = os.path.join(subjects_dir, subject_id, "surf")
    surf_meld  = os.path.join(subjects_dir, subject_id, "surf_meld")
    visual_dir = os.path.join(subjects_dir, subject_id, "visualizations")
    os.makedirs(visual_dir, exist_ok=True)

    # Hemisphere-specific surface, lesion, and curvature files
    hemis = {
        "left":  ("lh.white",  "lh.lesion_linked.mgh", "lh.curv"),
        "right": ("rh.white",  "rh.lesion_linked.mgh", "rh.curv")
    }

    for hemi, (surf_name, lesion_name, curv_name) in hemis.items():
        surf_file   = os.path.join(surf_dir, surf_name)
        lesion_file = os.path.join(surf_meld, lesion_name)
        curv_file   = os.path.join(surf_dir, curv_name)

        # Load mesh geometry and curvature shading
        coords, faces = surface.load_surf_mesh(surf_file)
        bg_map        = surface.load_surf_data(curv_file)

        # Load lesion data from MGH and flatten to vector
        img_data = nib.load(lesion_file).get_fdata()
        stat_map = np.ravel(img_data)

        # Plot lateral view with background
        fig = plotting.plot_surf_stat_map(
            surf_mesh=(coords, faces),
            stat_map=stat_map,
            hemi=hemi,
            view='lateral',
            bg_map=bg_map,
            threshold=None,
            cmap='hot',
            colorbar=False,
            black_bg=False
        )
        out_png = os.path.join(visual_dir, f"{subject_id}_{hemi}_lateral.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

        # Plot medial view
        fig = plotting.plot_surf_stat_map(
            surf_mesh=(coords, faces),
            stat_map=stat_map,
            hemi=hemi,
            view='medial',
            bg_map=bg_map,
            threshold=None,
            cmap='hot',
            colorbar=False,
            black_bg=False
        )
        out_png = os.path.join(visual_dir, f"{subject_id}_{hemi}_medial.png")
        fig.savefig(out_png, dpi=300)
        plt.close(fig)

    print(f"Saved all visualizations to: {visual_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render lesion overlays on cortical surfaces")
    parser.add_argument("subject_id",   help="Subject identifier, e.g. sub-00001")
    parser.add_argument("subjects_dir", help="Path to FreeSurfer subjects directory")
    args = parser.parse_args()
    main(args.subject_id, args.subjects_dir)
