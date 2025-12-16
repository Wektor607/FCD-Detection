import os
import csv
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import ants
from subprocess import Popen, PIPE
from nilearn import datasets
from nilearn.datasets import fetch_icbm152_2009
from atlasreader import create_output
from pathlib import Path

def has_lesion(h5, hemi):
    key = f"{hemi}/.on_lh.lesion.mgh"
    return key if key in h5 else None


def save_lesion_array_as_mgh_simple(h5_path, subjects_dir):
    # Extract subject_id from filename
    fname = os.path.basename(h5_path)
    sid = fname.split("_featurematrix")[0]

    with h5py.File(h5_path, 'r') as h5:
        for hemi in ('lh', 'rh'):
            key = has_lesion(h5, hemi)
            print(key)
            if key:
                data = h5[key][:]
                data = data[:, None, None]  # (NVERT,) → (NVERT, 1, 1)
                img = nib.MGHImage(data.astype(np.float32), affine=np.eye(4))
                out_dir = os.path.join(subjects_dir, sid, 'surf')
                os.makedirs(out_dir, exist_ok=True)
                nib.save(img, os.path.join(out_dir, f"{hemi}.lesion.mgh"))
                return hemi
    return None


def save_lesion_array_as_mgh(folder_name, h5_path, total_name, scanner, subject_id, subjects_dir):
    with h5py.File(h5_path, 'r') as h5:
        for hemi in ['lh', 'rh']:
            key = os.path.join(total_name, scanner, "patient", subject_id, hemi, ".on_lh.lesion.mgh")
            if key in h5:
                lesion = h5[key][:]
                if lesion.ndim != 1:
                    raise ValueError(f"Unexpected shape for lesion data: {lesion.shape}")
                lesion = lesion[:, np.newaxis, np.newaxis]
                img = nib.MGHImage(lesion.astype(np.float32), affine=np.eye(4))
                out_dir = os.path.join(subjects_dir, folder_name, "surf")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{hemi}.lesion.mgh")
                nib.save(img, out_path)
                return hemi  # use only one hemisphere
    raise RuntimeError(f"No lesion data for {subject_id} in {h5_path}")


def run_command(command, verbose=True):
    proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, encoding='utf-8')
    stdout, stderr = proc.communicate()
    if verbose:
        print(stdout)
    if proc.returncode != 0:
        print(f"❌ COMMAND FAILED: {command}\nERROR: {stderr}")
        raise RuntimeError(f"Command failed: {command}")


def convert_lesion_mgh_to_nii(folder_name, subjects_dir, hemi='lh', verbose=True):
    subjects_dir = Path(subjects_dir)
    fsaverage_root = Path("/home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing")
    fsavg_mri = fsaverage_root / "fsaverage_sym" / "mri"

    surf_path = subjects_dir / folder_name / "labels" / f"labels-{hemi}.mgh"
    template  = fsavg_mri / "orig.mgz"  # target template
    vol_nii   = subjects_dir / folder_name / f"{hemi}.lesion.nii.gz"

    # 1) Surface → volume projection in fsaverage_sym/orig.mgz space
    #    Write directly to .nii.gz to avoid calling mri_convert
    cmd = (
        f"SUBJECTS_DIR={fsaverage_root} "
        f"mri_surf2vol --identity fsaverage_sym "
        f"--template {template} "
        f"--o {vol_nii} "
        f"--hemi {hemi} "
        f"--surfval {surf_path} "
        f"--fillribbon"
    )
    run_command(cmd, verbose)

    return str(vol_nii)


def process_t1_mask_only(mask_path, output_dir, base_name='subject', t1_path=None):
    os.makedirs(output_dir, exist_ok=True)

    mni_path = fetch_icbm152_2009().t1
    mni_img = ants.image_read(mni_path)

    if t1_path is None:
        raise ValueError("T1 path must be provided.")

    t1_img = ants.image_read(t1_path)

    reg = ants.registration(fixed=mni_img, moving=t1_img, type_of_transform='Affine')

    lesion_img = ants.image_read(mask_path)
    mask_in_mni = ants.apply_transforms(
        fixed=mni_img,
        moving=lesion_img,
        transformlist=reg['fwdtransforms'],
        whichtoinvert=[False],
        interpolation='genericLabel'
    )

    mni_mask_path = os.path.join(output_dir, f"{base_name}_mask_mni.nii.gz")
    ants.image_write(mask_in_mni, mni_mask_path)

    atlas = datasets.fetch_atlas_harvard_oxford(
        'cort-maxprob-thr25-1mm',
        symmetric_split=True
    )
    atlas_img = atlas.maps
    atlas_data = atlas_img.get_fdata()
    z_val = int(np.max(np.unique(atlas_data)))

    nii = nib.load(mni_mask_path)
    nonzero_count = np.count_nonzero(nii.get_fdata())
    print(f"[debug] {base_name}: nonzero voxels in lesion = {nonzero_count}")

    if nonzero_count == 0:
        return None, None

    z_data = (nii.get_fdata() > 0).astype(np.float32) * z_val
    z_img = nib.Nifti1Image(z_data, affine=nii.affine, header=nii.header)
    z_img_path = os.path.join(output_dir, f"{base_name}_mask_zmap.nii.gz")
    nib.save(z_img, z_img_path)

    create_output(
        z_img_path,
        outdir=output_dir,
        cluster_extent=0,
        atlas=['harvard_oxford', 'aal']
    )

    reports = pd.read_csv(os.path.join(output_dir, f"{base_name}_mask_zmap_clusters.csv"))
    harv_rep = (
        reports["harvard_oxford"].dropna().iloc[0]
        if not reports["harvard_oxford"].dropna().empty
        else "N/A"
    )
    aal_rep = (
        reports["aal"].dropna().iloc[0]
        if not reports["aal"].dropna().empty
        else "N/A"
    )

    return harv_rep, aal_rep


def full_pipeline(data4sharing_root, report_log_path):
    fsaverage_root = "/home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing"
    t1_path = os.path.join(fsaverage_root, "fsaverage_sym/mri/T1.mgz")
    csv_path = os.path.join(data4sharing_root, "all_augmented_reports.csv")

    # --- Step 1: load already processed subject_ids
    processed_subjects = set()
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        if 'subject_id' in existing_df.columns:
            processed_subjects = set(existing_df['subject_id'].astype(str))
            print(f"[0] Already processed {len(processed_subjects)} subject_ids")

    is_new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:

        writer = csv.writer(csvfile)
        if is_new_file:
            writer.writerow(["subject_id", "report_harvard_oxford", "report_aal"])
        csvfile.flush()

        # --- Marker #1
        print(f"[1] Opened directory {data4sharing_root}")

        for folder_obj in reversed(os.listdir(data4sharing_root)):
            if 'ipynb' in folder_obj or 'csv' in folder_obj or 'reports' in folder_obj:
                continue

            label_files = []
            sid = folder_obj

            for labels in os.listdir(os.path.join(data4sharing_root, folder_obj, 'labels')):
                label_files.append(labels)

            # 2) Check if already processed
            if sid in processed_subjects:
                print(f"[!] {sid} already exists in all_reports.csv, skipping")
                continue

            # 3) Generate report
            report_dir = os.path.join(data4sharing_root, "reports", sid)
            os.makedirs(report_dir, exist_ok=True)

            print(sid)
            if "_C_" in sid:
                harv, aal = "No lesion detected", "No lesion detected"
            else:
                # --- Marker #4
                print(f"[4] Processing MGH files: {label_files}")

                meta = next((f for f in label_files if f == "labels-meta.npz"), None)
                with np.load(os.path.join(data4sharing_root, folder_obj, 'labels', meta)) as data:
                    if data["lh_has_lesion"]:
                        hemi = 'lh'
                    elif data["rh_has_lesion"]:
                        hemi = 'rh'

                nii_path = convert_lesion_mgh_to_nii(sid, data4sharing_root, hemi=hemi)

                if not os.path.isfile(nii_path):
                    print(f"❌ {sid}: .nii.gz not found, skipping")
                    continue

                harv, aal = process_t1_mask_only(
                    mask_path=nii_path,
                    output_dir=report_dir,
                    base_name=sid,
                    t1_path=t1_path
                )

            if harv is None and aal is None:
                print(f"❌ {sid}: zero mask")
                continue

            # 4) Write to CSV
            writer.writerow([sid, harv, aal])
            csvfile.flush()


full_pipeline(
    data4sharing_root="/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/meld_files",
    report_log_path="/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/report_log.txt"
)
