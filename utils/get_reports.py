import os
import sys
import re
import nibabel as nib
import numpy as np

os.environ['FSLDIR'] = '/home/s17gmikh/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

from fsl.wrappers import flirt
from nilearn import datasets
from nilearn.image import resample_to_img
from atlasreader import create_output
import pandas as pd

def process_data():
    # Get absolute path to the directory where the script is located
    script_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))),
        'meld_graph',
        'data',
        'input')
    list_folders = []
    error_folders = {}
    
    start, end = get_existing_subject_ids(script_dir)
    for i in range(start, end + 1):
        base_name = f"sub-{i:05d}"
        folder = f"{base_name}/anat"
        full_path = os.path.join(script_dir, folder)
        report_dir = os.path.join(full_path, 'report')

        # Skip if report folder exists and is non-empty
        if os.path.isdir(report_dir) and any(os.listdir(report_dir)):
            print(f"Skipping {base_name}: report already exists")
            list_folders.append(full_path)
            continue

        if os.path.isdir(full_path):
            res = process_single_subject(full_path, 'report', base_name=base_name)
            if res == 'Processing complete':
                list_folders.append(full_path)
            else:
                error_folders.update(res)
            print(f'DONE {i:05d}')
        else:
            print(f'Skipped {i:05d}: folder not found')

    
    print("Error folders: ", error_folders)
    print("Processed folders:", list_folders)
    # generate_full_data(list_folders, 'Finall_file.csv')

def process_pred_subj(subj_path, output_dir, base_name='sub-00016', standard='MNI152_T1_2mm.nii.gz'):
    pred_path = os.path.join(subj_path, "predictions", "prediction.nii.gz")
    pred_out_path = os.path.join(output_dir, base_name + '_prediction_to_mni.nii.gz')
    mat_path = os.path.join("/home/s17gmikh/FCD-Detection/data/ds004199", base_name, "anat/generated", base_name + "_flair2mni.mat")
    fsldir = os.environ['FSLDIR']
    ref_path = os.path.join(fsldir, 'data/standard', standard)

    if os.path.isfile(pred_path):
        flirt(
            src=pred_path,
            ref=ref_path,
            applyxfm=True,
            init=mat_path,  # ← матрица трансформации!
            interp='nearestneighbour',
            out=pred_out_path,
            verbose=True
        )
    else:
        print(f"⚠️ Prediction file not found: {pred_path}")

process_pred_subj(subj_path = '/home/s17gmikh/FCD-Detection/meld_graph/data/output/predictions_reports/sub-00146',
                  output_dir = '/home/s17gmikh/FCD-Detection/meld_graph/data/output/predictions_reports/sub-00146',
                  base_name='sub-00146')

def process_single_subject(subj_path, output_dir, base_name='sub-00016', standard='MNI152_T1_2mm.nii.gz'):
    """
    Process a single subject's MRI data to identify and localize potential epileptogenic zones.

    This function performs the following steps:
        1. **Automatic file matching**: Searches the provided folder for T1w, FLAIR, and FLAIR_roi NIfTI files
        based on their suffixes (e.g., 'T1w', 'FLAIR', 'FLAIR_roi').
        2. **Registration to MNI space**: Uses FSL's `flirt` to register all images to the MNI152_T1_2mm template.
        The resulting files are saved in a folder named `preprocessed/`.
        3. **Intermediate storage**: Transformation matrices and temporary files are saved in `generated/`.
        4. **Z-map creation**: Converts the registered ROI mask into a pseudo Z-score map for visualization.
        5. **Resampling and atlas mapping**: Aligns the Z-map with the Harvard-Oxford cortical atlas.
        6. **Report generation**: Runs `atlasreader.create_output(...)` to generate visual reports and
        cluster/peak tables that describe where the epileptogenic region is likely located.

    Parameters:
        - subj_path (str): Path to the subject folder containing raw NIfTI files.
        - output_dir (str): Path to the folder where the final report will be saved.
        - base_name (str): The name of the folder where all the files are located
        - standard (str): FSL template name for registering images
    
    Assumptions:
    - Input folder contains exactly one file each for `T1w`, `FLAIR` (excluding roi), and `FLAIR_roi`.
    """

    fsldir = os.environ['FSLDIR']
    ref_path = os.path.join(fsldir, 'data/standard', standard)

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = atlas.maps
    atlas_data = atlas_img.get_fdata()
    z_value = np.max(np.unique(atlas_data))
    
    t1_path = find_file_with_suffix(subj_path, 'T1w')
    flair_path = find_file_with_suffix(subj_path, 'FLAIR', exclude='roi')
    roi_path = find_file_with_suffix(subj_path, 'FLAIR_roi')

    if t1_path is None or flair_path is None or roi_path is None:
        if t1_path is None:
            missing = 'T1 scan'
        elif flair_path is None:
            missing = 'FLAIR scan'
        elif roi_path is None:
            missing = 'FLAIR ROI mask'
        print(f"This folder: {base_name} does not contain {missing}")

        return {base_name: f'does not contain {missing}'}

    generated_dir = os.path.join(subj_path, 'generated')
    os.makedirs(generated_dir, exist_ok=True)

    preprocessed_dir = os.path.join(subj_path, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    t1_out_path = os.path.join(preprocessed_dir, base_name + '_t1_to_mni.nii.gz')
    flair_out_path = os.path.join(preprocessed_dir, base_name + '_flair_to_mni.nii.gz')
    roi_out_path = os.path.join(preprocessed_dir, base_name + '_roi_to_mni.nii.gz')
    mat_path = os.path.join(generated_dir, base_name + '_flair2mni.mat')

    flirt(src=t1_path, ref=ref_path, omat=mat_path, out=t1_out_path, verbose=True)
    flirt(src=flair_path, ref=ref_path, omat=mat_path, out=flair_out_path, verbose=True)
    flirt(src=roi_path, ref=ref_path, applyxfm=True, init=mat_path, interp='nearestneighbour', out=roi_out_path, verbose=True)    

    img = nib.load(roi_out_path)
    data = img.get_fdata()
    z_data = np.where(data > 0, z_value, 0.0).astype(np.float32)
    z_img = nib.Nifti1Image(z_data, img.affine, img.header)
    resampled = resample_to_img(z_img, atlas_img, interpolation='nearest')
    
    output_dir = os.path.join(subj_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(generated_dir, f'{os.path.basename(roi_out_path)}_zmap.nii.gz')
    nib.save(resampled, tmp_path)
    create_output(tmp_path, outdir=output_dir, cluster_extent=0, atlas=['harvard_oxford', 'aal'])

    print(f"Processing complete. Reports saved in: {output_dir}")

    return 'Processing complete'

def find_file_with_suffix(folder, include, exclude=None):
    matches = [
        f for f in os.listdir(folder)
        if (f.endswith('.nii.gz') or f.endswith('.csv')) and include in f and (exclude is None or exclude not in f)
    ]

    if not matches:
        return None
    
    return os.path.join(folder, matches[0])

def get_existing_subject_ids(root_dir):
    """
    Returns a sorted list of all subject numbers (as integers) from folders matching 'sub-XXXXX'.
    """
    subject_ids = []
    for name in os.listdir(root_dir):
        match = re.match(r'sub-(\d{5})$', name)
        if match:
            subject_ids.append(int(match.group(1)))
    sorted_list = sorted(subject_ids)
    return np.min(sorted_list), np.max(sorted_list)

# process_data()
