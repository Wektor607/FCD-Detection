import os
import nibabel as nib
import numpy as np
from fsl.wrappers import flirt
from nilearn import datasets
from nilearn.image import resample_to_img
from atlasreader import create_output

os.environ['FSLDIR'] = '/home/german-rivman/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def find_file_with_suffix(folder, include, exclude=None):
    matches = [
        f for f in os.listdir(folder)
        if f.endswith('.nii.gz') and include in f and (exclude is None or exclude not in f)
    ]
    return os.path.join(folder, matches[0])


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
    ref_path = os.path.join(fsldir, 'data', 'standard', standard)

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
    atlas_img = atlas.maps
    atlas_data = atlas_img.get_fdata()
    z_value = np.max(np.unique(atlas_data))
    
    t1_path = find_file_with_suffix(subj_path, 'T1w')
    flair_path = find_file_with_suffix(subj_path, 'FLAIR', exclude='roi')
    roi_path = find_file_with_suffix(subj_path, 'FLAIR_roi')

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

    os.makedirs(output_dir, exist_ok=True)
    tmp_path = os.path.join(generated_dir, f'{os.path.basename(roi_out_path)}_zmap.nii.gz')
    nib.save(resampled, tmp_path)
    create_output(tmp_path, outdir=output_dir, cluster_extent=0, atlas=['harvard_oxford', 'aal'])

    print(f"Processing complete. Reports saved in: {output_dir}")

process_single_subject('sub-00016/anat', 'sub-00016/anat/report', 'sub-00016')
