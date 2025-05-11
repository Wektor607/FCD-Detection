import os
import sys
import re
import nibabel as nib
import numpy as np
from fsl.wrappers import flirt
from nilearn import datasets
from nilearn.image import resample_to_img
from atlasreader import create_output
import pandas as pd

os.environ['FSLDIR'] = '/home/german-rivman/fsl'
os.environ['PATH'] = os.environ['FSLDIR'] + '/bin:' + os.environ['PATH']
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'

def process_data():
    # Get absolute path to the directory where the script is located
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    list_folders = []
    error_folders = {}
    
    start, end = get_existing_subject_ids(script_dir)
    for i in range(start, end + 1):
        base_name = f"sub-{i:05d}"
        folder = f"{base_name}/anat"
        full_path = os.path.join(script_dir, folder)
        report_dir = os.path.join(full_path, 'report')

        # # Skip if report folder exists and is non-empty
        # if os.path.isdir(report_dir) and any(os.listdir(report_dir)):
        #     print(f"Skipping {base_name}: report already exists")
        #     list_folders.append(full_path)
        #     continue

        # if os.path.isdir(full_path):
        #     res = process_single_subject(full_path, 'report', base_name=base_name)
        #     if res == 'Processing complete':
        #         list_folders.append(full_path)
        #     else:
        #         error_folders.update(res)
        #     print(f'DONE {i:05d}')
        # else:
        #     print(f'Skipped {i:05d}: folder not found')

    
        print("Error folders: ", error_folders)
        print("Processed folders:", list_folders)
        full_pathh = os.path.join(full_path, 'report')
        if os.path.isdir(full_pathh):
            print(i)
            generate_full_data([full_path], 'Finall_file.csv')
        # generate_full_data(list_folders, 'Finall_file.csv')

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

def generate_full_data(list_paths, result_file):
    """
    Parameters:
        - 
    """
    new_csv = pd.DataFrame(columns=['T1w_path', 'FLAIR_path', 'FLAIR_ROI_path', 'harvard_oxford', 'aal'])

    region_aliases = {
        '%': ' percent of',
        '_': ' ',

        # Cenral region
        'Precentral': 'Precentral Gyrus',
        'Postcentral': 'Postcentral Gyrus',
        'Rolandic Oper': 'Rolandic operculum',

        # Frontal lobe
        ## Lateral surface
        'Frontal Sup': 'Superior Frontal Gyrus',
        'Frontal Mid': 'Middle Frontal Gyrus',
        'Frontal Inf Oper': 'Inferior Frontal Gyrus, Opercular part',
        'Frontal Inf Tri': 'Inferior Frontal Gyrus, Triangular part',
        ## Medial surface
        'Frontal Sup Medial': 'Superior Frontal Gyrus, Medial',
        'Supp Motor Area': 'Supplementary Motor Area',
        
        ## Orbital surface
        'Frontal Inf Orb': 'Inferior Frontal Gyrus, Orbital part',
        'Rectus': 'Gyrus rectus',
        'Olfactory': 'Olfactory cortex',

        # Temporal lobe
        ## Lateral surface
        'Temporal Sup': 'Superior Temporal Gyrus',
        'Heschl ': 'Heschl Gyrus ',
        'Temporal Mid': 'Middle Temporal Gyrus',
        'Temporal Inf': 'Inferior Temporal Gyrus',

        # Parietal lobe
        ## Lateral surface
        'Parietal Sup': 'Superior Parietal Gyrus',
        'Parietal Inf': 'Inferior Parietal Lobule (including supramarginal and angular gyri)',
        'Angular': 'Angular Gyrus',
        'SupraMarginal': 'Supramarginal Gyrus', 
        ## Medial surface
        
        # Occipital lobe
        ## Lateral surface
        'Occipital Sup': 'Superior Occipital Gyrus',
        'Occipital Mid': 'Middle Occipital Gyrus',
        'Occipital Inf': 'Inferior Occipital Gyrus',
        ## Medial and inferior surfaces
        ' Calcarine ': ' Calcarine fissure and surrounding cortex ',
        'Lingual': 'Lingual Gyrus',
        'Fusiform Left': 'Fusiform Gyrus Left',
        'Fusiform Right': 'Fusiform Gyrus Right',

        # Limbic lobe
        'Temporal Pole Sup': 'Temporal Pole: Superior Temporal Gyrus',
        'Cingulate Ant': 'Anterior Cingulate and Paracingulate gyri',
        'Cingulate Mid': 'Median Cingulate and Paracingulate gyri',
        'Cingulate Post': 'Posterior Cingulate gyrus',
        'Parahippocampal': 'Parahippocampal gyrus',

        # Sub cortical gray nuclei
        'Caudate': 'Caudate nucleus',
        'Putamen': 'Lenticular nucleus, putamen',

        'Gyrus Gyrus': 'Gyrus',
        'gyrus Gyrus': 'Gyrus',
        'no label': 'Unlabeled region'
    }

    for path in list_paths:
        t1_path = find_file_with_suffix(path, 'T1w')
        flair_path = find_file_with_suffix(path, 'FLAIR', exclude='roi')
        roi_path = find_file_with_suffix(path, 'FLAIR_roi')
        print(f"Processing report for: {path}")

        report_path = find_file_with_suffix(os.path.join(path, 'report'), 'clusters')
        data = pd.read_csv(report_path)

        for col_name in ['aal', 'harvard_oxford']:
            data[col_name] = apply_region_aliases(data[col_name], region_aliases)
            data[col_name] = data[col_name].apply(clean_region_string)

        new_csv.loc[len(new_csv)] = [
            t1_path,
            flair_path,
            roi_path,
            data['harvard_oxford'].iloc[0],
            data['aal'].iloc[0]
        ]

    if os.path.exists(result_file):
        old_csv = pd.read_csv(result_file)
        combined = pd.concat([old_csv, new_csv], ignore_index=True)
        combined.drop_duplicates(inplace=True)  # если нужно удалить повторы
        combined.to_csv(result_file, index=False)
    else:
        new_csv.to_csv(result_file, index=False)

def find_file_with_suffix(folder, include, exclude=None):
    matches = [
        f for f in os.listdir(folder)
        if (f.endswith('.nii.gz') or f.endswith('.csv')) and include in f and (exclude is None or exclude not in f)
    ]

    if not matches:
        return None
    
    return os.path.join(folder, matches[0])

def clean_region_string(text):
    """Remove 'Unlabeled' and normalize spacing/punctuation"""
    if not isinstance(text, str):
        return text
    # Remove 'Unlabeled' and 'no labels' entries
    text = re.sub(r'\s*\d+(\.\d+)?\s*percent of (Unlabeled region|no labels)', '', text)
    # Remove word dublicates: "Gyrus Gyrus", "cortex cortex", "Lobe Lobe", и т.д.
    # text = text.replace('Gyrus Gyrus', 'Gyrus')
    # Normalize semicolons
    text = re.sub(r';{2,}', ';', text).strip('; ').strip()
    return text

def apply_region_aliases(series, aliases):
    """Replace region name tokens in a Series using provided mapping"""
    for short, full in aliases.items():
        series = series.str.replace(short, full, regex=True)
    
    # Handle R/L at end of words (not followed by ;)
    series = series.apply(lambda text: re.sub(r'\bR(?=;|\s*$)', 'Right', text))
    series = series.apply(lambda text: re.sub(r'\bL(?=;|\s*$)', 'Left', text))
    return series

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

process_data()