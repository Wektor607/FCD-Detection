import os
import sys
import re
import ants
import numpy as np
import nibabel as nib
import json
from nilearn import datasets
from atlasreader import create_output
from nilearn.datasets import fetch_icbm152_2009
import csv

# ── NEW: load all pred_reports into a dict keyed by subject ID ──
import os
import re
import json

# удаляем любые кавычки: " ' “ ” ‘ ’
_RE_QUOTES = re.compile(r'[\"\'\u201c\u201d\u2018\u2019]')

def load_pred_reports(json_path):
    """
    Читает JSON с полями:
      - "image": ["/…/sub-XXXXX_…"]
      - "pred_report": "…"
    Возвращает dict: subj_id -> предобработанный текст отчёта
    """
    with open(json_path, 'r') as f:
        entries = json.load(f)

    report_map = {}
    for e in entries:
        img  = os.path.basename(e["image"][0])
        subj = img.split('_')[0]            # "sub-XXXXX"
        text = e.get("pred_report", "") \
                .replace('\n', ' ') \
                .strip()
        # 1) Сжать пробелы
        text = re.sub(r'\s+', ' ', text)
        # 2) Удалить кавычки
        text = _RE_QUOTES.sub('', text)
        report_map[subj] = text

    return report_map

def save_pred_reports_raw(report_map, out_path):
    """
    Writes a simple CSV without any quoting:
      subject_id,report_text
    Any commas in report_text will be literal commas (so this is not a fully compliant CSV).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["subject_id", "report_text"])
        for subj, report in sorted(report_map.items()):
            # убираем переносы строк, оставшиеся табуляции не сломают CSV, потому что они экранируются
            clean = report.replace('\n', ' ').replace('\r', ' ')
            writer.writerow([subj, clean])

# point this to your JSON
PRED_JSON = "/home/s17gmikh/FCD-Detection/meld_graph/data/pred_report.json"
PRED_REPORTS = load_pred_reports(PRED_JSON)
save_pred_reports_raw(PRED_REPORTS, "/home/s17gmikh/FCD-Detection/meld_graph/data/pred_reports_summary.csv")

# ────────────────────────────────────────────────────────────────

def process_data():
    # Get absolute path to the directory where the script is located
    script_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))),
        # 'meld_graph',
        'data',
        'input',
        'ds004199')
    list_folders = []
    error_folders = {}
    
    start, end = get_existing_subject_ids(script_dir)
    for i in range(start, end + 1):
        base_name = f"sub-{i:05d}"
        folder = f"{base_name}/anat"
        full_path = os.path.join(script_dir, folder)
        if find_file_with_suffix(full_path, 'FLAIR_roi') is None:
            continue
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

    preprocessed_dir = os.path.join(subj_path, 'preprocessed')
    os.makedirs(preprocessed_dir, exist_ok=True)

    report_dir = os.path.join(subj_path, output_dir)
    os.makedirs(report_dir, exist_ok=True)

    icbm = fetch_icbm152_2009()
    mni_path = icbm.t1
    mni_img = ants.image_read(mni_path)
    
    t1_path = find_file_with_suffix(subj_path, 'T1w')
    flair_path = find_file_with_suffix(subj_path, 'FLAIR', exclude='roi')
    roi_path = find_file_with_suffix(subj_path, 'FLAIR_roi')

    roi_out_path = os.path.join(preprocessed_dir, base_name + '_roi_to_mni.nii.gz')

    if t1_path is None or flair_path is None or roi_path is None:
        if t1_path is None:
            missing = 'T1 scan'
        elif flair_path is None:
            missing = 'FLAIR scan'
        elif roi_path is None:
            missing = 'FLAIR ROI mask'
        print(f"This folder: {base_name} does not contain {missing}")

        return {base_name: f'does not contain {missing}'}
    
    # Read data
    t1_img = ants.image_read(t1_path)
    flair_img = ants.image_read(flair_path)
    roi_img = ants.image_read(roi_path)

    # Registration: FLAIR → T1
    reg_flair2t1 = ants.registration(fixed=t1_img, 
                                    moving=flair_img, 
                                    type_of_transform='SyN')

    # Transform: ROI -> T1
    roi_in_t1 = ants.apply_transforms(
        fixed=t1_img,
        moving=roi_img,
        transformlist=reg_flair2t1['fwdtransforms'],
        whichtoinvert=[False, True], 
        interpolation='genericLabel'
    )

    # Registration: T1 → MNI
    reg_t1_to_mni = ants.registration(
        fixed=mni_img,
        moving=t1_img,
        type_of_transform='Affine',
    )

    # Transform: T1 -> MNI
    roi_in_mni = ants.apply_transforms(
        fixed=mni_img,
        moving=roi_in_t1,
        transformlist=reg_t1_to_mni['fwdtransforms'],
        whichtoinvert=[False], 
        interpolation='genericLabel'
    )

    ants.image_write(roi_in_mni, roi_out_path)

    # Download atlas
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', symmetric_split=True)
    atlas_img = atlas.maps
    atlas_data = atlas_img.get_fdata()
    z_value = int(np.max(np.unique(atlas_data)))

    # Get ROI-mask in nib format
    img = nib.load(roi_out_path)
    data = img.get_fdata()

    z_data = np.where(data > 0, z_value, 0).astype(np.float32)

    z_img = nib.Nifti1Image(z_data, affine=img.affine, header=img.header)
    roi_z_out_path = os.path.join(preprocessed_dir, base_name + '_roi_z_trans.nii.gz')
    nib.save(z_img, roi_z_out_path)

    # Generating report
    create_output(
        roi_z_out_path,
        outdir=report_dir,
        cluster_extent=0,
        atlas=['harvard_oxford', 'aal']
    )

    print(f"Processing complete. Reports saved in: {report_dir}")

    return 'Processing complete'

def find_file_with_suffix(folder, include, exclude=None):
    matches = [
        f for f in os.listdir(folder)
        if (f.endswith('.nii.gz') or f.endswith('.csv') or f.endswith('.hdf5')) and include in f and (exclude is None or exclude not in f)
    ]

    if not matches:
        return None
    
    return os.path.join(folder, matches[0])

def get_existing_subject_ids(root_dir):
    """
    Returns a sorted list of all subject numbers (as integers) from folders matching 'sub-XXXXX'.
    """
    subject_ids = []
    print(root_dir)
    for name in os.listdir(root_dir):
        match = re.match(r'sub-(\d{5})$', name)
        if match:
            subject_ids.append(int(match.group(1)))
    sorted_list = sorted(subject_ids)
    print(sorted_list)
    return np.min(sorted_list), np.max(sorted_list)

if __name__=="__main__":
    process_data()
