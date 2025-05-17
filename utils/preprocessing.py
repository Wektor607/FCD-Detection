# This file descive how preproccess images and their mask in the atlas format 
# and get descriptions from atlas for mask
# 1. This is 1st step to get data which MELD take as input (1 image preprocess ~3.5 hours if you use
# FastSurger, otherwise it take ~6 hours): ./meldgraph.sh run_script_segmentation.py --fastsurfer 
# 2. This is 2nd step to get final format hdf5: ./meldgraph.sh run_script_preprocessing.py -ids subjects_list.csv
# For getting subjects_list.csv use get_subj_list() function
# 3. Get predictions from MELD: ./meldgraph.sh run_script_prediction.py -ids subjects_list.csv
# 4. Convert predicitons and masks into atlas format and get descriptions from atlas mask

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from fsl.wrappers import flirt
from nilearn import datasets, image
from get_reports import process_data

from meld_graph.meld_graph.paths import MELD_DATA_PATH
from utils.get_reports import find_file_with_suffix

CURRENT_FILE = os.path.abspath(__file__)
# Maybe for your project you should change
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE, "..", "..")) 

sys.path.insert(0, PROJECT_ROOT)


def project_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)

def get_subj_list(folder, output_folder, output_file):
    filenames = os.listdir(project_path(folder))

    subject_ids = sorted(set(
        name.split('_')[0]
        for name in filenames
        if name.endswith(".hdf5")
    ))

    df = pd.DataFrame(subject_ids, columns=["ID"])
    df.to_csv(os.path.join(project_path(output_folder), output_file), index=False)

    print(f"✅ Saved {len(subject_ids)} subject IDs to {output_file}")

def preprocess_func(list_ids, subj_path):
    # Necessary for getting affine matrix for transformations
    # process_data()

    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', symmetric_split=True)
    atlas_img = atlas['maps']

    list_ids=os.path.join(MELD_DATA_PATH, list_ids)
    try:
        sub_list_df=pd.read_csv(list_ids)
        subject_ids=np.array(sub_list_df.ID.values)
    except:
        print(f'⚠️ Could not open CSV with pandas, trying loadtxt:', None, 'WARNING')
        try:
            subject_ids = np.loadtxt(list_ids, dtype='str', ndmin=1)
        except Exception as e2:
            sys.exit(f'❌ Could not load list_ids with any method: {e2}', None, 'ERROR')
    
    list_paths  = []
    report_paths = []
    print(subject_ids)
    for subj_id in subject_ids:
        # pred_path = project_path(f"meld_graph/data/output/predictions_reports/{subj_id}/predictions/prediction.nii.gz")
        featmat_path = project_path(f"meld_graph/data/output/preprocessed_surf_data/{subj_id}_featurematrix_combat.hdf5")
        roi_path  = find_file_with_suffix(project_path(f"meld_graph/data/input/{subj_id}/anat"), 'FLAIR_roi')

        subj_new = project_path(os.path.join(subj_path, subj_id))
        generated_dir = os.path.join(subj_new, "temp")
        os.makedirs(generated_dir, exist_ok=True)
        
        mat_path = os.path.join(project_path(f"meld_graph/data/input/{subj_id}/anat/generated"), f"{subj_id}_flair2mni.mat")
        roi_out = os.path.join(generated_dir, "roi_in_mni.nii.gz")
        # pred_out = os.path.join(generated_dir, "pred_in_mni.nii.gz")

        # === 1. FLIRT: приведение ROI и Prediction к пространству атласа ===
        # flirt(src=pred_path, ref=atlas_img, omat=mat_path, out=pred_out, applyxfm=True, init=mat_path,
        #     interp='nearestneighbour', verbose=True)
        flirt(src=roi_path, ref=atlas_img, omat=mat_path, out=roi_out, applyxfm=True, init=mat_path,
            interp='nearestneighbour', verbose=True)

        # === 2. Z-map: создаём из ROI или Prediction ===
        atlas_data = atlas_img.get_fdata()
        z_value = np.max(np.unique(atlas_data))
        img_roi = nib.load(roi_out)  # можно заменить на roi_out для ROI
        # img_pred = nib.load(pred_out)  # можно заменить на roi_out для ROI
        data_roi = img_roi.get_fdata()
        # data_pred = img_pred.get_fdata()
        z_data_roi = np.where(data_roi > 0, z_value, 0.0).astype(np.float32)
        # z_data_pred = np.where(data_pred > 0, z_value, 0.0).astype(np.float32)
        z_roi = nib.Nifti1Image(z_data_roi, img_roi.affine, img_roi.header)
        # z_pred = nib.Nifti1Image(z_data_pred, img_roi.affine, img_roi.header)

        # === 3. Пересемплируем в сетку атласа ===
        resampled_roi = image.resample_to_img(z_roi, atlas_img, interpolation='nearest')
        # resampled_pred = image.resample_to_img(z_pred, atlas_img, interpolation='nearest')

        # === 4. Сохраняем Z-map и запускаем create_output ===
        zroi_path = os.path.join(subj_new, "roi_in_atlas.nii.gz")
        # zpred_path = os.path.join(generated_dir, "pred_in_atlas.nii.gz")
        nib.save(resampled_roi, zroi_path)
        shutil.copy2(featmat_path, subj_new)
        # nib.save(resampled_pred, zpred_path)

        list_paths.append(subj_new)
        report_paths.append(project_path(f"meld_graph/data/input/{subj_id}/anat/report"))
    generate_full_data(list_paths, report_paths, project_path(os.path.join(subj_path, "NewFinal.csv")))

def generate_full_data(list_paths, report_paths, result_file):
    """
    Parameters:
        - 
    """
    # new_csv = pd.DataFrame(columns=['T1w_path', 'FLAIR_path', 'FLAIR_ROI_path', 'harvard_oxford', 'aal'])
    new_csv = pd.DataFrame(columns=['PRED_PATH', 'ROI_PATH', 'harvard_oxford', 'aal'])

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

    for path, report in zip(list_paths, report_paths):
        # t1_path = find_file_with_suffix(path, 'T1w')
        # flair_path = find_file_with_suffix(path, 'FLAIR', exclude='roi')
        # roi_path = find_file_with_suffix(path, 'FLAIR_roi')
        
        roi_path  = find_file_with_suffix(path, 'roi_in_atlas')
        pred_path = find_file_with_suffix(path, 'pred_in_atlas')
        print(f"Processing report for: {path}")

        report_path = find_file_with_suffix(report, 'clusters')
        data = pd.read_csv(report_path)

        for col_name in ['aal', 'harvard_oxford']:
            data[col_name] = apply_region_aliases(data[col_name], region_aliases)
            data[col_name] = data[col_name].apply(clean_region_string)

        new_csv.loc[len(new_csv)] = [
            pred_path,
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

# get_subj_list('meld_graph/data/output/preprocessed_surf_data',
#               'meld_graph/data',
#               'subjects_list.csv')


preprocess_func("subjects_list.csv", "example_output")