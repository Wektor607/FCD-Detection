# This file descive how preproccess images and their mask in the atlas format 
# and get descriptions from atlas for mask
# 1. This is 1st step to get data which MELD take as input (1 image preprocess ~3.5 hours if you use
# FastSurger, otherwise it take ~6 hours): ./meldgraph.sh run_script_segmentation.py --fastsurfer 
# 2. This is 2nd step to get final format hdf5: ./meldgraph.sh run_script_preprocessing.py -ids subjects_list.csv
# For getting subjects_list.csv use get_subj_list() function
# 3. Get predictions from MELD: ./meldgraph.sh run_script_prediction.py -ids subjects_list.csv
# 4. Convert predicitons and masks into atlas format and get descriptions from atlas mask

import os
import csv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from get_reports import process_data
from meld_graph.paths import MELD_DATA_PATH
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
        name for name in filenames
        if name.startswith("sub-") and len(name) == 9
    ))

    df = pd.DataFrame(subject_ids, columns=["ID"])
    df.to_csv(os.path.join(project_path(output_folder), output_file), index=False)

    print(f"✅ Saved {len(subject_ids)} subject IDs to {output_file}")

def preprocess_func(list_ids, subj_path, harmo_code, full_name, transform_mode, json_disc):
    # Getting a list of pre-processed files
    # get_subj_list('data/output/fs_outputs', 'data', 'subjects_list.csv')

    # sys.exit()
    # Necessary for getting affine matrix for transformations
    # process_data()
    
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
    
    list_paths      = []
    report_paths    = []
    roi_paths       = []
    for subj_id in subject_ids:
        featmat_path = project_path(f"data/output/preprocessed_surf_data/{subj_id}_featurematrix_combat.hdf5")
        subj_new = project_path(os.path.join(subj_path, "preprocessed", subj_id))
        os.makedirs(subj_new, exist_ok=True)
        
        feat_dst = os.path.join(subj_new, os.path.basename(featmat_path))

        if not os.path.isfile(feat_dst):
            shutil.copy2(featmat_path, feat_dst)

        if find_file_with_suffix(project_path(f"data/input/ds004199/{subj_id}/anat"), f"_FLAIR_roi.nii.gz") is not None:    
            roi_path  = find_file_with_suffix(project_path(f"data/output/preprocessed_surf_data/MELD"), f"{subj_id}_featurematrix.hdf5")
            # roi_path  = find_file_with_suffix(project_path(f"data/input/ds004199/{subj_id}/anat"), "_FLAIR_roi.nii.gz")
            roi_dst = os.path.join(subj_new, os.path.basename(roi_path))
            if not os.path.isfile(roi_dst):
                shutil.copy2(roi_path, roi_dst)

            report_path = project_path(f"data/input/ds004199/{subj_id}/anat/report")
        else:            
            roi_path    = None
            report_path = None

        report_paths.append(report_path)
        roi_paths.append(roi_path)
        list_paths.append(subj_new)

    print(json_disc)
    generate_full_data(list_paths, 
                       roi_paths, 
                       report_paths,
                       project_path(os.path.join(subj_path, "preprocessed", f"Res_{transform_mode}.csv")), 
                       full_name,
                       transform_mode,
                       json_disc)

def extract_hemisphere(text: str) -> str:
    """Find Left/Right in the text and return e.g. 'Right hemisphere'."""
    m = re.search(r'\b(Left|Right)\b', text)
    return f"{m.group(1)} hemisphere" if m else text


_LOBE_MAP = {
    'Frontal':      'Frontal lobe',
    'Parietal':     'Parietal lobe',
    'Temporal':     'Temporal lobe',
    'Occipital':    'Occipital lobe',
    'Insular':      'Insular lobe',
    'Cingulate':    'Limbic lobe',
    'Parahippocampal': 'Limbic lobe',
    'Caudate':      'Subcortical nuclei',
    'Putamen':      'Subcortical nuclei',
    'Fusiform':     'Occipital lobe',
}


def extract_hemisphere_lobes(text: str) -> str:
    """
    Take a cleaned string like
      "Right Superior Frontal Gyrus; Right Insular Cortex"
    and turn it into
      "Right Frontal lobe; Right Insular lobe"
    keeping only unique hemisphere+lobe entries.
    """
    entries = []
    for part in re.split(r'\s*;\s*', text):
        hemi_m = re.search(r'\b(Left|Right)\b', part)
        hemi   = hemi_m.group(1) if hemi_m else None
        lobe   = None
        for key, val in _LOBE_MAP.items():
            if key in part:
                lobe = val
                break
        if hemi and lobe:
            entries.append(f"{hemi} {lobe}")
    # remove duplicates, preserve order
    seen = set()
    out  = []
    for e in entries:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return "; ".join(out) if out else text


def generate_full_data(
    list_paths,
    roi_paths,
    report_paths,
    result_file,
    full_name: bool,
    transform_mode: str = "full",   # <-- new!
    json_disc: dict = None
):
    """
    transform_mode:
      - "full"            : keep the original cleaned region string
      - "hemisphere"      : reduce each entry to "Left hemisphere"/"Right hemisphere"
      - "hemisphere_lobe" : reduce each entry to "Right Frontal lobe" etc.
    """
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

    abbrev_map = {
        r'\bSup\b': 'Superior',
        r'\bInf\b': 'Inferior',
        r'\bTri\b': 'Triangular',
        r'\bOper\b': 'Opercular',
        r'\bOrb\b': 'Orbital',
        r'\bMid\b': 'Middle',
        r'\bMed\b': 'Medial',
        r'\bSupp\b': 'Supplementary',
        r'\bOFCpost\b': 'Orbitofrontal_cortex_posterior',
    }


    def clean_region_string(text: str) -> str:
        if not isinstance(text, str):
            return text
        parts = re.split(r'[;,]', text)
        cleaned = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if re.search(r'\b(no_label|Unlabeled|no labels?)\b',
                         part, flags=re.IGNORECASE):
                continue
            cleaned.append(part)
        return "; ".join(cleaned).replace("_", " ")

    def apply_region_aliases(series, aliases):
        s = series
        for short, full in aliases.items():
            s = s.str.replace(short, full, regex=True)
        
        s = s.str.replace(r'\bR(?=;|\s*$)', 'Right', regex=True)
        # print("After: ", s)
        s = s.str.replace(r'\bL(?=;|\s*$)', 'Left',  regex=True)
        return s

    if json_disc is not None:
        new_csv = pd.DataFrame(columns=[
        'DATA_PATH', 'ROI_PATH', 'description'
        ])
        df = pd.read_csv(json_disc, sep='\t')
        for path, roi_hdf5_path, rep_name in zip(list_paths, roi_paths, df['report_text']):
            pred_path   = find_file_with_suffix(path, '_combat')            
            print(rep_name)
            new_csv.loc[len(new_csv)] = [
                pred_path,
                roi_hdf5_path,
                rep_name,
            ]
    else:    
        new_csv = pd.DataFrame(columns=[
        'DATA_PATH', 'ROI_PATH', 'harvard_oxford', 'aal'
        ])
        for path, roi_hdf5_path, report in zip(list_paths, roi_paths, report_paths):
            print(roi_hdf5_path)
            pred_path   = find_file_with_suffix(path, '_combat')
            if roi_hdf5_path == None:
                new_csv.loc[len(new_csv)] = [
                    pred_path,
                    roi_hdf5_path,
                    'No lesion detected',
                    'No lesion detected',
                ]
            else:
                report_path = find_file_with_suffix(report, 'z_trans_clusters')
                df          = pd.read_csv(report_path)

                for col in ['harvard_oxford', 'aal']:
                    df[col] = df[col].apply(clean_region_string)
                    if full_name:
                        df[col] = apply_region_aliases(df[col], abbrev_map)#region_aliases)
                    df[col] = df[col].apply(clean_region_string)
                    # **new**: apply the chosen transform mode
                    if transform_mode == "hemisphere":
                        df[col] = df[col].apply(extract_hemisphere)
                    elif transform_mode == "hemisphere_lobe":
                        df[col] = df[col].apply(extract_hemisphere_lobes)
                    elif transform_mode == "full+hemisphere":
                        hemi = df[col].apply(extract_hemisphere)
                        df[col] = df[col] + '; ' + hemi
                new_csv.loc[len(new_csv)] = [
                    pred_path,
                    roi_hdf5_path,
                    df['harvard_oxford'].iloc[0],
                    df['aal'].iloc[0],
                ]

    Path(result_file).parent.mkdir(parents=True, exist_ok=True)
    new_csv.to_csv(
    result_file,
    sep=',',                # таб-разделитель
    index=False,
    quoting=csv.QUOTE_NONE,  # не оборачивать поля в кавычки
    escapechar='\\',         # экранировать любые спецсимволы бэкслэшем
    )

if __name__ == "__main__":
    preprocess_func("subjects_list.csv", "data", "fcd", full_name=True, transform_mode="full+hemisphere", json_disc=None)#"/home/s17gmikh/FCD-Detection/meld_graph/data/pred_reports_summary.csv")
    # get_subj_list('data/output/fs_outputs',
    #           'data',
    #           'healthy_subjects_list.csv')
