import re
import sys
import os
import csv
import pandas as pd
from pathlib import Path

CURRENT_FILE = os.path.abspath(__file__)
# Maybe for your project you should change
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE, "..", "..")) 

sys.path.insert(0, PROJECT_ROOT)

def project_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)

# --- Configuration ---
INPUT_CSV   = project_path(os.path.join("data", "preprocessed", "Res_full.csv"))   # исходный файл
MODE        = "no_percentage"      # варианты: "full", "hemisphere", "hemisphere_lobe", "full+hemisphere", "dominant", "no_percentage"
OUTPUT_CSV  = project_path(os.path.join("data", "preprocessed", f"Res_{MODE}.csv"))
INVERSE     = False

# --- Helper Functions ---
def extract_hemisphere(text: str) -> str:
    print(text)
    m = re.search(r"\b(Left|Right)\b", text)
    return f"{m.group(1)} Hemisphere" if m else text

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
    entries = []
    for part in re.split(r"\s*;\s*", text):
        hemi_m = re.search(r"\b(Left|Right)\b", part)
        hemi   = hemi_m.group(1) if hemi_m else None
        lobe   = None
        for key, val in _LOBE_MAP.items():
            if key in part:
                lobe = val
                break
        if hemi and lobe:
            entries.append(f"{hemi} {lobe}")
    # remove duplicates
    seen, out = set(), []
    for e in entries:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return "; ".join(out) if out else text


def extract_dominant(text: str) -> str:
    # берём первую зону до ;, удаляем проценты
    if not isinstance(text, str) or ";" not in text:
        return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", text)
    first = text.split(";", 1)[0].strip()
    return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", first)

# --- Main Processing ---

def transform_region(text: str, mode: str, inverse: bool) -> str:
    if mode == "full":
        return text
    if mode == "hemisphere":
        return extract_hemisphere(text)
    if mode == "hemisphere_lobe":
        return extract_hemisphere_lobes(text)
    if mode == "full+hemisphere":
        hemi = extract_hemisphere(text)
        if INVERSE:
            if hemi == 'Left Hemisphere':
                hemi = 'No Right Hemisphere'
            if hemi == 'Right Hemisphere':
                hemi = 'No Left Hemisphere'
    if mode == "no_percentage":
        text = re.sub(r'\b\d+(?:\.\d+)?%\s*', '', text)
    if mode == "dominant":
        return extract_dominant(text)
    return text


def main():
    df = pd.read_csv(INPUT_CSV)
    for col in ['harvard_oxford']:#, 'aal']:
        # new_col = f"{col}_{MODE}"
        df[col] = df[col].apply(lambda txt: transform_region(txt, MODE, INVERSE) if txt != 'No lesion detected' else txt)

    # Сохраняем результат
    
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"✅ Saved transformed data to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
