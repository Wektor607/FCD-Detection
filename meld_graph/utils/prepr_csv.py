import re
import sys
import os
import csv
import random
import pandas as pd
from pathlib import Path

CURRENT_FILE = os.path.abspath(__file__)
# Maybe for your project you should change
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_FILE, "..", "..")) 

sys.path.insert(0, PROJECT_ROOT)

def project_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)

# --- Configuration ---
file_name = 'MELD_BONN_dataset_augmented'
mode = input()
INPUT_CSV   = project_path(os.path.join("data", "preprocessed", f"{file_name}.csv"))   # исходный файл
MODE        = mode      # варианты: "full", "hemisphere", "lobe", "hemisphere_lobe", "full+hemisphere", "dominant", "no_percentage"
OUTPUT_CSV  = project_path(os.path.join("data", "preprocessed", f"MELD_BONN_{MODE}.csv"))
INVERSE     = False

# --- Helper Functions ---
def extract_hemisphere(text: str) -> str:
    m = re.search(r"\b(Left|Right)\b", text)
    return f"{m.group(1)} Hemisphere" if m else text

# _LOBE_MAP = {
#     'Frontal':      'Frontal lobe',
#     'Parietal':     'Parietal lobe',
#     'Temporal':     'Temporal lobe',
#     'Occipital':    'Occipital lobe',
#     'Insular':      'Insular lobe',
#     'Cingulate':    'Limbic lobe',
#     'Parahippocampal': 'Limbic lobe',
#     'Caudate':      'Subcortical nuclei',
#     'Putamen':      'Subcortical nuclei',
#     'Fusiform':     'Occipital lobe',
# }

def get_unique_lobes(df, col):
    lobes = set()
    for val in df[col]:
        if isinstance(val, str):
            val = val.replace("_", " ")
            val = re.sub(r'\b\d+(?:\.\d+)?%\s*', '', val)
            for lobe in [x.strip() for x in val.split(';')]:
                if lobe and lobe.lower() not in ['no lesion detected', 'no label']:
                    lobes.add(lobe)
    return sorted(lobes)

def extracts_wrong_lobe_hemi(text, unique_lobes, mode="wrong_lobe"):
    if not isinstance(text, str):
        return text
    parts = [x.strip() for x in text.split(';') if x.strip()]
    # Определяем все встречающиеся полушария
    hemis = set()
    for part in parts:
        m = re.match(r"^(Left|Right) ", part)
        if m:
            hemis.add(m.group(1))
    # Выбираем неверное полушарие
    wrong_hemi = None
    if hemis:
        wrong_hemi = "Left" if "Right" in hemis else "Right"
    else:
        wrong_hemi = random.choice(["Left", "Right"])
    # Убираем Left/Right из долей
    clean_parts = [re.sub(r"^(Left|Right) ", "", p) for p in parts]
    # Перемешиваем доли на неправильные
    new_parts = []
    for lobe in clean_parts:
        choices = [x for x in unique_lobes if re.sub(r"^(Left|Right) ", "", x) != lobe]
        if choices:
            wrong_lobe = random.choice(choices)
            wrong_lobe_clean = re.sub(r"^(Left|Right) ", "", wrong_lobe)
            new_parts.append(wrong_lobe_clean)
        else:
            new_parts.append(lobe)
    if mode == "wrong_lobe":
        # Добавляем неверное полушарие только один раз в начало
        return f"{wrong_hemi} {'; '.join(new_parts)}" if new_parts else text
    elif mode == "wrong_lobe_hemi":
        # Добавляем неверное полушарие в начало, убираем все Left/Right
        return f"{wrong_hemi} Hemisphere; {'; '.join(new_parts)}" if new_parts else text
    else:
        return "; ".join(new_parts)

def extract_hemisphere_lobes(text: str, mode: str) -> str:
    hemi_name = extract_hemisphere(text)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\s*%\s*", "", text)
    parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    clean_parts = []
    for part in parts:
        clean = re.sub(r"^(Left|Right) ", "", part)
        if clean not in clean_parts:
            clean_parts.append(clean)
    if mode == 'lobe':
        # Вынести Left/Right в начало, если все доли одного полушария
        hemis = set()
        for part in parts:
            m = re.match(r"^(Left|Right) ", part)
            if m:
                hemis.add(m.group(1))
        hemi_prefix = ''
        if len(hemis) == 1:
            hemi_prefix = list(hemis)[0] + ' '
        final = hemi_prefix + "; ".join(clean_parts)
        return final if clean_parts else text
    else:  # hemisphere_lobe
        # Добавляем hemi_name в начало, если его нет среди долей
        out = clean_parts
        if hemi_name and all(hemi_name not in s for s in out):
            out = [hemi_name] + out
        final = "; ".join(out)
        return final if out else text

def extract_dominant(text: str) -> str:
    # берём первую зону до ;, удаляем проценты
    if not isinstance(text, str) or ";" not in text:
        return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", text)
    first = text.split(";", 1)[0].strip()
    return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", first)

# --- Main Processing ---

def transform_region(text: str, unique_lobes: set, mode: str, inverse: bool) -> str:
    text = text.replace("_", " ")
    if mode == "full":
        return text
    if mode == "hemisphere":
        return extract_hemisphere(text)
    if mode == "hemisphere_lobe" or mode == 'lobe':
        return extract_hemisphere_lobes(text, mode)
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
    if mode == "replace_underscores":
        return text.replace("_", " ") if isinstance(text, str) else text
    if mode == "wrong_hemisphere":
        hemi = extract_hemisphere(text)
        swap = {"Left Hemisphere": "Right Hemisphere", "Right Hemisphere": "Left Hemisphere"}
        return swap.get(hemi, text)
    if mode == "wrong_lobe":
        text = extracts_wrong_lobe_hemi(text, unique_lobes, mode="wrong_lobe")
    if mode == "wrong_lobe_hemi":
        text = extracts_wrong_lobe_hemi(text, unique_lobes, mode="wrong_lobe_hemi")
    return text


def main():
    df = pd.read_csv(INPUT_CSV)
    unique_lobes = get_unique_lobes(df, 'harvard_oxford')
    for col in ['harvard_oxford']:#, 'aal']:
        df[col] = df[col].apply(lambda txt: transform_region(txt, unique_lobes, MODE, INVERSE) if txt != 'No lesion detected' else txt)

    # Сохраняем результат
    
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"✅ Saved transformed data to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
