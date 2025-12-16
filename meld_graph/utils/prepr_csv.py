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
# file_name = 'MELD_BONN_dataset_augmented'
file_name = 'MELD_BONN_dataset_augmented_final'
mode = input()
convert_lobe = True
INPUT_CSV   = project_path(os.path.join("data", "preprocessed", f"{file_name}.csv"))   # исходный файл
MODE        = mode      # варианты: "full", "hemisphere", "lobe", "hemisphere_lobe_regions", "full+hemisphere", "dominant", "no_percentage"

# if convert_lobe:
#     OUTPUT_CSV  = project_path(os.path.join("data", "preprocessed", f"MELD_BONN_{MODE}_converted_lobe.csv"))
# else:
#     OUTPUT_CSV  = project_path(os.path.join("data", "preprocessed", f"MELD_BONN_{MODE}.csv"))

# if convert_lobe:
#     OUTPUT_CSV  = project_path(os.path.join("data", "preprocessed", "final_aug_text", f"MELD_BONN_{MODE}_converted_lobe.csv"))
# else:
OUTPUT_CSV  = project_path(os.path.join("data", "preprocessed", "final_aug_text", f"MELD_BONN_{MODE}.csv"))

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

# --- Mapping of detailed regions to general lobes ---
REGION_TO_LOBE = {
    # Typo
    "Lateral Ventrical": "Lateral Ventricle",
    # Central region
    "Paracingulate Gyrus": "Frontal lobe",
    "Precentral gyrus": "Frontal lobe",
    "Postcentral gyrus": "Parietal lobe",
    "Rolandic operculum": "Frontal lobe",

    # Frontal lobe
    "Frontal Medial Cortex": "Frontal lobe",
    "Frontal Operculum Cortex": "Frontal lobe",
    "Frontal Orbital Cortex": "Frontal lobe",
    "Frontal Pole": "Frontal lobe",
    "Superior frontal gyrus dorsolateral": "Frontal lobe",
    "Superior Frontal Gyrus": "Frontal lobe",
    "Middle frontal gyrus": "Frontal lobe",
    "Inferior frontal gyrus opercular part": "Frontal lobe",
    "Inferior Frontal Gyrus pars opercularis": "Frontal lobe",
    "Inferior frontal gyrus pars opercularis": "Frontal lobe",
    "Inferior frontal gyrus triangular part": "Frontal lobe",
    "Inferior Frontal Gyrus pars triangularis": "Frontal lobe",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "Frontal lobe",

    "Superior frontal gyrus medial": "Frontal lobe",
    "Supplementary motor area": "Frontal lobe",
    "Paracentral lobule": "Frontal lobe",
    "Superior frontal gyrus orbital part": "Frontal lobe",
    "Middle frontal gyrus orbital part": "Frontal lobe",
    "Inferior frontal gyrus orbital part": "Frontal lobe",
    "Gyrus rectus": "Frontal lobe",
    "Olfactory cortex": "Frontal lobe",

    # Temporal lobe
    "Superior temporal gyrus": "Temporal lobe",
    "Heschl gyrus": "Temporal lobe",
    "Middle temporal gyrus": "Temporal lobe",
    "Inferior temporal gyrus": "Temporal lobe",
    "Planum Temporale": "Temporal lobe",
    "Planum Polare": "Temporal lobe",
    "Temporal Fusiform Cortex anterior division": "Temporal lobe",
    "Heschl's Gyrus (includes H1 and H2)": "Temporal lobe",
    "Temporal Fusiform Cortex posterior division": "Temporal lobe",
    "Temporal Occipital Fusiform Cortex": "Occipital lobe",

    # Parietal lobe
    "Superior parietal gyrus": "Parietal lobe",
    "Superior Parietal Lobule": "Parietal lobe",
    "Inferior parietal but supramarginal and angular gyrus": "Parietal lobe",
    "Angular gyrus": "Parietal lobe",
    "Supramarginal gyrus": "Parietal lobe",
    "Precuneus": "Parietal lobe",
    "Precuneous Cortex": "Parietal lobe",
    "Parietal Operculum Cortex": "Parietal lobe",

    # Occipital lobe
    "Superior occipital gyrus": "Occipital lobe",
    "Lateral Occipital Cortex superior division": "Occipital lobe",
    "Middle occipital gyrus": "Occipital lobe",
    "Inferior occipital gyrus": "Occipital lobe",
    "Cuneus": "Occipital lobe",
    "Calcarine fissure and surrounding cortex": "Occipital lobe",
    "Lingual gyrus": "Occipital lobe",
    "Fusiform gyrus": "Occipital lobe",
    "Occipital Pole": "Occipital lobe",
    "Supracalcarine Cortex": "Occipital lobe",
    "Lateral Occipital Cortex inferior division": "Occipital lobe",
    "Intracalcarine cortex": "Occipital lobe",
    "Cuneal Cortex": "Occipital lobe",
    # Limbic lobe
    "Temporal pole superior temporal gyrus": "Limbic lobe",
    "Temporal pole middle temporal gyrus": "Limbic lobe",
    "Temporal Pole": "Limbic lobe",
    "Anterior cingulate and paracingulate gyri": "Limbic lobe",
    "Median cingulate and paracingulate gyri": "Limbic lobe",
    "Posterior cingulate gyrus": "Limbic lobe",
    "Cingulate Gyrus posterior division": "Limbic lobe",
    "Hippocampus": "Limbic lobe",
    "Parahippocampal gyrus": "Limbic lobe",
    "Cingulate Gyrus anterior division": "Limbic lobe",

    # Insula
    "Insula": "Insular lobe",

    # Subcortical nuclei
    "Amygdala": "Subcortical nuclei",
    "Caudate nucleus": "Subcortical nuclei",
    "Lenticular nucleus putamen": "Subcortical nuclei",
    "Lenticular nucleus pallidum": "Subcortical nuclei",
    "Thalamus": "Subcortical nuclei",
    "Accumbens": "Subcortical nuclei",
}

def pick_dominant_lobe_from_percentages(text: str) -> str:
    """
    Получает строку формата:
        "30% Temporal lobe; 40% Temporal lobe; 30% Parietal lobe"
    Складывает проценты по лобам и возвращает ДОМАНИРУЮЩУЮ долю:
        "Temporal lobe"
    """
    if not isinstance(text, str):
        return text

    parts = [p.strip() for p in text.split(";") if p.strip()]
    lobe_totals = {}

    for part in parts:
        m = re.match(r"\s*(\d+(?:\.\d+)?)%\s*(.*)$", part)
        if not m:
            continue

        perc = float(m.group(1))
        lobe = m.group(2).strip()

        if lobe.lower() in ["no label", "no lesion detected"]:
            continue

        lobe_totals[lobe] = lobe_totals.get(lobe, 0.0) + perc

    if not lobe_totals:
        return "No lesion detected"

    # выбираем лоб с максимальным процентом
    dominant = max(lobe_totals, key=lobe_totals.get)
    return dominant

def extract_lobe_percentages(text: str) -> str:
    """
    Сохраняет проценты, конвертирует каждый регион → долю,
    сохраняет повторения (например 30% Temporal, 30% Temporal).
    """
    if not isinstance(text, str):
        return text

    parts = [p.strip() for p in text.split(";") if p.strip()]
    new_parts = []

    for part in parts:
        # Сохраняем процент (если есть)
        m = re.match(r"^\s*(\d+(?:\.\d+)?)%\s*(.*)$", part)
        if m:
            perc = m.group(1) + "%"
            region = m.group(2).strip()
        else:
            perc = ""
            region = part

        # Убираем Left/Right
        region_clean = re.sub(r"\b(Left|Right)\s+", "", region)

        # Ищем в словаре REGION_TO_LOBE
        found_lobe = None
        for key, lobe in REGION_TO_LOBE.items():
            if key.lower() in region_clean.lower():
                found_lobe = lobe
                break

        # Если не нашли — добавляем как есть
        lobe_name = found_lobe if found_lobe else region_clean

        if lobe_name == "no label":
            continue
        if perc:
            print(f"{perc} {lobe_name}")
            new_parts.append(f"{perc} {lobe_name}")
        else:
            new_parts.append(lobe_name)

    return "; ".join(new_parts)

def replace_region_with_lobe(text: str) -> str:
    """Заменяет конкретные анатомические регионы на их общие названия долей."""
    if not isinstance(text, str) or text.strip() == "":
        return text

    result_parts = []
    parts = [p.strip() for p in text.split(';') if p.strip()]
    for i, part in enumerate(parts):
        clean = part.replace("_", " ")
        clean = re.sub(r"[,;]", " ", clean)
        clean = re.sub(r"\b\d+(?:[.,]\d+)?\s*%\s*", "", clean)

        # ⚠️ Главное изменение: не убираем Left/Right для первого элемента, если это Hemisphere
        if not re.search(r"\bHemisphere\b", clean, re.IGNORECASE):
            clean = re.sub(r"\b(Left|Right)\s+", "", clean)

        clean = clean.strip()

        found_lobe = None
        for key, lobe in REGION_TO_LOBE.items():
            if key.lower() in clean.lower():
                found_lobe = lobe
                break

        result_parts.append(found_lobe if found_lobe else clean)

    seen = set()
    ordered_parts = []
    for p in result_parts:
        if p not in seen:
            seen.add(p)
            ordered_parts.append(p)
    return "; ".join(ordered_parts)

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

    # Определяем полушарие
    hemis = set()
    for part in parts:
        m = re.search(r"\b(Left|Right)\b", part)
        if m:
            hemis.add(m.group(1))
    wrong_hemi = "Left" if "Right" in hemis else "Right" if hemis else random.choice(["Left", "Right"])

    # Убираем проценты и стороны
    clean_parts = []
    for p in parts:
        p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
        p_clean = re.sub(r'\b(Left|Right)\s+', '', p_no_percent)
        clean_parts.append(p_clean)

    # Перемешиваем регионы
    new_parts = []
    for region in clean_parts:
        choices = [x for x in unique_lobes if re.sub(r'\b(Left|Right)\s+', '', x) != region]
        wrong_region = random.choice(choices) if choices else region
        new_parts.append(wrong_region)

    # 🟢 Добавляем преобразование регионов → лобов
    new_text = "; ".join(new_parts)
    new_text = replace_region_with_lobe(new_text)

    # Добавляем неверное полушарие
    if mode == "wrong_lobe_hemi":
        return f"{wrong_hemi} Hemisphere; {new_text}"
    elif mode == "wrong_hemisphere_only_correct_lobe":
        clean_text = "; ".join(clean_parts)
        clean_text = replace_region_with_lobe(clean_text)
        return f"{wrong_hemi} Hemisphere; {clean_text}"
    else:
        return new_text


def extracts_wrong_lobe_reg_hemi(text, unique_lobes, mode="wrong_lobe"):
    if not isinstance(text, str):
        return text
    parts = [x.strip() for x in text.split(';') if x.strip()]
    # Определяем все встречающиеся полушария
    hemis = set()
    print(parts)
    for part in parts:
        # Ищем Left или Right в любом месте строки
        m = re.search(r"\b(Left|Right)\b", part)
        if m:
            hemis.add(m.group(1))
    # Выбираем неверное полушарие
    print(hemis)
    wrong_hemi = None
    if hemis:
        wrong_hemi = "Left" if "Right" in hemis else "Right"
    else:
        wrong_hemi = random.choice(["Left", "Right"])
    
    if mode == "wrong_hemisphere_only_correct_regions":
        # Меняем только полушарие, доли оставляем как есть
        # Сначала убираем проценты, потом Left/Right
        clean_parts = []
        for p in parts:
            # Убираем проценты (в начале строки)
            p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
            # Убираем Left/Right с пробелом после
            p_clean = re.sub(r'\b(Left|Right)\s+', '', p_no_percent)
            clean_parts.append(p_clean)
        return f"{wrong_hemi} Hemisphere; {'; '.join(clean_parts)}" if clean_parts else text
    
    
    # Убираем проценты и Left/Right из долей
    clean_parts = []
    for p in parts:
        # Убираем проценты (в начале строки)
        p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
        # Убираем Left/Right с пробелом после
        p_clean = re.sub(r'\b(Left|Right)\s+', '', p_no_percent)
        clean_parts.append(p_clean)
    
    # Перемешиваем доли на неправильные
    new_parts = []
    for lobe in clean_parts:
        # Убираем Left/Right из unique_lobes для сравнения
        choices = []
        for x in unique_lobes:
            x_clean = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', x)
            x_clean = re.sub(r'\b(Left|Right)\s+', '', x_clean)
            if x_clean != lobe:
                choices.append(x_clean)
        if choices:
            wrong_lobe = random.choice(choices)
            new_parts.append(wrong_lobe)
        else:
            new_parts.append(lobe)
    if mode == "wrong_lobe_regions" or mode == "wrong_lobe_regions_hemi":
        # Добавляем неверное полушарие только один раз в начало
        return f"{wrong_hemi} Hemisphere; {'; '.join(new_parts)}" if new_parts else text
    else:
        return "; ".join(new_parts)


def extract_hemisphere_lobes(text: str, mode: str) -> str:
    hemi_name = extract_hemisphere(text)
    text = re.sub(r"\b\d+(?:[.,]\d+)?\s*%\s*", "", text)
    parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    clean_parts = []
    hemis = set()

    for part in parts:
        m = re.match(r"^(Left|Right)\b", part)
        if m:
            hemis.add(m.group(1))
        clean = re.sub(r"^(Left|Right)\s+", "", part)
        if clean not in clean_parts:
            clean_parts.append(clean)

    # Определяем полушарие
    hemi_prefix = ""
    if len(hemis) == 1:
        hemi_prefix = list(hemis)[0] + " Hemisphere"
    elif hemi_name:
        hemi_prefix = hemi_name

    # Формируем финальный порядок: сначала полушарие, потом доля(и)
    print(hemi_prefix)
    if mode == "lobe" or mode == "lobe_regions":
        # if hemi_prefix:
        #     return f"{hemi_prefix}; {'; '.join(clean_parts)}"
        # else:
        print(clean_parts)
        return "; ".join(clean_parts)
    else:  # hemisphere_lobe
        if hemi_prefix and all(hemi_prefix not in s for s in clean_parts):
            return f"{hemi_prefix}; {'; '.join(clean_parts)}"
        else:
            return "; ".join(clean_parts)

def extract_dominant(text: str) -> str:
    # берём первую зону до ;, удаляем проценты
    if not isinstance(text, str) or ";" not in text:
        return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", text)
    first = text.split(";", 1)[0].strip()
    return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", first)

# --- Main Processing ---

def transform_region(text: str, unique_lobes: set, mode: str, inverse: bool, convert_lobe: bool = True) -> str:
    text = text.replace("_", " ")
    # if mode == "full":
    #     return text
    if mode == "hemisphere":
        text = extract_hemisphere(text)
    if mode == "hemisphere_lobe" or mode == 'lobe' or mode == "hemisphere_lobe_regions" or mode == 'lobe_regions':
        text = extract_hemisphere_lobes(text, mode)
    if mode == "full+hemisphere":
        hemi = extract_hemisphere(text)
        if INVERSE:
            if hemi == 'Left Hemisphere':
                hemi = 'No Right Hemisphere'
            if hemi == 'Right Hemisphere':
                hemi = 'No Left Hemisphere'
    if mode == "no_percentage":
        text = re.sub(r'\b\d+(?:\.\d+)?%\s*', '', text)
    if mode == "dominant_lobe":
        text = extract_dominant(text)
    if mode == "replace_underscores":
        text = text.replace("_", " ") if isinstance(text, str) else text
    if mode == "wrong_hemisphere":
        hemi = extract_hemisphere(text)
        swap = {"Left Hemisphere": "Right Hemisphere", "Right Hemisphere": "Left Hemisphere"}
        text = swap.get(hemi, text)
    
    if mode == "wrong_lobe_hemi" or mode == "wrong_hemisphere_only_correct_lobe":
        text = extracts_wrong_lobe_hemi(text, unique_lobes, mode=mode)


    if mode == "wrong_lobe_regions":
        text = extracts_wrong_lobe_reg_hemi(text, unique_lobes, mode="wrong_lobe")
    if mode == "wrong_lobe_regions_hemi":
        text = extracts_wrong_lobe_reg_hemi(text, unique_lobes, mode="wrong_lobe_regions_hemi")
    if mode == "wrong_hemisphere_only_correct_regions":
        text = extracts_wrong_lobe_reg_hemi(text, unique_lobes, mode="wrong_hemisphere_only_correct_regions")
    
    if mode == "lobe_highest_percentages":
        # сохраняем проценты, но конвертируем каждый регион → его лоб
        text = extract_lobe_percentages(text)
        print(text)
        text = pick_dominant_lobe_from_percentages(text)
        print(text)
        print("------")
        return text

    if convert_lobe:
        text = replace_region_with_lobe(text)

    # Удаляем фрагменты, не несущие информации: 'no label' и 'no lesion detected'
    # Учитываем варианты с процентами в начале и разные разделители
    skip_re = re.compile(r'^(?:\s*\d+(?:\.\d+)?%\s*)?(no[_\s-]?label|no[_\s-]?lesion[_\s-]?detected)\s*$', re.IGNORECASE)
    parts = [p.strip() for p in text.split(';') if p.strip()]
    cleaned = []
    for p in parts:
        p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
        if skip_re.match(p_no_percent):
            continue
        cleaned.append(p)

    if not cleaned:
        return 'No lesion detected'

    return '; '.join(cleaned)


def main():
    df = pd.read_csv(INPUT_CSV)
    unique_lobes = get_unique_lobes(df, 'harvard_oxford')
    for col in ['harvard_oxford']:#, 'aal']:
        df[col] = df[col].apply(lambda txt: transform_region(txt, unique_lobes, MODE, INVERSE, convert_lobe) if txt != 'No lesion detected' else txt)

    # Сохраняем результат
    
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    print(f"✅ Saved transformed data to {OUTPUT_CSV}")

    # Дополнительно: собрать все уникальные значения в колонке(ах) и сохранить список
    unique_values = set()
    for col in ['harvard_oxford']:
        for val in df[col].dropna().astype(str):
            for part in [x.strip() for x in val.split(';') if x.strip()]:
                unique_values.add(part)

    unique_list = sorted(unique_values)
    uniq_out = project_path(os.path.join('data', 'preprocessed', f'{file_name}_{MODE}_unique_labels.txt'))
    with open(uniq_out, 'w', encoding='utf-8') as f:
        for u in unique_list:
            f.write(u + '\n')

    print(f"ℹ️ Saved {len(unique_list)} unique labels to {uniq_out}")

    # Также выведем первые 200 уникальных значений в консоль для быстрой проверки
    print('--- Sample unique labels (first 200) ---')
    for i, u in enumerate(unique_list, 1):
        print(f"{i:03d}: {u}")

if __name__ == '__main__':
    main()
