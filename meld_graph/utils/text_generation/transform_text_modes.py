from typing import List
import re
import sys
import os
import csv
import random
import pandas as pd
from pathlib import Path

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join( "..", ".."))

sys.path.insert(0, PROJECT_ROOT)


def project_path(relative_path):
    """Return an absolute path relative to the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)


# =========================
# Configuration
# =========================
FILE_NAME = "MELD_BONN_dataset_augmented_final"
MODE = input().strip()   # e.g. "hemisphere", "lobe", "hemisphere_lobe", etc.
CONVERT_LOBE = True
INVERSE = False

INPUT_CSV = project_path(
    os.path.join("data", "preprocessed", f"{FILE_NAME}.csv")
)

OUTPUT_CSV = project_path(
    os.path.join("data", "preprocessed", "final_aug_text", f"MELD_BONN_{MODE}.csv")
)

# Mapping from fine-grained atlas regions to coarse lobes
REGION_TO_LOBE = {
    "Lateral Ventrical": "Lateral Ventricle",

    # Central
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

def extract_hemisphere(text: str) -> str:
    """Extract hemisphere information from text."""
    m = re.search(r"\b(Left|Right)\b", text)
    return f"{m.group(1)} Hemisphere" if m else text

def extract_lobe_percentages(text: str) -> str:
    """
    Preserves percentages, converts each anatomical region to its
    corresponding lobe, and keeps repeated entries
    (e.g., "30% Temporal lobe; 30% Temporal lobe").
    """

    parts = [p.strip() for p in text.split(";") if p.strip()]
    new_parts = []

    for part in parts:
        # Preserve the percentage, if present
        m = re.match(r"^\s*(\d+(?:\.\d+)?)%\s*(.*)$", part)
        if m:
            perc = m.group(1) + "%"
            region = m.group(2).strip()
        else:
            perc = ""
            region = part

        # Remove Left/Right hemisphere markers
        region_clean = re.sub(r"\b(Left|Right)\s+", "", region)

        # Look up the region in the REGION_TO_LOBE mapping
        found_lobe = None
        for key, lobe in REGION_TO_LOBE.items():
            if key.lower() in region_clean.lower():
                found_lobe = lobe
                break

        # If no mapping is found, keep the original region name
        lobe_name = found_lobe if found_lobe else region_clean

        if lobe_name == "no label":
            continue

        if perc:
            new_parts.append(f"{perc} {lobe_name}")
        else:
            new_parts.append(lobe_name)

    return "; ".join(new_parts)

def replace_region_with_lobe(text: str) -> str:
    """
    Replace fine-grained anatomical regions with their corresponding lobes.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    parts = [p.strip() for p in text.split(';') if p.strip()]
    mapped = []

    for part in parts:
        clean = part.replace("_", " ")
        clean = re.sub(r"\b\d+(?:[.,]\d+)?\s*%\s*", "", clean)

        if not re.search(r"\bHemisphere\b", clean, re.IGNORECASE):
            clean = re.sub(r"\b(Left|Right)\s+", "", clean)

        found = None
        for key, lobe in REGION_TO_LOBE.items():
            if key.lower() in clean.lower():
                found = lobe
                break

        mapped.append(found if found else clean.strip())

    # Preserve order, remove duplicates
    seen = set()
    result = []
    for x in mapped:
        if x not in seen:
            seen.add(x)
            result.append(x)

    return "; ".join(result)

def extract_dominant(text: str) -> str:
    """Extract the dominant (first or highest-percentage) region/lobe."""

    if ";" not in text:
        return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", text)
    first = text.split(";", 1)[0]
    return re.sub(r"^[0-9]+(?:\.[0-9]+)?%\s*", "", first.strip())

def extracts_wrong_lobe_hemi(text: str, unique_lobes: List[str], mode: str = "wrong_lobe") -> str:
    """
    Generates an intentionally incorrect lesion description by modifying
    lobe and/or hemisphere information.

    Depending on the mode:
    - 'wrong_lobe_hemi': replaces lesion regions with incorrect lobes
      and assigns an incorrect hemisphere.
    - 'wrong_hemisphere_only_correct_lobe': keeps the correct lobe(s)
      but assigns an incorrect hemisphere.
    - otherwise: replaces regions with incorrect lobes only.
    """

    parts = [x.strip() for x in text.split(';') if x.strip()]

    # Determine the original hemisphere(s)
    hemis = set()
    for part in parts:
        m = re.search(r"\b(Left|Right)\b", part)
        if m:
            hemis.add(m.group(1))

    wrong_hemi = (
        "Left" if "Right" in hemis
        else "Right" if hemis
        else random.choice(["Left", "Right"])
    )

    # Remove percentages and hemisphere markers
    clean_parts = []
    for p in parts:
        p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
        p_clean = re.sub(r'\b(Left|Right)\s+', '', p_no_percent)
        clean_parts.append(p_clean)

    # Randomly replace regions with incorrect ones
    new_parts = []
    for region in clean_parts:
        choices = [
            x for x in unique_lobes
            if re.sub(r'\b(Left|Right)\s+', '', x) != region
        ]
        wrong_region = random.choice(choices) if choices else region
        new_parts.append(wrong_region)

    # Convert regions to lobes
    new_text = "; ".join(new_parts)
    new_text = replace_region_with_lobe(new_text)

    # Prepend incorrect hemisphere if required
    if mode == "wrong_lobe_hemi":
        return f"{wrong_hemi} Hemisphere; {new_text}"

    elif mode == "wrong_hemisphere_only_correct_lobe":
        clean_text = "; ".join(clean_parts)
        clean_text = replace_region_with_lobe(clean_text)
        return f"{wrong_hemi} Hemisphere; {clean_text}"

    else:
        return new_text

def extracts_wrong_lobe_reg_hemi(text: str, unique_lobes: List[str], mode: str = "wrong_lobe"):
    """
    Generates an intentionally incorrect lesion description by modifying
    lobe and/or hemisphere information at the region level.

    Supported modes:
    - 'wrong_lobe_regions':
        replaces lesion regions with incorrect ones and converts them to lobes.
    - 'wrong_lobe_regions_hemi':
        replaces lesion regions with incorrect ones and prepends an incorrect hemisphere.
    - 'wrong_hemisphere_only_correct_regions':
        keeps the correct regions but assigns an incorrect hemisphere.
    """

    parts = [x.strip() for x in text.split(';') if x.strip()]

    # Determine all hemispheres present in the original text
    hemis = set()
    for part in parts:
        m = re.search(r"\b(Left|Right)\b", part)
        if m:
            hemis.add(m.group(1))

    # Select an incorrect hemisphere
    if hemis:
        wrong_hemi = "Left" if "Right" in hemis else "Right"
    else:
        wrong_hemi = random.choice(["Left", "Right"])

    if mode == "wrong_hemisphere_only_correct_regions":
        # Change only the hemisphere, keep regions unchanged
        clean_parts = []
        for p in parts:
            # Remove percentages at the beginning of the string
            p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
            # Remove Left/Right hemisphere markers
            p_clean = re.sub(r'\b(Left|Right)\s+', '', p_no_percent)
            clean_parts.append(p_clean)

        return (
            f"{wrong_hemi} Hemisphere; {'; '.join(clean_parts)}"
            if clean_parts else text
        )

    # Remove percentages and hemisphere markers from regions
    clean_parts = []
    for p in parts:
        p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
        p_clean = re.sub(r'\b(Left|Right)\s+', '', p_no_percent)
        clean_parts.append(p_clean)

    # Randomly replace regions with incorrect ones
    new_parts = []
    for lobe in clean_parts:
        choices = []
        for x in unique_lobes:
            x_clean = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', x)
            x_clean = re.sub(r'\b(Left|Right)\s+', '', x_clean)
            if x_clean != lobe:
                choices.append(x_clean)

        wrong_lobe = random.choice(choices) if choices else lobe
        new_parts.append(wrong_lobe)

    # Prepend incorrect hemisphere if required
    if mode in ("wrong_lobe_regions", "wrong_lobe_regions_hemi"):
        return (
            f"{wrong_hemi} Hemisphere; {'; '.join(new_parts)}"
            if new_parts else text
        )

    return "; ".join(new_parts)

def pick_dominant_lobe_from_percentages(text: str) -> str:
    """
    Takes a string of the form:
        "30% Temporal lobe; 40% Temporal lobe; 30% Parietal lobe"
    Aggregates percentages by lobe and returns the DOMINANT lobe:
        "Temporal lobe"
    """

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

    # Select the lobe with the highest aggregated percentage
    dominant = max(lobe_totals, key=lobe_totals.get)
    return dominant

def extract_hemisphere_lobes(text: str, mode: str) -> str:
    """
    Extracts hemisphere and/or lobe information from a lesion description.

    Depending on the mode:
    - 'lobe' or 'lobe_regions': returns lobe information only.
    - 'hemisphere_lobe': returns hemisphere followed by lobe(s), if available.
    """
    hemi_name = extract_hemisphere(text)

    # Remove percentage values
    text = re.sub(r"\b\d+(?:[.,]\d+)?\s*%\s*", "", text)

    parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    clean_parts = []
    hemis = set()

    for part in parts:
        # Detect hemisphere markers
        m = re.match(r"^(Left|Right)\b", part)
        if m:
            hemis.add(m.group(1))

        # Remove hemisphere prefix from region/lobe name
        clean = re.sub(r"^(Left|Right)\s+", "", part)
        if clean not in clean_parts:
            clean_parts.append(clean)

    # Determine hemisphere prefix
    hemi_prefix = ""
    if len(hemis) == 1:
        hemi_prefix = f"{list(hemis)[0]} Hemisphere"
    elif hemi_name:
        hemi_prefix = hemi_name

    # Final ordering: hemisphere first, then lobe(s)
    if mode in ("lobe", "lobe_regions"):
        return "; ".join(clean_parts)

    else:  # hemisphere_lobe
        if hemi_prefix and all(hemi_prefix not in s for s in clean_parts):
            return f"{hemi_prefix}; {'; '.join(clean_parts)}"
        else:
            return "; ".join(clean_parts)

# =========================
# Main transformation logic
# =========================
def transform_region(
    text: str,
    unique_lobes: set,
    mode: str,
    inverse: bool,
    convert_lobe: bool = True
) -> str:
    """
    Transforms raw atlas-based lesion descriptions into different textual
    representations depending on the selected mode.

    Supported modes include:
    - hemisphere
    - lobe / lobe_regions
    - hemisphere_lobe / hemisphere_lobe_regions
    - dominant_lobe
    - no_percentage
    - wrong_* (negative text augmentation modes)
    """

    text = text.replace("_", " ")

    # --- Basic transformations ---
    if mode == "hemisphere":
        text = extract_hemisphere(text)

    if mode in ("hemisphere_lobe", "lobe", "hemisphere_lobe_regions", "lobe_regions"):
        text = extract_hemisphere_lobes(text, mode)

    if mode == "full+hemisphere":
        hemi = extract_hemisphere(text)
        if inverse:
            if hemi == "Left Hemisphere":
                text = "No Right Hemisphere"
            elif hemi == "Right Hemisphere":
                text = "No Left Hemisphere"
        else:
            text = hemi

    if mode == "no_percentage":
        text = re.sub(r'\b\d+(?:\.\d+)?%\s*', '', text)

    if mode == "dominant_lobe":
        text = extract_dominant(text)

    if mode == "replace_underscores":
        text = text.replace("_", " ")

    if mode == "wrong_hemisphere":
        hemi = extract_hemisphere(text)
        swap = {
            "Left Hemisphere": "Right Hemisphere",
            "Right Hemisphere": "Left Hemisphere",
        }
        text = swap.get(hemi, text)

    # --- Negative text augmentation modes ---
    if mode in ("wrong_lobe_hemi", "wrong_hemisphere_only_correct_lobe"):
        text = extracts_wrong_lobe_hemi(text, unique_lobes, mode=mode)

    if mode == "wrong_lobe_regions":
        text = extracts_wrong_lobe_reg_hemi(text, unique_lobes, mode="wrong_lobe")

    if mode == "wrong_lobe_regions_hemi":
        text = extracts_wrong_lobe_reg_hemi(
            text, unique_lobes, mode="wrong_lobe_regions_hemi"
        )

    if mode == "wrong_hemisphere_only_correct_regions":
        text = extracts_wrong_lobe_reg_hemi(
            text, unique_lobes, mode="wrong_hemisphere_only_correct_regions"
        )

    # --- Lobe with highest aggregated percentage ---
    if mode == "lobe_highest_percentages":
        text = extract_lobe_percentages(text)
        return pick_dominant_lobe_from_percentages(text)

    # --- Final cleanup and lobe conversion ---
    if convert_lobe:
        text = replace_region_with_lobe(text)

    # Remove non-informative labels
    skip_re = re.compile(
        r'^(?:\s*\d+(?:\.\d+)?%\s*)?(no[_\s-]?label|no[_\s-]?lesion[_\s-]?detected)\s*$',
        re.IGNORECASE
    )

    parts = [p.strip() for p in text.split(';') if p.strip()]
    cleaned = []
    for p in parts:
        p_no_percent = re.sub(r'^\s*\d+(?:\.\d+)?%\s*', '', p)
        if skip_re.match(p_no_percent):
            continue
        cleaned.append(p)

    if not cleaned:
        return "No lesion detected"

    return "; ".join(cleaned)


def main():
    df = pd.read_csv(INPUT_CSV)
    unique_lobes = set(df["harvard_oxford"].dropna())

    df["harvard_oxford"] = df["harvard_oxford"].apply(
        lambda t: transform_region(t, unique_lobes, MODE, INVERSE, CONVERT_LOBE)
        if t != "No lesion detected" else t
    )

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_NONE, escapechar="\\")

    print(f"✅ Saved transformed data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
