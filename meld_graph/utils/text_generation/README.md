---

# Text Generation Pipeline (Atlas-Based Descriptions)

This folder contains a multi-stage pipeline for generating **raw and derived textual descriptions of epileptogenic lesions** based on anatomical atlases (Harvard–Oxford, AAL).
The pipeline starts from **surface-based lesion representations** and produces multiple CSV files with different types of textual descriptions that can be used for multimodal learning (e.g. vision–language models).

---

## Overview of the Pipeline

The pipeline consists of **three main steps**:

1. **Generate raw atlas-based text descriptions** directly from lesion masks
2. **Attach data paths and raw text columns** to a dataset CSV
3. **Transform raw descriptions into different textual modes** (hemisphere, lobe, dominant lobe, etc.)
4. **(Optionally) Merge multiple text modes into a single “mixed-text” dataset**

Each step is implemented as a separate script and can be run independently once its inputs are prepared.

---

## 1. Raw Text Generation via Atlas Projection

**Script:**
`generate_raw_reports.py`

### Purpose

This script generates **raw textual lesion descriptions** by:

* projecting surface-based lesion masks (`.mgh`) into volumetric space,
* registering them to MNI space,
* intersecting them with anatomical atlases,
* and extracting region-level descriptions using `atlasreader`.

The output is a CSV file with **raw, unprocessed text descriptions** per subject.

---

### Required Input Data Structure

The script expects the following directory layout:

```
data4sharing_root/
├── SUBJECT_ID_1/
│   ├── labels/
│   │   ├── labels-lh.mgh
│   │   ├── labels-rh.mgh
│   │   └── labels-meta.npz
│   └── ...
├── SUBJECT_ID_2/
│   └── ...
```

To generate these files, it is necessary to run the updated MELD script: `../meld_graph/scripts/new_patient_pipeline/run_script_prediction_bonn_data.py`

Where:

* `labels-*.mgh` are surface-based lesion labels (FreeSurfer format)
* `labels-meta.npz` must contain boolean flags:

  * `lh_has_lesion`
  * `rh_has_lesion`

Subjects with `_C_` in their ID are treated as **controls** and automatically assigned:

```
"No lesion detected"
```

---

### External Dependencies

This script requires:

* **FreeSurfer** (for `mri_surf2vol`)
* **ANTsPy**
* **nilearn**
* **atlasreader**
* **nibabel**
* **h5py**

Environment variable `SUBJECTS_DIR` must point to a directory containing:

```
fsaverage_sym/
└── mri/
    └── orig.mgz
```

---

### Output

The script produces:

* a CSV file:

  ```
  all_augmented_reports.csv
  ```

  with columns:

  ```csv
  subject_id, report_harvard_oxford, report_aal
  ```
<!-- * per-subject intermediate files:

  * projected lesion volumes (`*.nii.gz`)
  * atlasreader cluster CSVs
* a log file with processing status -->

---

## 2. Attaching Data Paths and Raw Text Columns

**Script:**
`build_dataset_with_text.py`

### Purpose

This script:

* takes the raw report CSV from Step 1,
* matches each `subject_id` to its corresponding HDF5 feature file,
* and produces a **final dataset CSV** containing:

  * paths to input data,
  * optional ROI paths,
  * raw atlas-based text descriptions.

---

### Input

* CSV from Step 1:

```

all_augmented_reports.csv

```

* HDF5 feature files located in  
(to obtain the data in this format, please refer to the MELD documentation on how to convert your data:  
https://meld-graph.readthedocs.io/en/latest/run_prediction_pipeline.html):

```

meld_combats/
└── SUBJECTID_*_featurematrix*.hdf5

```

### Output

```
MELD_BONN_dataset_augmented_final.csv
```

with columns:

```csv
DATA_PATH, ROI_PATH, harvard_oxford, aal
```

Notes:

* `ROI_PATH` is empty for control subjects
* `DATA_PATH` always points to an HDF5 feature matrix

---

### Variant: Dataset Without Text / ROI

A simplified version of this script can be used to generate a dataset **without textual descriptions**, optionally without ROI information:

**Output:**

```csv
DATA_PATH, ROI_PATH
```

This is useful for:

* vision-only baselines
* ablation studies

---

## 3. Text Transformation and Augmentation Modes

**Script:**
`transform_text_modes.py`

### Purpose

This script converts **raw atlas-based descriptions** into multiple **controlled textual representations**, such as:

* hemisphere-only descriptions,
* lobe-level descriptions,
* dominant lobe,
* descriptions without percentages,
* deliberately incorrect (negative) descriptions.

---

### Input

* Dataset CSV from Step 2:

  ```
  MELD_BONN_dataset_augmented_final.csv
  ```

---

### Available Modes

The script supports the following modes (set via `MODE`):

| Mode name                               | Description                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------- |
| `full`                                  | Raw atlas output (original region-level descriptions)                       |
| `hemisphere`                            | Hemisphere information only (Left / Right Hemisphere)                       |
| `lobe`                                  | Lobe-level description (regions mapped to lobes)                            |
| `hemisphere_lobe`                       | Hemisphere followed by lobe(s)                                              |
| `lobe_regions`                          | Region-level descriptions mapped to lobes                                   |
| `hemisphere_lobe_regions`               | Hemisphere + lobe(s) derived from region-level mapping                      |
| `dominant_lobe`                         | Dominant lobe (first or most prominent region)                              |
| `lobe_highest_percentages`              | Lobe with the highest aggregated lesion percentage                          |
| `no_percentage`                         | Removes all percentage values from descriptions                             |
| `replace_underscores`                   | Replaces underscores with spaces in text                                    |
| `full+hemisphere`                       | Hemisphere-only description combined with raw atlas output                  |
| `wrong_hemisphere`                      | Incorrect hemisphere (Left ↔ Right)                                         |
| `wrong_lobe_hemi`                       | Incorrect lobe(s) and incorrect hemisphere                                  |
| `wrong_lobe_regions`                    | Incorrect region/lobe information only                                      |
| `wrong_lobe_regions_hemi`               | Incorrect region/lobe information with incorrect hemisphere                 |
| `wrong_hemisphere_only_correct_lobe`    | Correct lobe(s) with incorrect hemisphere                                   |
| `wrong_hemisphere_only_correct_regions` | Correct regions with incorrect hemisphere                                   |


**Important notes:**

- To generate **lobe-based descriptions** in any combination, set `CONVERT_LOBE = True`.
  Otherwise (e.g. when working with `lobe_regions`), set it to `False`.

- The `INVERSE` parameter is used **only** for the `full+hemisphere` mode.
  Set `INVERSE = True` if you want to explicitly generate *incorrect hemisphere labels*.

Internally, detailed atlas regions are mapped to generalized lobes using a predefined dictionary (`REGION_TO_LOBE`).

---

### Output

For each mode, the script generates:

```
final_aug_text/
└── MELD_BONN_<MODE>.csv
```

Each file contains:

```csv
DATA_PATH, ROI_PATH, <text_column>
```

Additionally:

* a file listing **all unique labels** for the given mode is saved for inspection.

---

## 4. Mixed-Text Dataset Generation

**Script:**
`build_mixed_text_dataset.py`

### Purpose

This script merges **multiple text representations** into a single dataset, enabling:

* random text selection,
* multi-view language supervision,
* text ablation experiments.

⚠️ This step **requires that Step 3 has already been run** for all selected modes.

---

### Input

Multiple CSV files generated in Step 3, e.g.:

```
final_aug_text/
├── MELD_BONN_hemisphere.csv
├── MELD_BONN_lobe.csv
├── MELD_BONN_hemisphere_lobe.csv
```

---

### Output

```
MELD_BONN_mixed.csv
```

with columns:

```csv
DATA_PATH,
ROI_PATH,
hemisphere_text,
lobe_text,
hemisphere_lobe_text,
no_text
```

Where:

* `no_text` is a fixed baseline string (`"full brain"`)
* missing descriptions are allowed (`NaN`)

---

