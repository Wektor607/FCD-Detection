# FCD-Detection

Development repository for **text-guided FCD detection** on top of
[MELD Graph](https://github.com/MELDProject/meld_graph): experiments, training and
evaluation code for our MICCAI 2026 paper. The cleaned public release of the code is
[FCD-TextGuide](https://github.com/Wektor607/FCD-TextGuide).

## Repository structure

```
FCD-Detection/
└── meld_graph/                  # MELD Graph pipeline + our model, Docker setup
    ├── languidemedseg_meld/     # Text-guided model: training, testing, inference
    ├── meld_graph/              # MELD Graph library (models, evaluation, data loading)
    ├── scripts/                 # MELD preprocessing and prediction pipelines
    ├── backend/, frontend/      # Web interface
    └── README.md                # Docker setup, training and testing guide
```

Setup, training and testing instructions: [meld_graph/README.md](meld_graph/README.md).

## Erratum (October 2026): MELD + distance-gating baseline

The **MELD + gating** row in the top part of Table 4 of our MICCAI 2026 paper is incorrect.
When evaluating the MELD baseline, `languidemedseg_meld/test.py --meld_check` gated MELD's
clusters with the **ground-truth** distance-to-lesion maps instead of predicted distances:
if no model was passed, it silently fell back to the ground truth. With ground-truth distances
the gate keeps every cluster that touches the lesion and removes every cluster in the controls,
which is why MELD + gating showed 100% specificity without any loss of sensitivity.

The script is fixed: `--dist_source meld` (MELD Graph's own distance head, computed with
`languidemedseg_meld/compute_meld_distance_maps.py`) or `--dist_source vlm` (our model) must
now be given explicitly, and the script raises an error if no predicted distance is available.
All other results in the paper are unaffected: the MELD + LM models were always gated with
their own predicted distances, and the gating thresholds were selected on the validation set
using predicted distances.

Corrected results, with MELD gated by the distance predicted by its own distance head
(median aggregation, threshold 0.48):

*Main test cohort (259 FCD patients, 193 controls)*

| Model | Text | Gate | Sensitivity | Specificity | IoU | Cluster PPV |
|---|---|---|---|---|---|---|
| MELD | – | – | 65.6% | 58.0% | 0.130 | 0.515 |
| MELD | – | median (as reported) | 65.6% | 100.0% | 0.131 | 0.900 |
| MELD | – | median (**corrected**) | **57.9%** | **86.5%** | **0.064** | **0.722** |
| MELD + LM | hemi+lobe | median | 69.5% | 93.3% | 0.140 | 0.494 |

*Independent Bonn cohort (82 FCD patients, 83 controls; gated results not reported in the paper)*

| Model | Text | Gate | Sensitivity | Specificity | IoU | Cluster PPV |
|---|---|---|---|---|---|---|
| MELD | – | – | 69.5% | 59.0% | 0.218 | 0.513 |
| MELD | – | median (previous evaluation) | 69.5% | 100.0% | 0.222 | 0.906 |
| MELD | – | median (**corrected**) | **59.8%** | **91.6%** | **0.188** | **0.833** |
| MELD + LM | hemi | median | 73.2% | 94.0% | 0.318 | 0.504 |
| MELD + LM | hemi+lobe | median | 70.7% | 94.0% | 0.292 | 0.545 |

Distance gating still improves MELD's specificity, but at the cost of sensitivity. The MELD + LM
results are unchanged. With gating, MELD + LM outperforms the corrected MELD + gating baseline in
sensitivity, specificity and IoU; gated MELD retains a higher cluster PPV, as it removes many
clusters including true lesions.

We thank the MELD Project team for identifying this discrepancy while reproducing our results.

## Results

Full comparison with the corrected MELD + gating baseline. Sensitivity: patients with at least
one cluster within 20 mm of the lesion; specificity: controls without any cluster; IoU: median
over patients; cluster PPV: TP / (TP + FP) clusters over all subjects. Gating uses the median
predicted distance per cluster (threshold 0.48).

*Main test cohort (259 FCD patients, 193 controls)*

| Model | Text | Gate | Sensitivity | Specificity | IoU | Cluster PPV |
|---|---|---|---|---|---|---|
| MELD | – | – | 65.6% | 58.0% | 0.130 | 0.515 |
| MELD | – | median (corrected) | 57.9% | 86.5% | 0.064 | 0.722 |
| MELD + LM | no text | – | 67.6% | 27.5% | 0.100 | 0.361 |
| MELD + LM | no text | median | 53.3% | 93.3% | 0.013 | 0.712 |
| MELD + LM | hemi | – | 69.1% | 27.5% | 0.135 | 0.335 |
| MELD + LM | hemi | median | 69.1% | 93.3% | 0.135 | 0.487 |
| MELD + LM | lobe | – | 69.1% | 27.5% | 0.130 | 0.347 |
| MELD + LM | lobe | median | 69.1% | 93.3% | 0.130 | 0.497 |
| MELD + LM | hemi+lobe | – | 69.5% | 27.5% | 0.140 | 0.343 |
| MELD + LM | hemi+lobe | median | 69.5% | 93.3% | 0.140 | 0.494 |

*Independent Bonn cohort (82 FCD patients, 83 controls; never used in training)*

| Model | Text | Gate | Sensitivity | Specificity | IoU | Cluster PPV |
|---|---|---|---|---|---|---|
| MELD | – | – | 69.5% | 59.0% | 0.218 | 0.513 |
| MELD | – | median (corrected) | 59.8% | 91.6% | 0.188 | 0.833 |
| MELD + LM | no text | – | 65.9% | 37.3% | 0.191 | 0.420 |
| MELD + LM | no text | median | 46.3% | 94.0% | 0.000 | 0.827 |
| MELD + LM | hemi | – | 78.0% | 37.3% | 0.306 | 0.275 |
| MELD + LM | hemi | median | 73.2% | 94.0% | 0.318 | 0.504 |
| MELD + LM | lobe | – | 72.0% | 37.3% | 0.251 | 0.341 |
| MELD + LM | lobe | median | 69.5% | 94.0% | 0.251 | 0.547 |
| MELD + LM | hemi+lobe | – | 74.4% | 37.3% | 0.292 | 0.337 |
| MELD + LM | hemi+lobe | median | 70.7% | 94.0% | 0.292 | 0.545 |
