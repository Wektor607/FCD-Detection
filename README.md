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

This erratum concerns our MICCAI 2026 paper
[*Text Guidance and Distance Gating Improve Detection and Segmentation of Focal Cortical Dysplasia*](https://papers.miccai.org/miccai-2026/paper/5707_paper.pdf).

The **MELD rows with distance gating (mean / med / min) in Table 4** are incorrect. When evaluating
the MELD baseline, `languidemedseg_meld/test.py --meld_check` gated MELD's clusters with the **ground-truth**
distance-to-lesion maps instead of predicted distances: if no model was passed, it silently fell
back to the ground truth. With ground-truth distances the gate keeps every cluster that touches the
lesion and removes every cluster in the controls, which is why MELD + gating showed 100%
specificity without any loss of sensitivity.

The script is fixed: the distance source must now be given explicitly (`--dist_source meld` for
MELD Graph's own distance head, `--dist_source vlm` for our model), and the script raises an error
if no predicted distance is available. All other numbers in the paper are unaffected:
Tables 2 and 3 report MELD without gating, the Vision+LM models were always gated with their own
predicted distances, and the gating thresholds were selected on the validation split using
predicted distances.

**Corrected Table 4** (main cohort, 259 FCD patients and 193 controls; 95% CI in parentheses).
MELD is now gated with the distance predicted by its own distance head, using the same thresholds
(0.48 / 0.48 / 0.16):

| Model | Text | Gate | Cl. PPV ↑ | IoU ↑ | Spec. ↑ | Sens. ↑ | FP/FCD ↓ |
|---|---|---|---|---|---|---|---|
| MELD | – | – | 0.515 | 0.130 (0.057, 0.190) | 58.0% | 65.6% | 0.28 |
| MELD | – | mean (as published) | ~~0.900~~ | ~~0.131 (0.057, 0.195)~~ | ~~100.0%~~ | ~~65.6%~~ | ~~0.07~~ |
| MELD | – | mean (**corrected**) | 0.721 | 0.064 (0.000, 0.156) | 86.5% | 57.5% | 0.12 |
| MELD | – | med (as published) | ~~0.900~~ | ~~0.131 (0.057, 0.195)~~ | ~~100.0%~~ | ~~65.6%~~ | ~~0.07~~ |
| MELD | – | med (**corrected**) | 0.722 | 0.064 (0.000, 0.156) | 86.5% | 57.9% | 0.12 |
| MELD | – | min (as published) | ~~0.945~~ | ~~0.131 (0.057, 0.195)~~ | ~~100.0%~~ | ~~65.6%~~ | ~~0.04~~ |
| MELD | – | min (**corrected**) | 0.795 | 0.004 (0.000, 0.085) | 93.3% | 52.1% | 0.08 |
| Vision+LM | hemi+lobe | – | 0.343 | 0.140 (0.079, 0.206) | 27.5% | 69.5% | 0.81 |
| Vision+LM | hemi+lobe | mean | 0.517 | 0.135 (0.078, 0.204) | 95.3% | 68.7% | 0.71 |
| Vision+LM | hemi+lobe | med | 0.494 | 0.140 (0.079, 0.206) | 93.3% | 69.5% | 0.76 |
| Vision+LM | hemi+lobe | min | 0.533 | 0.140 (0.079, 0.206) | 93.3% | 69.1% | 0.65 |

**Corrected statement in Section 4.** The paper states: *"As Table 4 shows, MELD with gating can
reach even higher specificity than Vision+LM (100.0% vs. 93.3% on the main cohort), but with lower
sensitivity."* With the corrected baseline, distance gating still improves MELD's specificity
(58.0% → 86.5% with median gating), but it costs MELD sensitivity (65.6% → 57.9%) and IoU
(0.130 → 0.064). With median gating, Vision+LM outperforms gated MELD in specificity (93.3% vs.
86.5%), sensitivity (69.5% vs. 57.9%) and IoU (0.140 vs. 0.064); gated MELD retains a higher
cluster PPV (0.722 vs. 0.494) because it removes many clusters, including true lesions.

We thank the MELD Project team for identifying this discrepancy while reproducing our results.

## Supplementary results (not in the paper)

Median distance gating (threshold 0.48), with MELD gated by its own predicted distance. Sensitivity:
patients with at least one cluster within 20 mm of the lesion; specificity: controls without any
cluster; IoU: median over patients; Cl. PPV: TP / (TP + FP) clusters over all subjects. Rows marked
with * are not reported in the paper; all other rows are identical to Tables 2 and 3.

*Main cohort (259 FCD patients, 193 controls)*

| Model | Text | Gate | Cl. PPV ↑ | IoU ↑ | Spec. ↑ | Sens. ↑ |
|---|---|---|---|---|---|---|
| MELD | – | – | 0.515 | 0.130 | 58.0% | 65.6% |
| MELD* | – | med | 0.722 | 0.064 | 86.5% | 57.9% |
| Vision+LM* | no text | – | 0.361 | 0.100 | 27.5% | 67.6% |
| Vision+LM* | no text | med | 0.712 | 0.013 | 93.3% | 53.3% |
| Vision+LM | hemi | – | 0.335 | 0.135 | 27.5% | 69.1% |
| Vision+LM | hemi | med | 0.487 | 0.135 | 93.3% | 69.1% |
| Vision+LM | lobe | – | 0.347 | 0.130 | 27.5% | 69.1% |
| Vision+LM | lobe | med | 0.497 | 0.130 | 93.3% | 69.1% |
| Vision+LM | hemi+lobe | – | 0.343 | 0.140 | 27.5% | 69.5% |
| Vision+LM | hemi+lobe | med | 0.494 | 0.140 | 93.3% | 69.5% |

*Independent Bonn cohort (82 FCD patients, 83 controls; never used in training)*

| Model | Text | Gate | Cl. PPV ↑ | IoU ↑ | Spec. ↑ | Sens. ↑ |
|---|---|---|---|---|---|---|
| MELD | – | – | 0.513 | 0.218 | 59.0% | 69.5% |
| MELD* | – | med | 0.833 | 0.188 | 91.6% | 59.8% |
| Vision+LM* | no text | – | 0.420 | 0.191 | 37.3% | 65.9% |
| Vision+LM* | no text | med | 0.827 | 0.000 | 94.0% | 46.3% |
| Vision+LM | hemi | – | 0.275 | 0.306 | 37.3% | 78.0% |
| Vision+LM | hemi | med | 0.504 | 0.318 | 94.0% | 73.2% |
| Vision+LM | lobe | – | 0.341 | 0.251 | 37.3% | 72.0% |
| Vision+LM | lobe | med | 0.547 | 0.251 | 94.0% | 69.5% |
| Vision+LM | hemi+lobe | – | 0.337 | 0.292 | 37.3% | 74.4% |
| Vision+LM | hemi+lobe | med | 0.545 | 0.292 | 94.0% | 70.7% |
