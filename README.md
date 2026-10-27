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

Corrected results on the main test cohort (259 FCD patients, 193 controls). MELD is gated with
the distance predicted by its own distance head (median aggregation, threshold 0.48):

| Model | Gate | Sensitivity | Specificity | IoU | Cluster PPV |
|---|---|---|---|---|---|
| MELD | – | 65.6% | 58.0% | 0.130 | 0.515 |
| MELD | median (as reported) | 65.6% | 100.0% | 0.131 | 0.900 |
| MELD | median (**corrected**) | **57.9%** | **86.5%** | **0.064** | **0.722** |

We thank the MELD Project team for identifying this discrepancy while reproducing our results.
