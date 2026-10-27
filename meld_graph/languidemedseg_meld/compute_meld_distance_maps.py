"""Run the released MELD Graph model and save its own per-vertex predictions and distance_map
(non_lesion_logits, normalised dist/300) for every subject of a split.

Output: <out_dir>/results_best_model/predictions.hdf5 with {sid}/{lh,rh}/{prediction,distance_map}.
Subjects already in the output file are skipped. Used by test.py --dist_source meld.
"""
import argparse
import os
import sys
import tempfile
import warnings

import h5py
import numpy as np
import pandas as pd

if not hasattr(np, "float"):
    np.float = float

from meld_graph.evaluation import Evaluator
from meld_graph.experiment import Experiment
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import BASE_PATH, EXPERIMENT_PATH, MELD_DATA_PATH, MODEL_PATH
from meld_graph.tools_pipeline import create_dataset_file

warnings.filterwarnings("ignore")

DEFAULT_OUT_DIR = os.path.join(MELD_DATA_PATH, "output", "meld_distance_predictions")
HDF5_FILE_ROOT = "{site_code}_{group}_featurematrix_combat.hdf5"


def predictions_file(out_dir):
    return os.path.join(out_dir, "results_best_model", "predictions.hdf5")


def done_subjects(out_dir):
    path = predictions_file(out_dir)
    if not os.path.isfile(path):
        return set()
    with h5py.File(path, "r") as f:
        return {sid for sid in f.keys() if "rh/distance_map" in f[sid]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split_path", required=True, help="csv with subject_id,split")
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--chunk", type=int, default=20, help="subjects per Evaluator run")
    p.add_argument("--limit", type=int, default=0, help="only first N subjects (0 = all)")
    p.add_argument("--subjects", default=None, help="comma-separated subject ids (overrides split)")
    a = p.parse_args()

    if a.subjects:
        sids = a.subjects.split(",")
    else:
        sids = pd.read_csv(a.split_path).query("split == @a.split")["subject_id"].tolist()
    if a.limit:
        sids = sids[:a.limit]
    todo = [s for s in sids if s not in done_subjects(a.out_dir)]
    print(f"[INFO] {len(sids)} subjects, {len(todo)} to predict -> {predictions_file(a.out_dir)}")

    experiment_path = os.path.join(EXPERIMENT_PATH, MODEL_PATH)
    for start in range(0, len(todo), a.chunk):
        chunk = todo[start:start + a.chunk]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as tmp:
            create_dataset_file(chunk, tmp.name)
            exp = Experiment.from_folder(experiment_path)
            exp.cohort = MeldCohort(hdf5_file_root=HDF5_FILE_ROOT, dataset=tmp.name, data_dir=BASE_PATH)
            exp.data_parameters["hdf5_file_root"] = HDF5_FILE_ROOT
            exp.data_parameters["dataset"] = tmp.name
            exp.data_parameters["augment_data"] = {}
            exp.experiment_path = experiment_path
            eva = Evaluator(
                experiment=exp,
                checkpoint_path=experiment_path,
                cohort=exp.cohort,
                subject_ids=chunk,
                save_dir=a.out_dir,
                aug_mode="test",
                mode="test",
                model_name="best_model",
                threshold="slope_threshold",
                thresh_and_clust=True,
            )
            eva.load_predict_data(store_predictions=False, save_prediction=True, roc_curves_thresholds=None)
        print(f"[INFO] done {start + len(chunk)}/{len(todo)}")


if __name__ == "__main__":
    main()
