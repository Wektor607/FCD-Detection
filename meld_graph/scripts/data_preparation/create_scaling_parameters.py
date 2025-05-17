# Script to create scaling parameters for the raw features

from meld_graph.paths import BASE_PATH
from meld_graph.meld_cohort import MeldCohort, MeldSubject
from meld_graph.data_preprocessing import Preprocess
import os
import numpy as np
import glob
# define cohort to compute scaling parameters from
site_codes = sorted(set(
    os.path.basename(p).split('_')[0]
    for p in glob.glob(os.path.join(BASE_PATH, "*.hdf5"))
))

cohort = MeldCohort(
    hdf5_file_root="{site_code}_featurematrix.hdf5",
    dataset=None,
    data_dir="data/output/preprocessed_surf_data/"
)

# define features to compute scaling parameters
features = [
    ".on_lh.curv.mgh",
    ".on_lh.gm_FLAIR_0.25.mgh",
    ".on_lh.gm_FLAIR_0.5.mgh",
    ".on_lh.gm_FLAIR_0.75.mgh",
    ".on_lh.gm_FLAIR_0.mgh",
    ".on_lh.pial.K_filtered.sm20.mgh",
    ".on_lh.sulc.mgh",
    ".on_lh.thickness.mgh",
    ".on_lh.w-g.pct.mgh",
    ".on_lh.wm_FLAIR_0.5.mgh",
    ".on_lh.wm_FLAIR_1.mgh",
]

# define scaling parameters file name
scaling_params_file = "scaling_params_GDL.json"

# create object preprocessing
scale = Preprocess(
    cohort,
    site_codes=site_codes,
    write_output_file=scaling_params_file,
    data_dir=BASE_PATH,
)

# compute scaling parameters
for feature in features:
    scale.compute_scaling_parameters(feature)
