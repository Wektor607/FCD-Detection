network_parameters = {
    "model_parameters": {
        "activation_fn": "leaky_relu",
        "conv_type": "SpiralConv",
        "dim": 2,
        "distance_head": False,
        "kernel_size": 3,
        "layer_sizes": [
            [32, 32, 32],
            [32, 32, 32],
            [64, 64, 64],
            [64, 64, 64],
            [128, 128, 128],
            [128, 128, 128],
            [256, 256, 256],
        ],
        "norm": None,
        "spiral_len": 7,
    },
    "name": "24-08-01_MRIN_dcp/s_0",
    "network_type": "MoNetUnet",
    "training_parameters": {
        "batch_size": 8,
        "deep_supervision": {
            "levels": [6, 5, 4, 3, 2, 1],
            "weight": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.0150765],
        },
        "init_weights": None,
        "loss_dictionary": {
            "cross_entropy": {"weight": 1},
            "dice": {"class_weights": [0.0, 1.0], "weight": 1},
            "distance_regression": {"loss": "mae", "weigh_by_gt": True, "weight": 1},
            "lesion_classification": {"apply_to_bottleneck": True, "weight": 1},
        },
        "lr_decay": 0.9,
        "max_epochs_lr_decay": 1000,
        "max_patience": 1000,
        "metric_smoothing": False,
        "metrics": [
            "dice_lesion",
            "dice_nonlesion",
            "precision",
            "recall",
            "tp",
            "fp",
            "fn",
            "auroc",
            "cl_precision",
            "cl_recall",
            "sub_auroc",
        ],
        "num_epochs": 1000,
        "optimiser": "sgd",
        "optimiser_parameters": {"lr": 0.0001, "momentum": 0.99, "nesterov": True},
        "oversampling": True,
        "shuffle_each_epoch": True,
        "start_epoch": 0,
        "stopping_metric": {"name": "loss", "sign": 1},
    },
}
data_parameters = {
    "augment_data": {
        "augment_lesion": {"p": 0.0},
        "blur": {"p": 0.2},
        "brightness": {"p": 0.15},
        "contrast": {"p": 0.15},
        "extend_lesion": {"p": 0.0},
        "flipping": {"file": "data/flipping/flipping_ico7_3.npy", "p": 0.5},
        "gamma": {"p": 0.15},
        "low_res": {"p": 0.25},
        "noise": {"p": 0.15},
        "spinning": {"file": "data/spinning/spinning_ico7_10.npy", "p": 0.2},
        "warping": {"file": "data/warping/warping_ico7_10.npy", "p": 0.2},
    },
    "combine_hemis": None,
    "dataset": "MELD_dataset_V6_mrineg&histo.csv",
    "distance_mask_medial_wall": True,
    "features": [
        ".combat.on_lh.pial.K_filtered.sm20.mgh",
        ".combat.on_lh.thickness.sm3.mgh",
        ".combat.on_lh.thickness_regression.sm3.mgh",
        ".combat.on_lh.w-g.pct.sm3.mgh",
        ".combat.on_lh.sulc.sm3.mgh",
        ".combat.on_lh.curv.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
        ".combat.on_lh.gm_FLAIR_0.sm3.mgh",
        ".combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
        ".combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.sulc.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.curv.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
        ".inter_z.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.pial.K_filtered.sm20.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.thickness_regression.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.w-g.pct.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.sulc.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.curv.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.75.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.5.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.25.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.gm_FLAIR_0.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_0.5.sm3.mgh",
        ".inter_z.asym.intra_z.combat.on_lh.wm_FLAIR_1.sm3.mgh",
    ],
    "features_to_exclude": [],
    "features_to_replace_with_0": [],
    "fold_n": 0,
    "group": "both",
    "hdf5_file_root": "{site_code}_{group}_featurematrix_combat_6_kernels_noCombat.hdf5",
    "icosphere_parameters": {"distance_type": "exact"},
    "lesion_bias": 0,
    "lobes": False,
    "number_of_folds": 5,
    "object_detection": False,
    "preprocessing_parameters": {
        "scaling": None,
        "zscore": "../data/feature_means_no_combat_mrineghisto.json",
    },
    "scanners": ["15T", "3T"],
    "site_codes": [
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H9",
        "H10",
        "H11",
        "H12",
        "H14",
        "H15",
        "H16",
        "H17",
        "H18",
        "H19",
        "H21",
        "H23",
        "H24",
        "H26",
    ],
    "smooth_labels": False,
    "subject_features_to_exclude": [],
    "subsample_cohort_fraction": False,
    "synth_on_the_fly": False,
    "synthetic_data": {
        "bias": 1,
        "jitter_factor": 2,
        "n_subs": 1000,
        "n_subtypes": 25,
        "proportion_features_abnormal": 0.2,
        "proportion_hemispheres_lesional": 0.9,
        "radius": 2,
        "run_synthetic": False,
        "smooth_lesion": False,
        "use_controls": True,
    },
}
