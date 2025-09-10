from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import BASE_PATH
from meld_graph.data_preprocessing import Preprocess as Prep
from torch.utils.data import Dataset


def load_config(config_file: str) -> Any:
    """load config.py file and return config object"""
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class EpilepDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: Any,
        mode: str = "train",
        meld_path: str = "",
        output_dir: str = "",
        feature_path: str = "",
        subject_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        assert subject_ids is not None, "subject_ids must be provided"

        self.mode = mode
        self.meld_path = meld_path
        self.output_dir = output_dir
        self.feature_path = feature_path
        self.subject_ids = subject_ids
        # good_ids = []
        # for sid in subject_ids:
        #     features_dir = Path(self.feature_path) / "preprocessed" / "meld_files" / sid / "features"
        #     dist_npz_path = features_dir / "distance_maps_gt.npz"
        #     if dist_npz_path.is_file():
        #         good_ids.append(sid)
        #     else:
        #         print(f"[WARN] Skip {sid}: missing {dist_npz_path}")

        # self.subject_ids = good_ids

        csv_path = Path(csv_path)
        self.data = pd.read_csv(
            csv_path, sep=",", engine="python", quoting=csv.QUOTE_NONE, escapechar="\\"
        )

        self.tokenizer = tokenizer

        # 2) extract sub-ID
        self.data["sub"] = self.data["DATA_PATH"].apply(
            lambda p: os.path.basename(p).split("_patient")[0].split("_control")[0]
            if isinstance(p, str)
            else None
        )

        # 3) set 'sub' as an index to make it easier to select
        self.data = self.data.set_index("sub")
        self.data = self.data.loc[subject_ids]

        cohort = MeldCohort(
            hdf5_file_root="{site_code}_{group}_featurematrix_combat.hdf5",
            dataset=None,
            data_dir=BASE_PATH,
        )

        self.config = load_config(
            "/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/final_ablation_full_with_combat_my.py"
        )
        params = (
            next(iter(self.config.losses))
            if isinstance(self.config.losses, list)
            else self.config.losses
        )
        self.prep = Prep(cohort=cohort, params=params["data_parameters"])

        # Descriptions may not be generated for some columns
        if "description" in self.data.columns:
            self.data["description"] = self.data["description"].fillna("")
            self.caption_list = list(self.data["description"])
            self.max_length = 384
        else:
            self.data["harvard_oxford"] = self.data["harvard_oxford"].fillna("")
            self.data["aal"] = self.data["aal"].fillna("")
            self.caption_list = list(self.data["harvard_oxford"])
            self.max_length = 256

        self.roi_list: List[Optional[str]] = list(self.data["ROI_PATH"])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Сheck existing features npz
        features_dir = (
            Path(self.feature_path)
            / "preprocessed"
            / "meld_files"
            / self.subject_ids[idx]
            / "features"
        )
        dist_npz_path = features_dir / "distance_maps_gt.npz"

        caption: str = self.caption_list[idx]
        subject_data_list: List[Dict[str, Any]] = self.prep.get_data_preprocessed(
            subject=self.subject_ids[idx],
            features=self.prep.params["features"],
            lobes=self.prep.params["lobes"],
            lesion_bias=False,
            distance_maps=False,
            harmo_code="fcd",  # TODO: Make a hyperparameter
            only_lesion=False,  # TODO: Make a hyperparameter
            only_features=self.roi_list[idx] is None,
            combine_hemis=self.prep.params["combine_hemis"],
        )

        labels_tensors: List[torch.Tensor] = []
        for d in subject_data_list:
            if d.get("labels") is None:
                n_verts = d["features"].shape[0]
                labels_tensors.append(torch.zeros(n_verts, dtype=torch.long))
            else:
                labels_tensors.append(torch.from_numpy(d["labels"]).long())

        roi: torch.Tensor = torch.stack(labels_tensors, dim=0)

        if not dist_npz_path.is_file():
            raise FileNotFoundError(
                f"Failed to generate NPZ for {self.subject_ids[idx]}"
            )
        dist_maps = torch.from_numpy(np.load(dist_npz_path)["arr_0"]).float()

        token_output: Dict[str, torch.Tensor] = self.tokenizer.encode_plus(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        token: torch.Tensor = token_output["input_ids"]
        mask: torch.Tensor = token_output["attention_mask"]

        text: Dict[str, torch.Tensor] = {
            "input_ids": token.squeeze(dim=0),
            "attention_mask": mask.squeeze(dim=0),
        }

        return {
            "subject_id": self.subject_ids[idx],
            "text": text,
            "roi": roi,
            "dist_maps": dist_maps,
        }
