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
import tempfile

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
        feature_path: str = "",
        subject_ids: Optional[List[str]] = None,
        cohort: MeldCohort = None
    ) -> None:
        super().__init__()

        assert subject_ids is not None, "subject_ids must be provided"

        self.feature_path = feature_path
        self.subject_ids = subject_ids

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
        self.data = self.data.set_index("sub").loc[subject_ids]

        self.config = load_config(
            "/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/final_ablation_full_with_combat_my.py"
        )
        params = (
            next(iter(self.config.losses))
            if isinstance(self.config.losses, list)
            else self.config.losses
        )
        self.prep = Prep(cohort=cohort, params=params["data_parameters"])

        if self.tokenizer is not None and "harvard_oxford" in self.data.columns:
            self.data["harvard_oxford"] = self.data["harvard_oxford"].fillna("")
            self.caption_list = list(self.data["harvard_oxford"])
        else:
            self.caption_list = [None] * len(self.subject_ids)

        self.max_length = 256
        self.roi_list: List[Optional[str]] = list(self.data["ROI_PATH"])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Сheck existing features npz
        features_dir = Path(self.feature_path) / "preprocessed" / "meld_files" / self.subject_ids[idx] / "features"
        dist_npz_path = features_dir / "distance_maps_gt.npz"

        # caption: str = self.caption_list[idx]
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

        if self.tokenizer is not None and self.caption_list[idx] is not None:
            token_output = self.tokenizer.encode_plus(
                self.caption_list[idx],
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text = {
                "input_ids": token_output["input_ids"].squeeze(0),
                "attention_mask": token_output["attention_mask"].squeeze(0),
            }
        else:
            text = {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            }

        return {
            "subject_id": self.subject_ids[idx],
            "text": text,
            "roi": roi,
            "dist_maps": dist_maps,
        }
    
class SingleEpilepSample(Dataset):
    def __init__(self, data: dict, description: str, tokenizer, cohort):
        super().__init__()
        self.keys = list(data.keys())
        self.description = description
        self.tokenizer = tokenizer
        self.cohort = cohort
        self.max_length = 256
        # Pre-tokenize description once to avoid repeated work in __getitem__
        if self.description and self.tokenizer is not None:
            token_output = self.tokenizer.encode_plus(
                self.description,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            # store token tensors to reuse
            self._text_input_ids = token_output["input_ids"].squeeze(0)
            self._text_attention_mask = token_output["attention_mask"].squeeze(0)
        else:
            self._text_input_ids = None
            self._text_attention_mask = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]
        
        # Текст — токенизация description
        if self._text_input_ids is not None:
            text = {"input_ids": self._text_input_ids, "attention_mask": self._text_attention_mask}
        else:
            text = {
                "input_ids": torch.zeros(self.max_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            }

        return {
            "subject_id": key,
            "text": text,
        }
