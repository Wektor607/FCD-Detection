from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import json
import random
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import BASE_PATH
from utils.utils import generate_random_text
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
        cohort: MeldCohort = None,
        max_length: int = 256,
        text_emb: bool = False,
        model_name: str = "",
        text_prob_json: str = None,
    ) -> None:
        super().__init__()

        assert subject_ids is not None, "subject_ids must be provided"

        self.feature_path = feature_path
        self.subject_ids = subject_ids
        self.text_emb = text_emb

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
        self.max_length = max_length

        # ---- Upload json ----
        self.text_probs = None
        if text_prob_json is not None:
            with open(text_prob_json, "r", encoding="utf-8") as f:
                self.text_probs = json.load(f)

        # ---- Text handling ----
        self.text_cols: List[str] = []
        self._single_input_ids = None
        self._single_attention = None
        self._multi_input_ids = None  # shape [num_cols, N, L]
        self._multi_attention = None  # shape [num_cols, N, L]

        if self.tokenizer is not None:
            # Detect available text columns
            priority_single = "harvard_oxford"
            optional_multi = [
                # "full_text",
                "hemisphere_text",
                "lobe_text",
                "dominant_lobe_text",
                "hemisphere_lobe_text",
                "no_text",
            ]

            if priority_single in self.data.columns and not any(
                c in self.data.columns for c in optional_multi
            ):
                # Only one text column scenario
                self.text_cols = [priority_single]
                captions = self.data[priority_single].fillna("").astype(str).tolist()
                # Pre-tokenize single column
                ids_list = []
                att_list = []
                for cap in captions:
                    token_output = self.tokenizer.encode_plus(
                        cap,
                        padding="max_length",
                        max_length=self.max_length,
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors="pt",
                    )
                    ids_list.append(token_output["input_ids"].squeeze(0))
                    att_list.append(token_output["attention_mask"].squeeze(0))
                self._single_input_ids = torch.stack(ids_list, dim=0)
                self._single_attention = torch.stack(att_list, dim=0)

            else:
                # Multi-column case
                self.text_cols = [c for c in optional_multi if c in self.data.columns]
                if not self.text_cols and priority_single in self.data.columns:
                    # fallback to single if present
                    self.text_cols = [priority_single]
                if self.text_cols:
                    # Prepare dictionary of raw captions
                    cap_matrix: List[List[str]] = []
                    for col in self.text_cols:
                        cap_matrix.append(self.data[col].fillna("").astype(str).tolist())
                    # Pre-tokenize all columns for all subjects
                    num_cols = len(self.text_cols)
                    N = len(self.subject_ids)
                    input_ids_tensor = torch.zeros(num_cols, N, self.max_length, dtype=torch.long)
                    attn_tensor = torch.zeros(num_cols, N, self.max_length, dtype=torch.long)
                    for c_idx, col_caps in enumerate(cap_matrix):
                        for s_idx, cap in enumerate(col_caps):
                            # Даже если строка пустая, токенизируем её, чтобы получить корректные спец-токены
                            # (например [CLS], [SEP]) и непустую attention_mask.
                            
                            # ---- Detect if control ----
                            data_path = self.data["DATA_PATH"].iloc[s_idx]
                            is_control = "_control_" in data_path
                            
                            # # ---- Replace text for controls ----
                            # if is_control and self.text_probs is not None:
                            #     if cap in ("No lesion detected", "full brain", "", " "):
                            #         cap = generate_random_text(self.text_probs)
                            #         # print("CONTROL -> GENERATED:", cap)

                            text_to_encode = cap if isinstance(cap, str) else ""

                            token_output = self.tokenizer.encode_plus(
                                text_to_encode,
                                padding="max_length",
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt",
                            )
                            input_ids_tensor[c_idx, s_idx] = token_output["input_ids"].squeeze(0)
                            attn_tensor[c_idx, s_idx] = token_output["attention_mask"].squeeze(0)
                    self._multi_input_ids = input_ids_tensor
                    self._multi_attention = attn_tensor

        self.roi_list: List[Optional[str]] = list(self.data["ROI_PATH"])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Сheck existing features npz
        features_dir = Path(self.feature_path) / "preprocessed" / "meld_files" / self.subject_ids[idx] / "features"
        dist_npz_path = features_dir / "distance_maps_gt.npz"

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

        if not self.text_emb:
            no_text_string = "full brain"
            token_output = self.tokenizer.encode_plus(
                no_text_string,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = token_output["input_ids"].squeeze(0)
            attention_mask = token_output["attention_mask"].squeeze(0)
            text = {"input_ids": input_ids, "attention_mask": attention_mask}

        elif self.tokenizer is not None:
            # Single-column pretokenized
            if self._single_input_ids is not None:
                input_ids = self._single_input_ids[idx]
                attention_mask = self._single_attention[idx]
            # Multi-column pretokenized
            elif self._multi_input_ids is not None:
                col_idx = random.randrange(self._multi_input_ids.shape[0])
                input_ids = self._multi_input_ids[col_idx, idx]
                attention_mask = self._multi_attention[col_idx, idx]
            else:  # No text
                input_ids = torch.zeros(self.max_length, dtype=torch.long)
                attention_mask = torch.zeros(self.max_length, dtype=torch.long)

            text = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        else:
            text = {"input_ids": torch.zeros(self.max_length, dtype=torch.long),
                    "attention_mask": torch.zeros(self.max_length, dtype=torch.long)}

        return {
            "subject_id": self.subject_ids[idx],
            "text": text,
            "roi": roi,
            "dist_maps": dist_maps,
        }
    
class SingleEpilepSample(Dataset):
    def __init__(self, data: dict, description: str, tokenizer, cohort, max_length: int = 256, text_emb: bool = True) -> None:
        super().__init__()
        self.keys = list(data.keys())
        self.description = description
        self.tokenizer = tokenizer
        self.cohort = cohort
        self.max_length = max_length
        # Pre-tokenize description once to avoid repeated work in __getitem__
        if not self.text_emb:
            # no-text experiment: фиксированный текст
            self.description = "full brain"
        else:
            self.description = description

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
        if not self.text_emb and self.tokenizer is not None:
            # заново токенизируем "full brain" на случай отсутствия в init
            token_output = self.tokenizer.encode_plus(
                "full brain",
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
        elif self._text_input_ids is not None:
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
