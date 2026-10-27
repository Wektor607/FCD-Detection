from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from meld_graph.data_preprocessing import Preprocess as Prep
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import FEATURE_PATH
from utils.config import SCRIPTS_DIR, DATA_DIR


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
        text_prob_json: str = None,
        atlas_type: str = "harvard_oxford",
        unclear_ratio: float = 0.33,
        seed: int = 42,
        training_mode: bool = False,
    ) -> None:
        super().__init__()

        assert subject_ids is not None, "subject_ids must be provided"

        self.feature_path = feature_path
        self.subject_ids = subject_ids
        self.text_emb = text_emb
        self.unclear_ratio = unclear_ratio
        self.seed = seed
        self.training = training_mode

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
            str(SCRIPTS_DIR / "config_files" / "final_ablation_full_with_combat_my.py")
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

        # >>> FIX: pre-tokenized no-text variant
        self._no_text_input_ids = None
        self._no_text_attention = None

        # Pre-compute per-subject patient flag for masking
        self._is_patient: List[bool] = []
        for sid in self.subject_ids:
            data_path = self.data.loc[sid, "DATA_PATH"]
            if isinstance(data_path, pd.Series):
                data_path = data_path.iloc[0]
            self._is_patient.append("_patient_" in str(data_path))

        # Patient and control indices for stratified epoch sampling
        self._pat_indices: List[int] = [i for i, is_p in enumerate(self._is_patient) if is_p]
        self._ctrl_indices: List[int] = [i for i, is_p in enumerate(self._is_patient) if not is_p]
        # Epoch state: active subject indices and which patients are masked this epoch
        self._epoch_active_indices: List[int] = list(range(len(self.subject_ids)))
        self._masked_patient_set: set = set()

        if self.tokenizer is not None:
            # Always pre-tokenize "unclear" for masking / no-text fallback
            token_output = self.tokenizer.encode_plus(
                "unclear",
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            self._no_text_input_ids = token_output["input_ids"].squeeze(0)
            self._no_text_attention = token_output["attention_mask"].squeeze(0)

            # Detect available text columns
            optional_multi = [
                # "full_text",
                "hemisphere_text",
                "lobe_text",
                "dominant_lobe_text",
                "hemisphere_lobe_text",
                # "no_text",  # removed: always "unclear", masking handled by set_epoch
            ]

            # Find the first column whose name contains atlas_type
            matched_col = next((c for c in self.data.columns if atlas_type in c), None)

            if matched_col is not None and not any(
                c in self.data.columns for c in optional_multi
            ):
                # Only one text column scenario
                captions = self.data[matched_col].fillna("").astype(str).tolist()
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

                ## CHECK LATER
                self.text_cols = [c for c in optional_multi if c in self.data.columns]
                if not self.text_cols and atlas_type in self.data.columns:
                    self.text_cols = [atlas_type]
                ##############################################
                if self.text_cols:
                    cap_matrix: List[List[str]] = []
                    for col in self.text_cols:
                        cap_matrix.append(self.data[col].fillna("").astype(str).tolist())

                    num_cols = len(self.text_cols)
                    N = len(self.subject_ids)
                    input_ids_tensor = torch.zeros(num_cols, N, self.max_length, dtype=torch.long)
                    attn_tensor = torch.zeros(num_cols, N, self.max_length, dtype=torch.long)

                    for c_idx, col_caps in enumerate(cap_matrix):
                        for s_idx, cap in enumerate(col_caps):
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

        if self.training:
            self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Per-epoch stratified sampling: k controls subsampled, same k patients masked to 'unclear'.
        The same patient can appear masked or unmasked in different epochs (per-epoch randomisation).
        k = min(round(unclear_ratio * n_patients), n_controls)
        """
        if not self.training or not self.text_emb or self.tokenizer is None:
            self._epoch_active_indices = list(range(len(self.subject_ids)))
            self._masked_patient_set = set()
            return

        rng = random.Random(self.seed + int(epoch))

        k = min(round(self.unclear_ratio * len(self._pat_indices)), len(self._ctrl_indices))

        # Subsample k controls for this epoch
        ctrl_chosen = rng.sample(self._ctrl_indices, k) if k > 0 else []

        # Randomly mask k patients to "unclear" (different each epoch)
        pat_masked = rng.sample(self._pat_indices, k) if k > 0 else []

        self._masked_patient_set = set(pat_masked)
        # Epoch dataset = all patients + only the k selected controls
        self._epoch_active_indices = self._pat_indices + ctrl_chosen

        n_pat_unclear = len(pat_masked)
        n_ctrl_unclear = len(ctrl_chosen)
        n_pat_with_text = len(self._pat_indices) - k
        print(
            f"[EpilepDataset] epoch={epoch}  unclear_ratio={self.unclear_ratio:.2f}  "
            f"k={k}  active={len(self._epoch_active_indices)}  "
            f"unclear: {n_pat_unclear} pat + {n_ctrl_unclear} ctrl (50/50)  "
            f"with_text: {n_pat_with_text} patients"
        )

    def __len__(self) -> int:
        if self.training:
            return len(self._epoch_active_indices)
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx: int = self._epoch_active_indices[idx] if self.training else idx
        features_dir = Path(FEATURE_PATH) / self.subject_ids[real_idx] / "features"
        dist_npz_path = features_dir / "distance_maps_gt.npz"

        subject_data_list = self.prep.get_data_preprocessed(
            subject=self.subject_ids[real_idx],
            features=self.prep.params["features"],
            lobes=self.prep.params["lobes"],
            lesion_bias=False,
            distance_maps=False,
            harmo_code="fcd",
            only_lesion=False,
            only_features=self.roi_list[real_idx] is None,
            combine_hemis=self.prep.params["combine_hemis"],
        )

        labels_tensors = []
        for d in subject_data_list:
            if d.get("labels") is None:
                labels_tensors.append(torch.zeros(d["features"].shape[0], dtype=torch.long))
            else:
                labels_tensors.append(torch.from_numpy(d["labels"]).long())

        roi = torch.stack(labels_tensors, dim=0)

        if not dist_npz_path.is_file():
            raise FileNotFoundError(f"Failed to generate NPZ for {self.subject_ids[real_idx]}")

        dist_maps = torch.from_numpy(np.load(dist_npz_path)["arr_0"]).float()

        if not self.text_emb:
            input_ids = self._no_text_input_ids
            attention_mask = self._no_text_attention

        elif self.tokenizer is not None:
            if self._single_input_ids is not None:
                input_ids = self._single_input_ids[real_idx]
                attention_mask = self._single_attention[real_idx]
            elif self._multi_input_ids is not None:
                col_idx = random.randrange(self._multi_input_ids.shape[0])
                input_ids = self._multi_input_ids[col_idx, real_idx]
                attention_mask = self._multi_attention[col_idx, real_idx]
            else:
                input_ids = torch.zeros(self.max_length, dtype=torch.long)
                attention_mask = torch.zeros(self.max_length, dtype=torch.long)

            # Stratified masking: patients selected by set_epoch get "unclear" this epoch
            if self.training and real_idx in self._masked_patient_set:
                input_ids = self._no_text_input_ids
                attention_mask = self._no_text_attention
            # Controls always have "unclear" text naturally — no extra logic needed
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        return {
            "subject_id": self.subject_ids[real_idx],
            "text": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            "roi": roi,
            "dist_maps": dist_maps,
        }

    
class SingleEpilepSample(Dataset):
    def __init__(
        self,
        data: dict,
        description: str,
        tokenizer,
        max_length: int = 256,
    ):
        super().__init__()
        self.keys = list(data.keys())
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not description:
            description = "full brain"

        if description and tokenizer is not None:
            token_output = tokenizer.encode_plus(
                description,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            self._text_input_ids = token_output["input_ids"].squeeze(0)
            self._text_attention_mask = token_output["attention_mask"].squeeze(0)
        else:
            self._text_input_ids = torch.zeros(self.max_length, dtype=torch.long)
            self._text_attention_mask = torch.zeros(self.max_length, dtype=torch.long)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        key = self.keys[idx]

        return {
            "subject_id": key,
            "text": {
                "input_ids": self._text_input_ids,
                "attention_mask": self._text_attention_mask,
            },
        }

# from __future__ import annotations

# import csv
# import json
# import os
# import random
# import sys
# from pathlib import Path
# from typing import Any, Dict, List, Optional

# REPO_ROOT = Path(__file__).resolve().parents[2]
# sys.path.insert(0, str(REPO_ROOT))

# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# from meld_graph.data_preprocessing import Preprocess as Prep
# from meld_graph.meld_cohort import MeldCohort
# from meld_graph.paths import FEATURE_PATH
# from utils.config import SCRIPTS_DIR, DATA_DIR


# def load_config(config_file: str) -> Any:
#     """load config.py file and return config object"""
#     import importlib.machinery
#     import importlib.util

#     loader = importlib.machinery.SourceFileLoader("config", config_file)
#     spec = importlib.util.spec_from_loader(loader.name, loader)
#     config = importlib.util.module_from_spec(spec)
#     loader.exec_module(config)
#     return config

# class EpilepDataset(Dataset):
#     def __init__(
#         self,
#         csv_path: str,
#         tokenizer: Any,
#         feature_path: str = "",
#         subject_ids: Optional[List[str]] = None,
#         cohort: MeldCohort = None,
#         max_length: int = 256,
#         text_emb: bool = False,
#         text_prob_json: str = None,
#         atlas_type: str = "harvard_oxford",
#         unclear_ratio: float = 0.3,
#         seed: int = 42,
#         training_mode: bool = False,
#     ) -> None:
#         super().__init__()

#         assert subject_ids is not None, "subject_ids must be provided"

#         self.feature_path = feature_path
#         self.subject_ids = subject_ids
#         self.text_emb = text_emb
#         self.unclear_ratio = unclear_ratio
#         self.training = training_mode
#         self.seed = seed

#         csv_path = Path(csv_path)
#         self.data = pd.read_csv(
#             csv_path, sep=",", engine="python", quoting=csv.QUOTE_NONE, escapechar="\\"
#         )

#         self.tokenizer = tokenizer

#         # 2) extract sub-ID
#         self.data["sub"] = self.data["DATA_PATH"].apply(
#             lambda p: os.path.basename(p).split("_patient")[0].split("_control")[0]
#             if isinstance(p, str)
#             else None
#         )
#         self.data = self.data.set_index("sub").loc[subject_ids]

#         self.config = load_config(
#             str(SCRIPTS_DIR / "config_files" / "final_ablation_full_with_combat_my.py")
#         )
#         params = (
#             next(iter(self.config.losses))
#             if isinstance(self.config.losses, list)
#             else self.config.losses
#         )
#         self.prep = Prep(cohort=cohort, params=params["data_parameters"])
#         self.max_length = max_length

#         # ---- Upload json ----
#         self.text_probs = None
#         if text_prob_json is not None:
#             with open(text_prob_json, "r", encoding="utf-8") as f:
#                 self.text_probs = json.load(f)

#         # ---- Text handling ----
#         self.text_cols: List[str] = []
#         self._single_input_ids = None
#         self._single_attention = None
#         self._multi_input_ids = None  # shape [num_cols, N, L]
#         self._multi_attention = None  # shape [num_cols, N, L]

#         # >>> FIX: pre-tokenized no-text variant
#         self._no_text_input_ids = None
#         self._no_text_attention = None

#         # Pre-compute per-subject patient flag for masking
#         self._is_patient: List[bool] = []
#         for sid in self.subject_ids:
#             data_path = self.data.loc[sid, "DATA_PATH"]
#             if isinstance(data_path, pd.Series):
#                 data_path = data_path.iloc[0]
#             self._is_patient.append("_patient_" in str(data_path))


#         # Indices for stratified "unclear" selection
#         self._ctrl_indices: List[int] = [i for i, is_p in enumerate(self._is_patient) if not is_p]
#         self._pat_indices: List[int]  = [i for i, is_p in enumerate(self._is_patient) if is_p]

#         # Set of indices masked to "unclear" for the current epoch
#         self._unclear_set: set[int] = set()

#         if self.tokenizer is not None:
#             # Always pre-tokenize "unclear" for masking / no-text fallback
#             token_output = self.tokenizer.encode_plus(
#                 "unclear",
#                 padding="max_length",
#                 max_length=self.max_length,
#                 truncation=True,
#                 return_attention_mask=True,
#                 return_tensors="pt",
#             )
#             self._no_text_input_ids = token_output["input_ids"].squeeze(0)
#             self._no_text_attention = token_output["attention_mask"].squeeze(0)

#             # Detect available text columns
#             optional_multi = [
#                 # "full_text",
#                 "hemisphere_text",
#                 "lobe_text",
#                 "dominant_lobe_text",
#                 "hemisphere_lobe_text",
#                 # "no_text",  # removed: always "unclear", masking handled by patient_text_mask_prob
#             ]

#             # Find the first column whose name contains atlas_type
#             matched_col = next((c for c in self.data.columns if atlas_type in c), None)

#             if matched_col is not None and not any(
#                 c in self.data.columns for c in optional_multi
#             ):
#                 # Only one text column scenario
#                 captions = self.data[matched_col].fillna("").astype(str).tolist()
#                 ids_list = []
#                 att_list = []
#                 for cap in captions:
#                     token_output = self.tokenizer.encode_plus(
#                         cap,
#                         padding="max_length",
#                         max_length=self.max_length,
#                         truncation=True,
#                         return_attention_mask=True,
#                         return_tensors="pt",
#                     )
#                     ids_list.append(token_output["input_ids"].squeeze(0))
#                     att_list.append(token_output["attention_mask"].squeeze(0))

#                 self._single_input_ids = torch.stack(ids_list, dim=0)
#                 self._single_attention = torch.stack(att_list, dim=0)

#             else:
#                 # Multi-column case

#                 ## CHECK LATER
#                 self.text_cols = [c for c in optional_multi if c in self.data.columns]
#                 if not self.text_cols and atlas_type in self.data.columns:
#                     self.text_cols = [atlas_type]
#                 ##############################################
#                 if self.text_cols:
#                     cap_matrix: List[List[str]] = []
#                     for col in self.text_cols:
#                         cap_matrix.append(self.data[col].fillna("").astype(str).tolist())

#                     num_cols = len(self.text_cols)
#                     N = len(self.subject_ids)
#                     input_ids_tensor = torch.zeros(num_cols, N, self.max_length, dtype=torch.long)
#                     attn_tensor = torch.zeros(num_cols, N, self.max_length, dtype=torch.long)

#                     for c_idx, col_caps in enumerate(cap_matrix):
#                         for s_idx, cap in enumerate(col_caps):
#                             text_to_encode = cap if isinstance(cap, str) else ""

#                             token_output = self.tokenizer.encode_plus(
#                                 text_to_encode,
#                                 padding="max_length",
#                                 max_length=self.max_length,
#                                 truncation=True,
#                                 return_attention_mask=True,
#                                 return_tensors="pt",
#                             )

#                             input_ids_tensor[c_idx, s_idx] = token_output["input_ids"].squeeze(0)
#                             attn_tensor[c_idx, s_idx] = token_output["attention_mask"].squeeze(0)

#                     self._multi_input_ids = input_ids_tensor
#                     self._multi_attention = attn_tensor

#         self.roi_list: List[Optional[str]] = list(self.data["ROI_PATH"])
#         # If training, initialize epoch 0 selection (optional, but удобнее)
#         if self.training:
#             self.set_epoch(0)

#     def __len__(self) -> int:
#         return len(self.subject_ids)

#     def set_epoch(self, epoch: int) -> None:
#         """Choose a fixed ratio of samples to receive 'unclear' text, stratified 50/50 by class."""
#         if not self.training or self.tokenizer is None or not self.text_emb:
#             # If not in training or no text encoder, do nothing
#             self._unclear_set = set()
#             return

#         rng = random.Random(self.seed + int(epoch))

#         N = len(self.subject_ids)
#         k_total = int(round(self.unclear_ratio * N))

#         # target split: half patients, half controls (as close as possible)
#         k_pat = k_total // 2
#         k_ctrl = k_total - k_pat

#         # if one class is smaller, clip and reassign the remainder to the other class
#         k_pat = min(k_pat, len(self._pat_indices))
#         k_ctrl = min(k_ctrl, len(self._ctrl_indices))

#         # if we clipped, try to fill remaining slots from the other class
#         k_filled = k_pat + k_ctrl
#         if k_filled < k_total:
#             remaining = k_total - k_filled
#             # fill from whichever class still has spare samples
#             pat_spare = len(self._pat_indices) - k_pat
#             ctrl_spare = len(self._ctrl_indices) - k_ctrl
#             take_pat = min(remaining, pat_spare)
#             take_ctrl = min(remaining - take_pat, ctrl_spare)
#             k_pat += take_pat
#             k_ctrl += take_ctrl

#         pat_chosen = rng.sample(self._pat_indices, k_pat) if k_pat > 0 else []
#         ctrl_chosen = rng.sample(self._ctrl_indices, k_ctrl) if k_ctrl > 0 else []

#         self._unclear_set = set(pat_chosen + ctrl_chosen)
#         # --- DEBUG: check stratified unclear split ---
#         n_total = len(self._unclear_set)
#         n_pat = sum(1 for i in self._unclear_set if i in set(self._pat_indices))
#         n_ctrl = n_total - n_pat
#         print(
#             f"[EpilepDataset] epoch={epoch} unclear_ratio={self.unclear_ratio:.2f} "
#             f"unclear_total={n_total}/{len(self.subject_ids)} "
#             f"(pat={n_pat}, ctrl={n_ctrl})"
#         )
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         features_dir = Path(FEATURE_PATH) /self.subject_ids[idx] / "features"
#         dist_npz_path = features_dir / "distance_maps_gt.npz"

#         subject_data_list = self.prep.get_data_preprocessed(
#             subject=self.subject_ids[idx],
#             features=self.prep.params["features"],
#             lobes=self.prep.params["lobes"],
#             lesion_bias=False,
#             distance_maps=False,
#             harmo_code="fcd",
#             only_lesion=False,
#             only_features=self.roi_list[idx] is None,
#             combine_hemis=self.prep.params["combine_hemis"],
#         )

#         labels_tensors = []
#         for d in subject_data_list:
#             if d.get("labels") is None:
#                 labels_tensors.append(torch.zeros(d["features"].shape[0], dtype=torch.long))
#             else:
#                 labels_tensors.append(torch.from_numpy(d["labels"]).long())

#         roi = torch.stack(labels_tensors, dim=0)

#         if not dist_npz_path.is_file():
#             raise FileNotFoundError(f"Failed to generate NPZ for {self.subject_ids[idx]}")

#         dist_maps = torch.from_numpy(np.load(dist_npz_path)["arr_0"]).float()

#         if not self.text_emb or self.tokenizer is None:
#             input_ids = self._no_text_input_ids if self._no_text_input_ids is not None else torch.zeros(self.max_length, dtype=torch.long)
#             attention_mask = self._no_text_attention if self._no_text_attention is not None else torch.zeros(self.max_length, dtype=torch.long)

#         elif self.tokenizer is not None:
#             if self._single_input_ids is not None:
#                 input_ids = self._single_input_ids[idx]
#                 attention_mask = self._single_attention[idx]
#             elif self._multi_input_ids is not None:
#                 col_idx = random.randrange(self._multi_input_ids.shape[0])
#                 input_ids = self._multi_input_ids[col_idx, idx]
#                 attention_mask = self._multi_attention[col_idx, idx]
#             else:
#                 input_ids = torch.zeros(self.max_length, dtype=torch.long)
#                 attention_mask = torch.zeros(self.max_length, dtype=torch.long)

#             # Stochastic text masking (train only) — equal "unclear" rate for patients and controls
#             # Fixed-ratio, stratified "unclear" masking (train only)
#             if self.training and (idx in self._unclear_set):
#                 input_ids = self._no_text_input_ids
#                 attention_mask = self._no_text_attention
#         else:
#             input_ids = torch.zeros(self.max_length, dtype=torch.long)
#             attention_mask = torch.zeros(self.max_length, dtype=torch.long)

#         return {
#             "subject_id": self.subject_ids[idx],
#             "text": {
#                 "input_ids": input_ids,
#                 "attention_mask": attention_mask,
#             },
#             "roi": roi,
#             "dist_maps": dist_maps,
#         }

    
# class SingleEpilepSample(Dataset):
#     def __init__(
#         self,
#         data: dict,
#         description: str,
#         tokenizer,
#         max_length: int = 256,
#     ):
#         super().__init__()
#         self.keys = list(data.keys())
#         self.tokenizer = tokenizer
#         self.max_length = max_length

#         if not description:
#             description = "full brain"

#         if description and tokenizer is not None:
#             token_output = tokenizer.encode_plus(
#                 description,
#                 padding="max_length",
#                 max_length=self.max_length,
#                 truncation=True,
#                 return_attention_mask=True,
#                 return_tensors="pt",
#             )
#             self._text_input_ids = token_output["input_ids"].squeeze(0)
#             self._text_attention_mask = token_output["attention_mask"].squeeze(0)
#         else:
#             self._text_input_ids = torch.zeros(self.max_length, dtype=torch.long)
#             self._text_attention_mask = torch.zeros(self.max_length, dtype=torch.long)

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx: int):
#         key = self.keys[idx]

#         return {
#             "subject_id": key,
#             "text": {
#                 "input_ids": self._text_input_ids,
#                 "attention_mask": self._text_attention_mask,
#             },
#         }

