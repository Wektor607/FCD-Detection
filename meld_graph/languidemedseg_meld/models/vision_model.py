from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GraphNorm, SAGEConv

from meld_graph.icospheres import IcoSpheres
from meld_graph.paths import FEATURE_PATH
from utils.config import DATA_DIR, REPO_ROOT


# TODO: Conduct experiments with different GNN layers (GAT, GCN, etc.)
class ResidualBlock(nn.Module):
    # def __init__(self, dim, dropout=0.1, aggr="mean", layerscale=0.1):
    #     super().__init__()
    #     self.norm = nn.LayerNorm(dim)              # стабильнее GraphNorm
    #     self.conv = SAGEConv(dim, dim, aggr=aggr)  # попробуй aggr="max" для очагов
    #     self.act  = nn.GELU()
    #     self.drop = nn.Dropout(dropout)
    #     self.alpha = nn.Parameter(torch.tensor(layerscale))  # скейл резидуала

    # def forward(self, x, edge_index, batch=None):
    #     h = self.norm(x)                # pre-norm
    #     h = self.conv(h, edge_index)
    #     h = self.act(h)
    #     h = self.drop(h)
    #     return x + self.alpha * h       # без пост-нормы: сохраняем identity

    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.conv = SAGEConv(dim, dim, aggr="max")
        self.norm1 = GraphNorm(dim)  # BatchNorm doesn't work here
        self.norm2 = GraphNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = self.norm1(x, batch)  # <- give higher perfomance
        h = self.conv(h, edge_index)
        # residual connection
        h = x + h
        h = self.relu(h)
        # h = self.dropout(h) # <- test it
        h = self.norm2(h, batch)  # <- necessary second normalization
        return h


class VisionModel(nn.Module):
    def __init__(
        self,
        feature_dim: List[int],
        device: str | torch.device,
        gnn_min_verts: int = 642,
        fold_number: int = 0
    ) -> None:
        super().__init__()

        ico_path = REPO_ROOT / "data" / "icospheres"

        self.fold_number = fold_number

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, int):
            # 0, 1, ... → GPU index, но если CUDA нет — fallback
            if torch.cuda.is_available():
                device = f"cuda:{device}"
            else:
                device = "cpu"
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                device = torch.device("cpu")
        else:
            device = str(device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

        self.device = device if isinstance(device, torch.device) else torch.device(device)


        self.icos = IcoSpheres(icosphere_path=str(ico_path))
        self._nverts_to_level = {
            len(self.icos.icospheres[level]["coords"]): level
            for level in self.icos.icospheres
        }
        # GNN layers for each of the first five stages
        self.gnn_min_verts = gnn_min_verts
        self.gnn_layers = nn.ModuleList(
            [
                ResidualBlock(feat_dim, dropout=0.1)  # MAKE HYPERPARAMETERS
                for feat_dim in feature_dim
            ]
        )

        # Precompute edge_index for stage1..stage7
        self.edge_index_per_stage = self._collect_edge_indices()

    def _collect_edge_indices(self) -> Dict[str, Tuple[int, torch.Tensor]]:
        """
        Build edge_index for each stage directly from icospheres.
        Assumes:
        stage1 -> ico1
        stage2 -> ico2
        ...
        stage7 -> ico7
        """

        edge_index_per_stage = {}
        H = 2  # number of hemispheres
        for level in range(1, 8):  # ico1 ... ico7
            stage = f"stage{level}"

            ico = self.icos.icospheres[level]
            N_i = ico["coords"].shape[0]     # vertices per hemisphere
            V_total = H * N_i

            # base edges for one hemisphere: [2, E]
            edge_lh = ico["t_edges"].clone()

            # duplicate for RH with index shift
            edge_rh = edge_lh.clone()
            edge_rh[0] += N_i
            edge_rh[1] += N_i

            edge_index = torch.cat([edge_lh, edge_rh], dim=1)

            # keep on CPU
            edge_index_per_stage[stage] = (V_total, edge_index)

        return edge_index_per_stage

    def forward(self, subject_ids: List[str]) -> Dict[str, List[Batch]]:
        # Only use stage1..stage7 keys
        stage_keys = sorted(
            self.edge_index_per_stage.keys(), key=lambda k: int(k.replace("stage", ""))
        )
        num_used_stages = len(stage_keys)  # expected = 7

        # Prepare container: one list of Data per used stage
        graph_list_per_stage: List[List[Data]] = [[] for _ in range(num_used_stages)]

        for subject_id in subject_ids:
            # Step 1: ensure MELD features exist for this subject
            npz_path = Path(FEATURE_PATH) / subject_id / "features" / "feature_maps.npz"

            if not npz_path.is_file():
                raise FileNotFoundError(
                    f"feature_maps.npz not found for subject '{subject_id}': {npz_path}"
                )

            # Step 2: load subject’s NPZ
            with np.load(npz_path, allow_pickle=False) as features:
                sorted_keys = sorted(
                    features.files, key=lambda k: int(k.replace("stage", ""))
                )
                # Only keep stage1..stage7
                sorted_keys = [
                    st for st in sorted_keys if st in self.edge_index_per_stage
                ]

                # Step 3: build graphs for each used stage
                for i, stage in enumerate(sorted_keys):
                    feat_torch = torch.from_numpy(features[stage])
                    # Move to device if possible; fallback to CPU on any error to avoid
                    # triggering CUDA initialization when drivers are missing.
                    try:
                        if getattr(self.device, 'type', None) == 'cuda' and torch.cuda.is_available():
                            _ = torch.cuda.current_device()
                            feat_torch = feat_torch.to(self.device)
                        else:
                            # keep on CPU
                            pass
                    except Exception:
                        # couldn't move to CUDA — keep tensor on CPU
                        pass

                    feat = feat_torch[self.fold_number]  # shape = (H, N_i, C_i)
                    H, N, C = feat.shape
                    feat_tensor = feat.view(H * N, C)

                    # Retrieve precomputed edge_index
                    _, edge_index = self.edge_index_per_stage[stage]
                    
                    data = Data(x=feat_tensor, edge_index=edge_index, num_nodes=H * N)

                    graph_list_per_stage[i].append(data)

        # Batch each stage’s list of Data
        batched_per_stage: List[Batch] = []
        for i, data_list in enumerate(graph_list_per_stage):
            batch = Batch.from_data_list(data_list)
            # Move batch to device only if the device is CUDA and CUDA can be initialized.
            if getattr(self.device, 'type', None) == 'cuda' and torch.cuda.is_available():
                try:
                    _ = torch.cuda.current_device()
                    batch = batch.to(self.device)
                except Exception:
                    # CUDA not available at runtime; keep batch on CPU
                    pass

            V_total, _ = batch.x.size()
            N = V_total // (2 * batch.num_graphs)  # H = 2

            if N >= self.gnn_min_verts:
                batch.x = self.gnn_layers[i](batch.x, batch.edge_index, batch.batch)

            batched_per_stage.append(batch)

        return {"feature": batched_per_stage}
