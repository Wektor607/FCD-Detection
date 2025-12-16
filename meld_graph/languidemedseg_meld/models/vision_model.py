from __future__ import annotations

from typing import List, Dict, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv, GraphNorm

from meld_graph.icospheres import IcoSpheres


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
        meld_script_path: str | Path,
        feature_path: str | Path,
        output_dir: str | Path,
        device: str | torch.device,
        gnn_min_verts: int = 642,
        fold_number: int = 0
    ) -> None:
        super().__init__()

        self.meld_script_path = Path(meld_script_path)
        self.feature_path = Path(feature_path)
        self.output_dir = Path(output_dir)
        ico_path = Path("data/icospheres")
        self.template_root = self.output_dir / "data4sharing"

        self.fold_number = fold_number
        self.device = (
            torch.device(device) if not isinstance(device, torch.device) else device
        )

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
        1) Scan TEMPLATE_ROOT for fsaverage templates (fsaverage_sym, fsaverage6, fsaverage5, fsaverage4, fsaverage3)
           and collect {V_tmpl: path}.
        2) Find one subject’s NPZ to read (stage1..stage5, N_i >= 642). If missing, run MELD to generate it.
        3) For each stage1..stage5, build edge_index from the matching template.
        """

        # Step B: locate any subject folder under feature_path/input
        input_root = self.feature_path / "input" / "data4sharing" / "meld_combats"
        if not input_root.is_dir():
            raise FileNotFoundError(f"Feature input directory not found: {input_root}")

        subject_dirs = []
        patient_subjects = []
        for d in input_root.iterdir():
            file_name = d.name
            if file_name.startswith("MELD_"):
                if "_control" in file_name:
                    subject_dirs.append(file_name.split("_control")[0])
                elif "_patient" in file_name:
                    subj = file_name.split("_patient")[0]
                    subject_dirs.append(subj)
                    patient_subjects.append(subj)

        if not subject_dirs:
            raise FileNotFoundError(f"No subject directories under {input_root}")
        if not patient_subjects:
            raise FileNotFoundError(f"No patient subject directories under {input_root}")

        # Use the first _patient subject as example
        example_subj = patient_subjects[0]
        example_npz = (
            self.feature_path
            / "preprocessed"
            / "meld_files"
            / example_subj
            / "features"
            / "feature_maps.npz"
        )
        if not example_npz.is_file():
            raise FileNotFoundError(f"Example NPZ not found: {example_npz}")

        # Step C: read the NPZ to get stage keys and vertex counts
        with np.load(example_npz, allow_pickle=False) as npz:
            all_stage_keys = sorted(
                npz.files, key=lambda k: int(k.replace("stage", ""))
            )

            edge_index_per_stage: Dict[str, Tuple[int, torch.Tensor]] = {}
            for st in all_stage_keys:
                _, H, N_i, _ = npz[st].shape
                V_total = N_i * H # total number of vertices: N_i vertices per hemisphere × H=2 hemispheres

                # find at what level of the icosphere exactly N_i vertices
                if N_i not in self._nverts_to_level:
                    raise ValueError(f"No level found for N_i={N_i}")
                level = self._nverts_to_level[N_i]

                # t_edges is [2, E] for one "hemisphere"
                t_edges = self.icos.icospheres[level]["t_edges"]

                # duplicate for the right hemisphere, shifting the indices by N_i
                edge_lh = t_edges
                edge_rh = t_edges.clone()
                edge_rh[0] += N_i
                edge_rh[1] += N_i

                # Keep edge_index on CPU during initialization to avoid triggering CUDA
                # initialization (which can fail if NVIDIA drivers are missing).
                edge_index = torch.cat([edge_lh, edge_rh], dim=1)

                edge_index_per_stage[st] = (V_total, edge_index)

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
            npz_path = (
                self.feature_path
                / "preprocessed"
                / "meld_files"
                / subject_id
                / "features"
                / "feature_maps.npz"
            )

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
                # print(subject_id)
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
