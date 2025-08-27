from typing import List, Dict, Tuple
import os
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import torch
import subprocess
from meld_graph.icospheres import IcoSpheres
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GraphNorm


# TODO: Conduct experiments with different GNN layers (GAT, GCN, etc.)
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.conv = SAGEConv(dim, dim, aggr="mean")
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
        h = self.dropout(h)
        h = self.norm2(h, batch)  # <- necessary second normalization
        return h


class VisionModel(nn.Module):
    def __init__(
        self,
        feature_dim: List[int],
        meld_script_path: str,
        feature_path: str,
        output_dir: str,
        device: str,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.meld_script_path = meld_script_path

        self.template_root = os.path.join(output_dir, "data4sharing")
        self.feature_path = feature_path
        self.output_dir = output_dir
        self.device = torch.device(device)
        ico_path = os.path.join("data", "icospheres")
        self.icos = IcoSpheres(icosphere_path=ico_path)
        self._nverts_to_level = {
            len(self.icos.icospheres[level]["coords"]): level
            for level in self.icos.icospheres
        }
        # GNN layers for each of the first five stages

        self.gnn_layers = nn.ModuleList(
            [
                ResidualBlock(feat_dim, dropout=0.1)  # MAKE HYPERPARAMETERS
                for feat_dim in feature_dim
            ]
        )

        # Precompute edge_index for stage1..stage7
        self.edge_index_per_stage = self._collect_edge_indices()

    def run_meld_prediction(self, subject_id: str):
        # command = [
        #     self.meld_script_path,
        #     'run_script_prediction.py',
        #     '-id', subject_id,
        #     '-harmo_code', 'fcd',
        #     '-demos', 'participants_with_scanner.tsv'
        # ]
        aug_flag = True  # CHANGE
        command = [
            self.meld_script_path,
            "run_script_prediction_meld.py",
            "-id",
            subject_id,
            "-harmo_code",
            "fcd",
            "-demos",
            "input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv",
            *(["--aug_mode", "train"] if aug_flag else []),
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running MELD prediction for {subject_id}: {e}")
            raise

    def _collect_edge_indices(self) -> Dict[str, Tuple[int, torch.Tensor]]:
        """
        1) Scan TEMPLATE_ROOT for fsaverage templates (fsaverage_sym, fsaverage6, fsaverage5, fsaverage4, fsaverage3)
           and collect {V_tmpl: path}.
        2) Find one subject’s NPZ to read (stage1..stage5, N_i >= 642). If missing, run MELD to generate it.
        3) For each stage1..stage5, build edge_index from the matching template.
        """

        # Step B: locate any subject folder under feature_path/input
        # input_root = os.path.join(self.feature_path, 'preprocessed', 'meld_files')
        input_root = os.path.join(
            self.feature_path, "input", "data4sharing", "meld_combats"
        )
        if not os.path.isdir(input_root):
            raise FileNotFoundError(f"Feature input directory not found: {input_root}")

        subject_dirs = []
        for d in os.listdir(input_root):
            if d.startswith("MELD_"):
                if "_control" in d:
                    subject_dirs.append(d.split("_control")[0])
                elif "_patient" in d:
                    subject_dirs.append(d.split("_patient")[0])

        if not subject_dirs:
            raise FileNotFoundError(f"No subject directories under {input_root}")

        # Use the first subject as example
        example_subj = subject_dirs[0]
        example_npz = os.path.join(
            self.feature_path,
            "preprocessed",
            "meld_files",
            example_subj,
            "features",
            "feature_maps.npz",
        )

        # If features NPZ does not exist, run MELD for that subject to generate it
        if not os.path.isfile(example_npz):
            self.run_meld_prediction(example_subj)
            if not os.path.isfile(example_npz):
                raise FileNotFoundError(
                    f"Failed to generate NPZ for subject {example_subj}: {example_npz}"
                )

        # Step C: read the NPZ to get stage keys and vertex counts
        npz = np.load(example_npz)
        all_stage_keys = sorted(npz.files, key=lambda k: int(k.replace("stage", "")))

        edge_index_per_stage: Dict[str, Tuple[int, torch.Tensor]] = {}
        for st in all_stage_keys:
            _, H, N_i, _ = npz[st].shape
            V_total = (
                N_i * H
            )  # total number of vertices: N_i vertices per hemisphere × H=2 hemispheres

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

            edge_index = torch.cat([edge_lh, edge_rh], dim=1).to(self.device)

            edge_index_per_stage[st] = (V_total, edge_index)

        return edge_index_per_stage

    def forward(self, subject_ids: List[str]):
        # Only use stage1..stage7 keys
        stage_keys = sorted(
            self.edge_index_per_stage.keys(), key=lambda k: int(k.replace("stage", ""))
        )
        num_used_stages = len(stage_keys)  # expected = 7

        # Prepare container: one list of Data per used stage
        graph_list_per_stage: List[List[Data]] = [[] for _ in range(num_used_stages)]

        for subject_id in subject_ids:
            # Step 1: ensure MELD features exist for this subject
            features_dir = os.path.join(
                self.feature_path, "preprocessed", "meld_files", subject_id, "features"
            )
            npz_path = os.path.join(features_dir, "feature_maps.npz")
            if os.path.isfile(npz_path):
                features = np.load(npz_path)
            else:
                raise FileNotFoundError(f"Failed to generate NPZ for {subject_id}")

            # Step 2: load subject’s NPZ

            sorted_keys = sorted(
                features.files, key=lambda k: int(k.replace("stage", ""))
            )
            # Only keep stage1..stage7
            sorted_keys = [st for st in sorted_keys if st in self.edge_index_per_stage]

            # Step 3: build graphs for each used stage

            for i, stage in enumerate(sorted_keys):
                feat_torch = torch.from_numpy(features[stage]).to(self.device)
                feat = torch.mean(feat_torch, dim=0)  # shape = (H, N_i, C_i)
                H, N, C = feat.shape
                feat_tensor = feat.view(H * N, C)

                # Retrieve precomputed edge_index
                _, edge_index = self.edge_index_per_stage[stage]
                data = Data(x=feat_tensor, edge_index=edge_index, num_nodes=H * N)

                graph_list_per_stage[i].append(data)

        # Batch each stage’s list of Data
        batched_per_stage: List[Batch] = []
        for i, data_list in enumerate(graph_list_per_stage):
            batch = Batch.from_data_list(data_list).to(self.device)

            V_total, _ = batch.x.size()
            N = V_total // (2 * batch.num_graphs)  # H = 2
            # TODO: Conduct experiments with/without GNN on small graphs
            if N >= 642:
                batch.x = self.gnn_layers[i](batch.x, batch.edge_index, batch.batch)

            batched_per_stage.append(batch)

        return {"feature": batched_per_stage}
