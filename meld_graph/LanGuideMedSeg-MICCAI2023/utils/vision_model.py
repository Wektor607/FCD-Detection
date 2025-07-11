from typing import List, Dict, Tuple
import os
import sys
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import numpy as np
import torch
import subprocess
from nibabel.freesurfer import read_geometry
from torch_geometric.data import Data, Batch
from torch_geometric.nn import TransformerConv, GraphNorm
import grafog.transforms as T

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        # in_channels=dim, out_channels=dim — сохраняем размерность
        self.conv = SAGEConv(dim, dim, aggr='mean')
        self.norm1 = GraphNorm(dim) # BatchNorm doesn't work here
        self.norm2 = GraphNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.norm1(x) # <- give higher perfomance
        h = self.conv(x, edge_index)
        # residual connection
        h = x + h
        h = self.relu(h)
        # h = self.dropout(h)
        h = self.norm2(h) # <- necessary second normalization
        return h
    
class VisionModel(nn.Module):
    def __init__(
        self,
        feature_dim: List[int],
        meld_script_path: str,
        feature_path: str,
        output_dir: str,
        device: str
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.meld_script_path = meld_script_path
        
        self.template_root = os.path.join(output_dir, "fs_outputs")
        self.feature_path = feature_path
        self.output_dir = output_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # GNN layers for each of the first five stages  
        
        self.gnn_layers = nn.ModuleList([
            ResidualBlock(feat_dim, dropout=0.1) # MAKE HYPERPARAMETERS
            for feat_dim in feature_dim
        ])

        # Precompute edge_index for stage1..stage7
        self.edge_index_per_stage = self._collect_edge_indices()

    def run_meld_prediction(self, subject_id: str):
        command = [
            self.meld_script_path,
            'run_script_prediction.py',
            '-id', subject_id,
            '-harmo_code', 'fcd',
            '-demos', 'participants_with_scanner.tsv'
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error running MELD prediction for {subject_id}: {e}')
            raise
        
    def _collect_edge_indices(self) -> Dict[str, Tuple[int, torch.Tensor]]:
        """
        1) Scan TEMPLATE_ROOT for fsaverage templates (fsaverage_sym, fsaverage6, fsaverage5, fsaverage4, fsaverage3)
           and collect {V_tmpl: path}.
        2) Find one subject’s NPZ to read (stage1..stage5, N_i >= 642). If missing, run MELD to generate it.
        3) For each stage1..stage5, build edge_index from the matching template.
        """
        # Step A: find all candidate templates in TEMPLATE_ROOT
        candidate_templates: Dict[int, str] = {}
        for name in os.listdir(self.template_root):
            if 'sub' in name or name == 'fsaverage':
                continue 

            tmpl_dir = os.path.join(self.template_root, name)
            surf_dir = os.path.join(tmpl_dir, "surf")
            lh_pial = os.path.join(surf_dir, "lh.pial")
            rh_pial = os.path.join(surf_dir, "rh.pial")
            if os.path.isdir(tmpl_dir) and os.path.isdir(surf_dir) and os.path.isfile(lh_pial) and os.path.isfile(rh_pial):
                coords_lh, _ = read_geometry(lh_pial)
                V_lh = coords_lh.shape[0]
                coords_rh, _ = read_geometry(rh_pial)
                V_rh = coords_rh.shape[0]
                if V_lh != V_rh:
                    continue
                candidate_templates[V_lh] = tmpl_dir
        
        # Step B: locate any subject folder under feature_path/input
        input_root = os.path.join(self.feature_path, "preprocessed")#"input", "ds004199")
        if not os.path.isdir(input_root):
            raise FileNotFoundError(f"Feature input directory not found: {input_root}")

        subject_dirs = [
            d for d in os.listdir(input_root)
            if d.startswith("sub-") and os.path.isdir(os.path.join(input_root, d))
        ]
        
        if not subject_dirs:
            raise FileNotFoundError(f"No subject directories under {input_root}")

        # Use the first subject as example
        example_subj = subject_dirs[0]
        # example_npz = os.path.join(input_root, example_subj, "anat", "features", "feature_maps.npz")
        example_npz = os.path.join(input_root, example_subj, "features", "feature_maps.npz")

        # If features NPZ does not exist, run MELD for that subject to generate it
        if not os.path.isfile(example_npz):
        # not os.path.isfile(os.path.join(self.output_dir, "predictions_reports", f"{example_subj}", "predictions/prediction.nii.gz")) or \
            self.run_meld_prediction(example_subj)
            if not os.path.isfile(example_npz):
                raise FileNotFoundError(f"Failed to generate NPZ for subject {example_subj}: {example_npz}")

        # Step C: read the NPZ to get stage keys and vertex counts
        npz = np.load(example_npz)
        all_stage_keys = sorted(npz.files, key=lambda k: int(k.replace("stage", "")))

        edge_index_per_stage: Dict[str, Tuple[int, torch.Tensor]] = {}
        for st in all_stage_keys:
            _, H, N_i, _ = npz[st].shape
            V_total = N_i * H   # общее число вершин: N_i вершин на полушарие × H=2 полушарий

            if N_i < 642:
                # Для малых стадий просто создаём random graph
                # возьмём, скажем, 2*V_total случайных рёбер
                num_rand_edges = V_total * 2
                src = torch.randint(0, V_total, (num_rand_edges,), device=self.device)
                dst = torch.randint(0, V_total, (num_rand_edges,), device=self.device)
                edge_index = torch.stack([src, dst], dim=0)
                edge_index_per_stage[st] = (V_total, edge_index)
                continue

            # Существующая логика для stage1..stage5:
            if N_i not in candidate_templates:
                raise ValueError(f"Template for {st} with N_i={N_i} not found in {self.template_root}")
            tmpl_dir = candidate_templates[N_i]
            lh_pial = os.path.join(tmpl_dir, "surf", "lh.pial")
            rh_pial = os.path.join(tmpl_dir, "surf", "rh.pial")

            _, edge_lh = self._build_surf_edge_index_from_pial(lh_pial)
            _, edge_rh = self._build_surf_edge_index_from_pial(rh_pial)

            # shift RH indices by N_i
            edge_rh_shifted = edge_rh.clone()
            edge_rh_shifted[0, :] += N_i
            edge_rh_shifted[1, :] += N_i

            edge_comb = torch.cat([edge_lh, edge_rh_shifted], dim=1).to(self.device)

            edge_index_per_stage[st] = (V_total, edge_comb)

        return edge_index_per_stage

    @staticmethod
    def _build_surf_edge_index_from_pial(pial_file: str) -> Tuple[int, torch.Tensor]:
        coords, faces = read_geometry(pial_file)
        V = coords.shape[0]
        i = faces[:, 0]
        j = faces[:, 1]
        k = faces[:, 2]
        src = np.concatenate([i, j, k, j, k, i], axis=0)
        dst = np.concatenate([j, k, i, i, j, k], axis=0)
        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        return V, edge_index

    def forward(self, subject_ids: List[str]):
        # Only use stage1..stage7 keys
        stage_keys = sorted(
            self.edge_index_per_stage.keys(),
            key=lambda k: int(k.replace("stage", ""))
        )
        num_used_stages = len(stage_keys)  # expected = 7

        # Prepare container: one list of Data per used stage
        graph_list_per_stage: List[List[Data]] = [[] for _ in range(num_used_stages)]

        for subject_id in subject_ids:
            # Step 1: ensure MELD features exist for this subject
            # features_dir = os.path.join(self.feature_path, "input", "ds004199", subject_id, "anat", "features")
            features_dir = os.path.join(self.feature_path, "preprocessed", subject_id, "features")
            npz_path = os.path.join(features_dir, "feature_maps.npz")
            if os.path.isfile(npz_path):
                features = np.load(npz_path)
            else:
                raise FileNotFoundError(f"Failed to generate NPZ for {subject_id}")

            # Step 2: load subject’s NPZ
            
            sorted_keys = sorted(features.files, key=lambda k: int(k.replace("stage", "")))
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
                data = Data(x=feat_tensor, 
                            edge_index=edge_index, 
                            num_nodes=H * N)

                if N < 642:
                    data['gnn_x'] = data.x
                else:
                    data['gnn_x'] = self.gnn_layers[i](data.x, data.edge_index)

                graph_list_per_stage[i].append(data)

        # Batch each stage’s list of Data
        batched_per_stage: List[Batch] = []
        for i in range(num_used_stages):
            batch_i = Batch.from_data_list(graph_list_per_stage[i]).to(self.device)
            batched_per_stage.append(batch_i)

        return {"feature": batched_per_stage, "project": None}