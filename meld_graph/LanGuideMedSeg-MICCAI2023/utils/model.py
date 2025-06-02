import os
import sys
from typing import List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from utils.layers import GuideDecoder
from torch_geometric.nn import GraphConv
from transformers import AutoTokenizer, AutoModel

from typing import List, Tuple, Optional

# Библиотека для графов
from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch


class BERTModel(nn.Module):

    def __init__(self, bert_type, project_dim):

        super(BERTModel, self).__init__()

        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        # freeze the parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        # get 1+2+last layer
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
        embed = self.project_head(embed)

        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):
    def __init__(self, project_dim: int, meld_script_path: str, feature_path: str, output_dir: str, device: str, feature_dim: list):
        super().__init__()
        

        self.meld_script_path = meld_script_path
        self.feature_path     = feature_path
        self.output_dir       = output_dir
        self.feature_dim      = feature_dim
        # Добавляем по одному GraphConv для каждой стадии
        self.gnn_layers = nn.ModuleList()
        for feat_dim in self.feature_dim:
            self.gnn_layers.append(GraphConv(feat_dim, feat_dim))


        # Кэшируем edge_index для каждой стадии, чтобы не пересоздавать их каждый шаг
        self.edge_index_cache: List[Optional[torch.Tensor]] = [None] * len(self.feature_dim)

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')


    def run_meld_prediction(self, subject_id):
        command = [
            self.meld_script_path,
            "run_script_prediction.py",
            "-id", subject_id,
            "-harmo_code", "fcd",
            "-demos", "participants_with_scanner.tsv"
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running MELD prediction: {e}")
            raise
    
    def build_edge_index(self, num_nodes: int, stage_idx: int) -> torch.Tensor:
        """
        Строим edge_index для «линейного» графа длины num_nodes:
        ребро между i и i+1 (в обе стороны). Кэшируем по stage_idx.
        """
        if self.edge_index_cache[stage_idx] is not None:
            return self.edge_index_cache[stage_idx]
        
        # Если узлов меньше 2, просто пустой граф
        if num_nodes < 2:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.edge_index_cache[stage_idx] = edge_index
            return edge_index
        
        # Создаём ребра i→i+1 и i+1→i
        src = torch.arange(0, num_nodes - 1, device=self.device, dtype=torch.long)
        dst = torch.arange(1, num_nodes, device=self.device, dtype=torch.long)
        # [0,1,2,...,N−2] → [1,2,3,...,N−1]
        e1 = torch.stack([src, dst], dim=0)
        e2 = torch.stack([dst, src], dim=0)
        edge_index = torch.cat([e1, e2], dim=1)  # форма [2, 2*(N−1)]
        self.edge_index_cache[stage_idx] = edge_index
        return edge_index
    
    def forward(self, subject_ids:List[str]):
        num_stages = len(self.feature_dim) # Fix it later
        
        # Заготовим контейнер: для каждой стадии i ‒ свой список графов длины B
        graph_list_per_stage: List[List[Data]] = [[] for _ in range(num_stages)]

        for subject_id in subject_ids:

            # Step 1: run MELD prediction
            if not os.path.isfile(os.path.join(self.output_dir, "predictions_reports", f"{subject_id}", "predictions/prediction.nii.gz")):
                self.run_meld_prediction(subject_id)

            # Step 2: get features from all model layers
            features_path = os.path.join(self.feature_path, "input", subject_id, "anat", "features", "feature_maps.npz")
            features      = np.load(features_path)
            sorted_keys   = sorted(features.files, key=lambda k: int(k.replace('stage', '')))
            
            # Step 3: We squeeze and project the embeddings to the same dimension
            for i, stage in enumerate(sorted_keys):
                feat_np = features[stage]    # shape (1, Ni, Ci)
                feat    = torch.from_numpy(feat_np).to(self.device).squeeze(0)  # [Ni, Ci]
                Ni, Ci  = feat.shape

                edge_index = self.build_edge_index(Ni, stage_idx=i)  # [2, 2*(Ni−1)]

                data       = Data(x=feat, edge_index=edge_index, num_nodes=Ni)
                data.x     = self.gnn_layers[i](data.x, data.edge_index)
                graph_list_per_stage[i].append(data)

        # Теперь для каждой стадии i склеим List[Data] → один Batch
        batched_per_stage: List[Batch] = []
        for i in range(num_stages):
            batch_i = Batch.from_data_list(graph_list_per_stage[i]).to(self.device)
            batched_per_stage.append(batch_i)
    
        return {"feature": batched_per_stage, "project": None}

class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, 
                 meld_script_path, 
                 feature_path, 
                 output_dir,
                 project_dim=512,
                 device='cpu'):

        super(LanGuideMedSeg, self).__init__()

        # Layer stage1 — shape: torch.Size([1, 163842, 32])
        # Layer stage2 — shape: torch.Size([1, 40962, 32])
        # Layer stage3 — shape: torch.Size([1, 10242, 64])
        # Layer stage4 — shape: torch.Size([1, 2562, 64])
        # Layer stage5 — shape: torch.Size([1, 642, 128])
        # Layer stage6 — shape: torch.Size([1, 162, 128])
        # Layer stage7 — shape: torch.Size([1, 42, 256])
        
        feature_dim         = [32, 32, 64, 64, 128, 128, 256] # stage_i[2]
        text_lens           = [256, 256, 256, 256, 256, 256]
        self.spatial_dim    = [
            (1, 1, 163842),
            (1, 1, 40962),
            (1, 1, 10242),
            (1, 1, 2562),
            (1, 1, 642),
            (1, 1, 162),
            (1, 1, 42),
        ]       

        self.num_stages = len(self.spatial_dim)

        self.encoder = VisionModel(project_dim, meld_script_path, 
                                   feature_path, output_dir, device, 
                                   feature_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.decoders = nn.ModuleList()
        for i in range(self.num_stages-1, 0, -1):
            in_channels   = feature_dim[i]
            skip_channels = feature_dim[i-1]
            text_len      = text_lens[i-1]

            decoder       = GuideDecoder(in_channels  = in_channels, 
                                         out_channels = skip_channels,
                                         text_len     = text_len)
            self.decoders.append(decoder)

        # 4) Финальный линейный слой: превращаем [Ni, C_skip] → [Ni, 1] (логит) на мелкой стадии
        final_in = feature_dim[0]  # C_skip для самой мелкой стадии (stage1)
        self.final_lin = nn.Linear(final_in, 1)
        self.to(device)
    
    def forward(self, data):

        subject_ids, text = data
        B = len(subject_ids)

        graph_output                = self.encoder(subject_ids)
        graph_features: List[Batch] = graph_output['feature']
        text_output                 = self.text_encoder(text['input_ids'],
                                                        text['attention_mask'])
        text_hidden_last            = text_output['feature'][-1]

        # 3) Будем поочерёдно «подниматься» от глубокой стадии к мелкой
        #    current_graphs — список из B объектов Data (каждый Data для одной стадии)
        current_graphs = []
        # Достаём из Batch(stage=6) по одному Data
        for b in range(B):
            current_graphs.append(
                Data(
                    x=graph_features[-1].x[graph_features[-1].batch == b],
                    edge_index=graph_features[-1].edge_index
                )
            )
        ###################################################
        # У current_graphs[j].x форма [N6, C6]

        # Идём по декодерам: первый декодер берёт (stageN → stageN-1), 
        # второй — (stageN-1 → stageN-2), и т. д.
        for idx, decoder in enumerate(self.decoders):
            # Определим стадию, в которую «поднимаемся». idx=0 → i=num_stages-1 → «stageN→stageN-1»
            stage_from = self.num_stages - 1 - idx    # номер стадии (0-based) откуда
            stage_to = stage_from - 1                 # куда «поднимаемся»

            # Подготовим «пропускаемые» узлы (skip connections) аналогично:
            next_graphs = []
            for b in range(B):
                next_graphs.append(
                    Data(
                        x=graph_features[stage_to].x[graph_features[stage_to].batch == b],
                        edge_index=graph_features[stage_to].edge_index
                    )
                )
            # next_graphs[j].x имеет shape [N(stage_to), C(stage_to)]

            updated_graphs: List[Data] = []
            for j in range(B):
                vis_feat = current_graphs[j].x.unsqueeze(0)   # [1, N_from, C_from]
                skip_feat = next_graphs[j].x.unsqueeze(0)     # [1, N_to, C_to]
                txt_emb = text_hidden_last[j].unsqueeze(0)    # [1, L_seq, 768]

                N_from = vis_feat.size(1)    # число «грубых» вершин, откуда мы «поднимаемся»
                N_to   = skip_feat.size(1)   # число «четких» вершин, куда ведет skip

                # Будем считать, что N_to / N_from ≈ целое число, например 4.
                factor = np.round(N_to / N_from)

                # Тогда каждому индексу i в [0..N_to-1] мы сопоставим coarse-индекс:
                #   assign[i] = i // factor
                # Итого assign.shape == [N_to], а значения лежат в [0 .. N_from-1].
                assign = torch.arange(N_to, device=vis_feat.device) // factor
                assign = assign.to(dtype=torch.int64)
                
                # Запускаем decoder
                out_feat = decoder(vis_feat, skip_feat, txt_emb, assign)  # [1, N_out, C_to]

                # Сохраняем обратно в Data: обновляем x у графа стадии stage_to
                new_data = Data(
                    x=out_feat.squeeze(0),         # [N_out, C_to]
                    edge_index=next_graphs[j].edge_index,
                    num_nodes=out_feat.size(1)
                )
                updated_graphs.append(new_data)

            # После цикла обновляем current_graphs на «модельные» данные stage_to
            current_graphs = updated_graphs

        # 4) Теперь current_graphs = список B графов для самой «мелкой» стадии (stage1)
        #    Каждый из них хранит x: [N1, C1]. Наша задача — выдать узловой логит
        logits_list: List[torch.Tensor] = []
        for j in range(B):
            node_feats = current_graphs[j].x   # [N1, C1]
            logit = self.final_lin(node_feats) # [N1, 1]
            logits_list.append(logit.squeeze(-1))

        logits = torch.stack(logits_list, dim=0)
        return logits  # List длины B: [Ni, 1] для каждого субъекта
    
