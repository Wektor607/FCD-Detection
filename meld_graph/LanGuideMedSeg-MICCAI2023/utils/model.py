import os
import sys
from typing import List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from utils.layers import GuideDecoder

from typing import List
from torch_geometric.data import Data, Batch

from .vision_model import VisionModel
from .language_model import BERTModel

class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, 
                 meld_script_path,
                 feature_path, 
                 output_dir,
                 project_dim=512,
                 device='cpu',
                 warmup_epochs=0):

        super(LanGuideMedSeg, self).__init__()

        # Layer stage1 — shape: torch.Size([5, 2, 163842, 32])
        # Layer stage2 — shape: torch.Size([5, 2, 40962, 32])
        # Layer stage3 — shape: torch.Size([5, 2, 10242, 64])
        # Layer stage4 — shape: torch.Size([5, 2, 2562, 64])
        # Layer stage5 — shape: torch.Size([5, 2, 642, 128])
        # Layer stage6 — shape: torch.Size([5, 2, 162, 128]) SKIP
        # Layer stage7 — shape: torch.Size([5, 2, 42, 256])  SKIP
        
        feature_dim         = [32, 32, 64, 64, 128] #, 128, 256] # stage_i[2]
        text_lens           = [256, 256, 128, 128, 64]  #, 256]

        self.num_stages = len(feature_dim)
        self.warmup_epochs = warmup_epochs
        self.encoder = VisionModel(feature_dim, meld_script_path,
                                   feature_path, output_dir, device)
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

        final_in = feature_dim[0]
        self.final_lin = nn.Linear(final_in, 1)
        self.to(device)
    
    def forward(self, data, current_epoch):

        subject_ids, text = data
        B = len(subject_ids)

        graph_output                = self.encoder(subject_ids)
        graph_features: List[Batch] = graph_output['feature']
        text_output                 = self.text_encoder(text['input_ids'],
                                                        text['attention_mask'])
        text_hidden_last            = text_output['feature'][-1]

        # 3) Будем поочерёдно «подниматься» от глубокой стадии к мелкой
        #    current_graphs — список из B объектов Data (каждый Data для одной стадии)
        current_graphs = graph_features[-1].to_data_list()
        # Идём по декодерам: первый декодер берёт (stageN → stageN-1), 
        # второй — (stageN-1 → stageN-2), и т. д.
        for idx, decoder in enumerate(self.decoders):
            # Определим стадию, в которую «поднимаемся». idx=0 → i=num_stages-1 → «stageN→stageN-1»
            stage_from = self.num_stages - 1 - idx    # номер стадии (0-based) откуда
            stage_to = stage_from - 1                 # куда «поднимаемся»

            # Подготовим «пропускаемые» узлы (skip connections) аналогично:
            next_graphs     = graph_features[stage_to].to_data_list()            
            
            updated_graphs: List[Data] = []
            for j in range(B):
                vis_feat  = current_graphs[j].gnn_x.unsqueeze(0)   # [1, N_from, C_from]
                skip_feat = next_graphs[j].x.unsqueeze(0)     # [1, N_to, C_to]
                if current_epoch >= self.warmup_epochs:
                    txt_emb = text_hidden_last[j].unsqueeze(0)    # [1, L_seq, 768]
                else:
                    txt_emb = None

                N_from = vis_feat.size(1)    # число «грубых» вершин, откуда мы «поднимаемся»
                N_to   = skip_feat.size(1)   # число «четких» вершин, куда ведет skip

                # Будем считать, что N_to / N_from ≈ целое число, например 4.
                factor = int(round(N_to / N_from))

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
                    gnn_x=out_feat.squeeze(0),
                    edge_index=next_graphs[j].edge_index,
                    num_nodes=out_feat.size(1)
                )
                updated_graphs.append(new_data)

            # После цикла обновляем current_graphs на «модельные» данные stage_to
            current_graphs = updated_graphs

        # 4) Теперь current_graphs = список B графов для самой «мелкой» стадии (stage1)
        logits_list: List[torch.Tensor] = []
        for g in current_graphs:
            logit = self.final_lin(g.x)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=0)
        return logits  # List длины B: [Ni, 1] для каждого субъекта
    
