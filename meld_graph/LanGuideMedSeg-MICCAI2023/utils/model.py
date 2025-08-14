import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import numpy as np
from utils.layers import GuideDecoder
from meld_graph.icospheres import IcoSpheres
from meld_graph.spiralconv import SpiralConv
from typing import List
from torch_geometric.data import Data, Batch

from .vision_model import VisionModel
from .language_model import BERTModel

class HexUnpool(nn.Module):
    """
    Mean unpooling for Icospheres.
    """

    def __init__(self, upsample_indices, target_size):
        super(HexUnpool, self).__init__()
        self.upsample_indices = upsample_indices
        self.target_size = target_size

    def forward(self, x, device):
        # print(x.shape)
        B, H, N_from, C = x.shape

        # new_x: [B, H, target_size, C]
        new_x = torch.zeros(B, H, self.target_size, C, device=device)
        # print(new_x.shape)

        # 1) копируем старые фичи
        #    по всем батчам и полушариям: берем первые N_from вершин
        new_x[:, :, :N_from, :] = x

        # 2) считаем усреднённые фичи для новых вершин
        #    x[:,:, self.upsample_indices, :] → [B, H, N_new, 2, C]
        upsampled = x[:, :, self.upsample_indices, :].mean(dim=3)

        # 3) вставляем их в new_x после оригинальных N_from позиций
        new_x[:, :, N_from:, :] = upsampled
        
        new_x = new_x.view(B, H*self.target_size, C)
        return new_x
    
class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, 
                 meld_script_path,
                 feature_path, 
                 output_dir,
                 project_dim=512,
                 device='cpu',
                 tokenizer=None,
                 max_len=384):

        super(LanGuideMedSeg, self).__init__()

        # Layer stage1 — shape: torch.Size([5, 2, 163842, 32])
        # Layer stage2 — shape: torch.Size([5, 2, 40962, 32])
        # Layer stage3 — shape: torch.Size([5, 2, 10242, 64])
        # Layer stage4 — shape: torch.Size([5, 2, 2562, 64])
        # Layer stage5 — shape: torch.Size([5, 2, 642, 128])
        # Layer stage6 — shape: torch.Size([5, 2, 162, 128]) 
        # Layer stage7 — shape: torch.Size([5, 2, 42, 256])

        feature_dim             = [32, 32, 64, 64, 128, 128, 256]
        if max_len == 256:
            text_lens           = [128, 64, 64, 32, 32, 16, 16] # [256, 128, 128, 64, 64, 32, 32]
        else:
            text_lens           = [384, 384, 256, 256, 128, 128, 64] 

        self.num_stages = len(feature_dim)
        self.tokenizer = tokenizer
        self.encoder = VisionModel(feature_dim, meld_script_path,
                                   feature_path, output_dir, device)
        self.text_encoder = BERTModel(bert_type, project_dim, tokenizer=self.tokenizer)

        self.decoders = nn.ModuleList()
        skip_dims = []
        for i in range(self.num_stages-1, 0, -1):
            in_channels   = feature_dim[i]
            skip_channels = feature_dim[i-1]
            skip_dims.append(skip_channels)
            text_len      = text_lens[i-1]

            decoder       = GuideDecoder(in_channels    = in_channels, 
                                         out_channels   = skip_channels,
                                         text_len       = text_len,
                                         input_text_len = max_len)
            self.decoders.append(decoder)

        ico_path = os.path.join('data', 'icospheres')
        icos = IcoSpheres(icosphere_path=ico_path)
            
        # 151550
        layer_sizes = [
            [32,32,32],
            [32,32,32],
            [64,64,64],
            [64,64,64],  
            [128,128,128], 
            # [128,128,128], 
            [256,256,128],
            ]

        # 151551
        # layer_sizes = [
        #     [64, 32, 32],       # level 2 → 1
        #     [96, 64, 32],       # level 3 → 2
        #     [128, 64, 64],      # level 4 → 3
        #     [192, 128, 64],     # level 5 → 4
        #     [256, 128, 128],    # level 6 → 5
        #     [384, 256, 128],    # level 7 → 6
        # ]
        
        # 151507
        # layer_sizes = [ # <- works on GPU
        #     [32, 32],        # вместо [32,32,32]
        #     [64, 32],        # ...
        #     [64, 64],
        #     [128, 64],
        #     [128, 128],
        #     [256, 128],
        # ]

        self.unpool_layers = nn.ModuleList()
        self.decoder_conv_layers = nn.ModuleList()
        spiral_len, level = 7, 2 # make it automatically
        in_size = feature_dim[-1]
        for i in range(self.num_stages - 1):
            upsample = icos.get_upsample(target_level=level)
            num = len(icos.get_neighbours(level=level)) # we multipy on number of hemispheres
            # print(upsample.shape, num)
            self.unpool_layers.append(HexUnpool(upsample_indices=upsample, target_size=num))

            # 2. SpiralConv
            icos.create_spirals(level=level)
            indices = icos.get_spirals(level=level)
            indices = indices[:, :spiral_len]

            block = []
            input_dim = in_size + skip_dims[i]
            print(input_dim)
            for _, out_size in enumerate(layer_sizes[::-1][i]):
                conv = SpiralConv(input_dim, out_size, indices=indices)
                block.append(conv)
                input_dim = out_size
            
            self.decoder_conv_layers.append(nn.ModuleList(block))
            in_size = input_dim

            level+=1
        
        final_in = feature_dim[0]
        self.final_lin = nn.Linear(final_in, 1)
        # self.final_lin = nn.Linear(final_in, 2)
        # self.dist_lin = nn.Linear(final_in, 1)
        nn.init.constant_(self.final_lin.bias, 1.0)#-4.0)
    
    def forward(self, data):

        subject_ids, text = data
        B = len(subject_ids)

        graph_output                = self.encoder(subject_ids)
        graph_features: List[Batch] = graph_output['feature']
        text_output                 = self.text_encoder(text['input_ids'],
                                                        text['attention_mask'])
        text_hidden_last            = text_output['feature']
        # text_hidden_last            = text_output['feature'][-1]
        
        # prepare per-level container
        ds_levels = len(self.decoders)
        ds_logits: List[List[torch.Tensor]] = [[] for _ in range(ds_levels)]

        # 3) Будем поочерёдно «подниматься» от глубокой стадии к мелкой
        #    current_graphs — список из B объектов Data (каждый Data для одной стадии)
        current_graphs = graph_features[-1].to_data_list()
        # Идём по декодерам: первый декодер берёт (stageN → stageN-1), 
        # второй — (stageN-1 → stageN-2), и т. д.
        for idx, decoder in enumerate(self.decoders):
            # Определим стадию, в которую «поднимаемся». idx=0 → i=num_stages-1 → «stageN→stageN-1»
            stage_from  = self.num_stages - 1 - idx    # номер стадии (0-based) откуда
            stage_to    = stage_from - 1                 # куда «поднимаемся»
            
            unpool      = self.unpool_layers[idx]
            spiral_conv = self.decoder_conv_layers[idx]
 
            next_graphs = graph_features[stage_to].to_data_list()            
            
            updated_graphs: List[Data] = []
            for j in range(B):
                vis_feat  = current_graphs[j].x.unsqueeze(0)   # [1, N_from, C_from]
                skip_feat = next_graphs[j].x.unsqueeze(0)     # [1, N_to, C_to]

                N_from    = vis_feat.size(1)    # число «грубых» вершин, откуда мы «поднимаемся»
                
                # if stage_to > 1:                    
                if N_from < 40962:
                    txt_emb = text_hidden_last[j].unsqueeze(0)    # [1, L_seq, 768]
                else:
                    txt_emb = None

                chunk = (N_from > 40962)
                out_feat = decoder(vis_feat, skip_feat, txt_emb, unpool, spiral_conv, chunk)  # [1, N_out, C_to]

                # Collect for this level j-th batch
                # ds_levels_logits = self.ds_heads[idx](out_feat.squeeze(0)).unsqueeze(0)  # [1, N_to, 1]
                # ds_logits[idx].append(ds_levels_logits)

                # Сохраняем обратно в Data: обновляем x у графа стадии stage_to
                new_data = Data(
                    x=out_feat.squeeze(0),         # [N_out, C_to]
                    # gnn_x=out_feat.squeeze(0),
                    edge_index=next_graphs[j].edge_index,
                    num_nodes=out_feat.size(1)
                )
                updated_graphs.append(new_data)

            # После цикла обновляем current_graphs на «модельные» данные stage_to
            # stack batch outputs for this level
            # ds_logits[idx] = torch.cat(ds_logits[idx], dim=0)  # [B, N_to,1]
            current_graphs = updated_graphs
        
        # outputs = {
        #     "log_softmax":       [],
        #     "non_lesion_logits": [] 
        # }
        # for g in current_graphs:
        #     seg_logits  = self.final_lin(g.x)
        #     dist_logits = self.dist_lin(g.x)
            
        #     outputs["log_softmax"].append(
        #         nn.LogSoftmax(dim=1)(seg_logits)  # [N,2]
        #     )
        #     outputs["non_lesion_logits"].append(
        #         dist_logits.squeeze(-1)                     # [N]
        #     )
            
        # outputs["log_softmax"] = torch.cat(outputs["log_softmax"], dim=0)       # [B*H*V, 2]
        # outputs["non_lesion_logits"] = torch.cat(outputs["non_lesion_logits"], dim=0)  # [B*H*V]

        # return outputs, None
        # 4) Теперь current_graphs = список B графов для самой «мелкой» стадии (stage1)
        logits_list: List[torch.Tensor] = []
        for g in current_graphs:
            logit = self.final_lin(g.x)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=0)
        return logits, ds_logits  # List длины B: [Ni, 1] для каждого субъекта
    
