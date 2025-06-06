import sys
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=2000000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]



class GuideDecoderLayer(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=256, embed_dim:int=768, chunk_size:int=4096):

        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels
        self.chunk_size  = chunk_size 
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(1.421),requires_grad=True)

    def _chunked_attention(self, q, k, v, attn_module, chunk_size=4096):
        chunks = zip(q.split(chunk_size, dim=1),
                    k.split(chunk_size, dim=1),
                    v.split(chunk_size, dim=1))
        out_chunks = [attn_module(qc, kc, value=vc)[0] for qc, kc, vc in chunks]
        return torch.cat(out_chunks, dim=1)

    # def _chunked_cross_attention(self, vis, txt, attn_module, vis_pos, txt_pos, chunk_size=4096):
    #     chunks = vis.split(chunk_size, dim=1)
    #     out_chunks = []
    #     for chunk in chunks:
    #         norm_chunk = self.norm2(chunk)
    #         out_chunk, _ = attn_module(query=vis_pos(norm_chunk), key=txt_pos(txt), value=txt)
    #         out_chunks.append(out_chunk)
    #     return torch.cat(out_chunks, dim=1)

    def forward(self, x, txt, chunk_threshold=100000):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        txt = self.text_project(txt)
        
        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        # vis2, _ = self.self_attn(q, k, value=vis2)
        vis2 = self._chunked_attention(  # [B, N_vis, C]
            q = q,
            k = k,
            v = vis2,
            attn_module = self.self_attn,
            chunk_size = 16384
        )
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2
        
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2,_ = self.cross_attn(
                    query=self.vis_pos(vis2),
                    key=self.txt_pos(txt),
                    value=txt)
        
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis

class GuideDecoder(nn.Module):

    def __init__(self,in_channels, out_channels, text_len) -> None:

        super().__init__()

        self.guide_layer = GuideDecoderLayer(in_channels, text_len)   # for skip
        # 2. После этого «склеиваем» признаки vis + skip_vis и делаем простой Linear→BatchNorm→ReLU
        self.lin1 = nn.Linear(in_channels + out_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)


    def forward(self, vis, skip_vis, txt, assign):
        B, N_in, C_in = vis.shape
        B2, N_skip, C_skip = skip_vis.shape
        assert B == B2, "Batch size mismatch между vis и skip_vis"


        if txt is not None:
            vis_coarse = self.guide_layer(vis, txt)
        
        # 2) Graph Unpooling: «разворачиваем» coarse→fine через assign
        # assign: [N_fine], в диапазоне [0..N_coarse−1]
        # сделаем gather: из vis_coarse2 по dim=1
        #  a) расширяем assign на батч
        assign_expand = assign.unsqueeze(0).expand(B, -1)            # [B, N_fine]
        assign_expand = assign_expand.unsqueeze(-1).expand(-1, -1, C_in)  # [B, N_fine, C_in]

        #  b) берём каждый fine-индекс i: parent_idx = assign[i],
        #     и доставляем vis_coarse2[b, parent_idx, :] в vis_upsampled[b, i, :].
        vis_upsampled = torch.gather(vis_coarse, dim=1, index=assign_expand)
        # → [B, N_fine, C_in]

        cat = torch.cat([vis_upsampled, skip_vis], dim=-1)
        B, N, Ctot = cat.shape

        out = self.lin1(cat.view(-1, Ctot))               # [B*N_max, out_channels]
        out = self.norm(out)                              # [B*N_max, out_channels]
        out = F.relu(out)
        out = out.view(B, N, -1)                      # [B, N_max, out_channels]

        return out
       


