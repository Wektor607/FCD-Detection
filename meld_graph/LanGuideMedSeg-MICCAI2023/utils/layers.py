import sys
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GraphNorm
from performer_pytorch import SelfAttention

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

        self.in_channels      = in_channels
        self.chunk_size       = chunk_size 
        self.self_attn_norm   = nn.LayerNorm(in_channels)
        self.cross_attn_norm  = nn.LayerNorm(in_channels)

        # self.self_attn_lin = SelfAttention(
        #     dim=in_channels,      
        #     heads=1,              
        #     causal=False          
        # )
        self.self_attn  = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        # self.text_project = nn.Sequential(
        #     nn.Linear(embed_dim, 2 * in_channels),
        #     nn.LayerNorm(2 * in_channels),
        #     nn.GELU(),
        #     nn.Linear(2 * in_channels, in_channels),
        #     nn.LayerNorm(in_channels),
        #     nn.GELU(),
        # )

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos  = PositionalEncoding(in_channels)
        self.txt_pos  = PositionalEncoding(in_channels,max_len=output_text_len)

        self.norm1    = nn.LayerNorm(in_channels)
        self.norm2    = nn.LayerNorm(in_channels)
        self.txt_norm = nn.LayerNorm(in_channels)
        
        self.scale = nn.Parameter(torch.tensor(1.421),requires_grad=True)
        # self.scale = nn.Parameter(torch.tensor(0.5),requires_grad=True)


    def _chunked_attention(self, q, k, v, attn_module, chunk_size=4096):
        chunks = zip(q.split(chunk_size, dim=1),
                    k.split(chunk_size, dim=1),
                    v.split(chunk_size, dim=1))
        out_chunks = [attn_module(qc, kc, value=vc)[0] for qc, kc, vc in chunks]
        return torch.cat(out_chunks, dim=1)

    def _adaptive_chunked_attn(self, q, k, v, attn_module,
                               chunk=16384, min_chunk=512):
        """
        Пробуем разбивать по init_chunk, и если всё ещё OOM, 
        каждый раз делим на 2, пока не упадём ниже min_chunk.
        """
        while True:
            try:
                return self._chunked_attention(
                    q=q, k=k, v=v,
                    attn_module=attn_module,
                    chunk_size=chunk
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and chunk > min_chunk:
                    torch.cuda.empty_cache()
                    chunk = chunk // 2
                    print(f"[DEBUG] OOM in chunked_attn, reducing chunk to {chunk}")
                    continue
                # если либо другая ошибка, либо уже слишком мало – пробрасываем
                raise
    
    def forward(self, x, txt, chunk):
        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)

        if chunk:
            vis2 = self._adaptive_chunked_attn(  # [B, N_vis, C]
                q = q,
                k = k,
                v = vis2,
                attn_module=self.self_attn)
        else:
            vis2, _ = self.self_attn(q, k, value=vis2)

        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2
        
        if txt is not None:
            # Cross-Attention
            vis2 = self.norm2(vis)
            
            txt  = self.text_project(txt)
            
            # txt = self.txt_norm(txt) # <- a little bit lower accuracy if we use text normalization
            
            vis2, _ = self.cross_attn(
                        query=self.vis_pos(vis2),
                        key=self.txt_pos(txt),
                        value=txt)
            
            vis2 = self.cross_attn_norm(vis2)
                        
            # alpha = F.softplus(self.scale) 
            # vis = alpha * vis + (1 - alpha) * vis2
            vis = vis + self.scale * vis2
            # print(f"[DEBUG] α = {alpha.item():.4f}")

        return vis

class GuideDecoder(nn.Module):

    def __init__(self,in_channels, out_channels, text_len, input_text_len) -> None:

        super().__init__()
        
        self.guide_layer = GuideDecoderLayer(in_channels, text_len, input_text_len)   # for skip
        
        total_dim = in_channels + out_channels
        hidden = total_dim // 2
        self.hidden = hidden

        # MLP
        # self.ln_tot = nn.Linear(total_dim, out_channels)
        
        self.lin1    = nn.Linear(total_dim, hidden)
        self.gn1     = GraphNorm(hidden)
        self.lnorm1  = nn.LayerNorm(hidden)
        self.lin2    = nn.Linear(hidden, out_channels)
        self.gn2     = GraphNorm(out_channels)
        self.lnorm2  = nn.LayerNorm(out_channels)
        self.act     = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.BatchNorm1d(hidden), 
            nn.ReLU(), # Worse with GELU
            nn.Dropout(0.1),
            nn.Linear(hidden, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(), # Worse with GELU
            # nn.Dropout(0.1)            
        )

        self.res_proj = nn.Linear(total_dim, out_channels)
        # self.mlp = nn.Sequential(
        #     nn.Linear(total_dim, hidden),
        #     nn.GELU(),
        #     nn.Linear(hidden, out_channels),
        # )

    def forward(self, vis, skip_vis, txt, assign, chunk):
        B, N_in, C_in = vis.shape
        B2, N_skip, C_skip = skip_vis.shape
        assert B == B2, "Batch size mismatch между vis и skip_vis"

        vis_coarse = self.guide_layer(vis, txt, chunk)

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

        cat         = torch.cat([vis_upsampled, skip_vis], dim=-1)
        B, N, Ctot  = cat.shape
        cat_flat    = cat.view(B * N, Ctot)
        # out_flat    = self.mlp(cat_flat) # <- bad results

        batch = torch.arange(B, device=cat.device).repeat_interleave(N)

        x           = self.lin1(cat_flat)                # [B·N, hidden]
        x           = self.lnorm1(self.gn1(x, batch))                 # GraphNorm по каждому графу
        x           = self.act(x)
        x           = self.dropout(x)
        x           = self.lin2(x)                       # [B·N, out_dim]
        x           = self.lnorm2(self.gn2(x, batch))
        out_flat    = self.act(x)

        # MLP-1
        # x           = self.ln_tot(cat_flat)                # [B·N, hidden]
        # x           = self.lnorm2(self.gn2(x, batch))                 # GraphNorm по каждому графу
        # x           = self.act(x)
        # x           = self.dropout(x)
        # out_flat    = x.view(B, N, -1)                      # [B, N_max]

        # MLP-2 + Residual 
        out         = (self.res_proj(cat_flat) + out_flat).view(B, N, -1)                      # [B, N_max]
        # out         = out_flat.view(B, N, -1)
        
        # With out MLP
        # out = self.res_proj(cat_flat).view(B, N, -1)                      # [B, N_max]

        return out
    