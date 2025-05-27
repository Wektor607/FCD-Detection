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

    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=256, embed_dim:int=768):

        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=2,batch_first=True)
        # self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        print(input_text_len,output_text_len, embed_dim,in_channels)
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


    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        
        txt = self.text_project(txt)
        
        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        
        q = torch.as_tensor(q)
        k = torch.as_tensor(k)
        vis2 = torch.as_tensor(vis2)

        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2
        
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2,_ = self.cross_attn(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis

class GuideDecoder(nn.Module):

    def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:

        super().__init__()

        self.guide_layer = GuideDecoderLayer(in_channels, text_len)   # for skip
        self.D, self.H, self.W = spatial_size
        self.decoder = UnetrUpBlock(3,in_channels,out_channels,3,2,norm_name='BATCH')

    def pad_to_length(self, tensor, target_len):
        """
        tensor: [B, N, C]
        target_len: int
        """
        B, N, C = tensor.shape
        if N >= target_len:
            return tensor[:, :target_len, :]
        pad_len = target_len - N
        pad = torch.zeros(B, pad_len, C, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)

    def forward(self, vis, skip_vis, txt):

        if txt is not None:
            vis = self.guide_layer(vis, txt)

        vis = rearrange(vis, 'B (D H W) C -> B C D H W', D=self.D, H=self.H, W=self.W)

        target_len = (self.D * 2) * (self.H * 2) * (self.W * 2)
        skip_vis = self.pad_to_length(skip_vis, target_len)

        skip_vis = rearrange(skip_vis, 'B (D H W) C -> B C D H W', D=self.D * 2, H=self.H * 2, W=self.W * 2)
        print(vis.shape)
        print(skip_vis.shape)
        print()
        output = self.decoder(vis, skip_vis)

        output = rearrange(output, 'B C D H W -> B (D H W) C')

        return output
    # def forward(self, vis, skip_vis, txt):

    #     if txt is not None:
    #         vis =  self.guide_layer(vis, txt)
        
    #     vis = rearrange(vis,'B (H W) C -> B C H W',H=self.H,W=self.W)

    #     target_len = (self.H * 2) * (self.W * 2)
    #     skip_vis = self.pad_to_length(skip_vis, target_len)

    #     skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.H * 2, W=self.W * 2)
    #     output = self.decoder(vis,skip_vis)

    #     output = rearrange(output,'B C H W -> B (H W) C')

    #     return output


