import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import numpy as np
from utils.layers import GuideDecoder
from meld_graph.icospheres import IcoSpheres
from meld_graph.spiralconv import SpiralConv
from typing import List
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from .vision_model import VisionModel
from .language_model import BERTModel
from engine.pooling import HexUnpool


class LanGuideMedSeg(nn.Module):
    def __init__(
        self,
        bert_type,
        meld_script_path,
        feature_path,
        output_dir,
        project_dim=512,
        device="cpu",
        tokenizer=None,
        max_len=384,
    ):
        super(LanGuideMedSeg, self).__init__()

        # Layer stage1 — shape: torch.Size([bs, 2, 163842, 32])
        # Layer stage2 — shape: torch.Size([bs, 2, 40962, 32])
        # Layer stage3 — shape: torch.Size([bs, 2, 10242, 64])
        # Layer stage4 — shape: torch.Size([bs, 2, 2562, 64])
        # Layer stage5 — shape: torch.Size([bs, 2, 642, 128])
        # Layer stage6 — shape: torch.Size([bs, 2, 162, 128])
        # Layer stage7 — shape: torch.Size([bs, 2, 42, 256])

        feature_dim = [32, 32, 64, 64, 128, 128, 256]
        if max_len == 256:
            text_lens = [128, 64, 64, 32, 32, 16, 16]  # [256, 128, 128, 64, 64, 32, 32]
        else:
            text_lens = [384, 384, 256, 256, 128, 128, 64]

        self.num_stages = len(feature_dim)
        self.tokenizer = tokenizer
        self.encoder = VisionModel(
            feature_dim, meld_script_path, feature_path, output_dir, device
        )
        self.text_encoder = BERTModel(bert_type, project_dim, tokenizer=self.tokenizer)

        self.decoders = nn.ModuleList()
        skip_dims = []
        for i in range(self.num_stages - 1, 0, -1):
            in_channels = feature_dim[i]
            skip_channels = feature_dim[i - 1]
            skip_dims.append(skip_channels)
            text_len = text_lens[i - 1]

            decoder = GuideDecoder(
                in_channels=in_channels,
                out_channels=skip_channels,
                text_len=text_len,
                input_text_len=max_len,
            )
            self.decoders.append(decoder)
        
        # TODO: read from config
        layer_sizes = [
            [32, 32, 32],
            [32, 32, 32],
            [64, 64, 64],
            [64, 64, 64],
            [128, 128, 128],
            [128, 128, 128],
        ]

        ico_path = os.path.join("data", "icospheres")
        icos = IcoSpheres(icosphere_path=ico_path)
        self.unpool_layers = nn.ModuleList()
        self.decoder_conv_layers = nn.ModuleList()
        # TODO: parameters
        spiral_len, level = 7, 2  # make it automatically
        in_size = feature_dim[-1]
        for i in range(self.num_stages - 1):
            upsample = icos.get_upsample(target_level=level)
            num = len(icos.get_neighbours(level=level))

            self.unpool_layers.append(
                HexUnpool(upsample_indices=upsample, target_size=num)
            )

            # 2. SpiralConv
            icos.create_spirals(level=level)
            indices = icos.get_spirals(level=level)
            indices = indices[:, :spiral_len]

            block = []
            input_dim = in_size + skip_dims[i]

            for _, out_size in enumerate(layer_sizes[::-1][i]):
                conv = SpiralConv(input_dim, out_size, indices=indices)
                block.append(conv)
                input_dim = out_size

            self.decoder_conv_layers.append(nn.ModuleList(block))
            in_size = input_dim

            level += 1

        # ----------------------------
        # Deep Supervision heads
        # ----------------------------

        # The first value represents the proportion of lesion pixels across all scans,
        # and the second value represents the proportion of background pixels
        pos_bias = -3.0 # np.log(0.006917 / 0.993083)

        self.ds_heads = nn.ModuleDict()
        self.ds_dist_heads = nn.ModuleDict()
        self.ds_levels = []

        level = 2  # first level after upsampling
        for i in range(self.num_stages - 1):
            out_ch = skip_dims[i]  # C_to for stage_to
            head = nn.Linear(out_ch, 2)
            dist_head = nn.Linear(out_ch, 1)

            with torch.no_grad():
                head.bias[0].fill_(0.0)  # class "background"
                head.bias[1].fill_(pos_bias)  # class "lesion"
                dist_head.bias.fill_(0.0)

            self.ds_heads[str(level)] = head
            self.ds_dist_heads[str(level)] = dist_head
            self.ds_levels.append(level)
            level += 1

        # ----------------------------
        final_in = feature_dim[0]
        
        self.activation_function = nn.LeakyReLU()
        self.hemi_classification_head = nn.ModuleList([
            nn.Conv1d(final_in, 1, kernel_size=1),
            nn.Linear(2 * len(icos.icospheres[7]["coords"]), 2)
        ])

        self.final_lin_one_class = nn.Linear(final_in, 1)
        self.final_lin = nn.Linear(final_in, 2)
        self.dist_lin = nn.Linear(final_in, 1)
        with torch.no_grad():
            self.final_lin.bias[0].fill_(0.0)
            self.final_lin.bias[1].fill_(pos_bias)
            self.dist_lin.bias.fill_(0.0)

            self.final_lin_one_class.bias.fill_(pos_bias)

    def forward(self, data):
        subject_ids, text = data
        B = len(subject_ids)

        graph_output                = self.encoder(subject_ids)
        graph_features: List[Batch] = graph_output["feature"]
        text_output                 = self.text_encoder(text["input_ids"], text["attention_mask"])
        text_hidden_last            = text_output["feature"]  # [B, L_seq, 768]

        outputs = {"log_softmax": [], "non_lesion_logits": [], "log_sumexp": []}

        # prepare per-level container
        for lvl in self.ds_levels:
            outputs[f"ds{lvl}_log_softmax"]         = []
            outputs[f"ds{lvl}_non_lesion_logits"]   = []

        # Climb from deep to shallow stages (list of B Data objects per stage)
        current_graphs  = graph_features[-1].to_data_list()

        # Each decoder maps stage N → N-1, then N-1 → N-2, etc.
        for idx, decoder in enumerate(self.decoders):
            stage_from  = self.num_stages - 1 - idx
            stage_to    = stage_from - 1

            unpool      = self.unpool_layers[idx]
            spiral_conv = self.decoder_conv_layers[idx]

            next_graphs = graph_features[stage_to].to_data_list()

            ds_logp_level               = []
            ds_dist_level               = []
            updated_graphs: List[Data]  = []
            cur_level = self.ds_levels[
                idx
            ]  # level for Deep Supervision head at this stage
            for j in range(B):
                vis_feat = current_graphs[j].x.unsqueeze(0)  # [1, N_from, C_from]
                skip_feat = next_graphs[j].x.unsqueeze(0)  # [1, N_to, C_to]

                N_from = vis_feat.size(1)

                # TODO: hyperparameter
                if N_from < 40962:
                    txt_emb = text_hidden_last[j].unsqueeze(0)  # [1, L_seq, 768]
                else:
                    txt_emb = None

                # TODO: hyperparameter
                chunk = N_from > 40962
                out_feat = decoder(
                    vis_feat, skip_feat, txt_emb, unpool, spiral_conv, chunk
                )  # [1, N_out, C_to]

                x_lvl = out_feat.squeeze(0)  # [N_to, C_to]

                # --- Deep Supervision head for current level ---
                if str(cur_level) in self.ds_heads:  # Linear(C_to -> 2)
                    head = self.ds_heads[str(cur_level)]
                    assert head.in_features == x_lvl.size(1), (
                        f"Level {cur_level}: head expects {head.in_features}, got {x_lvl.size(1)}"
                    )

                    logits = head(x_lvl)  # [N_to, 2]
                    logp = nn.LogSoftmax(dim=1)(logits)  # [N_to, 2]
                    ds_logp_level.append(logp)

                    dist = self.ds_dist_heads[str(cur_level)](x_lvl)  # [N_to, 1]
                    ds_dist_level.append(dist)

                # Save back to Data: update x in stage_to stage graph
                new_data = Data(
                    x=out_feat.squeeze(0),  # [N_out, C_to]
                    edge_index=next_graphs[j].edge_index,
                    num_nodes=out_feat.size(1),
                )
                updated_graphs.append(new_data)

            # stack batch outputs for this level
            if ds_logp_level:
                outputs[f"ds{cur_level}_log_softmax"] = torch.cat(
                    ds_logp_level, dim=0
                )  # [B*H*V_level, 2]
                outputs[f"ds{cur_level}_non_lesion_logits"] = torch.cat(
                    ds_dist_level, dim=0
                )  # [B*H*V_level, 1]
            current_graphs = updated_graphs

        # 4) Final level
        final_logp_list     = []
        final_classification   = []
        final_dist_list     = []

        for g in current_graphs:
            seg_logits      = self.final_lin(g.x)  # [N1, 2]
            log_seg_logits  = nn.LogSoftmax(dim=1)(seg_logits)
            final_logp_list.append(log_seg_logits)  # [N1, 2]
            
            hemi_classification = self.activation_function(self.hemi_classification_head[0](g.x.unsqueeze(2)))
            hemi_classification = self.hemi_classification_head[1](hemi_classification.view(-1))
            hemi_classification = nn.LogSoftmax(dim=0)(hemi_classification)
            final_classification.append(hemi_classification)
            # distance head
            if hasattr(self, "dist_lin"):
                dist_logits = self.dist_lin(g.x)  # [N1, 1]
                final_dist_list.append(dist_logits.squeeze(-1))  # [N1]

        outputs["log_softmax"]          = torch.cat(final_logp_list, dim=0)  # [B*H*V1, 2]
        outputs["non_lesion_logits"]    = torch.cat(final_dist_list, dim=0)  # [B*H*V1]
        outputs["hemi_log_softmax"]     = torch.cat(final_classification,dim=0)
        return outputs
        # logits_list: List[torch.Tensor] = []
        # for g in current_graphs:
        #     logit = self.final_lin_one_class(g.x)
        #     logits_list.append(logit)

        # logits = torch.stack(logits_list, dim=0)

        # return logits, outputs
