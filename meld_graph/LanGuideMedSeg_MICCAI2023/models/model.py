from typing import List, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from pathlib import Path
import torch.nn as nn
from LanGuideMedSeg_MICCAI2023.models.layers import GuideDecoder
from meld_graph.icospheres import IcoSpheres
from LanGuideMedSeg_MICCAI2023.engine.pooling import HexPool, HexUnpool
from meld_graph.spiralconv import SpiralConv
from torch_geometric.data import Data, Batch

from .vision_model import VisionModel
from .language_model import BERTModel


class LanGuideMedSeg(nn.Module):
    def __init__(
        self,
        bert_type: str,
        meld_script_path: str,
        feature_path: str,
        output_dir: str,
        layer_sizes: List[List[int]],
        device: str,
        feature_dim: List[int],
        text_lens: List[int],
        max_len: int,
        gnn_min_verts: int,
        num_unfreeze_layers: int = 0,
        fold_number: int = 0,
        att_mechanism: bool = False,
        text_emb: bool = False,
    ) -> None:
        super().__init__()

        # Layer stage1 — shape: torch.Size([bs, 2, 163842, 32])
        # Layer stage2 — shape: torch.Size([bs, 2, 40962, 32])
        # Layer stage3 — shape: torch.Size([bs, 2, 10242, 64])
        # Layer stage4 — shape: torch.Size([bs, 2, 2562, 64])
        # Layer stage5 — shape: torch.Size([bs, 2, 642, 128])
        # Layer stage6 — shape: torch.Size([bs, 2, 162, 128])
        # Layer stage7 — shape: torch.Size([bs, 2, 42, 256])

        self.num_stages: int = len(feature_dim)
        self.text_emb = text_emb
        self.encoder = VisionModel(
            feature_dim, meld_script_path, feature_path, output_dir, device,
            gnn_min_verts=gnn_min_verts, fold_number=fold_number
        )
        self.text_encoder = BERTModel(bert_type, num_unfreeze_layers=num_unfreeze_layers)

        self.decoders = nn.ModuleList()
        skip_dims: List[int] = []
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
                att_mechanism=att_mechanism,
            )
            self.decoders.append(decoder)

        ico_path = Path("data/icospheres")
        icos = IcoSpheres(icosphere_path=ico_path)
        self.unpool_layers = nn.ModuleList()
        self.decoder_conv_layers = nn.ModuleList()
        # TODO: parameters
        spiral_len, level = 7, 2  # make it automatically
        in_size: int = feature_dim[-1]
        for i in range(self.num_stages - 1):
            upsample = icos.get_upsample(target_level=level)
            num: int = len(icos.get_neighbours(level=level))

            self.unpool_layers.append(
                HexUnpool(upsample_indices=upsample, target_size=num)
            )

            # 2. SpiralConv
            icos.create_spirals(level=level)
            indices = icos.get_spirals(level=level)
            indices = indices[:, :spiral_len]

            block: List[SpiralConv] = []
            input_dim: int = in_size + skip_dims[i]

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

        self.ds_heads = nn.ModuleDict()
        self.ds_dist_heads = nn.ModuleDict()
        self.ds_levels: List[int] = []

        level = 2  # first level after upsampling
        for i in range(self.num_stages - 1):
            out_ch: int = skip_dims[i]  # C_to for stage_to
            head = nn.Linear(out_ch, 2)
            dist_head = nn.Linear(out_ch, 1)

            str_level = str(level)
            self.ds_heads[str_level] = head
            self.ds_dist_heads[str_level] = dist_head
            self.ds_levels.append(level)
            level += 1

        # ----------------------------
        self.pool_layers: Dict[int, HexPool] = {
            level: HexPool(icos.get_downsample(target_level=level))
            for level in range(1, 7)[::-1]
        }

        final_in: int = feature_dim[0]

        self.activation_function = nn.LeakyReLU()
        self.hemi_classification_head = nn.ModuleList(
            [
                nn.Conv1d(final_in, 1, kernel_size=1),
                nn.Linear(len(icos.icospheres[1]["coords"]), 2),
            ]
        )

        self.final_lin = nn.Linear(final_in, 2)
        self.dist_lin = nn.Linear(final_in, 1)

    def forward(
        self, data: Tuple[List[str], Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        subject_ids, text = data
        B = len(subject_ids)

        graph_output: Dict[str, List[Batch]] = self.encoder(subject_ids)
        graph_features: List[Batch] = graph_output["feature"]
        text_output: Dict[str, torch.Tensor] = self.text_encoder(text)
        text_hidden_last: torch.Tensor = text_output["feature"]  # [B, L_seq, 768]

        outputs: Dict[str, torch.Tensor] = {}

        # Climb from deep to shallow stages (list of B Data objects per stage)
        current_graphs: List[Data] = graph_features[-1].to_data_list()

        # Each decoder maps stage N → N-1, then N-1 → N-2, etc.
        for idx, (decoder, unpool, spiral_conv) in enumerate(
            zip(self.decoders, self.unpool_layers, self.decoder_conv_layers)
        ):
            stage_from: int = self.num_stages - 1 - idx
            stage_to: int = stage_from - 1

            next_graphs: List[Data] = graph_features[stage_to].to_data_list()
            cur_level: int = self.ds_levels[idx]  # level for Deep Supervision head at this stage

            ds_logp_level: List[torch.Tensor] = []
            ds_dist_level: List[torch.Tensor] = []
            updated_graphs: List[Data] = []

            for j in range(B):
                vis_feat: torch.Tensor = current_graphs[j].x.unsqueeze(0)  # [1, N_from, C_from]
                skip_feat: torch.Tensor = next_graphs[j].x.unsqueeze(0)  # [1, N_to, C_to]
                
                if self.text_emb:
                    # TODO: hyperparameter
                    # if vis_feat.shape[1] <= 162:
                    txt_emb: torch.Tensor = text_hidden_last[j].unsqueeze(0)  # [1, L_seq, 768]
                    # else:
                    # txt_emb = None
                else:
                    txt_emb = None

                # TODO: hyperparameter
                chunk: bool = vis_feat.size(1) > 40962
                out_feat: torch.Tensor = decoder(
                    vis_feat, skip_feat, txt_emb, unpool, spiral_conv, chunk
                )  # [1, N_out, C_to]
                x_lvl: torch.Tensor = out_feat.squeeze(0)  # [N_to, C_to]

                # --- Deep Supervision head for current level ---
                if str(cur_level) in self.ds_heads:  # Linear(C_to -> 2)
                    head: nn.Module = self.ds_heads[str(cur_level)]

                    logits: torch.Tensor = head(x_lvl)  # [N_to, 2]
                    logp: torch.Tensor = nn.LogSoftmax(dim=1)(logits)  # [N_to, 2]
                    ds_logp_level.append(logp)

                    dist: torch.Tensor = self.ds_dist_heads[str(cur_level)](
                        x_lvl
                    )  # [N_to, 1]
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
            if ds_dist_level:
                outputs[f"ds{cur_level}_non_lesion_logits"] = torch.cat(
                    ds_dist_level, dim=0
                )  # [B*H*V_level, 1]

            current_graphs = updated_graphs

        # 4) Final level
        final_logp_list: List[torch.Tensor] = []
        final_cls_list: List[torch.Tensor] = []
        final_dist_list: List[torch.Tensor] = []

        for g in current_graphs:
            seg_logits: torch.Tensor = self.final_lin(g.x)  # [N1, 2]
            log_seg_logits: torch.Tensor = nn.LogSoftmax(dim=1)(seg_logits)
            final_logp_list.append(log_seg_logits)  # [N1, 2]

            pool_g: torch.Tensor = g.x.unsqueeze(0).unsqueeze(0)  # [1,1,N1,C]
            for lvl in range(6, 0, -1):
                pool_g = self.pool_layers[lvl](pool_g)
            pool_g = pool_g.squeeze(0).squeeze(0)

            hemi_classification: torch.Tensor = self.activation_function(
                self.hemi_classification_head[0](pool_g.unsqueeze(2))
            )
            hemi_classification = self.hemi_classification_head[1](
                hemi_classification.view(-1)
            )
            hemi_classification = nn.LogSoftmax(dim=0)(hemi_classification)
            final_cls_list.append(hemi_classification)

            # distance head
            if hasattr(self, "dist_lin"):
                final_dist_list.append(self.dist_lin(g.x).squeeze(-1))  # [N1]

        outputs["log_softmax"] = torch.cat(final_logp_list, dim=0)  # [B*H*V1, 2]
        outputs["hemi_log_softmax"] = torch.cat(final_cls_list, dim=0)
        if final_dist_list:
            outputs["non_lesion_logits"] = torch.cat(final_dist_list, dim=0)  # [B*H*V1]

        return outputs
