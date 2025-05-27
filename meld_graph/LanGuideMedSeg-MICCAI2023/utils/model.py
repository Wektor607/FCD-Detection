import os
import sys
from typing import List
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import subprocess
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from utils.layers import GuideDecoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample
from transformers import AutoTokenizer, AutoModel



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
    def __init__(self, project_dim: int, meld_script_path: str, feature_path: str, output_dir: str, device: str):
        super().__init__()
        
        self.project_heads = nn.ModuleList([
            nn.Linear(16, project_dim),   # for (163842, 32)
            nn.Linear(32, project_dim),   # for (163842, 32)
            nn.Linear(32, project_dim),   # for (40962, 32)
            nn.Linear(64, project_dim),   # for (10242, 64)
            nn.Linear(64, project_dim),   # for (2562, 64)
            nn.Linear(128, project_dim),  # for (642, 128)
            nn.Linear(128, project_dim),  # for (162, 128)
            # nn.Linear(256, project_dim),  # for (42, 256)
        ])

        self.meld_script_path = meld_script_path
        self.feature_path     = feature_path
        self.output_dir       = output_dir
        
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

    def forward(self, subject_ids:List[str]):
        features_path = os.path.join(self.feature_path, "input", subject_ids[0], "anat", "features", "feature_maps.npz")
        features = np.load(features_path)
        sorted_keys = sorted(features.files, key=lambda k: int(k.replace('stage', '')))
        stage_embeddings = [[] for _ in sorted_keys]

        for subject_id in subject_ids:

            # Step 1: run MELD prediction
            if not os.path.isfile(os.path.join(self.output_dir, "predictions_reports", f"{subject_id}", "predictions/prediction.nii.gz")):
                self.run_meld_prediction(subject_id)

            # Step 2: get features from all model layers
            features_path = os.path.join(self.feature_path, "input", subject_id, "anat", "features", "feature_maps.npz")
            features = np.load(features_path)

            # Step 3: We squeeze and project the embeddings to the same dimension
            for i, stage in enumerate(sorted_keys):
                feat_squeeze = torch.tensor(features[stage], dtype=torch.float32, device=self.device).squeeze()
                stage_embeddings[i].append(feat_squeeze)

        # Remove first 2 stages:
        stage_embeddings = stage_embeddings[3:]

        batch_tensors = []
        for i, level_batch in enumerate(stage_embeddings):
            try:
                level_tensor = torch.stack(level_batch, dim=0)  # [B, N, D]
            except RuntimeError:
                print(f"[!] Stage {i}: Cannot stack due to size mismatch. Keeping as list.")
                level_tensor = level_batch

            batch_tensors.append(level_tensor)

        return {"feature": batch_tensors, "project": None}

class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, 
                 meld_script_path, 
                 feature_path, 
                 output_dir,
                 project_dim=512,
                 device='cpu'):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(project_dim, meld_script_path, feature_path, output_dir, device)
        self.text_encoder = BERTModel(bert_type, project_dim)

        # Layer stage1 — shape: torch.Size([1, 163842, 32])
        # Layer stage2 — shape: torch.Size([1, 40962, 32])
        # Layer stage3 — shape: torch.Size([1, 10242, 64])
        # Layer stage4 — shape: torch.Size([1, 2562, 64])
        # Layer stage5 — shape: torch.Size([1, 642, 128])
        # Layer stage6 — shape: torch.Size([1, 162, 128])
        # Layer stage7 — shape: torch.Size([1, 42, 256])
        
        feature_dim      = [256, 128, 128, 64, 64, 32, 32] # stage_i[2]
        self.spatial_dim = [
            (1, 6, 7),     # stage7: 42
            (2, 12, 14),   # stage6: 336
            (4, 24, 28),   # stage5: 2688
            (8, 48, 56),   # stage4: 21,504
            # === CUDA out of memory ===
            # (16, 96, 112),  # stage3: 172,032
            # (32, 192, 224), # stage2: 1,376,256
            # (64, 384, 448)  # stage1: 11,010,048
        ]       
        text_lens        = [128, 64, 64, 48, 48, 24, 24]

        self.decoder7 = GuideDecoder(feature_dim[0], feature_dim[1], self.spatial_dim[0], text_lens[0])
        self.decoder6 = GuideDecoder(feature_dim[1], feature_dim[2],self.spatial_dim[1], text_lens[1])
        self.decoder5 = GuideDecoder(feature_dim[2], feature_dim[3],self.spatial_dim[2], text_lens[2])
        self.out = UnetOutBlock(3, in_channels=feature_dim[3], out_channels=1)
        # self.decoder4 = GuideDecoder(feature_dim[3], feature_dim[4],self.spatial_dim[3], text_lens[3])
        # self.upsample = SubpixelUpsample(3, feature_dim[4], 32, 12)
        
        # self.decoder3 = GuideDecoder(feature_dim[4], feature_dim[5],self.spatial_dim[4], text_lens[4])
        # self.decoder2 = GuideDecoder(feature_dim[5], feature_dim[6],self.spatial_dim[5], text_lens[5])
        # self.out = UnetOutBlock(3, in_channels=feature_dim[6], out_channels=1)
        # self.out = UnetOutBlock(2, in_channels=48, out_channels=1)
    
    def forward(self, data):

        subject_ids, text = data

        # if image.shape[1] == 1:   
        #     image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(subject_ids)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c d h w -> b (d h w) c') for item in image_features] 
        
        # os8 = image_features[6]
        # os7  = self.decoder7(os8, image_features[5], text_embeds[-1])
        # os6  = self.decoder6(os7, image_features[4], text_embeds[-1])
        # os5  = self.decoder5(os6, image_features[3], text_embeds[-1])
        # os4  = self.decoder4(os5, image_features[2], text_embeds[-1])
        # os3  = self.decoder3(os4, image_features[1], text_embeds[-1])
        # os2  = self.decoder2(os3, image_features[0], text_embeds[-1])

        
        os8 = image_features[3]
        os7 = self.decoder7(os8, image_features[2], text_embeds[-1])
        os6 = self.decoder6(os7, image_features[1], text_embeds[-1])
        os5 = self.decoder5(os6, image_features[0], text_embeds[-1])
        # os4 = self.decoder4(os5, image_features[0], text_embeds[-1])

        D, H, W = self.spatial_dim[3]
        vol = rearrange(os5, 'B (D H W) C -> B C D H W', D=D, H=H, W=W)
        
        vol = F.interpolate(vol, scale_factor=2, mode='trilinear', align_corners=False)
        
        out = self.out(vol).sigmoid()
        
        print('Out_shape_before_resampling:', out.shape)
        
        # os2 = rearrange(os2, 'B (D H W) C -> B C D H W', D=D, H=H,W=W)
        # os1 = self.decoder1(os2)
        # print(os1.shape)
        return out
    
