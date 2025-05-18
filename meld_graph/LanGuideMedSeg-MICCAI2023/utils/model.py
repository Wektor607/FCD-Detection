import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import nibabel as nib
import subprocess
import torch.nn as nn
from fsl.wrappers import flirt
from nilearn import datasets, image
from einops import rearrange, repeat
from layers import GuideDecoder
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
    def __init__(self, project_dim, meld_script_path, output_dir, subject_id, input_shape=(218, 182, 218)):
        super().__init__()
        self.project_head = nn.Sequential(
            nn.Linear(np.prod(input_shape), 1024),
            nn.ReLU(),
            nn.Linear(1024, project_dim)
        )
        self.meld_script_path = meld_script_path
        self.output_dir       = output_dir
        self.subject_id       = subject_id
        self.input_shape      = input_shape
        self.pred_path        = '../../dataset'

    def run_meld_prediction(self):
        command = [
            self.meld_script_path,
            'run_script_prediction.py',
            "-id", self.subject_id,
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running MELD prediction: {e}")
            raise

    def load_prediction_nifti(self):
        pred_path = os.path.join(
            self.output_dir,
            "predictions_reports",
            f"{self.subject_id}",
            "predictions",
            "prediction.nii.gz"
        )
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"Prediction file not found: {pred_path}")

        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm', symmetric_split=True)
        atlas_img = atlas['maps']
        pred_img = nib.load(pred_path)
        
        resampled_pred = image.resample_to_img(pred_img, atlas_img, interpolation='nearest')

        pred_data = resampled_pred.get_fdata()
        atlas_data = atlas_img.get_fdata()
        z_value = np.max(np.unique(atlas_data))
        pred_data_z = (pred_data > 0).astype(np.float32) * z_value
        z_pred = nib.Nifti1Image(pred_data_z, resampled_pred.affine, resampled_pred.header)
        
        zpred_path = os.path.join(self.pred_path, self.subject_id, "pred_in_atlas.nii.gz")
        nib.save(z_pred, zpred_path)
        return resampled_pred

    def forward(self):
        # Step 1: run MELD prediction
        if not os.path.isfile(os.path.join(
            self.output_dir, 
            "predictions_reports", 
            f"{self.subject_id}",
            "predictions",
            "prediction.nii.gz"
        )):
            self.run_meld_prediction()

        # Step 2: load and process prediction.nii.gz
        x = self.load_prediction_nifti()
        print(x)
        raise(0)
        # Step 3: to tensor and flatten
        pred_tensor = torch.tensor(pred_data, dtype=torch.float32).view(1, -1)  # [1, D]

        # Step 4: project to embedding
        embedding = self.project_head(pred_tensor)

        return {"feature": None, "project": embedding}

vision = VisionModel(
    project_dim=512,
    meld_script_path="../../meldgraph.sh",
    output_dir="../../data/output",
    subject_id="sub-00003"
)

out = vision()
print(out["project"].shape)
# class VisionModel(nn.Module):

#     def __init__(self, vision_type, project_dim):
#         super(VisionModel, self).__init__()

#         self.project_head = nn.Linear(768, project_dim)
#         self.spatial_dim = 768

#     def forward(self, x):

#         output = self.model(x, output_hidden_states=True)
#         embeds = output['pooler_output'].squeeze()
#         project = self.project_head(embeds)

#         return {"feature":output['hidden_states'], "project":project}


class LanGuideMedSeg(nn.Module):

    def __init__(self, bert_type, vision_type, project_dim=512):

        super(LanGuideMedSeg, self).__init__()

        self.encoder = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)

        self.spatial_dim = [7,14,28,56]    # 224*224
        feature_dim = [768,384,192,96]

        self.decoder16 = GuideDecoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = GuideDecoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = GuideDecoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):

        image, text = data
        if image.shape[1] == 1:   
            image = repeat(image,'b 1 h w -> b c h w',c=3)

        image_output = self.encoder(image)
        image_features, image_project = image_output['feature'], image_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, text_project = text_output['feature'],text_output['project']

        if len(image_features[0].shape) == 4: 
            image_features = image_features[1:]  # 4 8 16 32   convnext: Embedding + 4 layers feature map
            image_features = [rearrange(item,'b c h w -> b (h w) c') for item in image_features] 

        os32 = image_features[3]
        os16 = self.decoder16(os32,image_features[2], text_embeds[-1])
        os8 = self.decoder8(os16,image_features[1], text_embeds[-1])
        os4 = self.decoder4(os8,image_features[0], text_embeds[-1])
        os4 = rearrange(os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        os1 = self.decoder1(os4)

        out = self.out(os1).sigmoid()

        return out
    
