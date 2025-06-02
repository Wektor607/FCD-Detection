import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import nibabel as nib
import torch
import pandas as pd
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import BASE_PATH
from meld_graph.data_preprocessing import Preprocess as Prep
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,
                              RandCoarseShuffled,RandRotated,RandZoomd, Resized, 
                              ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class EpilepDataset(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[160, 256, 256]):

        super(EpilepDataset, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        
        self.roi_list = list(self.data['ROI_PATH'])

        target_shape = image_size
        valid_indices = []

        cohort = MeldCohort(
            hdf5_file_root="{site_code}_featurematrix.hdf5",
            dataset=None,
            data_dir=BASE_PATH
        )
        config = load_config("/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/example_experiment_config.py")
        self.prep = Prep(cohort=cohort, params=config.data_parameters)
        # Exclude data where roi shape != target_shape
        # for i, roi_path in enumerate(self.roi_list):
        #     full_path = os.path.join(root_path, roi_path)
        #     try:
        #         img = nib.load(full_path)
        #         if list(img.shape) == target_shape:
        #             valid_indices.append(i)
        #         else:
        #             print(f"Skipping {roi_path}, shape = {img.shape}")
        #     except Exception as e:
        #         print(f"[!] Failed to load {roi_path}: {e}")

        # Descriptions may not be generated for some columns
        self.data['harvard_oxford'] = self.data['harvard_oxford'].fillna('')
        self.data['aal'] = self.data['aal'].fillna('')

        self.caption_list = list(self.data['harvard_oxford'] + '; ' + self.data['aal'])

        # Getting filtered data
        # self.roi_list = [self.roi_list[i] for i in valid_indices]
        # self.caption_list = [self.caption_list[i] for i in valid_indices]

        # TODO: Make a hyperparameters
        start_split, end_split = 0.8, 0.9
        total_len = len(self.roi_list)

        if mode == 'train':
            idx_range = slice(0, int(start_split * total_len))
        elif mode == 'valid':
            idx_range = slice(int(start_split * total_len), int(end_split * total_len))
        else:  # test
            idx_range = slice(int(end_split * total_len), total_len)

        self.subject_ids = [os.path.basename(path).split('_')[0] for path in self.roi_list[idx_range]]

        self.roi_list = self.roi_list[idx_range]
        self.caption_list = self.caption_list[idx_range]
        
        self.root_path = root_path
        self.image_size = image_size

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.roi_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        roi = os.path.join(self.root_path, self.roi_list[idx])
        caption = self.caption_list[idx]

        subject_data_list = next(iter(self.prep.get_data_preprocessed(
            subject=self.subject_ids[idx],
            features=self.prep.params["features"],
            lobes=self.prep.params["lobes"],
            lesion_bias=False,
        )))
        
        roi = subject_data_list['labels']
        
        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=256, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'],token_output['attention_mask']

        data = {'roi':roi, 'token':token, 'mask':mask}
        data = trans(data)

        roi, token, mask = data['roi'], data['token'], data['mask']
        roi = roi.squeeze(0)

        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        return ([self.subject_ids[idx], text], roi, self.roi_list[idx])

    def transform(self,image_size=[160, 256, 256]):

        trans = Compose([
            # LoadImaged(["roi"], reader='NibabelReader'),
            # EnsureChannelFirstd(keys=["roi"]),
            # Resized(keys=["roi"], spatial_size=image_size, mode="nearest"),
            ToTensord(["roi", "token", "mask"]),
        ])

        return trans


