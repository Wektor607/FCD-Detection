import json
import os
import torch
import numpy as np
import pandas as pd
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,
                              RandCoarseShuffled,RandRotated,RandZoomd, Resized, 
                              ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

class EpilepDataset(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', image_size=[160, 256, 256]):

        super(EpilepDataset, self).__init__()

        self.mode = mode

        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(f)
        
        # self.hdf5_list = list(self.data['DATA_PATH'])
        self.roi_list = list(self.data['ROI_PATH'])
        self.caption_list = list(self.data['harvard_oxford'] + '; ' + self.data['aal'])

        # TODO: Make a hyperparameters
        start_split, end_split = 0.6, 0.8
        # total_len = len(self.hdf5_list)
        total_len = len(self.roi_list)

        if mode == 'train':
            idx_range = slice(0, int(start_split * total_len))
        elif mode == 'valid':
            idx_range = slice(int(start_split * total_len), int(end_split * total_len))
        else:  # test
            idx_range = slice(int(end_split * total_len), total_len)

        # self.hdf5_list = self.hdf5_list[idx_range] ## MAYBE DELETE, BECAUSE I USE ONLY INDEXES
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

        # hdf5_file = os.path.join(self.root_path, self.hdf5_list[idx])
        roi = os.path.join(self.root_path, self.roi_list[idx])
        caption = self.caption_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=256, 
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'],token_output['attention_mask']

        # data = {'hdf5_data':hdf5_file, 'roi':roi, 'token':token, 'mask':mask}
        data = {'roi':roi, 'token':token, 'mask':mask}
        data = trans(data)

        # hdf5_data, roi, token, mask = data['hdf5_data'], data['roi'], data['token'], data['mask']
        roi, token, mask = data['roi'], data['token'], data['mask']
        roi = roi.squeeze(0)

        text = {'input_ids':token.squeeze(dim=0), 'attention_mask':mask.squeeze(dim=0)} 
        return ([self.subject_ids[idx], text], roi)
        # return (self.subject_ids[idx], [hdf5_data, text], roi)

    def transform(self,image_size=[160, 256, 256]):

        trans = Compose([
            LoadImaged(["roi"], reader='NibabelReader'),
            EnsureChannelFirstd(keys=["roi"]),
            Resized(keys=["roi"], spatial_size=image_size, mode="nearest"),
            ToTensord(["roi", "token", "mask"]),
        ])

        return trans


