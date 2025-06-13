import os
import sys
import re
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
        
        self.tokenizer = tokenizer
        self.roi_list = list(self.data['ROI_PATH'])

        cohort = MeldCohort(
            hdf5_file_root="{site_code}_featurematrix.hdf5",
            dataset=None,
            data_dir=BASE_PATH
        )
        config = load_config("/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/example_experiment_config.py")
        self.prep = Prep(cohort=cohort, params=config.data_parameters)

        # Descriptions may not be generated for some columns
        self.data['harvard_oxford'] = self.data['harvard_oxford'].fillna('')
        self.data['aal'] = self.data['aal'].fillna('')

        # self.caption_list = list(self.data['harvard_oxford'] + '; ' + self.data['aal'])

        # 2) Собираем все уникальные ROI-токены
        unique_tokens = set()
        processed_captions = []
        for harv, aal in zip(self.data['harvard_oxford'], self.data['aal']):
            # объединённая строка, разбиваем по ';'
            phrases = (harv + '; ' + aal).split(';')
            proc_phrases = []
            for ph in phrases:
                ph = ph.strip()
                if not ph:
                    continue
                # ожидаем формат "30.62 percent of Right Insular Cortex"
                m = re.match(r'([\d\.]+)\s*percent of\s*(.+)', ph, flags=re.IGNORECASE)
                if not m:
                    continue
                val, region = m.groups()
                token = self.normalize_roi_name(region)
                unique_tokens.add(token)
                proc_phrases.append(f"{val}% {token}")
            
            processed_captions.append("; ".join(proc_phrases))
        
        # 3) Добавляем новые токены в tokenizer и обновляем эмбеддинги
        self.tokenizer.add_tokens(list(unique_tokens))

        # 4) Используем уже предобработанные тексты
        self.caption_list = processed_captions
        
        # TODO: Make a hyperparameters
        start_split, end_split = 0.7, 0.9
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

        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

    def __len__(self):
        return len(self.roi_list)

    def __getitem__(self, idx):

        trans = self.transform(self.image_size)

        roi = os.path.join(self.root_path, self.roi_list[idx])
        caption = self.caption_list[idx]

        subject_data_list = self.prep.get_data_preprocessed(
            subject=self.subject_ids[idx],
            features=self.prep.params["features"],
            lobes=self.prep.params["lobes"],
            lesion_bias=False,
            harmo_code="fcd", #TODO: Make a hyperparameter
            only_lesion=True  #TODO: Make a hyperparameter
        )

        roi = torch.cat(
            [torch.from_numpy(d['labels']).unsqueeze(0) for d in subject_data_list],
            dim=0
        )

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
        return ([self.subject_ids[idx], text], roi)

    def transform(self,image_size=[160, 256, 256]):

        trans = Compose([
            # LoadImaged(["roi"], reader='NibabelReader'),
            # EnsureChannelFirstd(keys=["roi"]),
            # Resized(keys=["roi"], spatial_size=image_size, mode="nearest"),
            ToTensord(["roi", "token", "mask"]),
        ])

        return trans

    def normalize_roi_name(self, name: str) -> str:
        name = name.lower()
        # убрать всё в скобках
        name = re.sub(r'\s*\(.+?\)', '', name)
        # разбить на слова
        toks = name.split()
        # вынести ориентацию
        if toks[0] in ('left','right'):
            orient = toks.pop(0)
        elif toks[-1] in ('left','right'):
            orient = toks.pop(-1)
        else:
            orient = ''
        # простые аббревиатуры
        abbrev = {'cortex':'ctx', 'gyrus':'gyr', 'lobule':'lob'}
        toks = [abbrev.get(t, t) for t in toks]
        toks = [t.replace('-', '_') for t in toks]
        # собрать обратно
        parts = ([orient] if orient else []) + toks
        return "_".join(parts)
