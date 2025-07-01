import os
import sys
import re
import csv
import numpy as np
import nibabel as nb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import random
import pandas as pd
import nibabel as nib
import subprocess
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import BASE_PATH
from meld_graph.data_preprocessing import Preprocess as Prep
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,
                              RandCoarseShuffled,RandRotated,RandZoomd, Resized, 
                              ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import Dataset

def load_config(config_file):
    '''load config.py file and return config object'''
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader('config', config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config

def split_to_channels(roi_text: str):
    '''
    В результате получаем:
      names = ['Right Parietal Operculum Cortex', …]
      pcts  = [0.3062, 0.3048, …]
    '''
    names, pcts = [], []
    for part in roi_text.split(';'):
        part = part.strip()
        m = re.match(r'([\d\.]+)%\s*(.+)', part)
        if not m:
            continue
        pct = float(m.group(1)) / 100.0
        name = m.group(2).strip()
        pcts.append(pct)
        names.append(name)
    return names, pcts

def run_meld_prediction(meld_script_path: str, subject_id: str):
    command = [
        meld_script_path,
        'run_script_prediction.py',
        '-id', subject_id,
        '-harmo_code', 'fcd',
        '-demos', 'participants_with_scanner.tsv'
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error running MELD prediction for {subject_id}: {e}')
        raise
    
class EpilepDataset(Dataset):

    def __init__(self, csv_path=None, root_path=None, tokenizer=None, mode='train', meld_path='', output_dir='', feature_path=''):

        super(EpilepDataset, self).__init__()

        self.mode = mode
        self.meld_path = meld_path
        self.output_dir = output_dir
        self.feature_path = feature_path
        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(
                f,
                sep=',',
                engine='python',            # нужен движок Python, чтобы поддерживался escapechar
                quoting=csv.QUOTE_NONE,     # не искать кавычки
                escapechar='\\'             # '\' перед ',' будет означать «не разделитель»
            )
        
        self.tokenizer = tokenizer
        self.roi_list = list(self.data['ROI_PATH'])

        cohort = MeldCohort(
            hdf5_file_root='{site_code}_featurematrix.hdf5',
            dataset=None,
            data_dir=BASE_PATH
        )
        config = load_config('/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/example_experiment_config.py')
        self.prep = Prep(cohort=cohort, params=config.data_parameters)


        # Descriptions may not be generated for some columns
        if 'description' in self.data.columns:
            self.data['description'] = self.data['description'].fillna('')
            self.caption_list = list(self.data['description'])
            self.max_length = 384
        else:
            self.data['harvard_oxford'] = self.data['harvard_oxford'].fillna('')
            self.data['aal'] = self.data['aal'].fillna('')
            self.max_length = 256
            # 4) Preproccesed text
            # self.caption_list = processed_captions
            sep = self.tokenizer.sep_token
            self.caption_list = self.data['harvard_oxford'].to_list()
            # self.names_list = []
            # self.pcts_list = []
            # for txt in self.data['harvard_oxford']:
            #     names, pcts = split_to_channels(txt)
            #     self.names_list.append('; '.join(names))
            #     self.pcts_list.append(torch.tensor(pcts, dtype=torch.float32))
            # self.caption_list = (self.data['harvard_oxford'].str.strip()
            #                      + f' {sep} '
            #                      + self.data['aal'].str.strip()
            #                     ).to_list()
            
            # if mode == 'train':
            #     txt = [shuffle_regions(text) for text in self.caption_list]
            #     self.caption_list = [drop_token(text, drop_prob=0.3) for text in txt]
                # items = []
                # for item, item_new in zip(self.caption_list, self.new_caption_list):
                #     print(item, '\n', item_new, '\n\n')
                #     items.append(len(item_new))

        # # 2) Store all unique ROI tokens
        # unique_tokens = set()
        # processed_captions = []
        # for harv, aal in zip(self.data['harvard_oxford'], self.data['aal']):            
        #     phrases = (harv + '; ' + aal).split(';')
        #     proc_phrases = []
        #     for ph in phrases:
        #         ph = ph.strip()
        #         if not ph:
        #             continue
        #         # Expected format '30.62 percent of Right Insular Cortex'
        #         m = re.match(r'([\d\.]+)\s*percent of\s*(.+)', ph, flags=re.IGNORECASE)
        #         if not m:
        #             continue
        #         val, region = m.groups()
        #         token = self.normalize_roi_name(region)
        #         unique_tokens.add(token)
        #         proc_phrases.append(f'{val}% {token}')
        #     print(proc_phrases)
        #     sys.exit()
        #     processed_captions.append('; '.join(proc_phrases))
        
        # # 3) Add tokens in tokenizer and update embeddings
        # self.tokenizer.add_tokens(list(unique_tokens))
        
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
        # self.names_list = self.names_list[idx_range]
        # self.pcts_list = self.pcts_list[idx_range]
        self.caption_list = self.caption_list[idx_range]
        
        # # Shuffle indices
        # indices = np.arange(len(self.roi_list))
        # np.random.seed(42)
        # np.random.shuffle(indices)

        # # Сплит
        # start_split, end_split = 0.7, 0.9
        # n = len(indices)
        # train_end = int(start_split * n)
        # val_end   = int(end_split   * n)

        # if mode == 'train':
        #     sel = indices[:train_end]
        # elif mode == 'valid':
        #     sel = indices[train_end:val_end]
        # else:  # test
        #     sel = indices[val_end:]

        # # Применяем отобранные индексы к исходным спискам
        # self.roi_list     = [self.roi_list[i]     for i in sel]
        # self.caption_list = [self.caption_list[i] for i in sel]
        # self.subject_ids  = [
        #     os.path.basename(path).split('_')[0]
        #     for path in self.roi_list
        # ]

        # print(self.subject_ids)
        
        self.root_path = root_path


    def __len__(self):
        return len(self.roi_list)

    def __getitem__(self, idx):

        trans = self.transform()

        roi = os.path.join(self.root_path, self.roi_list[idx])

        caption = self.caption_list[idx]

        subject_data_list = self.prep.get_data_preprocessed(
            subject=self.subject_ids[idx],
            features=self.prep.params['features'],
            lobes=self.prep.params['lobes'],
            lesion_bias=False,
            harmo_code='fcd', #TODO: Make a hyperparameter
            only_lesion=True  #TODO: Make a hyperparameter
        )

        roi = torch.cat(
            [torch.from_numpy(d['labels']).unsqueeze(0) for d in subject_data_list],
            dim=0
        )

        # pred_path = os.path.join(self.output_dir, 'predictions_reports', f'{self.subject_ids[idx]}', 'predictions/prediction.nii.gz')
        # if not os.path.isfile(pred_path):
        #     run_meld_prediction(self.meld_path, self.subject_ids[idx])
        
        # features_dir = os.path.join(self.feature_path, 'input', self.subject_ids[idx], 'anat', 'features')
        # res_path = os.path.join(features_dir, 'result.npz')

        # Step 2: load subject’s NPZ
        # meld_pred = np.load(res_path)['result']
        # names = self.names_list[idx]
        # pcts = self.pcts_list[idx]

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=self.max_length,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'],token_output['attention_mask']

        data = {'roi':roi, 'token':token, 'mask':mask,} # 'meld_pred': meld_pred}#, 'num_feats': pcts}
        data = trans(data)

        roi, token, mask = data['roi'], data['token'], data['mask'] #, data['meld_pred']#, data['num_feats']
        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)} 
        return ([self.subject_ids[idx], text], roi)

    def transform(self):

        trans = Compose([
            # LoadImaged(['roi'], reader='NibabelReader'),
            # EnsureChannelFirstd(keys=['roi']),
            # Resized(keys=['roi'], spatial_size=image_size, mode='nearest'),
            ToTensord(['roi', 'token', 'mask',]), # 'meld_pred']),#'num_feats']),
        ])

        return trans

    def normalize_roi_name(self, name: str) -> str:
        name = name.lower()
        name = re.sub(r'\s*\(.+?\)', '', name)        
        toks = name.split()
        if toks[0] in ('left','right'):
            orient = toks.pop(0)
        elif toks[-1] in ('left','right'):
            orient = toks.pop(-1)
        else:
            orient = ''
        
        # simple abbreviations
        abbrev = {'cortex':'ctx', 'gyrus':'gyr', 'lobule':'lob'}
        toks = [abbrev.get(t, t) for t in toks]
        toks = [t.replace('-', '_') for t in toks]
        
        parts = ([orient] if orient else []) + toks
        return '_'.join(parts)