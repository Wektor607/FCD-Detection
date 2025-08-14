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
from meld_graph.augment import Augment
from meld_graph.graph_tools import GraphTools
from meld_graph.icospheres import IcoSpheres
from monai.transforms import (AddChanneld, Compose, Lambdad, NormalizeIntensityd,
                              RandCoarseShuffled,RandRotated,RandZoomd, Resized, 
                              ToTensord, LoadImaged, EnsureChannelFirstd)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_config(config_file):
    '''load config.py file and return config object'''
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader('config', config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config

class EpilepDataset(Dataset):

    def __init__(self, csv_path=None, 
                 root_path=None, 
                 tokenizer=None, 
                 mode='train', 
                 meld_path='', 
                 output_dir='', 
                 feature_path='', 
                 subject_ids=None,
                 aug_flag=False):

        super(EpilepDataset, self).__init__()

        self.mode           = mode
        self.meld_path      = meld_path
        self.output_dir     = output_dir
        self.feature_path   = feature_path
        self.subject_ids    = subject_ids
        self.aug_flag       = aug_flag
        with open(csv_path, 'r') as f:
            self.data = pd.read_csv(
                f,
                sep=',',
                engine='python',            # нужен движок Python, чтобы поддерживался escapechar
                quoting=csv.QUOTE_NONE,     # не искать кавычки
                escapechar='\\'             # '\' перед ',' будет означать «не разделитель»
            )
        
        self.tokenizer  = tokenizer
        self.roi_list   = list(self.data['ROI_PATH'])

        # 2) вытаскиваем sub-ID
        self.data['sub'] = self.data['DATA_PATH'].apply(
            lambda p: os.path.basename(p).split('_patient')[0].split('_control')[0]#.split('_')[0] 
                    if isinstance(p, str) else None
        )

        # 3) задаём sub как индекс, чтобы удобнее было выбирать
        self.data   = self.data.set_index('sub')
        
        self.data = self.data.loc[subject_ids]
        
        cohort      = MeldCohort(
            # hdf5_file_root='{site_code}_featurematrix.hdf5',
            hdf5_file_root='{site_code}_{group}_featurematrix_combat.hdf5',
            dataset=None,
            data_dir=BASE_PATH
        )

        self.config = load_config('/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/example_experiment_config.py')
        self.prep   = Prep(cohort=cohort, params=self.config.data_parameters)

        # Descriptions may not be generated for some columns
        if 'description' in self.data.columns:
            self.data['description']    = self.data['description'].fillna('')
            self.caption_list           = list(self.data['description'])
            self.max_length             = 384
        else:
            self.data['harvard_oxford'] = self.data['harvard_oxford'].fillna('')
            self.data['aal']            = self.data['aal'].fillna('')
            self.max_length             = 256
        
        self.roi_list     = list(self.data['ROI_PATH'])
        self.caption_list = list(self.data['harvard_oxford'])

        self.root_path = root_path


    def __len__(self):
        return len(self.roi_list)

    def run_meld_prediction(self, subject_id: str, mode: str, aug_flag: bool):
        # command = [
        #     self.meld_path,
        #     'run_script_prediction.py',
        #     '-id', subject_id,
        #     '-harmo_code', 'fcd',
        #     '-demos', 'participants_with_scanner.tsv',
        #     *(['--aug_mode', 'train'] if aug_flag else []),
        # ]
        # TEST IT
        command = [
            self.meld_path,
            'run_script_prediction_meld.py',
            '-id', subject_id,
            '-harmo_code', 'fcd',
            '-demos', 'input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv',
            *(['--aug_mode', 'train'] if aug_flag else []),
        ]
        # Run: ./meldgraph.sh run_script_prediction_meld.py --list_ids /home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv --demographic_file /home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing/demographics_qc_allgroups_withH27H28H101.csv
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error running MELD prediction for {subject_id}: {e}')
            raise
        
    def __getitem__(self, idx):

        caption = self.caption_list[idx]
        subject_data_list = self.prep.get_data_preprocessed(
            subject=self.subject_ids[idx],
            features=self.prep.params['features'],
            lobes=self.prep.params['lobes'],
            lesion_bias=False,
            distance_maps=False,
            harmo_code='fcd', #TODO: Make a hyperparameter
            only_lesion=False,  #TODO: Make a hyperparameter
            only_features= self.roi_list[idx] is None,
            combine_hemis=self.prep.params["combine_hemis"]
        )

        # Generating features
        features_dir = os.path.join(self.feature_path, 'preprocessed', 'meld_files', self.subject_ids[idx], 'features')
        npz_path = os.path.join(features_dir, 'feature_maps.npz')
        if not os.path.isfile(npz_path):
            self.run_meld_prediction(self.subject_ids[idx], self.mode, aug_flag=self.aug_flag)
            if not os.path.isfile(npz_path):
                raise FileNotFoundError(f'Failed to generate NPZ for {self.subject_ids[idx]}')

        labels_tensors = []
        for d in subject_data_list:
            if d.get('labels') is None:
                n_verts = d['features'].shape[0]
                labels_tensors.append(
                    torch.zeros(n_verts, dtype=torch.long)
                )
            else:
                labels_tensors.append(
                    torch.from_numpy(d['labels']).long()
                )
        roi = torch.stack(labels_tensors, dim=0)

        token_output = self.tokenizer.encode_plus(caption, padding='max_length',
                                                        max_length=self.max_length,
                                                        truncation=True,
                                                        return_attention_mask=True,
                                                        return_tensors='pt')
        token, mask = token_output['input_ids'],token_output['attention_mask']

        text = {'input_ids': token.squeeze(dim=0), 'attention_mask': mask.squeeze(dim=0)} 
        return ([self.subject_ids[idx], text], roi)