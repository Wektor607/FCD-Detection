import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import wandb
import utils.config as config
from os.path import join as opj
import torch.nn as nn
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex, Precision
from torch.optim import lr_scheduler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import subprocess

from utils.data import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl    
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

from meld_graph.paths import (FS_SUBJECTS_PATH, 
                              MELD_DATA_PATH,
                                )
from meld_graph.meld_cohort import MeldCohort
from test import test_model
from sklearn.model_selection import train_test_split

import random
import numpy as np
import pandas as pd
import torch
import csv

SEED = 42

# Python RNG
random.seed(SEED)

# NumPy RNG
np.random.seed(SEED)

# Torch RNG for CPU and CUDA
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Гарантируем детерминированность cuDNN (но может замедлить обучение)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Если вы используете DataLoader с несколькими воркерами:
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--meld_check',
                        default=False,
                        type=bool,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.meld_check = args.meld_check
    return cfg

def mgh_cleaner():
    cleanup_cmd = [
        "find",
        os.path.join(FS_SUBJECTS_PATH),
        "-type", "f",
        "-path", "*/xhemi/classifier/*",
        "-delete"
    ]
    subprocess.run(cleanup_cmd, check=True)

def _filter_existing(data: pd.DataFrame, ids: list[str]) -> list[str]:
    """Оставить только те ids, что есть в data.index."""
    valid_ids = [sid for sid in ids if sid in data.index]
    missing   = sorted(set(ids) - set(valid_ids))
    if missing:
        print(f"⚠️ Эти участники отсутствуют и будут пропущены: {missing}")
    return valid_ids

if __name__ == '__main__':
    args = get_parser()
    
    test_ids = ['sub-00001']

    ckpt_path = './save_model/medseg-v1.ckpt'

    # --- Запуск тестового прогона ---    
    print('start testing')
    # test_model(model_ckpt.best_model_path, test_ids)
    test_model(ckpt_path, test_ids)
    # wandb.finish()
    

