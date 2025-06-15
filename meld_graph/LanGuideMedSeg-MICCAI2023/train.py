import sys
import torch
import wandb
import utils.config as config
from os.path import join as opj
import os
from torch.optim import lr_scheduler
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from utils.dataset_bonn import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl    
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse

from meld_graph.paths import (FS_SUBJECTS_PATH, 
                              MELD_DATA_PATH,
                                )
from utils.save_predictions import SavePredictionsCallback

def get_parser():
    parser = argparse.ArgumentParser(
        description='Language-guide Medical Image Segmentation')
    parser.add_argument('--config',
                        default='./config/training.yaml',
                        type=str,
                        help='config file')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)

    return cfg

import random
import numpy as np
import torch

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


if __name__ == '__main__':

    args = get_parser()
    
    # wandb_logger = WandbLogger(
    #     project=args.project_name,
    #     log_model=True
    # )   

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
    ds_train = EpilepDataset(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=tokenizer,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = EpilepDataset(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=tokenizer,
                    image_size=args.image_size,
                    mode='valid')


    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size, pin_memory=True, worker_init_fn=worker_init_fn)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size, pin_memory=True, worker_init_fn=worker_init_fn)
    
    ## 2. setting trainer
    if torch.cuda.is_available():
        accelerator = "gpu"
        # TODO: Using more than 1 GPU require syncronization, because I got worse perfomance, than in 1 GPU
        devices     =  "auto" #torch.cuda.device_count()
        strategy    = "ddp_sharded"
    else:
        accelerator = "cpu"
        args.device ="cpu"
        devices     = 1
        strategy    = None

    ckpt_path = None #'/home/s17gmikh/FCD-Detection/meld_graph/LanGuideMedSeg-MICCAI2023/save_model/medseg-v4.ckpt'
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
        model = LanGuideMedSegWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            args=args,
            root_path=ds_train.root_path,
            tokenizer=tokenizer
        )
    else:
        if ckpt_path is not None:
            print(f"[WARNING] Checkpoint file not found at {ckpt_path}. Initializing new model.")
        model = LanGuideMedSegWrapper(args, ds_train.root_path, tokenizer=tokenizer)
    
    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    early_stopping = EarlyStopping(monitor = 'val_loss',
                            patience=args.patience,
                            mode = 'min'
    )


    torch.set_float32_matmul_precision('high')
    # Add new parameters in Trainer very accurate, because it arise problems with memory 
    classifier_output_dir    = opj(MELD_DATA_PATH, 'output', 'classifier_outputs', model_ckpt.__class__.__name__)
    train_prediction_file    = opj(classifier_output_dir, 'results_best_model', 'train_predictions.hdf5')
    val_prediction_file    = opj(classifier_output_dir, 'results_best_model', 'val_predictions.hdf5')
    test_prediction_file     = opj(classifier_output_dir, 'results_best_model', 'predictions.hdf5')
    predictions_output_dir   = opj(MELD_DATA_PATH, 'output', 'predictions_reports')

    save_pred = SavePredictionsCallback(
        subjects_dir           = FS_SUBJECTS_PATH,
        train_prediction_file  = train_prediction_file,
        val_prediction_file    = val_prediction_file,
        test_prediction_file   = test_prediction_file,
        predictions_output_dir = predictions_output_dir,
        verbose = True
    )

    trainer = pl.Trainer(
                        # logger=wandb_logger,
                        min_epochs=args.min_epochs,
                        max_epochs=args.max_epochs,
                        accelerator=accelerator, 
                        devices=devices,
                        # strategy=strategy, ###################
                        callbacks=[model_ckpt, early_stopping],
                        enable_progress_bar=True,
                        
                    ) 

    ## 3. start training
    print('start training')
    trainer.fit(model,dl_train,dl_valid)
    print('done training')

    # --- Подготовка тестовой выборки и DataLoader ---
    ds_test = EpilepDataset(
        csv_path=args.test_csv_path,             # новый CSV для теста
        root_path=args.test_root_path,           # путь к тестовым данным
        tokenizer=tokenizer,
        image_size=args.image_size,
        mode='test'
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.valid_batch_size,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # --- Запуск тестового прогона ---
    # Используем чекпоинт с наилучшим val_loss
    print('start testing')
    test_results = trainer.test(
        model,
        dataloaders=dl_test,
        ckpt_path='best',    # можно также указать model_ckpt.best_model_path
    )
    print('test results:', test_results)
    # wandb.finish()

