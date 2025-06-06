import sys
import torch
import wandb
import utils.config as config

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from utils.dataset_bonn import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl    
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse


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


if __name__ == '__main__':

    args = get_parser()
    
    # wandb_logger = WandbLogger(
    #     project=args.project_name,
    #     log_model=True
    # )   
    
    ds_train = EpilepDataset(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='train')

    ds_valid = EpilepDataset(csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=args.bert_type,
                    image_size=args.image_size,
                    mode='valid')


    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.train_batch_size, pin_memory=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size, pin_memory=True)

    ## 2. setting trainer
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"
        strategy = "ddp_sharded"
    else:
        accelerator = "cpu"
        args.device="cpu"
        devices = 1
        strategy = None

    model = LanGuideMedSegWrapper(args, ds_train.root_path)

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
    trainer = pl.Trainer(
                        # logger=wandb_logger,
                        min_epochs=args.min_epochs,max_epochs=args.max_epochs,
                        accelerator=accelerator, 
                        devices=devices,
                        callbacks=[model_ckpt,],#early_stopping],
                        enable_progress_bar=True,
                        # overfit_batches=1,
                        # strategy=strategy
                    ) 

    ## 3. start training
    print('start training')
    trainer.fit(model,dl_train,dl_valid)
    print('done training')

    # wandb.finish()

