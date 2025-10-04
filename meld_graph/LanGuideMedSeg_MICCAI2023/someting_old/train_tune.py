import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import utils.config as config
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import subprocess

from utils.data import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch.multiprocessing

import argparse

from meld_graph.paths import (
    FS_SUBJECTS_PATH,
)

import random
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument(
        "--config", default="./config/training.yaml", type=str, help="config file"
    )
    parser.add_argument("--meld_check", default=False, type=bool, help="config file")

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.meld_check = args.meld_check
    return cfg


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


if __name__ == "__main__":
    for alpha in [0.8, 0.85, 0.9, 0.95]:
        for gamma in [2.0, 2.5, 3.0]:
            if alpha == 0.8 and (gamma == 2.0 or gamma == 2.5):
                continue
            for coef in [(0.6, 0.2, 0.2), (0.5, 0.3, 0.2), (0.4, 0.4, 0.2)]:
                cleanup_cmd = [
                    "find",
                    os.path.join(FS_SUBJECTS_PATH),
                    "-type",
                    "f",
                    "-path",
                    "*/xhemi/classifier/*",
                    "-delete",
                ]
                subprocess.run(cleanup_cmd, check=True)

                args = get_parser()

                check = args.meld_check
                # wandb_logger = WandbLogger(
                #     project=args.project_name,
                #     log_model=True
                # )

                tokenizer = AutoTokenizer.from_pretrained(
                    args.bert_type, trust_remote_code=True
                )
                ds_train = EpilepDataset(
                    csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=tokenizer,
                    mode="train",
                    meld_path=args.meld_script_path,
                    output_dir=args.output_dir,
                    feature_path=args.feature_path,
                )

                ds_valid = EpilepDataset(
                    csv_path=args.train_csv_path,
                    root_path=args.train_root_path,
                    tokenizer=tokenizer,
                    mode="valid",
                    meld_path=args.meld_script_path,
                    output_dir=args.output_dir,
                    feature_path=args.feature_path,
                )

                # --- Подготовка тестовой выборки и DataLoader ---
                ds_test = EpilepDataset(
                    csv_path=args.test_csv_path,  # новый CSV для теста
                    root_path=args.test_root_path,  # путь к тестовым данным
                    tokenizer=tokenizer,
                    mode="test",
                    meld_path=args.meld_script_path,
                    output_dir=args.output_dir,
                    feature_path=args.feature_path,
                )

                dl_train = DataLoader(
                    ds_train,
                    batch_size=args.train_batch_size,
                    shuffle=True,
                    num_workers=args.train_batch_size,
                    pin_memory=True,
                    worker_init_fn=worker_init_fn,
                    persistent_workers=True,
                )

                dl_valid = DataLoader(
                    ds_valid,
                    batch_size=args.valid_batch_size,
                    shuffle=False,
                    num_workers=args.valid_batch_size,
                    pin_memory=True,
                    worker_init_fn=worker_init_fn,
                    persistent_workers=True,
                )
                dl_test = DataLoader(
                    ds_test,
                    batch_size=args.valid_batch_size,
                    shuffle=False,
                    num_workers=args.valid_batch_size,
                    # num_workers=1,
                    pin_memory=True,
                    worker_init_fn=worker_init_fn,
                    persistent_workers=True,
                )

                ## 2. setting trainer
                if torch.cuda.is_available():
                    accelerator = "gpu"
                    # TODO: Using more than 1 GPU require syncronization, because I got worse perfomance, than in 1 GPU
                    devices = "auto"  # torch.cuda.device_count()
                    strategy = "ddp_sharded"
                else:
                    accelerator = "cpu"
                    args.device = "cpu"
                    devices = 1
                    strategy = None

                # if alpha == 0.8 and gamma == 2 and coef == (0.6, 0.2, 0.2):
                #     ckpt_path = './save_model/medseg.ckpt'
                # else:
                ckpt_path = None
                # Vis+Text: './save_model/medseg-v5.ckpt'
                print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
                if ckpt_path is not None:
                    model = LanGuideMedSegWrapper.load_from_checkpoint(
                        checkpoint_path=ckpt_path,
                        args=args,
                        tokenizer=tokenizer,
                        max_len=ds_test.max_length,
                        alpha=alpha,
                        gamma=gamma,
                        coef=coef,
                    )
                else:
                    model = LanGuideMedSegWrapper(
                        args,
                        tokenizer=tokenizer,
                        max_len=ds_train.max_length,
                        alpha=alpha,
                        gamma=gamma,
                        coef=coef,
                    )

                ## 1. setting recall function
                model_ckpt = ModelCheckpoint(
                    dirpath=args.model_save_path,
                    filename=args.model_save_filename,
                    monitor="val_loss",
                    save_top_k=1,
                    mode="min",
                    verbose=True,
                )

                early_stopping = EarlyStopping(
                    monitor="val_loss", patience=args.patience, mode="min"
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

                # if alpha != 0.8 and gamma != 2 and coef != (0.6, 0.2, 0.2):
                # # 3. start training
                print("start training")
                trainer.fit(model, dl_train, dl_valid)
                print("done training")

                # --- Запуск тестового прогона ---
                print("start testing")
                if ckpt_path is not None:
                    test_results = trainer.test(
                        model,
                        dataloaders=dl_test,
                        ckpt_path=ckpt_path,  # можно также указать model_ckpt.best_model_path
                    )
                else:
                    test_results = trainer.test(
                        model,
                        dataloaders=dl_test,
                        ckpt_path="best",  # можно также указать model_ckpt.best_model_path
                    )
                print("test results:", test_results)
                # wandb.finish()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
