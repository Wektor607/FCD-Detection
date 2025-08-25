import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import wandb
import utils.config as config

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import subprocess

from utils.data import EpilepDataset

# from engine.wrapper_new_loss import LanGuideMedSegWrapper
from engine.wrapper import LanGuideMedSegWrapper

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
import argparse

from meld_graph.paths import (
    FS_SUBJECTS_PATH,
)
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


def mgh_cleaner():
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


def _filter_existing(data: pd.DataFrame, ids: list[str]) -> list[str]:
    """Оставить только те ids, что есть в data.index."""
    valid_ids = [sid for sid in ids if sid in data.index]
    missing = sorted(set(ids) - set(valid_ids))
    if missing:
        print(f"⚠️ Эти участники отсутствуют и будут пропущены: {missing}")
    return valid_ids


import random
from torch.utils.data import Sampler


class LesionOversampleSampler(Sampler):
    """
    Сэмплер, который берёт ВСЕ healthy-примеры ровно по одному разу,
    а lesion-примеры — с replacement, чтобы заполнить всю эпоху.
    """

    def __init__(self, labels, seed=42):
        self.labels = labels
        random.seed(seed)
        # индексы здоровых и lesion
        self.hc_idx = [i for i, l in enumerate(labels) if l == 0]
        self.les_idx = [i for i, l in enumerate(labels) if l == 1]
        # хотим ровно len(labels) выборок за эпoху
        self.epoch_size = len(labels)

    def __iter__(self):
        # начинаем с всех hc-индексов
        idxs = self.hc_idx.copy()
        # сколько нужно докинуть lesion'ов
        n_les_to_sample = self.epoch_size - len(idxs)
        # добавляем lesion с replacement
        idxs += random.choices(self.les_idx, k=n_les_to_sample)
        # перемешиваем всю эпоху
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return self.epoch_size


if __name__ == "__main__":
    args = get_parser()

    check = args.meld_check
    wandb_logger = WandbLogger(project=args.project_name, log_model=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)

    with open(args.train_csv_path, "r") as f:
        data = pd.read_csv(
            f,
            sep=",",
            engine="python",  # нужен движок Python, чтобы поддерживался escapechar
            quoting=csv.QUOTE_NONE,  # не искать кавычки
            escapechar="\\",  # '\' перед ',' будет означать «не разделитель»
        )

    # 2) вытаскиваем sub-ID
    data["sub"] = data["DATA_PATH"].apply(
        lambda p: os.path.basename(p).split("_")[0] if isinstance(p, str) else None
    )

    # 3) задаём sub как индекс, чтобы удобнее было выбирать
    data = data.set_index("sub")

    df = pd.read_csv("../data/input/ds004199/participants.tsv", sep="\t")

    train_ids = df[df.split == "train"]["participant_id"].tolist()
    # train_ids = df[(df.split == 'train') & (df.group != 'hc')]['participant_id'].tolist()
    print(train_ids)
    test_all = df[df.split == "test"]["participant_id"].tolist()

    train_ids = _filter_existing(data, train_ids)
    test_all = _filter_existing(data, test_all)

    val_ids, test_ids = train_test_split(
        test_all, test_size=0.3, random_state=42, shuffle=True
    )
    print(val_ids)
    print(len(train_ids), len(val_ids), len(test_ids))

    ds_train = EpilepDataset(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        tokenizer=tokenizer,
        mode="train",
        meld_path=args.meld_script_path,
        output_dir=args.output_dir,
        feature_path=args.feature_path,
        subject_ids=train_ids,
        aug_flag=True,
    )  # CHANGE

    ds_valid = EpilepDataset(
        csv_path=args.train_csv_path,
        root_path=args.train_root_path,
        tokenizer=tokenizer,
        mode="valid",
        meld_path=args.meld_script_path,
        output_dir=args.output_dir,
        feature_path=args.feature_path,
        subject_ids=val_ids,
    )


    # # Split fcd and hc patients near equal
    hc_set = set(df[df.group == "hc"]["participant_id"])
    subject_ids = ds_train.subject_ids  # или как вы их храните
    labels = [0 if sid in hc_set else 1 for sid in subject_ids]
    # class_counts = np.bincount(labels)
    # print(f"[DEBUG] Class counts: {class_counts}")
    # weights = 1. / class_counts
    # sample_weights = [weights[l] for l in labels]
    # print(sample_weights)
    # sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    sampler = LesionOversampleSampler(labels, seed=SEED)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.train_batch_size,
        # shuffle=True,
        sampler=sampler,
        num_workers=2,  # 0
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        # persistent_workers=True
    )

    # for sample in ds_train:
    #     print('Hi')
    #     print(sample)
    #     sys.exit()
    dl_valid = DataLoader(
        ds_valid,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=2,  # 0
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        # persistent_workers=True
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
    ckpt_path = None  #'./save_model/medseg-v2.ckpt'
    # Vis+Text: './save_model/medseg-v5.ckpt'
    print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
    if ckpt_path is not None:
        model = LanGuideMedSegWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            args=args,
            tokenizer=tokenizer,
            max_len=ds_valid.max_length,
        )
    else:
        model = LanGuideMedSegWrapper(
            args,
            tokenizer=tokenizer,
            max_len=ds_train.max_length,
        )

    ## 1. setting recall function
    model_ckpt = ModelCheckpoint(
        dirpath=args.model_save_path,
        filename=args.model_save_filename,
        monitor="val_dice",  #'val_loss',
        save_top_k=1,
        mode="max",  #'min',
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_dice",  #'val_loss',
        patience=args.patience,
        mode="max",  #'min',
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        # strategy=strategy, ###################
        callbacks=[model_ckpt, early_stopping],
        enable_progress_bar=True,
        log_every_n_steps=18,
    )

    # 3. start training
    print("start training")
    trainer.fit(model, dl_train, dl_valid)
    print("done training")

    # --- Запуск тестового прогона ---
    print("start testing")
    test_model(model_ckpt.best_model_path, test_ids)
    # test_model(ckpt_path, test_ids)
    wandb.finish()
