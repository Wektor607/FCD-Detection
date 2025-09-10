import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import argparse
import random
import numpy as np
import pandas as pd

import torch.multiprocessing
from utils.utils import LesionOversampleSampler
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import KFold
from transformers import AutoTokenizer

import utils.config as config
from utils.data import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper

torch.multiprocessing.set_sharing_strategy("file_system")


SEED = 42
pl.seed_everything(SEED, workers=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Полная детерминированность и отключение TF32 (даёт меньший дрейф на Ampere/Ada)
# torch.use_deterministic_algorithms(True, warn_only=True)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False
# (опц.) для единообразной матричной точности
# torch.set_float32_matmul_precision("high")

torch.use_deterministic_algorithms(True, warn_only=True)  # предупреждать, но не падать
torch.set_float32_matmul_precision("high")  # позволяет TF32, даёт ускорение на A100


# Если вы используете DataLoader с несколькими воркерами:
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cfg():
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument(
        "--config", default="./config/training.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="optional checkpoint to load"
    )

    cli = parser.parse_args()

    if cli.config is None:
        parser.error("--config is required")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.ckpt_path = cli.ckpt_path
    return cfg


if __name__ == "__main__":
    args = get_cfg()

    wandb_logger = WandbLogger(project=args.project_name, log_model=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)

    df = pd.read_csv(args.split_path, sep=",")

    train_ids = df[df.split == "trainval"]["subject_id"].tolist()

    fold_metrics = []
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_index, val_index) in enumerate(kf.split(train_ids)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_fold_ids = [train_ids[i] for i in train_index]
        val_fold_ids_full = [train_ids[i] for i in val_index]

        # Prune controls from validation set and add them in train
        val_fold_ids = [sid for sid in val_fold_ids_full if "FCD" in sid]
        val_fold_ids_controls = [sid for sid in val_fold_ids_full if "_C_" in sid]
        train_fold_ids.extend(val_fold_ids_controls)
        print(f"Train IDs: {len(train_fold_ids)}")
        print(f"Validation IDs: {len(val_fold_ids)}")

        # 1. setting dataset and dataloader
        ds_train = EpilepDataset(
            csv_path=args.train_csv_path,
            tokenizer=tokenizer,
            mode="train",
            meld_path=args.meld_script_path,
            output_dir=args.output_dir,
            feature_path=args.feature_path,
            subject_ids=train_fold_ids,
        )

        ds_valid = EpilepDataset(
            csv_path=args.train_csv_path,
            tokenizer=tokenizer,
            mode="valid",
            meld_path=args.meld_script_path,
            output_dir=args.output_dir,
            feature_path=args.feature_path,
            subject_ids=val_fold_ids,
        )

        hc_set = set(
            [sid for sid in train_fold_ids if sid.split("_")[3].startswith("C")]
        )
        labels = [0 if sid in hc_set else 1 for sid in ds_train.subject_ids]

        sampler = LesionOversampleSampler(labels, seed=SEED)

        dl_train = DataLoader(
            ds_train,
            batch_size=args.train_batch_size,
            sampler=sampler,
            num_workers=args.train_batch_size,  # 1,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

        dl_valid = DataLoader(
            ds_valid,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=args.valid_batch_size,  # 1,
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

        ckpt_path = args.ckpt_path  # None
        print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
        if ckpt_path is not None:
            model = LanGuideMedSegWrapper.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                args=args,
            )
        else:
            model = LanGuideMedSegWrapper(
                args,
            )

        if ckpt_path is None:
            filename = f"{args.model_save_filename}_fold{fold + 1}"
        else:
            filename = os.path.splitext(os.path.basename(ckpt_path))[0]

        ## 1. setting recall function
        model_ckpt = ModelCheckpoint(
            dirpath=args.model_save_path,
            filename=filename,
            monitor="val_dice",  # 'val_loss',
            save_top_k=1,
            mode="max",  # 'min',
            verbose=True,
        )

        early_stopping = EarlyStopping(
            monitor="val_dice",  # "val_loss",
            patience=args.patience,
            mode="max",  # 'min',
        )

        trainer = pl.Trainer(
            logger=wandb_logger,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[model_ckpt, early_stopping],
            enable_progress_bar=True,
        )

        # 3. start training
        print("start training")
        trainer.fit(model, dl_train, dl_valid)  # , ckpt_path=ckpt_path)
        print("done training")
