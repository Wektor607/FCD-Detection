import os
import torch
import argparse
import subprocess
import random
import csv
import numpy as np
import pandas as pd

import torch.multiprocessing
from torch.utils.data import DataLoader, Sampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from sklearn.model_selection import KFold
from transformers import AutoTokenizer

import wandb
import utils.config as config
from utils.data import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper
from meld_graph.paths import FS_SUBJECTS_PATH

# теперь можно вызвать
torch.multiprocessing.set_sharing_strategy("file_system")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Полная детерминированность и отключение TF32 (даёт меньший дрейф на Ampere/Ada)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# (опц.) для единообразной матричной точности
# torch.set_float32_matmul_precision("high")


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


class LesionOversampleSampler(Sampler):
    """
    Сэмплер, который берёт ВСЕ healthy-примеры ровно по одному разу,
    а lesion-примеры — с replacement, чтобы заполнить всю эпоху.
    """

    def __init__(self, labels, seed=42):
        self.labels = labels
        random.seed(seed)
        # индексы здоровых и lesion
        self.hc_idx = [i for i, label in enumerate(labels) if label == 0]
        self.les_idx = [i for i, label in enumerate(labels) if label == 1]
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


def iter_labels_from_batch(batch):
    """
    Приведи к своей структуре: верни тензор меток на финальном уровне.
    Должен быть bool/0-1, shape [B, H, N] или [B, N] или [B*N].
    """
    # пример — подстрой под свой batch
    # из твоего пайплайна часто есть что-то вроде batch['labels'] / batch['y'] / labels_pooled[...]
    if isinstance(batch, dict):
        for k in ["labels", "y", "target", "gt"]:
            if k in batch:
                return batch[k]
    # если batch = (inputs, labels) или что-то похожее
    if isinstance(batch, (list, tuple)):
        for item in batch:
            if torch.is_tensor(item):
                # эвристика: бинарные метки
                if item.dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
                    return item
    raise RuntimeError(
        "Не удалось найти тензор меток в batch — поправь iter_labels_from_batch()."
    )


def compute_class_prior(dls, max_batches=None):
    total = 0
    pos = 0
    seen = 0
    for dl in dls:
        for i, batch in enumerate(dl):
            y = iter_labels_from_batch(batch)  # [B, H, N] / [B,N] / [B*N]
            y = y.detach()
            # приведём к [B*H*N]
            y_flat = y.view(-1).to(torch.float32)
            pos += y_flat.sum().item()
            total += y_flat.numel()
            seen += 1
            print(pos, total)
            if max_batches is not None and seen >= max_batches:
                break
    p1 = pos / (total + 1e-12)
    p0 = 1.0 - p1
    return p0, p1, int(total), int(pos)


if __name__ == "__main__":
    args = get_parser()

    check = args.meld_check
    wandb_logger = WandbLogger(project=args.project_name, log_model=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)

    df = pd.read_csv(args.split_path, sep=",")

    train_ids = df[df.split == "trainval"]["subject_id"].tolist()
    # test_ids  = df[df.split=='test']['subject_id'].tolist()
    test_ids = df[(df["split"] == "test") & (df["subject_id"].str.contains("FCD"))][
        "subject_id"
    ].tolist()

    fold_metrics = []
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_index, val_index) in enumerate(kf.split(train_ids)):
        # if fold+1 < 4:
        #     continue
        print(f"Fold {fold + 1}/{n_splits}")

        train_fold_ids = [train_ids[i] for i in train_index]
        val_fold_ids = [train_ids[i] for i in val_index]

        print(f"Train IDs: {len(train_fold_ids)}")
        print(f"Validation IDs: {len(val_fold_ids)}")

        # 1. setting dataset and dataloader
        ds_train = EpilepDataset(
            csv_path=args.train_csv_path,
            root_path=args.train_root_path,
            tokenizer=tokenizer,
            mode="train",
            meld_path=args.meld_script_path,
            output_dir=args.output_dir,
            feature_path=args.feature_path,
            subject_ids=train_fold_ids,
            aug_flag=True,
        )

        ds_valid = EpilepDataset(
            csv_path=args.train_csv_path,
            root_path=args.train_root_path,
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
            sampler=sampler,  # shuffle=True,
            num_workers=0,  # args.train_batch_size,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # persistent_workers=True
        )

        dl_valid = DataLoader(
            ds_valid,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=0,  # args.valid_batch_size,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            # persistent_workers=True
        )

        # Считаем на train (+ при желании val)
        # p0, p1, tot, pos = compute_class_prior([dl_train, dl_valid], max_batches=None)  # или только [dl_train]
        # print(f"[PRIOR] total={tot}, positives={pos}, p1={p1:.6f}, p0={p0:.6f}")
        # sys.exit(0)
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

        ckpt_path = None
        print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
        if ckpt_path is not None:
            model = LanGuideMedSegWrapper.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                args=args,
                tokenizer=tokenizer,
                max_len=ds_valid.max_length,
                alpha=0.75,
                gamma=2,
            )
        else:
            model = LanGuideMedSegWrapper(
                args,
                tokenizer=tokenizer,
                max_len=ds_train.max_length,
                alpha=0.75,
                gamma=2,
            )

        ## 1. setting recall function
        model_ckpt = ModelCheckpoint(
            dirpath=args.model_save_path,
            filename=f"{args.model_save_filename}_fold{fold + 1}",
            monitor="val_dice",  # 'val_loss',
            save_top_k=1,
            mode="max",  # 'min',
            verbose=True,
        )

        early_stopping = EarlyStopping(
            monitor="val_dice",  # 'val_loss',
            patience=args.patience,
            mode="max",  # 'min',
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
            # precision=32,                  # без AMP
            # gradient_clip_val=1.0,         # сглаживает всплески
            # accumulate_grad_batches=2,     # сглаживает шум (по желанию)
        )

        # 3. start training
        print("start training")
        trainer.fit(model, dl_train, dl_valid)  # , ckpt_path=ckpt_path)
        print("done training")

        # --- Запуск тестового прогона ---
        print("start testing")
        # mgh_cleaner()

        # --- Подготовка тестовой выборки и DataLoader ---
        ds_test = EpilepDataset(
            csv_path=args.test_csv_path,  # новый CSV для теста
            root_path=args.test_root_path,  # путь к тестовым данным
            tokenizer=tokenizer,
            mode="test",
            meld_path=args.meld_script_path,
            output_dir=args.output_dir,
            feature_path=args.feature_path,
            subject_ids=test_ids,
        )

        dl_test = DataLoader(
            ds_test,
            batch_size=args.valid_batch_size,
            shuffle=False,
            num_workers=2,
            # num_workers=1,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )

        # 3) Evaluate on TEST
        model.eval()
        test_results = trainer.test(
            model,
            dataloaders=dl_test,
            ckpt_path=ckpt_path,
            verbose=True,
        )
        print("=== TEST metrics ===")
        print(test_results)
        metrics = test_results[0]  # обычно там только один элемент

        # Добавляем номер фолда к метрикам
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

        csv_output_path = os.path.join(args.output_dir, "kfold_metrics.csv")
        keys = fold_metrics[0].keys()
        with open(csv_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(fold_metrics)

        print(f"📊 Обновлён CSV с метриками после фолда {fold + 1}: {csv_output_path}")
        wandb.finish()
        break
