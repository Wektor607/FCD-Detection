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
import subprocess
from torch.utils.data import DataLoader

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

def mgh_cleaner():
    cleanup_cmd = [
        "find",
        os.path.join(FS_SUBJECTS_PATH),
        "-type", "f",
        "-path", "*/xhemi/classifier/*",
        "-delete"
    ]
    subprocess.run(cleanup_cmd, check=True)


def test_model(model_ckpt=None, test_ids=None):
    mgh_cleaner()

    args = get_parser()
    
    check = args.meld_check

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)

     # --- Подготовка тестовой выборки и DataLoader ---
    ds_test = EpilepDataset(
        csv_path=args.test_csv_path,             # новый CSV для теста
        root_path=args.test_root_path,           # путь к тестовым данным
        tokenizer=tokenizer,
        mode='test',
        meld_path=args.meld_script_path,
        output_dir=args.output_dir,
        feature_path=args.feature_path,
        subject_ids=test_ids
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=args.valid_batch_size,
        # num_workers=1,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    
    # Checking MELD results on test data
    if check:
        test_metrics = nn.ModuleDict({
            "acc":  Accuracy(task="binary"),
            "dice": Dice(),
            "ppv":  Precision(task="binary"),
            "MIoU": BinaryJaccardIndex()
        })
        cohort = MeldCohort()
        
        cortex_mask = torch.from_numpy(cohort.cortex_mask)   # shape [N], dtype=torch.bool

        metrics_history = { name: [] for name in test_metrics.keys() }

        for batch in dl_test:
            (subject_ids, text), y = batch
            B, H, N = y.shape

            for b, sid in enumerate(subject_ids):
                # извлекаем GT для кортикальных вершин
                gt = y[b]                        # [2, N]
                gt_cortex = gt[:, cortex_mask]   # [2, N_cortex]
                gt_flat = gt_cortex.flatten()    # [2*N_cortex]

                # загружаем ваши предсказания
                nii_path = os.path.join(
                    MELD_DATA_PATH, f"input/{sid}/anat/features", "result.npz",
                )
                arr = np.load(nii_path)['result']
                pred = torch.from_numpy(arr.astype("float32"))  # [2, N]

                # 3) обновляем метрики, сохраняем в history
                for name, metric in test_metrics.items():
                    metric.update(pred, gt_flat.long())
                    val = metric.compute().item()
                    metric.reset()

                    print(f"{sid} {name} = {val:.4f}")
                    metrics_history[name].append(val)

                print('---')

        # 4) после всех субъектов считаем медиану и 2.5–97.5 перцентили
        import numpy as np

        print("\n=== Сводная статистика по всем субъектам ===")
        for name, values in metrics_history.items():
            vals = np.array(values)
            med = np.median(vals)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            print(f"{name:>5}: median={med:.4f}  [{lo:.4f}–{hi:.4f}]")

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

    if model_ckpt is not None:
        ckpt_path = model_ckpt
    else:
        ckpt_path = './save_model/medseg-v1.ckpt'
    print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
    model = LanGuideMedSegWrapper.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        args=args,
        tokenizer=tokenizer,
        max_len = ds_test.max_length
    )

    trainer = pl.Trainer(
                        accelerator=accelerator, 
                        devices=devices,
                        enable_progress_bar=True,
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


if __name__ == "__main__":
    test_model()