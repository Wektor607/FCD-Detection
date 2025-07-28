import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# import wandb
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
import pandas as pd
import csv
from meld_graph.paths import (FS_SUBJECTS_PATH, 
                              MELD_DATA_PATH,
                                )
from meld_graph.meld_cohort import MeldCohort

import random
import numpy as np
import torch
from scipy.ndimage import label
from sklearn.model_selection import train_test_split

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
                    # MELD_DATA_PATH, f"input/{sid}/anat/features", "result.npz",
                    MELD_DATA_PATH, f"preprocessed/{sid}/features", "result.npz",
                )
                arr = np.load(nii_path)['result']
                pred = torch.from_numpy(arr.astype("float32"))  # [2, N]

                pred_label = pred.argmax(dim=0) 

                # 3) бинаризуем: 1 — потенциальный объект, 0 — шум
                binary_pred = (pred_label == 1).cpu().numpy()  # ndarray [N_cortex]

                # 4) находим связные компоненты (кластеризацию)
                clusters, num_clusters = label(binary_pred)
                print(f"{sid} num_clusters = {num_clusters}")
                metrics_history.setdefault('num_clusters', []).append(num_clusters)

                # 6) качество по каждому кластеру
                for cl_id in range(1, num_clusters + 1):
                    cluster_mask = (clusters == cl_id)
                    if cluster_mask.sum() == 0:
                        continue
                    for name, metric in test_metrics.items():
                        metric.update(
                            torch.from_numpy(cluster_mask).unsqueeze(0),   # [1, n_pts_total]
                            torch.from_numpy(gt_flat.reshape(2, -1)[1, :][cluster_mask]).unsqueeze(0)
                        )
                        val = metric.compute().item()
                        metric.reset()
                        print(f"{sid} {name} кластер {cl_id} = {val:.4f}")
                        key = f"{name}_cluster_{cl_id}"
                        metrics_history.setdefault(key, []).append(val)

                print('---')
                
                # 3) обновляем метрики, сохраняем в history
                for name, metric in test_metrics.items():
                    metric.update(pred, gt_flat.long())
                    val = metric.compute().item()
                    metric.reset()

                    print(f"{sid} {name} = {val:.4f}")
                    metrics_history[name].append(val)

                print('---')

        # 4) после всех субъектов считаем медиану и 2.5–97.5 перцентили

        print("\n=== Сводная статистика по всем субъектам ===")
        for name, values in metrics_history.items():
            vals = np.array(values)
            med = np.median(vals)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            print(f"{name:>5}: median={med:.4f}  [{lo:.4f}–{hi:.4f}]")

    # sys.exit()
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

def _filter_existing(data: pd.DataFrame, ids: list[str]) -> list[str]:
    """Оставить только те ids, что есть в data.index."""
    valid_ids = [sid for sid in ids if sid in data.index]
    missing   = sorted(set(ids) - set(valid_ids))
    if missing:
        print(f"⚠️ Эти участники отсутствуют и будут пропущены: {missing}")
    return valid_ids

if __name__ == '__main__':
    args = get_parser()

    with open(args.train_csv_path, 'r') as f:
        data = pd.read_csv(
            f,
            sep=',',
            engine='python',            # нужен движок Python, чтобы поддерживался escapechar
            quoting=csv.QUOTE_NONE,     # не искать кавычки
            escapechar='\\'             # '\' перед ',' будет означать «не разделитель»
        )

    # 2) вытаскиваем sub-ID
    data['sub'] = data['DATA_PATH'].apply(
        lambda p: os.path.basename(p).split('_')[0] 
                if isinstance(p, str) else None
    )

    # 3) задаём sub как индекс, чтобы удобнее было выбирать
    data = data.set_index('sub')

    df = pd.read_csv('../data/input/ds004199/participants.tsv', sep='\t')

    test_all  = df[df.split=='test']['participant_id'].tolist()

    test_all = _filter_existing(data, test_all)

    val_ids, test_ids = train_test_split(test_all, test_size=0.3, random_state=42, shuffle=True)
    
    ckpt_path = './save_model/medseg-v1.ckpt'

    # --- Запуск тестового прогона ---    
    print('start testing')
    # test_model(model_ckpt.best_model_path, test_ids)
    test_model(ckpt_path, test_ids)

