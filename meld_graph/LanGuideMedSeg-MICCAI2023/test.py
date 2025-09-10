import os

from tqdm import tqdm
import torch
import argparse
import random
import numpy as np
import pandas as pd

import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryJaccardIndex, Precision, BinaryF1Score

import torch.multiprocessing
from utils.utils import summarize_ci, compute_adaptive_threshold
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from transformers import AutoTokenizer

import utils.config as config
from utils.data import EpilepDataset
from engine.wrapper import LanGuideMedSegWrapper
from meld_graph.meld_cohort import MeldCohort
from meld_graph.paths import MELD_DATA_PATH

# теперь можно вызвать
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
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# (опц.) для единообразной матричной точности
# torch.set_float32_matmul_precision("high")


# Если вы используете DataLoader с несколькими воркерами:
def worker_init_fn(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def get_cfg():
    parser = argparse.ArgumentParser(
        description="Language-guide Medical Image Segmentation"
    )
    parser.add_argument(
        "--config", default="./config/training.yaml", type=str, help="config file"
    )
    parser.add_argument(
        "--meld_check", action="store_true", help="enable MELD test check mode"
    )
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="optional checkpoint to load"
    )

    cli = parser.parse_args()

    if cli.config is None:
        parser.error("--config is required")

    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.meld_check = cli.meld_check
    cfg.ckpt_path = cli.ckpt_path
    return cfg


if __name__ == "__main__":
    args = get_cfg()

    wandb_logger = WandbLogger(project=args.project_name, log_model=True)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)

    df = pd.read_csv(args.split_path, sep=",")

    test_ids = df[(df["split"] == "test") & (df["subject_id"].str.contains("FCD"))][
        "subject_id"
    ].tolist()

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

    print("start testing")

    # --- Подготовка тестовой выборки и DataLoader ---
    ds_test = EpilepDataset(
        csv_path=args.test_csv_path,  # новый CSV для теста
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
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
    )

    # Checking MELD results on test data
    if args.meld_check:
        num_non_predict_samples = 0
        test_metrics = nn.ModuleDict(
            {
                "acc": Accuracy(task="binary"),
                "dice": BinaryF1Score(),  # Dice(),
                "ppv": Precision(task="binary"),
                "IoU": BinaryJaccardIndex(),
            }
        )
        cohort = MeldCohort()

        cortex_mask = torch.from_numpy(
            cohort.cortex_mask.astype(bool)
        )  # shape [N], dtype=torch.bool

        metrics_history = {name: [] for name in test_metrics.keys()}

        for batch in tqdm(dl_test):
            subject_ids = batch["subject_id"]  # list[str]
            y = batch["roi"]  # torch.Tensor
            B, H, N = y.shape

            for b, sid in enumerate(subject_ids):
                # извлекаем GT для кортикальных вершин
                gt = y[b]  # [2, N]
                gt_cortex = gt[:, cortex_mask]  # [2, N_cortex]
                gt_flat = gt_cortex.reshape(-1).long()  # [2*N_cortex]
                # загружаем ваши предсказания
                nii_path = os.path.join(
                    MELD_DATA_PATH,
                    f"preprocessed/meld_files/{sid}/features",
                    "result.npz",
                )
                with np.load(nii_path, allow_pickle=False) as npz:
                    arr = npz["result"].astype("float32")
                pred = torch.from_numpy(arr.astype("float32")).reshape(H, -1)  # [2, N]

                probs_bin = torch.zeros_like(pred, dtype=torch.float32)

                for h in range(H):  # 2 полушария
                    pv = pred[h]  # [V_cortex], torch
                    pv_np = pv.detach().cpu().numpy()
                    th = compute_adaptive_threshold(pv_np)
                    probs_bin[h] = (pv >= th).float()

                probs_flat = probs_bin.view(-1)
                # pred_label = pred.argmax(dim=0)

                # # 3) бинаризуем: 1 — потенциальный объект, 0 — шум
                # binary_pred = (pred_label == 1).cpu().numpy()  # ndarray [N_cortex]

                # # 4) находим связные компоненты (кластеризацию)
                # clusters, num_clusters = label(binary_pred)
                # print(f"{sid} num_clusters = {num_clusters}")
                # metrics_history.setdefault("num_clusters", []).append(num_clusters)

                # # 6) качество по каждому кластеру
                # for cl_id in range(1, num_clusters + 1):
                #     cluster_mask = clusters == cl_id
                #     if cluster_mask.sum() == 0:
                #         continue
                #     for name, metric in test_metrics.items():
                #         metric.update(
                #             torch.from_numpy(cluster_mask).unsqueeze(
                #                 0
                #             ),  # [1, n_pts_total]
                #             torch.from_numpy(
                #                 gt_flat.reshape(2, -1)[1, :][cluster_mask]
                #             ).unsqueeze(0),
                #         )
                #         val = metric.compute().item()
                #         metric.reset()
                #         print(f"{sid} {name} кластер {cl_id} = {val:.4f}")
                #         key = f"{name}_cluster_{cl_id}"
                #         metrics_history.setdefault(key, []).append(val)

                # print("---")

                # 3) обновляем метрики, сохраняем в history
                for name, metric in test_metrics.items():
                    metric.update(probs_flat, gt_flat)
                    val = metric.compute().item()
                    metric.reset()
                    if name == "dice" and val == 0.0:
                        num_non_predict_samples += 1
                        print(num_non_predict_samples)
                    # print(f"{sid} {name} = {val:.4f}")
                    metrics_history[name].append(val)

                # print("---")

        # 4) после всех субъектов считаем медиану и 2.5–97.5 перцентили

        print("\n=== Сводная статистика по всем субъектам ===")
        for name, values in metrics_history.items():
            med, lo, hi = summarize_ci(values)

            print(f"{name:>5}: median={med:.4f}  [95% CI {lo:.4f}–{hi:.4f}]")
    else:
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

        # 3) Evaluate on TEST
        model.eval()
        if ckpt_path is None:
            filename = args.model_save_filename
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

        test_results = trainer.test(
            model,
            dataloaders=dl_test,
            ckpt_path=ckpt_path,
            verbose=True,
        )
        print("=== TEST metrics ===")
        print(test_results)

    # metrics = test_results[0]  # обычно там только один элемент

    # # Добавляем номер фолда к метрикам
    # metrics["fold"] = fold + 1
    # fold_metrics.append(metrics)

    # csv_output_path = os.path.join(args.output_dir, "kfold_metrics.csv")
    # keys = fold_metrics[0].keys()
    # with open(csv_output_path, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=keys)
    #     writer.writeheader()
    #     writer.writerows(fold_metrics)

    # print(f"📊 Обновлён CSV с метриками после фолда {fold + 1}: {csv_output_path}")
    # wandb.finish()
    # break
