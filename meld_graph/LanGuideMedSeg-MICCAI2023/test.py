import os
import sys

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
from utils.utils import summarize_ci
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from utils.data import EpilepDataset
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
from engine.wrapper import LanGuideMedSegWrapper
from meld_graph.paths import MELD_DATA_PATH
import utils.config as config

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

    eva, cohort = config.inference_config()
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
        cohort=cohort
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

        metrics_history = {name: [] for name in test_metrics.keys()}

        dice_metric = []
        iou_metric = []
        ppv_metric = []
        
        for batch in tqdm(dl_test):
            subject_ids = batch["subject_id"]  # list[str]
            y = batch["roi"]  # torch.Tensor
            B, H, N = y.shape

            for b, sid in enumerate(subject_ids):
                gt = y[b]  # [2, N]
                gt_cortex = gt[:, cohort.cortex_mask]  # [2, N_cortex]
                gt_flat = gt_cortex.reshape(-1)  # [2*N_cortex]
                
                nii_path = os.path.join(
                    MELD_DATA_PATH,
                    f"preprocessed/meld_files/{sid}/features",
                    "result.npz",
                )
                with np.load(nii_path, allow_pickle=False) as npz:
                    arr = npz["result"].astype("float32")

                mini = {sid: {"result": arr.copy()}}
                out = eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
                probs_flat = out[sid]["cluster_thresholded"]           # (2*N_cortex,)
               
                mask = torch.as_tensor(np.array(probs_flat > 0)).long()
                labels = torch.as_tensor(np.array(gt_flat).astype(bool)).long()
                dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)

                tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
                iou = tp / (tp + fp + fn + 1e-8)
                ppv = tp / (tp + fp + 1e-8)

                dice_metric.append(dices[1])
                ppv_metric.append(ppv)
                iou_metric.append(iou)
                print(f"[{sid}] Dice lesional={dices[1]:.3f}, IoU={iou:.3f}, PPV={ppv:.3f}, "
                    f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        print("\n=== Сводная статистика по всем субъектам ===")
        d_med, d_lo, d_hi = summarize_ci(dice_metric)
        p_med, p_lo, p_hi = summarize_ci(ppv_metric)
        i_med, i_lo, i_hi = summarize_ci(iou_metric)

        print("\n=== OVERALL TEST METRICS ===")
        print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
        print(f"PPV  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
        print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
    else:
        ckpt_path = args.ckpt_path  # None
        print(f"[INFO] Loading model from checkpoint: {ckpt_path}")
        if ckpt_path is not None:
            model = LanGuideMedSegWrapper.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                args=args,
                eva=eva
            )
        else:
            model = LanGuideMedSegWrapper(
                args,
                eva=eva
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
