from utils.model import LanGuideMedSeg
from engine.comb_loss import CombinedSegmentationLoss
from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss
from torchmetrics import Accuracy, Dice

from meld_graph.paths import (MELD_DATA_PATH, DEFAULT_HDF5_FILE_ROOT)

import os
from meld_graph.meld_cohort import MeldCohort
from scripts.manage_results.register_back_to_xhemi import register_subject_to_xhemi
from torchmetrics.classification import BinaryJaccardIndex, Precision, Specificity
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import torch.nn.functional as F
import pandas as pd
import sys
import numpy as np
import datetime
import nibabel as nb

class LanGuideMedSegWrapper(pl.LightningModule):
    def __init__(self, args, tokenizer, max_len, 
                 alpha=0.85, gamma=2, ):
                #  coef=(0.1, 0.1, 0.8)):
        super(LanGuideMedSegWrapper, self).__init__()
        self.save_hyperparameters(args)
        # ---- инициализируем саму модель ----
        self.model = LanGuideMedSeg(
            args.bert_type,
            args.meld_script_path,
            args.feature_path,
            args.output_dir,
            args.project_dim,
            args.device,
            args.warmup_epochs,
            tokenizer,
            max_len

        )
        
        self.alpha = alpha
        self.gamma = gamma
        # self.foc_coef, self.fp_coef, self.dice_coef = coef
        self.loss_fn = CombinedSegmentationLoss(alpha=alpha, gamma=gamma)

        self.train_metrics = nn.ModuleDict({
            "acc"   : Accuracy(task="binary"),
            "spec"  : Specificity(task="binary"),
            "dice"  : Dice(),
            "ppv"   : Precision(task="binary"),
            "IoU"   : BinaryJaccardIndex()
        })
        self.val_metrics   = deepcopy(self.train_metrics)
        self.test_metrics  = deepcopy(self.train_metrics)

        self.dice = Dice().to(args.device)                 
        self.ppv  = Precision(task="binary").to(args.device)
        self.iou  = BinaryJaccardIndex().to(args.device)
        self.acc  = Accuracy(task="binary").to(args.device)

        self.dice_fn = DiceLoss(include_background=False, sigmoid=True)

        self.c = MeldCohort()

        self.history = {}
        self.test_dice_scores = []
        self.test_ppv_scores  = []
        self.test_iou_scores  = []
        self.test_acc_scores  = []

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-2)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr * 10, # <- better than 0.3 and 0.003
            total_steps=self.trainer.estimated_stepping_batches, # 600
            pct_start=0.1, # 0.1, 0.3
            anneal_strategy='cos',
        )
        # return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

        # TRY THIS
        return {"optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",      # важный пункт!
                    "frequency": 1,
                    "name": "one_cycle_lr"
                }
            }

    def forward(self, x):
        return self.model(x)
    
    # def compute_loss(self, logits, target):
    #     prob = torch.sigmoid(logits)
    #     focal = sigmoid_focal_loss(logits, target, 
    #                                gamma=self.gamma, 
    #                                alpha=self.alpha, 
    #                                reduction="mean")
    #     dice = self.dice_fn(logits, target)

    #     tn = ((1 - prob) * (1 - target)).sum()
    #     fp = (prob * (1 - target)).sum()
    #     fp_loss = 1 - (tn / (tn + fp + 1e-6))

    #     return self.foc_coef * focal + self.fp_coef * fp_loss + self.dice_coef * dice

    def shared_step(self, batch, batch_idx, stage: str):
        """
        General code for train/val/test.
        batch = (x, y, ...), where
            x: input features (→ self.model(x) yields [B, n_nodes])
            y: binary labels {0,1} [B, n_nodes]
        ignore the rest of the batch elements
        stage: "train" | "val" | "test"
        """
        x, y = batch
        subject_ids, text = x
        y = y.float()

        logits = self(x).view_as(y)

        logits  = logits[:, :, self.c.cortex_mask]
        y       = y[:, :, self.c.cortex_mask]

        # loss = self.compute_loss(logits, y)
        loss = self.loss_fn(logits, y)

        # Remain only cortex vertices
        probs   = torch.sigmoid(logits)
        
        probs_flat = probs.view(-1)
        y          = y.long()
        y_flat     = y.view(-1)

        if stage == "test":
            for name, m in self.test_metrics.items():
                m.update(probs_flat, y_flat)

        else:
            metrics = self.train_metrics if stage=="train" else self.val_metrics
            for name, m in metrics.items():
                m.update(probs_flat, y_flat)
                val = m.compute()
                if name == "spec": 
                    val = 1 - val
                self.log(name, val, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
                
        if stage == 'test':
            for sid, pred, tgt in zip(subject_ids, probs, y):
                dice_i = self.dice(pred, tgt).item()
                ppv_i  = self.ppv (pred, tgt).item()
                iou_i  = self.iou (pred, tgt).item()
                acc_i  = self.acc (pred, tgt).item()

                # накапливаем для общего сводного отчёта
                self.test_dice_scores.append(dice_i)
                self.test_ppv_scores .append(ppv_i)
                self.test_iou_scores .append(iou_i)
                self.test_acc_scores .append(acc_i)
                
                print(f"[TEST sample {sid!r}] Dice={dice_i:.3f}, PPV={ppv_i:.3f}, "
                f"IoU={iou_i:.3f}, Acc={acc_i:.3f}")

            subjects_dir = "/home/s17gmikh/FCD-Detection/meld_graph/data/output/fs_outputs"
            predictions_output_dir = os.path.join(MELD_DATA_PATH,'output','predictions_reports')
            for sid, p in zip(subject_ids, logits):
                # create classifier directory if not exist
                classifier_dir = os.path.join(subjects_dir, sid, "xhemi", "classifier")
                if not os.path.isdir(classifier_dir):
                    os.mkdir(classifier_dir)

                predictions = torch.sigmoid(p)
                
                for idx, hemi in enumerate(["lh", "rh"]):
                    # prediction_h = predictions[idx][self.c.cortex_mask].detach().cpu().numpy()
                    prediction_h = predictions[idx].detach().cpu().numpy()
                    overlay = np.zeros_like(self.c.cortex_mask, dtype=np.float32)
                    overlay[self.c.cortex_mask] = prediction_h
                    try:
                        demo = nb.load(os.path.join(subjects_dir, sid, "xhemi", "surf_meld", f"{hemi}.on_lh.thickness.mgh"))
                    except:
                        print(f'Could not load {os.path.join(subjects_dir, sid, "xhemi", "surf_meld", f"{hemi}.on_lh.thickness.mgh")} ', sid, 'ERROR') 
                        return False   
                    filename = os.path.join(subjects_dir, sid, "xhemi", "classifier", f"{hemi}.prediction.mgh")
                    self.save_mgh(filename, overlay, demo)
                    print(filename)
                                
                register_subject_to_xhemi(subject_id=sid, subjects_dir=subjects_dir, output_dir=predictions_output_dir, verbose=False, plus_name=sid)

                nii_path = os.path.join(predictions_output_dir, sid, "predictions", f"prediction_{sid}.nii.gz")
                if not os.path.exists(nii_path):
                    raise FileNotFoundError(f"Не найден файл {nii_path}")

        return loss

        # if batch_idx == 0 and stage == "train":
        #     print(
        #         f"\n[{stage.upper()} step {self.global_step}] "
        #         f"patch-loss(avg) = {loss.item():.4f} "
        #     )
            
        #     # 1) Average probability over all pixels:
        #     mean_prob = preds_prob.mean().item()

        #     # 2) Share of predicted "pluses" at threshold 0.5:
        #     pred_pos_frac = (preds_prob > 0.5).float().mean().item()

        #     # 3) Compare with share of real "pluses":
        #     true_pos_frac = (y > 0.5).float().mean().item()

        #     print(f"mean_prob={mean_prob:.4f}, pred_pos_frac={pred_pos_frac:.4f}, true_pos_frac={true_pos_frac:.4f}")

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="test")
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss
    
    def shared_epoch_end(self, outputs, stage="train"):
        """
            outputs: list from the results of the corresponding step:
            - for train: list[{"loss": tensor, ...}] (Lightning will pack the dictionary)
            - for val/test: list[tensor]
            Our task is to correctly extract the loss tensor from each element.
        """
        losses = []
        for o in outputs:
            if isinstance(o, tuple):
                losses.append(o[0].detach())
            elif isinstance(o, dict):
                losses.append(o["loss"].detach())
            elif isinstance(o, torch.Tensor):
                losses.append(o.detach())
            else:
                raise TypeError(f"[ERROR] Unexpected output type in epoch_end: {type(o)}")
        
        losses = torch.stack(losses)  # [num_batches]
        avg_loss = losses.mean().item()

        stats = {
            "epoch": self.current_epoch,
            f"{stage}_loss": avg_loss
        }

        if stage == 'train':
            metrics = self.train_metrics
        elif stage == 'val':
            metrics = self.val_metrics
        elif stage == 'test':
            metrics = self.test_metrics

        for name, metric in metrics.items():
            val = metric.compute().item()
            metric.reset()
            if name == 'spec':
                val = 1 - val

            stats[f"{stage}_{name}"] = val

        if self.test_dice_scores != []:
            # Summarize test metrics
            stats[f"{stage}_Dice_med"] = np.median(self.test_dice_scores)
            stats[f"{stage}_PPV_med"]  = np.median(self.test_ppv_scores)
            stats[f"{stage}_IoU_med"]  = np.median(self.test_iou_scores)
            stats[f"{stage}_ACC_med"]  = np.median(self.test_acc_scores)

            stats[f"{stage}_dice"]  = np.mean(self.test_dice_scores)
            stats[f"{stage}_ppv"]   = np.mean(self.test_ppv_scores)
            stats[f"{stage}_IoU"]   = np.mean(self.test_iou_scores)
            stats[f"{stage}_acc"]   = np.mean(self.test_acc_scores)

        if stage != "test":
            self.history[self.current_epoch] = stats.copy()

        return stats

    def training_epoch_end(self, outputs):
        stats = self.shared_epoch_end(outputs, stage="train")
        print(
            f"\n[TRAIN epoch {stats['epoch']}] "
            f"loss={stats['train_loss']:.4f}, "
            f"acc={stats['train_acc']:.4f}, "
            f"fp={stats['train_spec']:.4f}, "
            f"ppv={stats['train_ppv']:.4f}, "
            f"dice={stats['train_dice']:.4f}, "
            f"IoU={stats['train_IoU']:.4f}"
        )
        
        self.log_dict({k: v for k, v in stats.items() if k != "epoch"},
                      prog_bar=False, logger=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        stats = self.shared_epoch_end(outputs, stage="val")
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + f" {nowtime}")
        print(
            f"[VAL   epoch {stats['epoch']}] "
            f"loss={stats['val_loss']:.4f}, "
            f"acc={stats['val_acc']:.4f}, "
            f"fp={stats['val_spec']:.4f}, "
            f"ppv={stats['val_ppv']:.4f}, "
            f"dice={stats['val_dice']:.4f}, "
            f"IoU={stats['val_IoU']:.4f}"
        )
        self.log_dict({k: v for k, v in stats.items() if k != "epoch"},
                      prog_bar=False, logger=True, sync_dist=True)
        
        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb is not None:
            monitor = ckpt_cb.monitor
            mode    = ckpt_cb.mode  # "min" или "max"
            arr_scores = pd.DataFrame(self.history).T[monitor].values
            if mode == "max":
                best_idx = np.argmax(arr_scores)
            else:
                best_idx = np.argmin(arr_scores)
            if best_idx == len(arr_scores) - 1:
                print(
                    f"<<<<<< reach best {monitor} : {arr_scores[best_idx]:.4f} >>>>>>",
                    file=sys.stderr
                )

    def test_epoch_end(self, outputs):
        stats = self.shared_epoch_end(outputs, stage="test")
        print(
            f"\n[TEST  epoch {stats['epoch']}] "
            f"loss={stats['test_loss']:.4f}, "
            f"acc={stats['test_acc']:.4f}, "
            f"fp={stats['test_spec']:.4f}, ",
            f"ppv={stats['test_ppv']:.4f}, "
            f"dice={stats['test_dice']:.4f}, "
            f"IoU={stats['test_IoU']:.4f}"
        )
        self.log_dict({k: v for k, v in stats.items() if k != "epoch"},
                      prog_bar=False, logger=True, sync_dist=True)

        def summarize(scores):
            med = np.median(scores)
            lo, hi = np.percentile(scores, [2.5, 97.5])
            return med, lo, hi

        d_med, d_lo, d_hi = summarize(self.test_dice_scores)
        p_med, p_lo, p_hi = summarize(self.test_ppv_scores)
        i_med, i_lo, i_hi = summarize(self.test_iou_scores)
        a_med, a_lo, a_hi = summarize(self.test_acc_scores)

    
        metrics = {
            "Dice_med": d_med, #"dice_lo": d_lo, "dice_hi": d_hi,
            "PPV_med": p_med,  #"ppv_lo": p_lo,  "ppv_hi": p_hi,
            "IoU_med": i_med,  #"iou_lo": i_lo,  "iou_hi": i_hi,
            "ACC_med": a_med,  #"acc_lo": a_lo,  "acc_hi": a_hi,
        }

        print("\n=== OVERALL TEST METRICS ===")
        print(f"Dice : {d_med:.3f} ({d_lo:.3f}-{d_hi:.3f})")
        print(f"PPV  : {p_med:.3f} ({p_lo:.3f}-{p_hi:.3f})")
        print(f"IoU  : {i_med:.3f} ({i_lo:.3f}-{i_hi:.3f})")
        print(f"Acc  : {a_med:.3f} ({a_lo:.3f}-{a_hi:.3f})")

        return metrics

    @staticmethod    
    def save_mgh(filename, vertex_values, demo_img):
        """save mgh file using nibabel and imported demo mgh file"""
        shape = demo_img.header.get_data_shape()
        data = np.zeros(shape, dtype=np.float32)
        data.flat[:] = vertex_values
        # Save result
        new_img = nb.MGHImage(data, demo_img.affine, demo_img.header)
        nb.save(new_img, filename)

    def on_test_epoch_start(self):
        self.test_dice_scores.clear()
        self.test_ppv_scores.clear()
        self.test_iou_scores.clear()
        self.test_acc_scores.clear()
        for m in self.test_metrics.values():
            m.reset()
            
    def get_history(self):
        return pd.DataFrame(self.history.values())