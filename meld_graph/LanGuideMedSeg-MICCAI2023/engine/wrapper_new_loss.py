from utils.model_new_loss import LanGuideMedSeg
from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss
from torchmetrics import Accuracy, Dice

from meld_graph.paths import (MELD_DATA_PATH, DEFAULT_HDF5_FILE_ROOT)

import os
from meld_graph.meld_cohort import MeldCohort
from scripts.manage_results.register_back_to_xhemi import register_subject_to_xhemi
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import csv
from engine.loss import calculate_loss
import torch.nn.functional as F

def load_config(config_file):
    '''load config.py file and return config object'''
    import importlib.machinery, importlib.util

    loader = importlib.machinery.SourceFileLoader('config', config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class LanGuideMedSegWrapper(pl.LightningModule):
    def __init__(self, args, tokenizer, max_len, alpha=0.85, gamma=2.0, coef=(0.8, 0.1, 0.1)):
        super(LanGuideMedSegWrapper, self).__init__()

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
        self.foc_coef, self.fp_coef, self.dice_coef = coef
        self.lr = args.lr
        self.history = {}
        self.test_dice_scores = []
        self.test_ppv_scores  = []
        self.test_iou_scores  = []
        self.test_acc_scores  = []

        self.config = load_config('/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/final_ablation_full_with_combat_my.py')
        self.dice_fn = DiceLoss(include_background=False, sigmoid=True)

        # ---- метрики (они принимают прогнозы [B, N] и метки [B, N]) ----
        self.train_metrics = nn.ModuleDict({
            "acc" : Accuracy(task="binary"),
            "spec"  : Specificity(task="binary"),
            "dice": Dice(),
            "ppv" : Precision(task="binary"),
            "MIoU": BinaryJaccardIndex()
        })
        
        self.dice = Dice().to(args.device)                 
        self.ppv  = Precision(task="binary").to(args.device)
        self.iou  = BinaryJaccardIndex().to(args.device)
        self.acc  = Accuracy(task="binary").to(args.device)

        self.val_metrics   = deepcopy(self.train_metrics)
        self.test_metrics  = deepcopy(self.train_metrics)

        self.save_hyperparameters(args)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=300, eta_min=1e-4
        # ) # Worse than OneCycleLR
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0003,
            total_steps=1500, # 600
            pct_start=0.1, # 0.3
            anneal_strategy='cos'
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_test_epoch_start(self):
        # очищаем перед каждой тестовой эпохой
        self.test_dice_scores.clear()
        self.test_ppv_scores.clear()
        self.test_iou_scores.clear()
        self.test_acc_scores.clear()
    
    def save_mgh(self, filename, vertex_values, demo_img):
        """save mgh file using nibabel and imported demo mgh file"""
        shape = demo_img.header.get_data_shape()
        # print(shape)
        # New empty volume
        data = np.zeros(shape, dtype=np.float32)

        data.flat[:] = vertex_values

        # Save result
        new_img = nb.MGHImage(data, demo_img.affine, demo_img.header)
        nb.save(new_img, filename)

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
        B, H, N = y.shape

        if stage == "train":
            self.model.train()
            outputs = self.model(x)  # [B * H * N, 2]
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)  # [B * H * N, 2]

        # Convert: [B * H * N, 2] -> [B, H, num_classes, N]
        preds_logits = outputs['log_softmax'].view(B, H, 2, N)
        # Convert: [B * H * N] -> [B, H, N]
        preds_logits_hemi = outputs['non_lesion_logits'].view(B, H, N)

        if stage == 'test':
            subjects_dir = "/home/s17gmikh/FCD-Detection/meld_graph/data/output/fs_outputs"
            predictions_output_dir = os.path.join(MELD_DATA_PATH,'output','predictions_reports')
            for sid, p in zip(subject_ids, preds_logits):
                c = MeldCohort()
                # create classifier directory if not exist
                classifier_dir = os.path.join(subjects_dir, sid, "xhemi", "classifier")
                if not os.path.isdir(classifier_dir):
                    os.mkdir(classifier_dir)

                predictions = torch.exp(p)[:, :, 1, :]# torch.sigmoid(p)
                
                for idx, hemi in enumerate(["lh", "rh"]):
                    prediction_h = predictions[idx][c.cortex_mask].detach().cpu().numpy()
                    overlay = np.zeros_like(c.cortex_mask, dtype=np.float32)
                    overlay[c.cortex_mask] = prediction_h
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

        c = MeldCohort()
        preds_logits        = preds_logits[:, :, :, c.cortex_mask]
        preds_logits_hemi   = preds_logits_hemi[:, :, c.cortex_mask]
        y                   = y[:, :, c.cortex_mask]

        if stage == 'test':
            for sid, p, t in zip(subject_ids, preds_logits, y):
                print(f"---- {sid} ----")
                # update each metric
                for name, metric in self.test_metrics.items():
                    metric.update(p.flatten(), t.flatten().long())
                    val = metric.compute().item()
                    if name == 'spec':
                        val = 1 - val
                    metric.reset()
                    print(f"{name:>4} = {val:.4f}")
        
        loss_hemi = []
        for hemi in [0, 1]:
            logits_hemi = preds_logits[:, hemi, :, :]
            target = y[:, hemi, :].long()
            preds_prob_hemi = preds_logits_hemi[:, hemi, :]

            logits_hemi     = logits_hemi.permute(0, 2, 1).reshape(-1, H)
            preds_prob_hemi = preds_prob_hemi.reshape(-1)#.permute(0, 2, 1).reshape(-1, H)
            target          = target.reshape(-1)
            
            estimates = {
                'log_softmax': logits_hemi,
                'non_lesion_logits': preds_prob_hemi,
                # 'hemi_log_softmax': estimates['hemi_log_softmax'] # Lates
            }
            losses = calculate_loss(
                self.config.losses[0]['network_parameters']['training_parameters']['loss_dictionary'], 
                estimates, 
                target, 
                # distance_map=preds_prob_hemi, 
                #xyzr=xyzr,
                deep_supervision_level=None, 
                device=self.device, 
                n_vertices=N
            )
            loss_hemi.append(sum(losses.values()).mean())

        # for i, level in enumerate(sorted(self.deep_supervision["levels"])):
        #     cur_labels = getattr(data, f"output_level{level}")
        #     cur_distance_map = getattr(data, f"output_level{level}_distance_map", None)
        #     n_vertices = len(self.experiment.model.icospheres.icospheres[level]["coords"])
        #     ds_losses = calculate_loss(
        #         self.params["loss_dictionary"],
        #         preds_prob,
        #         cur_labels,
        #         distance_map=cur_distance_map,
        #         deep_supervision_level=level,
        #         device=self.device,
        #         n_vertices=n_vertices,
        #     )
        #     losses.update(
        #         {f"ds{level}_{key}": self.deep_supervision["weight"][i] * val for key, val in ds_losses.items()}
        #     )
        # calculate overall loss
        loss = torch.stack(loss_hemi, dim=0).mean()
        preds_prob      = torch.exp(preds_logits[:, :, 1, :]) # [B, H, N] - это вероятности "плюсов" (lesion)
        if batch_idx == 0 and stage == "train":
            print(
                f"\n[{stage.upper()} step {self.global_step}] "
                f"patch-loss(avg) = {loss.item():.4f} "
            )
            
            # 1) Average probability over all pixels:
            mean_prob = preds_prob.mean().item()

            # 2) Share of predicted "pluses" at threshold 0.5:
            pred_pos_frac = (preds_prob > 0.5).float().mean().item()

            # 3) Compare with share of real "pluses":
            true_pos_frac = (y > 0.5).float().mean().item()

            print(f"mean_prob={mean_prob:.4f}, pred_pos_frac={pred_pos_frac:.4f}, true_pos_frac={true_pos_frac:.4f}")

            # sum_prob = torch.sigmoid(preds_logits).sum().item()
            # print(f"[DEBUG] sum(y_all) = {sum_pos:.0f}, sum(sigmoid_all) = {sum_prob:.1f}")

        metrics = (
            self.train_metrics if stage == "train" else
            self.val_metrics   if stage == "val"   else
            self.test_metrics
        )
        for name, metric in metrics.items():
            if name == 'spec':
                value = 1 - metric(preds_prob, y.long())
            else:
                value = metric(preds_prob, y.long())
            if stage == "train":
                self.log(name, value, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)

        # if stage == 'test':
        #     y = y.long()
        #     for pred, tgt in zip(preds_prob, y):
        #         self.test_dice_scores.append(self.dice(pred, tgt).item())
        #         self.test_ppv_scores .append(self.ppv(pred, tgt).item())
        #         self.test_iou_scores .append(self.iou(pred, tgt).item())
        #         self.test_acc_scores .append(self.acc(pred, tgt).item())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="train")

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # loss, prods = self.shared_step(batch, batch_idx, stage="val")
        loss = self.shared_step(batch, batch_idx, stage="val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss, #prods  

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
            # print(f"[DEBUG] o = {o}, type = {type(o)}")
            if isinstance(o, tuple):
                losses.append(o[0])#.detach())
            elif isinstance(o, dict):
                losses.append(o["loss"].detach())
            elif isinstance(o, torch.Tensor):
                losses.append(o.detach())
            else:
                raise TypeError(f"[ERROR] Unexpected output type in epoch_end: {type(o)}")
        losses = torch.stack(losses)  # [num_batches]
        avg_loss = losses.mean().item()

        metrics = (
            self.train_metrics if stage == "train" else
            self.val_metrics   if stage == "val"   else
            self.test_metrics
        )

        stats = {
            "epoch": self.current_epoch,
            f"{stage}_loss": avg_loss
        }

        for name, metric in metrics.items():
            val = metric.compute().item()
            metric.reset()
            if name == 'spec':
                val = 1 - val

            stats[f"{stage}_{name}"] = val

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
            f"MIoU={stats['train_MIoU']:.4f}"
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
            f"MIoU={stats['val_MIoU']:.4f}"
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
            f"MIoU={stats['test_MIoU']:.4f}"
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

        params = {
            "lr": self.hparams.lr,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "focal_coef": self.foc_coef,
            "fp_coef": self.fp_coef,
            "dice_coef": self.dice_coef
        }
        metrics = {
            "dice_med": d_med, "dice_lo": d_lo, "dice_hi": d_hi,
            "ppv_med": p_med,  "ppv_lo": p_lo,  "ppv_hi": p_hi,
            "iou_med": i_med,  "iou_lo": i_lo,  "iou_hi": i_hi,
            "acc_med": a_med,  "acc_lo": a_lo,  "acc_hi": a_hi,
        }

        # 2) Путь к файлу результатов
        results_file = os.path.join(self.hparams.output_dir, "all_results.csv")
        file_exists = os.path.isfile(results_file)

        # 3) Открываем в режиме аппенд и дозаписываем строку
        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(params.keys()) + list(metrics.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow({**params, **metrics})

        print("\n=== OVERALL TEST METRICS ===")
        print(f"Dice : {d_med:.3f} ({d_lo:.3f}-{d_hi:.3f})")
        print(f"PPV  : {p_med:.3f} ({p_lo:.3f}-{p_hi:.3f})")
        print(f"IoU  : {i_med:.3f} ({i_lo:.3f}-{i_hi:.3f})")
        print(f"Acc  : {a_med:.3f} ({a_lo:.3f}-{a_hi:.3f})")

    def get_history(self):
        return pd.DataFrame(self.history.values())