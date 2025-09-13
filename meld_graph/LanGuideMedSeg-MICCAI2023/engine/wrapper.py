from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union

from utils.model import LanGuideMedSeg
from torchmetrics import Accuracy, Dice

from meld_graph.meld_cohort import MeldCohort
from meld_graph.icospheres import IcoSpheres
from torchmetrics.classification import (
    BinaryJaccardIndex,
    Precision,
    Specificity,
    BinaryF1Score,
)

from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from engine.loss_meld import dice_coeff, tp_fp_fn_tn
import pandas as pd
from utils.utils import summarize_ci
import sys
import numpy as np
import datetime
from engine.converter_mgh_to_nifti import *
from engine.pooling import HexPool
from engine.loss_meld import calculate_loss
from utils.utils import convert_preds_to_nifti

def load_config(config_file):
    """load config.py file and return config object"""
    import importlib.machinery
    import importlib.util

    loader = importlib.machinery.SourceFileLoader("config", config_file)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    config = importlib.util.module_from_spec(spec)
    loader.exec_module(config)
    return config


class LanGuideMedSegWrapper(pl.LightningModule):
    def __init__(self, args: Any, eva: Any) -> None:
        super().__init__()

        self.save_hyperparameters(args)
        self.config: Any = load_config(
            "/home/s17gmikh/FCD-Detection/meld_graph/scripts/config_files/final_ablation_full_with_combat_my.py"
        )
        self.params: Dict[str, Any] = (
            next(iter(self.config.losses))
            if isinstance(self.config.losses, list)
            else self.config.losses
        )
        layer_sizes: List[List[int]] = list(
            self.params["data_parameters"]["layer_sizes"]
        )
        self.eva = eva
        self.ckpt_path = Path(args.ckpt_path).stem if args.ckpt_path is not None else None
        self.model = LanGuideMedSeg(
            args.bert_type,
            args.meld_script_path,
            args.feature_path,
            args.output_dir,
            layer_sizes,
            args.device,
            args.feature_dim,
            args.text_lens,
            args.max_len,
        )

        self.train_metrics: nn.ModuleDict[str, nn.Module] = nn.ModuleDict(
            {
                "acc": Accuracy(task="binary"),
                "spec": Specificity(task="binary"),
                "dice": BinaryF1Score(),  # Dice(),
                "ppv": Precision(task="binary"),
                "IoU": BinaryJaccardIndex(),
            }
        )
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)

        self.dice: nn.Module = Dice().to(args.device)
        self.ppv: nn.Module = Precision(task="binary").to(args.device)
        self.iou: nn.Module = BinaryJaccardIndex().to(args.device)
        self.acc: nn.Module = Accuracy(task="binary").to(args.device)

        self.c = MeldCohort()

        self.history: Dict[int, Dict[str, Union[float, int]]] = {}
        self.test_dice_scores: List[float] = []
        self.test_ppv_scores: List[float] = []
        self.test_iou_scores: List[float] = []
        self.test_acc_scores: List[float] = []

        self.base_path: str = args.feature_path
        self.icospheres = IcoSpheres()

        self.ds_levels: List[int] = self.params["network_parameters"][
            "training_parameters"
        ]["deep_supervision"]["levels"]
        self.ds_weights: List[float] = self.params["network_parameters"][
            "training_parameters"
        ]["deep_supervision"]["weight"]
        self.pool_layers: Dict[int, HexPool] = {
            level: HexPool(self.icospheres.get_downsample(target_level=level))
            for level in range(min(self.ds_levels), 7)[::-1]
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-3
        )  # 1e-2

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,  # 3e-3
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,  # короче разгон, чаще помогает
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "one_cycle",
            },
        }

    def forward(
        self, x: Tuple[List[str], Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def shared_step(
        self, batch: Dict[str, Any], batch_idx: int, stage: str
    ) -> torch.Tensor:
        """
        General code for train/val/test.
        batch = (x, y, ...), where
            x: input features (→ self.model(x) yields [B, n_nodes])
            y: binary labels {0,1} [B, n_nodes]
        ignore the rest of the batch elements
        stage: "train" | "val" | "test"
        """
        subject_ids = batch["subject_id"]  # list[str]
        text = batch["text"]  # dict with input_ids, attention_mask
        y = batch["roi"]  # torch.Tensor
        dist_maps = batch["dist_maps"]  # torch.Tensor

        # Pooling layers of targets
        B, H, V7 = y.shape
        dist_maps = dist_maps.view(B, 2, -1)
        labels_pooled = {7: y.long()}  # [B,H,V7]
        dists_pooled = {7: dist_maps}
        self.cortex_mask = torch.from_numpy(self.c.cortex_mask).to(y.device)
        cortex_pooled = {7: self.cortex_mask.to(y.device).bool()}  # [V7] bool

        for level in range(min(self.ds_levels), 7)[::-1]:
            pooled = self.pool_layers[level](labels_pooled[level + 1].float())
            labels_pooled[level] = (pooled >= 0.5).long()  # [B,H,V_level]

            dists_pooled[level] = self.pool_layers[level](
                dists_pooled[level + 1], center_pool=True
            )
            dists_pooled[level] = torch.clip(dists_pooled[level], 0, 300)

            cortex_mask = (
                cortex_pooled[level + 1].float().unsqueeze(0).unsqueeze(0)
            )  # [1,1,V_{l+1}]
            cortex_mask = self.pool_layers[level](cortex_mask)  # [1,1,V_level] float
            cortex_pooled[level] = cortex_mask.squeeze(0).squeeze(0).bool()

        y = y.float()
        outputs = self([subject_ids, text])

        # Loss configuration
        B, H, V7 = y.shape
        loss_cfg = self.params["network_parameters"]["training_parameters"][
            "loss_dictionary"
        ]

        # ---------- Loss on final layer (S7) ----------

        # [B*H*V, 2] -> [B, H, V, 2] -> [B, 2, H, V]
        logp = outputs["log_softmax"].view(B, H, V7, 2).permute(0, 3, 1, 2)  # [B,2,H,V]
        logp = logp[:, :, :, self.cortex_mask]  # [B,2,H,V_cortex]
        logp = logp.reshape(B, 2, -1)  # [B,2,H*V_cortex]

        y_mask = y[:, :, self.cortex_mask]  # [B,H,V_cortex]
        target = y_mask.view(B, -1).long()

        dist_maps_cortex = dist_maps[:, :, self.cortex_mask]
        dist_maps_cortex = dist_maps_cortex.view(B, -1)

        estimates = {}
        estimates["log_softmax"] = logp
        estimates["hemi_log_softmax"] = outputs["hemi_log_softmax"]
        # distance head
        if "non_lesion_logits" in outputs:
            non_lesion_logits_cortex = outputs["non_lesion_logits"].view(B, 2, -1)[
                :, :, self.cortex_mask
            ]
            estimates["non_lesion_logits"] = non_lesion_logits_cortex.reshape(B, -1)

        losses = {}

        losses_main = calculate_loss(
            loss_cfg,
            estimates,
            labels=target,
            distance_map=dist_maps_cortex,
            deep_supervision_level=None,
            device=self.device,
            n_vertices=y.shape[2],
        )
        total_loss = sum(losses_main.values())
        losses.update({f"main/{k}": v for k, v in losses_main.items()})

        # ---------- Losses on DS-levels ----------

        for weight, level in zip(self.ds_weights, self.ds_levels):
            key = f"ds{level}_log_softmax"
            if key not in outputs:
                continue

            num_vert_ds = labels_pooled[level].size(-1)
            cortex_mask = cortex_pooled[level]  # [V_l] bool
            y_l = labels_pooled[level][
                :, :, cortex_mask
            ]  # [B,H,V_l] -> [B,H,V_l_cortex]
            y_l = y_l.reshape(y_l.shape[0], -1)  # [B*H*V_l]

            dist_map_l = dists_pooled[level]
            dist_map_l = dist_map_l[:, :, cortex_mask]
            dist_map_l = dist_map_l.view(B, -1)

            estimates_ds = {}
            logp_ds = (
                outputs[f"ds{level}_log_softmax"]
                .view(B, H, num_vert_ds, 2)
                .permute(0, 3, 1, 2)
            )  # [B,2,H,V_l]
            logp_ds = logp_ds[:, :, :, cortex_pooled[level]]  # [B,2,H,V_cortex_l]
            logp_ds = logp_ds.reshape(B, 2, -1)  # [B,2,H*V_cortex_l]

            estimates_ds["log_softmax"] = logp_ds
            estimates_ds["non_lesion_logits"] = (
                outputs[f"ds{level}_non_lesion_logits"]
                .view(B, 2, -1)[:, :, cortex_mask]
                .view(B, -1)
            )

            ds_losses = calculate_loss(
                loss_cfg,
                estimates_ds,
                labels=y_l,
                distance_map=dist_map_l,
                deep_supervision_level=level,
                device=self.device,
                n_vertices=num_vert_ds,
            )

            for _, val_loss in ds_losses.items():
                total_loss = total_loss + weight * val_loss
            losses.update(
                {
                    f"ds{level}/{name_loss}": weight * loss_val
                    for name_loss, loss_val in ds_losses.items()
                }
            )

        # ---------- logging ----------
        self.log(
            f"{stage}/loss_total",
            total_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # Calculate metrics on cortex only
        probs = logp[:, 1, :].exp()
        pprobs = probs.view(B, H, -1).contiguous()  # [B, H, V_cortex]
        target = target.view(B, H, -1)

        all_preds = []
        # Обновляем метрики тем же образом, как делал раньше
        for sid, pred, tgt in zip(subject_ids, pprobs, target):
        # for sid, pred, tgt in zip(subject_ids, probs_bin, target):
            pred_np = torch.cat([pred[0], pred[1]], dim=0).detach().cpu().numpy().astype("float32")
            mini = {sid: {"result": pred_np}}
            out = self.eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)
            
            probs_flat = out[sid]["cluster_thresholded"]        # (2*N_cortex,)
            all_preds.append(torch.from_numpy(probs_flat).view(H, -1).contiguous())
            gt_flat = tgt.reshape(-1)
            
            mask = torch.as_tensor(np.array(probs_flat > 0)).long()
            labels = torch.as_tensor(np.array(gt_flat).astype(bool)).long()
            dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)

            tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
            iou = tp / (tp + fp + fn + 1e-8)
            ppv = tp / (tp + fp + 1e-8)

            self.test_dice_scores.append(dices[1])
            self.test_ppv_scores.append(ppv)
            self.test_iou_scores.append(iou)
            print(f"[{sid}] Dice lesional={dices[1]:.3f}, IoU={iou:.3f}, PPV={ppv:.3f}, "
                f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

        # ---------- Save predictions as MGH and NIfTI ----------
        if stage == "test":
            convert_preds_to_nifti(self.ckpt_path, subject_ids, all_preds, self.c)

        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, batch_idx, stage="train")
        # print(f"[TRAIN epoch {self.current_epoch}] " +
        #   ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, batch_idx, stage="val")
        # print(f"[VAL   epoch {self.current_epoch}] " +
        #   ", ".join(f"{k}={v:.4f}" for k, v in losses.items()))
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, batch_idx, stage="test")
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def shared_epoch_end(
        self,
        outputs: List[Union[Dict[str, torch.Tensor], torch.Tensor, Tuple[Any, ...]]],
        stage: str = "train",
    ) -> Dict[str, Union[int, float]]:
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
                raise TypeError(
                    f"[ERROR] Unexpected output type in epoch_end: {type(o)}"
                )

        losses = torch.stack(losses)  # [num_batches]
        avg_loss = losses.mean().item()

        stats = {"epoch": self.current_epoch, f"{stage}_loss": avg_loss}

        if stage == "train":
            metrics = self.train_metrics
        elif stage == "val":
            metrics = self.val_metrics
        elif stage == "test":
            metrics = self.test_metrics

        # for name, metric in metrics.items():
        #     val = metric.compute().item()
        #     metric.reset()
        #     if name == "spec":
        #         val = 1 - val

        #     stats[f"{stage}_{name}"] = val

        if self.test_dice_scores != []:
            # Summarize test metrics
            stats[f"{stage}_Dice_med"] = np.median(self.test_dice_scores)
            stats[f"{stage}_PPV_med"] = np.median(self.test_ppv_scores)
            stats[f"{stage}_IoU_med"] = np.median(self.test_iou_scores)
            # stats[f"{stage}_ACC_med"] = np.median(self.test_acc_scores)

            stats[f"{stage}_dice"] = np.mean(self.test_dice_scores)
            stats[f"{stage}_ppv"] = np.mean(self.test_ppv_scores)
            stats[f"{stage}_IoU"] = np.mean(self.test_iou_scores)
            # stats[f"{stage}_acc"] = np.mean(self.test_acc_scores)

        if stage != "test":
            self.history[self.current_epoch] = stats.copy()

        return stats

    def training_epoch_end(self, outputs: List[Any]) -> None:
        stats = self.shared_epoch_end(outputs, stage="train")
        print(
            f"\n[TRAIN epoch {stats['epoch']}] "
            f"loss={stats['train_loss']:.4f}, "
            # f"acc={stats['train_acc']:.4f}, "
            # f"fp={stats['train_spec']:.4f}, "
            f"ppv={stats['train_ppv']:.4f}, "
            f"dice={stats['train_dice']:.4f}, "
            f"IoU={stats['train_IoU']:.4f}"
        )

        self.log_dict(
            {k: v for k, v in stats.items() if k != "epoch"},
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        stats = self.shared_epoch_end(outputs, stage="val")
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 80 + f" {nowtime}")
        print(
            f"[VAL   epoch {stats['epoch']}] "
            f"loss={stats['val_loss']:.4f}, "
            # f"acc={stats['val_acc']:.4f}, "
            # f"fp={stats['val_spec']:.4f}, "
            f"ppv={stats['val_ppv']:.4f}, "
            f"dice={stats['val_dice']:.4f}, "
            f"IoU={stats['val_IoU']:.4f}"
        )
        self.log_dict(
            {k: v for k, v in stats.items() if k != "epoch"},
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb is not None:
            monitor = ckpt_cb.monitor
            mode = ckpt_cb.mode  # "min" или "max"
            arr_scores = pd.DataFrame(self.history).T[monitor].values
            if mode == "max":
                best_idx = np.argmax(arr_scores)
            else:
                best_idx = np.argmin(arr_scores)
            if best_idx == len(arr_scores) - 1:
                print(
                    f"<<<<<< reach best {monitor} : {arr_scores[best_idx]:.4f} >>>>>>",
                    file=sys.stderr,
                )

    def test_epoch_end(self, outputs: List[Any]) -> None:
        stats = self.shared_epoch_end(outputs, stage="test")
        print(
            f"\n[TEST  epoch {stats['epoch']}] "
            f"loss={stats['test_loss']:.4f}, "
            # f"acc={stats['test_acc']:.4f}, "
            # f"fp={stats['test_spec']:.4f}, ",
            f"ppv={stats['test_ppv']:.4f}, "
            f"dice={stats['test_dice']:.4f}, "
            f"IoU={stats['test_IoU']:.4f}",
        )
        self.log_dict(
            {k: v for k, v in stats.items() if k != "epoch"},
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        d_med, d_lo, d_hi = summarize_ci(self.test_dice_scores)
        p_med, p_lo, p_hi = summarize_ci(self.test_ppv_scores)
        i_med, i_lo, i_hi = summarize_ci(self.test_iou_scores)
        # a_med, a_lo, a_hi = summarize_ci(self.test_acc_scores)

        metrics = {
            "Dice_med": d_med,  # "dice_lo": d_lo, "dice_hi": d_hi,
            "PPV_med": p_med,  # "ppv_lo": p_lo,  "ppv_hi": p_hi,
            "IoU_med": i_med,  # "iou_lo": i_lo,  "iou_hi": i_hi,
            # "ACC_med": a_med,  # "acc_lo": a_lo,  "acc_hi": a_hi,
        }

        print("\n=== OVERALL TEST METRICS ===")
        print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
        print(f"PPV  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
        print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
        # print(f"Acc  : {a_med:.3f} (95% CI {a_lo:.3f}-{a_hi:.3f})")

        return metrics

    def on_test_epoch_start(self) -> None:
        self.test_dice_scores.clear()
        self.test_ppv_scores.clear()
        self.test_iou_scores.clear()
        self.test_acc_scores.clear()
        for m in self.test_metrics.values():
            m.reset()

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history.values())
