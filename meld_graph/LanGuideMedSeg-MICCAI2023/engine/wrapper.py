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
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
from utils.utils import summarize_ci, compute_adaptive_threshold
import sys
import numpy as np
import datetime
from engine.pooling import HexPool
from engine.loss_meld import calculate_loss


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
    def __init__(self, args: Any) -> None:
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

        probs_bin = torch.empty_like(pprobs)
        for i in range(B):
            for h in range(H):  # 2 hemispheres
                pv = pprobs[i, h]  # [V_cortex]
                th = compute_adaptive_threshold(pv.detach().cpu().numpy())
                probs_bin[i, h] = (pv >= th).float()

        # frac_pos = probs_bin.float().mean()
        # print(f"\n{stage}/frac_positive: ", frac_pos)
        # print(f"\n{stage}/mean_prob_lesion: ", pprobs.mean())

        y_flat = target.view(-1)  # [B*N_cortex]
        p_flat = probs_bin.view(-1)

        # Обновляем метрики тем же образом, как делал раньше
        if stage == "test":
            # for sid, pred, tgt in zip(subject_ids, pprobs, target):
            for sid, pred, tgt in zip(subject_ids, probs_bin, target):
                dice_i = self.dice(pred, tgt).item()
                ppv_i = self.ppv(pred, tgt).item()
                iou_i = self.iou(pred, tgt).item()
                acc_i = self.acc(pred, tgt).item()

                self.test_dice_scores.append(dice_i)
                self.test_ppv_scores.append(ppv_i)
                self.test_iou_scores.append(iou_i)
                self.test_acc_scores.append(acc_i)

                print(
                    f"[TEST sample {sid!r}] Dice={dice_i:.3f}, PPV={ppv_i:.3f}, "
                    f"IoU={iou_i:.3f}, Acc={acc_i:.3f}"
                )
        else:
            metrics = self.train_metrics if stage == "train" else self.val_metrics
            for name, metric in metrics.items():
                val = metric(p_flat, y_flat)
                # У тебя "spec" логируется как FPR (1 - specificity)
                if name == "spec":
                    val = 1 - val
                self.log(
                    name,
                    val,
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
            # ---------- Save predictions as MGH and NIfTI ----------

            # subjects_fs_dir = (
            #     "/home/s17gmikh/FCD-Detection/meld_graph/data/input/data4sharing"
            # )
            # predictions_output_root = os.path.join(
            #     MELD_DATA_PATH, "output", "predictions_reports"
            # )

            # for (sid, pred) in zip(subject_ids, probs_bin):
            #     predictions = pred.detach().cpu().numpy()

            #     classifier_dir = os.path.join(
            #         subjects_fs_dir, sid, "xhemi", "classifier"
            #     )
            #     predictions_dir = os.path.join(
            #         predictions_output_root, sid, "predictions"
            #     )
            #     os.makedirs(classifier_dir, exist_ok=True)
            #     os.makedirs(predictions_dir, exist_ok=True)

            #     h5_path = os.path.join(
            #         subjects_fs_dir,
            #         "meld_combats",
            #         f"{sid}_patient_featurematrix_combat.hdf5",
            #     )
            #     for idx, hemi in enumerate(["lh", "rh"]):
            #         # Prediction overlay
            #         overlay = np.zeros_like(self.c.cortex_mask, dtype=np.float32)
            #         overlay[self.c.cortex_mask] = predictions[idx]

            #         # Read template thickness to get shape/affine
            #         combat_file = get_combat_feature_path(
            #             os.path.join(subjects_fs_dir, "meld_combats"), sid
            #         )
            #         with h5py.File(combat_file, "r") as f:
            #             key = ".combat.on_lh.thickness.sm3.mgh"
            #             if key not in f[hemi]:
            #                 raise KeyError(f"No dataset {key!r} in group {hemi}")
            #             base_arr = f[hemi][key][:]

            #         mgh_img = nb.MGHImage(
            #             base_arr[np.newaxis, :, np.newaxis],
            #             affine=nb.load(
            #                 os.path.join(
            #                     subjects_fs_dir, "fsaverage_sym", "mri", "T1.mgz"
            #                 )
            #             ).affine,
            #         )

            #         # Save prediction MGH → NIfTI
            #         out_mgh_pred = os.path.join(
            #             classifier_dir, f"{hemi}.prediction.mgh"
            #         )
            #         save_mgh(out_mgh_pred, overlay, mgh_img)
            #         print(f"Saved PRED MGH: {out_mgh_pred}")
            #         convert_prediction_mgh_to_nii(
            #             subjects_fs_dir,
            #             out_mgh_pred,
            #             hemi,
            #             predictions_dir,
            #             verbose=True,
            #         )

            #         mgh_gt = save_gt_as_mgh(
            #             h5_path, hemi, predictions_dir, subjects_fs_dir
            #         )
            #         if mgh_gt:
            #             convert_gt_to_nii(subjects_fs_dir, mgh_gt, hemi, verbose=True)

            #     # Combine both hemispheres for prediction
            #     lh_nii = os.path.join(predictions_dir, "lh.prediction.nii.gz")
            #     rh_nii = os.path.join(predictions_dir, "rh.prediction.nii.gz")
            #     final_nii = os.path.join(predictions_dir, f"prediction_{sid}.nii.gz")
            #     cmd = f"mri_concat --i {lh_nii} --i {rh_nii} --o {final_nii} --combine"
            #     run_command(cmd, verbose=True)
            #     print(f"🎉 Final combined PRED NIfTI: {final_nii}")

            #     # Combine both hemispheres for ground‐truth
            #     gt_lh_nii = os.path.join(predictions_dir, "lh.gt.nii.gz")
            #     gt_rh_nii = os.path.join(predictions_dir, "rh.gt.nii.gz")
            #     gt_final = os.path.join(predictions_dir, f"ground_truth_{sid}.nii.gz")
            #     cmd_gt = f"mri_concat --i {gt_lh_nii} --i {gt_rh_nii} --o {gt_final} --combine"
            #     run_command(cmd_gt, verbose=False)
            #     print(f"🎉 Final combined GT   NIfTI: {gt_final}")

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

        for name, metric in metrics.items():
            val = metric.compute().item()
            metric.reset()
            if name == "spec":
                val = 1 - val

            stats[f"{stage}_{name}"] = val

        if self.test_dice_scores != []:
            # Summarize test metrics
            stats[f"{stage}_Dice_med"] = np.median(self.test_dice_scores)
            stats[f"{stage}_PPV_med"] = np.median(self.test_ppv_scores)
            stats[f"{stage}_IoU_med"] = np.median(self.test_iou_scores)
            stats[f"{stage}_ACC_med"] = np.median(self.test_acc_scores)

            stats[f"{stage}_dice"] = np.mean(self.test_dice_scores)
            stats[f"{stage}_ppv"] = np.mean(self.test_ppv_scores)
            stats[f"{stage}_IoU"] = np.mean(self.test_iou_scores)
            stats[f"{stage}_acc"] = np.mean(self.test_acc_scores)

        if stage != "test":
            self.history[self.current_epoch] = stats.copy()

        return stats

    def training_epoch_end(self, outputs: List[Any]) -> None:
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
            f"acc={stats['val_acc']:.4f}, "
            f"fp={stats['val_spec']:.4f}, "
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
            f"acc={stats['test_acc']:.4f}, "
            f"fp={stats['test_spec']:.4f}, ",
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
        a_med, a_lo, a_hi = summarize_ci(self.test_acc_scores)

        metrics = {
            "Dice_med": d_med,  # "dice_lo": d_lo, "dice_hi": d_hi,
            "PPV_med": p_med,  # "ppv_lo": p_lo,  "ppv_hi": p_hi,
            "IoU_med": i_med,  # "iou_lo": i_lo,  "iou_hi": i_hi,
            "ACC_med": a_med,  # "acc_lo": a_lo,  "acc_hi": a_hi,
        }

        print("\n=== OVERALL TEST METRICS ===")
        print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
        print(f"PPV  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
        print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
        print(f"Acc  : {a_med:.3f} (95% CI {a_lo:.3f}-{a_hi:.3f})")

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
