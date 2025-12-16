from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union

from LanGuideMedSeg_MICCAI2023.models.model import LanGuideMedSeg

from meld_graph.meld_cohort import MeldCohort
from meld_graph.icospheres import IcoSpheres

from pathlib import Path
import torch
import pytorch_lightning as pl
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
    def __init__(self, 
                 args: Any, 
                 eva: Any, 
                 mode: str = None, 
                 fold_number: int = 0, 
                 model_type: str = None,
                 att_mechanism: bool = False, 
                 text_emb: bool = False
        ) -> None:
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
        self.fold_number = fold_number
        self.mode = mode
        self.final_predictions = []
        self.epi_dict = None
        if self.mode == 'inference':
            args.feature_path = Path("meld_graph") / "data"
            self.ckpt_path = model_type
        else:
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
            args.gnn_min_verts,
            args.num_unfreeze_layers,
            self.fold_number,
            att_mechanism=att_mechanism,
            text_emb=text_emb
        )

        self.c = MeldCohort()

        # cache cortex mask on CPU to avoid recreating tensor every batch
        self.cortex_mask_cpu = torch.from_numpy(self.c.cortex_mask)

        self.history: Dict[int, Dict[str, Union[float, int]]] = {}
        
        metrics = ["dice_scores", "ppv_scores", "iou_scores", "number_fp_clusters", "number_tp_clusters"]
        stages = ["train", "val", "test"]

        for stage in stages:
            for metric in metrics:
                setattr(self, f"{stage}_{metric}", [])

        self.results = []
        self.base_path: str = args.feature_path
        self.icospheres = IcoSpheres()

        self.ds_levels: List[int] = self.params["network_parameters"]["training_parameters"]["deep_supervision"]["levels"]
        self.ds_weights: List[float] = self.params["network_parameters"]["training_parameters"]["deep_supervision"]["weight"]
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
            pct_start=0.1,  # попробовать 0.2
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

        # ensure cortex mask is available on the current device (cached)
        if not hasattr(self, "_cortex_mask_device") or self._cortex_mask_device.device != getattr(self, "device", torch.device("cpu")):
            self._cortex_mask_device = self.cortex_mask_cpu.to(getattr(self, "device", torch.device("cpu")))
        self.cortex_mask = self._cortex_mask_device

        outputs = self([subject_ids, text])

        y = batch["roi"]  # torch.Tensor
        B, H, V7 = y.shape
        dist_maps = batch["dist_maps"]  # torch.Tensor

        # Pooling layers of targets
        dist_maps = dist_maps.view(B, 2, -1)
        # cortex_mask already moved to device above; build pooled dicts
        labels_pooled = {7: y.long()}  # [B,H,V7]
        dists_pooled = {7: dist_maps}
        cortex_pooled = {7: self.cortex_mask.bool()}  # [V7] bool

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
        # Loss configuration
        loss_cfg = self.params["network_parameters"]["training_parameters"][
            "loss_dictionary"
        ]

        # ---------- Loss on final layer (S7) ----------

        # [B*H*V, 2] -> [B, H, V, 2] -> [B, 2, H, V]
        logp = outputs["log_softmax"].view(B, H, V7, 2).permute(0, 3, 1, 2)  # [B,2,H,V]
        logp = logp[..., self.cortex_mask]  # [B,2,H,V_cortex]
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
            dist_map_l = dist_map_l[..., cortex_mask]
            dist_map_l = dist_map_l.view(B, -1)

            estimates_ds = {}
            logp_ds = (
                outputs[f"ds{level}_log_softmax"]
                .view(B, H, num_vert_ds, 2)
                .permute(0, 3, 1, 2)
            )  # [B,2,H,V_l]
            logp_ds = logp_ds[..., cortex_pooled[level]]  # [B,2,H,V_cortex_l]
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

        # Move predicted probabilities and dist_maps to CPU once to avoid repeated device transfers
        pprobs_cpu = pprobs.detach().cpu().numpy()  # shape [B, H, V_cortex]
        if isinstance(dist_maps_cortex, torch.Tensor):
            dist_maps_cortex_cpu = dist_maps_cortex.detach().cpu().numpy()
        else:
            dist_maps_cortex_cpu = dist_maps_cortex

        all_preds = []
        dice_step = []
        ppv_step = []
        iou_step = []

        # Iterate over subjects using CPU arrays to minimize small allocations
        for i, sid in enumerate(subject_ids):
            pred_np = np.concatenate([pprobs_cpu[i, 0], pprobs_cpu[i, 1]]).astype("float32")
            mini = {sid: {"result": pred_np}}
            out = self.eva.threshold_and_cluster(data_dictionary=mini, save_prediction=False)

            probs_flat = out[sid]["cluster_thresholded"]  # (2*N_cortex,)
            boundary_zone = dist_maps_cortex_cpu[i] < 20

            # normalize probs_flat to numpy
            if isinstance(probs_flat, torch.Tensor):
                probs_flat_cpu = probs_flat.detach().cpu().numpy()
            else:
                probs_flat_cpu = probs_flat

            difference = np.setdiff1d(
                np.unique(probs_flat_cpu), np.unique(probs_flat_cpu[boundary_zone])
            )

            difference = difference[difference > 0]
            n_fp_clusters = len(difference)
            correct_values = np.unique(probs_flat_cpu[boundary_zone])
            correct_values = correct_values[correct_values > 0]
            n_tp_clusters = len(correct_values)

            all_preds.append(torch.from_numpy(probs_flat_cpu).view(H, -1).contiguous())

            # ground truth for this subject (on device)
            tgt = target[i]
            gt_flat = tgt.reshape(-1)

            mask = torch.from_numpy(probs_flat_cpu > 0).long().to(gt_flat.device)
            labels = gt_flat.bool().long()
            dices = dice_coeff(torch.nn.functional.one_hot(mask, num_classes=2), labels)

            tp, fp, fn, tn = tp_fp_fn_tn(mask, labels)
            iou = tp / (tp + fp + fn + 1e-8)
            ppv = tp / (tp + fp + 1e-8)

            getattr(self, f"{stage}_dice_scores").append(float(dices[1].detach().cpu()))
            getattr(self, f"{stage}_ppv_scores").append(float(ppv.detach().cpu()))
            getattr(self, f"{stage}_iou_scores").append(float(iou.detach().cpu()))
            getattr(self, f"{stage}_number_fp_clusters").append(n_fp_clusters)
            getattr(self, f"{stage}_number_tp_clusters").append(n_tp_clusters)

            dice_step.append(float(dices[1].detach().cpu()))
            ppv_step.append(float(ppv.detach().cpu()))
            iou_step.append(float(iou.detach().cpu()))

            if stage == "test":
                print(
                    f"[{sid}] Dice lesional={dices[1]:.3f}, IoU={iou:.3f}, PPV={ppv:.3f}, "
                    f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
                )
                self.results.append(
                    {
                        "subject_id": sid,
                        "number FP clusters": n_fp_clusters,
                        "number TP clusters": n_tp_clusters,
                        "dice": float(dices[1]),
                        "iou": float(iou),
                        "ppv_voxel": float(ppv),
                    }
                )
        
        if stage != "test":
            mean_dice = np.mean(dice_step)
            mean_ppv = np.mean(ppv_step)
            mean_iou = np.mean(iou_step)

            self.log(f"dice", mean_dice, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=True)
            self.log(f"ppv", mean_ppv, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=True)
            self.log(f"iou", mean_iou, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=True)

        # ---------- Save predictions as MGH and NIfTI ----------
        if stage == "test":
            final_nii = convert_preds_to_nifti(self.ckpt_path, subject_ids, all_preds, self.c)
        
        return total_loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss = self.shared_step(batch, batch_idx, stage="train")
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

        return {"loss": loss}

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
                if hasattr(o["loss"], "detach"):
                    losses.append(o["loss"].detach())
                else:
                    losses.append(o["loss"])
            elif isinstance(o, torch.Tensor):
                losses.append(o.detach())
            else:
                raise TypeError(
                    f"[ERROR] Unexpected output type in epoch_end: {type(o)}"
                )

        losses = torch.stack(losses)  # [num_batches]
        avg_loss = losses.mean().item()

        stats = {"epoch": self.current_epoch, f"{stage}_loss": avg_loss}

        dice_scores = getattr(self, f"{stage}_dice_scores")
        ppv_scores  = getattr(self, f"{stage}_ppv_scores")
        iou_scores  = getattr(self, f"{stage}_iou_scores")
        n_fp_clusters = getattr(self, f"{stage}_number_fp_clusters")
        n_tp_clusters = getattr(self, f"{stage}_number_tp_clusters")

        ppv_clusters = np.sum(n_tp_clusters) / np.sum(n_tp_clusters + n_fp_clusters)

        if len(dice_scores) > 0:
            stats[f"{stage}_dice"] = np.mean(dice_scores)
            stats[f"{stage}_ppv_pixels"] = np.mean(ppv_scores)
            stats[f"{stage}_IoU"] = np.mean(iou_scores)
            stats[f"{stage}_ppv_clusters"] = ppv_clusters

        if stage != "test":
            self.history[self.current_epoch] = stats.copy()
        
        return stats

    def training_epoch_end(self, outputs: List[Any]) -> None:
        stats = self.shared_epoch_end(outputs, stage="train")
        print(
            f"\n[TRAIN epoch {stats['epoch']}] "
            f"loss={stats['train_loss']:.4f}, "
            f"ppv_pixels={stats['train_ppv_pixels']:.4f}, "
            f"ppv_clusters={stats['train_ppv_clusters']:.4f}, "
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
            f"ppv_pixels={stats['val_ppv_pixels']:.4f}, "
            f"ppv_clusters={stats['val_ppv_clusters']:.4f}, "
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
            f"ppv={stats['test_ppv_pixels']:.4f}, "
            f"ppv={stats['test_ppv_clusters']:.4f}, "
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

        total = len(self.test_number_tp_clusters)
        found = sum(1 for t in self.test_number_tp_clusters if t > 0)
        pct = found / total if total > 0 else 0.0
        
        print("\n=== OVERALL TEST METRICS ===")
        print(f"Dice : {d_med:.3f} (95% CI {d_lo:.3f}-{d_hi:.3f})")
        print(f"PPV_pixels  : {p_med:.3f} (95% CI {p_lo:.3f}-{p_hi:.3f})")
        print(f"PPV_clusters  : ", stats['test_ppv_clusters'])
        print(f"IoU  : {i_med:.3f} (95% CI {i_lo:.3f}-{i_hi:.3f})")
        print(f"Detected {found} / {total} FCDs ({pct:.1%})")

        df = pd.DataFrame(self.results)
        df.to_csv(f"{self.ckpt_path}_results.csv", index=False)

    def on_train_epoch_start(self) -> None:
        self.train_dice_scores.clear()
        self.train_ppv_scores.clear()
        self.train_iou_scores.clear()

    def on_validation_epoch_start(self) -> None:
        self.val_dice_scores.clear()
        self.val_ppv_scores.clear()
        self.val_iou_scores.clear()

    def on_test_epoch_start(self) -> None:
        self.test_dice_scores.clear()
        self.test_ppv_scores.clear()
        self.test_iou_scores.clear()


    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history.values())
