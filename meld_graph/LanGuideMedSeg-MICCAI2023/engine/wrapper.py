from utils.model import LanGuideMedSeg
from monai.losses import DiceLoss
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime


class LanGuideMedSegWrapper(pl.LightningModule):
    def __init__(self, args, root_path):
        super(LanGuideMedSegWrapper, self).__init__()

        # ---- инициализируем саму модель ----
        self.model = LanGuideMedSeg(
            args.bert_type,
            args.meld_script_path,
            args.feature_path,
            args.output_dir,
            args.project_dim,
            args.device
        )
        
        self.root_dir = root_path
        self.lr = args.lr
        self.history = {}

        # ---- «патчевый» запас вокруг каждой области ROI (в единицах индексов) ----
        self.patch_margin = getattr(args, "patch_margin", 500)

        # ---- функции потерь ----
        # BCEWithLogitsLoss: принимает логиты [B, n_vertices] и метки [B, n_vertices]
        # TODO: calculate coef automatically
        # If pos_weight very high, model start predict everything as 1 
        # Initial: 100
        pos_weight = torch.tensor([15.0], device=self.device)
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # DiceLoss из MONAI: мы сами передаём ему [B, 1, N] → поэтому sigmoid=False
        self.dice_fn = DiceLoss(include_background=False, sigmoid=False)

        # ---- метрики (они принимают прогнозы [B, N] и метки [B, N]) ----
        self.train_metrics = nn.ModuleDict({
            "acc":  Accuracy(task="binary"),
            "dice": Dice(),
            "MIoU": BinaryJaccardIndex()
        })
        self.val_metrics   = deepcopy(self.train_metrics)
        self.test_metrics  = deepcopy(self.train_metrics)

        self.save_hyperparameters(args)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x):
        # Ожидаем, что self.model(x) вернёт логиты размера [B, n_vertices]
        return self.model(x)

    def shared_step(self, batch, batch_idx, stage: str, strategy: str='full'):
        """
        Общий код для train/val/test.
        batch = (x, y, ...), где
          x: входные признаки (→ self.model(x) даёт [B, n_vertices])
          y: бинарные метки {0,1} [B, n_vertices]
          остальные элементы батча игнорируем
        stage: "train" | "val" | "test"
        """
        # Распаковываем первые два элемента кортежа
        x, y = batch
        y = y.float()  # приводим к float для BCE

        B, H, N = y.shape

        # 1) считаем логиты
        if stage == "train":
            self.model.train()
            preds_logits = self.model(x)  # [B, H * N]
        else:
            self.model.eval()
            with torch.no_grad():
                preds_logits = self.model(x)  # [B, H * N]

        # Convert: [B,H*N] -> [B, H, N]
        preds_logits = preds_logits.view(B, H, N)
        # 2) calculate loss by using patches
        per_sample_losses = []
        for i in range(B):
            for hemi in range(H):
                y_i = y[i, hemi, :]            # [N]
                p_i = preds_logits[i, hemi, :] # [N]

                pos_idxs = (y_i > 0).nonzero(as_tuple=False).flatten()

                if pos_idxs.numel() == 0:
                    bce_c  = self.bce_fn(p_i, y_i)         # BCE on raw logits
                    
                    p_sig = torch.sigmoid(p_i)                  # [patch_size]
                    p_sig = p_sig.unsqueeze(0).unsqueeze(0)     # [1, 1, patch_size]
                    y_chan = y_i.unsqueeze(0).unsqueeze(0) 
                    dice_c = self.dice_fn(p_sig, y_chan)         # DiceLoss на сигмоиде и метках

                    per_sample_losses.append(bce_c + dice_c)
                  
                elif pos_idxs.numel() > 0:
                    if strategy == 'patch':
                        # старое поведение: один общий интервал
                        min_idx = pos_idxs.min().item()
                        max_idx = pos_idxs.max().item()
                        start = max(min_idx - self.patch_margin, 0)
                        end   = min(max_idx + self.patch_margin + 1, N)

                        p_crop = p_i[start:end]
                        y_crop = y_i[start:end]

                        p_sig = torch.sigmoid(p_crop)               # [patch_size]
                        p_sig = p_sig.unsqueeze(0).unsqueeze(0)     # [1,1,patch_size]
                        y_chan = y_crop.unsqueeze(0).unsqueeze(0)   # [1,1,patch_size]

                        bce_c  = self.bce_fn(p_crop, y_crop)         # BCE на «сырых» логитах и метках
                        dice_c = self.dice_fn(p_sig, y_chan)         # DiceLoss на сигмоиде и метках

                        per_sample_losses.append(bce_c + dice_c)

                    else:            
                        sorted_pos = pos_idxs.sort().values  # строго возрастающий список позиций {…}
                        clusters = []
                        cluster = [sorted_pos[0].item()]
                        for idx in sorted_pos[1:]:
                            idx = idx.item()
                            if idx - cluster[0] <= 2*self.patch_margin:
                                cluster.append(idx)
                            else:
                                clusters.append(cluster)
                                cluster = [idx]
                        clusters.append(cluster)
                        # print(f"Sample {i}: found {len(clusters)} clusters")
                        for cluster in clusters:
                            min_idx, max_idx = cluster[0], cluster[-1]
                            start = max(min_idx - self.patch_margin, 0)
                            end   = min(max_idx + self.patch_margin + 1, N)
                            # print(f"  cluster {cluster[:3]}…{cluster[-3:]}")

                            p_crop = p_i[start:end]
                            y_crop = y_i[start:end]
                            p_crop_chan = p_crop.unsqueeze(0).unsqueeze(0)
                            y_crop_chan = y_crop.unsqueeze(0).unsqueeze(0)

                            bce_c  = self.bce_fn(p_crop, y_crop)
                            dice_c = self.dice_fn(torch.sigmoid(p_crop_chan), y_crop_chan)
                            per_sample_losses.append(bce_c + dice_c)
                else:
                    continue  # если в y_i нет ни одной «1»

        if len(per_sample_losses) > 0:
            loss = torch.stack(per_sample_losses).mean()
        else:
            # fallback: если во всём батче нет ни одной «1»
            bce_full  = self.bce_fn(preds_logits, y)
            preds_chan = preds_logits.unsqueeze(1)  # [B,1,N]
            y_chan     = y.unsqueeze(1)             # [B,1,N]
            dice_full  = self.dice_fn(torch.sigmoid(preds_chan), y_chan)
            loss = bce_full + dice_full

        # 3) отладочный вывод (раз в эпоху, первый батч train)
        if batch_idx == 0 and stage == "train":
            num_active = len(per_sample_losses)
            print(
                f"\n[{stage.upper()} step {self.global_step}] "
                f"patch-loss(avg) = {loss.item():.4f} "
                f"(активных патчей: {num_active} из {B})"
            )
            sum_pos  = y.sum().item()
            sum_prob = torch.sigmoid(preds_logits).sum().item()
            print(f"[DEBUG] sum(y_all) = {sum_pos:.0f}, sum(sigmoid_all) = {sum_prob:.1f}")

        # 4) считаем метрики на полном векторе [B, N]
        preds_prob = torch.sigmoid(preds_logits)
        metrics = (
            self.train_metrics if stage == "train" else
            self.val_metrics   if stage == "val"   else
            self.test_metrics
        )
        for name, metric in metrics.items():
            value = metric(preds_prob, y.long())
            if stage == "train":
                # логируем только на шаге train
                self.log(name, value, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def training_step(self, batch, batch_idx):
        # shared_step возвращает тензор «loss»
        loss = self.shared_step(batch, batch_idx, stage="train")

        # логируем «train_loss» для агрегированного (epoch) отображения
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # Lightning ждёт ключ «loss» в возвращаемом словаре
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="val")
        # логируем «val_loss»
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss  # возвращаем тензор, Lightning его примет в shared_epoch_end

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="test")
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def shared_epoch_end(self, outputs, stage="train"):
        """
        outputs: list из результатов соответствующего step:
          - для train: list[{"loss": tensor, ...}]  (Lightning упакует словарь)
          - для val/test:  list[tensor]
        Наша задача корректно извлечь тензор loss из каждого элемента.
        """
        # 1) Собираем тензоры loss
        losses = []
        for o in outputs:
            if isinstance(o, dict):
                # training_step вернул dict с ключом "loss"
                losses.append(o["loss"].detach())
            else:
                # validation_step / test_step возвращали тензор
                losses.append(o.detach())
        losses = torch.stack(losses)  # [num_batches]
        avg_loss = losses.mean().item()

        # 2) Собираем метрики
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
            stats[f"{stage}_{name}"] = val

        # 3) Сохраняем историю (для train/val)
        if stage != "test":
            self.history[self.current_epoch] = stats.copy()

        return stats

    def training_epoch_end(self, outputs):
        stats = self.shared_epoch_end(outputs, stage="train")
        print(
            f"\n[TRAIN epoch {stats['epoch']}] "
            f"loss={stats['train_loss']:.4f}, "
            f"acc={stats['train_acc']:.4f}, "
            f"dice={stats['train_dice']:.4f}, "
            f"MIoU={stats['train_MIoU']:.4f}"
        )
        # Лог всех показателей во внешние логгеры (TensorBoard, W&B)
        self.log_dict({k: v for k, v in stats.items() if k != "epoch"},
                      prog_bar=False, logger=True)

    def validation_epoch_end(self, outputs):
        stats = self.shared_epoch_end(outputs, stage="val")
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "=" * 80 + f" {nowtime}")
        print(
            f"[VAL   epoch {stats['epoch']}] "
            f"loss={stats['val_loss']:.4f}, "
            f"acc={stats['val_acc']:.4f}, "
            f"dice={stats['val_dice']:.4f}, "
            f"MIoU={stats['val_MIoU']:.4f}"
        )
        self.log_dict({k: v for k, v in stats.items() if k != "epoch"},
                      prog_bar=False, logger=True)

        # Мониторим «лучший» чекпоинт по val_loss или другой метрике
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
            f"dice={stats['test_dice']:.4f}, "
            f"MIoU={stats['test_MIoU']:.4f}"
        )
        self.log_dict({k: v for k, v in stats.items() if k != "epoch"},
                      prog_bar=False, logger=True)

    def get_history(self):
        return pd.DataFrame(self.history.values())