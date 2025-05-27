from utils.model import LanGuideMedSeg
from monai.losses import DiceCELoss
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch.nn.functional as F
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt


class LanGuideMedSegWrapper(pl.LightningModule):

    def __init__(self, args):
        
        super(LanGuideMedSegWrapper, self).__init__()
        
        self.model = LanGuideMedSeg(args.bert_type, args.meld_script_path, 
                                    args.feature_path, args.output_dir,
                                    # args.vision_type, 
                                    args.project_dim, args.device)
        self.lr = args.lr
        self.history = {}
        
        self.loss_fn = DiceCELoss()

        metrics_dict = {"acc":Accuracy(task='binary'),"dice":Dice(),"MIoU":BinaryJaccardIndex()}
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics = deepcopy(self.train_metrics)
        self.test_metrics = deepcopy(self.train_metrics)
        
        self.save_hyperparameters(args)

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.model.parameters(),lr = self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =200, eta_min=1e-6)

        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
        
    def forward(self,x):
       
       return self.model.forward(x)
    
    def save_pred_and_mask(self, preds, y, step, batch_idx, output_dir="./debug_preds"):
        os.makedirs(output_dir, exist_ok=True)
        preds_np = preds.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        B = preds_np.shape[0]
        for i in range(B):
            pred_slice = preds_np[i, 0, preds_np.shape[2] // 2]
            y_slice = y_np[i, 0, y_np.shape[2] // 2]

            fig, axs = plt.subplots(1, 2, figsize=(6, 3))
            axs[0].imshow(pred_slice, cmap='viridis', vmin=0, vmax=1)
            axs[0].set_title('Prediction')
            axs[1].imshow(y_slice, cmap='gray')
            axs[1].set_title('Ground Truth')

            for ax in axs:
                ax.axis('off')

            fname = f"step_{step}_batch_{batch_idx}_idx_{i}.png"
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')
            plt.close(fig)


    def shared_step(self,batch,batch_idx):

        x, y = batch
        
        preds = self(x)
        if y.dim() == 4:
            y = y.unsqueeze(1)
        
        if y.shape[2:] != preds.shape[2:]:
            y = F.interpolate(y, size=preds.shape[2:], mode='nearest')

        # print("preds stats:", preds.min().item(), preds.max().item(), preds.mean().item())
        loss = self.loss_fn(preds, y)

        # ===== debug visualization =====
        if self.global_step < 10:  # сохранить только первые несколько шагов
            self.save_pred_and_mask(preds, y, step=self.global_step, batch_idx=batch_idx)

        return {'loss': loss, 'preds': preds, 'y': y}    
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def test_step(self, batch, batch_idx):
        return self.shared_step(batch,batch_idx)
    
    def predict_step(self, batch, batch_idx):
        if isinstance(batch,list) and len(batch)==2:
            return self(batch[0])
        else:
            return self(batch)
        
    def shared_step_end(self,outputs,stage):
        preds = outputs['preds']
        target = outputs['y']

        target = target.long()

        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        for name in metrics:
            step_metric = metrics[name](preds, target).item()
            if stage=="train":
                self.log(name,step_metric,prog_bar=True)
        return outputs["loss"].mean()
        
    def training_step_end(self, outputs):
        return {'loss':self.shared_step_end(outputs,"train")}
            
    def validation_step_end(self, outputs):
        return {'val_loss':self.shared_step_end(outputs,"val")}
            
    def test_step_end(self, outputs):
        return {'test_loss':self.shared_step_end(outputs,"test")}
            
    def shared_epoch_end(self,outputs,stage="train"):
        metrics = self.train_metrics if stage=="train" else (
            self.val_metrics if stage=="val" else self.test_metrics)
        
        epoch = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor([t[(stage+"_loss").replace('train_','')] for t in outputs])).item()
        dic = {"epoch":epoch,stage+"_loss":stage_loss}
        
        for name in metrics:
            epoch_metric = metrics[name].compute().item() 
            metrics[name].reset()
            dic[stage+"_"+name] = epoch_metric 
        if stage!='test':
            self.history[epoch] = dict(self.history.get(epoch,{}),**dic)    
        return dic 
    
    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="train")
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="val")
        self.print_bar()
        self.print(dic)
        dic.pop("epoch",None)
        self.log_dict(dic, logger=True)
        
        #log when reach best score
        ckpt_cb = self.trainer.checkpoint_callback
        monitor = ckpt_cb.monitor 
        mode = ckpt_cb.mode 
        arr_scores = self.get_history()[monitor]
        best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
        if best_score_idx==len(arr_scores)-1:   
            self.print("<<<<<< reach best {0} : {1} >>>>>>".format(
                monitor,arr_scores[best_score_idx]),file = sys.stderr)
    
    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs,stage="test")
        dic.pop("epoch",None)
        self.print(dic)
        self.log_dict(dic, logger=True)
        
    def get_history(self):
        return pd.DataFrame(self.history.values()) 
    
    def print_bar(self): 
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n"+"="*80 + "%s"%nowtime)