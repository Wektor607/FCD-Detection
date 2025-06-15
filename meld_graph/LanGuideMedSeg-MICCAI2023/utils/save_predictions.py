import os
import sys
import h5py
import torch
import pytorch_lightning as pl
from scripts.manage_results.move_predictions_to_mgh import move_predictions_to_mgh
from scripts.manage_results.register_back_to_xhemi import register_subject_to_xhemi
from meld_graph.tools_pipeline import get_m

class SavePredictionsCallback(pl.Callback):
    def __init__(self,
                 subjects_dir: str,
                 train_prediction_file: str,
                 val_prediction_file: str,
                 test_prediction_file: str,
                 predictions_output_dir: str,
                 verbose: bool = False):
        super().__init__()
        self.subjects_dir = subjects_dir
        self.train_pred_file = train_prediction_file
        self.val_pred_file   = val_prediction_file
        self.test_pred_file  = test_prediction_file
        self.predictions_output_dir = predictions_output_dir
        self.verbose = verbose

    def _init_storage(self, stage, pl_module):
        setattr(pl_module, f"{stage}_subject_ids", [])
        setattr(pl_module, f"{stage}_preds", [])

    def on_train_epoch_start(self, trainer, pl_module):
        self._init_storage("train", pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._init_storage("val", pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        self._init_storage("test", pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # batch = (x, y, subject_ids)
        loss, preds = outputs  # preds = torch.sigmoid(logits) [B,H,N]
        sids = batch[0][0]
        for sid, p in zip(sids, preds.detach().cpu()):
            pl_module.train_subject_ids.append(sid)
            pl_module.train_preds.append(p.numpy())

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss, preds = outputs
        sids = batch[0][0]
        for sid, p in zip(sids, preds.detach().cpu()):
            pl_module.val_subject_ids.append(sid)
            pl_module.val_preds.append(p.numpy())


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss, preds = outputs
        sids = batch[0][0]
        for sid, p in zip(sids, preds.detach().cpu()):
            pl_module.test_subject_ids.append(sid)
            pl_module.test_preds.append(p.numpy())

    def _save_and_post(self, stage, pl_module):
        # HDF5
        if stage == "train":
            pred_file = self.train_pred_file
        elif stage == "val":
            pred_file = self.val_pred_file
        else:
            pred_file = self.test_pred_file

        os.makedirs(os.path.dirname(pred_file), exist_ok=True)
        with h5py.File(pred_file, 'w') as f:
            subj_ids = getattr(pl_module, f"{stage}_subject_ids")
            preds    = getattr(pl_module, f"{stage}_preds")
            for sid, p in zip(subj_ids, preds):
                grp = f.create_group(sid)
                
                for elem in [(0, "lh"), (1, "rh")]:
                    idx, hemi = elem
                    hemi_grp = grp.create_group(hemi)
                    hemi_grp.create_dataset("cluster_thresholded_salient", data=p[idx], compression='gzip')
        if self.verbose:
            print(f"[Callback] Saved {stage} predictions → {pred_file}")

        if stage=="val":
            failed=[]
            for sid in getattr(pl_module, "val_subject_ids"):
                # STEP 2
                print(get_m(f'Move predictions into volume', sid, 'STEP 2'))
                ok = move_predictions_to_mgh(
                    subject_id=sid,
                    subjects_dir=self.subjects_dir,
                    prediction_file=self.val_pred_file,
                    verbose=self.verbose
                )
                if not ok:
                    failed.append(sid)
                    continue
                # STEP 3
                print(get_m(f'Move prediction back to native space', sid, 'STEP 3'))
                ok = register_subject_to_xhemi(
                    subject_id=sid,
                    subjects_dir=self.subjects_dir,
                    output_dir=self.predictions_output_dir,
                    verbose=self.verbose
                )
                if not ok:
                    failed.append(sid)
            if failed:
                print(get_m(f'Post-processing failed for: {failed}', None, 'ERROR'))
            else:
                print("[Callback] Test post-processing done.")
        # post-process только для теста
        if stage=="test":
            failed=[]
            for sid in getattr(pl_module, "test_subject_ids"):
                # STEP 2
                print(get_m(f'Move predictions into volume', sid, 'STEP 2'))
                ok = move_predictions_to_mgh(
                    subject_id=sid,
                    subjects_dir=self.subjects_dir,
                    prediction_file=self.test_pred_file,
                    verbose=self.verbose
                )
                if not ok:
                    failed.append(sid)
                    continue
                # STEP 3
                print(get_m(f'Move prediction back to native space', sid, 'STEP 3'))
                ok = register_subject_to_xhemi(
                    subject_id=sid,
                    subjects_dir=self.subjects_dir,
                    output_dir=self.predictions_output_dir,
                    verbose=self.verbose
                )
                if not ok:
                    failed.append(sid)
            if failed:
                print(get_m(f'Post-processing failed for: {failed}', None, 'ERROR'))
            else:
                print("[Callback] Test post-processing done.")

    def on_train_epoch_end(self, trainer, pl_module):
        self._save_and_post("train", pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._save_and_post("val", pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._save_and_post("test", pl_module)
