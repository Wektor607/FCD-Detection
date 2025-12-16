import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from torch.utils.data import DataLoader

import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

from sklearn.model_selection import KFold
from transformers import AutoTokenizer

import utils.config as config

from utils.data import EpilepDataset
from utils.utils import LesionOversampleSampler
from engine.wrapper import LanGuideMedSegWrapper

# Ensure repository root is on sys.path so imports like `meld_graph` resolve
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch.multiprocessing.set_sharing_strategy("file_system")


SEED = 42

# Determinism + reasonable TF32 settings for fast GPUs (A100 etc.)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_float32_matmul_precision("medium")


def worker_init_fn(worker_id: int):
    """Initialize RNGs for DataLoader workers reproducibly."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cfg():
    parser = argparse.ArgumentParser(description="Language-guide Medical Image Segmentation")
    parser.add_argument("--config", default="./config/training.yaml", type=str, help="config file")
    parser.add_argument("--ckpt_path", default=None, type=str, help="optional checkpoint to load")
    parser.add_argument("--job_name", default=None, type=str, help="optional job name to include in filenames")
    cli = parser.parse_args()
    if cli.config is None:
        parser.error("--config is required")
    cfg = config.load_cfg_from_cfg_file(cli.config)
    cfg.ckpt_path = cli.ckpt_path
    cfg.job_name = cli.job_name
    return cfg


class FreezeDecoderCallback(Callback):
    """Optional callback to freeze/unfreeze decoder layers for the first N epochs."""

    def __init__(self, unfreeze_at_epoch: int = 10):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def on_train_start(self, trainer, pl_module):
        for name, param in pl_module.model.named_parameters():
            if any(k in name for k in [
                "decoder_conv_layers", "ds_heads", "ds_dist_heads",
                "hemi_classification_head", "final_lin", "dist_lin"
            ]):
                param.requires_grad = False
        print(f"[INFO] Decoder frozen until epoch {self.unfreeze_at_epoch}")

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_at_epoch:
            for name, param in pl_module.model.named_parameters():
                if any(k in name for k in [
                    "decoder_conv_layers", "ds_heads", "ds_dist_heads",
                    "hemi_classification_head", "final_lin", "dist_lin"
                ]):
                    param.requires_grad = True
            print(f"[INFO] Decoder unfrozen at epoch {trainer.current_epoch}")


def set_seed(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_filename(args, ckpt_path: str, fold: int) -> str:
    """Derive a descriptive filename for model checkpoints."""
    job_name = args.job_name or os.environ.get("JOB_NAME") or os.environ.get("SLURM_JOB_NAME") or os.environ.get("job_name")
    if job_name:
        return f"{job_name}_fold{fold + 1}"
    if ckpt_path is None:
        return f"{args.model_save_filename}_fold{fold + 1}"
    return os.path.splitext(os.path.basename(ckpt_path))[0]


def make_dataloaders(args, tokenizer, cohort, train_fold_ids: List[str], val_fold_ids: List[str], fold_seed: int):
    print("INFO Dataset name: ", args.csv_path)
    ds_train = EpilepDataset(csv_path=args.csv_path, 
                tokenizer=tokenizer, 
                feature_path=args.feature_path, 
                subject_ids=train_fold_ids, 
                cohort=cohort, 
                max_length=args.max_len, 
                text_emb=True,
#!!!!!!!!!!!!!!!!!!!!!!! Make like a parameter!
                text_prob_json="/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/mixed/train_prob.json")
    ds_valid = EpilepDataset(csv_path=args.csv_path, 
                tokenizer=tokenizer, 
                feature_path=args.feature_path, 
                subject_ids=val_fold_ids, 
                cohort=cohort, 
                max_length=args.max_len, 
                text_emb=True,
                text_prob_json="/home/s17gmikh/FCD-Detection/meld_graph/data/preprocessed/mixed/train_prob.json")
    hc_set = set([sid for sid in train_fold_ids if sid.split("_")[3].startswith("C")])
    labels = [0 if sid in hc_set else 1 for sid in ds_train.subject_ids]
    sampler = LesionOversampleSampler(labels, seed=fold_seed)
    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, sampler=sampler, num_workers=args.train_batch_size, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.valid_batch_size, pin_memory=True, worker_init_fn=worker_init_fn, persistent_workers=True)
    
    return dl_train, dl_valid

if __name__ == "__main__":
    args = get_cfg()

    eva, cohort, exp_flags = config.inference_config()
    # wandb_logger = WandbLogger(project=args.project_name, log_model=True)
    
    df = pd.read_csv(args.split_path, sep=",")
    # CHANGE BACK WITH HEALTHY CONTROLS #################################
    train_ids = df[df.split == "trainval"]["subject_id"].tolist()
    # train_ids = df[(df.split == "trainval") & (df["subject_id"].str.contains("FCD"))]["subject_id"].tolist()
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    for fold, (train_index, val_index) in enumerate(kf.split(train_ids)):
        # optional: process only specific folds (was earlier skipping folds < 4)
        if fold < 3:
            continue

        print(f"Fold {fold + 1}/{n_splits}")
        fold_seed = SEED + fold
        print(f"[INFO] Using fold-specific seed: {fold_seed}")
        set_seed(fold_seed)

        ckpt_path = f"{args.ckpt_path}_fold{fold + 1}.ckpt"
        filename = get_filename(args, ckpt_path, fold)

        # experiment flags (affects tokenization / model internals)
        att_mechanism = False
        text_emb = False
        for exp, flags in exp_flags.items():
            if exp in (args.job_name or ""):
                att_mechanism = flags.get("self_att_mechanism", False)
                text_emb = flags.get("text_emb", False)
                print(f"[INFO] Experiment '{exp}' flags: self_att_mechanism={att_mechanism}, text_emb={text_emb}")
                break

        tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True) if text_emb else None
        train_fold_ids = [train_ids[i] for i in train_index]
        val_fold_ids_full = [train_ids[i] for i in val_index]
        val_fold_ids = [sid for sid in val_fold_ids_full if "FCD" in sid]
        val_fold_ids_controls = [sid for sid in val_fold_ids_full if "_C_" in sid]
        train_fold_ids.extend(val_fold_ids_controls)
        print(f"Train IDs: {len(train_fold_ids)}")
        print(f"Validation IDs: {len(val_fold_ids)}")

        dl_train, dl_valid = make_dataloaders(args, tokenizer, cohort, train_fold_ids, val_fold_ids, fold_seed)

        # trainer device setup
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = "auto"
            strategy = "ddp_sharded"
        else:
            accelerator = "cpu"
            args.device = "cpu"
            devices = 1
            strategy = None

        model = LanGuideMedSegWrapper(args, eva, fold_number=fold, att_mechanism=att_mechanism, text_emb=text_emb)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print(f"[INFO] Attempting to load pretrained weights from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            src_sd = ckpt.get("state_dict", ckpt if isinstance(ckpt, dict) else None)
            if isinstance(src_sd, dict):
                try:
                    res = model.load_state_dict(src_sd, strict=False)
                    print(f"[INFO] load_state_dict into wrapper returned: {res}")
                except Exception as e:
                    print(f"[WARN] load_state_dict into wrapper failed: {e}")
            else:
                print("[WARN] checkpoint does not contain a state_dict-like mapping; skipping direct load into wrapper.")
        else:
            print(f"[INFO] No checkpoint file found at {ckpt_path}; training from scratch.")

        print(f"[INFO] Model checkpoint filename pattern: {filename}")

        model_ckpt = ModelCheckpoint(dirpath=args.model_save_path, filename=filename, monitor="val_dice", save_top_k=1, mode="max", verbose=True)
        early_stopping = EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")

        # FREEZE DECODER CALLBACK
        freeze_cb = FreezeDecoderCallback(unfreeze_at_epoch=5)

        trainer = pl.Trainer(min_epochs=args.min_epochs, 
                            max_epochs=args.max_epochs, 
                            accelerator=accelerator, 
                            devices=devices, 
                            callbacks=[model_ckpt, early_stopping, 
                            # ],
                            freeze_cb], 
                            # logger=wandb_logger,
                            enable_progress_bar=True)

        print("start training")
        trainer.fit(model, dl_train, dl_valid)
        print("done training")

        break

