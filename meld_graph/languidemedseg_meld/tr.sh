#!/bin/bash
#SBATCH --job-name=exp3_no_gnn_full_freeeze
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH -w node-06
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --output=/home/s17gmikh/FCD-Detection/log_outputs/log/reconall_%A_%a.out
#SBATCH --error=/home/s17gmikh/FCD-Detection/log_outputs/error/reconall_%A_%a.err

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s17gmikh@uni-bonn.de

# nvidia-smi

mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/log
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/error

source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate FCD-meld
export PATH=/home/s17gmikh/miniconda3/envs/FCD-meld/bin:$PATH
# hash -r
which python

export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/s17gmikh/FCD-Detection/meld_graph
export PYTHONPATH=$(pwd):$PYTHONPATH

# WANDB_MODE=disabled 
# WANDB_MODE=disabled python3 train_Kfold.py --job_name exp1_5_gnn_aug
WANDB_MODE=disabled python3 languidemedseg_meld/train_Kfold.py \
  --config languidemedseg_meld/config/training.yaml \
  --ckpt_path ./languidemedseg_meld/save_model/exp1_3_gnn_aug \
  --job_name exp3_mixed_3_gnn_aug

WANDB_MODE=disabled python3 languidemedseg_meld/train_Kfold.py   --config languidemedseg_meld/config/training.yaml   --job_name exp1
# --ckpt_path ./save_model/exp1_3_gnn_aug   --job_name exp3_mixed_3_gnn_aug

# --ckpt_path ./save_model/exp1_no_gnn_full_aug --job_name exp3_dominant_nognn_nocontrols_noaug_freeze

# --ckpt_path ./save_model/exp1_no_gnn_full_aug --job_name exp3_no_gnn_hemi_lobe
# --job_name exp1_no_gnn_full_aug
# --ckpt_path /home/s17gmikh/FCD-Detection/meld_graph/LanGuideMedSeg-MICCAI2023/save_model/exp1.ckpt
# --ckpt_path /home/s17gmikh/FCD-Detection/meld_graph/LanGuideMedSeg-MICCAI2023/save_model/exp1_loss.ckpt
# --meld_check True