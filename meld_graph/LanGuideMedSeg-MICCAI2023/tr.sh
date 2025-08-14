#!/bin/bash
#SBATCH --job-name=exp3-lobe+hemi
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
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
conda activate MELD-env
export PATH=/home/s17gmikh/miniconda3/envs/MELD-env/bin:$PATH
# hash -r
which python

export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/s17gmikh/FCD-Detection/meld_graph/LanGuideMedSeg-MICCAI2023
export PYTHONPATH=$(pwd):$PYTHONPATH

# python3 train.py

WANDB_MODE=disabled python3 train_meld_bonn_Kfold.py