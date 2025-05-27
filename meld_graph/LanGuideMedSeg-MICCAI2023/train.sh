#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --output=/home/s17gmikh/FCD-Detection/log_outputs/log/reconall_%A_%a.out
#SBATCH --error=/home/s17gmikh/FCD-Detection/log_outputs/error/reconall_%A_%a.err

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s17gmikh@uni-bonn.de

# Папки
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/log
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/error

# Окружение
export FREESURFER_HOME=/home/s17gmikh/freesurfer
export FS_LICENSE=/home/s17gmikh/license.txt
export OMP_NUM_THREADS=2

source /home/s17gmikh/freesurfer/SetUpFreeSurfer.sh

# Сразу очищаем FSL-овский python
unalias python 2>/dev/null || true
unset PYTHONPATH
unset PYTHONHOME

source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate MELD-env
export PATH="$CONDA_PREFIX/bin:$PATH"
export FASTSURFER_HOME=/home/s17gmikh/FastSurfer
export PATH=$FASTSURFER_HOME:$PATH

cd /home/s17gmikh/FCD-Detection/meld_graph/LanGuideMedSeg-MICCAI2023
export PYTHONPATH=$(pwd):$PYTHONPATH

export PATH="/home/s17gmikh/miniconda3/envs/MELD-env/bin:$PATH"
hash -r 
which python

python3 train_bonn.py