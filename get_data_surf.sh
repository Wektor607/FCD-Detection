#!/bin/bash
#SBATCH --job-name=reconall_array
#SBATCH --partition=A40short
#SBATCH --time=8:00:00
#SBATCH --gpus=0
#SBATCH --cpus-per-task=1
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
export SUBJECTS_DIR=/home/s17gmikh/FCD-Detection/data/ds004199/freesurfer_subjects
export FS_LICENSE=/home/s17gmikh/license.txt
export OMP_NUM_THREADS=2

# source $FREESURFER_HOME/SetUpFreeSurfer.sh
source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh
conda activate FCD

python3 /home/s17gmikh/FCD-Detection/LanGuideMedSeg-MICCAI2023/utils/dataset_bonn.py