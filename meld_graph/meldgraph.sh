#!/bin/bash
#SBATCH --job-name=re_desc
#SBATCH --partition=A100medium
#SBATCH --time=24:00:00
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

# source $FREESURFER_HOME/SetUpFreeSurfer.sh
source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh

eval "$(conda shell.bash hook)"
conda activate FCD #meld_graph
export PATH="$CONDA_PREFIX/bin:$PATH"
export FASTSURFER_HOME=/home/s17gmikh/FastSurfer
export PATH=$FASTSURFER_HOME:$PATH
if [ $1 = 'pytest' ]; then
  pytest ${@:2}
else
  python scripts/new_patient_pipeline/$1 ${@:2}
fi
