#!/bin/bash
#SBATCH --job-name=prepr
#SBATCH --partition=A100short
#SBATCH --time=8:00:00
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --output=/home/s17gmikh/FCD-Detection/log_outputs/log/prepr_%A_%a.out
#SBATCH --error=/home/s17gmikh/FCD-Detection/log_outputs/error/prepr_%A_%a.err

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=s17gmikh@uni-bonn.de

# Создание папок логов
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/log
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/error

# Настройка окружения FreeSurfer (если используется)
export FREESURFER_HOME=/home/s17gmikh/freesurfer
export FS_LICENSE=/home/s17gmikh/license.txt
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# FastSurfer (если используется)
export FASTSURFER_HOME=/home/s17gmikh/FastSurfer
export PATH=$FASTSURFER_HOME:$PATH

# Активация conda-окружения
source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate MELD-env
export PATH="$CONDA_PREFIX/bin:$PATH"

# Общие переменные
export OMP_NUM_THREADS=2
export PYTHONPATH=/home/s17gmikh/FCD-Detection/meld_graph:$PYTHONPATH

# Перемещение в рабочую директорию
cd /home/s17gmikh/FCD-Detection/meld_graph

# Отладка: покажем, какой Python используется
which python
python -c "import sys; print(sys.executable)"

# Основной вызов
if [ "$1" = 'pytest' ]; then
  pytest "${@:2}"
else
  python scripts/new_patient_pipeline/"$1" "${@:2}"
fi
