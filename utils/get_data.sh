#!/bin/bash
#SBATCH --partition=A40medium
#SBATCH --time=3:00:00
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=2

#SBATCH --output=/home/s17gmikh/FCD-Detection/log_outputs/log/get_data_Benchmark_%j.output
#SBATCH --error=/home/s17gmikh/FCD-Detection/log_outputs/error/get_data_Benchmark_%j.error

#SBATCH --mail-type=ALL
#SBATCH --mail-user=s17gmikh@uni-bonn.de

mkdir -p /home/s17gmikh/FCD-Detection/log_outputs
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/log
mkdir -p /home/s17gmikh/FCD-Detection/log_outputs/error

source /home/s17gmikh/miniconda3/etc/profile.d/conda.sh

conda activate fcd_env

cd /home/s17gmikh/FCD-Detection

module load Python
module load CUDA/11.7.0
module purge

# export OMP_NUM_THREADS=5

python3 utils/get_reports.py