#!/bin/bash
#SBATCH --partition=public-cpu,shared-cpu
#SBATCH --time=4:00:00
#SBATCH --mem=2GB
#SBATCH --output=/home/users/k/kleins/MLproject/surVAE/jobs/

_image_location=/home/users/k/kleins/MLproject/surVAE/container/tensorflow_latest.sif

# Log directory for what you want to monitor:
_log_directory=/home/users/k/kleins/MLproject/surVAE/images/logs/gas

module load GCC/9.3.0 Singularity/3.7.3-Go-1.14
srun singularity exec ${_image_location} tensorboard --bind_all --logdir ${_log_directory}