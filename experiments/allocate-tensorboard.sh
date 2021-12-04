#!/bin/bash
#SBATCH --partition=public-cpu,shared-cpu
#SBATCH --time=4:00:00
#SBATCH --mem=2GB
#SBATCH --output=/home/users/k/kleins/MLproject/funnels/jobs/slurm-%A-%x_%a.out

_image_location=/home/users/k/kleins/MLproject/funnels/container/tensorflow_latest.sif

# Log directory for what you want to monitor:
_log_directory=/home/users/k/kleins/MLproject/funnels/images/logs/

module load GCC/9.3.0 Singularity/3.7.3-Go-1.14
srun singularity exec ${_image_location} tensorboard --bind_all --logdir ${_log_directory}