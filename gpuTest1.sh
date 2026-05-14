#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --partition=h200q
#SBATCH --job-name=test
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1
#SBATCH --array=0-9

set -x
module load python
export TMPDIR=$HOME/tmp
source /home/tkumar/nqsWeight/.gpuvenv/bin/activate

python -u gpuTraining1.py --seed=$SLURM_ARRAY_TASK_ID 
