#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --partition=h200q
#SBATCH --job-name=test2
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1

set -x
module load python
export TMPDIR=$HOME/tmp
source /home/tkumar/nqsWeight/.gpuvenv/bin/activate

python -u gpuTraining2.py
