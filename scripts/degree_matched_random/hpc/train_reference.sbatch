#!/bin/bash
#SBATCH --job-name=flyvis-ref
#SBATCH -p d3
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -o /home/janne.lappalainen/hpc_runs/logs/ref-%A_%a.out
#SBATCH -e /home/janne.lappalainen/hpc_runs/logs/ref-%A_%a.err

# Trains an ensemble on the DEFAULT connectome with the training regime of the
# runs behind the published models, to check that this pipeline reproduces the
# published ensemble (flow/0000) rather than merely resembling it.
#
# Every member has a byte-identical config -- as the published ensemble's 50
# members do. Network and decoder initialization are deterministic, and nothing
# in the training path seeds the global RNG, so members differ only through data
# order and augmentation. That is the same source of ensemble diversity the
# published models have, so no per-member argument is needed.
#
# Usage: sbatch --array=0-49 train_reference.sbatch <ensemble_id> [n_iters] [stop_iter]
#
#   sbatch --array=0-49 train_reference.sbatch 0310

set -euo pipefail

ENSEMBLE=${1:?ensemble id required, e.g. 0310}
N_ITERS=${2:-250000}
STOP_ITER=${3:-200000}
ID=$(printf "%03d" "$SLURM_ARRAY_TASK_ID")

REPO=$HOME/hpc_runs/flyvis-dmn
export FLYVIS_ROOT_DIR=$HOME/hpc_runs/flyvis-dmn-data
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate flyvis-dmn

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO"

python flyvis_cli/training/train_single.py \
    task_name=flow \
    ensemble_and_network_id="${ENSEMBLE}/${ID}" \
    description="reference reproduction of the published training regime" \
    task=task_original \
    scheduler=scheduler_original \
    task.n_iters="$N_ITERS" \
    scheduler.sched_stop_iter="$STOP_ITER" \
    delete_if_exists=true

echo "TRAIN DONE ${ENSEMBLE}/${ID}"
