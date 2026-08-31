#!/bin/bash
#SBATCH --job-name=flyvis-dmr
#SBATCH -p d3
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gpus=h200:1
#SBATCH -t 2-00:00:00
#SBATCH -o /home/janne.lappalainen/hpc_runs/logs/dmr-%A_%a.out
#SBATCH -e /home/janne.lappalainen/hpc_runs/logs/dmr-%A_%a.err

# Trains one network per array task on its own degree-matched random connectome,
# with the training regime of the runs behind the published models.
#
# The array task id is both the network id and the rewiring seed, so network
# 0300/003 is the one trained on the connectome rewired with seed 3. The seed is
# stored in the network's config, so each model records the connectome it used.
#
# Usage:
#   sbatch --array=0-5 train_array.sbatch <ensemble_id> [connectome] [n_iters] [stop_iter]
#
#   sbatch --array=0-5 train_array.sbatch 0301                       # full run
#   sbatch --array=0-5 train_array.sbatch 0398 \
#       connectome_locality_preserving_random 100 80                 # trial

set -euo pipefail

ENSEMBLE=${1:?ensemble id required, e.g. 0301}
CONNECTOME=${2:-connectome_locality_preserving_random}
N_ITERS=${3:-250000}
STOP_ITER=${4:-200000}
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

# chkpt_every_epoch is 334 in scheduler_original, which is ~4000 iterations; a
# short trial would checkpoint only at iteration 0, so shorten it for trials
CHKPT_ARG=()
if [ "$N_ITERS" -lt 10000 ]; then
    CHKPT_ARG=(scheduler.chkpt_every_epoch=2)
fi

python flyvis_cli/training/train_single.py \
    task_name=flow \
    ensemble_and_network_id="${ENSEMBLE}/${ID}" \
    description="${CONNECTOME} with rewiring seed ${SLURM_ARRAY_TASK_ID}" \
    network/connectome="${CONNECTOME}" \
    network.connectome.seed="$SLURM_ARRAY_TASK_ID" \
    task=task_original \
    scheduler=scheduler_original \
    task.n_iters="$N_ITERS" \
    scheduler.sched_stop_iter="$STOP_ITER" \
    delete_if_exists=true \
    "${CHKPT_ARG[@]}"

echo "TRAIN DONE ${ENSEMBLE}/${ID}"
