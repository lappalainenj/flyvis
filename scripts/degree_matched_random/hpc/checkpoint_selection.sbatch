#!/bin/bash
#SBATCH --job-name=flyvis-ckpt
#SBATCH -p d3
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH -t 8:00:00
#SBATCH -o /home/janne.lappalainen/hpc_runs/logs/ckpt-%j.out
#SBATCH -e /home/janne.lappalainen/hpc_runs/logs/ckpt-%j.err

# Scores every checkpoint of every member of an ensemble by EPE on the common
# protocol, to separate a training deficit from a checkpoint-selection artifact.
#
# Both ensembles trained here select their best checkpoint by l2norm, because
# that is what training records; the published models were selected by EPE. So
# this applies to the null ensemble as much as to the reference one.
#
# Usage: sbatch checkpoint_selection.sbatch <ensemble> <n_members> [reference_gap]
#
#   sbatch checkpoint_selection.sbatch flow/0310 48 0.0500
#   sbatch checkpoint_selection.sbatch flow/0301 6

set -euo pipefail

ENSEMBLE=${1:?ensemble required, e.g. flow/0310}
N=${2:?number of members required}
GAP=${3:-}

REPO=$HOME/hpc_runs/flyvis-dmn
export FLYVIS_ROOT_DIR=$HOME/hpc_runs/flyvis-dmn-data
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate flyvis-dmn

nvidia-smi --query-gpu=name --format=csv,noheader
cd "$REPO/scripts/degree_matched_random"

TAG=$(echo "$ENSEMBLE" | tr '/' '_')
ARGS=(--ensemble "$ENSEMBLE" --networks "$N"
      --out "$FLYVIS_ROOT_DIR/checkpoint_selection_${TAG}.json")
if [ -n "$GAP" ]; then
    ARGS+=(--reference-gap "$GAP")
fi

python -u checkpoint_selection.py "${ARGS[@]}"
echo "CKPT SELECTION DONE $ENSEMBLE"
