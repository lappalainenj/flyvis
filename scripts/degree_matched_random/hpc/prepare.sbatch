#!/bin/bash
#SBATCH --job-name=flyvis-dmr-prep
#SBATCH -p d3
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /home/janne.lappalainen/hpc_runs/logs/dmr-prep-%j.out
#SBATCH -e /home/janne.lappalainen/hpc_runs/logs/dmr-prep-%j.err

# Builds and validates one degree-matched random connectome per ensemble member.
#
# Run this before the training array: each connectome takes ~20 s to build and
# datamate caches it on disk, so pre-building here means the array jobs cannot
# race each other writing the same directory.
#
# Usage: sbatch prepare.sbatch [n_networks] [mode]

set -euo pipefail

N=${1:-6}
MODE=${2:-filter_symmetry}
SEEDS=$(seq 0 $((N - 1)) | tr '\n' ' ')

REPO=$HOME/hpc_runs/flyvis-dmn
export FLYVIS_ROOT_DIR=$HOME/hpc_runs/flyvis-dmn-data
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate flyvis-dmn

cd "$REPO"
python -u scripts/degree_matched_random/validate_rewiring.py \
    --seeds $SEEDS \
    --mode "$MODE" \
    --out "$FLYVIS_ROOT_DIR/rewiring_report_${MODE}.json"

echo "PREPARE DONE"
