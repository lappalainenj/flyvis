#!/bin/bash
#SBATCH --job-name=flyvis-eval
#SBATCH -p d3
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH -o /home/janne.lappalainen/hpc_runs/logs/eval-%j.out
#SBATCH -e /home/janne.lappalainen/hpc_runs/logs/eval-%j.err

# Evaluates ensembles on one common protocol. CPU only, but long enough that it
# belongs on a compute node rather than the login node.
#
# Usage: sbatch evaluate.sbatch <out_name> <val_set> <ensemble> [ensemble ...]
#
#   sbatch evaluate.sbatch eval_0310_fold fold flow/0310 flow/0000

set -euo pipefail

OUT=${1:?output name required, e.g. eval_0310}
VAL_SET=${2:?val set required: config|original|fold|intersection}
shift 2
ENSEMBLES=("$@")

REPO=$HOME/hpc_runs/flyvis-dmn
export FLYVIS_ROOT_DIR=$HOME/hpc_runs/flyvis-dmn-data
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate flyvis-dmn

cd "$REPO"
python -u scripts/degree_matched_random/evaluate_ensemble.py \
    --ensembles "${ENSEMBLES[@]}" \
    --val-set "$VAL_SET" \
    --min-iterations 250000 \
    --out "$FLYVIS_ROOT_DIR/${OUT}.json"

echo "EVAL DONE"
