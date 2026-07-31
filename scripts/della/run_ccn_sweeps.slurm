#!/bin/bash
#SBATCH --job-name=ccn-sweeps
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=cpu          # TODO: set to your allocation's partition (e.g. gpu partition, see below)
# #SBATCH --gres=gpu:1                 # TODO: uncomment (+ switch partition) if you want these on GPU
# #SBATCH --account=your_allocation    # TODO: set if your group requires an account/allocation name
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=YOUR_NETID@princeton.edu   # TODO

# One array task per independent sweep script -- they don't share state so
# they run fully in parallel as separate SLURM tasks.
#
# Submit from the repo root:
#   sbatch scripts/della/run_ccn_sweeps.slurm
#
# Check status:
#   squeue -u $USER
#
# Array index -> script mapping:
#   0: fixed_gamma/context_identification_bottleneck.py  -> data/context_identification_full_sweep.json
#   1: online_gamma/online_gamma_20seeds_sweep.py         -> data/online_gamma_20seeds_sweep.json
#   2: pos_codes/pos_codes_comparison_sweep.py            -> data/pos_codes_comparison_sweep.json (slowest: 120 runs x 5000 epochs)

set -euo pipefail

module purge
module load anaconda3/2024.6            # TODO: adjust to whatever anaconda/python module Della exposes (`module avail anaconda`)
conda activate adaptive_bottleneck      # TODO: adjust to your env name
# If you're using uv instead of conda, replace the two lines above with:
# module load python/3.13
# source /path/to/repo/.venv/bin/activate

REPO_ROOT="$SLURM_SUBMIT_DIR"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "Could not locate repo root (pyproject.toml) above $SLURM_SUBMIT_DIR" >&2
    exit 1
fi
mkdir -p "$REPO_ROOT/logs"

# These scripts save/load with paths relative to CWD (e.g. "./data/...", not
# relative to the script file) -- each task MUST cd into its own script
# directory before running, or it'll read/write the wrong data/ folder.
SCRIPT_DIRS=(
    "scripts/context_identification/fixed_gamma"
    "scripts/context_identification/online_gamma"
    "scripts/context_identification/pos_codes"
)
SCRIPTS=(
    "context_identification_bottleneck.py"
    "online_gamma_20seeds_sweep.py"
    "pos_codes_comparison_sweep.py"
)

TASK_DIR="${SCRIPT_DIRS[$SLURM_ARRAY_TASK_ID]}"
TASK_SCRIPT="${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"

echo "[task $SLURM_ARRAY_TASK_ID] $(date): running $TASK_SCRIPT in $TASK_DIR"
cd "$REPO_ROOT/$TASK_DIR"
python "$TASK_SCRIPT"
echo "[task $SLURM_ARRAY_TASK_ID] $(date): done"
