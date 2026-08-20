#!/bin/bash
# ============================================================
# ContextIdentification weight-init-scale sweep -- job array.
# 7 scale values g (idx 0-6): 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2.
# Each array task runs 3 conditions x 2 label_modes x N_SEEDS seeds for its
# assigned scale, at the fixed epoch count below (more epochs at smaller g):
#
#   idx 0: g=1e-8 -> 128,000 epochs   <- LONGEST task, not idx 6
#   idx 1: g=1e-7 ->  64,000 epochs
#   idx 2: g=1e-6 ->  32,000 epochs
#   idx 3: g=1e-5 ->  16,000 epochs
#   idx 4: g=1e-4 ->   8,000 epochs
#   idx 5: g=1e-3 ->   4,000 epochs
#   idx 6: g=1e-2 ->   2,000 epochs   <- shortest task
#
# idx 0 alone (128k epochs x 120 runs) accounts for ~50% of the total
# epoch-run budget across all 7 tasks combined -- size --time for idx 0,
# not the average across the array.
#
# FILL IN after profiling (run slurm/weight_scale_pos_codes_profile.sh first):
#   N_SEEDS         = 20  (fixed by sweep design; only adjust mem/time)
#   TIME_PER_EPOCH  = read from profile job output (wall time / epochs, for
#                     ONE run of --scale-idx 0 --conditions adaptive
#                     --label-modes feature --n-seeds 1, i.e. the single
#                     most expensive combo)
#   N_RUNS_PER_TASK = 3 conditions * 2 label_modes * N_SEEDS = 120
#   total_time(idx=0) = N_RUNS_PER_TASK * 128000 * TIME_PER_EPOCH * 1.2  (20% buffer)
#   MEM_GB          = peak_rss_gb_from_profile * 1.15
#   Update the #SBATCH lines marked "SET AFTER PROFILING" below.
#
# Also create the log directory once before first submission:
#   mkdir -p slurm/logs
#
# Submit:
#   sbatch --array=0-6 slurm/weight_scale_pos_codes_sweep.sh
# ============================================================

#SBATCH --job-name=weight_scale_pos_codes_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # SET AFTER PROFILING
#SBATCH --mem=8G                     # SET AFTER PROFILING
#SBATCH --time=24:00:00              # SET AFTER PROFILING -- size for idx=0 (128k epochs)
#SBATCH --output=slurm/logs/weight_scale_pos_codes_sweep_%A_%a.out

N_SEEDS=20   # fixed sweep design

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source /path/to/your/env         # FILL IN: this repo has no documented cluster env-setup
                                  # convention yet -- point this at whatever activates
                                  # uv/the right Python environment for your account.

uv run python scripts/context_identification/pos_codes/weight_scale_pos_codes_sweep.py \
    --scale-idx "$SLURM_ARRAY_TASK_ID" \
    --n-seeds "$N_SEEDS" \
    --save-models
