#!/bin/bash
# ============================================================
# ContextIdentification weight-init-scale sweep -- job array.
# 7 scale values g x 3 conditions x 2 label_modes = 42 combos (idx 0-41).
# Each array task runs N_SEEDS seeds sequentially for ONE (scale, condition,
# label_mode) combo, at the fixed epoch count for its scale (more epochs at
# smaller g):
#
#   scale_idx 0: g=1e-8 -> 128,000 epochs   <- LONGEST combos, not scale_idx 6
#   scale_idx 1: g=1e-7 ->  64,000 epochs
#   scale_idx 2: g=1e-6 ->  32,000 epochs
#   scale_idx 3: g=1e-5 ->  16,000 epochs
#   scale_idx 4: g=1e-4 ->   8,000 epochs
#   scale_idx 5: g=1e-3 ->   4,000 epochs
#   scale_idx 6: g=1e-2 ->   2,000 epochs   <- shortest combos
#
# Index -> combo mapping (COMBO_ID = $SLURM_ARRAY_TASK_ID, 0-41):
#   SCALE_IDX     = COMBO_ID / 6        (0-6, index into the scale list above)
#   REM           = COMBO_ID % 6
#   CONDITION_IDX = REM / 2             (0=adaptive, 1=adaptive_pos, 2=attn_pos)
#   LABEL_MODE_IDX= REM % 2             (0=feature, 1=position)
#
# PROFILED 2026-08-20 (job 5672099): --scale-idx 0 --n-seeds 1
# --conditions adaptive --label-modes feature (1 run, g=1e-8, 128,000
# epochs) -> Job Wall-clock time 00:15:52 (952s), Memory Utilized 453.30 MB,
# Cores per node 2 @ 75.26% CPU efficiency (~1.5 cores used).
#   TIME_PER_EPOCH = 952s / 128,000 epochs = 0.00744 s/epoch
#   total_time(scale_idx=0 combo) = N_SEEDS(20) * 128,000 * 0.00744 * 1.2 (20% buffer)
#                                  = 22,856s = 6.35h
# Per-combo estimated wall time at N_SEEDS=20 (same 20% buffer), same for
# every condition/label_mode at a given scale since they share its epoch count:
#   scale_idx 0 (g=1e-8, 128k epochs): 6.35h   <- sizes --time below
#   scale_idx 1 (g=1e-7,  64k epochs): 3.17h
#   scale_idx 2 (g=1e-6,  32k epochs): 1.59h
#   scale_idx 3 (g=1e-5,  16k epochs): 0.79h
#   scale_idx 4 (g=1e-4,   8k epochs): 0.40h
#   scale_idx 5 (g=1e-3,   4k epochs): 0.20h
#   scale_idx 6 (g=1e-2,   2k epochs): 0.10h
# NOTE: a single --array submission uses one shared --time for every combo,
# so scale_idx 1-6 combos will be allocated far more walltime than they
# need. Fine on a scheduler that only queues by core availability (as here);
# reconsider if your cluster's fair-share instead penalizes requested (not
# used) walltime.
#
# Also create the log directory once before first submission:
#   mkdir -p slurm/logs
#
# Submit:
#   sbatch --array=0-41 slurm/weight_scale_pos_codes_sweep.sh
# ============================================================

#SBATCH --job-name=weight_scale_pos_codes_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # profiling showed ~1.5 cores used at cpus-per-task=1
#SBATCH --mem=2G                     # profiling showed 453MB peak for 1 run; margin for repeated-run overhead across 20 sequential runs/task
#SBATCH --time=08:00:00              # sized for scale_idx=0 combos (6.35h estimate + margin); see per-scale estimates above
#SBATCH --output=slurm/logs/weight_scale_pos_codes_sweep_%A_%a.out

N_SEEDS=20   # fixed sweep design

CONDITIONS=(adaptive adaptive_pos attn_pos)
LABEL_MODES=(feature position)

SCALE_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
CONDITION=${CONDITIONS[$(( REM / 2 ))]}
LABEL_MODE=${LABEL_MODES[$(( REM % 2 ))]}

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source "$HOME/.local/bin/env"   # puts uv on PATH (default uv install location); adjust for your cluster

uv run python scripts/context_identification/pos_codes/weight_scale_pos_codes_sweep.py \
    --scale-idx "$SCALE_IDX" \
    --conditions "$CONDITION" \
    --label-modes "$LABEL_MODE" \
    --n-seeds "$N_SEEDS" \
    --save-models
