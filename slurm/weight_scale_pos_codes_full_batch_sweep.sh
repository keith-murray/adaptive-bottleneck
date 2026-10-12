#!/bin/bash
# ============================================================
# ContextIdentification weight-init-scale sweep -- job array, full-batch GD.
# 7 scale values g x 3 conditions x 2 label_modes = 42 combos (idx 0-41).
# Each array task runs all N_SEEDS seeds for ONE (scale, condition,
# label_mode) combo TOGETHER via jax.shard_map, partitioning seeds across
# N_DEVICES CPU devices (see train_seeds_shard_map() in
# weight_scale_pos_codes_full_batch_sweep.py) -- not sequentially, and not a
# single-device jax.vmap either (see IMPORTANT note below for why). One
# jax.jit compile per combo (not per seed), at the fixed full-batch step
# count for its scale (more steps at smaller g, c.f. STEPS_FOR_SCALE in
# weight_scale_pos_codes_full_batch_sweep.py):
#
#   scale_idx 0: g=1e-8 -> 64,000 steps   <- LONGEST combos, not scale_idx 6
#   scale_idx 1: g=1e-7 -> 32,000 steps
#   scale_idx 2: g=1e-6 -> 16,000 steps
#   scale_idx 3: g=1e-5 ->  8,000 steps
#   scale_idx 4: g=1e-4 ->  4,000 steps
#   scale_idx 5: g=1e-3 ->  2,000 steps
#   scale_idx 6: g=1e-2 ->  1,000 steps   <- shortest combos
#
# Index -> combo mapping (COMBO_ID = $SLURM_ARRAY_TASK_ID, 0-41), identical
# to weight_scale_pos_codes_sweep.sh's scheme:
#   SCALE_IDX     = COMBO_ID / 6        (0-6, index into the scale list above)
#   REM           = COMBO_ID % 6
#   CONDITION_IDX = REM / 2             (0=adaptive, 1=adaptive_pos, 2=attn_pos)
#   LABEL_MODE_IDX= REM % 2             (0=feature, 1=position)
#
# NOT YET PROFILED ON THE CLUSTER -- run
# slurm/weight_scale_pos_codes_full_batch_profile.sh (scale_idx=0, the
# longest combo, all 20 seeds batched -- that's the real per-task workload
# now) first and replace the #SBATCH --time/--mem/--cpus-per-task below with
# real numbers, the same way weight_scale_pos_codes_sweep.sh's header was
# derived.
#
# IMPORTANT: a local (8-core, non-HPC) test at L=6, n_seeds=20, 300 steps
# found the single-device jax.vmap backend gave NO speedup over the old
# sequential-per-seed version (~64s either way -- tiny per-step matmuls,
# plausibly dispatch- rather than compute-bound). jax.shard_map (this
# script's backend), which partitions seeds across genuinely separate
# per-device XLA programs instead of relying on intra-op threading of one
# big op, DID show a real, reproducible speedup on that same machine:
#   1 device (=vmap):  64.1s, 323% CPU
#   4 devices:          39.7s, 522% CPU
#   5 devices:          38.4s, 504% CPU
#   10 devices:         37.1s, 526% CPU  <- plateaus past ~4-5 devices (8 physical cores)
# This is why N_DEVICES below defaults to a shard_map value, not vmap's
# single device. But this was measured on a laptop, not this cluster's CPU
# backend/topology -- do NOT assume the same plateau point or speedup ratio
# holds here. Profile at a couple of N_DEVICES values (divisors of 20:
# 1/2/4/5/10/20) before trusting the number below.
#
# Also create the log directory once before first submission:
#   mkdir -p slurm/logs
#
# Submit:
#   sbatch --array=0-41 slurm/weight_scale_pos_codes_full_batch_sweep.sh
# ============================================================

#SBATCH --job-name=weight_scale_pos_codes_full_batch_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20            # SET AFTER PROFILING -- must divide N_SEEDS=20 evenly; see IMPORTANT note above for local (non-HPC) plateau point
#SBATCH --mem=40G                     # SET AFTER PROFILING -- placeholder, full population (27,684 x 7 x 6 floats, more with pos codes) kept resident for the whole array task
#SBATCH --time=04:00:00              # SET AFTER PROFILING -- placeholder sized for scale_idx=0 combos
#SBATCH --output=slurm/logs/weight_scale_pos_codes_full_batch_sweep_%A_%a.out

N_SEEDS=20   # fixed sweep design
N_DEVICES="$SLURM_CPUS_PER_TASK"   # must evenly divide N_SEEDS -- keep in sync with --cpus-per-task above
export XLA_FLAGS="--xla_force_host_platform_device_count=${N_DEVICES}"   # MUST be set before `uv run python` starts -- JAX's CPU backend exposes only 1 device by default, and this can't be changed after it initializes

CONDITIONS=(adaptive adaptive_pos attn_pos)
LABEL_MODES=(feature position)

SCALE_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
REM=$(( SLURM_ARRAY_TASK_ID % 6 ))
CONDITION=${CONDITIONS[$(( REM / 2 ))]}
LABEL_MODE=${LABEL_MODES[$(( REM % 2 ))]}

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source "$HOME/.local/bin/env"   # puts uv on PATH (default uv install location); adjust for your cluster

uv run python scripts/context_identification/pos_codes/weight_scale_pos_codes_full_batch_sweep.py \
    --scale-idx "$SCALE_IDX" \
    --conditions "$CONDITION" \
    --label-modes "$LABEL_MODE" \
    --n-seeds "$N_SEEDS" \
    --backend shard_map \
    --save-models
