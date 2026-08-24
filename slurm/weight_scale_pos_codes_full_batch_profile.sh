#!/bin/bash
# ============================================================
# Profiling job for the full-batch weight-init-scale sweep (see
# slurm/weight_scale_pos_codes_full_batch_sweep.sh). Single task, no array.
#
# Runs the single most expensive combo of the full sweep: scale idx 0
# (g=1e-8, 64,000 full-batch steps at L=6), one condition, one label mode,
# but ALL 20 SEEDS -- since seeds are now trained together in one shard_map'd
# computation (train_seeds_shard_map() in
# weight_scale_pos_codes_full_batch_sweep.py), --n-seeds 20 IS the real
# per-task workload, not --n-seeds 1 as before this seeds-in-parallel
# refactor. Use the resulting wall time and peak RSS to fill in the "SET
# AFTER PROFILING" #SBATCH lines in
# slurm/weight_scale_pos_codes_full_batch_sweep.sh.
#
# `/usr/bin/time -v` reports elapsed wall time and Maximum resident set
# size directly in the .out file. If unavailable on your cluster, use
# `seff $SLURM_JOB_ID` after the job completes instead.
#
# NOT YET PROFILED ON THE CLUSTER. #SBATCH --mem/--time below are generous
# placeholders. IMPORTANT: a local (8-core, non-HPC) test at n_seeds=20, 300
# steps found the single-device jax.vmap backend (--backend vmap) gave NO
# speedup over the old sequential-per-seed version, but jax.shard_map (this
# script's --backend, below) DID give a real, reproducible ~1.6x speedup,
# plateauing past ~4-5 devices on that 8-core machine:
#   1 device (=vmap):  64.1s   4 devices: 39.7s   5 devices: 38.4s   10 devices: 37.1s
# This was measured on a laptop, not this cluster -- re-run this profiling
# script with N_DEVICES set to a couple different divisors of 20 (edit the
# line below, e.g. 1, 4, 5, 10) before assuming the same plateau point holds
# here, or that --backend shard_map beats --backend vmap at all on this
# hardware/XLA build.
#
# Submit:
#   sbatch slurm/weight_scale_pos_codes_full_batch_profile.sh
# ============================================================

#SBATCH --job-name=weight_scale_pos_codes_full_batch_profile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4   # EDIT AND RE-RUN with other divisors of 20 (1, 5, 10, 20) -- compare, see IMPORTANT note above
#SBATCH --mem=2G            # generous placeholder; this run's peak RSS is what you're measuring
#SBATCH --time=02:00:00     # generous placeholder for a single-condition, full-step-count, all-20-seeds run
#SBATCH --output=slurm/logs/weight_scale_pos_codes_full_batch_profile_%j.out

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source /usr/people/km3199/.local/bin/env

N_DEVICES="$SLURM_CPUS_PER_TASK"   # must evenly divide --n-seeds below; keep in sync with --cpus-per-task above
export XLA_FLAGS="--xla_force_host_platform_device_count=${N_DEVICES}"   # MUST be set before `uv run python` starts

/usr/bin/time -v uv run python scripts/context_identification/pos_codes/weight_scale_pos_codes_full_batch_sweep.py \
    --scale-idx 0 --n-seeds 20 --conditions adaptive --label-modes feature \
    --backend shard_map \
    --data-dir ./data/_profile_weight_scale_pos_codes_full_batch_sweep

# To compare against the vmap backend instead, comment out the export/--backend
# lines above and add --backend vmap to the invocation.
