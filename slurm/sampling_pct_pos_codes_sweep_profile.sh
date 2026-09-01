#!/bin/bash
# ============================================================
# Profiling job for the sampling-percentage sweep (see
# slurm/sampling_pct_pos_codes_sweep.sh). Single task, no array.
#
# Runs the worst-case combo of the full sweep: combo-idx 2 (adaptive_pos,
# feature) -- positional_codes=True (larger in_dim=16, slower per step) AND
# the adaptive branch (joint gamma gradient, slower per step than fixed-gamma)
# combined, both the compute and memory worst case. --n-seeds 1 still runs the
# FULL percentage schedule (log + linear, 20 percentages) since num_steps is
# fixed across percentages by design (per-run compute cost is roughly constant
# regardless of percentage, unlike the old full-batch sweeps where a step's
# cost scaled with dataset size) -- so this one run sweeps through every
# percentage regime including the 90% linear-schedule point, which is also
# where peak memory occurs (numpy fancy-indexing X[train_idx] at that
# percentage very nearly doubles the resident positional-codes population
# array, ~7.5GB -> ~15GB transiently -- see run_sampling_pct_sweep_for_combo's
# docstring).
#
# `/usr/bin/time -v` reports elapsed wall time and Maximum resident set size
# directly in the .out file. If unavailable on your cluster, use `seff
# $SLURM_JOB_ID` after the job completes instead.
#
# Local (laptop, 8-core, 8GB RAM) profiling of the underlying train/eval steps
# gave ~42.3s/run worst case (300,000 steps @ ~7,657 steps/sec + capped eval);
# this job measures the real thing on the actual cluster hardware -- use its
# wall time and peak RSS to fill in the "SET AFTER PROFILING" #SBATCH lines in
# slurm/sampling_pct_pos_codes_sweep.sh. Do NOT submit the real array job
# using only the local estimate below.
#
# Submit:
#   sbatch slurm/sampling_pct_pos_codes_sweep_profile.sh
# ============================================================

#SBATCH --job-name=sampling_pct_pos_codes_profile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G          # generous placeholder (local worst-case estimate ~15.5GB transient peak + margin); this run's peak RSS is what you're measuring
#SBATCH --time=04:00:00    # generous placeholder for 20 runs (full percentage schedule) at 1 seed
#SBATCH --output=slurm/logs/sampling_pct_pos_codes_profile_%j.out

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source /usr/people/km3199/.local/bin/env

/usr/bin/time -v uv run python scripts/context_identification/pos_codes/sampling_pct_pos_codes_sweep.py \
    --combo-idx 2 --n-seeds 1 \
    --data-dir ./data/_profile_sampling_pct_pos_codes_sweep
