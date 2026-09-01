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
# FIRST PROFILE (job 5990989, 2026-09-01): 3:46:35 wall clock, 24.18GB peak
# RSS -- caught a real bug: capped periodic eval was routed through a
# jdl.DataLoader(shuffle=True) and broken out of early, but jax_dataloader's
# __iter__ generates a full permutation of the ENTIRE test array (2,793,888
# rows) on every fresh iteration regardless of the early break, so the cap did
# nothing to cap the actual cost. Confirmed directly: the two train_frac=1.0
# runs in that profile (no test set, zero eval calls) finished in ~100-200s,
# vs. ~755s for every other run -- ~550-650s/run of pure waste, matching a
# direct local timing of the culprit call (~2,578ms per full-array permutation
# x 199 checkpoints = ~513s). Fixed in sampling_pct_pos_codes_sweep.py:
# periodic eval now draws a subsample via plain numpy indexing (~0.3ms even at
# this scale) instead of a DataLoader permutation. This job's fix should fix
# wall-clock time dramatically; memory is a separate question (JAX's CPU
# allocator typically doesn't return freed memory to the OS within a process,
# so the 24GB figure may reflect accumulated allocations across the profiling
# run's 20 sequential training runs rather than any single run's real peak) --
# RE-RUN THIS PROFILER after the fix and use its fresh numbers, don't assume
# memory shrinks just because time did.
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
