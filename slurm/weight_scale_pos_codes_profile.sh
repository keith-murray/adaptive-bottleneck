#!/bin/bash
# ============================================================
# Profiling job for the weight-init-scale sweep (see
# slurm/weight_scale_pos_codes_sweep.sh). Single task, no array.
#
# Runs the single most expensive combo of the full sweep: scale idx 0
# (g=1e-8, 128,000 epochs), one condition, one label mode, one seed. Use
# the resulting wall time and peak RSS to fill in the "SET AFTER
# PROFILING" #SBATCH lines in slurm/weight_scale_pos_codes_sweep.sh.
#
# `/usr/bin/time -v` reports elapsed wall time and Maximum resident set
# size directly in the .out file. If unavailable on your cluster, use
# `seff $SLURM_JOB_ID` after the job completes instead.
#
# PROFILED 2026-08-20 (job 5672099, spockmk2): Wall-clock 00:15:52 (952s),
# Memory Utilized 453.30 MB, 2 cores @ 75.26% CPU efficiency. See
# slurm/weight_scale_pos_codes_sweep.sh header for the derived --time/--mem
# for the real array job. Mem/time below kept generous in case this script
# is reused to re-profile a different --scale-idx/--conditions combo.
#
# Submit:
#   sbatch slurm/weight_scale_pos_codes_profile.sh
# ============================================================

#SBATCH --job-name=weight_scale_pos_codes_profile
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G          # generous placeholder; this run's peak RSS is what you're measuring
#SBATCH --time=12:00:00   # generous placeholder for a single-condition, full-epoch-count run
#SBATCH --output=slurm/logs/weight_scale_pos_codes_profile_%j.out

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source /usr/people/km3199/.local/bin/env

/usr/bin/time -v uv run python scripts/context_identification/pos_codes/weight_scale_pos_codes_sweep.py \
    --scale-idx 0 --n-seeds 1 --conditions adaptive --label-modes feature \
    --data-dir ./data/_profile_weight_scale_pos_codes_sweep
