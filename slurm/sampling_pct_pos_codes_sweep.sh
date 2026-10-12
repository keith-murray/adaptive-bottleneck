#!/bin/bash
# ============================================================
# ContextIdentification sampling-percentage sweep -- job array
# (claude/2026_08_31/TODO.md, claude/2026_09_02/TODO.md).
#
# 3 conditions x 2 label_modes x 10 seeds = 60 (combo, seed) pairs (idx 0-59).
# Each array task runs the FULL percentage schedule (log + linear, 20
# percentages) for ONE (condition, label_mode, seed), at a fixed 300,000-step
# budget per run (see train_one_run's docstring).
#
# Parallelizing over seeds too (not just the 6 combos) is deliberate: the
# profiled workload is not CPU-parallel internally (129% CPU with
# --cpus-per-task=2, since JAX's CPU backend runs one train step at a time),
# so running all 10 seeds sequentially inside one task would leave most of
# the cluster's 250-core budget idle while wall clock stretched to
# ~5h44m/combo (10 x the profiled 20-run time). Splitting into 60 tasks x 2
# cores = 120 cores (< 250 budget) instead brings each task down to
# ~34min -- the SAME work as the profiling run below, just run 10x in
# parallel instead of in series. The only added cost is repeating the
# per-task population-generation step (generate_all_examples(), ~100s
# overhead per task -- see the profile history below) 60 times instead of 6,
# which is trivial next to the time saved.
#
# task-id -> (combo_idx, seed) decomposition (matches
# CONDITION_LABEL_MODE_COMBOS in sampling_pct_pos_codes_sweep.py):
#   combo_idx = task_id / 10   (integer division)
#   seed      = task_id % 10
#   combo_idx 0: adaptive,     feature
#   combo_idx 1: adaptive,     position
#   combo_idx 2: adaptive_pos, feature      <- worst case: pos codes (in_dim=16) + adaptive (joint gamma grad)
#   combo_idx 3: adaptive_pos, position     <- worst case
#   combo_idx 4: attn_pos,     feature      <- pos codes (in_dim=16), fixed gamma
#   combo_idx 5: attn_pos,     position
#
# PROFILED with slurm/sampling_pct_pos_codes_sweep_profile.sh (combo-idx 2,
# the worst case, 1 seed x full percentage schedule = 20 runs -- exactly the
# per-array-task workload here).
#
#   FIRST profile (job 5990989, 2026-09-01, pre-fix): 3:46:35 wall clock,
#   24.18GB peak RSS. Caught a real bug -- capped periodic eval was still
#   paying a full O(N) permutation cost per checkpoint via jax_dataloader's
#   shuffle=True DataLoader, regardless of the early break after
#   max_eval_batches (see sampling_pct_pos_codes_sweep_profile.sh and
#   train_one_run's docstring for the full story). Fixed by drawing periodic
#   eval subsamples via plain numpy indexing instead.
#
#   SECOND profile (job 5995808, 2026-09-01, post-fix): 34:24.66 wall clock
#   (down from 3:46:35, confirming the fix -- ~103.2s/run average across all
#   20 percentages), 24,194,680 KB (~23.1GiB) peak RSS -- essentially
#   unchanged from the pre-fix run, as expected: the eval fix addressed
#   wasted CPU time, not memory (the population array + per-run indexing
#   copies drive peak RSS, both independent of the eval bug). This is now a
#   real measurement of the per-array-task workload (1 seed x full schedule,
#   worst-case combo), not an estimate.
#
#   --time and --mem below are generous margins over this real measurement:
#   --time 1h vs. the measured 34:24.66 (~1.75x, covers cluster-to-cluster
#   variance); --mem 32G vs. the measured ~23.1GiB (~1.4x margin).
#
#   Disk: each run writes one JSON with a full 300,000-step history (~18MB
#   uncompressed with indent=2, roughly constant size regardless of
#   percentage). 60 tasks x 20 runs = 1,200 files total, ~21.6GB -- check
#   your cluster's disk quota AND inode/file-count quota before submitting.
#
# Also create the log directory once before first submission:
#   mkdir -p slurm/logs
#
# Submit:
#   sbatch --array=0-59 slurm/sampling_pct_pos_codes_sweep.sh
# ============================================================

#SBATCH --job-name=sampling_pct_pos_codes_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # profile measured 129% CPU utilization -- not CPU-parallel, 2 cores is already generous
#SBATCH --mem=24G                    # profile (job 5995808, post-fix) measured 24,194,680 KB (~23.1GiB) peak -- ~4% margin (kept tight deliberately; bump if array tasks start OOM-killing)
#SBATCH --time=01:00:00              # profile (job 5995808, post-fix) measured 34:24.66 wall clock -- ~1.75x margin
#SBATCH --output=slurm/logs/sampling_pct_pos_codes_sweep_%A_%a.out

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source "$HOME/.local/bin/env"   # puts uv on PATH (default uv install location); adjust for your cluster

combo_idx=$(( SLURM_ARRAY_TASK_ID / 10 ))
seed=$(( SLURM_ARRAY_TASK_ID % 10 ))

uv run python scripts/context_identification/pos_codes/sampling_pct_pos_codes_sweep.py \
    --combo-idx "$combo_idx" \
    --seed "$seed"
