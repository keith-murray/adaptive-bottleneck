#!/bin/bash
# ============================================================
# ContextIdentification sampling-percentage sweep -- job array
# (claude/2026_08_31/TODO.md).
#
# 3 conditions x 2 label_modes = 6 combos (idx 0-5). Each array task runs the
# FULL percentage schedule (log + linear, 20 percentages) x N_SEEDS seeds
# sequentially for ONE (condition, label_mode) combo, at a fixed 300,000-step
# budget per run (see train_one_run's docstring) -- unlike the weight-scale
# sweeps, num_steps does NOT vary by combo here, so every array task does
# roughly the same amount of work (differing only by positional_codes=True/False
# and adaptive/fixed-gamma branch, not by an order of magnitude like the old
# per-scale epoch counts did).
#
# combo-idx -> (condition, label_mode) mapping (matches
# CONDITION_LABEL_MODE_COMBOS in sampling_pct_pos_codes_sweep.py; no bash-side
# decomposition needed, --combo-idx is passed straight through):
#   0: adaptive,     feature
#   1: adaptive,     position
#   2: adaptive_pos, feature      <- worst case: pos codes (in_dim=16) + adaptive (joint gamma grad)
#   3: adaptive_pos, position     <- worst case
#   4: attn_pos,     feature      <- pos codes (in_dim=16), fixed gamma
#   5: attn_pos,     position
#
# PROFILE FIRST with slurm/sampling_pct_pos_codes_sweep_profile.sh (combo-idx 2,
# the worst case) and fill in the "SET AFTER PROFILING" lines below from its
# real wall-clock/peak-RSS. Do NOT trust the numbers below -- they are NOT a
# reliable estimate, see the history below.
#
#   FIRST profile (job 5990989, 2026-09-01, pre-fix): 3:46:35 wall clock for
#   20 runs (1 seed x full percentage schedule), 24.18GB peak RSS. This caught
#   a real bug -- capped periodic eval was still paying a full O(N) permutation
#   cost per checkpoint via jax_dataloader's shuffle=True DataLoader,
#   regardless of the early break after max_eval_batches (see
#   sampling_pct_pos_codes_sweep_profile.sh and train_one_run's docstring for
#   the full story). Fixed by drawing periodic eval subsamples via plain numpy
#   indexing instead. Directly confirmed the fix eliminates the ~550-650s/run
#   of wasted permutation work (local timing: ~2,578ms/call x 199 checkpoints
#   = ~513s old vs. ~67ms total new, at the real 2,793,888-row scale) -- but
#   this was NOT yet re-measured on the actual cluster with the fix applied.
#   Rough bound from the profile's two train_frac=1.0 runs (no test set, so
#   already unaffected by the bug): ~100-200s/run for pure training, which is
#   the best available proxy for fixed-code per-run cost until re-profiled.
#   Memory is a separate question the eval fix does NOT obviously address --
#   JAX's CPU allocator typically doesn't return freed memory to the OS within
#   a process, so 24.18GB may reflect accumulated allocations across the
#   profiling run's 20 sequential training runs rather than any single run's
#   real peak; do not assume it drops just because time did.
#
#   --time and --mem below are therefore placeholders pending a fresh profile
#   run with the fix -- generous multiples of the rough bounds above, not a
#   real measurement. RE-RUN THE PROFILER before submitting this array job.
#
#   Disk: each run writes one JSON with a full 300,000-step history (~18MB
#   uncompressed with indent=2, roughly constant size regardless of
#   percentage). 6 combos x 200 runs = 1,200 files, ~21.6GB total -- check
#   your cluster's disk quota AND inode/file-count quota (1,200 files is modest
#   but some clusters cap inode count separately from bytes) before submitting.
#
# Also create the log directory once before first submission:
#   mkdir -p slurm/logs
#
# Submit:
#   sbatch --array=0-5 slurm/sampling_pct_pos_codes_sweep.sh
# ============================================================

#SBATCH --job-name=sampling_pct_pos_codes_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2            # SET AFTER PROFILING -- tiny per-step compute (L=8, batch=64), unlikely to benefit from more cores
#SBATCH --mem=32G                    # SET AFTER PROFILING -- pre-fix profile measured 24.18GB peak (eval fix doesn't obviously change memory) + margin; RE-PROFILE, don't trust this
#SBATCH --time=18:00:00              # SET AFTER PROFILING -- rough bound from pre-fix profile's unaffected (no-test-set) runs, ~200s/run x 200 runs/combo + generous margin; RE-PROFILE, don't trust this
#SBATCH --output=slurm/logs/sampling_pct_pos_codes_sweep_%A_%a.out

N_SEEDS=10   # fixed sweep design

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source /usr/people/km3199/.local/bin/env

uv run python scripts/context_identification/pos_codes/sampling_pct_pos_codes_sweep.py \
    --combo-idx "$SLURM_ARRAY_TASK_ID" \
    --n-seeds "$N_SEEDS"
