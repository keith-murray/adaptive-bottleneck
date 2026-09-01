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
# real wall-clock/peak-RSS -- the numbers below are LOCAL (laptop, 8-core, 8GB
# RAM) estimates only, NOT yet verified on the cluster:
#
#   Local train/eval-step profiling: worst case (adaptive_pos, in_dim=16)
#   ~7,657 steps/sec train + capped eval -> ~42.3s/run; best case (attn_pos or
#   adaptive, in_dim=8, fixed gamma) ~12,787 steps/sec -> ~26.0s/run. With a
#   20% buffer (matching the weight-scale sweep's convention):
#     worst case: ~50.8s/run x 200 runs/combo (20 pct x 10 seeds) = ~2.82h
#     best case:  ~31.1s/run x 200 runs/combo                     = ~1.73h
#   All 6 combos fall in this ~1.7-2.8h range (see note above on why, unlike
#   the weight-scale sweep, no combo is dramatically longer than another) --
#   --time below is sized for the worst case + margin.
#
#   Memory: positional_codes=True combos (2-5) transiently peak around ~15.5GB
#   at the 90% linear-schedule percentage (numpy fancy-indexing nearly doubles
#   the resident ~7.5GB population array there -- see
#   run_sampling_pct_sweep_for_combo's docstring); combos 0-1 (no pos codes)
#   peak around ~7.8GB. --mem below is sized for the worst case + margin; a
#   single array submission shares one --mem across every task, so combos 0-1
#   will be over-provisioned -- fine on a scheduler that only queues by
#   core/mem availability (as here); reconsider if your cluster's fair-share
#   instead penalizes requested (not used) memory.
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
#SBATCH --mem=24G                    # SET AFTER PROFILING -- local worst-case estimate ~15.5GB transient peak + margin
#SBATCH --time=04:00:00              # SET AFTER PROFILING -- local worst-case estimate ~2.82h + margin
#SBATCH --output=slurm/logs/sampling_pct_pos_codes_sweep_%A_%a.out

N_SEEDS=10   # fixed sweep design

cd "$HOME/adaptive-bottleneck"   # FILL IN: adjust if the repo lives elsewhere on the cluster

source /usr/people/km3199/.local/bin/env

uv run python scripts/context_identification/pos_codes/sampling_pct_pos_codes_sweep.py \
    --combo-idx "$SLURM_ARRAY_TASK_ID" \
    --n-seeds "$N_SEEDS"
