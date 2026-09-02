# Session Summary — 2026-09-02 (continuing claude/2026_08_31/TODO.md)

## Goal

Finish what claude/2026_08_31/TODO.md started: submit the real sampling-percentage
sweep to the cluster (feasibility, memory safety, and a post-fix profiling run were
already done in the prior session -- see claude/2026_08_31/SUMMARY.md), then build
figures to answer the open question -- is ARCA without positional codes simply more
sample-efficient than the positional-code conditions, which was the hypothesis
motivating this whole sweep (README.md's original small-sample comparison showed a
large gap between conditions).

## What we did

1. **Read the post-fix profile** (job 5995808): 20 runs (1 seed, full percentage
   schedule, worst-case combo `adaptive_pos`/`feature`) took 34:24.66 wall clock,
   24,194,680 KB (~23.1GiB) peak RSS -- confirming the eval-loop fix from the prior
   session fixed wall-clock time (down from 3:46:35) without changing memory (as
   expected, since the fix addressed wasted CPU time, not the resident population
   array).

2. **Parallelized the sweep across seeds, not just conditions.** The original plan
   was a 6-task array (one per `(condition, label_mode)` combo, each running all 10
   seeds sequentially -- ~5h44m wall clock, using only 12 of the cluster's 250-core
   budget). Since the workload isn't CPU-parallel internally (129% CPU utilization
   at `--cpus-per-task=2`), the fix was to parallelize over seeds instead of cores:
   added a `--seed` CLI flag to `sampling_pct_pos_codes_sweep.py` (runs exactly one
   seed, distinct from the existing `--n-seeds` used for local smoke-testing) and
   reworked `slurm/sampling_pct_pos_codes_sweep.sh` into a 60-task array
   (`combo_idx = task_id / 10`, `seed = task_id % 10`), using 120 of 250 cores and
   cutting wall clock to ~34 minutes per task -- the same work as one profiling run,
   just run 10x in parallel instead of in series. Verified the new `--seed` flag
   directly (CLI parsing + a small end-to-end run) before submission, and updated
   `slurm/sampling_pct_pos_codes_sweep_profile.sh`'s history/comments to match.

3. **Caught a stale comment after submission.** After committing, `--mem` was
   tightened from the recommended 32G down to 24G (~4% margin over the measured
   23.1GiB peak, deliberately tight) but the comment still claimed "~1.4x margin" --
   flagged this and corrected the comment to describe the real (thin) margin rather
   than revert the value.

4. **The real sweep ran successfully** (60 array tasks, 1,200 result files, ~7GB on
   the cluster -- smaller than the ~21.6GB worst-case estimate from the prior
   session, since real per-run file sizes came in lower than the indent=2 JSON
   estimate).

5. **Built `plot_sampling_pct_pos_codes_results.py`.** Loads each of the 1,200
   per-run result JSONs one at a time, keeps only the final train/test accuracy
   (`history["train_acc"][-1]` / `history["test_acc"][-1]`, NaN when a run has no
   held-out set at `train_frac==1.0`) and caches the result as a small summary CSV
   -- so the ~7GB of full per-step histories only ever needs to be scanned once,
   and restyling a figure later doesn't require the cluster at all. Reused
   `CONDITIONS`/`LABEL_MODES`/`build_schedules`/`run_filename` directly from the
   sweep script to avoid drift. Verified against synthetic data matching the real
   schema before running for real.

6. **Iterated on figure layout per feedback**: originally 4 separate figures (one
   per schedule x metric); consolidated into 2 figures (one per schedule), each a
   2x2 grid with training accuracy on top and test accuracy on the bottom, feature
   and positional labels as columns -- dodged, jittered strip plots (one strip per
   condition per percentage bin, consistent color + marker per condition across
   every bin: `adaptive`=purple circle, `adaptive_pos`=red square, `attn_pos`=
   yellow triangle).

7. **Ran the plotting script on the cluster** (via a plot-only SLURM script) and
   committed the results: `sampling_pct_pos_codes_linear.png`,
   `sampling_pct_pos_codes_log.png`, and `sampling_pct_pos_codes_summary.csv` under
   `results/context_identification/pos_codes/`.

## Result

The figures **contradict the ordering in README.md's original positional-codes
table**, which was the whole motivation for this sweep. That table (small,
undersampled comparison) showed ARCA without positional codes clearly ahead of both
positional-code conditions, especially on the positional label:

| Condition | Feature | Position |
| --- | --- | --- |
| ARCA (no Pos) | 98.5 ± 0.4 | 99.0 ± 0.2 |
| ARCA (with Pos) | 79.9 ± 5.2 | 80.7 ± 5.7 |
| Cross attention (with Pos) | 78.0 ± 5.1 | 72.3 ± 7.9 |

In the new sweep (mean final test accuracy across all 10 seeds and all 19 tested
percentages, 0.01% to 100% of the full `L=8` population, 300,000 mini-batch steps
per run):

| Condition | Feature | Position |
| --- | --- | --- |
| ARCA (no pos) | 94.4 ± 0.4% | 99.97 ± 0.02% |
| ARCA (pos) | 96.2 ± 0.3% | 99.96 ± 0.02% |
| Cross attention (pos) | 97.6 ± 0.3% | 98.77 ± 0.51% |

On the **feature label**, ARCA without positional codes is now the *worst* of the
three conditions at nearly every percentage tested, not the best -- the opposite of
the README's ordering. On the **positional label**, the three conditions are
essentially indistinguishable near ceiling (all >98.7%), rather than the ~27-point
spread the README reported. This holds even at the smallest tested percentage
(0.01% of the population, ~1,397 training examples -- already larger than the
README experiment's "~500 or so" samples) and at 100% of the population.

One notable anomaly: `attn_pos`/position is the only condition with real low-outlier
seeds -- one seed hit 46.6% test accuracy at the smallest percentage (log schedule,
0.01%), a few more sat in the 52-73% range at the next couple of log percentages,
and seed 3 specifically also dropped to 72.9% at 80% train data (linear schedule) --
suggesting an occasional convergence failure for this condition rather than a purely
sample-size effect, since that last case has plenty of data.

## Open question / next steps

The user flagged that this result is unexpected and needs more analysis before
drawing conclusions -- this sweep alone doesn't explain *why* the ordering flipped.
Candidate directions for next session:

- Reconcile training regimes: the README's original comparison likely used a
  different setup (full-batch vs. this sweep's mini-batch training, a different
  step/epoch budget, or a different weight-init scale) than this sweep's fixed
  300,000-mini-batch-step, batch-size-64, `lr=1e-2`/`weight_decay=1e-3` config
  (`sampling_pct_pos_codes_sweep.py` defaults) -- worth checking whether matching
  the original hyperparameters at a comparable sample count reproduces the README's
  gap, which would point to training regime rather than sample size as the real
  explanatory variable.
- Investigate the `attn_pos`/position seed-3 instability at 80% train data
  specifically (not a low-sample-count case) -- is this a real occasional
  convergence failure for this condition, and if so, does it also affect
  `attn_pos`/feature or the other conditions at any percentage?
- The per-run summary CSV (`results/context_identification/pos_codes/sampling_pct_pos_codes_summary.csv`)
  already has everything needed to slice this further (per-seed, per-percentage)
  without returning to the cluster.
