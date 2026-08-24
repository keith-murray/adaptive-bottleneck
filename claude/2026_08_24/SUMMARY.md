# Session Summary — 2026-08-24

## Goal

Following from 2026-08-20's finding that ARCA + positional codes and cross-attention
+ positional codes show highly variable (bimodal) convergence under SGD, redo the
weight-init-scale analysis with **full-batch gradient descent** to test whether
mini-batch noise ("competing gradient information" between mini-batches) — rather
than the positional-code architecture or weight-init scale itself — was the cause
of the instability.

## What we did

1. **Combinatorics** (`scripts/context_identification/pos_codes/full_batch_combinatorics.py`):
   derived and verified (against brute-force enumeration) a closed-form formula for
   `N(L)`, the exact number of unique valid `ContextIdentification` trials for a
   given `L`. `N(L)` grows like `L^(L+1)` (`N(10) ≈ 1.07×10^10`, intractable), so we
   chose `L=6` (`N(6)=27,684`) as tractable for exhaustive enumeration.
2. Added `ContextIdentification.generate_all_examples()` to
   `src/adaptive_bottleneck/tasks.py` — exhaustively enumerates the full valid
   population (vs. the existing constructive rejection sampling), producing exactly
   `N(L)` rows.
3. Implemented full-batch GD training
   (`scripts/context_identification/pos_codes/pos_codes_full_batch_comparison.py`):
   no train/test split (there's nothing left to hold out once training sees the
   entire population), x-axis is training step, not epoch. Verified all three
   conditions (`adaptive`, `adaptive_pos`, `attn_pos`) converge near 100% train
   accuracy at `L=6`.
4. Implemented the weight-scale sweep under full-batch GD
   (`weight_scale_pos_codes_full_batch_sweep.py`), mirroring
   `weight_scale_pos_codes_sweep.py`'s structure (`STEPS_FOR_SCALE` in place of
   `EPOCHS_FOR_SCALE`).
5. Parallelized the sweep across the 20 seeds per `(scale, condition, label_mode)`
   combo for the HPC. First via `jax.vmap` (batch all seeds into one computation) —
   a local test showed this gave **no speedup** over the old sequential-per-seed
   loop (tiny per-step matmuls at `L=6`, plausibly dispatch- rather than
   compute-bound). Then via `jax.shard_map`, explicitly partitioning seeds across
   CPU devices instead of relying on XLA's intra-op threading — this gave a real,
   reproducible **~1.6x local speedup** (64.1s → ~37-40s at 4+ devices on an 8-core
   machine). Both backends are available via `--backend {vmap,shard_map}`; the
   sweep script and SLURM scripts default to `shard_map`.
6. Ran the full sweep on the HPC: 20 seeds × 7 scales (`g ∈ {1e-8 .. 1e-2}`) × 3
   conditions × 2 label modes.

## Result

Full-batch GD converges near 100% (mean accuracy mostly ≥98%, worst case 94.5%,
SEM mostly <2%) across **every** weight-init scale and **all three** conditions —
including `adaptive_pos` and `attn_pos`, the conditions that were highly unreliable
under SGD.

Comparison against the SGD baseline (`README.md`, `g≈0.02`, feature / position):

| Condition | SGD baseline | Full-batch GD (range across all scales) |
| --- | --- | --- |
| ARCA (no pos codes) | 98.5±0.4 / 99.0±0.2 | 98.4–100% / 94.5–100% |
| ARCA (pos codes) | 79.9±5.2 / 80.7±5.7 | 94.6–99.9% / ~100% |
| Cross attention (pos codes) | 78.0±5.1 / 72.3±7.9 | 97.2–100% / 98.3–100% |

Full results: `results/context_identification/pos_codes/weight_scale_pos_codes_full_batch_accuracy.png`,
`weight_scale_pos_codes_full_batch_table.csv`.

## Conclusion

The bimodal, high-variance failure mode for the positional-code conditions under
SGD is essentially eliminated under full-batch GD, regardless of weight-init scale.
This is strong evidence that **mini-batch noise**, not the positional-code
architecture or weight-init scale per se, was the primary cause of the original
convergence instability.

A secondary, much weaker pattern: full-batch accuracy dips slightly at the
*larger* scales (1e-2, 1e-3) for feature labels — the opposite direction from the
2026-08-20 SGD scale-sweep hypothesis (which found *smaller* scales helped SGD
convergence).

## Open question / next steps

New hypothesis under consideration: the `README.md` SGD results were trained on a
comparatively small i.i.d.-sampled training set (`num_samples` in the hundreds to
low thousands) relative to the full training distribution's size (`N(6)=27,684`
for the small `L` used here; the README's `L=10` setting would be far larger
still, `N(10)≈1.07×10^10`). Under this view, ARCA without positional codes may
simply be **more sample-efficient** — able to learn from a small i.i.d. sample
where the positional-code conditions need to see much more (or all) of the
distribution to reliably converge. *Why* ARCA-without-pos-codes would be more
sample-efficient is an open question, but the first step is to understand the
effect of training-set size/coverage directly: sweep `num_samples` (or the
fraction of the full population used) and see where each condition's accuracy
saturates. Deferred to a future session.
