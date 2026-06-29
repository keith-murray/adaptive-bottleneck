# Session Summary — 2026-06-29

## Goal

Add individual seed trajectories to `pos_codes_comparison_accuracy.png` to clarify whether the high variability in the positional-codes conditions reflects bimodal outcomes (some seeds learn, some fail) or uniform partial learning.

## Change

Modified `scripts/context_identification/pos_codes/plot_pos_codes_comparison_results.py`:

- Added `plot_individual_trajectories()` helper that draws per-seed curves (linewidth=0.4, alpha=0.3) behind the mean ± SEM band.
- Called it inside `plot_accuracy()` before the existing `plot_mean_sem()` call, so the mean line remains visually dominant.

## Result

The updated figure confirms the suspected bimodal pattern:

- **Adaptive + pos codes** and **Attention + pos codes**: many seeds fail to learn the task, while a minority do succeed — explaining the large SEM (~5–8%).
- **Adaptive (no pos codes)**: nearly all seeds converge to high accuracy (~98–99%), with very small SEM (~0.2–0.4%).

## Conclusion

Positional codes alter the learning dynamics or loss landscape in a way that makes optimization unreliable. The task is not fundamentally harder for those architectures — some seeds do solve it — but the basin of attraction for a good solution is apparently much smaller or harder to find when positional codes are present.
