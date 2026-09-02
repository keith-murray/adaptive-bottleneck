"""Plot the sampling-percentage sweep as dodged strip plots
(claude/2026_09_02/TODO.md, Task 2).

Loads every {schedule}__pct{pct_idx:02d}__{condition}__{label_mode}__seed{seed}.json
file written by sampling_pct_pos_codes_sweep.py (60 array tasks x 20 runs =
1,200 files -- see run_filename there), extracts only each run's FINAL
training/test accuracy (history["train_acc"][-1] / history["test_acc"][-1],
or NaN if a run has no test set at train_frac==1.0), and caches the result as
a small summary CSV -- avoids re-parsing the full ~7GB of per-step histories
every time a figure is restyled.

Meant to run ON THE CLUSTER where the sweep's data/ lives (the raw data is
too large to pull locally): `uv run python plot_sampling_pct_pos_codes_results.py`
writes small PNGs (and the summary CSV) to --save-dir, which alone are cheap
enough to copy back.

Produces two figures, one per sweep schedule, each a 2x2 grid of dodged strip
plots (rows = final train/test accuracy, top/bottom; columns = label_mode):
x-axis = percentage of data used to train (one categorical bin per sweep
percentage), one dodge-offset+jittered strip per condition per bin (same
color+marker for a condition across every bin):
  1. sampling_pct_pos_codes_linear.png
  2. sampling_pct_pos_codes_log.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from adaptive_bottleneck.utils import load_plot_config
from sampling_pct_pos_codes_sweep import CONDITIONS, LABEL_MODES, build_schedules, run_filename


CONDITION_LABELS = {
    "adaptive":     "ARCA (no pos codes)",
    "adaptive_pos": "ARCA (pos codes)",
    "attn_pos":     "Cross attention (pos codes)",
}
COL_TITLES = {"feature": "Feature label", "position": "Positional label"}
METRIC_LABELS = {
    "train_acc_final": "Final training accuracy",
    "test_acc_final":  "Final test accuracy",
}
SCHEDULE_TITLES = {"linear": "Linear schedule", "log": "Log schedule"}

CONDITION_COLORS = {
    "adaptive":     plt.cm.plasma(0.2),
    "adaptive_pos": plt.cm.plasma(0.55),
    "attn_pos":     plt.cm.plasma(0.85),
}
# Secondary (non-color) encoding for condition identity, on top of
# CONDITION_COLORS -- kept distinguishable under color-vision deficiency and
# in grayscale printouts.
CONDITION_MARKERS = {"adaptive": "o", "adaptive_pos": "s", "attn_pos": "^"}
# Fixed dodge offsets (same order as CONDITIONS) so each condition's strip
# lands at a consistent horizontal position relative to its percentage bin,
# across every figure.
DODGE_OFFSETS = {"adaptive": -0.24, "adaptive_pos": 0.0, "attn_pos": 0.24}


def format_pct(train_frac):
    """0.0001 -> '0.01%', 0.359 -> '35.9%', 1.0 -> '100%'."""
    pct = 100 * train_frac
    if pct >= 1:
        return f"{pct:.0f}%" if pct == round(pct) else f"{pct:.1f}%"
    return f"{pct:.3g}%"


# ============================================================
# Data loading (with a small on-disk cache -- see module docstring)
# ============================================================

def collect_summary(data_dir, schedules, seeds, cache_path=None, refresh_cache=False):
    """One row per (schedule, pct_idx, condition, label_mode, seed) with that
    run's final train/test accuracy. Reads and discards each ~MB-scale JSON
    file one at a time (never holds more than one run's full step history in
    memory) and keeps only the two scalars needed for these figures.

    Returns a DataFrame; also writes/reads a CSV cache at cache_path (unless
    refresh_cache) so restyling a figure doesn't require re-scanning data_dir.
    """
    if cache_path is not None and Path(cache_path).exists() and not refresh_cache:
        print(f"Loading cached summary from {cache_path} (pass --refresh-cache to rescan data_dir)")
        return pd.read_csv(cache_path)

    data_dir = Path(data_dir)
    combos = [
        (schedule_name, pct_idx, train_frac, condition, label_mode, seed)
        for schedule_name, pcts in schedules.items()
        for pct_idx, train_frac in enumerate(pcts)
        for condition in CONDITIONS
        for label_mode in LABEL_MODES
        for seed in seeds
    ]

    rows, missing = [], []
    for schedule_name, pct_idx, train_frac, condition, label_mode, seed in tqdm(combos, desc="Loading result files"):
        path = data_dir / run_filename(schedule_name, pct_idx, condition, label_mode, seed)
        if not path.exists():
            missing.append(path.name)
            continue
        with path.open() as f:
            hist = json.load(f)["history"]
        rows.append({
            "schedule": schedule_name,
            "pct_idx": pct_idx,
            "train_frac": train_frac,
            "condition": condition,
            "label_mode": label_mode,
            "seed": seed,
            "train_acc_final": hist["train_acc"][-1] if hist["train_acc"] else float("nan"),
            "test_acc_final": hist["test_acc"][-1] if hist["test_acc"] else float("nan"),
        })

    if missing:
        print(f"WARNING: {len(missing)}/{len(combos)} expected result files were missing "
              f"under {data_dir}, e.g.: {missing[:5]}")

    df = pd.DataFrame(rows)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"Saved summary cache to {cache_path}")
    return df


# ============================================================
# Figure: dodged strip plot (2 metric rows x 2 label-mode cols)
# ============================================================

def _draw_strip_panel(ax, sub_mode, metric, n_bins, jitter_width, rng):
    """Draw one (metric, label_mode) panel: one dodge-offset+jittered strip
    per condition per percentage bin, plus a mean-trend line. NaN values (no
    test set at train_frac==1.0) are silently dropped -- that bin's strip for
    that condition is simply empty, which is the correct depiction of the
    sweep design rather than an error.
    """
    for cond in CONDITIONS:
        color = CONDITION_COLORS[cond]
        marker = CONDITION_MARKERS[cond]
        offset = DODGE_OFFSETS[cond]
        sub_cond = sub_mode[sub_mode["condition"] == cond]

        xs_mean, means = [], []
        for pct_idx in range(n_bins):
            vals = sub_cond.loc[sub_cond["pct_idx"] == pct_idx, metric].dropna().to_numpy()
            if vals.size == 0:
                continue
            x0 = pct_idx + offset
            x = x0 + rng.uniform(-jitter_width, jitter_width, size=vals.shape)
            ax.scatter(x, vals, s=14, color=color, marker=marker, alpha=0.5, linewidths=0)
            xs_mean.append(x0)
            means.append(vals.mean())

        if means:
            ax.plot(xs_mean, means, color=color, marker=marker, markersize=5,
                     linewidth=1.3, alpha=0.9, zorder=5, label=CONDITION_LABELS[cond])

    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)


def plot_schedule_figure(df, schedule_name, pct_labels, save_path, jitter_width=0.08, rng_seed=0):
    """One figure per schedule: rows = (train, test) final accuracy, columns
    = label_mode.
    """
    sub_sched = df[df["schedule"] == schedule_name]
    rng = np.random.default_rng(rng_seed)
    n_bins = len(pct_labels)
    metrics = ("train_acc_final", "test_acc_final")

    fig, axes = plt.subplots(
        len(metrics), len(LABEL_MODES), figsize=(12, 8.5), sharex=True, sharey=True,
    )

    for row, metric in enumerate(metrics):
        for col, mode in enumerate(LABEL_MODES):
            ax = axes[row, col]
            sub_mode = sub_sched[sub_sched["label_mode"] == mode]
            _draw_strip_panel(ax, sub_mode, metric, n_bins, jitter_width, rng)

            if row == 0:
                ax.set_title(COL_TITLES[mode], fontsize=10)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            if row == len(metrics) - 1:
                ax.set_xticks(range(n_bins))
                ax.set_xticklabels(pct_labels, rotation=45, ha="right")
                ax.set_xlabel("% of data used for training")

    axes[0, 0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle(f"{SCHEDULE_TITLES[schedule_name]}", fontsize=12)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=str, default="./data/sampling_pct_pos_codes_sweep")
    p.add_argument("--save-dir", type=str, default="./results/context_identification/pos_codes")
    p.add_argument("--n-seeds", type=int, default=10, help="Seeds 0..n_seeds-1 to look for per combo.")
    p.add_argument("--cache-path", type=str, default=None,
                    help="Summary CSV cache path. Default: <save-dir>/sampling_pct_pos_codes_summary.csv")
    p.add_argument("--refresh-cache", action="store_true",
                    help="Rescan --data-dir instead of using an existing cache.")
    return p


if __name__ == "__main__":
    load_plot_config()
    args = build_arg_parser().parse_args()

    save_dir = Path(args.save_dir)
    cache_path = args.cache_path or (save_dir / "sampling_pct_pos_codes_summary.csv")

    schedules = build_schedules()
    df = collect_summary(
        data_dir=args.data_dir,
        schedules=schedules,
        seeds=range(args.n_seeds),
        cache_path=cache_path,
        refresh_cache=args.refresh_cache,
    )

    for schedule_name in ("linear", "log"):
        pct_labels = [format_pct(p) for p in schedules[schedule_name]]
        plot_schedule_figure(
            df, schedule_name, pct_labels,
            save_path=save_dir / f"sampling_pct_pos_codes_{schedule_name}.png",
        )
