"""Plot the full-batch weight-init-scale sweep as a 6-panel jitter figure.

Adapted from plot_weight_scale_pos_codes_results.py: full-batch GD has no
held-out test set, so the y-axis is final *train* accuracy (the full
population), not test accuracy.

Loads data/weight_scale_pos_codes_full_batch_sweep/scale_<tag>__<condition>__<label_mode>.json
for every (scale, condition, label_mode) combo -- 7 x 3 x 2 = 42 files total
(must all exist -- run weight_scale_pos_codes_full_batch_sweep.py for every
--scale-idx/--conditions/--label-modes combo first, e.g. via the SLURM array
in slurm/weight_scale_pos_codes_full_batch_sweep.sh).

Produces:
  1. weight_scale_pos_codes_full_batch_accuracy.png  -- 6-panel jitter figure
  2. weight_scale_pos_codes_full_batch_table.csv     -- final train accuracy
                                                         mean ± sem for all
                                                         condition x label_mode
                                                         x scale combinations
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_bottleneck.utils import load_plot_config
from weight_scale_pos_codes_full_batch_sweep import SCALES, scale_tag


CONDITIONS = ("adaptive", "adaptive_pos", "attn_pos")
LABEL_MODES = ("feature", "position")

ROW_LABELS = {
    "adaptive":     "ARCA",
    "adaptive_pos": "ARCA (pos. codes)",
    "attn_pos":     "Cross attention (pos. codes)",
}
COL_TITLES = {"feature": "Feature label", "position": "Positional label"}

CONDITION_COLORS = {
    "adaptive":     plt.cm.plasma(0.2),
    "adaptive_pos": plt.cm.plasma(0.55),
    "attn_pos":     plt.cm.plasma(0.85),
}


def load_all_scales(data_dir):
    """scale -> results[condition][label_mode][seed] -> summary dict.

    Loads one file per (scale, condition, label_mode) combo -- see
    weight_scale_pos_codes_full_batch_sweep.py's combo_filename -- and merges
    them back into the same nested structure a single-file save would have had.
    """
    data_dir = Path(data_dir)
    all_results = {s: {cond: {} for cond in CONDITIONS} for s in SCALES}
    for s in SCALES:
        tag = scale_tag(s)
        for cond in CONDITIONS:
            for mode in LABEL_MODES:
                path = data_dir / f"scale_{tag}__{cond}__{mode}.json"
                d = json.load(open(path))
                all_results[s][cond][mode] = d[cond][mode]
    return all_results


def collect_final_accs(all_results, condition, label_mode):
    """scale -> np.array of per-seed final_train_acc."""
    return {
        s: np.array([v["final_train_acc"]
                      for v in all_results[s][condition][label_mode].values()])
        for s in SCALES
    }


# ============================================================
# Figure: weight-scale jitter plot (3 rows x 2 cols)
# ============================================================

def plot_weight_scale_jitter(
    data_dir="./data/weight_scale_pos_codes_full_batch_sweep",
    save_path="./results/context_identification/pos_codes/weight_scale_pos_codes_full_batch_accuracy.png",
    jitter_width=0.06,
    rng_seed=0,
):
    all_results = load_all_scales(data_dir)
    rng = np.random.default_rng(rng_seed)
    log_scales = np.log10(SCALES)

    fig, axes = plt.subplots(
        len(CONDITIONS), len(LABEL_MODES),
        figsize=(8, 9), sharey=True, sharex=True,
    )

    for row, cond in enumerate(CONDITIONS):
        for col, mode in enumerate(LABEL_MODES):
            ax = axes[row, col]
            color = CONDITION_COLORS[cond]
            per_scale = collect_final_accs(all_results, cond, mode)

            means = []
            for x0, s in zip(log_scales, SCALES):
                accs = per_scale[s]
                x = x0 + rng.uniform(-jitter_width, jitter_width, size=accs.shape)
                ax.scatter(x, accs, s=10, color=color, alpha=0.4, linewidths=0)
                means.append(accs.mean())

            ax.plot(log_scales, means, color=color, marker="D", markersize=5,
                     linewidth=1.5, zorder=5)

            ax.set_xticks(log_scales)
            ax.set_xticklabels([f"$10^{{{int(e)}}}$" for e in log_scales], rotation=45)

            if row == 0:
                ax.set_title(COL_TITLES[mode], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{ROW_LABELS[cond]}\nFinal train accuracy", fontsize=8)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("Weight-init scale $g$")

            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Table: final train accuracy mean ± sem per scale
# ============================================================

def make_table(
    data_dir="./data/weight_scale_pos_codes_full_batch_sweep",
    save_path="./results/context_identification/pos_codes/weight_scale_pos_codes_full_batch_table.csv",
):
    all_results = load_all_scales(data_dir)

    rows = []
    for cond in CONDITIONS:
        for mode in LABEL_MODES:
            per_scale = collect_final_accs(all_results, cond, mode)
            for s in SCALES:
                accs = per_scale[s]
                mean = accs.mean()
                sem = accs.std(ddof=1) / np.sqrt(len(accs))
                rows.append({
                    "condition":  cond,
                    "label_mode": mode,
                    "scale":      f"{s:.0e}",
                    "mean":       f"{mean:.4f}",
                    "sem":        f"{sem:.4f}",
                    "mean_pct":   f"{100*mean:.1f}",
                    "sem_pct":    f"{100*sem:.1f}",
                    "formatted":  f"{100*mean:.1f} ± {100*sem:.1f}",
                })

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "label_mode", "scale", "mean", "sem", "mean_pct", "sem_pct", "formatted"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {save_path}")
    return rows


# ============================================================
# Main
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./data/weight_scale_pos_codes_full_batch_sweep")
    p.add_argument("--save-dir", type=str, default="./results/context_identification/pos_codes")
    return p


if __name__ == "__main__":
    load_plot_config()
    args = build_arg_parser().parse_args()

    plot_weight_scale_jitter(
        data_dir=args.data_dir,
        save_path=f"{args.save_dir}/weight_scale_pos_codes_full_batch_accuracy.png",
    )

    make_table(
        data_dir=args.data_dir,
        save_path=f"{args.save_dir}/weight_scale_pos_codes_full_batch_table.csv",
    )
