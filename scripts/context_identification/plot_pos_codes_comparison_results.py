"""Plot and tabulate results from the positional-codes comparison sweep.

Produces:
  1. pos_codes_comparison_accuracy.png  — test-accuracy curves (2×3 grid:
                                          rows = label modes, cols = conditions)
  2. pos_codes_comparison_trajectory.png — gamma trajectory (1×2 panels,
                                            adaptive conditions only)
  3. pos_codes_comparison_table.csv     — final test accuracy mean ± sem for
                                          all conditions × label modes (paper table)

Result structure: data[condition][label_mode][seed] -> history dict with keys
  'test_acc', 'gamma'.

Conditions:
  adaptive     — online gamma (init=0.5), no positional codes
  adaptive_pos — online gamma (init=0.5), with positional codes
  attn_pos     — fixed gamma=0 (self-attention), with positional codes
"""

import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_bottleneck.utils import load_plot_config


CONDITIONS = ("adaptive", "adaptive_pos", "attn_pos")
LABEL_MODES = ("feature", "position")

CONDITION_LABELS = {
    "adaptive":     "Adaptive (no pos codes)",
    "adaptive_pos": "Adaptive + pos codes",
    "attn_pos":     "Attention + pos codes",
}
MODE_TITLES = {"feature": "Feature", "position": "Position"}

CONDITION_COLORS = {
    "adaptive":     plt.cm.plasma(0.2),
    "adaptive_pos": plt.cm.plasma(0.55),
    "attn_pos":     plt.cm.plasma(0.85),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_curves(seed_dict, key):
    """Stack per-seed histories into (S, E) array."""
    return np.stack(
        [np.array(hist[key]) for hist in seed_dict.values()],
        axis=0,
    )


def plot_mean_sem(ax, epochs, curves, color, label=None):
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0) / np.sqrt(curves.shape[0])
    ax.plot(epochs, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.15)


# ============================================================
# Figure 1: test accuracy curves (2 rows × 3 cols)
# ============================================================

def plot_accuracy(
    results_path="./data/pos_codes_comparison_sweep.json",
    save_path="./results/pos_codes_comparison_accuracy.png",
):
    results = load_json(results_path)

    fig, axes = plt.subplots(
        len(LABEL_MODES), len(CONDITIONS),
        figsize=(12, 5), sharey=True, sharex=True,
    )

    for row, mode in enumerate(LABEL_MODES):
        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            seed_dict = results[cond][mode]
            curves = collect_curves(seed_dict, "test_acc")
            epochs = np.arange(curves.shape[1])
            plot_mean_sem(ax, epochs, curves, color=CONDITION_COLORS[cond])

            if row == 0:
                ax.set_title(CONDITION_LABELS[cond], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{MODE_TITLES[mode]}\nTest accuracy")
            if row == len(LABEL_MODES) - 1:
                ax.set_xlabel("Epoch")

            ax.set_ylim(0.0, 1.05)
            ax.grid(alpha=0.2)

    fig.suptitle("Positional codes comparison: test accuracy", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 2: gamma trajectory (adaptive conditions only)
# ============================================================

def plot_gamma_trajectory(
    results_path="./data/pos_codes_comparison_sweep.json",
    save_path="./results/pos_codes_comparison_trajectory.png",
):
    results = load_json(results_path)
    adaptive_conds = ("adaptive", "adaptive_pos")

    fig, axes = plt.subplots(
        len(LABEL_MODES), len(adaptive_conds),
        figsize=(8, 5), sharey=True, sharex=True,
    )

    for row, mode in enumerate(LABEL_MODES):
        for col, cond in enumerate(adaptive_conds):
            ax = axes[row, col]
            seed_dict = results[cond][mode]
            curves = collect_curves(seed_dict, "gamma")
            epochs = np.arange(curves.shape[1])
            plot_mean_sem(ax, epochs, curves, color=CONDITION_COLORS[cond])

            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

            if row == 0:
                ax.set_title(CONDITION_LABELS[cond], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{MODE_TITLES[mode]}\n$\\gamma$")
            if row == len(LABEL_MODES) - 1:
                ax.set_xlabel("Epoch")

            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)

    fig.suptitle("Positional codes comparison: $\\gamma$ trajectory", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Table: final test accuracy mean ± sem
# ============================================================

def make_table(
    results_path="./data/pos_codes_comparison_sweep.json",
    save_path="./results/pos_codes_comparison_table.csv",
    n_final=100,   # average over last n_final epochs for stability
):
    """Compute mean ± sem of final test accuracy and save as CSV.

    Args:
        results_path: path to results JSON
        save_path: path to write CSV
        n_final: number of final epochs to average over per seed
    """
    results = load_json(results_path)

    rows = []
    for cond in CONDITIONS:
        for mode in LABEL_MODES:
            seed_dict = results[cond][mode]
            curves = collect_curves(seed_dict, "test_acc")  # (S, E)
            # average the last n_final epochs per seed, then stats across seeds
            final_acc = curves[:, -n_final:].mean(axis=1)   # (S,)
            mean = final_acc.mean()
            sem  = final_acc.std(ddof=1) / np.sqrt(len(final_acc))
            rows.append({
                "condition":  cond,
                "label_mode": mode,
                "mean":       f"{mean:.4f}",
                "sem":        f"{sem:.4f}",
                "mean_pct":   f"{100*mean:.1f}",
                "sem_pct":    f"{100*sem:.1f}",
                "formatted":  f"{100*mean:.1f} ± {100*sem:.1f}",
            })

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["condition", "label_mode", "mean", "sem", "mean_pct", "sem_pct", "formatted"]
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Also print the paper-ready table to stdout
    print(f"\nFinal test accuracy (mean ± sem %, last {n_final} epochs)\n")
    print(f"{'Condition':<25} {'Feature':>15} {'Position':>15}")
    print("-" * 57)
    for cond in CONDITIONS:
        feature_row  = next(r for r in rows if r["condition"] == cond and r["label_mode"] == "feature")
        position_row = next(r for r in rows if r["condition"] == cond and r["label_mode"] == "position")
        print(f"{CONDITION_LABELS[cond]:<25} {feature_row['formatted']:>15} {position_row['formatted']:>15}")

    print(f"\nSaved {save_path}")
    return rows


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    load_plot_config()

    plot_accuracy(
        results_path="./data/pos_codes_comparison_sweep.json",
        save_path="./results/pos_codes_comparison_accuracy.png",
    )

    plot_gamma_trajectory(
        results_path="./data/pos_codes_comparison_sweep.json",
        save_path="./results/pos_codes_comparison_trajectory.png",
    )

    make_table(
        results_path="./data/pos_codes_comparison_sweep.json",
        save_path="./results/pos_codes_comparison_table.csv",
        n_final=100,
    )
