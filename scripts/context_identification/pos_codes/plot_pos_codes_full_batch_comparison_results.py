"""Plot and tabulate results from the full-batch positional-codes comparison.

Adapted from plot_pos_codes_comparison_results.py: full-batch GD has no held-out
test set (there is nothing left to hold out once training sees the entire
population), so all curves use 'train_acc'/'train_loss' and the x-axis is
"Training step" rather than "Epoch".

Produces:
  1. pos_codes_full_batch_comparison_accuracy.png   — train-accuracy curves (3×2 grid:
                                                        rows = conditions, cols = label modes)
  2. pos_codes_full_batch_comparison_trajectory.png — gamma trajectory (1×2 panels,
                                                        adaptive conditions only)
  3. pos_codes_full_batch_comparison_table.csv      — final train accuracy mean ± sem for
                                                        all conditions × label modes

Result structure: data[condition][label_mode][seed] -> history dict with keys
  'train_acc', 'train_loss', 'gamma'.

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
ROW_LABELS = {
    "adaptive":     "ARCA",
    "adaptive_pos": "ARCA (pos. codes)",
    "attn_pos":     "Cross attention (pos. codes)",
}
MODE_TITLES = {"feature": "Feature", "position": "Position"}
COL_TITLES = {"feature": "Feature label", "position": "Positional label"}

CONDITION_COLORS = {
    "adaptive":     plt.cm.plasma(0.2),
    "adaptive_pos": plt.cm.plasma(0.55),
    "attn_pos":     plt.cm.plasma(0.85),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_curves(seed_dict, key):
    """Stack per-seed histories into (S, T) array."""
    return np.stack(
        [np.array(hist[key]) for hist in seed_dict.values()],
        axis=0,
    )


def plot_mean_sem(ax, steps, curves, color, label=None):
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0) / np.sqrt(curves.shape[0])
    ax.plot(steps, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(steps, mean - sem, mean + sem, color=color, alpha=0.15)


def plot_individual_trajectories(ax, steps, curves, color):
    for traj in curves:
        ax.plot(steps, traj, color=color, linewidth=0.4, alpha=0.3)


# ============================================================
# Figure 1: train accuracy curves (3 rows x 2 cols)
# ============================================================

def plot_accuracy(
    results_path="./data/pos_codes_full_batch_comparison_sweep.json",
    save_path="./results/context_identification/pos_codes/pos_codes_full_batch_comparison_accuracy.png",
):
    results = load_json(results_path)

    fig, axes = plt.subplots(
        len(CONDITIONS), len(LABEL_MODES),
        figsize=(8, 7), sharey=True, sharex=True,
    )

    for row, cond in enumerate(CONDITIONS):
        for col, mode in enumerate(LABEL_MODES):
            ax = axes[row, col]
            seed_dict = results[cond][mode]
            curves = collect_curves(seed_dict, "train_acc")
            steps = np.arange(curves.shape[1])
            plot_individual_trajectories(ax, steps, curves, color=CONDITION_COLORS[cond])
            plot_mean_sem(ax, steps, curves, color=CONDITION_COLORS[cond])

            if row == 0:
                ax.set_title(COL_TITLES[mode], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{ROW_LABELS[cond]}\nTrain accuracy", fontsize=8)
            if row == len(CONDITIONS) - 1:
                ax.set_xlabel("Training step")

            ax.set_ylim(0.0, 1.05)
            ax.grid(alpha=0.2)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 2: gamma trajectory (adaptive conditions only)
# ============================================================

def plot_gamma_trajectory(
    results_path="./data/pos_codes_full_batch_comparison_sweep.json",
    save_path="./results/context_identification/pos_codes/pos_codes_full_batch_comparison_trajectory.png",
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
            steps = np.arange(curves.shape[1])
            plot_mean_sem(ax, steps, curves, color=CONDITION_COLORS[cond])

            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

            if row == 0:
                ax.set_title(CONDITION_LABELS[cond], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{MODE_TITLES[mode]}\n$\\gamma$")
            if row == len(LABEL_MODES) - 1:
                ax.set_xlabel("Training step")

            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.2)

    fig.suptitle("Positional codes comparison (full-batch GD): $\\gamma$ trajectory", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Table: final train accuracy mean ± sem
# ============================================================

def make_table(
    results_path="./data/pos_codes_full_batch_comparison_sweep.json",
    save_path="./results/context_identification/pos_codes/pos_codes_full_batch_comparison_table.csv",
    n_final=100,   # average over last n_final steps for stability
):
    """Compute mean ± sem of final train accuracy and save as CSV.

    Args:
        results_path: path to results JSON
        save_path: path to write CSV
        n_final: number of final steps to average over per seed
    """
    results = load_json(results_path)

    rows = []
    for cond in CONDITIONS:
        for mode in LABEL_MODES:
            seed_dict = results[cond][mode]
            curves = collect_curves(seed_dict, "train_acc")  # (S, T)
            # average the last n_final steps per seed, then stats across seeds
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
    print(f"\nFinal train accuracy (mean ± sem %, last {n_final} steps)\n")
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
        results_path="./data/pos_codes_full_batch_comparison_sweep.json",
        save_path="./results/context_identification/pos_codes/pos_codes_full_batch_comparison_accuracy.png",
    )

    plot_gamma_trajectory(
        results_path="./data/pos_codes_full_batch_comparison_sweep.json",
        save_path="./results/context_identification/pos_codes/pos_codes_full_batch_comparison_trajectory.png",
    )

    make_table(
        results_path="./data/pos_codes_full_batch_comparison_sweep.json",
        save_path="./results/context_identification/pos_codes/pos_codes_full_batch_comparison_table.csv",
        n_final=100,
    )
