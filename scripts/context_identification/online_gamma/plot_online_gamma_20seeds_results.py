"""Plot results from the online gamma 20-seeds sweep.

Produces two figures saved to results/:
  1. online_gamma_20seeds_accuracy.png   — test-accuracy curves (1×2 panels)
  2. online_gamma_20seeds_trajectory.png — per-epoch gamma trajectory (1×2 panels)

Result structure: data[label_mode][seed] -> history dict with keys
  'test_acc', 'gamma'.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_bottleneck.utils import load_plot_config


LABEL_MODES = ("feature", "position")
MODE_TITLES = {"feature": "Feature", "position": "Position"}


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
# Figure 1: test accuracy curves
# ============================================================

def plot_accuracy(
    results_path="./data/online_gamma_20seeds_sweep.json",
    save_path="./results/online_gamma_20seeds_accuracy.png",
):
    results = load_json(results_path)
    color = plt.cm.plasma(0.6)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    for ax, mode in zip(axes, LABEL_MODES):
        seed_dict = results[mode]
        curves = collect_curves(seed_dict, "test_acc")
        epochs = np.arange(curves.shape[1])
        plot_mean_sem(ax, epochs, curves, color=color)

        ax.set_title(MODE_TITLES[mode])
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Online $\\gamma$ (init=0.5, 20 seeds): test accuracy", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 2: gamma trajectory
# ============================================================

def plot_gamma_trajectory(
    results_path="./data/online_gamma_20seeds_sweep.json",
    save_path="./results/online_gamma_20seeds_trajectory.png",
):
    results = load_json(results_path)
    color = plt.cm.plasma(0.6)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

    for ax, mode in zip(axes, LABEL_MODES):
        seed_dict = results[mode]
        curves = collect_curves(seed_dict, "gamma")
        epochs = np.arange(curves.shape[1])

        plot_mean_sem(ax, epochs, curves, color=color)

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax.set_title(MODE_TITLES[mode])
        ax.set_xlabel("Epoch")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(r"$\gamma$")
    fig.suptitle("Online $\\gamma$ (init=0.5, 20 seeds): $\\gamma$ trajectory", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 3: dual plot
# ============================================================

def plot_accuracy_gamma_combined(
    results_path="./data/online_gamma_20seeds_sweep.json",
    save_path="./results/online_gamma_20seeds_combined.png",
    epoch_start=0,
    epoch_end=None,   # None = plot all epochs
    shade="sem",
):
    results = load_json(results_path)

    acc_color  = plt.cm.plasma(0.3)
    gam_color  = plt.cm.plasma(0.75)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)

    COL_TITLES = {"feature": "Feature label", "position": "Positional label"}

    for col, (ax_acc, mode) in enumerate(zip(axes, LABEL_MODES)):

        # ── collect curves ──────────────────────────────────────
        acc_curves = collect_curves(results[mode], "test_acc")   # (S, E)
        gam_curves = collect_curves(results[mode], "gamma")      # (S, E)

        n_epochs = acc_curves.shape[1]
        end      = epoch_end if epoch_end is not None else n_epochs
        sl       = slice(epoch_start, end)
        epochs   = np.arange(epoch_start, end)

        acc_curves = acc_curves[:, sl]
        gam_curves = gam_curves[:, sl]

        # ── stats ────────────────────────────────────────────────
        def mean_err(curves):
            mean = curves.mean(axis=0)
            std  = curves.std(axis=0, ddof=1) if curves.shape[0] > 1 else np.zeros_like(mean)
            if shade.lower() == "sem":
                err = std / np.sqrt(curves.shape[0])
            elif shade.lower() == "std":
                err = std
            else:
                raise ValueError("shade must be 'sem' or 'std'")
            return mean, err

        acc_mean, acc_err = mean_err(acc_curves)
        gam_mean, gam_err = mean_err(gam_curves)

        # ── left axis: test accuracy ─────────────────────────────
        ax_acc.plot(epochs, acc_mean, color=acc_color, linewidth=1.5, label="Test acc")
        ax_acc.fill_between(epochs, acc_mean - acc_err, acc_mean + acc_err,
                            color=acc_color, alpha=0.20)
        ax_acc.set_ylim(0.0, 1.05)
        if col == 0:
            ax_acc.set_ylabel("Test accuracy", fontsize=8, color=acc_color)
        ax_acc.tick_params(axis="y", labelcolor=acc_color, labelsize=8)
        ax_acc.set_title(COL_TITLES[mode], fontsize=9)
        ax_acc.grid(alpha=0.2)
        ax_acc.set_xlabel("Epoch", fontsize=8)

        # ── right axis: gamma ────────────────────────────────────
        ax_gam = ax_acc.twinx()
        ax_gam.plot(epochs, gam_mean, color=gam_color, linewidth=1.5,
                    linestyle="--", label="$\\gamma$")
        ax_gam.fill_between(epochs, gam_mean - gam_err, gam_mean + gam_err,
                            color=gam_color, alpha=0.15)
        ax_gam.set_ylim(-0.05, 1.05)
        if col == len(LABEL_MODES) - 1:
            ax_gam.set_ylabel("$\\gamma$", fontsize=8, color=gam_color)
            ax_gam.tick_params(axis="y", labelcolor=gam_color, labelsize=8)
        else:
            ax_gam.tick_params(axis="y", right=False, labelright=False)

        # ── combined legend ──────────────────────────────────────
        lines  = ax_acc.get_lines() + ax_gam.get_lines()
        labels = [l.get_label() for l in lines]
        ax_acc.legend(lines, labels, fontsize=7, loc="center right")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    load_plot_config()

    # plot_accuracy(
    #     results_path="./data/online_gamma_20seeds_sweep.json",
    #     save_path="./results/online_gamma_20seeds_accuracy.png",
    # )

    # plot_gamma_trajectory(
    #     results_path="./data/online_gamma_20seeds_sweep.json",
    #     save_path="./results/online_gamma_20seeds_trajectory.png",
    # )

    # Zoom into late training
    plot_accuracy_gamma_combined(
        epoch_start=0,
        epoch_end=2500,
        save_path="./results/context_identification/online_gamma/online_gamma_20seeds_combined_zoom.png"
    )
