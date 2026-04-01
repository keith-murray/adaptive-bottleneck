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
# Main
# ============================================================

if __name__ == "__main__":
    load_plot_config()

    plot_accuracy(
        results_path="./data/online_gamma_20seeds_sweep.json",
        save_path="./results/online_gamma_20seeds_accuracy.png",
    )

    plot_gamma_trajectory(
        results_path="./data/online_gamma_20seeds_sweep.json",
        save_path="./results/online_gamma_20seeds_trajectory.png",
    )
