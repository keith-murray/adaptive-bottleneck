"""Plot results from the XOR online gamma comparison.

Produces one figure saved to results/xor/:
  xor_online_gamma_comparison.png — 2×2 panels:
    top row:    test-accuracy curves (left: indexing, right: decoder)
    bottom row: gamma trajectories   (left: indexing, right: decoder)

All three conditions (rca, cross_attn, arca) are overlaid in each panel.

Result structure: data[arch_kind][condition][seed] -> history dict with keys
  'test_acc', 'gamma'.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_bottleneck.utils import load_plot_config


ARCH_KINDS = ("indexing", "decoder")
ARCH_TITLES = {"indexing": "Indexing (no decoder)", "decoder": "Linear decoder"}

CONDITIONS = ("rca", "cross_attn", "arca")
CONDITION_LABELS = {
    "rca":        "RCA ($\\gamma=1$)",
    "cross_attn": "Cross-attention ($\\gamma=0$)",
    "arca":       "ARCA ($\\gamma_0=0.5$)",
}
CONDITION_COLORS = {
    "rca":        plt.cm.plasma(0.85),
    "cross_attn": plt.cm.plasma(0.15),
    "arca":       plt.cm.plasma(0.5),
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


def plot_individual_trajectories(ax, epochs, curves, color):
    for seed_curve in curves:
        ax.plot(epochs, seed_curve, color=color, linewidth=0.5, alpha=0.2)


def plot_comparison(
    results_path="./data/xor_online_gamma_comparison.json",
    save_path="./results/xor/xor_online_gamma_comparison.png",
):
    results = load_json(results_path)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

    for col, arch_kind in enumerate(ARCH_KINDS):
        ax_acc = axes[0, col]
        ax_gam = axes[1, col]

        for condition in CONDITIONS:
            seed_dict = results[arch_kind][condition]
            label = CONDITION_LABELS[condition]
            color = CONDITION_COLORS[condition]

            acc_curves = collect_curves(seed_dict, "test_acc")
            gam_curves = collect_curves(seed_dict, "gamma")
            epochs = np.arange(acc_curves.shape[1])

            plot_mean_sem(ax_acc, epochs, acc_curves, color=color, label=label)
            plot_mean_sem(ax_gam, epochs, gam_curves, color=color)

            if condition == "arca":
                plot_individual_trajectories(ax_acc, epochs, acc_curves, color=color)
                plot_individual_trajectories(ax_gam, epochs, gam_curves, color=color)

        ax_acc.set_title(ARCH_TITLES[arch_kind])
        ax_acc.set_ylim(0.0, 1.05)
        ax_acc.grid(alpha=0.2)

        ax_gam.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_gam.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_gam.set_xlabel("Epoch")
        ax_gam.set_ylim(-0.05, 1.05)
        ax_gam.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Test accuracy")
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[0, 1].legend(fontsize=8, loc="lower right")

    fig.suptitle("XOR: online $\\gamma$ comparison (20 seeds)", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


if __name__ == "__main__":
    load_plot_config()

    plot_comparison(
        results_path="./data/xor_online_gamma_comparison.json",
        save_path="./results/xor/xor_online_gamma_comparison.png",
    )
