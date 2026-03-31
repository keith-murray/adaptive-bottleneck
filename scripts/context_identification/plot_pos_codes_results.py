"""Plot results from the ContextIdentification positional-codes sweeps.

Produces three figures saved to results/:
  1. pos_codes_fixed_gamma_sweep.png    — fixed-γ test-accuracy curves (1×2 panels)
  2. pos_codes_learnable_gamma_sweep.png — learnable-γ accuracy + γ trajectory (2×2 panels)
  3. pos_codes_meta_gamma_sweep.png     — meta-γ accuracy + γ trajectory (2×2 panels)

Data structures:
  Fixed:    results[label_mode]["softmax"][gamma_str][seed_str] -> history
  Learnable/Meta: results[label_mode][init_gamma_str][seed_str] -> history
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from adaptive_bottleneck.utils import load_plot_config


LABEL_MODES = ("feature", "position")
MODE_TITLES = {"feature": "Feature", "position": "Position"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def collect_curves(seed_dict, key="test_acc"):
    """Stack per-seed histories into (S, E) array."""
    return np.stack(
        [np.array(hist[key]) for hist in seed_dict.values()],
        axis=0,
    )


def plot_mean_sem(ax, epochs, curves, color, label=None):
    mean = curves.mean(axis=0)
    sem = curves.std(axis=0) / np.sqrt(curves.shape[0])
    ax.plot(epochs, mean, color=color, linewidth=1.5, alpha=0.9, label=label)
    ax.fill_between(epochs, mean - sem, mean + sem, color=color, alpha=0.15)


# ============================================================
# Figure 1: fixed-gamma sweep (softmax only)
# ============================================================

def plot_fixed_gamma_sweep(
    results_path="./data/context_id_pos_codes_gamma_sweep.json",
    save_path="./results/pos_codes_fixed_gamma_sweep.png",
):
    results = load_json(results_path)

    gammas = sorted(float(g) for g in results["feature"]["softmax"].keys())
    cmap = plt.cm.plasma
    norm = Normalize(vmin=min(gammas), vmax=max(gammas))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    for ax, mode in zip(axes, LABEL_MODES):
        for g in gammas:
            seed_dict = results[mode]["softmax"][str(g)]
            curves = collect_curves(seed_dict, "test_acc")
            epochs = np.arange(curves.shape[1])
            plot_mean_sem(ax, epochs, curves, color=cmap(norm(g)))

        ax.set_title(MODE_TITLES[mode])
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Test accuracy")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], location="right", pad=0.02)
    cbar.set_label(r"$\gamma$")

    fig.suptitle("Positional codes: fixed-$\\gamma$ test accuracy (softmax)", fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 2: learnable-gamma sweep
# ============================================================

def plot_learnable_gamma_sweep(
    results_path="./data/context_id_pos_codes_learnable_gamma_sweep.json",
    save_path="./results/pos_codes_learnable_gamma_sweep.png",
):
    results = load_json(results_path)

    init_gammas = sorted(float(g) for g in results["feature"].keys())
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(init_gammas) - 1)) for i in range(len(init_gammas))]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

    for col, mode in enumerate(LABEL_MODES):
        ax_acc = axes[0, col]
        ax_gam = axes[1, col]

        for i, g in enumerate(init_gammas):
            seed_dict = results[mode][str(g)]
            acc_curves = collect_curves(seed_dict, "test_acc")
            gam_curves = collect_curves(seed_dict, "gamma")
            epochs = np.arange(acc_curves.shape[1])

            label = f"$\\gamma_0$={g:.2f}"
            plot_mean_sem(ax_acc, epochs, acc_curves, color=colors[i], label=label)
            plot_mean_sem(ax_gam, epochs, gam_curves, color=colors[i])

        ax_acc.set_title(MODE_TITLES[mode])
        ax_acc.set_ylim(0.0, 1.05)
        ax_acc.grid(alpha=0.2)

        ax_gam.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_gam.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_gam.set_ylim(-0.05, 1.05)
        ax_gam.set_xlabel("Epoch")
        ax_gam.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Test accuracy")
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[0, 1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Positional codes: learnable-$\\gamma$ dynamics (softmax)", fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 3: meta-gamma sweep
# ============================================================

def plot_meta_gamma_sweep(
    results_path="./data/context_id_pos_codes_meta_gamma_sweep.json",
    save_path="./results/pos_codes_meta_gamma_sweep.png",
):
    results = load_json(results_path)

    init_gammas = sorted(float(g) for g in results["feature"].keys())
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(init_gammas) - 1)) for i in range(len(init_gammas))]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

    for col, mode in enumerate(LABEL_MODES):
        ax_acc = axes[0, col]
        ax_gam = axes[1, col]

        for i, g in enumerate(init_gammas):
            seed_dict = results[mode][str(g)]
            acc_curves = collect_curves(seed_dict, "test_acc")
            gam_curves = collect_curves(seed_dict, "gamma")
            epochs = np.arange(acc_curves.shape[1])

            label = f"$\\gamma_0$={g:.2f}"
            plot_mean_sem(ax_acc, epochs, acc_curves, color=colors[i], label=label)
            plot_mean_sem(ax_gam, epochs, gam_curves, color=colors[i])

        ax_acc.set_title(MODE_TITLES[mode])
        ax_acc.set_ylim(0.0, 1.05)
        ax_acc.grid(alpha=0.2)

        ax_gam.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_gam.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
        ax_gam.set_ylim(-0.05, 1.05)
        ax_gam.set_xlabel("Epoch")
        ax_gam.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Test accuracy")
    axes[1, 0].set_ylabel(r"$\gamma$")
    axes[0, 1].legend(fontsize=8, loc="lower right")

    fig.suptitle("Positional codes: meta-$\\gamma$ dynamics (softmax)", fontsize=14)
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

    plot_fixed_gamma_sweep(
        results_path="./data/context_id_pos_codes_gamma_sweep.json",
        save_path="./results/pos_codes_fixed_gamma_sweep.png",
    )

    plot_learnable_gamma_sweep(
        results_path="./data/context_id_pos_codes_learnable_gamma_sweep.json",
        save_path="./results/pos_codes_learnable_gamma_sweep.png",
    )

    plot_meta_gamma_sweep(
        results_path="./data/context_id_pos_codes_meta_gamma_sweep.json",
        save_path="./results/pos_codes_meta_gamma_sweep.png",
    )
