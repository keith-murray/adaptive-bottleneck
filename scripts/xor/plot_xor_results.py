"""Plot results from XOR fixed-gamma and learnable-gamma sweeps.

Produces two figures:
  1. results/xor_gamma_sweep.png       — fixed-gamma test-accuracy curves
  2. results/xor_learnable_gamma_sweep.png — learnable-gamma accuracy + gamma trajectories
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from adaptive_bottleneck.utils import load_plot_config


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
# Figure 1: fixed-gamma sweep
# ============================================================

def plot_fixed_gamma_sweep(
    results_path="./data/xor_gamma_sweep.json",
    save_path="./results/xor_gamma_sweep.png",
):
    results = load_json(results_path)

    attn_kinds = ["linear", "softmax"]
    gammas = sorted(float(g) for g in results[attn_kinds[0]].keys())

    cmap = plt.cm.plasma
    norm = Normalize(vmin=min(gammas), vmax=max(gammas))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)

    for ax, attn_kind in zip(axes, attn_kinds):
        for g in gammas:
            g_key = str(g)
            seed_dict = results[attn_kind][g_key]
            curves = collect_curves(seed_dict, "test_acc")
            epochs = np.arange(curves.shape[1])
            plot_mean_sem(ax, epochs, curves, color=cmap(norm(g)))

        ax.set_title(attn_kind)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Test accuracy")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[-1], location="right", pad=0.02)
    cbar.set_label(r"$\gamma$")

    fig.suptitle("XOR: fixed-$\\gamma$ test-accuracy dynamics", fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 2: learnable-gamma sweep
# ============================================================

def plot_learnable_gamma_sweep(
    results_path="./data/xor_learnable_gamma_sweep.json",
    save_path="./results/xor_learnable_gamma_sweep.png",
):
    results = load_json(results_path)

    attn_kinds = ["linear", "softmax"]
    init_gammas = sorted(float(g) for g in results[attn_kinds[0]].keys())

    cmap = plt.cm.plasma
    colors = [cmap(i / (len(init_gammas) - 1)) for i in range(len(init_gammas))]

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

    for col, attn_kind in enumerate(attn_kinds):
        ax_acc = axes[0, col]
        ax_gam = axes[1, col]

        for i, g in enumerate(init_gammas):
            g_key = str(g)
            seed_dict = results[attn_kind][g_key]

            acc_curves = collect_curves(seed_dict, "test_acc")
            gam_curves = collect_curves(seed_dict, "gamma")
            epochs = np.arange(acc_curves.shape[1])

            label = f"$\\gamma_0$={g:.2f}"
            plot_mean_sem(ax_acc, epochs, acc_curves, color=colors[i], label=label)
            plot_mean_sem(ax_gam, epochs, gam_curves, color=colors[i])

        ax_acc.set_title(attn_kind)
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

    fig.suptitle("XOR: learnable-$\\gamma$ dynamics", fontsize=14)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 3: fixed vs learnable gamma comparison (softmax only)
# ============================================================

def plot_fixed_vs_learnable(
    fixed_path="./data/xor_gamma_sweep.json",
    learnable_path="./data/xor_learnable_gamma_sweep.json",
    save_path="./results/xor_fixed_vs_learnable.png",
):
    fixed = load_json(fixed_path)
    learnable = load_json(learnable_path)

    gammas = sorted(float(g) for g in fixed["softmax"].keys())
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(gammas) - 1)) for i in range(len(gammas))]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    for ax, (results, title) in zip(axes, [
        (fixed["softmax"],     "Fixed $\\gamma$"),
        (learnable["softmax"], "Learnable $\\gamma$"),
    ]):
        for i, g in enumerate(gammas):
            seed_dict = results[str(g)]
            curves = collect_curves(seed_dict, "test_acc")
            epochs = np.arange(curves.shape[1])
            label = f"$\\gamma$={g:.2f}" if title.startswith("Fixed") else f"$\\gamma_0$={g:.2f}"
            plot_mean_sem(ax, epochs, curves, color=colors[i], label=label)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")

    axes[0].set_ylabel("Test accuracy")

    fig.suptitle("XOR (softmax): fixed vs learnable $\\gamma$", fontsize=14)
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
        results_path="./data/xor_gamma_sweep.json",
        save_path="./results/xor_gamma_sweep.png",
    )

    plot_learnable_gamma_sweep(
        results_path="./data/xor_learnable_gamma_sweep.json",
        save_path="./results/xor_learnable_gamma_sweep.png",
    )

    plot_fixed_vs_learnable(
        fixed_path="./data/xor_gamma_sweep.json",
        learnable_path="./data/xor_learnable_gamma_sweep.json",
        save_path="./results/xor_fixed_vs_learnable.png",
    )
