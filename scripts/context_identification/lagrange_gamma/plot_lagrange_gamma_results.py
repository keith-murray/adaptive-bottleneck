"""Plot results from the closed-form optimal gamma sweep.

Produces two figures saved to results/context_identification/lagrange_gamma/:
  1. closed_form_gamma_accuracy.png  — test-accuracy curves (1×3 panels)
  2. closed_form_gamma_trajectory.png — per-epoch mean gamma (1×3 panels)

Result structure: data[label_mode][seed] -> history dict with keys
  'test_acc' / 'test_acc_f' / 'test_acc_p' (both mode), 'gamma'.
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
    results_path="./data/closed_form_gamma_sweep.json",
    save_path="./results/context_identification/lagrange_gamma/closed_form_gamma_accuracy.png",
):
    results = load_json(results_path)
    color = plt.cm.plasma(0.6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, mode in zip(axes, LABEL_MODES):
        seed_dict = results[mode]

        if mode == "both":
            curves_f = collect_curves(seed_dict, "test_acc_f")
            curves_p = collect_curves(seed_dict, "test_acc_p")
            epochs = np.arange(curves_f.shape[1])
            plot_mean_sem(ax, epochs, curves_f, color=plt.cm.plasma(0.3), label="feature")
            plot_mean_sem(ax, epochs, curves_p, color=plt.cm.plasma(0.75), label="position")
            ax.legend(fontsize=8, loc="lower right")
        else:
            curves = collect_curves(seed_dict, "test_acc")
            epochs = np.arange(curves.shape[1])
            plot_mean_sem(ax, epochs, curves, color=color)

        ax.set_title(MODE_TITLES[mode])
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Closed-form $\\gamma$: test-accuracy dynamics", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


# ============================================================
# Figure 2: gamma trajectory
# ============================================================

def plot_gamma_trajectory(
    results_path="./data/closed_form_gamma_sweep.json",
    save_path="./results/context_identification/lagrange_gamma/closed_form_gamma_trajectory.png",
):
    results = load_json(results_path)
    color = plt.cm.plasma(0.6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

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

    axes[0].set_ylabel(r"$\gamma$ (per-epoch mean)")
    fig.suptitle("Closed-form $\\gamma$: gamma trajectory", fontsize=13)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()

# ============================================================
# Figure 3: test accuracy curves
# ============================================================

def plot_mse(
    results_path="./data/closed_form_gamma_sweep.json",
    save_path="./results/context_identification/lagrange_gamma/closed_form_gamma_accuracy.png",
):
    results = load_json(results_path)
    color = plt.cm.plasma(0.6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

    for ax, mode in zip(axes, LABEL_MODES):
        seed_dict = results[mode]

        if mode == "both":
            curves_f = collect_curves(seed_dict, "test_loss_f")
            curves_p = collect_curves(seed_dict, "test_loss_p")
            epochs = np.arange(curves_f.shape[1])
            plot_mean_sem(ax, epochs, curves_f, color=plt.cm.plasma(0.3), label="feature")
            plot_mean_sem(ax, epochs, curves_p, color=plt.cm.plasma(0.75), label="position")
            ax.legend(fontsize=8, loc="lower right")
        else:
            curves = collect_curves(seed_dict, "test_loss")
            epochs = np.arange(curves.shape[1])
            plot_mean_sem(ax, epochs, curves, color=color)

        ax.set_title(MODE_TITLES[mode])
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Test Loss")
    fig.suptitle("Closed-form $\\gamma$: test-accuracy dynamics", fontsize=13)
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
        results_path="./data/lagrange_gamma_sweep.json",
        save_path="./results/context_identification/lagrange_gamma/lagrange_gamma_accuracy.png",
    )

    plot_gamma_trajectory(
        results_path="./data/lagrange_gamma_sweep.json",
        save_path="./results/context_identification/lagrange_gamma/lagrange_gamma_trajectory.png",
    )
    
    # plot_mse(
    #     results_path="./data/regress_lagrange_gamma_sweep.json",
    #     save_path="./results/context_identification/lagrange_gamma/regress_lagrange_gamma_loss.png",
    # )
