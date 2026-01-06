"""Train and save 4 ContextIdentification models.

This script trains four models:
  1) gamma=0 on feature task
  2) gamma=1 on feature task
  3) gamma=0 on position task
  4) gamma=1 on position task

For each run, we save:
  - learning dynamics (train/test loss/acc over epochs) as JSON
  - trained attention parameters as a pickle (JAX pytree)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as random

from adaptive_bottleneck.tasks import ContextIdentification
from adaptive_bottleneck.save import _ensure_dir, _save_json, _save_params_pkl

# Reuse the existing training loop implemented in the sweep script.
from context_identification_bottleneck import train_one_run


@dataclass(frozen=True)
class RunSpec:
    label_mode: str  # "feature" | "position"
    gamma: float


def _make_datasets_for_label_mode(
    *,
    key_train: jax.Array,
    key_test: jax.Array,
    label_mode: str,
    num_samples: int,
    L: int,
    allow_ref_in_context: bool,
    batch_size: int,
) -> Tuple[Any, Any]:
    """Build TF datasets for one label mode."""

    task_train = ContextIdentification(
        key_train,
        num_samples=num_samples,
        num_vectors_and_vector_len=L,
        allow_ref_in_context=allow_ref_in_context,
        label_mode=label_mode,
    )
    task_test = ContextIdentification(
        key_test,
        num_samples=num_samples,
        num_vectors_and_vector_len=L,
        allow_ref_in_context=allow_ref_in_context,
        label_mode=label_mode,
    )

    ds_train = task_train.generate_tf_dataset(batch_size=batch_size, shuffle=True)
    ds_test = task_test.generate_tf_dataset(batch_size=batch_size, shuffle=False)
    return ds_train, ds_test


def run_four_models(
    *,
    seed: int = 0,
    L: int = 10,
    num_samples: int = 2000,
    allow_ref_in_context: bool = False,
    batch_size: int = 64,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    attn_kind: str = "softmax",
    show_progress: bool = True,
    out_dir: str = "./data/context_identification_four_models",
) -> Dict[str, Dict[str, Any]]:
    """Train four (label_mode, gamma) combinations and save weights + dynamics.

    Returns a nested dict:
        results[label_mode][gamma_str] -> {"history": ..., "params_path": ...}
    """

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Base RNG streams
    key = random.PRNGKey(int(seed))
    key_train_base, key_test_base = random.split(key, 2)

    # Fixed set of four runs
    run_specs = (
        RunSpec(label_mode="feature", gamma=0.0),
        RunSpec(label_mode="feature", gamma=1.0),
        RunSpec(label_mode="position", gamma=0.0),
        RunSpec(label_mode="position", gamma=1.0),
    )

    # Build datasets once per label_mode (reused across gammas)
    datasets: Dict[str, Tuple[Any, Any]] = {}
    for lm in ("feature", "position"):
        # Fold-in label_mode so train/test datasets differ across label modes but are reproducible.
        lm_tag = 123 if lm == "feature" else 456
        key_train = random.fold_in(key_train_base, lm_tag)
        key_test = random.fold_in(key_test_base, lm_tag)
        datasets[lm] = _make_datasets_for_label_mode(
            key_train=key_train,
            key_test=key_test,
            label_mode=lm,
            num_samples=num_samples,
            L=L,
            allow_ref_in_context=allow_ref_in_context,
            batch_size=batch_size,
        )

    results: Dict[str, Dict[str, Any]] = {"feature": {}, "position": {}}

    for spec in run_specs:
        ds_train, ds_test = datasets[spec.label_mode]

        # Fresh init per run; fold in gamma + label_mode + attn_kind for determinism.
        # (This mirrors the sweep logic but is simpler.)
        key_init = random.fold_in(key, 10_000 + int(1e6 * float(spec.gamma)))
        key_init = random.fold_in(key_init, 111 if spec.label_mode == "feature" else 222)
        key_init = random.fold_in(key_init, 333 if attn_kind == "linear" else 444)

        history, params_attn = train_one_run(
            key_init=key_init,
            gamma=float(spec.gamma),
            label_mode=spec.label_mode,
            attn_kind=attn_kind,
            ds_train=ds_train,
            ds_test=ds_test,
            L=L,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            show_progress=show_progress,
        )

        gamma_str = f"{spec.gamma:.1f}"  # "0.0" or "1.0"
        run_name = f"context_id_{spec.label_mode}_gamma{gamma_str}_{attn_kind}_seed{seed}"

        params_path = out_dir / f"{run_name}_params.pkl"
        hist_path = out_dir / f"{run_name}_history.json"

        _save_params_pkl(params_attn, params_path)
        _save_json(history, hist_path)

        results[spec.label_mode][gamma_str] = {
            "history": history,
            "params_path": str(params_path),
            "history_path": str(hist_path),
            "run_name": run_name,
        }

    # Save a manifest to make downstream analysis easy
    manifest = {
        "seed": seed,
        "L": L,
        "num_samples": num_samples,
        "allow_ref_in_context": allow_ref_in_context,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "attn_kind": attn_kind,
        "results": results,
    }
    _save_json(manifest, out_dir / "manifest.json")

    return results


if __name__ == "__main__":
    # You can override any of these via environment variables if you want.
    seed = 0
    L = 10
    num_samples = 3200
    allow_ref_in_context = False
    batch_size = 32
    epochs = 2500
    lr = 1e-2
    weight_decay = 1e-4
    attn_kind = "softmax"
    show_progress = True
    out_dir = "../data/context_identification_models"

    run_four_models(
        seed=seed,
        L=L,
        num_samples=num_samples,
        allow_ref_in_context=allow_ref_in_context,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        attn_kind=attn_kind,
        show_progress=show_progress,
        out_dir=out_dir,
    )
