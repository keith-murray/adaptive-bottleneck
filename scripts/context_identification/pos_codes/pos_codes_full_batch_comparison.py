"""Positional-codes comparison for ContextIdentification, trained with full-batch
gradient descent instead of SGD (c.f. pos_codes_comparison_sweep.py).

Full batch = the exhaustive enumeration of every valid task instance for a chosen
L (see ContextIdentification.generate_all_examples() and
full_batch_combinatorics.py for why L must stay small, e.g. L=6 -> 27,684 unique
instances). One gradient step now sees the *entire* population, so there is no
train/test split -- there is nothing left to hold out for generalization -- and
history is indexed by training step, not epoch.

Compares three conditions on the feature and position tasks separately, across 20
seeds each:

  adaptive     — online gamma (init=0.5), no positional codes
  adaptive_pos — online gamma (init=0.5), with positional codes
  attn_pos     — fixed gamma=0 (self-attention), with positional codes

With positional codes, in_dim = 2*L (feature one-hot + position one-hot).
Without positional codes, in_dim = L.

Result format: results[condition][label_mode][seed] -> history
"""

from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    gamma_value_cross_attention_forward,
)
from adaptive_bottleneck.tasks import ContextIdentification
from adaptive_bottleneck.save import save_results_json, _save_params_pkl, _ensure_dir


CONDITIONS = ("adaptive", "adaptive_pos", "attn_pos")


def ce_with_logits(logits, y):
    """Cross-entropy for integer labels."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def acc_from_logits(logits, y):
    """Accuracy from logits."""
    return (jnp.argmax(logits, axis=-1) == y).mean()


def forward_logits(params_attn, X, gamma):
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]
    out, attn_w = gamma_value_cross_attention_forward(
        params_attn, X_c, X_q, gamma=gamma, linear=False,
    )
    return out[:, 0, :], attn_w


# ============================================================
# Training steps (identical scaffold to pos_codes_comparison_sweep.py;
# only the caller now passes the full population instead of a mini-batch)
# ============================================================

def make_train_step_adaptive(optimizer, gamma_optimizer):
    """Full-batch training step: joint gradient on params and gamma."""
    @jax.jit
    def train_step(params_attn, gamma, opt_state, gamma_opt_state, X, y):
        def loss_fn(p, g):
            logits, _ = forward_logits(p, X, gamma=g)
            return ce_with_logits(logits, y)

        loss, (grads_attn, grads_gamma) = jax.value_and_grad(
            loss_fn, argnums=(0, 1)
        )(params_attn, gamma)

        updates, opt_state = optimizer.update(grads_attn, opt_state, params_attn)
        params_attn = optax.apply_updates(params_attn, updates)

        gamma_updates, gamma_opt_state = gamma_optimizer.update(grads_gamma, gamma_opt_state, gamma)
        gamma = optax.apply_updates(gamma, gamma_updates)
        gamma = jnp.clip(gamma, 0.0, 1.0)

        logits, _ = forward_logits(params_attn, X, gamma=gamma)
        acc = acc_from_logits(logits, y)

        return params_attn, gamma, opt_state, gamma_opt_state, loss, acc

    return train_step


def make_train_step_fixed(optimizer, gamma):
    """Full-batch training step with fixed gamma (self-attention baseline)."""
    @jax.jit
    def train_step(params_attn, opt_state, X, y):
        def loss_fn(p):
            logits, _ = forward_logits(p, X, gamma=gamma)
            return ce_with_logits(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params_attn)
        updates, opt_state = optimizer.update(grads, opt_state, params_attn)
        params_attn = optax.apply_updates(params_attn, updates)

        logits, _ = forward_logits(params_attn, X, gamma=gamma)
        acc = acc_from_logits(logits, y)

        return params_attn, opt_state, loss, acc

    return train_step


# ============================================================
# Training function
# ============================================================

def train_one_run(
    *,
    key_init,
    condition,
    label_mode,
    X,
    y,
    in_dim,
    L,
    num_steps,
    lr,
    weight_decay,
    gamma_lr,
    init_gamma=0.5,
    show_progress=True,
):
    """Train one full-batch run for a given condition.

    Args:
        key_init: JAX PRNG key
        condition: 'adaptive', 'adaptive_pos', or 'attn_pos'
        label_mode: 'feature' or 'position'
        X, y: the full enumerated population (see ContextIdentification.generate_all_examples)
        in_dim: input dimension (L without pos codes, 2*L with)
        L: feature vector length (= number of classes)
        num_steps: number of full-batch gradient steps
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        gamma_lr: learning rate for gamma (adaptive conditions only)
        init_gamma: initial gamma value (adaptive conditions only)
        show_progress: whether to show progress bar

    Returns:
        history: dict with per-step train metrics and gamma evolution
        params_attn: trained parameters
    """
    is_adaptive = condition in ("adaptive", "adaptive_pos")

    params_attn = init_gamma_value_attn_params(key_init, in_dim=in_dim, attn_dim=L, val_dim=L)

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params_attn)

    if is_adaptive:
        gamma_param = jnp.array(init_gamma, dtype=jnp.float32)
        gamma_optimizer = optax.sgd(learning_rate=gamma_lr)
        gamma_opt_state = gamma_optimizer.init(gamma_param)
        train_step = make_train_step_adaptive(optimizer, gamma_optimizer)
    else:
        gamma_fixed = jnp.array(0.0, dtype=jnp.float32)
        train_step = make_train_step_fixed(optimizer, gamma=gamma_fixed)

    history = {"train_loss": [], "train_acc": [], "gamma": []}

    step_iter = range(num_steps)
    pbar = None
    if show_progress:
        pbar = tqdm(step_iter, desc=f"{condition} | {label_mode}", total=num_steps, leave=True)
        step_iter = pbar

    for _step in step_iter:
        if is_adaptive:
            params_attn, gamma_param, opt_state, gamma_opt_state, loss, acc = train_step(
                params_attn, gamma_param, opt_state, gamma_opt_state, X, y
            )
        else:
            params_attn, opt_state, loss, acc = train_step(params_attn, opt_state, X, y)

        history["train_loss"].append(float(loss))
        history["train_acc"].append(float(acc))
        history["gamma"].append(float(gamma_param) if is_adaptive else 0.0)

        if pbar is not None:
            postfix = {"loss": f"{float(loss):.3f}", "acc": f"{float(acc):.3f}"}
            if is_adaptive:
                postfix["γ"] = f"{float(gamma_param):.3f}"
            pbar.set_postfix(postfix)

    if pbar is not None:
        pbar.close()

    return history, params_attn


# ============================================================
# Comparison function
# ============================================================

def run_comparison(
    *,
    seeds=tuple(range(20)),
    conditions=CONDITIONS,
    label_modes=("feature", "position"),
    L=6,
    num_steps=2000,
    lr=1e-2,
    weight_decay=1e-3,
    gamma_lr=1e-2,
    init_gamma=0.5,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Run the full-batch positional-codes comparison.

    Args:
        seeds: random seeds for parameter initialization (default: 20)
        conditions: which conditions to run (default: all three)
        label_modes: 'feature' and/or 'position'
        L: feature vector length (= number of classes); also fixes context_size=L,
            so the full population has N(L) instances (see full_batch_combinatorics.py)
        num_steps: full-batch gradient steps
        lr: learning rate for model parameters
        weight_decay: weight decay
        gamma_lr: learning rate for gamma (adaptive conditions)
        init_gamma: initial gamma (adaptive conditions)
        show_progress: per-run progress bars
        show_combo_progress: overall progress bar
        models_dir: directory to save model parameters
        save_path: path to save results JSON

    Returns:
        results[condition][label_mode][seed] -> history
    """
    if models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {cond: {lm: {} for lm in label_modes} for cond in conditions}

    # The enumerated population is deterministic (no PRNG dependence) and only
    # varies with (positional_codes, label_mode), so build it once per combo and
    # reuse it across every seed/condition that shares that combo.
    datasets = {}
    for use_pos_codes in (False, True):
        for label_mode in label_modes:
            task = ContextIdentification(
                random.PRNGKey(0),
                num_vectors=L + 1,
                vector_len=L,
                input_mode="cross",
                label_mode=label_mode,
                positional_codes=use_pos_codes,
            )
            X, y = task.generate_all_examples()
            datasets[(use_pos_codes, label_mode)] = (jnp.asarray(X), jnp.asarray(y))

    total_runs = len(seeds) * len(conditions) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Pos Codes Full-Batch Comparison", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))

        for condition in conditions:
            use_pos_codes = condition in ("adaptive_pos", "attn_pos")
            in_dim = 2 * L if use_pos_codes else L

            for label_mode in label_modes:
                X, y = datasets[(use_pos_codes, label_mode)]

                key_init = random.fold_in(key, hash((condition, label_mode)) % 10000)

                history, params_attn = train_one_run(
                    key_init=key_init,
                    condition=condition,
                    label_mode=label_mode,
                    X=X,
                    y=y,
                    in_dim=in_dim,
                    L=L,
                    num_steps=num_steps,
                    lr=lr,
                    weight_decay=weight_decay,
                    gamma_lr=gamma_lr,
                    init_gamma=init_gamma,
                    show_progress=show_progress,
                )

                results[condition][label_mode][int(seed)] = history

                if models_dir is not None:
                    run_name = f"pos_codes_full_batch_{condition}_{label_mode}_seed{seed}"
                    _save_params_pkl(params_attn, models_dir / f"{run_name}_params.pkl")

                if outer is not None:
                    outer.update(1)

    if outer is not None:
        outer.close()

    if save_path is not None:
        save_results_json(results, save_path)

    return results


if __name__ == "__main__":
    results = run_comparison(
        seeds=tuple(range(20)),
        conditions=CONDITIONS,
        label_modes=("feature", "position"),
        L=6,
        num_steps=2000,
        lr=1e-2,
        weight_decay=1e-3,
        gamma_lr=1e-2,
        init_gamma=0.5,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/pos_codes_full_batch_comparison_models/",
        save_path="./data/pos_codes_full_batch_comparison_sweep.json",
    )
