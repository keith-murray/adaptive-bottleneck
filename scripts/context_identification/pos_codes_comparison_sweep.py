"""Positional-codes comparison sweep for ContextIdentification.

Compares three conditions on the feature and position tasks separately,
across 20 seeds each:

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
# Training steps
# ============================================================

def make_train_step_adaptive(optimizer, gamma_optimizer):
    """Online training step: joint gradient on params and gamma."""
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
    """Training step with fixed gamma (self-attention baseline)."""
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


def make_eval_step_adaptive():
    @jax.jit
    def eval_step(params_attn, X, y, gamma):
        logits, _ = forward_logits(params_attn, X, gamma=gamma)
        return ce_with_logits(logits, y), acc_from_logits(logits, y)
    return eval_step


def make_eval_step_fixed(gamma):
    @jax.jit
    def eval_step(params_attn, X, y):
        logits, _ = forward_logits(params_attn, X, gamma=gamma)
        return ce_with_logits(logits, y), acc_from_logits(logits, y)
    return eval_step


# ============================================================
# Training function
# ============================================================

def train_one_run(
    *,
    key_init,
    condition,
    label_mode,
    ds_train,
    ds_test,
    in_dim,
    L,
    epochs,
    lr,
    weight_decay,
    gamma_lr,
    init_gamma=0.5,
    show_progress=True,
):
    """Train one run for a given condition.

    Args:
        key_init: JAX PRNG key
        condition: 'adaptive', 'adaptive_pos', or 'attn_pos'
        label_mode: 'feature' or 'position'
        ds_train: training dataset
        ds_test: test dataset
        in_dim: input dimension (L without pos codes, 2*L with)
        L: feature vector length (= number of classes)
        epochs: number of training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        gamma_lr: learning rate for gamma (adaptive conditions only)
        init_gamma: initial gamma value (adaptive conditions only)
        show_progress: whether to show progress bar

    Returns:
        history: dict with train/test metrics and gamma evolution
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
        eval_step = make_eval_step_adaptive()
    else:
        gamma_fixed = jnp.array(0.0, dtype=jnp.float32)
        train_step = make_train_step_fixed(optimizer, gamma=gamma_fixed)
        eval_step = make_eval_step_fixed(gamma=gamma_fixed)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "gamma": [],
    }

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"{condition} | {label_mode}", total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs = [], []

        if is_adaptive:
            for X, y in ds_train:
                params_attn, gamma_param, opt_state, gamma_opt_state, loss, acc = train_step(
                    params_attn, gamma_param, opt_state, gamma_opt_state, X, y
                )
                train_losses.append(loss)
                train_accs.append(acc)
        else:
            for X, y in ds_train:
                params_attn, opt_state, loss, acc = train_step(params_attn, opt_state, X, y)
                train_losses.append(loss)
                train_accs.append(acc)

        tr_loss = float(jnp.mean(jnp.stack(train_losses)))
        tr_acc = float(jnp.mean(jnp.stack(train_accs)))

        test_losses, test_accs = [], []
        if is_adaptive:
            for X, y in ds_test:
                loss, acc = eval_step(params_attn, X, y, gamma_param)
                test_losses.append(loss)
                test_accs.append(acc)
        else:
            for X, y in ds_test:
                loss, acc = eval_step(params_attn, X, y)
                test_losses.append(loss)
                test_accs.append(acc)

        te_loss = float(jnp.mean(jnp.stack(test_losses)))
        te_acc = float(jnp.mean(jnp.stack(test_accs)))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["gamma"].append(float(gamma_param) if is_adaptive else 0.0)

        if pbar is not None:
            postfix = {"loss": f"{tr_loss:.3f}", "acc": f"{tr_acc:.3f}"}
            if is_adaptive:
                postfix["γ"] = f"{float(gamma_param):.3f}"
            pbar.set_postfix(postfix)

    if pbar is not None:
        pbar.close()

    return history, params_attn


# ============================================================
# Sweep function
# ============================================================

def run_sweep(
    *,
    seeds=tuple(range(20)),
    conditions=CONDITIONS,
    label_modes=("feature", "position"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=False,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    gamma_lr=1e-3,
    init_gamma=0.5,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Run positional-codes comparison sweep.

    Args:
        seeds: random seeds (default: 20)
        conditions: which conditions to run (default: all three)
        label_modes: 'feature' and/or 'position'
        num_samples: samples per dataset
        L: feature vector length (= number of classes)
        allow_ref_in_context: whether reference can appear in context
        batch_size: batch size
        epochs: training epochs
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

    total_runs = len(seeds) * len(conditions) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Pos Codes Comparison", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test = random.split(key, 2)

        for condition in conditions:
            use_pos_codes = condition in ("adaptive_pos", "attn_pos")
            in_dim = 2 * L if use_pos_codes else L

            for label_mode in label_modes:
                task_train = ContextIdentification(
                    key_train,
                    num_samples=num_samples,
                    num_vectors=L + 1,
                    vector_len=L,
                    allow_ref_in_context=allow_ref_in_context,
                    input_mode="cross",
                    label_mode=label_mode,
                    positional_codes=use_pos_codes,
                )
                task_test = ContextIdentification(
                    key_test,
                    num_samples=num_samples,
                    num_vectors=L + 1,
                    vector_len=L,
                    allow_ref_in_context=allow_ref_in_context,
                    input_mode="cross",
                    label_mode=label_mode,
                    positional_codes=use_pos_codes,
                )

                ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
                ds_test = task_test.generate_dataset(batch_size=batch_size, shuffle=False)

                key_init = random.fold_in(key, hash((condition, label_mode)) % 10000)

                history, params_attn = train_one_run(
                    key_init=key_init,
                    condition=condition,
                    label_mode=label_mode,
                    ds_train=ds_train,
                    ds_test=ds_test,
                    in_dim=in_dim,
                    L=L,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    gamma_lr=gamma_lr,
                    init_gamma=init_gamma,
                    show_progress=show_progress,
                )

                results[condition][label_mode][int(seed)] = history

                if models_dir is not None:
                    run_name = f"pos_codes_{condition}_{label_mode}_seed{seed}"
                    _save_params_pkl(params_attn, models_dir / f"{run_name}_params.pkl")

                if outer is not None:
                    outer.update(1)

    if outer is not None:
        outer.close()

    if save_path is not None:
        save_results_json(results, save_path)

    return results


if __name__ == "__main__":
    results = run_sweep(
        seeds=tuple(range(20)),
        conditions=CONDITIONS,
        label_modes=("feature", "position"),
        num_samples=500,
        L=10,
        allow_ref_in_context=False,
        batch_size=20,
        epochs=5000,
        lr=1e-2,
        weight_decay=1e-3,
        gamma_lr=1e-2,
        init_gamma=0.5,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/pos_codes_comparison_models/",
        save_path="./data/pos_codes_comparison_sweep.json",
    )
