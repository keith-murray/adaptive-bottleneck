"""Learnable gamma sweep for ContextIdentification with positional codes.

This script is a variant of learnable_gamma_sweep.py that adds one-hot positional
codes to the context items. The input dimension grows from L to 2*L.

Label modes: 'feature' and 'position' only (no dual-decoder 'both' mode).
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


def ce_with_logits(logits, y):
    """Cross-entropy for integer labels."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def acc_from_logits(logits, y):
    """Accuracy from logits."""
    preds = jnp.argmax(logits, axis=-1)
    return (preds == y).mean()


def forward_single_logits(params_attn, X, gamma):
    """Forward pass for single-output modes (feature or position).

    Args:
        params_attn: attention parameters
        X: (B, L+1, 2L) input with reference at [:, 0, :] and context at [:, 1:, :]
        gamma: scalar or array, the bottleneck parameter

    Returns:
        logits: (B, val_dim) class logits
        attn_w: (B, 1, L) attention weights
    """
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]

    out, attn_w = gamma_value_cross_attention_forward(
        params_attn,
        X_c,
        X_q,
        gamma=gamma,
        linear=False,  # softmax only
    )

    logits = out[:, 0, :]
    return logits, attn_w


def make_train_step_single(optimizer, gamma_optimizer):
    """Create training step for single-decoder modes."""
    @jax.jit
    def train_step(params_attn, gamma, opt_state, gamma_opt_state, X, y):
        def loss_fn(p_attn, g):
            logits, _ = forward_single_logits(p_attn, X, gamma=g)
            return ce_with_logits(logits, y)

        loss, (grads_attn, grads_gamma) = jax.value_and_grad(loss_fn, argnums=(0, 1))(params_attn, gamma)

        updates, opt_state = optimizer.update(grads_attn, opt_state, params_attn)
        params_attn = optax.apply_updates(params_attn, updates)

        gamma_updates, gamma_opt_state = gamma_optimizer.update(grads_gamma, gamma_opt_state, gamma)
        gamma = optax.apply_updates(gamma, gamma_updates)
        gamma = jnp.clip(gamma, 0.0, 1.0)

        logits, _ = forward_single_logits(params_attn, X, gamma=gamma)
        acc = acc_from_logits(logits, y)

        return params_attn, gamma, opt_state, gamma_opt_state, loss, acc

    return train_step


def make_eval_step_single():
    """Create eval step for single-decoder modes."""
    @jax.jit
    def eval_step(params_attn, X, y, gamma):
        logits, _ = forward_single_logits(params_attn, X, gamma=gamma)
        loss = ce_with_logits(logits, y)
        acc = acc_from_logits(logits, y)
        return loss, acc

    return eval_step


def train_one_run(
    *,
    key_init,
    init_gamma,
    label_mode,
    ds_train,
    ds_test,
    in_dim,
    L,
    epochs,
    lr,
    weight_decay,
    gamma_lr,
    show_progress=True,
):
    """Train with learnable gamma on ContextIdentification with positional codes.

    Args:
        key_init: JAX PRNG key
        init_gamma: initial value for gamma
        label_mode: 'feature' or 'position'
        ds_train: training dataset
        ds_test: test dataset
        in_dim: input dimension (= 2*L with positional codes)
        L: feature vector length (= number of classes)
        epochs: number of training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        gamma_lr: learning rate for gamma
        show_progress: whether to show progress bar

    Returns:
        history: dict with train/test metrics and gamma evolution
        params_attn: trained parameters
    """
    params_attn = init_gamma_value_attn_params(key_init, in_dim=in_dim, attn_dim=L, val_dim=L)

    gamma_param = jnp.array(init_gamma, dtype=jnp.float32)

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params_attn)

    gamma_optimizer = optax.sgd(learning_rate=gamma_lr)
    gamma_opt_state = gamma_optimizer.init(gamma_param)

    train_step = make_train_step_single(optimizer, gamma_optimizer)
    eval_step = make_eval_step_single()

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "gamma": []
    }

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        desc = f"{label_mode} | init_gamma={init_gamma:.2f}"
        pbar = tqdm(epoch_iter, desc=desc, total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs = [], []
        for X, y in ds_train:
            params_attn, gamma_param, opt_state, gamma_opt_state, loss, acc = train_step(
                params_attn, gamma_param, opt_state, gamma_opt_state, X, y
            )
            train_losses.append(loss)
            train_accs.append(acc)

        tr_loss = float(jnp.mean(jnp.stack(train_losses)))
        tr_acc = float(jnp.mean(jnp.stack(train_accs)))

        test_losses, test_accs = [], []
        for X, y in ds_test:
            loss, acc = eval_step(params_attn, X, y, gamma_param)
            test_losses.append(loss)
            test_accs.append(acc)

        te_loss = float(jnp.mean(jnp.stack(test_losses)))
        te_acc = float(jnp.mean(jnp.stack(test_accs)))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["gamma"].append(float(gamma_param))

        if pbar is not None:
            pbar.set_postfix({
                "loss": f"{tr_loss:.3f}",
                "acc": f"{tr_acc:.3f}",
                "γ": f"{float(gamma_param):.3f}",
            })

    if pbar is not None:
        pbar.close()

    return history, params_attn


def run_sweep(
    *,
    seeds=(0, 1, 2, 3, 4),
    init_gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
    label_modes=("feature", "position"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=False,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    gamma_lr=1e-3,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Run learnable gamma sweep on ContextIdentification with positional codes.

    Returns:
        results: nested dict results[label_mode][init_gamma][seed] -> history
    """
    context_size = L  # num_vectors = L+1, so context_size = L
    in_dim = L + context_size  # = 2*L

    if models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {lm: {float(g): {} for g in init_gammas} for lm in label_modes}

    total_runs = len(seeds) * len(init_gammas) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Learnable γ Pos Codes Sweep", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test = random.split(key, 2)

        for label_mode in label_modes:
            task_train = ContextIdentification(
                key_train,
                num_samples=num_samples,
                num_vectors=L + 1,
                vector_len=L,
                allow_ref_in_context=allow_ref_in_context,
                input_mode="cross",
                label_mode=label_mode,
                positional_codes=True,
            )
            task_test = ContextIdentification(
                key_test,
                num_samples=num_samples,
                num_vectors=L + 1,
                vector_len=L,
                allow_ref_in_context=allow_ref_in_context,
                input_mode="cross",
                label_mode=label_mode,
                positional_codes=True,
            )

            ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
            ds_test = task_test.generate_dataset(batch_size=batch_size, shuffle=False)

            for init_gamma in init_gammas:
                key_init = random.fold_in(key, 10_000 + int(1e6 * float(init_gamma)))
                key_init = random.fold_in(key_init, hash(label_mode) % 10000)

                history, params_attn = train_one_run(
                    key_init=key_init,
                    init_gamma=float(init_gamma),
                    label_mode=label_mode,
                    ds_train=ds_train,
                    ds_test=ds_test,
                    in_dim=in_dim,
                    L=L,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    gamma_lr=gamma_lr,
                    show_progress=show_progress,
                )

                results[label_mode][float(init_gamma)][int(seed)] = history

                if models_dir is not None:
                    gamma_str = f"{init_gamma:.2f}".replace(".", "p")
                    run_name = f"learnable_gamma_pos_codes_{label_mode}_init{gamma_str}_seed{seed}"
                    params_path = models_dir / f"{run_name}_params.pkl"
                    _save_params_pkl(params_attn, params_path)

                if outer is not None:
                    outer.update(1)

    if outer is not None:
        outer.close()

    if save_path is not None:
        save_results_json(results, save_path)

    return results


if __name__ == "__main__":
    results = run_sweep(
        seeds=(0, 1, 2, 3, 4),
        init_gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        label_modes=("feature", "position"),
        num_samples=500,
        L=10,
        allow_ref_in_context=False,
        batch_size=20,
        epochs=5000,
        lr=1e-2,
        weight_decay=1e-3,
        gamma_lr=1e-2,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/context_id_pos_codes_learnable_models/",
        save_path="./data/context_id_pos_codes_learnable_gamma_sweep.json",
    )
