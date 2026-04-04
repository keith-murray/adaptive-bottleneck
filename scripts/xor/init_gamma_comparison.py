"""Gamma initialization comparison for the XOR task.

ARCA with online SGD for gamma, swept across three initializations:
  - init_gamma = 0.0
  - init_gamma = 0.5
  - init_gamma = 1.0

Two architectural variants:
  - 'indexing': attention output at position 0 used directly as 2-class logits (CE loss)
  - 'decoder':  attention output flattened and passed through a linear classifier (BCE loss)

Result structure: results[arch_kind][init_gamma][seed] -> history
"""

from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import jax.nn as nn
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    init_classifier_params,
    gamma_value_self_attention_forward,
    forward_logits,
)
from adaptive_bottleneck.tasks import XOR
from adaptive_bottleneck.save import save_results_json, _save_params_pkl, _ensure_dir


ARCH_KINDS  = ("indexing", "decoder")
INIT_GAMMAS = (0.0, 0.5, 1.0)


# ============================================================
# Loss and accuracy functions
# ============================================================

def ce_with_logits(logits, y):
    """2-class cross-entropy. y in {0.0, 1.0} (cast to int internally)."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, y.astype(jnp.int32)).mean()


def ce_accuracy(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return (preds == y.astype(jnp.int32)).mean()


def bce_with_logits(logits, y):
    """Binary cross-entropy. y in {0.0, 1.0}."""
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()


def bce_accuracy(logits, y):
    preds = (nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
    return (preds == y).mean()


# ============================================================
# Forward functions
# ============================================================

def forward_indexing(params, X, gamma):
    """Self-attention; output at position 0 → (B, 2) logits."""
    out, attn_w = gamma_value_self_attention_forward(params, X, gamma=gamma, linear=False)
    logits = out[:, 0, :]  # (B, 2)
    return logits, attn_w


def forward_decoder(params, X, gamma):
    """Self-attention + linear classifier → (B,) scalar logits."""
    return forward_logits(params["attn"], params["clf"], X, gamma=gamma, linear=False)


# ============================================================
# Training steps
# ============================================================

def make_train_step(optimizer, gamma_optimizer, arch_kind):
    loss_fn = {"indexing": ce_with_logits, "decoder": bce_with_logits}[arch_kind]
    acc_fn  = {"indexing": ce_accuracy,    "decoder": bce_accuracy}[arch_kind]
    fwd     = {"indexing": forward_indexing, "decoder": forward_decoder}[arch_kind]

    @jax.jit
    def train_step(params, gamma, opt_state, gamma_opt_state, X, y):
        def _loss(p, g):
            logits, _ = fwd(p, X, gamma=g)
            return loss_fn(logits, y)

        loss, (grads, grads_gamma) = jax.value_and_grad(_loss, argnums=(0, 1))(params, gamma)

        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)

        gamma_updates, gamma_opt_state2 = gamma_optimizer.update(grads_gamma, gamma_opt_state, gamma)
        gamma2 = optax.apply_updates(gamma, gamma_updates)
        gamma2 = jnp.clip(gamma2, 0.0, 1.0)

        logits, _ = fwd(params2, X, gamma=gamma2)
        acc = acc_fn(logits, y)
        return params2, gamma2, opt_state2, gamma_opt_state2, loss, acc

    return train_step


def make_eval_step(arch_kind):
    loss_fn = {"indexing": ce_with_logits, "decoder": bce_with_logits}[arch_kind]
    acc_fn  = {"indexing": ce_accuracy,    "decoder": bce_accuracy}[arch_kind]
    fwd     = {"indexing": forward_indexing, "decoder": forward_decoder}[arch_kind]

    @jax.jit
    def eval_step(params, X, y, gamma):
        logits, _ = fwd(params, X, gamma=gamma)
        return loss_fn(logits, y), acc_fn(logits, y)

    return eval_step


# ============================================================
# Training function
# ============================================================

def train_one_run(
    *,
    key_init,
    arch_kind,
    init_gamma,
    ds_train,
    ds_test,
    epochs,
    lr,
    weight_decay,
    gamma_lr,
    show_progress=True,
):
    """Train ARCA on XOR with online gamma optimization for one run.

    Args:
        key_init: JAX PRNG key
        arch_kind: 'indexing' or 'decoder'
        init_gamma: initial value for gamma
        ds_train: training dataset
        ds_test: test dataset
        epochs: number of training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        gamma_lr: learning rate for gamma
        show_progress: whether to show a progress bar

    Returns:
        history: dict with train/test metrics and gamma evolution
        params: trained parameters
    """
    if arch_kind == "indexing":
        params = init_gamma_value_attn_params(key_init, in_dim=2, attn_dim=2, val_dim=2)
    else:
        k_attn, k_clf = random.split(key_init)
        params = {
            "attn": init_gamma_value_attn_params(k_attn, in_dim=2, attn_dim=2, val_dim=2),
            "clf":  init_classifier_params(k_clf, val_dim=2, seq_len=2),
        }

    gamma_param = jnp.array(init_gamma, dtype=jnp.float32)

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    gamma_optimizer = optax.sgd(learning_rate=gamma_lr)
    gamma_opt_state = gamma_optimizer.init(gamma_param)

    train_step = make_train_step(optimizer, gamma_optimizer, arch_kind)
    eval_step  = make_eval_step(arch_kind)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "gamma": []}

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"{arch_kind} | γ₀={init_gamma:.1f}", total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs = [], []
        for X, y in ds_train:
            params, gamma_param, opt_state, gamma_opt_state, loss, acc = train_step(
                params, gamma_param, opt_state, gamma_opt_state, X, y
            )
            train_losses.append(loss)
            train_accs.append(acc)

        test_losses, test_accs = [], []
        for X, y in ds_test:
            loss, acc = eval_step(params, X, y, gamma_param)
            test_losses.append(loss)
            test_accs.append(acc)

        history["train_loss"].append(float(jnp.mean(jnp.stack(train_losses))))
        history["train_acc"].append(float(jnp.mean(jnp.stack(train_accs))))
        history["test_loss"].append(float(jnp.mean(jnp.stack(test_losses))))
        history["test_acc"].append(float(jnp.mean(jnp.stack(test_accs))))
        history["gamma"].append(float(gamma_param))

        if pbar is not None:
            pbar.set_postfix({
                "loss": f"{history['train_loss'][-1]:.4f}",
                "acc":  f"{history['train_acc'][-1]:.3f}",
                "γ":    f"{float(gamma_param):.3f}",
            })

    if pbar is not None:
        pbar.close()

    return history, params


# ============================================================
# Sweep function
# ============================================================

def run_sweep(
    *,
    seeds=tuple(range(20)),
    arch_kinds=ARCH_KINDS,
    init_gammas=INIT_GAMMAS,
    num_samples=500,
    batch_size=20,
    epochs=2500,
    lr=1e-2,
    weight_decay=1e-3,
    gamma_lr=1e-2,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Run gamma initialization comparison for XOR.

    Returns:
        results: nested dict results[arch_kind][init_gamma][seed] -> history
    """
    if models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {ak: {float(g): {} for g in init_gammas} for ak in arch_kinds}

    total_runs = len(seeds) * len(arch_kinds) * len(init_gammas)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="XOR Init γ Comparison", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test = random.split(key, 2)

        task_train = XOR(key_train, num_samples=num_samples)
        task_test  = XOR(key_test,  num_samples=num_samples)

        ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
        ds_test  = task_test.generate_dataset(batch_size=batch_size,  shuffle=False)

        for arch_kind in arch_kinds:
            for init_gamma in init_gammas:
                key_init = random.fold_in(key, hash((arch_kind, init_gamma)) % 10000)

                history, params = train_one_run(
                    key_init=key_init,
                    arch_kind=arch_kind,
                    init_gamma=float(init_gamma),
                    ds_train=ds_train,
                    ds_test=ds_test,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    gamma_lr=gamma_lr,
                    show_progress=show_progress,
                )

                results[arch_kind][float(init_gamma)][int(seed)] = history

                if models_dir is not None:
                    g_str = f"{init_gamma:.1f}".replace(".", "p")
                    run_name = f"xor_init_gamma_{arch_kind}_g{g_str}_seed{seed}"
                    _save_params_pkl(params, models_dir / f"{run_name}_params.pkl")

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
        arch_kinds=ARCH_KINDS,
        init_gammas=INIT_GAMMAS,
        num_samples=500,
        batch_size=20,
        epochs=50,
        lr=1e-2,
        weight_decay=1e-3,
        gamma_lr=1e-2,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/xor_init_gamma_models/",
        save_path="./data/xor_init_gamma_comparison.json",
    )
