"""Fixed gamma sweep for the XOR task.

Sweeps gamma across [0, 1] for both linear and softmax attention variants,
recording train/test accuracy across multiple seeds.

Results structure: results[attn_kind][gamma][seed] -> history
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
    forward_logits,
)
from adaptive_bottleneck.tasks import XOR
from adaptive_bottleneck.save import save_results_json, _save_params_pkl


def bce_with_logits(logits, y):
    """Binary cross-entropy from logits. y in {0.0, 1.0}."""
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()


def accuracy_from_logits(logits, y):
    preds = (nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
    return (preds == y).mean()


def make_train_step(optimizer, gamma, linear):
    @jax.jit
    def train_step(params, opt_state, X, y):
        def loss_fn(p):
            logits, _ = forward_logits(p["attn"], p["clf"], X, gamma=gamma, linear=linear)
            return bce_with_logits(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)

        logits, _ = forward_logits(params2["attn"], params2["clf"], X, gamma=gamma, linear=linear)
        acc = accuracy_from_logits(logits, y)
        return params2, opt_state2, loss, acc

    return train_step


def make_eval_step(gamma, linear):
    @jax.jit
    def eval_step(params, X, y):
        logits, _ = forward_logits(params["attn"], params["clf"], X, gamma=gamma, linear=linear)
        return bce_with_logits(logits, y), accuracy_from_logits(logits, y)

    return eval_step


def train_one_run(
    *,
    key_init,
    gamma,
    linear,
    ds_train,
    ds_test,
    epochs,
    lr,
    weight_decay,
    show_progress=True,
):
    """Train the adaptive bottleneck on XOR for one (gamma, seed) run.

    Returns:
        history: dict with train_loss/train_acc/test_loss/test_acc lists.
        params: trained parameters {"attn", "clf"}.
    """
    k_attn, k_clf = random.split(key_init)
    params_attn = init_gamma_value_attn_params(k_attn, in_dim=2, attn_dim=2, val_dim=2)
    params_clf = init_classifier_params(k_clf, val_dim=2, seq_len=2)
    params = {"attn": params_attn, "clf": params_clf}

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    gamma_jax = jnp.array(gamma, dtype=jnp.float32)
    train_step = make_train_step(optimizer, gamma=gamma_jax, linear=linear)
    eval_step = make_eval_step(gamma=gamma_jax, linear=linear)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"linear={linear} | gamma={gamma:.2f}", total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs = [], []
        for X, y in ds_train:
            params, opt_state, loss, acc = train_step(params, opt_state, X, y)
            train_losses.append(loss)
            train_accs.append(acc)

        test_losses, test_accs = [], []
        for X, y in ds_test:
            loss, acc = eval_step(params, X, y)
            test_losses.append(loss)
            test_accs.append(acc)

        tr_loss = float(jnp.mean(jnp.stack(train_losses)))
        tr_acc = float(jnp.mean(jnp.stack(train_accs)))
        te_loss = float(jnp.mean(jnp.stack(test_losses)))
        te_acc = float(jnp.mean(jnp.stack(test_accs)))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if pbar is not None:
            pbar.set_postfix({
                "tr_loss": f"{tr_loss:.4f}",
                "tr_acc": f"{tr_acc:.3f}",
                "te_loss": f"{te_loss:.4f}",
                "te_acc": f"{te_acc:.3f}",
            })

    if pbar is not None:
        pbar.close()

    return history, params


def run_sweep(
    *,
    seeds=(0, 1, 2, 3, 4),
    gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
    attn_kinds=("linear", "softmax"),
    num_samples=2000,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Full gamma sweep for XOR.

    Returns:
        results[attn_kind][gamma][seed] -> history
    """
    if models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

    results = {ak: {float(g): {} for g in gammas} for ak in attn_kinds}

    total_runs = len(seeds) * len(gammas) * len(attn_kinds)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="XOR Sweep", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test = random.split(key, 2)

        task_train = XOR(key_train, num_samples=num_samples)
        task_test = XOR(key_test, num_samples=num_samples)

        ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
        ds_test = task_test.generate_dataset(batch_size=batch_size, shuffle=False)

        for attn_kind in attn_kinds:
            linear = (attn_kind == "linear")
            for gamma in gammas:
                key_init = random.fold_in(key, 10_000 + int(1e6 * float(gamma)))
                key_init = random.fold_in(key_init, 789 if linear else 987)

                history, params = train_one_run(
                    key_init=key_init,
                    gamma=float(gamma),
                    linear=linear,
                    ds_train=ds_train,
                    ds_test=ds_test,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    show_progress=show_progress,
                )

                results[attn_kind][float(gamma)][int(seed)] = history

                if models_dir is not None:
                    gamma_str = f"{gamma:.1f}"
                    run_name = f"xor_gamma{gamma_str}_{attn_kind}_seed{seed}"
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
        seeds=(0, 1, 2, 3, 4),
        gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        attn_kinds=("linear", "softmax"),
        num_samples=500,
        batch_size=20,
        epochs=2500,
        lr=1e-2,
        weight_decay=1e-3,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/xor_models/",
        save_path="./data/xor_gamma_sweep.json",
    )
