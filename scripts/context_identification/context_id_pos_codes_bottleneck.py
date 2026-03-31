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
from adaptive_bottleneck.save import save_results_json, _save_params_pkl


def ce_with_logits(logits, y):
    """Cross-entropy for integer labels.

    Args:
        logits: (B, L)
        y: (B,) int
    """
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def acc_from_logits(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return (preds == y).mean()


def forward_task_logits(params_attn, X, gamma, linear):
    """Compute logits for ContextIdentification with positional codes.

    X: (B, L+1, 2L) — X[:, 0, :] is the reference (zero-padded positional part),
                       X[:, 1:, :] is the context (features + one-hot positional codes).

    Returns logits of shape (B, val_dim) for CE over L classes.
    """
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]

    out, attn_w = gamma_value_cross_attention_forward(
        params_attn,
        X_c,
        X_q,
        gamma=gamma,
        linear=linear,
    )

    logits = out[:, 0, :]
    return logits, attn_w


def make_train_step(optimizer, gamma, linear):
    @jax.jit
    def train_step(params, opt_state, X, y):
        def loss_fn(p):
            logits, _ = forward_task_logits(p, X, gamma=gamma, linear=linear)
            return ce_with_logits(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)

        logits, _ = forward_task_logits(params2, X, gamma=gamma, linear=linear)
        acc = acc_from_logits(logits, y)
        return params2, opt_state2, loss, acc

    return train_step


def make_eval_step(gamma, linear):
    @jax.jit
    def eval_step(params, X, y):
        logits, _ = forward_task_logits(params, X, gamma=gamma, linear=linear)
        return ce_with_logits(logits, y), acc_from_logits(logits, y)
    return eval_step


def train_one_run(
    *,
    key_init,
    gamma,
    linear,
    label_mode,
    ds_train,
    ds_test,
    in_dim,
    L,
    epochs,
    lr,
    weight_decay,
    show_progress=True,
):
    """Train adaptive bottleneck attention on ContextIdentification (with positional codes)
    for one run.

    Returns:
        history: dict with train_loss/train_acc/test_loss/test_acc lists.
        params_attn: trained attention params.
    """
    params_attn = init_gamma_value_attn_params(
        key_init,
        in_dim=in_dim,
        attn_dim=L,
        val_dim=L,
    )

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params_attn)

    gamma_jax = jnp.array(gamma, dtype=jnp.float32)
    train_step = make_train_step(optimizer, gamma=gamma_jax, linear=linear)
    eval_step = make_eval_step(gamma=gamma_jax, linear=linear)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"{label_mode} | linear={linear} | gamma={gamma:.2f}", total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs = [], []
        for X, y in ds_train:
            params_attn, opt_state, loss, acc = train_step(params_attn, opt_state, X, y)
            train_losses.append(loss)
            train_accs.append(acc)

        test_losses, test_accs = [], []
        for X, y in ds_test:
            loss, acc = eval_step(params_attn, X, y)
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
            pbar.set_postfix(
                {
                    "tr_loss": f"{tr_loss:.4f}",
                    "tr_acc": f"{tr_acc:.3f}",
                    "te_loss": f"{te_loss:.4f}",
                    "te_acc": f"{te_acc:.3f}",
                }
            )

    if pbar is not None:
        pbar.close()

    return history, params_attn


def run_sweep(
    *,
    seeds=(0, 1, 2, 3, 4),
    gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
    label_modes=("feature", "position"),
    attn_kinds=("linear", "softmax"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=True,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Full sweep for ContextIdentification with positional codes.

    Runs all combinations of (seed, gamma, label_mode, attn_kind) and returns a nested
    dict suitable for JSON saving.

    Returns:
        results[label_mode][attn_kind][gamma][seed] -> history
    """
    # With positional codes: in_dim = vector_len + context_size = L + L = 2*L
    context_size = L  # num_vectors = L+1, so context_size = L
    in_dim = L + context_size  # = 2*L

    if models_dir is not None:
        models_dir = Path(models_dir)
    results = {lm: {ak: {float(g): {} for g in gammas} for ak in attn_kinds} for lm in label_modes}

    total_runs = len(seeds) * len(gammas) * len(label_modes) * len(attn_kinds)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Sweep", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test = random.split(key, 2)

        task_train = ContextIdentification(
            key_train,
            num_samples=num_samples,
            num_vectors=L + 1,
            vector_len=L,
            allow_ref_in_context=allow_ref_in_context,
            input_mode="cross",
            label_mode="feature",  # overwritten per label_mode
            positional_codes=True,
        )
        task_test = ContextIdentification(
            key_test,
            num_samples=num_samples,
            num_vectors=L + 1,
            vector_len=L,
            allow_ref_in_context=allow_ref_in_context,
            input_mode="cross",
            label_mode="feature",
            positional_codes=True,
        )

        for label_mode in label_modes:
            task_train.label_mode = label_mode
            task_test.label_mode = label_mode

            ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
            ds_test = task_test.generate_dataset(batch_size=batch_size, shuffle=False)

            for attn_kind in attn_kinds:
                linear = True if attn_kind == "linear" else False
                for gamma in gammas:
                    key_init = random.fold_in(key, 10_000 + int(1e6 * float(gamma)))
                    key_init = random.fold_in(key_init, 123 if label_mode == "feature" else 456)
                    key_init = random.fold_in(key_init, 789 if attn_kind == "linear" else 987)

                    history, params_attn = train_one_run(
                        key_init=key_init,
                        gamma=float(gamma),
                        linear=linear,
                        label_mode=label_mode,
                        ds_train=ds_train,
                        ds_test=ds_test,
                        in_dim=in_dim,
                        L=L,
                        epochs=epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        show_progress=show_progress,
                    )

                    results[label_mode][attn_kind][float(gamma)][int(seed)] = history

                    if models_dir is not None:
                        gamma_str = f"{gamma:.1f}"
                        run_name = f"context_id_pos_codes_{label_mode}_gamma{gamma_str}_{attn_kind}_seed{seed}"
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
        gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        label_modes=("feature", "position"),
        attn_kinds=("softmax"),
        num_samples=500,
        L=10,
        allow_ref_in_context=False,
        batch_size=20,
        epochs=2500,
        lr=1e-2,
        weight_decay=1e-3,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/context_id_pos_codes_models/",
        save_path="./data/context_id_pos_codes_gamma_sweep.json",
    )
