"""Gamma sweep for partially-relational task with dual decoders."""

from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    init_dual_decoder_params,
    dual_decoder_cross_attention_forward,
)
from adaptive_bottleneck.tasks import ContextIdentification
from adaptive_bottleneck.save import save_results_json, _save_params_pkl, _ensure_dir


def ce_with_logits(logits, y):
    """Cross-entropy for integer labels."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def acc_from_logits(logits, y):
    preds = jnp.argmax(logits, axis=-1)
    return (preds == y).mean()


def forward_dual_logits(params_attn, params_dec, X, gamma, linear):
    """Compute dual logits for partially-relational task.

    X: (B, L, D) one-hot block. X[:, 0, :] is the reference, X[:, 1:, :] is the context.

    Returns:
        logits_f: (B, num_classes) feature logits
        logits_p: (B, num_classes) position logits
        attn_w: (B, 1, L_c) attention weights
    """
    X_q = X[:, 0:1, :]  # (B, 1, D)
    X_c = X[:, 1:, :]   # (B, L_c, D)

    logits_f, logits_p, attn_w = dual_decoder_cross_attention_forward(
        params_attn,
        params_dec,
        X_c,
        X_q,
        gamma=gamma,
        linear=linear,
    )

    return logits_f, logits_p, attn_w


def make_train_step(optimizer, gamma, linear):
    @jax.jit
    def train_step(params, opt_state, X, y_f, y_p):
        def loss_fn(p):
            logits_f, logits_p, _ = forward_dual_logits(
                p["attn"], p["dec"], X, gamma=gamma, linear=linear
            )
            loss_f = ce_with_logits(logits_f, y_f)
            loss_p = ce_with_logits(logits_p, y_p)
            return loss_f + loss_p

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)

        logits_f, logits_p, _ = forward_dual_logits(
            params2["attn"], params2["dec"], X, gamma=gamma, linear=linear
        )
        acc_f = acc_from_logits(logits_f, y_f)
        acc_p = acc_from_logits(logits_p, y_p)
        return params2, opt_state2, loss, acc_f, acc_p

    return train_step


def make_eval_step(gamma, linear):
    @jax.jit
    def eval_step(params, X, y_f, y_p):
        logits_f, logits_p, _ = forward_dual_logits(
            params["attn"], params["dec"], X, gamma=gamma, linear=linear
        )
        loss_f = ce_with_logits(logits_f, y_f)
        loss_p = ce_with_logits(logits_p, y_p)
        acc_f = acc_from_logits(logits_f, y_f)
        acc_p = acc_from_logits(logits_p, y_p)
        return loss_f + loss_p, acc_f, acc_p

    return eval_step


def train_one_run(
    *,
    key_init,
    gamma,
    linear,
    ds_train,
    ds_test,
    L,
    num_classes_f,
    num_classes_p,
    epochs,
    lr,
    weight_decay,
    use_bias=False,
    show_progress=True,
):
    """Train dual decoder model on partially-relational task for one run.

    Returns:
        history: dict with train/test loss and accuracy for both feature and position.
        params: trained parameters {"attn": ..., "dec": ...}.
    """
    k1, k2 = random.split(key_init, 2)

    params_attn = init_gamma_value_attn_params(
        k1,
        in_dim=L,
        attn_dim=L,
        val_dim=L,
    )
    params_dec = init_dual_decoder_params(
        k2,
        val_dim=L,
        num_classes=num_classes_f,  # feature and position have same num_classes here
        use_bias=use_bias,
    )
    params = {"attn": params_attn, "dec": params_dec}

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    gamma_jax = jnp.array(gamma, dtype=jnp.float32)
    train_step = make_train_step(optimizer, gamma=gamma_jax, linear=linear)
    eval_step = make_eval_step(gamma=gamma_jax, linear=linear)

    history = {
        "train_loss": [],
        "train_acc_f": [],
        "train_acc_p": [],
        "test_loss": [],
        "test_acc_f": [],
        "test_acc_p": [],
    }

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(
            epoch_iter,
            desc=f"both | linear={linear} | gamma={gamma:.2f}",
            total=epochs,
            leave=True,
        )
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs_f, train_accs_p = [], [], []
        for X, y_f, y_p in ds_train:
            params, opt_state, loss, acc_f, acc_p = train_step(
                params, opt_state, X, y_f, y_p
            )
            train_losses.append(loss)
            train_accs_f.append(acc_f)
            train_accs_p.append(acc_p)

        test_losses, test_accs_f, test_accs_p = [], [], []
        for X, y_f, y_p in ds_test:
            loss, acc_f, acc_p = eval_step(params, X, y_f, y_p)
            test_losses.append(loss)
            test_accs_f.append(acc_f)
            test_accs_p.append(acc_p)

        tr_loss = float(jnp.mean(jnp.stack(train_losses)))
        tr_acc_f = float(jnp.mean(jnp.stack(train_accs_f)))
        tr_acc_p = float(jnp.mean(jnp.stack(train_accs_p)))
        te_loss = float(jnp.mean(jnp.stack(test_losses)))
        te_acc_f = float(jnp.mean(jnp.stack(test_accs_f)))
        te_acc_p = float(jnp.mean(jnp.stack(test_accs_p)))

        history["train_loss"].append(tr_loss)
        history["train_acc_f"].append(tr_acc_f)
        history["train_acc_p"].append(tr_acc_p)
        history["test_loss"].append(te_loss)
        history["test_acc_f"].append(te_acc_f)
        history["test_acc_p"].append(te_acc_p)

        if pbar is not None:
            pbar.set_postfix(
                {
                    "loss": f"{tr_loss:.3f}",
                    "f": f"{tr_acc_f:.3f}",
                    "p": f"{tr_acc_p:.3f}",
                    "te_f": f"{te_acc_f:.3f}",
                    "te_p": f"{te_acc_p:.3f}",
                }
            )

    if pbar is not None:
        pbar.close()

    return history, params


def run_sweep(
    *,
    seeds=(0, 1, 2, 3, 4),
    gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
    attn_kinds=("linear", "softmax"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=False,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    use_bias=False,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Full sweep for partially-relational task with dual decoders.

    Returns:
        results[attn_kind][gamma][seed] -> history
    """
    _ensure_dir(models_dir)

    results = {ak: {float(g): {} for g in gammas} for ak in attn_kinds}

    total_runs = len(seeds) * len(gammas) * len(attn_kinds)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Sweep", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test = random.split(key, 2)

        # Build tasks with label_mode="both"
        task_train = ContextIdentification(
            key_train,
            num_samples=num_samples,
            num_vectors=L + 1,
            vector_len=L,
            allow_ref_in_context=allow_ref_in_context,
            input_mode="cross",
            label_mode="both",
        )
        task_test = ContextIdentification(
            key_test,
            num_samples=num_samples,
            num_vectors=L + 1,
            vector_len=L,
            allow_ref_in_context=allow_ref_in_context,
            input_mode="cross",
            label_mode="both",
        )

        ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
        ds_test = task_test.generate_dataset(batch_size=batch_size, shuffle=False)

        for attn_kind in attn_kinds:
            linear = attn_kind == "linear"
            for gamma in gammas:
                key_init = random.fold_in(key, 10_000 + int(1e6 * float(gamma)))
                key_init = random.fold_in(key_init, 789 if linear else 987)

                history, params = train_one_run(
                    key_init=key_init,
                    gamma=float(gamma),
                    linear=linear,
                    ds_train=ds_train,
                    ds_test=ds_test,
                    L=L,
                    num_classes_f=L,  # feature classes = vector_len
                    num_classes_p=L,  # position classes = context_size
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    use_bias=use_bias,
                    show_progress=show_progress,
                )

                results[attn_kind][float(gamma)][int(seed)] = history

                if models_dir is not None:
                    gamma_str = f"{gamma:.2f}".replace(".", "p")
                    run_name = f"partial_rel_gamma{gamma_str}_{attn_kind}_seed{seed}"
                    params_path = models_dir / f"{run_name}_params.pkl"
                    _save_params_pkl(params, params_path)

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
        L=10,
        allow_ref_in_context=False,
        batch_size=20,
        epochs=2500,
        lr=1e-2,
        weight_decay=1e-3,
        use_bias=False,
        show_progress=False,
        show_combo_progress=True,
        models_dir=Path("./data/partial_relational_models/"),
        save_path="./data/partial_relational_sweep.json",
    )
