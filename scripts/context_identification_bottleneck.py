import jax
import jax.random as random
import jax.numpy as jnp
import jax.nn as nn
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    gamma_value_attention_forward,
)
from adaptive_bottleneck.tasks import ContextIdentification
from adaptive_bottleneck.save import save_results_json


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


def forward_task_logits(params_attn, X, gamma, label_mode):
    """Compute logits for ContextIdentification.

    X: (B, L, L) one-hot block. X[:, 0, :] is the reference, X[:, 1:, :] is the context.

    We return logits of shape (B, L) for CE over L classes.

    Notes on the two modes:
      - feature: predict the *feature index* (0..L-1) of the closest context item.
        We use the reference token's output feature vector: out[:, 0, :]  -> (B, L).

      - position: predict the *position in the block* (0..L-1) of the closest context item.
        To emphasize that value-mixing can be detrimental for positional decoding, we
        intentionally use the VALUE output (not the attention distribution): out[:, :, 0] -> (B, L)
        when val_dim==1.

    Args:
        params_attn: gamma-value attention parameters.
        X: (B, L, L)
        gamma: scalar float32
        label_mode: 'feature' or 'position'

    Returns:
        logits: (B, L)
        attn_w: (B, L, L)
    """
    out, attn_w = gamma_value_attention_forward(
        params_attn,
        X,
        gamma=gamma,
    )

    if label_mode == "feature":
        logits = out[:, 0, :]  # (B, L)  requires val_dim == L
    elif label_mode == "position":
        logits = out[:, :, 0]  # (B, L)  typical when val_dim == 1
    else:
        raise ValueError(f"Unknown label_mode={label_mode}")

    return logits, attn_w


def make_train_step(optimizer, gamma, label_mode):
    @jax.jit
    def train_step(params, opt_state, X, y):
        def loss_fn(p):
            logits, _ = forward_task_logits(p, X, gamma=gamma, label_mode=label_mode)
            return ce_with_logits(logits, y)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)

        logits, _ = forward_task_logits(params2, X, gamma=gamma, label_mode=label_mode)
        acc = acc_from_logits(logits, y)
        return params2, opt_state2, loss, acc

    return train_step


def make_eval_step(gamma, label_mode):
    @jax.jit
    def eval_step(params, X, y):
        logits, _ = forward_task_logits(params, X, gamma=gamma, label_mode=label_mode)
        return ce_with_logits(logits, y), acc_from_logits(logits, y)
    return eval_step


def train_one_gamma(
    *,
    key_init,
    gamma,
    label_mode,
    ds_train,
    ds_test,
    L,
    epochs,
    lr,
    weight_decay,
    show_progress=True,
):
    """Train adaptive bottleneck attention on ContextIdentification for one gamma.

    Returns:
        history: dict with train_loss/train_acc/test_loss/test_acc lists.
        params_attn: trained attention params.
    """
    params_attn = init_gamma_value_attn_params(
        key_init,
        in_dim=L,
        attn_dim=L,
        val_dim=L,
    )

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params_attn)

    gamma_jax = jnp.array(gamma, dtype=jnp.float32)
    train_step = make_train_step(optimizer, gamma=gamma_jax, label_mode=label_mode)
    eval_step = make_eval_step(gamma=gamma_jax, label_mode=label_mode)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"{label_mode} | gamma={gamma:.2f}", total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        train_losses, train_accs = [], []
        for X_np, y_np in ds_train.as_numpy_iterator():
            X = jnp.asarray(X_np, dtype=jnp.float32)
            y = jnp.asarray(y_np).astype(jnp.int32).reshape(-1)
            params_attn, opt_state, loss, acc = train_step(params_attn, opt_state, X, y)
            train_losses.append(loss)
            train_accs.append(acc)

        test_losses, test_accs = [], []
        for X_np, y_np in ds_test.as_numpy_iterator():
            X = jnp.asarray(X_np, dtype=jnp.float32)
            y = jnp.asarray(y_np).astype(jnp.int32).reshape(-1)
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


def run_gamma_sweep(
    *,
    seed=0,
    gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
    label_modes=("feature", "position"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=True,
    batch_size=64,
    epochs=100,
    lr=1e-3,
    weight_decay=1e-4,
    show_progress=True,
    save_path=None,
):
    """Sweep gamma for ContextIdentification under multiple label modes.

    Returns:
        results: dict[label_mode][gamma] -> history
    """
    key = random.PRNGKey(seed)
    key_train, key_test, key_init = random.split(key, 3)

    # Train/test tasks (separate RNG streams)
    task_train = ContextIdentification(
        key_train,
        num_samples=num_samples,
        num_vectors_and_vector_len=L,
        allow_ref_in_context=allow_ref_in_context,
        label_mode="feature",  # overwritten per mode in loop (dataset must match)
    )

    task_test = ContextIdentification(
        key_test,
        num_samples=num_samples,
        num_vectors_and_vector_len=L,
        allow_ref_in_context=allow_ref_in_context,
        label_mode="feature",
    )

    results = {}

    for label_mode in label_modes:
        # rebuild datasets per mode so the labels are correct
        task_train.label_mode = label_mode
        task_test.label_mode = label_mode

        ds_train = task_train.generate_tf_dataset(batch_size=batch_size, shuffle=True)
        ds_test = task_test.generate_tf_dataset(batch_size=batch_size, shuffle=False)

        results[label_mode] = {}

        # fresh init stream per mode
        key_mode_init = random.fold_in(key_init, 123 if label_mode == "feature" else 456)

        for gamma in gammas:
            history, _ = train_one_gamma(
                key_init=key_mode_init,
                gamma=float(gamma),
                label_mode=label_mode,
                ds_train=ds_train,
                ds_test=ds_test,
                L=L,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                show_progress=show_progress,
            )
            results[label_mode][float(gamma)] = history

            # change init key per gamma to reduce accidental coupling
            key_mode_init = random.fold_in(key_mode_init, int(1e6 * float(gamma)) + 1)

    if save_path is not None:
        save_results_json(results, save_path)

    return results


if __name__ == "__main__":
    results = run_gamma_sweep(
        seed=0,
        gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        label_modes=("feature", "position"),
        num_samples=250,
        L=10,
        allow_ref_in_context=False,
        batch_size=16,
        epochs=500,
        lr=1e-3,
        weight_decay=1e-4,
        show_progress=True,
        save_path="./data/context_identification_gamma_sweep.json",
    )
