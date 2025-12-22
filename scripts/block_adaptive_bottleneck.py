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
from adaptive_bottleneck.tasks import BlockComparison
from adaptive_bottleneck.save import save_results_json

## Accuracy and loss functions
def bce_with_logits(logits, y):
    """
    y: (B,) in {0,1}
    """
    # optax.sigmoid_binary_cross_entropy expects labels in [0,1]
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()


def accuracy_from_logits(logits, y):
    preds = (nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
    return (preds == y).mean()


## Training functions
def make_train_step(optimizer, gamma,):
    @jax.jit
    def train_step(params, opt_state, X, y):
        params_attn, params_clf = params["attn"], params["clf"]

        def loss_fn(p_attn, p_clf):
            logits, _ = forward_logits(p_attn, p_clf, X, gamma=gamma,)
            return bce_with_logits(logits, y)

        grads_attn, grads_clf = jax.grad(loss_fn, argnums=(0, 1))(params_attn, params_clf)
        grads = {"attn": grads_attn, "clf": grads_clf}

        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # metrics
        logits, _ = forward_logits(params["attn"], params["clf"], X, gamma=gamma,)
        loss = bce_with_logits(logits, y)
        acc = accuracy_from_logits(logits, y)
        return params, opt_state, loss, acc

    return train_step


@jax.jit
def eval_step(params, X, y, gamma,):
    logits, _ = forward_logits(params["attn"], params["clf"], X, gamma=gamma,)
    return bce_with_logits(logits, y), accuracy_from_logits(logits, y)


def train_one_gamma(
    *,
    key_init,
    gamma,
    ds_train,
    ds_test,
    num_vectors_and_vector_len,
    epochs,
    lr,
    weight_decay,
    show_progress=True,
):
    """Train the adaptive bottleneck model for a single gamma.

    Returns:
        history (dict): keys train_loss/train_acc/test_loss/test_acc, each a list length `epochs`.
        params (dict): trained params with keys {"attn", "clf"}.
    """
    # init params
    k_attn, k_clf = random.split(key_init)
    params_attn = init_gamma_value_attn_params(
        k_attn,
        in_dim=num_vectors_and_vector_len,
        attn_dim=num_vectors_and_vector_len,
        val_dim=num_vectors_and_vector_len,
    )

    params_clf = init_classifier_params(
        k_clf,
        val_dim=num_vectors_and_vector_len,
        seq_len=num_vectors_and_vector_len,
    )
    params = {"attn": params_attn, "clf": params_clf}

    # optimizer
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    gamma_jax = jnp.array(gamma, dtype=jnp.float32)
    train_step = make_train_step(optimizer, gamma=gamma_jax)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"gamma={gamma:.2f}", total=epochs, leave=True)
        epoch_iter = pbar

    for epoch in epoch_iter:
        # ---- Train epoch ----
        train_losses = []
        train_accs = []
        for X_np, y_np in ds_train.as_numpy_iterator():
            X = jnp.asarray(X_np, dtype=jnp.float32)
            y = jnp.asarray(y_np, dtype=jnp.float32).reshape(-1)  # (B,)
            params, opt_state, loss, acc = train_step(params, opt_state, X, y)
            train_losses.append(loss)
            train_accs.append(acc)

        # ---- Eval epoch ----
        test_losses = []
        test_accs = []
        for X_np, y_np in ds_test.as_numpy_iterator():
            X = jnp.asarray(X_np, dtype=jnp.float32)
            y = jnp.asarray(y_np, dtype=jnp.float32).reshape(-1)
            loss, acc = eval_step(params, X, y, gamma=gamma_jax)
            test_losses.append(loss)
            test_accs.append(acc)

        # aggregate
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

    return history, params


## Main function
def run_gamma_sweep(
    seed=0,
    gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
    num_samples=2000,
    num_vectors_and_vector_len=2,
    batch_size=64,
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    show_progress=True,
    save_path=None,
):
    """Run a sweep over gamma values and (optionally) save results.

    Args:
        show_progress: if True, show a per-gamma tqdm bar with train/test metrics.
        save_path: if provided, save JSON results to this path.

    Returns:
        results (dict): mapping gamma -> history dict.
    """
    # Create task (train/test separate keys for clean eval)
    key = random.PRNGKey(seed)
    key_train, key_test, key_init = random.split(key, 3)

    task_train = BlockComparison(
        key_train,
        num_samples=num_samples,
        num_vectors_and_vector_len=num_vectors_and_vector_len,
    )
    ds_train = task_train.generate_tf_dataset(batch_size=batch_size, shuffle=True)

    task_test = BlockComparison(
        key_test,
        num_samples=num_samples,
        num_vectors_and_vector_len=num_vectors_and_vector_len,
    )
    ds_test = task_test.generate_tf_dataset(batch_size=batch_size, shuffle=False)

    results = {}

    for gamma in gammas:
        history, _ = train_one_gamma(
            key_init=key_init,
            gamma=float(gamma),
            ds_train=ds_train,
            ds_test=ds_test,
            num_vectors_and_vector_len=num_vectors_and_vector_len,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            show_progress=show_progress,
        )
        results[float(gamma)] = history

        # update init key so each gamma run gets a fresh init
        key_init = random.fold_in(key_init, int(1e6 * float(gamma)) + 1)

    if save_path is not None:
        save_results_json(results, save_path)

    return results


if __name__ == "__main__":
    results = run_gamma_sweep(
        seed=0,
        gammas=(0.0, 0.25, 0.5, 0.75, 1.0),
        num_samples=2000,
        num_vectors_and_vector_len=2,
        batch_size=64,
        epochs=200,
        lr=1e-3,
        weight_decay=1e-4,
        show_progress=True,
        save_path="./data/block_comparison_gamma_sweep.json",
    )