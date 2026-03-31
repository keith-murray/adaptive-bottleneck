"""Closed-form optimal gamma sweep for ContextIdentification across all label modes.

Instead of learning gamma via backpropagation, this script computes the
analytically optimal gamma at each training step by maximising the speed of
loss reduction ||∇_W L(γ)||², following the Lagrangian derivation in
docs/optimization/lagrangian_opt.tex.

The document derives that for the interpolated model

    f(X) = A(W,X)(1-γ) + R(W,X)γ

the speed of loss reduction is Q(γ)², where Q is a scalar quadratic in γ
with coefficients (α₀, α₁, ρ₀, ρ₁) that depend on the current batch.
The constrained maximum over γ ∈ [0,1] is:
  - Interior:  γ* = (α₀ + α₁ - ρ₀) / 2(α₁ - ρ₁), valid when it is in [0,1]
               and gives a true maximum (second-derivative check).
  - Boundary:  γ=0 if Q(0)²=‖g_A‖² > Q(1)²=‖g_R‖², else γ=1.

Because the document's derivation uses MSE with scalar outputs, we adapt it
to cross-entropy by evaluating the gradient norm squared numerically at
γ ∈ {0, 0.5, 1}, fitting a quadratic to these three values, and finding the
maximum of that quadratic on [0,1].  This recovers the document's boundary
conditions exactly (Q(0)²=‖g_A‖², Q(1)²=‖g_R‖²) and gives a principled
interior estimate without requiring explicit cross-term computation.

Label modes:
- 'feature': predict which feature is closest
- 'position': predict which position contains the closest item
- 'both': predict both feature and position (dual decoder)

Gamma is NOT a persistent parameter — it is recomputed analytically each step.
"""

from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    init_dual_decoder_params,
    gamma_value_cross_attention_forward,
    dual_decoder_cross_attention_forward,
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


def _pytree_norm_sq(tree):
    """Squared L2 norm of a pytree (sum of squared elements)."""
    return jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(x ** 2), tree, 0.0
    )


def _optimal_gamma(h0, h05, h1):
    """Find γ ∈ [0,1] that maximises a quadratic fit to gradient norm squared.

    Fits h(γ) = a*γ² + b*γ + c to the three evaluations h(0), h(0.5), h(1),
    then returns the constrained maximiser on [0,1].

    Args:
        h0:  ‖∇_W L(γ=0.0)‖²  (attention-only gradient norm squared)
        h05: ‖∇_W L(γ=0.5)‖²
        h1:  ‖∇_W L(γ=1.0)‖²  (relational-only gradient norm squared)

    Returns:
        γ ∈ [0,1] (JAX scalar)
    """
    # Quadratic coefficients from Lagrange interpolation at γ=0, 0.5, 1
    c = h0
    a = 2.0 * h0 - 4.0 * h05 + 2.0 * h1
    b = -3.0 * h0 + 4.0 * h05 - h1

    # Interior critical point γ* = -b / (2a), valid maximum when a < 0
    gamma_interior = jnp.clip(-b / (2.0 * a + 1e-10), 0.0, 1.0)
    h_interior = a * gamma_interior ** 2 + b * gamma_interior + c

    # Boundary maximum
    gamma_boundary = jnp.where(h0 >= h1, 0.0, 1.0)
    h_boundary = jnp.maximum(h0, h1)

    # Use interior solution only when quadratic is concave (a < 0) and
    # the interior value beats both boundaries
    return jnp.where(
        (a < 0) & (h_interior >= h_boundary),
        gamma_interior,
        gamma_boundary,
    )


# ============================================================
# Single decoder forward (feature or position mode)
# ============================================================

def forward_single_logits(params_attn, X, gamma):
    """Forward pass for single-output modes (feature or position).

    Args:
        params_attn: attention parameters
        X: (B, L+1, L) input with reference at [:, 0, :] and context at [:, 1:, :]
        gamma: scalar or array, the bottleneck parameter

    Returns:
        logits: (B, L) class logits
        attn_w: (B, 1, L) attention weights
    """
    X_q = X[:, 0:1, :]  # (B, 1, L)
    X_c = X[:, 1:, :]   # (B, L, L)

    out, attn_w = gamma_value_cross_attention_forward(
        params_attn,
        X_c,
        X_q,
        gamma=gamma,
        linear=False,  # softmax only
    )

    logits = out[:, 0, :]  # (B, L)
    return logits, attn_w


# ============================================================
# Dual decoder forward (both mode)
# ============================================================

def forward_dual_logits(params_attn, params_dec, X, gamma, signed):
    """Forward pass for dual-output mode (both).

    Args:
        params_attn: attention parameters
        params_dec: dual decoder parameters
        X: (B, L+1, L) input
        gamma: scalar or array
        signed: bool, whether to use signed interpolation

    Returns:
        logits_f: (B, L) feature logits
        logits_p: (B, L) position logits
        attn_w: (B, 1, L) attention weights
    """
    X_q = X[:, 0:1, :]  # (B, 1, L)
    X_c = X[:, 1:, :]   # (B, L, L)

    logits_f, logits_p, attn_w = dual_decoder_cross_attention_forward(
        params_attn,
        params_dec,
        X_c,
        X_q,
        gamma=gamma,
        linear=False,  # softmax only
        signed=signed,
    )

    return logits_f, logits_p, attn_w


# ============================================================
# Training steps for single decoder
# ============================================================

def make_train_step_single(optimizer):
    """Create training step for single-decoder modes with closed-form gamma.

    Evaluates gradient norm squared at γ ∈ {0, 0.5, 1}, fits a quadratic,
    finds the constrained maximum, then updates model params at that gamma.
    """
    @jax.jit
    def train_step(params_attn, opt_state, X, y):
        def loss_fn(p_attn, g):
            logits, _ = forward_single_logits(p_attn, X, gamma=g)
            return ce_with_logits(logits, y)

        # Evaluate gradient norm squared at three gamma values
        h0  = _pytree_norm_sq(jax.grad(loss_fn, argnums=0)(params_attn, 0.0))
        h05 = _pytree_norm_sq(jax.grad(loss_fn, argnums=0)(params_attn, 0.5))
        h1  = _pytree_norm_sq(jax.grad(loss_fn, argnums=0)(params_attn, 1.0))

        # Closed-form optimal gamma
        gamma = _optimal_gamma(h0, h05, h1)

        # Update params at the optimal gamma
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params_attn, gamma)
        updates, opt_state = optimizer.update(grads, opt_state, params_attn)
        params_attn = optax.apply_updates(params_attn, updates)

        # Metrics
        logits, _ = forward_single_logits(params_attn, X, gamma=gamma)
        acc = acc_from_logits(logits, y)

        return params_attn, gamma, opt_state, loss, acc

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


# ============================================================
# Training steps for dual decoder
# ============================================================

def make_train_step_dual(optimizer, signed):
    """Create training step for dual-decoder mode with closed-form gamma."""
    @jax.jit
    def train_step(params, opt_state, X, y_f, y_p):
        def loss_fn(p, g):
            logits_f, logits_p, _ = forward_dual_logits(
                p["attn"], p["dec"], X, gamma=g, signed=signed
            )
            return ce_with_logits(logits_f, y_f) + ce_with_logits(logits_p, y_p)

        # Evaluate gradient norm squared at three gamma values
        h0  = _pytree_norm_sq(jax.grad(loss_fn, argnums=0)(params, 0.0))
        h05 = _pytree_norm_sq(jax.grad(loss_fn, argnums=0)(params, 0.5))
        h1  = _pytree_norm_sq(jax.grad(loss_fn, argnums=0)(params, 1.0))

        # Closed-form optimal gamma
        gamma = _optimal_gamma(h0, h05, h1)

        # Update params at the optimal gamma
        loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params, gamma)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Metrics
        logits_f, logits_p, _ = forward_dual_logits(
            params["attn"], params["dec"], X, gamma=gamma, signed=signed
        )
        acc_f = acc_from_logits(logits_f, y_f)
        acc_p = acc_from_logits(logits_p, y_p)

        return params, gamma, opt_state, loss, acc_f, acc_p

    return train_step


def make_eval_step_dual(signed):
    """Create eval step for dual-decoder mode."""
    @jax.jit
    def eval_step(params, X, y_f, y_p, gamma):
        logits_f, logits_p, _ = forward_dual_logits(
            params["attn"], params["dec"], X, gamma=gamma, signed=signed
        )
        loss_f = ce_with_logits(logits_f, y_f)
        loss_p = ce_with_logits(logits_p, y_p)
        acc_f = acc_from_logits(logits_f, y_f)
        acc_p = acc_from_logits(logits_p, y_p)
        return loss_f + loss_p, acc_f, acc_p

    return eval_step


# ============================================================
# Unified training function
# ============================================================

def train_one_run(
    *,
    key_init,
    label_mode,
    ds_train,
    ds_test,
    L,
    epochs,
    lr,
    weight_decay,
    signed=False,
    show_progress=True,
):
    """Train with closed-form optimal gamma for one run.

    Args:
        key_init: JAX PRNG key
        label_mode: 'feature', 'position', or 'both'
        ds_train: training dataset
        ds_test: test dataset
        L: dimension (vector length)
        epochs: number of training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        signed: whether to use signed interpolation (for 'both' mode)
        show_progress: whether to show progress bar

    Returns:
        history: dict with train/test metrics and per-epoch mean gamma
        params: trained parameters
    """
    is_dual = (label_mode == "both")

    # Initialize parameters
    if is_dual:
        k1, k2 = random.split(key_init, 2)
        params_attn = init_gamma_value_attn_params(k1, in_dim=L, attn_dim=L, val_dim=L)
        params_dec = init_dual_decoder_params(k2, val_dim=L, num_classes=L, use_bias=True)
        params = {"attn": params_attn, "dec": params_dec}
    else:
        params = init_gamma_value_attn_params(key_init, in_dim=L, attn_dim=L, val_dim=L)

    # Optimizer (model params only — no gamma optimizer)
    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    # Training/eval steps
    if is_dual:
        train_step = make_train_step_dual(optimizer, signed=signed)
        eval_step = make_eval_step_dual(signed=signed)
    else:
        train_step = make_train_step_single(optimizer)
        eval_step = make_eval_step_single()

    # History tracking
    if is_dual:
        history = {
            "train_loss": [], "train_acc_f": [], "train_acc_p": [],
            "test_loss": [], "test_acc_f": [], "test_acc_p": [],
            "gamma": [],
        }
    else:
        history = {
            "train_loss": [], "train_acc": [],
            "test_loss": [], "test_acc": [],
            "gamma": [],
        }

    # Progress bar
    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        desc = f"{label_mode} | closed-form γ"
        pbar = tqdm(epoch_iter, desc=desc, total=epochs, leave=True)
        epoch_iter = pbar

    # Training loop
    for _epoch in epoch_iter:
        if is_dual:
            train_losses, train_accs_f, train_accs_p, epoch_gammas = [], [], [], []
            for X, y_f, y_p in ds_train:
                params, gamma, opt_state, loss, acc_f, acc_p = train_step(
                    params, opt_state, X, y_f, y_p
                )
                train_losses.append(loss)
                train_accs_f.append(acc_f)
                train_accs_p.append(acc_p)
                epoch_gammas.append(float(gamma))

            tr_loss  = float(jnp.mean(jnp.stack(train_losses)))
            tr_acc_f = float(jnp.mean(jnp.stack(train_accs_f)))
            tr_acc_p = float(jnp.mean(jnp.stack(train_accs_p)))
            mean_gamma = float(sum(epoch_gammas) / len(epoch_gammas))

            # Eval using mean gamma from this epoch's training
            test_losses, test_accs_f, test_accs_p = [], [], []
            for X, y_f, y_p in ds_test:
                loss, acc_f, acc_p = eval_step(params, X, y_f, y_p, mean_gamma)
                test_losses.append(loss)
                test_accs_f.append(acc_f)
                test_accs_p.append(acc_p)

            te_loss  = float(jnp.mean(jnp.stack(test_losses)))
            te_acc_f = float(jnp.mean(jnp.stack(test_accs_f)))
            te_acc_p = float(jnp.mean(jnp.stack(test_accs_p)))

            history["train_loss"].append(tr_loss)
            history["train_acc_f"].append(tr_acc_f)
            history["train_acc_p"].append(tr_acc_p)
            history["test_loss"].append(te_loss)
            history["test_acc_f"].append(te_acc_f)
            history["test_acc_p"].append(te_acc_p)

            if pbar is not None:
                pbar.set_postfix({
                    "loss": f"{tr_loss:.3f}",
                    "f": f"{tr_acc_f:.3f}",
                    "p": f"{tr_acc_p:.3f}",
                    "γ": f"{mean_gamma:.3f}",
                })

        else:
            train_losses, train_accs, epoch_gammas = [], [], []
            for X, y in ds_train:
                params, gamma, opt_state, loss, acc = train_step(
                    params, opt_state, X, y
                )
                train_losses.append(loss)
                train_accs.append(acc)
                epoch_gammas.append(float(gamma))

            tr_loss  = float(jnp.mean(jnp.stack(train_losses)))
            tr_acc   = float(jnp.mean(jnp.stack(train_accs)))
            mean_gamma = float(sum(epoch_gammas) / len(epoch_gammas))

            # Eval using mean gamma from this epoch's training
            test_losses, test_accs = [], []
            for X, y in ds_test:
                loss, acc = eval_step(params, X, y, mean_gamma)
                test_losses.append(loss)
                test_accs.append(acc)

            te_loss = float(jnp.mean(jnp.stack(test_losses)))
            te_acc  = float(jnp.mean(jnp.stack(test_accs)))

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["test_loss"].append(te_loss)
            history["test_acc"].append(te_acc)

            if pbar is not None:
                pbar.set_postfix({
                    "loss": f"{tr_loss:.3f}",
                    "acc": f"{tr_acc:.3f}",
                    "γ": f"{mean_gamma:.3f}",
                })

        history["gamma"].append(mean_gamma)

    if pbar is not None:
        pbar.close()

    return history, params


# ============================================================
# Sweep function
# ============================================================

def run_sweep(
    *,
    seeds=(0, 1, 2, 3, 4),
    label_modes=("feature", "position", "both"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=False,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    signed=True,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Run closed-form gamma sweep across all label modes.

    Unlike the learnable/meta gamma sweeps, there is no init_gammas dimension
    because gamma is computed analytically at each step, not initialised.

    Args:
        seeds: random seeds for reproducibility
        label_modes: which label modes to run ('feature', 'position', 'both')
        num_samples: number of samples per dataset
        L: vector dimension
        allow_ref_in_context: whether reference can appear in context
        batch_size: batch size for training
        epochs: number of training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        signed: whether to use signed interpolation (for 'both' mode)
        show_progress: show per-run progress bars
        show_combo_progress: show overall sweep progress
        models_dir: directory to save model parameters
        save_path: path to save results JSON

    Returns:
        results: nested dict results[label_mode][seed] -> history
    """
    if models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {lm: {} for lm in label_modes}

    total_runs = len(seeds) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Closed-form γ Sweep", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test, key_init = random.split(key, 3)

        for label_mode in label_modes:
            # Create datasets
            task_train = ContextIdentification(
                key_train,
                num_samples=num_samples,
                num_vectors=L + 1,
                vector_len=L,
                allow_ref_in_context=allow_ref_in_context,
                input_mode="cross",
                label_mode=label_mode,
            )
            task_test = ContextIdentification(
                key_test,
                num_samples=num_samples,
                num_vectors=L + 1,
                vector_len=L,
                allow_ref_in_context=allow_ref_in_context,
                input_mode="cross",
                label_mode=label_mode,
            )

            ds_train = task_train.generate_dataset(batch_size=batch_size, shuffle=True)
            ds_test  = task_test.generate_dataset(batch_size=batch_size, shuffle=False)

            # Unique init key per (seed, label_mode)
            key_run = random.fold_in(key_init, hash(label_mode) % 10000)

            history, params = train_one_run(
                key_init=key_run,
                label_mode=label_mode,
                ds_train=ds_train,
                ds_test=ds_test,
                L=L,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                signed=signed,
                show_progress=show_progress,
            )

            results[label_mode][int(seed)] = history

            # Save model
            if models_dir is not None:
                signed_str = "signed" if signed else "unsigned"
                run_name = f"closed_form_gamma_{label_mode}_{signed_str}_seed{seed}"
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
        label_modes=("feature", "position", "both"),
        num_samples=500,
        L=10,
        allow_ref_in_context=False,
        batch_size=20,
        epochs=2500,
        lr=1e-2,
        weight_decay=1e-3,
        signed=True,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/closed_form_gamma_models/",
        save_path="./data/closed_form_gamma_sweep.json",
    )
