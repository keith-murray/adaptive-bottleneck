"""Lagrangian-optimal gamma sweep for ContextIdentification.

Implements the closed-form gamma update derived in
docs/optimization/lagrangian_opt.tex exactly as written.

The model interpolates between self-attention A (γ=0) and relational
bottleneck R (γ=1):

    f(X, γ) = (1-γ) A(W,X) + γ R(W,X)

Because the model is linear in γ, A and R are the endpoints of the
interpolation and are computed as separate components.

Loss: MSE with one-hot targets, L = (1/2) mean((f - Y)²), matching the
document's regression objective exactly.

The four γ-independent gradient quantities are computed via vjp:

    α₀ = ⟨δ_A · J_A⟩    α₁ = ⟨(δ_A-δ_R) · J_A⟩
    ρ₀ = ⟨δ_A · J_R⟩    ρ₁ = ⟨(δ_A-δ_R) · J_R⟩

where δ_A = Y - A, δ_R = Y - R, and J_A, J_R are the Jacobians of A and R
w.r.t. the shared parameters W.

The constrained optimal gamma is (document eq.):

    γ* = (α₀ + α₁ - ρ₀) / (2(α₁ - ρ₁))    if γ* ∈ [0,1] and is a maximum
    γ* = 0                                    if Q(0)² ≥ Q(1)²
    γ* = 1                                    otherwise

Since α₀, α₁, ρ₀, ρ₁ are pytrees (same shape as W), the interior
formula is evaluated by projecting onto α₀: e.g. scalar_α₁ = ⟨α₁_vec, α₀_vec⟩.
The boundary condition compares actual gradient norms ‖Q(0)_vec‖² vs
‖Q(1)_vec‖² to avoid the directional bias of projecting boundary terms.

The gradient at optimal γ follows analytically (no extra forward pass):

    ∇_W L(γ*) = -Q(γ*)_vec
              = -(α₀_vec + (ρ₀_vec - α₀_vec - α₁_vec)γ* + (α₁_vec - ρ₁_vec)γ*²)

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


# ============================================================
# Loss and metrics
# ============================================================

def mse_loss(logits, y):
    """MSE loss with one-hot targets: L = (1/2) mean((logits - one_hot(y))²).

    The factor of 1/2 matches the document's L = (1/2)(Y - f(X))².
    """
    y_onehot = jax.nn.one_hot(y, logits.shape[-1])
    return 0.5 * jnp.mean((logits - y_onehot) ** 2)


def acc_from_logits(logits, y):
    return (jnp.argmax(logits, axis=-1) == y).mean()


# ============================================================
# Pytree helpers
# ============================================================

def _pytree_dot(t1, t2):
    """Dot product of two pytrees: sum of element-wise products."""
    leaves1 = jax.tree_util.tree_leaves(t1)
    leaves2 = jax.tree_util.tree_leaves(t2)
    return sum(jnp.sum(a * b) for a, b in zip(leaves1, leaves2))


def _pytree_lc(pairs):
    """Linear combination of pytrees: sum of (scalar * pytree) pairs."""
    result = jax.tree.map(lambda x: pairs[0][0] * x, pairs[0][1])
    for coeff, tree in pairs[1:]:
        result = jax.tree.map(lambda r, x: r + coeff * x, result, tree)
    return result


# ============================================================
# Lagrangian optimal gamma
# ============================================================

def _optimal_gamma_lagrange(alpha_0, alpha_1, rho_0, rho_1, norm_sq_Q1):
    """Document's closed-form γ* from scalar projections.

    Args:
        alpha_0:    ⟨α₀_vec, α₀_vec⟩ = ‖α₀_vec‖² = ‖Q(0)_vec‖²  (always ≥ 0)
        alpha_1:    ⟨α₁_vec, α₀_vec⟩
        rho_0:      ⟨ρ₀_vec, α₀_vec⟩
        rho_1:      ⟨ρ₁_vec, α₀_vec⟩
        norm_sq_Q1: ‖(ρ₀_vec - ρ₁_vec)‖² = ‖Q(1)_vec‖²

    Returns:
        γ* ∈ [0,1] (JAX scalar)
    """
    # Interior critical point: γ* = (α₀ + α₁ - ρ₀) / (2(α₁ - ρ₁))
    denom = 2.0 * (alpha_1 - rho_1)
    gamma_star = (alpha_0 + alpha_1 - rho_0) / (denom + 1e-10)

    # Q evaluated at γ*: Q(γ) = α₀ + (ρ₀-α₀-α₁)γ + (α₁-ρ₁)γ²
    Q_star = (alpha_0
              + (rho_0 - alpha_0 - alpha_1) * gamma_star
              + (alpha_1 - rho_1) * gamma_star ** 2)

    # γ* is a maximum when (α₁-ρ₁)·Q(γ*) < 0 (document second-derivative test)
    is_maximum = ((alpha_1 - rho_1) * Q_star < 0)
    in_range = (gamma_star >= 0.0) & (gamma_star <= 1.0)

    # Boundary: compare actual gradient norms ‖Q(0)‖² vs ‖Q(1)‖².
    # Q(0)_vec = α₀_vec  →  ‖Q(0)‖² = alpha_0
    # Q(1)_vec = ρ₀_vec - ρ₁_vec  →  ‖Q(1)‖² = norm_sq_Q1
    # Use γ=0 if the attention gradient is larger; γ=1 if the relational gradient is larger.
    gamma_boundary = jnp.where(alpha_0 >= norm_sq_Q1, 0.0, 1.0)

    return jnp.where(is_maximum & in_range, gamma_star, gamma_boundary)


# ============================================================
# Separate A / R forward passes
# ============================================================

def forward_A(params_attn, X):
    """Self-attention output A(W,X) at γ=0.

    Args:
        params_attn: attention parameters
        X: (B, L+1, L) input

    Returns:
        logits: (B, L)
    """
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]
    out, _ = gamma_value_cross_attention_forward(
        params_attn, X_c, X_q, gamma=0.0, linear=False
    )
    return out[:, 0, :]  # (B, L)


def forward_R(params_attn, X):
    """Relational bottleneck output R(W,X) at γ=1.

    Args:
        params_attn: attention parameters
        X: (B, L+1, L) input

    Returns:
        logits: (B, L)
    """
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]
    out, _ = gamma_value_cross_attention_forward(
        params_attn, X_c, X_q, gamma=1.0, linear=False
    )
    return out[:, 0, :]  # (B, L)


# ============================================================
# Training step
# ============================================================

def make_train_step(optimizer):
    """Create training step using the Lagrangian closed-form gamma.

    At each step:
      1. Compute A = forward_A(params, X) and R = forward_R(params, X)
         via two vjp calls that simultaneously set up the backward passes.
      2. Compute δ_A = Y - A, δ_R = Y - R.
      3. Use the vjp functions to compute α₀, α₁, ρ₀, ρ₁ as pytrees.
      4. Project onto α₀ to get the four scalars needed for γ*.
      5. Compute γ* via the document's formula.
      6. Compute the gradient analytically: ∇L(γ*) = -Q(γ*)_vec.
      7. Update params with the optimizer.
    """
    @jax.jit
    def train_step(params_attn, opt_state, X, y):
        L_dim = X.shape[-1]
        y_onehot = jax.nn.one_hot(y, L_dim)  # (B, L)

        # --- Step 1: separate forward passes with vjp setup ---
        logits_A, vjp_A_fn = jax.vjp(forward_A, params_attn, X)
        logits_R, vjp_R_fn = jax.vjp(forward_R, params_attn, X)

        # --- Step 2: residuals ---
        delta_A    = y_onehot - logits_A   # (B, L)
        delta_R    = y_onehot - logits_R   # (B, L)
        delta_diff = delta_A - delta_R     # (B, L)

        # Scale: mean over batch (B) and class (L) dimensions
        scale = 1.0 / (delta_A.shape[0] * delta_A.shape[1])

        # --- Step 3: compute α₀, α₁, ρ₀, ρ₁ as pytrees via vjp ---
        # vjp_fn(co_vector) returns (grad_params, grad_X); we take [0] for params.
        alpha_0_vec = vjp_A_fn(delta_A    * scale)[0]
        alpha_1_vec = vjp_A_fn(delta_diff * scale)[0]
        rho_0_vec   = vjp_R_fn(delta_A    * scale)[0]
        rho_1_vec   = vjp_R_fn(delta_diff * scale)[0]

        # --- Step 4: project onto α₀ to get scalars ---
        alpha_0 = _pytree_dot(alpha_0_vec, alpha_0_vec)  # = ‖α₀_vec‖² = ‖Q(0)‖²
        alpha_1 = _pytree_dot(alpha_1_vec, alpha_0_vec)
        rho_0   = _pytree_dot(rho_0_vec,   alpha_0_vec)
        rho_1   = _pytree_dot(rho_1_vec,   alpha_0_vec)

        # Actual norm of Q(1)_vec = ρ₀_vec - ρ₁_vec for unbiased boundary check
        rho_01_vec = _pytree_lc([(1.0, rho_0_vec), (-1.0, rho_1_vec)])
        norm_sq_Q1 = _pytree_dot(rho_01_vec, rho_01_vec)  # = ‖Q(1)‖²

        # --- Step 5: document's closed-form γ* ---
        gamma = _optimal_gamma_lagrange(alpha_0, alpha_1, rho_0, rho_1, norm_sq_Q1)

        # --- Step 6: gradient at γ* analytically ---
        # ∇_W L(γ) = -Q(γ)_vec
        # Q(γ)_vec = α₀_vec + (ρ₀_vec - α₀_vec - α₁_vec)γ + (α₁_vec - ρ₁_vec)γ²
        grads = _pytree_lc([
            (1.0,         alpha_0_vec),
            (gamma,       _pytree_lc([(1.0, rho_0_vec), (-1.0, alpha_0_vec), (-1.0, alpha_1_vec)])),
            (gamma ** 2,  _pytree_lc([(1.0, alpha_1_vec), (-1.0, rho_1_vec)])),
        ])
        grads = jax.tree.map(lambda x: -x, grads)

        # --- Step 7: optimizer update ---
        updates, opt_state = optimizer.update(grads, opt_state, params_attn)
        params_attn = optax.apply_updates(params_attn, updates)

        # Metrics (evaluated at γ* with updated params)
        logits_f = (1.0 - gamma) * logits_A + gamma * logits_R
        loss = 0.5 * jnp.mean((logits_f - y_onehot) ** 2)
        acc  = acc_from_logits(logits_f, y)

        return params_attn, gamma, opt_state, loss, acc

    return train_step


def make_eval_step():
    """Create eval step."""
    @jax.jit
    def eval_step(params_attn, X, y, gamma):
        logits_A = forward_A(params_attn, X)
        logits_R = forward_R(params_attn, X)
        logits_f = (1.0 - gamma) * logits_A + gamma * logits_R
        loss = mse_loss(logits_f, y)
        acc  = acc_from_logits(logits_f, y)
        return loss, acc

    return eval_step


# ============================================================
# Training function
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
    show_progress=True,
):
    """Train with Lagrangian-optimal gamma for one run.

    Args:
        key_init: JAX PRNG key
        label_mode: 'feature' or 'position'
        ds_train: training dataset
        ds_test: test dataset
        L: vector dimension
        epochs: number of training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        show_progress: whether to show progress bar

    Returns:
        history: dict with train/test metrics and per-epoch mean gamma
        params_attn: trained parameters
    """
    params_attn = init_gamma_value_attn_params(key_init, in_dim=L, attn_dim=L, val_dim=L)

    optimizer  = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state  = optimizer.init(params_attn)

    train_step = make_train_step(optimizer)
    eval_step  = make_eval_step()

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "gamma": [],
    }

    epoch_iter = range(epochs)
    pbar = None
    if show_progress:
        pbar = tqdm(epoch_iter, desc=f"{label_mode} | Lagrange γ", total=epochs, leave=True)
        epoch_iter = pbar

    for _epoch in epoch_iter:
        # Training
        train_losses, train_accs, epoch_gammas = [], [], []
        for X, y in ds_train:
            params_attn, gamma, opt_state, loss, acc = train_step(
                params_attn, opt_state, X, y
            )
            train_losses.append(loss)
            train_accs.append(acc)
            epoch_gammas.append(float(gamma))

        tr_loss    = float(jnp.mean(jnp.stack(train_losses)))
        tr_acc     = float(jnp.mean(jnp.stack(train_accs)))
        mean_gamma = float(sum(epoch_gammas) / len(epoch_gammas))

        # Eval using epoch mean gamma
        test_losses, test_accs = [], []
        for X, y in ds_test:
            loss, acc = eval_step(params_attn, X, y, mean_gamma)
            test_losses.append(loss)
            test_accs.append(acc)

        te_loss = float(jnp.mean(jnp.stack(test_losses)))
        te_acc  = float(jnp.mean(jnp.stack(test_accs)))

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["gamma"].append(mean_gamma)

        if pbar is not None:
            pbar.set_postfix({
                "loss": f"{tr_loss:.3f}",
                "acc":  f"{tr_acc:.3f}",
                "γ":    f"{mean_gamma:.3f}",
            })

    if pbar is not None:
        pbar.close()

    return history, params_attn


# ============================================================
# Sweep function
# ============================================================

def run_sweep(
    *,
    seeds=(0, 1, 2, 3, 4),
    label_modes=("feature", "position"),
    num_samples=2000,
    L=10,
    allow_ref_in_context=False,
    batch_size=64,
    epochs=200,
    lr=1e-3,
    weight_decay=1e-4,
    show_progress=False,
    show_combo_progress=True,
    models_dir=None,
    save_path=None,
):
    """Run Lagrangian gamma sweep across label modes.

    Args:
        seeds: random seeds
        label_modes: 'feature' and/or 'position'
        num_samples: samples per dataset
        L: vector dimension
        allow_ref_in_context: whether reference can appear in context
        batch_size: batch size
        epochs: training epochs
        lr: learning rate for model parameters
        weight_decay: weight decay
        show_progress: per-run progress bars
        show_combo_progress: overall progress bar
        models_dir: directory to save model parameters
        save_path: path to save results JSON

    Returns:
        results[label_mode][seed] -> history
    """
    if models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {lm: {} for lm in label_modes}

    total_runs = len(seeds) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc="Lagrange γ Sweep", leave=True)

    for seed in seeds:
        key = random.PRNGKey(int(seed))
        key_train, key_test, key_init = random.split(key, 3)

        for label_mode in label_modes:
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

            key_run = random.fold_in(key_init, hash(label_mode) % 10000)

            history, params_attn = train_one_run(
                key_init=key_run,
                label_mode=label_mode,
                ds_train=ds_train,
                ds_test=ds_test,
                L=L,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                show_progress=show_progress,
            )

            results[label_mode][int(seed)] = history

            if models_dir is not None:
                run_name = f"lagrange_gamma_{label_mode}_seed{seed}"
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
        seeds=(0, 1, 2, 3, 4),
        label_modes=("feature", "position"),
        num_samples=500,
        L=10,
        allow_ref_in_context=False,
        batch_size=20,
        epochs=2500,
        lr=1e-4,
        weight_decay=0,
        show_progress=False,
        show_combo_progress=True,
        models_dir="./data/lagrange_gamma_models/",
        save_path="./data/lagrange_gamma_sweep.json",
    )
