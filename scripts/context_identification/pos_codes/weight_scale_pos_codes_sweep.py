"""Weight-init-scale sweep for ContextIdentification pos-codes comparison.

Sweeps the weight-init scale g of init_gamma_value_attn_params across
{1e-8, ..., 1e-2} for all three conditions (adaptive, adaptive_pos, attn_pos)
and both label modes, with more training epochs at smaller g (competing-
gradient-information hypothesis: does shrinking the positional-code weights'
initial scale fix the bimodal learning trajectories seen in adaptive_pos /
attn_pos in pos_codes_comparison_sweep.py?).

One JSON file is written per scale (not one combined file), because at
g=1e-8 (128,000 epochs) a single scale's full-history JSON would be roughly
25x pos_codes_comparison_sweep.json's 72MB (5,000 epochs) if stored the same
way. Instead, each run's history is reduced to scalar summary stats
immediately after training and the full per-epoch history is discarded.

Designed to run as one SLURM array task per scale value
(--scale-idx == $SLURM_ARRAY_TASK_ID, 0-6); each task runs all
3 conditions x 2 label_modes x N seeds for its assigned scale.

Result format (one file per scale):
  data/weight_scale_pos_codes_sweep/scale_<tag>.json
  results[condition][label_mode][seed] -> summary dict
"""

import argparse
from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    gamma_value_cross_attention_forward,
)
from adaptive_bottleneck.tasks import ContextIdentification
from adaptive_bottleneck.save import save_results_json, _save_params_pkl, _ensure_dir


CONDITIONS = ("adaptive", "adaptive_pos", "attn_pos")

SCALES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

EPOCHS_FOR_SCALE = {
    1e-2: 2_000,
    1e-3: 4_000,
    1e-4: 8_000,
    1e-5: 16_000,
    1e-6: 32_000,
    1e-7: 64_000,
    1e-8: 128_000,
}


def scale_tag(scale):
    """'1e-08' style filename-safe tag."""
    return f"{scale:.0e}"


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
    scale=0.02,
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
        scale: weight-init scale g for init_gamma_value_attn_params
        show_progress: whether to show progress bar

    Returns:
        history: dict with train/test metrics and gamma evolution
        params_attn: trained parameters
    """
    is_adaptive = condition in ("adaptive", "adaptive_pos")

    params_attn = init_gamma_value_attn_params(
        key_init, in_dim=in_dim, attn_dim=L, val_dim=L, scale=scale
    )

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


def summarize_history(history, *, n_final, scale, epochs):
    """Reduce a full per-epoch history to scalars.

    Called immediately after train_one_run returns, so the full history
    list is discarded before the next run starts.
    """
    n = min(n_final, epochs)
    return {
        "final_test_acc": float(np.mean(history["test_acc"][-n:])),
        "final_train_acc": float(np.mean(history["train_acc"][-n:])),
        "final_test_loss": float(np.mean(history["test_loss"][-n:])),
        "final_gamma": float(history["gamma"][-1]),
        "scale": scale,
        "epochs": epochs,
        "n_final": n,
    }


# ============================================================
# Sweep function
# ============================================================

def run_weight_scale_sweep(
    *,
    scale,
    epochs,
    seeds=tuple(range(20)),
    conditions=CONDITIONS,
    label_modes=("feature", "position"),
    num_samples=500,
    L=10,
    allow_ref_in_context=False,
    batch_size=20,
    lr=1e-2,
    weight_decay=1e-3,
    gamma_lr=1e-2,
    init_gamma=0.5,
    n_final=100,
    show_progress=False,
    show_combo_progress=True,
    save_models=False,
    models_dir=None,
    save_path=None,
):
    """Run all conditions x label_modes x seeds for ONE scale value.

    Args:
        scale: weight-init scale g
        epochs: number of training epochs for this scale
        seeds: random seeds
        conditions: which conditions to run
        label_modes: 'feature' and/or 'position'
        num_samples: samples per dataset
        L: feature vector length (= number of classes)
        allow_ref_in_context: whether reference can appear in context
        batch_size: batch size
        lr: learning rate for model parameters
        weight_decay: weight decay
        gamma_lr: learning rate for gamma (adaptive conditions)
        init_gamma: initial gamma (adaptive conditions)
        n_final: number of final epochs to average over per seed
        show_progress: per-run progress bars
        show_combo_progress: overall progress bar
        save_models: whether to save trained params
        models_dir: directory to save model parameters (required if save_models)
        save_path: path to save results JSON

    Returns:
        results[condition][label_mode][seed] -> summary dict
    """
    if save_models and models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {cond: {lm: {} for lm in label_modes} for cond in conditions}

    total_runs = len(seeds) * len(conditions) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc=f"Weight scale g={scale:g}", leave=True)

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
                    scale=scale,
                    show_progress=show_progress,
                )

                results[condition][label_mode][int(seed)] = summarize_history(
                    history, n_final=n_final, scale=scale, epochs=epochs
                )

                if save_models and models_dir is not None:
                    run_name = f"pos_codes_{condition}_{label_mode}_seed{seed}"
                    _save_params_pkl(params_attn, models_dir / f"{run_name}_params.pkl")

                if outer is not None:
                    outer.update(1)

    if outer is not None:
        outer.close()

    if save_path is not None:
        save_results_json(results, save_path)

    return results


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scale-idx", type=int, default=None, choices=range(len(SCALES)),
        help="Index into SCALES (0=1e-8 .. 6=1e-2). Typically $SLURM_ARRAY_TASK_ID. "
             "If omitted, all 7 scales run sequentially in-process "
             "(local smoke-test use only -- do not use this for the full HPC sweep).",
    )
    p.add_argument("--epochs-override", type=int, default=None,
                    help="Override the scale->epoch table (for smoke tests).")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    p.add_argument("--label-modes", type=str, default="feature,position")
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--L", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--gamma-lr", type=float, default=1e-2)
    p.add_argument("--init-gamma", type=float, default=0.5)
    p.add_argument("--n-final", type=int, default=100)
    p.add_argument("--data-dir", type=str, default="./data/weight_scale_pos_codes_sweep")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--progress", action="store_true", help="show per-run tqdm bars")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    conditions = tuple(args.conditions.split(","))
    label_modes = tuple(args.label_modes.split(","))
    scale_indices = [args.scale_idx] if args.scale_idx is not None else list(range(len(SCALES)))

    for idx in scale_indices:
        scale = SCALES[idx]
        epochs = args.epochs_override or EPOCHS_FOR_SCALE[scale]
        tag = scale_tag(scale)
        data_dir = Path(args.data_dir)
        models_dir = data_dir / "models" / f"scale_{tag}" if args.save_models else None

        run_weight_scale_sweep(
            scale=scale,
            epochs=epochs,
            seeds=tuple(range(args.n_seeds)),
            conditions=conditions,
            label_modes=label_modes,
            num_samples=args.num_samples,
            L=args.L,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gamma_lr=args.gamma_lr,
            init_gamma=args.init_gamma,
            n_final=args.n_final,
            show_progress=args.progress,
            show_combo_progress=True,
            save_models=args.save_models,
            models_dir=models_dir,
            save_path=str(data_dir / f"scale_{tag}.json"),
        )
