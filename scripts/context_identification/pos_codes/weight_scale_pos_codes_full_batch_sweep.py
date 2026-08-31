"""Weight-init-scale sweep for ContextIdentification pos-codes comparison,
trained with full-batch gradient descent instead of SGD (c.f.
weight_scale_pos_codes_sweep.py and pos_codes_full_batch_comparison.py).

Sweeps the weight-init scale g of init_gamma_value_attn_params across
{1e-8, ..., 1e-2} for all three conditions (adaptive, adaptive_pos, attn_pos)
and both label modes, with more full-batch training steps at smaller g
(competing-gradient-information hypothesis: does shrinking the positional-code
weights' initial scale fix the bimodal learning trajectories seen in
adaptive_pos / attn_pos?) -- now with mini-batch SGD noise removed as a
confound.

Full batch = the exhaustive enumeration of every valid task instance for a
chosen L (see ContextIdentification.generate_all_examples() and
full_batch_combinatorics.py -- L=6 gives 27,684 unique instances, small enough
to keep in memory for the whole sweep). One gradient step sees the entire
population, so there is no train/test split; only final train accuracy/loss
are reported.

The enumerated population is deterministic (no PRNG dependence) and depends
only on (positional_codes, label_mode), not on scale or seed, so it is built
once per (condition-family, label_mode) combo and reused across every seed --
unlike the SGD version, where "seed" controlled both dataset sampling and
parameter initialization, here it controls only parameter initialization.

All N_SEEDS seeds for a given (scale, condition, label_mode) combo are
trained together, batched over an extra leading "seed" axis on params/gamma/
opt_state while X, y stay shared/unbatched -- one jax.jit compile per combo
instead of one per seed, and lets --cpus-per-task on the SLURM array task
actually parallelize the sweep. Two backends implement this, selected via
--backend, both driving the SAME per-seed train_step body
(make_train_step_adaptive/fixed):

  - "vmap" (train_seeds(), default): a single jax.vmap over all N_SEEDS,
    relying on XLA's CPU backend to internally parallelize the batched
    matmuls across whatever threads it has available. Simple, but a local
    smoke test showed no wall-clock improvement over the old sequential
    version at L=6 (tiny per-step matmuls -- plausibly dispatch-bound, not
    compute-bound).
  - "shard_map" (train_seeds_shard_map()): explicitly partitions the batch
    of seeds across jax.device_count() CPU devices via jax.shard_map,
    wrapping the SAME vmapped train_step as the per-device body (vmap
    within a shard, shard_map across shards/devices) -- genuinely separate
    per-device XLA programs rather than relying on intra-op threading of one
    big op. REQUIRES XLA_FLAGS=--xla_force_host_platform_device_count=<N> to
    be set in the environment BEFORE this process starts (JAX's CPU backend
    exposes only 1 device by default, and this must take effect before JAX
    initializes, so it can't be set from inside this script -- set it in the
    SLURM script, or export it before a local run). N_SEEDS must be evenly
    divisible by the device count.

Neither backend's real-world speedup on the actual HPC hardware is verified
yet -- profile both with slurm/weight_scale_pos_codes_full_batch_profile.sh
before picking one for the real sweep.

One JSON file is written per (scale, condition, label_mode) combo, matching
weight_scale_pos_codes_sweep.py's file-per-combo convention, so concurrent
SLURM array tasks sharing a scale never write the same file:
  data/weight_scale_pos_codes_full_batch_sweep/scale_<tag>__<condition>__<label_mode>.json
  results[condition][label_mode][seed] -> summary dict

Designed to run as one SLURM array task per (scale, condition, label_mode)
combo, mirroring slurm/weight_scale_pos_codes_sweep.sh's index->combo mapping.
"""

import argparse
from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import PartitionSpec as P

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    gamma_value_cross_attention_forward,
)
from adaptive_bottleneck.tasks import ContextIdentification
from adaptive_bottleneck.save import save_results_json, _save_params_pkl, _ensure_dir


CONDITIONS = ("adaptive", "adaptive_pos", "attn_pos")

SCALES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)

# Full-batch steps converge much faster per-step than mini-batch SGD epochs
# (Step 2 found ~2000 full-batch steps sufficient at g~=1e-2, L=6), so the base
# step count is smaller than EPOCHS_FOR_SCALE's, but keeps the same "halve g,
# double the budget" relationship in case smaller scales are similarly harder
# to converge under full-batch GD. Revisit after the local smoke test below.
STEPS_FOR_SCALE = {
    1e-2: 1_000,
    1e-3: 2_000,
    1e-4: 4_000,
    1e-5: 8_000,
    1e-6: 16_000,
    1e-7: 32_000,
    1e-8: 64_000,
}


def scale_tag(scale):
    """'1e-08' style filename-safe tag."""
    return f"{scale:.0e}"


def combo_filename(scale, condition, label_mode):
    """'scale_1e-08__adaptive__feature.json' style per-combo filename.

    One file per (scale, condition, label_mode) so array tasks that share a
    scale but run different conditions/label_modes never write the same path.
    """
    return f"scale_{scale_tag(scale)}__{condition}__{label_mode}.json"


def ce_with_logits(logits, y):
    """Cross-entropy for integer labels."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def acc_from_logits(logits, y):
    """Accuracy from logits."""
    return (jnp.argmax(logits, axis=-1) == y).mean()


def _broadcast_unbatched_leaves(pytree, n_seeds):
    """Broadcast any scalar (ndim==0) leaf in `pytree` to shape (n_seeds,).

    optax state leaves that don't depend on the mapped input (e.g. adamw's
    integer step `count = jnp.zeros([], jnp.int32)`, independent of params)
    come back from jax.vmap(optimizer.init)(...) still unbatched (shape ()),
    since vmap only adds a batch axis to outputs that actually depend on the
    batched input. Every leaf this codebase's params/mu/nu ever produces has
    ndim>=1 (they mirror W_q/W_k/W_v, always matrices), so ndim==0 is an
    unambiguous signal this is one of those data-independent leaves -- force
    it to a uniform (n_seeds,) leading axis so it can be sliced/mapped like
    every other per-seed leaf in train_step's vmap.
    """
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, (n_seeds,)) if leaf.ndim == 0 else leaf,
        pytree,
    )


def forward_logits(params_attn, X, gamma):
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]
    out, attn_w = gamma_value_cross_attention_forward(
        params_attn, X_c, X_q, gamma=gamma, linear=False,
    )
    return out[:, 0, :], attn_w


# ============================================================
# Training steps (full-batch AND batched over seeds: called once per step on
# the entire population, jax.vmap'd over an extra leading "seed" axis on
# params_attn/gamma/opt_state so all seeds for a (condition, label_mode)
# combo run as one computation instead of one per seed. X, y (the shared
# enumerated population) are NOT batched -- every seed trains on the
# identical population, so they use in_axes=None.
#
# These return the RAW vmapped function, deliberately not jax.jit-wrapped --
# the two backends below compose it differently:
#   - train_seeds() wraps it in a plain jax.jit (single device).
#   - train_seeds_shard_map() reuses it UNCHANGED as the per-device body of
#     a jax.shard_map (vmap over the local per-device seed shard, shard_map
#     across devices), then jax.jit's the whole thing.
# ============================================================

def make_train_step_adaptive(optimizer, gamma_optimizer):
    """Full-batch training step: joint gradient on params and gamma."""
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

    # (params_attn, gamma, opt_state, gamma_opt_state) are per-seed -> axis 0;
    # (X, y) are shared across seeds -> None.
    return jax.vmap(train_step, in_axes=(0, 0, 0, 0, None, None))


def make_train_step_fixed(optimizer, gamma):
    """Full-batch training step with fixed gamma (self-attention baseline)."""
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

    # gamma is a closed-over scalar constant (same for every seed), so it
    # isn't a vmapped argument at all -- only (params_attn, opt_state) batch.
    return jax.vmap(train_step, in_axes=(0, 0, None, None))


# ============================================================
# Training function
# ============================================================

def _init_batched_state(keys_init, condition, in_dim, L, scale, lr, weight_decay, gamma_lr, init_gamma):
    """Shared batched-init logic for both backends: (n_seeds, ...)-leading
    params_attn/gamma/opt_state, plus the optimizer objects the caller needs
    to build its own train_step wrapper (jit(vmap(...)) or
    jit(shard_map(vmap(...), ...))).
    """
    is_adaptive = condition in ("adaptive", "adaptive_pos")
    n_seeds = keys_init.shape[0]

    params_attn = jax.vmap(
        init_gamma_value_attn_params, in_axes=(0, None, None, None, None)
    )(keys_init, in_dim, L, L, scale)

    optimizer = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    opt_state = _broadcast_unbatched_leaves(jax.vmap(optimizer.init)(params_attn), n_seeds)

    if is_adaptive:
        gamma_param = jnp.full((n_seeds,), init_gamma, dtype=jnp.float32)
        gamma_optimizer = optax.sgd(learning_rate=gamma_lr)
        gamma_opt_state = _broadcast_unbatched_leaves(jax.vmap(gamma_optimizer.init)(gamma_param), n_seeds)
    else:
        gamma_param = jnp.zeros((n_seeds,), dtype=jnp.float32)  # for history/summary only
        gamma_optimizer = None
        gamma_opt_state = None

    return is_adaptive, n_seeds, params_attn, opt_state, optimizer, gamma_param, gamma_opt_state, gamma_optimizer


def _run_training_loop(
    *, train_step, is_adaptive, params_attn, gamma_param, opt_state, gamma_opt_state,
    X, y, num_steps, desc, show_progress,
):
    """Shared per-step loop for both backends -- same (params_attn, gamma,
    opt_state, gamma_opt_state) -> (..., loss, acc) calling convention
    either way, just a differently-wrapped train_step.

    Returns:
        train_loss, train_acc, gamma_hist: each a (n_seeds, num_steps) numpy array
        params_attn: trained parameters, pytree with a leading (n_seeds, ...) axis per leaf
    """
    train_loss_steps, train_acc_steps, gamma_steps = [], [], []

    step_iter = range(num_steps)
    pbar = None
    if show_progress:
        pbar = tqdm(step_iter, desc=desc, total=num_steps, leave=True)
        step_iter = pbar

    for _step in step_iter:
        if is_adaptive:
            params_attn, gamma_param, opt_state, gamma_opt_state, loss, acc = train_step(
                params_attn, gamma_param, opt_state, gamma_opt_state, X, y
            )
        else:
            params_attn, opt_state, loss, acc = train_step(params_attn, opt_state, X, y)

        train_loss_steps.append(loss)
        train_acc_steps.append(acc)
        gamma_steps.append(gamma_param)

        if pbar is not None:
            postfix = {
                "loss": f"{float(jnp.mean(loss)):.3f}",
                "acc": f"{float(jnp.mean(acc)):.3f}",
            }
            if is_adaptive:
                postfix["γ"] = f"{float(jnp.mean(gamma_param)):.3f}"
            pbar.set_postfix(postfix)

    if pbar is not None:
        pbar.close()

    # (num_steps, n_seeds) -> (n_seeds, num_steps), one row per seed so each
    # can be sliced straight into the existing summarize_history().
    train_loss = np.asarray(jnp.stack(train_loss_steps, axis=0)).T
    train_acc = np.asarray(jnp.stack(train_acc_steps, axis=0)).T
    gamma_hist = np.asarray(jnp.stack(gamma_steps, axis=0)).T

    return train_loss, train_acc, gamma_hist, params_attn


def train_seeds(
    *,
    keys_init,
    condition,
    label_mode,
    X,
    y,
    in_dim,
    L,
    num_steps,
    lr,
    weight_decay,
    gamma_lr,
    init_gamma=0.5,
    scale=0.02,
    show_progress=True,
):
    """Train ALL seeds for one (condition, label_mode) combo together, batched
    over seeds via a single jax.vmap on one device (see make_train_step_adaptive/
    fixed above). See train_seeds_shard_map() below for the multi-device variant.

    Args:
        keys_init: (n_seeds,) array of JAX PRNG keys, one per seed
        condition: 'adaptive', 'adaptive_pos', or 'attn_pos'
        label_mode: 'feature' or 'position'
        X, y: the full enumerated population (see ContextIdentification.generate_all_examples),
            shared/unbatched across all seeds
        in_dim: input dimension (L without pos codes, 2*L with)
        L: feature vector length (= number of classes)
        num_steps: number of full-batch gradient steps for this scale
        lr: learning rate for model parameters
        weight_decay: weight decay for model parameters
        gamma_lr: learning rate for gamma (adaptive conditions only)
        init_gamma: initial gamma value (adaptive conditions only)
        scale: weight-init scale g for init_gamma_value_attn_params
        show_progress: whether to show progress bar

    Returns:
        train_loss, train_acc, gamma_hist: each a (n_seeds, num_steps) numpy array
        params_attn: trained parameters, pytree with a leading (n_seeds, ...) axis per leaf
    """
    is_adaptive, n_seeds, params_attn, opt_state, optimizer, gamma_param, gamma_opt_state, gamma_optimizer = (
        _init_batched_state(keys_init, condition, in_dim, L, scale, lr, weight_decay, gamma_lr, init_gamma)
    )

    if is_adaptive:
        train_step = jax.jit(make_train_step_adaptive(optimizer, gamma_optimizer))
    else:
        gamma_fixed = jnp.array(0.0, dtype=jnp.float32)
        train_step = jax.jit(make_train_step_fixed(optimizer, gamma=gamma_fixed))

    return _run_training_loop(
        train_step=train_step,
        is_adaptive=is_adaptive,
        params_attn=params_attn,
        gamma_param=gamma_param,
        opt_state=opt_state,
        gamma_opt_state=gamma_opt_state,
        X=X,
        y=y,
        num_steps=num_steps,
        desc=f"{condition} | {label_mode} ({n_seeds} seeds, vmap)",
        show_progress=show_progress,
    )


def train_seeds_shard_map(
    *,
    keys_init,
    condition,
    label_mode,
    X,
    y,
    in_dim,
    L,
    num_steps,
    lr,
    weight_decay,
    gamma_lr,
    init_gamma=0.5,
    scale=0.02,
    show_progress=True,
):
    """Same contract as train_seeds() above, but explicitly partitions the
    batch of seeds across jax.device_count() CPU devices via jax.shard_map,
    instead of relying on a single jax.vmap on one device to be internally
    parallelized by XLA. Reuses the SAME per-seed train_step body
    (make_train_step_adaptive/fixed) as the per-device shard_map body --
    each device runs a genuinely local jax.vmap over its own slice of seeds
    (vmap within a shard, shard_map across shards/devices).

    IMPORTANT: requires XLA_FLAGS=--xla_force_host_platform_device_count=<N>
    set in the environment BEFORE this process starts -- JAX's CPU backend
    exposes only 1 device by default, and this must take effect before JAX
    initializes its backend, so it cannot be set from inside this script
    (set it in the calling shell / SLURM script). len(keys_init) (n_seeds)
    must be evenly divisible by jax.device_count(): each device gets an
    equal-size local shard of seeds.
    """
    is_adaptive, n_seeds, params_attn, opt_state, optimizer, gamma_param, gamma_opt_state, gamma_optimizer = (
        _init_batched_state(keys_init, condition, in_dim, L, scale, lr, weight_decay, gamma_lr, init_gamma)
    )

    n_devices = jax.device_count()
    if n_seeds % n_devices != 0:
        raise ValueError(
            f"n_seeds={n_seeds} must be evenly divisible by jax.device_count()={n_devices} "
            "for the shard_map backend (each device gets an equal-size shard of seeds). "
            "Check XLA_FLAGS=--xla_force_host_platform_device_count=<N> was set before this "
            "process started."
        )
    mesh = jax.make_mesh((n_devices,), ("seed",))
    seed_spec, shared_spec = P("seed"), P()

    if is_adaptive:
        local_step = make_train_step_adaptive(optimizer, gamma_optimizer)  # vmap over the LOCAL per-device seed shard
        train_step = jax.jit(jax.shard_map(
            local_step,
            mesh=mesh,
            in_specs=(seed_spec, seed_spec, seed_spec, seed_spec, shared_spec, shared_spec),
            out_specs=(seed_spec, seed_spec, seed_spec, seed_spec, seed_spec, seed_spec),
        ))
    else:
        gamma_fixed = jnp.array(0.0, dtype=jnp.float32)
        local_step = make_train_step_fixed(optimizer, gamma=gamma_fixed)
        train_step = jax.jit(jax.shard_map(
            local_step,
            mesh=mesh,
            in_specs=(seed_spec, seed_spec, shared_spec, shared_spec),
            out_specs=(seed_spec, seed_spec, seed_spec, seed_spec),
        ))

    return _run_training_loop(
        train_step=train_step,
        is_adaptive=is_adaptive,
        params_attn=params_attn,
        gamma_param=gamma_param,
        opt_state=opt_state,
        gamma_opt_state=gamma_opt_state,
        X=X,
        y=y,
        num_steps=num_steps,
        desc=f"{condition} | {label_mode} ({n_seeds} seeds, shard_map x{n_devices} devices)",
        show_progress=show_progress,
    )


def summarize_history(history, *, n_final, scale, num_steps):
    """Reduce a full per-step history to scalars.

    Called immediately after unpacking one seed's slice from train_seeds()'s
    batched output, so the full per-step history is discarded before the
    next seed's summary is computed.
    """
    n = min(n_final, num_steps)
    return {
        "final_train_acc": float(np.mean(history["train_acc"][-n:])),
        "final_train_loss": float(np.mean(history["train_loss"][-n:])),
        "final_gamma": float(history["gamma"][-1]),
        "scale": scale,
        "num_steps": num_steps,
        "n_final": n,
    }


# ============================================================
# Sweep function
# ============================================================

def run_weight_scale_sweep(
    *,
    scale,
    num_steps,
    seeds=tuple(range(20)),
    conditions=CONDITIONS,
    label_modes=("feature", "position"),
    L=6,
    backend="vmap",
    lr=1e-2,
    weight_decay=1e-3,
    gamma_lr=1e-2,
    init_gamma=0.5,
    n_final=100,
    show_progress=False,
    show_combo_progress=True,
    save_models=False,
    models_dir=None,
    save_dir=None,
):
    """Run all conditions x label_modes x seeds for ONE scale value.

    Args:
        scale: weight-init scale g
        num_steps: number of full-batch gradient steps for this scale
        seeds: random seeds (parameter initialization only -- the enumerated
            population does not depend on seed, see module docstring)
        conditions: which conditions to run
        label_modes: 'feature' and/or 'position'
        L: feature vector length (= number of classes); also fixes
            context_size=L, so the full population has N(L) instances
            (full_batch_combinatorics.py)
        backend: 'vmap' (train_seeds, default) or 'shard_map' (train_seeds_shard_map,
            requires XLA_FLAGS=--xla_force_host_platform_device_count=<N> set before
            this process started, and len(seeds) divisible by that N) -- see the
            module docstring for the tradeoffs between the two.
        lr: learning rate for model parameters
        weight_decay: weight decay
        gamma_lr: learning rate for gamma (adaptive conditions)
        init_gamma: initial gamma (adaptive conditions)
        n_final: number of final steps to average over per seed
        show_progress: per-run progress bars
        show_combo_progress: overall progress bar
        save_models: whether to save trained params
        models_dir: directory to save model parameters (required if save_models)
        save_dir: directory to save one results JSON per (condition, label_mode)
            combo actually run (see combo_filename) -- NOT a single combined file,
            so that array tasks sharing a scale but running different
            conditions/label_modes never clobber each other's output.

    Returns:
        results[condition][label_mode][seed] -> summary dict
    """
    if backend not in ("vmap", "shard_map"):
        raise ValueError(f"backend must be 'vmap' or 'shard_map', got {backend!r}")
    train_fn = train_seeds if backend == "vmap" else train_seeds_shard_map
    if save_models and models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    results = {cond: {lm: {} for lm in label_modes} for cond in conditions}

    # The enumerated population is deterministic and only varies with
    # (positional_codes, label_mode), so build it once and reuse across every
    # seed/condition that shares that combo.
    datasets = {}
    for use_pos_codes in (False, True):
        for label_mode in label_modes:
            task = ContextIdentification(
                random.PRNGKey(0),
                num_vectors=L + 1,
                vector_len=L,
                input_mode="cross",
                label_mode=label_mode,
                positional_codes=use_pos_codes,
            )
            X, y = task.generate_all_examples()
            datasets[(use_pos_codes, label_mode)] = (jnp.asarray(X), jnp.asarray(y))

    # Seeds within a combo are no longer trained one at a time: train_seeds
    # batches all of them into a single jax.vmap'd computation, so progress
    # is tracked per (condition, label_mode) combo, not per seed.
    total_combos = len(conditions) * len(label_modes)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_combos, desc=f"Weight scale g={scale:g} (full-batch, {backend}, {len(seeds)} seeds/combo)", leave=True)

    for condition in conditions:
        use_pos_codes = condition in ("adaptive_pos", "attn_pos")
        in_dim = 2 * L if use_pos_codes else L

        for label_mode in label_modes:
            X, y = datasets[(use_pos_codes, label_mode)]

            keys_init = jnp.stack([
                random.fold_in(random.PRNGKey(int(seed)), hash((condition, label_mode)) % 10000)
                for seed in seeds
            ])

            train_loss, train_acc, gamma_hist, params_attn = train_fn(
                keys_init=keys_init,
                condition=condition,
                label_mode=label_mode,
                X=X,
                y=y,
                in_dim=in_dim,
                L=L,
                num_steps=num_steps,
                lr=lr,
                weight_decay=weight_decay,
                gamma_lr=gamma_lr,
                init_gamma=init_gamma,
                scale=scale,
                show_progress=show_progress,
            )

            for i, seed in enumerate(seeds):
                seed_history = {
                    "train_loss": train_loss[i],
                    "train_acc": train_acc[i],
                    "gamma": gamma_hist[i],
                }
                results[condition][label_mode][int(seed)] = summarize_history(
                    seed_history, n_final=n_final, scale=scale, num_steps=num_steps
                )

                if save_models and models_dir is not None:
                    params_i = jax.tree_util.tree_map(lambda leaf, i=i: leaf[i], params_attn)
                    run_name = f"pos_codes_full_batch_{condition}_{label_mode}_seed{seed}"
                    _save_params_pkl(params_i, models_dir / f"{run_name}_params.pkl")

            if outer is not None:
                outer.update(1)

    if outer is not None:
        outer.close()

    if save_dir is not None:
        save_dir = Path(save_dir)
        for condition in conditions:
            for label_mode in label_modes:
                combo_results = {condition: {label_mode: results[condition][label_mode]}}
                save_results_json(combo_results, save_dir / combo_filename(scale, condition, label_mode))

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
    p.add_argument("--steps-override", type=int, default=None,
                    help="Override the scale->step-count table (for smoke tests).")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--conditions", type=str, default=",".join(CONDITIONS))
    p.add_argument("--label-modes", type=str, default="feature,position")
    p.add_argument("--L", type=int, default=6)
    p.add_argument(
        "--backend", type=str, default="vmap", choices=("vmap", "shard_map"),
        help="'vmap' (default): single jax.vmap on one device. 'shard_map': "
             "partition seeds across jax.device_count() CPU devices -- requires "
             "XLA_FLAGS=--xla_force_host_platform_device_count=<N> set in the "
             "environment BEFORE this process starts, and --n-seeds divisible by N. "
             "See the module docstring for the tradeoffs.",
    )
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--gamma-lr", type=float, default=1e-2)
    p.add_argument("--init-gamma", type=float, default=0.5)
    p.add_argument("--n-final", type=int, default=100)
    p.add_argument("--data-dir", type=str, default="./data/weight_scale_pos_codes_full_batch_sweep")
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
        num_steps = args.steps_override or STEPS_FOR_SCALE[scale]
        tag = scale_tag(scale)
        data_dir = Path(args.data_dir)
        models_dir = data_dir / "models" / f"scale_{tag}" if args.save_models else None

        run_weight_scale_sweep(
            scale=scale,
            num_steps=num_steps,
            seeds=tuple(range(args.n_seeds)),
            conditions=conditions,
            label_modes=label_modes,
            L=args.L,
            backend=args.backend,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gamma_lr=args.gamma_lr,
            init_gamma=args.init_gamma,
            n_final=args.n_final,
            show_progress=args.progress,
            show_combo_progress=True,
            save_models=args.save_models,
            models_dir=models_dir,
            save_dir=data_dir,
        )
