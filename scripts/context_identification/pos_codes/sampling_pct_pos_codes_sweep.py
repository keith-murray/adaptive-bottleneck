"""Sampling-percentage sweep for ContextIdentification pos-codes comparison
(claude/2026_08_31/TODO.md).

Sweeps the fraction of the full enumerated L=8 population (N(8) = 13,969,440,
see full_batch_combinatorics.py) used for mini-batch training, across all three
conditions (adaptive, adaptive_pos, attn_pos) and both label modes, to test
whether ARCA without positional codes is simply more sample-efficient than the
positional-code conditions (open question from the 2026-08-24 session).

Train/test split: train_test_split_indices() (tasks.py) reserves up to 20% of
the population for testing, shrinking to whatever's left over above 80% training
(10% at train_frac=0.9, 0% -- omitted -- at train_frac=1.0), via one seed-derived
permutation shared across every condition/label_mode/percentage at that seed, so
comparisons are apples-to-apples (see train_test_split_indices' docstring).

Fixed step count, not epochs: batches/epoch varies from ~21 to ~218,000 across
this sweep's percentages (batch_size=64), so training is specified as a fixed
global mini-batch step count (train_one_run's `num_steps`) instead -- ds_train is
iterated repeatedly (reshuffled each pass) until exactly num_steps updates have
been taken. Train loss/acc is logged every step; test loss/acc (full ds_test,
profiled as cheap even at ~2.8M rows) is logged every `eval_every` steps.
"""

import argparse
import itertools
from pathlib import Path

import jax
import jax.random as random
import jax.numpy as jnp
import jax_dataloader as jdl
import numpy as np
import optax

from tqdm import tqdm

from adaptive_bottleneck.models import (
    init_gamma_value_attn_params,
    gamma_value_cross_attention_forward,
)
from adaptive_bottleneck.tasks import ContextIdentification, train_test_split_indices
from adaptive_bottleneck.save import save_results_json, _save_params_pkl, _ensure_dir


CONDITIONS = ("adaptive", "adaptive_pos", "attn_pos")
LABEL_MODES = ("feature", "position")
CONDITION_LABEL_MODE_COMBOS = list(itertools.product(CONDITIONS, LABEL_MODES))


def forward_logits(params_attn, X, gamma):
    X_q = X[:, 0:1, :]
    X_c = X[:, 1:, :]
    out, attn_w = gamma_value_cross_attention_forward(
        params_attn, X_c, X_q, gamma=gamma, linear=False,
    )
    return out[:, 0, :], attn_w


def ce_with_logits(logits, y):
    """Cross-entropy for integer labels."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def acc_from_logits(logits, y):
    """Accuracy from logits."""
    return (jnp.argmax(logits, axis=-1) == y).mean()


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
    batch_size,
    num_steps,
    eval_every,
    max_eval_batches=200,
    lr=1e-2,
    weight_decay=1e-3,
    gamma_lr=1e-2,
    init_gamma=0.5,
    scale=0.02,
    show_progress=True,
):
    """Train one run for a fixed number of mini-batch steps (not epochs).

    ds_train is iterated repeatedly -- each pass reshuffled, since ds_train is
    built with shuffle=True -- wrapping around as many times as needed for small
    percentages and stopping mid-pass for large ones, until exactly num_steps
    mini-batch updates have been taken.

    Train loss/acc is recorded every step (cheap: one batch_size=64 batch). Test
    loss/acc is recorded every `eval_every` steps, but the test set stays a fixed
    ~20% of the *full* population regardless of train_frac (see
    train_test_split_indices) -- for small train_frac this can be orders of
    magnitude larger than the training set. Periodic checkpoints evaluate only a
    random subsample of `max_eval_batches` batches, drawn directly via numpy
    indexing (np.random.Generator.choice, ~0.3ms even at ~2.8M rows -- genuinely
    O(k), not O(N)); only the final step (step == num_steps) evaluates the
    complete ds_test in order, for one authoritative full-population metric per
    run. If ds_test is None (train_frac == 1.0, no held-out data), test
    evaluation is skipped entirely.

    IMPORTANT: ds_test is plain (X_test, y_test) arrays, NOT a jax_dataloader
    DataLoader -- profiling on the cluster caught a real bug in an earlier
    version that routed capped eval through a shuffle=True DataLoader and broke
    out early after max_eval_batches: jax_dataloader's DataLoaderJAX.__iter__
    (loaders/jax.py) calls jrand.permutation() on the ENTIRE index array on
    every fresh iteration, regardless of how many batches are consumed before
    the early break -- so "capping" the batch count did nothing to cap the
    actual cost, which stayed O(N) per checkpoint. Confirmed via the profiler:
    runs with a test set took ~755s each (199 checkpoints x ~3s/full-array
    permutation), while the two train_frac=1.0 runs (no test set, zero eval
    calls) took only ~100-200s -- the gap was pure wasted permutation work, not
    training or legitimate eval cost. Do not reintroduce a DataLoader for
    periodic eval; if ds_train's cycling behavior is ever changed, keep this
    reasoning in mind.

    Args:
        key_init: JAX PRNG key.
        condition: 'adaptive', 'adaptive_pos', or 'attn_pos'.
        label_mode: 'feature' or 'position'.
        ds_train: training DataLoader (shuffle=True).
        ds_test: (X_test, y_test) arrays, or None to skip evaluation.
        in_dim: input dimension (L without pos codes, 2*L with).
        L: feature vector length (= number of classes).
        batch_size: evaluation batch size (training batch size is implicit in
            ds_train; this must match it for eval_step's jit to reuse cache).
        num_steps: total mini-batch training steps.
        eval_every: evaluate ds_test every this many steps (plus the final step).
        max_eval_batches: cap on batches evaluated at non-final checkpoints
            (~12,800 examples at the default 200 batches x batch_size=64 -- SEM
            on the accuracy estimate is ~0.4%). Ignored at the final step, which
            always evaluates the complete ds_test.
        lr, weight_decay: AdamW hyperparameters for params_attn.
        gamma_lr: SGD learning rate for gamma (adaptive conditions only).
        init_gamma: initial gamma value (adaptive conditions only).
        scale: weight-init scale g for init_gamma_value_attn_params.
        show_progress: whether to show a tqdm progress bar.

    Returns:
        history: dict with "step"/"train_loss"/"train_acc"/"gamma" (one entry
            per training step) and "test_step"/"test_loss"/"test_acc"/
            "test_n_batches" (one entry per eval_every steps, plus the final
            step; empty if ds_test is None).
        params_attn: trained parameters.
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
        gamma_param = gamma_fixed
        train_step = make_train_step_fixed(optimizer, gamma=gamma_fixed)
        eval_step = make_eval_step_fixed(gamma=gamma_fixed)

    history = {
        "step": [], "train_loss": [], "train_acc": [], "gamma": [],
        "test_step": [], "test_loss": [], "test_acc": [], "test_n_batches": [],
    }

    X_test, y_test = ds_test if ds_test is not None else (None, None)
    n_test = X_test.shape[0] if X_test is not None else 0
    # Seeded from key_init so periodic subsamples are reproducible per run but
    # differ across checkpoints (advances each run_eval call) and across seeds.
    eval_rng = np.random.default_rng(int(jax.random.randint(key_init, (), 0, 2**31 - 1)))

    def run_eval(step, max_batches):
        if ds_test is None:
            return
        if max_batches is None:
            # Final step: complete, unshuffled pass -- order doesn't matter for
            # an aggregate metric, so no permutation cost at all.
            idx = np.arange(n_test)
        else:
            k = min(max_batches * batch_size, n_test)
            idx = eval_rng.choice(n_test, size=k, replace=False)

        n_complete = len(idx) - len(idx) % batch_size
        idx = idx[:n_complete]

        test_losses, test_accs = [], []
        for i in range(0, n_complete, batch_size):
            batch_idx = idx[i:i + batch_size]
            Xb, yb = X_test[batch_idx], y_test[batch_idx]
            if is_adaptive:
                loss, acc = eval_step(params_attn, Xb, yb, gamma_param)
            else:
                loss, acc = eval_step(params_attn, Xb, yb)
            test_losses.append(loss)
            test_accs.append(acc)

        history["test_step"].append(step)
        history["test_loss"].append(float(jnp.mean(jnp.stack(test_losses))))
        history["test_acc"].append(float(jnp.mean(jnp.stack(test_accs))))
        history["test_n_batches"].append(n_complete // batch_size)

    pbar = None
    if show_progress:
        pbar = tqdm(total=num_steps, desc=f"{condition} | {label_mode}", leave=True)

    step = 0
    while step < num_steps:
        step_before_pass = step
        for X, y in ds_train:
            if step >= num_steps:
                break

            if is_adaptive:
                params_attn, gamma_param, opt_state, gamma_opt_state, loss, acc = train_step(
                    params_attn, gamma_param, opt_state, gamma_opt_state, X, y
                )
            else:
                params_attn, opt_state, loss, acc = train_step(params_attn, opt_state, X, y)

            step += 1
            history["step"].append(step)
            history["train_loss"].append(float(loss))
            history["train_acc"].append(float(acc))
            history["gamma"].append(float(gamma_param) if is_adaptive else 0.0)

            is_last_step = step == num_steps
            if step % eval_every == 0 or is_last_step:
                run_eval(step, max_batches=None if is_last_step else max_eval_batches)

            if pbar is not None:
                pbar.update(1)
                if step % eval_every == 0 or is_last_step:
                    postfix = {"loss": f"{loss:.3f}", "acc": f"{acc:.3f}"}
                    if is_adaptive:
                        postfix["γ"] = f"{float(gamma_param):.3f}"
                    if history["test_acc"]:
                        postfix["test_acc"] = f"{history['test_acc'][-1]:.3f}"
                    pbar.set_postfix(postfix)

        if step == step_before_pass:
            raise RuntimeError(
                "ds_train yielded 0 batches in a full pass (dataset smaller than "
                "one batch, with drop_last=True) -- would loop forever. Increase "
                "the training set size or decrease batch_size."
            )

    if pbar is not None:
        pbar.close()

    return history, params_attn


# ============================================================
# Schedules
# ============================================================

def build_schedules():
    """The two sampling-percentage sweep schedules, sharing the p=1.0 endpoint.

    log: np.logspace(-4, 0, num=10) -- 0.01% to 100%, train_n from ~1,397 to
        13,969,440 at L=8, 4 orders of magnitude.
    linear: 10%, 20%, ..., 100% -- train_n from ~1.4M to 13,969,440 at L=8.
    """
    log_schedule = tuple(float(p) for p in np.logspace(-4, 0, num=10))
    linear_schedule = tuple(round(float(p), 2) for p in np.arange(0.1, 1.01, 0.1))
    return {"log": log_schedule, "linear": linear_schedule}


# ============================================================
# Sweep function
# ============================================================

def run_filename(schedule_name, pct_idx, condition, label_mode, seed):
    """'log__pct03__adaptive_pos__feature__seed7.json' style per-run filename.

    One file per (schedule, percentage, condition, label_mode, seed) -- not one
    combined file per combo -- both to avoid combining ~1,140 runs' full
    per-step histories into a single unwieldy file, and because every run's
    history is the same size (num_steps entries) regardless of percentage, so
    per-run files stay a predictable, manageable size (concurrent SLURM array
    tasks also never collide on a shared file this way).
    """
    return f"{schedule_name}__pct{pct_idx:02d}__{condition}__{label_mode}__seed{seed}.json"


def run_sampling_pct_sweep_for_combo(
    *,
    condition,
    label_mode,
    seeds,
    schedules,
    L,
    batch_size,
    num_steps,
    eval_every,
    max_eval_batches,
    lr,
    weight_decay,
    gamma_lr,
    init_gamma,
    data_dir,
    save_models=False,
    models_dir=None,
    show_progress=False,
    show_combo_progress=True,
):
    """Run every (schedule, percentage, seed) combo for one (condition, label_mode).

    Generates the full L-population once via generate_all_examples() (shared
    across every percentage/seed for this combo -- positional_codes is fixed by
    condition, label_mode is fixed by the label_mode arg), then for each
    (schedule, percentage, seed) draws that seed's shared train/test split (see
    train_test_split_indices -- identical split across every condition/label_mode
    at a given seed and percentage, for apples-to-apples comparison) and trains
    for a fixed step count (train_one_run).

    At train_frac == 1.0, train_idx covers the whole population; indexing with it
    (X[train_idx]) would still make a full copy via numpy fancy indexing, so that
    case reuses X, y directly instead.
    """
    positional_codes = condition in ("adaptive_pos", "attn_pos")
    in_dim = 2 * L if positional_codes else L

    task = ContextIdentification(
        random.PRNGKey(0), num_vectors=L + 1, vector_len=L,
        input_mode="cross", label_mode=label_mode, positional_codes=positional_codes,
    )
    X, y = task.generate_all_examples()
    n_total = X.shape[0]

    data_dir = Path(data_dir)
    _ensure_dir(data_dir)
    if save_models and models_dir is not None:
        models_dir = Path(models_dir)
        _ensure_dir(models_dir)

    total_runs = sum(len(pcts) for pcts in schedules.values()) * len(seeds)
    outer = None
    if show_combo_progress:
        outer = tqdm(total=total_runs, desc=f"Sampling pct sweep: {condition} | {label_mode}", leave=True)

    for schedule_name, pcts in schedules.items():
        for pct_idx, train_frac in enumerate(pcts):
            for seed in seeds:
                seed = int(seed)
                train_idx, test_idx = train_test_split_indices(n_total, seed, train_frac)

                if train_idx.size < batch_size:
                    raise ValueError(
                        f"schedule={schedule_name!r} pct_idx={pct_idx} (train_frac={train_frac}) "
                        f"seed={seed}: train_idx has only {train_idx.size} rows, below "
                        f"batch_size={batch_size} -- with drop_last=True this yields 0 batches "
                        f"per pass and would loop forever. Increase train_frac's lower bound, "
                        f"increase L, or decrease batch_size."
                    )

                if train_idx.size == n_total:
                    X_train, y_train = X, y
                else:
                    X_train, y_train = X[train_idx], y[train_idx]

                loader_key = random.fold_in(
                    random.PRNGKey(seed), hash((schedule_name, pct_idx, "loader")) % 10000
                )
                ds_train = jdl.DataLoader(
                    jdl.ArrayDataset(X_train, y_train), "jax",
                    batch_size=batch_size, shuffle=True, drop_last=True,
                    generator=loader_key,
                )

                # Plain arrays, not a DataLoader -- train_one_run draws its own
                # cheap numpy subsamples for periodic eval (see its docstring
                # for why routing this through jdl's shuffle=True DataLoader
                # was a real bug: full O(N) permutation cost per checkpoint,
                # regardless of an early break after max_eval_batches).
                ds_test = None
                if test_idx.size > 0:
                    ds_test = (X[test_idx], y[test_idx])

                key_init = random.fold_in(
                    random.PRNGKey(seed),
                    hash((condition, label_mode, schedule_name, pct_idx, "init")) % 10000,
                )

                history, params_attn = train_one_run(
                    key_init=key_init,
                    condition=condition,
                    label_mode=label_mode,
                    ds_train=ds_train,
                    ds_test=ds_test,
                    in_dim=in_dim,
                    L=L,
                    batch_size=batch_size,
                    num_steps=num_steps,
                    eval_every=eval_every,
                    max_eval_batches=max_eval_batches,
                    lr=lr,
                    weight_decay=weight_decay,
                    gamma_lr=gamma_lr,
                    init_gamma=init_gamma,
                    show_progress=show_progress,
                )

                result = {
                    "condition": condition,
                    "label_mode": label_mode,
                    "schedule": schedule_name,
                    "pct_idx": pct_idx,
                    "train_frac": train_frac,
                    "seed": seed,
                    "L": L,
                    "n_total": n_total,
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "batch_size": batch_size,
                    "num_steps": num_steps,
                    "eval_every": eval_every,
                    "max_eval_batches": max_eval_batches,
                    "history": history,
                }
                save_results_json(
                    result,
                    data_dir / run_filename(schedule_name, pct_idx, condition, label_mode, seed),
                )

                if save_models and models_dir is not None:
                    run_name = f"sampling_pct_{schedule_name}_pct{pct_idx}_{condition}_{label_mode}_seed{seed}"
                    _save_params_pkl(params_attn, models_dir / f"{run_name}_params.pkl")

                if outer is not None:
                    outer.update(1)

    if outer is not None:
        outer.close()


# ============================================================
# CLI
# ============================================================

def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--combo-idx", type=int, default=None, choices=range(len(CONDITION_LABEL_MODE_COMBOS)),
        help="Index into (condition, label_mode) pairs (0=adaptive/feature .. "
             "5=attn_pos/position). Typically $SLURM_ARRAY_TASK_ID. If omitted, "
             "all 6 combos run sequentially in-process (local smoke-test use "
             "only -- do not use this for the full HPC sweep).",
    )
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=None,
                    help="Evaluate ds_test every this many steps. Default: num_steps // 200.")
    p.add_argument("--max-eval-batches", type=int, default=200,
                    help="Cap on batches evaluated at non-final checkpoints; the final step "
                         "always evaluates the complete test set.")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--gamma-lr", type=float, default=1e-2)
    p.add_argument("--init-gamma", type=float, default=0.5)
    p.add_argument("--data-dir", type=str, default="./data/sampling_pct_pos_codes_sweep")
    p.add_argument("--save-models", action="store_true")
    p.add_argument("--progress", action="store_true", help="show per-run tqdm bars")
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    eval_every = args.eval_every or (args.num_steps // 200)
    schedules = build_schedules()

    combo_indices = (
        [args.combo_idx] if args.combo_idx is not None else list(range(len(CONDITION_LABEL_MODE_COMBOS)))
    )

    data_dir = Path(args.data_dir)
    for idx in combo_indices:
        condition, label_mode = CONDITION_LABEL_MODE_COMBOS[idx]
        models_dir = data_dir / "models" / f"{condition}_{label_mode}" if args.save_models else None

        run_sampling_pct_sweep_for_combo(
            condition=condition,
            label_mode=label_mode,
            seeds=tuple(range(args.n_seeds)),
            schedules=schedules,
            L=args.L,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            eval_every=eval_every,
            max_eval_batches=args.max_eval_batches,
            lr=args.lr,
            weight_decay=args.weight_decay,
            gamma_lr=args.gamma_lr,
            init_gamma=args.init_gamma,
            data_dir=data_dir,
            save_models=args.save_models,
            models_dir=models_dir,
            show_progress=args.progress,
            show_combo_progress=True,
        )
