"""Sanity check for the sampling-percentage sweep's fixed-step-count training
loop (sampling_pct_pos_codes_sweep.train_one_run), end-to-end on a small L."""

import math

import jax.random as random
import jax_dataloader as jdl
import numpy as np

from adaptive_bottleneck.tasks import ContextIdentification, train_test_split_indices

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "context_identification" / "pos_codes"))
from sampling_pct_pos_codes_sweep import train_one_run  # noqa: E402


def build_loaders(L, positional_codes, label_mode, seed, train_frac, batch_size):
    task = ContextIdentification(
        random.PRNGKey(0), num_vectors=L + 1, vector_len=L,
        input_mode="cross", label_mode=label_mode, positional_codes=positional_codes,
    )
    X, y = task.generate_all_examples()
    train_idx, test_idx = train_test_split_indices(X.shape[0], seed, train_frac)

    ds_train = jdl.DataLoader(
        jdl.ArrayDataset(X[train_idx], y[train_idx]), "jax",
        batch_size=batch_size, shuffle=True, drop_last=True,
        generator=random.PRNGKey(seed),
    )
    # ds_test is plain arrays, not a DataLoader -- see train_one_run's docstring
    # for why (jdl's shuffle=True DataLoader pays a full O(N) permutation cost
    # per __iter__ call regardless of an early break, which defeated eval
    # capping when profiled on the cluster).
    ds_test = None
    if test_idx.size > 0:
        ds_test = (X[test_idx], y[test_idx])
    return ds_train, ds_test, test_idx.size // batch_size


def main():
    L = 6
    num_steps = 50
    eval_every = 10
    batch_size = 8
    max_eval_batches = 2  # small on purpose, to exercise the capping behavior

    for condition in ("adaptive", "attn_pos"):
        for train_frac in (0.5, 1.0):
            positional_codes = condition in ("adaptive_pos", "attn_pos")
            in_dim = 2 * L if positional_codes else L

            ds_train, ds_test, full_test_batches = build_loaders(
                L, positional_codes, "feature", seed=0, train_frac=train_frac, batch_size=batch_size,
            )

            history, params_attn = train_one_run(
                key_init=random.PRNGKey(1),
                condition=condition,
                label_mode="feature",
                ds_train=ds_train,
                ds_test=ds_test,
                in_dim=in_dim,
                L=L,
                batch_size=batch_size,
                num_steps=num_steps,
                eval_every=eval_every,
                max_eval_batches=max_eval_batches,
                lr=1e-2,
                weight_decay=1e-3,
                gamma_lr=1e-2,
                show_progress=False,
            )

            assert len(history["step"]) == num_steps, history["step"]
            assert history["step"] == list(range(1, num_steps + 1))
            assert len(history["train_loss"]) == num_steps
            assert len(history["train_acc"]) == num_steps
            assert len(history["gamma"]) == num_steps
            assert not any(math.isnan(v) for v in history["train_loss"])

            if train_frac == 1.0:
                assert ds_test is None
                assert history["test_step"] == [] and history["test_loss"] == []
            else:
                assert ds_test is not None
                expected_evals = num_steps // eval_every
                assert len(history["test_step"]) == expected_evals, history["test_step"]
                assert history["test_step"] == list(range(eval_every, num_steps + 1, eval_every))
                assert not any(math.isnan(v) for v in history["test_loss"])

                # Non-final checkpoints are capped at max_eval_batches; only the
                # final step (== num_steps) evaluates the complete test set.
                assert full_test_batches > max_eval_batches, "test fixture too small to exercise capping"
                for s, n_batches in zip(history["test_step"], history["test_n_batches"]):
                    if s == num_steps:
                        assert n_batches == full_test_batches, (s, n_batches, full_test_batches)
                    else:
                        assert n_batches == max_eval_batches, (s, n_batches, max_eval_batches)

            print(f"condition={condition:>10} train_frac={train_frac}: "
                  f"final train_acc={history['train_acc'][-1]:.3f} "
                  f"test_acc={(history['test_acc'][-1] if history['test_acc'] else float('nan')):.3f} OK")

    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
