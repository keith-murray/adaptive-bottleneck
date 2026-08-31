"""Sanity check for train_test_split_indices, the sampling-percentage sweep's
train/test split of an enumerated ContextIdentification population."""

import numpy as np

from adaptive_bottleneck.tasks import train_test_split_indices


def main():
    n_total = 27_684  # N(6), a real enumerated population size
    seed = 0

    percentages = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    splits = {}
    for p in percentages:
        train_idx, test_idx = train_test_split_indices(n_total, seed, p)
        splits[p] = (train_idx, test_idx)

        expected_test_frac = min(0.2, 1.0 - p)
        assert len(train_idx) == round(p * n_total), (p, len(train_idx))
        assert len(test_idx) == round(expected_test_frac * n_total) or len(test_idx) == n_total - len(train_idx)
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, f"train/test overlap at p={p}"
        assert train_idx.max(initial=-1) < n_total and test_idx.max(initial=-1) < n_total
        print(f"p={p:>6.3f}: train_n={len(train_idx):>7,} test_n={len(test_idx):>6,} "
              f"(test_frac={expected_test_frac:.2f})")

    # p=1.0 must have an empty test set.
    assert len(splits[1.0][1]) == 0, "train_frac=1.0 should omit the test set"

    # p=0.9 must reserve exactly the "remaining 10%", not the full 20%.
    assert len(splits[0.9][1]) == round(0.1 * n_total), "train_frac=0.9 should leave 10% for test"

    # Nesting: train_idx(p) subset of train_idx(p') for p < p', same seed.
    for p_small, p_big in zip(percentages[:-1], percentages[1:]):
        small_train = set(splits[p_small][0].tolist())
        big_train = set(splits[p_big][0].tolist())
        assert small_train.issubset(big_train), f"train({p_small}) not nested in train({p_big})"
    print("Nesting property verified across all percentages.")

    # Determinism: same seed -> identical split.
    train_a, test_a = train_test_split_indices(n_total, seed=7, train_frac=0.3)
    train_b, test_b = train_test_split_indices(n_total, seed=7, train_frac=0.3)
    assert np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)

    # Different seeds -> different splits.
    train_c, _ = train_test_split_indices(n_total, seed=8, train_frac=0.3)
    assert not np.array_equal(train_a, train_c)
    print("Determinism (same seed) and seed-sensitivity (different seeds) verified.")

    # Shared across "conditions"/"label_modes": the function takes no such argument,
    # so calling it twice with the same (n_total, seed, train_frac) -- as every
    # condition/label_mode sharing a population would -- yields the same split.
    for _ in range(3):
        train_d, test_d = train_test_split_indices(n_total, seed=0, train_frac=0.5)
        assert np.array_equal(train_d, splits[0.5][0]) and np.array_equal(test_d, splits[0.5][1])
    print("Cross-condition/label_mode sharing verified (split depends only on n_total, seed, train_frac).")

    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
