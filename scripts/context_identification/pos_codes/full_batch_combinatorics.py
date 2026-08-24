"""Functional relationship between L and the number of unique ContextIdentification
task instances, to pick a tractable L for full-batch (exhaustive) gradient descent.

A trial is (ref_idx, ctx_1, ..., ctx_L), each in {0, ..., L-1} (num_vectors=L+1,
vector_len=L, context_size=L, matching the convention in weight_scale_pos_codes_sweep.py
and pos_codes_comparison_sweep.py). It is *valid* -- i.e. reachable by
ContextIdentification._sample_one_example's constructive rejection sampling, which
is built specifically to avoid distance ties -- iff (a) no context item is at
distance 0 from the reference (the sampled target distance d always has d >= 1,
since jax.random.geometric's support is {1, 2, ...}, and the "rest" of the context
is drawn from values strictly farther than d) and (b) exactly one context slot
achieves the minimum distance |ctx_i - ref_idx| to the reference. y is then fully
determined by X (both feature and position labels), so N(L) below is also the
number of unique valid (X, y) pairs.

Closed form: for a fixed ref_idx, let n_d = number of feature values at distance
exactly d from ref_idx (d = 1..max_d, n_d in {1, 2}), and F_d = sum_{d' > d} n_d' =
number of values strictly farther than d. The number of valid trials with minimum
distance level d is L * n_d * F_d**(L-1) (choose which of the L slots is closest,
which value achieves that distance, and independently fill the remaining L-1 slots
from the F_d farther values). Terms with F_d == 0 vanish automatically, so no special
casing of the max distance is needed.

Produces:
  1. full_batch_combinatorics.png -- N(L) vs L, log-scale y-axis
  2. full_batch_combinatorics.csv -- table of L, N(L), and brute-force cross-check
"""

import csv
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_bottleneck.utils import load_plot_config


def n_valid_formula(L):
    """N(L) via the closed-form combinatorial formula above."""
    total = 0
    for ref_idx in range(L):
        dists = np.abs(np.arange(L) - ref_idx)
        max_d = dists.max()
        n_d = np.array([np.sum(dists == d) for d in range(0, max_d + 1)])
        farther_cumsum = np.cumsum(n_d[::-1])[::-1]  # farther_cumsum[d] = sum_{d'>=d} n_d'
        for d in range(1, max_d + 1):
            F_d = farther_cumsum[d + 1] if d + 1 <= max_d else 0
            total += L * n_d[d] * (F_d ** (L - 1))
    return int(total)


def n_valid_brute_force(L):
    """N(L) via brute-force enumeration of {0,...,L-1}^(L+1), for small L only.

    A context item at distance 0 from the reference (i.e. equal to ref_idx) never
    occurs in a real trial: _sample_one_example's target distance d is drawn with
    support {1, 2, ...} and the "rest" pool only contains values strictly farther
    than d >= 1. So besides requiring a unique minimum distance, that minimum must
    also be >= 1.
    """
    count = 0
    for ref_idx in range(L):
        for ctx in itertools.product(range(L), repeat=L):
            dists = [abs(c - ref_idx) for c in ctx]
            min_d = min(dists)
            if min_d >= 1 and dists.count(min_d) == 1:
                count += 1
    return count


def verify_formula(L_values=range(2, 7)):
    """Cross-check the closed-form formula against brute-force enumeration."""
    print(f"{'L':>3} {'N(L) formula':>15} {'N(L) brute force':>18} {'match':>6}")
    all_match = True
    for L in L_values:
        formula = n_valid_formula(L)
        brute = n_valid_brute_force(L)
        match = formula == brute
        all_match &= match
        print(f"{L:>3} {formula:>15} {brute:>18} {str(match):>6}")
    if not all_match:
        raise AssertionError("Closed-form formula does not match brute-force enumeration.")
    print("Formula verified against brute force for all L above.\n")
    return all_match


def build_table(L_values=range(2, 13)):
    rows = []
    for L in L_values:
        N = n_valid_formula(L)
        rows.append({"L": L, "N": N, "N_per_class": N / L})
    return rows


def plot_n_vs_L(
    rows,
    save_path="./results/context_identification/pos_codes/full_batch_combinatorics.png",
):
    Ls = [r["L"] for r in rows]
    Ns = [r["N"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(Ls, Ns, marker="o", color=plt.cm.plasma(0.4), linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("$L$ (number of classes)")
    ax.set_ylabel("Number of unique valid task instances $N(L)$")
    ax.set_xticks(Ls)
    ax.grid(alpha=0.2, which="both")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved {save_path}")
    plt.show()


def save_table_csv(
    rows,
    save_path="./results/context_identification/pos_codes/full_batch_combinatorics.csv",
):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["L", "N", "N_per_class"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    load_plot_config()

    verify_formula(L_values=range(2, 7))

    rows = build_table(L_values=range(2, 13))
    print(f"{'L':>3} {'N(L)':>12}")
    for r in rows:
        print(f"{r['L']:>3} {r['N']:>12}")

    save_table_csv(rows)
    plot_n_vs_L(rows)
