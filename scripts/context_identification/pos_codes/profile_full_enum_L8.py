"""Feasibility check: time and peak-memory profile of
ContextIdentification.generate_all_examples() at L=8 (N(8) = 13,969,440), before
designing the sampling-percentage sweep (claude/2026_08_31/TODO.md).

Run each `--positional-codes` setting as its own process invocation so peak-RSS
measurements don't compound across variants:

    uv run python scripts/context_identification/pos_codes/profile_full_enum_L8.py --positional-codes 0
    uv run python scripts/context_identification/pos_codes/profile_full_enum_L8.py --positional-codes 1
"""

import argparse
import importlib.util
import platform
import resource
import signal
import time
from pathlib import Path

import jax.random as random

from adaptive_bottleneck.tasks import ContextIdentification

L = 8


class _Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise _Timeout("generate_all_examples() exceeded the timeout")


def _load_n_valid_formula():
    """Import n_valid_formula from the sibling combinatorics script, rather than
    hardcoding N(8), so this check stays consistent with the verified closed form."""
    combinatorics_path = Path(__file__).parent / "full_batch_combinatorics.py"
    spec = importlib.util.spec_from_file_location("full_batch_combinatorics", combinatorics_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.n_valid_formula


def profile(positional_codes, timeout_s):
    n_valid_formula = _load_n_valid_formula()
    expected_n = n_valid_formula(L)

    task = ContextIdentification(
        random.PRNGKey(0),
        num_vectors=L + 1,
        vector_len=L,
        input_mode="cross",
        label_mode="both",
        positional_codes=positional_codes,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_unit = "bytes" if platform.system() == "Darwin" else "KB"

    old_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(timeout_s)
    start = time.perf_counter()
    try:
        X, y_feature, y_position = task.generate_all_examples()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    elapsed_s = time.perf_counter() - start

    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    n_rows = X.shape[0]
    match = n_rows == expected_n

    print(f"positional_codes={positional_codes}  L={L}")
    print(f"  expected N(L)        = {expected_n:,}")
    print(f"  actual rows          = {n_rows:,}  (match: {match})")
    print(f"  X.shape              = {tuple(X.shape)}")
    print(f"  X.nbytes             = {X.nbytes:,} bytes ({X.nbytes / 1e9:.2f} GB)")
    print(f"  y_feature.nbytes     = {y_feature.nbytes:,} bytes")
    print(f"  y_position.nbytes    = {y_position.nbytes:,} bytes")
    print(f"  wall time            = {elapsed_s:.1f} s")
    print(f"  process peak RSS before = {rss_before:,} {rss_unit}")
    print(f"  process peak RSS after  = {rss_after:,} {rss_unit}")

    if not match:
        raise AssertionError(f"Row count mismatch: expected {expected_n:,}, got {n_rows:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positional-codes", type=int, choices=[0, 1], required=True)
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=1800,
        help="Wall-clock timeout for generate_all_examples() (default: 30 min).",
    )
    args = parser.parse_args()

    profile(bool(args.positional_codes), args.timeout_s)
