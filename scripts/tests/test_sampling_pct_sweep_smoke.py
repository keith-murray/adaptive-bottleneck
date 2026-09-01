"""End-to-end smoke test for run_sampling_pct_sweep_for_combo: small L, tiny
schedules, writes to a scratch directory (not the repo's data/ dir)."""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "context_identification" / "pos_codes"))
from sampling_pct_pos_codes_sweep import run_sampling_pct_sweep_for_combo, run_filename  # noqa: E402


def main():
    data_dir = Path("/private/tmp/claude-501/-Users-ktmurray-Projects-adaptive-bottleneck/1d8f380a-c4ab-46cc-a174-c2335926e4a6/scratchpad/sampling_pct_smoke")
    if data_dir.exists():
        shutil.rmtree(data_dir)

    schedules = {"log": (0.01, 0.5, 1.0), "linear": (0.5, 1.0)}
    seeds = (0, 1)
    condition, label_mode = "adaptive_pos", "position"  # exercises positional_codes=True

    run_sampling_pct_sweep_for_combo(
        condition=condition,
        label_mode=label_mode,
        seeds=seeds,
        schedules=schedules,
        L=6,
        batch_size=8,
        num_steps=20,
        eval_every=5,
        max_eval_batches=2,
        lr=1e-2,
        weight_decay=1e-3,
        gamma_lr=1e-2,
        init_gamma=0.5,
        data_dir=data_dir,
        show_progress=False,
        show_combo_progress=False,
    )

    expected_files = []
    for schedule_name, pcts in schedules.items():
        for pct_idx, train_frac in enumerate(pcts):
            for seed in seeds:
                expected_files.append(run_filename(schedule_name, pct_idx, condition, label_mode, seed))

    actual_files = sorted(p.name for p in data_dir.glob("*.json"))
    assert actual_files == sorted(expected_files), (actual_files, expected_files)
    print(f"{len(actual_files)} result files written, all names match expected.")

    for schedule_name, pcts in schedules.items():
        for pct_idx, train_frac in enumerate(pcts):
            for seed in seeds:
                path = data_dir / run_filename(schedule_name, pct_idx, condition, label_mode, seed)
                result = json.loads(path.read_text())

                assert result["condition"] == condition
                assert result["label_mode"] == label_mode
                assert result["schedule"] == schedule_name
                assert result["pct_idx"] == pct_idx
                assert abs(result["train_frac"] - train_frac) < 1e-9
                assert result["seed"] == seed
                assert len(result["history"]["step"]) == 20

                if train_frac == 1.0:
                    assert result["n_test"] == 0
                    assert result["history"]["test_step"] == []
                else:
                    assert result["n_test"] > 0
                    assert len(result["history"]["test_step"]) == 20 // 5

    print("All result files verified: correct metadata, history length, "
          "and train_frac=1.0 correctly omits the test set.")
    print("\nSanity check passed!")


if __name__ == "__main__":
    main()
