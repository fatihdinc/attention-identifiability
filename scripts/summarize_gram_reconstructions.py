from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from identifiability_llm.ten_task_attention import TASK_NAMES  # noqa: E402
from identifiability_llm.ten_task_effective_score import (  # noqa: E402
    ALL_EFFECTIVE_SCORE_METHODS,
)


def atomic_csv_create(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite final table: {path}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json_create(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite final audit: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def minimum_rank_table(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task, method), group in primary.groupby(["task", "method"], sort=True):
        mean_by_rank = group.groupby("rank")["accuracy"].mean().sort_index()
        baseline = float(group.groupby("seed")["full_model_accuracy"].first().mean())
        row: dict[str, Any] = {
            "task": task,
            "method": method,
            "mean_full_model_accuracy": baseline,
        }
        for label, fraction in (("K95", 0.95), ("K99", 0.99)):
            passing = mean_by_rank[mean_by_rank >= fraction * baseline]
            row[label] = int(passing.index.min()) if not passing.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    result_root = arguments.result_root.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    methods = [str(value) for value in config["sweep"]["methods"]]
    if methods != list(ALL_EFFECTIVE_SCORE_METHODS):
        raise RuntimeError("Method list differs from implementation")
    part_root = result_root / "parts"
    primary_paths = [
        part_root / f"seed_{seed}" / f"{task}.csv"
        for seed in seeds
        for task in TASK_NAMES
    ]
    control_paths = [
        part_root / f"seed_{seed}" / f"{task}_controls.csv"
        for seed in seeds
        for task in TASK_NAMES
    ]
    missing = [str(path) for path in primary_paths + control_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} analysis parts: {missing[:5]}")
    primary = pd.concat([pd.read_csv(path) for path in primary_paths], ignore_index=True)
    controls = pd.concat([pd.read_csv(path) for path in control_paths], ignore_index=True)
    training_rows: list[dict[str, Any]] = []
    run_id = str(config["experiment"]["run_id"])
    for seed in seeds:
        training_path = result_root / "training" / f"seed_{seed}.json"
        training = json.loads(training_path.read_text(encoding="utf-8"))
        for task in TASK_NAMES:
            training_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "validation_accuracy": training["validation_accuracy"][task],
                    "test_accuracy": training["test_accuracy"][task],
                    "task_batch_count": training["task_batch_counts"][task],
                    "task_example_count": training["task_example_counts"][task],
                    "outer_updates_completed": training["outer_updates_completed"],
                    "device": training["device"]["selected"],
                }
            )
    training_frame = pd.DataFrame(training_rows)
    k_table = minimum_rank_table(primary)
    mean_curves = (
        primary.groupby(["task", "method", "rank"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            retention_mean=("retention", "mean"),
            retention_std=("retention", "std"),
            attention_kl_mean=("mean_attention_kl", "mean"),
            centered_score_mse_mean=("centered_score_mse", "mean"),
            svd_cumulative_power_mean=("svd_cumulative_power", "mean"),
            svd_cumulative_power_std=("svd_cumulative_power", "std"),
        )
        .sort_values(["task", "method", "rank"])
    )
    table_root = result_root / "tables"
    atomic_csv_create(primary, table_root / "reconstruction_results.csv")
    atomic_csv_create(controls, table_root / "control_results.csv")
    atomic_csv_create(training_frame, table_root / "training_accuracies.csv")
    atomic_csv_create(k_table, table_root / "k95_k99.csv")
    atomic_csv_create(mean_curves, table_root / "mean_rank_curves.csv")
    expected_primary_rows = len(seeds) * len(TASK_NAMES) * len(methods) * len(ranks)
    expected_control_rows = len(seeds) * len(TASK_NAMES) * 16
    duplicates = int(
        primary.duplicated(subset=["seed", "task", "method", "rank"]).sum()
    )
    full_rank = primary[primary["rank"] == 128]
    rank_zero = primary[primary["rank"] == 0]
    audit_paths = [result_root / "audits" / f"seed_{seed}.json" for seed in seeds]
    audit_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
    checks = {
        "all_primary_rows_present": len(primary) == expected_primary_rows,
        "all_control_rows_present": len(controls) == expected_control_rows,
        "no_duplicate_primary_conditions": duplicates == 0,
        "all_tasks_present": set(primary["task"]) == set(TASK_NAMES),
        "all_methods_present": set(primary["method"]) == set(methods),
        "all_ranks_present": set(primary["rank"].astype(int)) == set(ranks),
        "all_seeds_present": set(primary["seed"].astype(int)) == set(seeds),
        "all_rank_bounds_satisfied": bool(primary["rank_bound_satisfied"].all()),
        "rank_zero_conditions_complete": len(rank_zero)
        == len(seeds) * len(TASK_NAMES) * len(methods),
        "full_rank_conditions_complete": len(full_rank)
        == len(seeds) * len(TASK_NAMES) * len(methods),
        "full_rank_matrices_recovered": bool(
            (full_rank["relative_matrix_error"] <= 1e-10).all()
        ),
        "full_rank_accuracies_recovered": bool(
            ((full_rank["accuracy"] - full_rank["full_model_accuracy"]).abs() <= 1e-12).all()
        ),
        "all_seed_audits_pass": all(payload["passed"] for payload in audit_payloads),
        "equal_task_batch_exposure": bool(
            (training_frame["task_batch_count"] == int(config["confirmatory"]["outer_updates"])).all()
        ),
        "equal_task_example_exposure": bool(
            (
                training_frame["task_example_count"]
                == int(config["confirmatory"]["outer_updates"])
                * int(config["confirmatory"]["task_batch_size"])
            ).all()
        ),
        "no_device_fallbacks": set(training_frame["device"])
        == {str(config["experiment"]["device_order"][0])},
    }
    correctness = {
        "run_id": run_id,
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "primary_rows": len(primary),
            "expected_primary_rows": expected_primary_rows,
            "control_rows": len(controls),
            "expected_control_rows": expected_control_rows,
            "training_rows": len(training_frame),
            "duplicate_primary_conditions": duplicates,
        },
        "diagnostics": {
            "maximum_full_rank_matrix_relative_error": float(
                full_rank["relative_matrix_error"].max()
            ),
            "maximum_projector_symmetry_error": float(
                pd.to_numeric(primary["projector_symmetry_error"], errors="coerce").max()
            ),
            "maximum_projector_idempotence_error": float(
                pd.to_numeric(primary["projector_idempotence_error"], errors="coerce").max()
            ),
            "mean_teacher_test_accuracy": float(training_frame["test_accuracy"].mean()),
        },
    }
    atomic_json_create(correctness, table_root / "correctness_audit.json")
    print(json.dumps({"event": "effective_score_summary_complete", **correctness}, indent=2))
    if not correctness["passed"]:
        raise RuntimeError(f"Correctness audit failed: {checks}")


if __name__ == "__main__":
    main()
