from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [
    4103,
    5209,
    6311,
    7421,
    8537,
    9643,
    10753,
    11863,
    12973,
    14081,
    15187,
    16291,
    17393,
    18499,
    19603,
    20707,
    21817,
    22921,
    24029,
    25147,
]
EXPECTED_TASKS = list(range(10))
EXPECTED_RANKS = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
EXPECTED_HORIZONS = [5000, 10000]
TRAINABLE_RANKS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {path}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def threshold_rank(curve: pd.DataFrame, fraction: float) -> int | None:
    baseline = float(curve["full_model_accuracy"].mean())
    for rank, accuracy in (
        curve.groupby("rank", as_index=False)["accuracy"].mean().sort_values("rank")
        [["rank", "accuracy"]]
        .itertuples(index=False, name=None)
    ):
        if float(accuracy) >= fraction * baseline:
            return int(rank)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-root", type=Path, required=True)
    parser.add_argument("--audits-root", type=Path, required=True)
    parser.add_argument("--analytic-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    part_paths = sorted(arguments.parts_root.resolve().glob("seed_*/task_*/effective_score.csv"))
    audit_paths = sorted(arguments.audits_root.resolve().glob("seed_*/task_*/effective_score.json"))
    expected_groups = len(EXPECTED_SEEDS) * len(EXPECTED_TASKS)
    if len(part_paths) != expected_groups or len(audit_paths) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} parts and audits, found "
            f"{len(part_paths)} and {len(audit_paths)}"
        )
    trained = pd.concat([pd.read_csv(path) for path in part_paths], ignore_index=True)
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
    key = ["seed", "task_id", "method", "rank", "optimization_horizon"]
    expected_rows = len(EXPECTED_SEEDS) * len(EXPECTED_TASKS) * 2 * len(EXPECTED_RANKS)
    checks = {
        "row_count": len(trained) == expected_rows,
        "no_duplicate_conditions": not trained.duplicated(key).any(),
        "all_seeds_present": sorted(trained["seed"].unique().tolist()) == EXPECTED_SEEDS,
        "all_tasks_present": sorted(trained["task_id"].unique().tolist()) == EXPECTED_TASKS,
        "all_ranks_present": sorted(trained["rank"].unique().tolist()) == EXPECTED_RANKS,
        "all_horizons_present": sorted(trained["optimization_horizon"].unique().tolist())
        == EXPECTED_HORIZONS,
        "all_group_audits_pass": all(bool(audit["passed"]) for audit in audits),
        "all_rank_bounds_satisfied": bool(trained["rank_bound_satisfied"].all()),
        "all_terminal_rank_bounds_satisfied": bool(
            trained["terminal_rank_bound_satisfied"].all()
        ),
        "all_teachers_frozen": bool(trained["all_teacher_parameters_frozen"].all()),
        "all_teacher_gradients_absent": bool(
            trained["all_teacher_gradients_absent"].all()
        ),
        "all_teacher_parameters_unchanged": bool(
            trained["teacher_parameters_unchanged"].all()
        ),
        "all_trajectories_reach_10000": all(
            int(audit["completed_steps"]) == 10000 for audit in audits
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Aggregate audit failed: {checks}")

    analytic = pd.read_csv(arguments.analytic_table.resolve())
    trained_plot = trained[
        [
            "seed",
            "task_id",
            "task",
            "method",
            "rank",
            "accuracy",
            "full_model_accuracy",
            "retention",
            "svd_cumulative_power",
        ]
    ].copy()
    combined = pd.concat([analytic, trained_plot], ignore_index=True, sort=False)

    paired = trained[trained["rank"].isin(TRAINABLE_RANKS)].pivot(
        index=["seed", "task_id", "task", "rank"],
        columns="optimization_horizon",
        values=[
            "accuracy",
            "final_normalized_logit_mse",
            "selected_step",
        ],
    )
    paired.columns = [f"{name}_{int(horizon)}" for name, horizon in paired.columns]
    paired = paired.reset_index()
    paired["accuracy_delta_10k_minus_5k"] = (
        paired["accuracy_10000"] - paired["accuracy_5000"]
    )
    paired["normalized_logit_mse_delta_10k_minus_5k"] = (
        paired["final_normalized_logit_mse_10000"]
        - paired["final_normalized_logit_mse_5000"]
    )
    tolerance = 1e-12
    paired["accuracy_outcome"] = "tie"
    paired.loc[
        paired["accuracy_delta_10k_minus_5k"] > tolerance, "accuracy_outcome"
    ] = "win"
    paired.loc[
        paired["accuracy_delta_10k_minus_5k"] < -tolerance, "accuracy_outcome"
    ] = "loss"

    thresholds: list[dict[str, Any]] = []
    for task, task_data in trained.groupby("task", sort=False):
        for horizon, horizon_data in task_data.groupby("optimization_horizon"):
            thresholds.append(
                {
                    "task": task,
                    "optimization_horizon": int(horizon),
                    "method": f"trained_low_rank_{int(horizon // 1000)}k",
                    "K95": threshold_rank(horizon_data, 0.95),
                    "K99": threshold_rank(horizon_data, 0.99),
                }
            )
    threshold_frame = pd.DataFrame(thresholds)
    threshold_paired = threshold_frame.pivot(
        index="task", columns="optimization_horizon", values=["K95", "K99"]
    )
    threshold_paired.columns = [f"{metric}_{int(horizon)}" for metric, horizon in threshold_paired.columns]
    threshold_paired = threshold_paired.reset_index()
    threshold_paired["K95_savings_10k_vs_5k"] = (
        threshold_paired["K95_5000"] - threshold_paired["K95_10000"]
    )
    threshold_paired["K99_savings_10k_vs_5k"] = (
        threshold_paired["K99_5000"] - threshold_paired["K99_10000"]
    )

    outcome_counts = paired["accuracy_outcome"].value_counts().to_dict()
    scientific_summary = {
        "conditions_compared": int(len(paired)),
        "accuracy_wins_10k": int(outcome_counts.get("win", 0)),
        "accuracy_ties": int(outcome_counts.get("tie", 0)),
        "accuracy_losses_10k": int(outcome_counts.get("loss", 0)),
        "mean_accuracy_delta_10k_minus_5k": float(
            paired["accuracy_delta_10k_minus_5k"].mean()
        ),
        "median_accuracy_delta_10k_minus_5k": float(
            paired["accuracy_delta_10k_minus_5k"].median()
        ),
        "mean_normalized_logit_mse_delta_10k_minus_5k": float(
            paired["normalized_logit_mse_delta_10k_minus_5k"].mean()
        ),
        "conditions_with_lower_10k_selected_loss": int(
            (paired["normalized_logit_mse_delta_10k_minus_5k"] < -1e-12).sum()
        ),
        "mean_K95_5k": float(threshold_frame[threshold_frame["optimization_horizon"] == 5000]["K95"].mean()),
        "mean_K95_10k": float(threshold_frame[threshold_frame["optimization_horizon"] == 10000]["K95"].mean()),
        "mean_K99_5k": float(threshold_frame[threshold_frame["optimization_horizon"] == 5000]["K99"].mean()),
        "mean_K99_10k": float(threshold_frame[threshold_frame["optimization_horizon"] == 10000]["K99"].mean()),
        "tasks_with_lower_K95_at_10k": int((threshold_paired["K95_savings_10k_vs_5k"] > 0).sum()),
        "tasks_with_lower_K99_at_10k": int((threshold_paired["K99_savings_10k_vs_5k"] > 0).sum()),
    }

    output = arguments.output.resolve()
    atomic_csv(trained, output / "trained_reconstruction_results.csv")
    atomic_csv(combined, output / "analytic_and_trained_reconstruction_results.csv")
    atomic_csv(paired, output / "paired_5k_vs_10k_conditions.csv")
    atomic_csv(threshold_frame, output / "trained_k95_k99.csv")
    atomic_csv(threshold_paired, output / "paired_5k_vs_10k_k95_k99.csv")
    atomic_json(scientific_summary, output / "scientific_summary.json")
    audit_payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "parts": len(part_paths),
            "group_audits": len(audit_paths),
            "trained_rows": len(trained),
            "combined_rows": len(combined),
            "paired_trainable_conditions": len(paired),
        },
    }
    atomic_json(audit_payload, output / "aggregate_audit.json")
    print(
        json.dumps(
            {
                "event": "effective_score_distillation_summary_complete",
                "output": str(output),
                "audit": audit_payload,
                "scientific_summary": scientific_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
