from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from identifiability_llm.paths import DATA_ROOT  # noqa: E402
from identifiability_llm.ten_task_attention import (  # noqa: E402
    TASK_NAMES,
    initialize_model,
    make_transforms,
    stable_seed,
)
from identifiability_llm.support_projected_svd import (  # noqa: E402
    build_support_projected_svd,
    reconstruct_support_projected_svd,
    support_projected_svd_power,
)
from identifiability_llm.ten_task_effective_score import (  # noqa: E402
    effective_score_matrix,
    effective_score_reconstruction_audit,
    evaluate_cached_effective_score,
    prepare_effective_score_cache,
)


RUN_ID = "ten_task_effective_score_20seeds_v1"
TRAINED_EXTENSION = "trained_low_rank_effective_score_20seeds_v1_steps5000_10000"
EXTENSION_ID = "support_projected_svd_v1"
METHOD = "support_projected_m_svd"
SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE = 1e-10

RESULT_COLUMNS = [
    "seed",
    "task_id",
    "task",
    "method",
    "rank",
    "accuracy",
    "full_model_accuracy",
    "retention",
    "svd_cumulative_power",
    "mean_attention_kl",
    "centered_score_mse",
    "requested_rank",
    "numerical_rank",
    "rank_bound_satisfied",
    "relative_matrix_error",
    "projector_symmetry_error",
    "projector_idempotence_error",
    "reconstruction_target",
    "query_support_rank",
    "key_support_rank",
    "supported_matrix_rank",
    "support_eigenvalue_relative_tolerance",
]


def atomic_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite support-SVD result: {path}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_csv_frame(
    frame: pd.DataFrame, path: Path, *, replace_existing: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or (path.exists() and not replace_existing):
        raise FileExistsError(f"Refusing to overwrite support-SVD table: {path}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(
    payload: Any, path: Path, *, replace_existing: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or (path.exists() and not replace_existing):
        raise FileExistsError(f"Refusing to overwrite support-SVD audit: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _maximum_score_error(
    cache: list[Any], original: torch.Tensor, supported: torch.Tensor
) -> float:
    maximum = 0.0
    for batch in cache:
        original_scores = torch.einsum(
            "bd,de,bne->bn",
            batch.query,
            original.to(device=batch.query.device, dtype=batch.query.dtype),
            batch.context,
        )
        supported_scores = torch.einsum(
            "bd,de,bne->bn",
            batch.query,
            supported.to(device=batch.query.device, dtype=batch.query.dtype),
            batch.context,
        )
        maximum = max(
            maximum,
            float((original_scores - supported_scores).abs().max().item()),
        )
    return maximum


def run_seed(config: dict[str, Any], seed: int) -> None:
    run_id = str(config["experiment"]["run_id"])
    if run_id != RUN_ID:
        raise RuntimeError(f"Expected parent run {RUN_ID}, got {run_id}")
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    result_root = DATA_ROOT / "results" / RUN_ID / "extensions" / EXTENSION_ID
    part_path = result_root / "parts" / f"seed_{seed}.csv"
    audit_path = result_root / "audits" / f"seed_{seed}.json"
    if part_path.exists() and audit_path.exists():
        print(json.dumps({"event": "support_svd_skip_existing", "seed": seed}))
        return
    if part_path.exists() or audit_path.exists():
        raise RuntimeError(
            f"Incomplete atomic output pair for seed {seed}: {part_path}, {audit_path}"
        )

    checkpoint_path = (
        DATA_ROOT
        / "artifacts"
        / RUN_ID
        / "confirmatory/checkpoints"
        / f"seed_{seed}.pt"
    )
    basis_root = (
        DATA_ROOT
        / "artifacts"
        / RUN_ID
        / "confirmatory/four_gram_bases"
        / f"seed_{seed}"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    confirmatory = config["confirmatory"]
    generator = config["generator"]
    model, _ = initialize_model(seed, float(confirmatory["gamma"]))
    model.load_state_dict(checkpoint["final_state_dict"])
    model.eval().to(torch.device("cpu"))
    transforms = make_transforms(seed)
    matrix = effective_score_matrix(model)
    rows: list[dict[str, Any]] = []
    task_audits: dict[str, Any] = {}

    for task_id, task_name in enumerate(TASK_NAMES):
        payload = torch.load(
            basis_root / f"{task_name}.pt", map_location="cpu", weights_only=False
        )
        if not bool(payload["gram_audit"]["passed"]):
            raise AssertionError(f"Cached Gram audit failed for {seed}/{task_name}")
        basis = payload["basis"]
        if not torch.equal(basis.matrix, matrix):
            raise AssertionError(f"Cached M differs from checkpoint M for {seed}/{task_name}")
        supported = build_support_projected_svd(
            basis,
            relative_eigenvalue_tolerance=SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE,
        )
        sensitivity_ranks = {
            str(tolerance): (
                candidate.query_support_rank,
                candidate.key_support_rank,
            )
            for tolerance in (1e-8, 1e-10, 1e-12)
            for candidate in (
                build_support_projected_svd(
                    basis, relative_eigenvalue_tolerance=tolerance
                ),
            )
        }
        query_projector = supported.query_vectors @ supported.query_vectors.T
        key_projector = supported.key_vectors @ supported.key_vectors.T
        identity = torch.eye(query_projector.shape[0], dtype=torch.float64)
        grams = payload["grams"]
        query_support_residual = float(
            torch.linalg.norm((identity - query_projector) @ grams.query_input).item()
            / max(float(torch.linalg.norm(grams.query_input).item()), 1e-12)
        )
        key_support_residual = float(
            torch.linalg.norm(grams.key_input @ (identity - key_projector)).item()
            / max(float(torch.linalg.norm(grams.key_input).item()), 1e-12)
        )
        full_supported = reconstruct_support_projected_svd(supported, 128)
        supported_rank = int(
            (
                supported.singular_values
                > supported.singular_values.max()
                * SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE
            )
            .sum()
            .item()
        )
        test_seed = stable_seed(
            str(config["data"]["confirmatory_test_namespace"]),
            seed,
            "test",
            task_id,
        )
        cache = prepare_effective_score_cache(
            model,
            task_id,
            int(confirmatory["test_count"]),
            int(confirmatory["evaluation_batch_size"]),
            test_seed,
            transforms,
            float(generator["rho"]),
            float(generator["three_vote_distractor_rho"]),
            torch.device("cpu"),
        )
        baseline = evaluate_cached_effective_score(model, cache, matrix)
        full_supported_metrics = evaluate_cached_effective_score(
            model, cache, full_supported
        )
        maximum_score_error = _maximum_score_error(cache, matrix, full_supported)
        expected_accuracy = float(checkpoint["test_accuracy"][task_name])
        checks = {
            "baseline_matches_parent_test_accuracy": abs(
                baseline["accuracy"] - expected_accuracy
            )
            <= 1e-12,
            "full_support_accuracy_matches_raw_m": abs(
                full_supported_metrics["accuracy"] - baseline["accuracy"]
            )
            <= 1e-12,
            "full_support_score_error_small": maximum_score_error <= 2e-5,
            "supported_rank_within_input_supports": supported_rank
            <= min(supported.query_support_rank, supported.key_support_rank),
            "support_ranks_threshold_stable": len(set(sensitivity_ranks.values())) == 1,
            "query_support_residual_small": query_support_residual <= 1e-10,
            "key_support_residual_small": key_support_residual <= 1e-10,
            "query_projector_symmetric": float(
                torch.linalg.norm(query_projector - query_projector.T).item()
            )
            <= 1e-10,
            "key_projector_symmetric": float(
                torch.linalg.norm(key_projector - key_projector.T).item()
            )
            <= 1e-10,
            "query_projector_idempotent": float(
                torch.linalg.norm(query_projector @ query_projector - query_projector).item()
            )
            <= 1e-10,
            "key_projector_idempotent": float(
                torch.linalg.norm(key_projector @ key_projector - key_projector).item()
            )
            <= 1e-10,
        }
        if not all(checks.values()):
            raise AssertionError(f"Support audit failed for {seed}/{task_name}: {checks}")

        for rank in ranks:
            reconstructed = reconstruct_support_projected_svd(supported, rank)
            numerical = effective_score_reconstruction_audit(
                full_supported, reconstructed, None, rank
            )
            if not numerical["rank_bound_satisfied"]:
                raise AssertionError(
                    f"Rank audit failed for {seed}/{task_name}/{METHOD}/{rank}"
                )
            metrics = evaluate_cached_effective_score(model, cache, reconstructed)
            if rank == 128:
                if numerical["relative_matrix_error"] > 1e-10:
                    raise AssertionError("Full support-projected matrix recovery failed")
                if abs(metrics["accuracy"] - baseline["accuracy"]) > 1e-12:
                    raise AssertionError("Full support-projected accuracy recovery failed")
            rows.append(
                {
                    "seed": seed,
                    "task_id": task_id,
                    "task": task_name,
                    "method": METHOD,
                    "rank": rank,
                    "accuracy": metrics["accuracy"],
                    "full_model_accuracy": baseline["accuracy"],
                    "retention": metrics["accuracy"]
                    / max(baseline["accuracy"], 1e-12),
                    "svd_cumulative_power": support_projected_svd_power(
                        supported, rank
                    ),
                    "mean_attention_kl": metrics["mean_attention_kl"],
                    "centered_score_mse": metrics["centered_score_mse"],
                    "requested_rank": numerical["requested_rank"],
                    "numerical_rank": numerical["numerical_rank"],
                    "rank_bound_satisfied": numerical["rank_bound_satisfied"],
                    "relative_matrix_error": numerical["relative_matrix_error"],
                    "projector_symmetry_error": "",
                    "projector_idempotence_error": "",
                    "reconstruction_target": "P_q M P_x",
                    "query_support_rank": supported.query_support_rank,
                    "key_support_rank": supported.key_support_rank,
                    "supported_matrix_rank": supported_rank,
                    "support_eigenvalue_relative_tolerance": (
                        SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE
                    ),
                }
            )
        task_audits[task_name] = {
            "passed": all(checks.values()),
            "checks": checks,
            "query_support_rank": supported.query_support_rank,
            "key_support_rank": supported.key_support_rank,
            "supported_matrix_rank": supported_rank,
            "maximum_full_support_score_absolute_error": maximum_score_error,
            "query_support_residual": query_support_residual,
            "key_support_residual": key_support_residual,
            "threshold_sensitivity_support_ranks": sensitivity_ranks,
            "full_model_accuracy": baseline["accuracy"],
            "full_support_accuracy": full_supported_metrics["accuracy"],
        }

    expected_rows = len(TASK_NAMES) * len(ranks)
    audit = {
        "extension_id": EXTENSION_ID,
        "parent_run_id": RUN_ID,
        "post_hoc": True,
        "seed": seed,
        "method": METHOD,
        "support_definition": "non-null eigenspaces of query-input and key-input Grams",
        "support_eigenvalue_relative_tolerance": (
            SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE
        ),
        "passed": len(rows) == expected_rows
        and all(task["passed"] for task in task_audits.values()),
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "tasks": task_audits,
    }
    if not audit["passed"]:
        raise AssertionError(f"Seed-level support-SVD audit failed: {seed}")
    atomic_csv_rows(rows, part_path)
    atomic_json(audit, audit_path)
    print(
        json.dumps(
            {
                "event": "support_svd_seed_complete",
                "seed": seed,
                "rows": len(rows),
                "part": str(part_path),
            }
        ),
        flush=True,
    )


def _threshold_rank(group: pd.DataFrame, fraction: float) -> int | None:
    baseline = float(group.groupby("seed")["full_model_accuracy"].first().mean())
    mean_curve = group.groupby("rank")["accuracy"].mean().sort_index()
    passing = mean_curve[mean_curve >= fraction * baseline]
    return int(passing.index.min()) if not passing.empty else None


def summarize(config: dict[str, Any], *, replace_derived: bool = False) -> None:
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    result_root = DATA_ROOT / "results" / RUN_ID / "extensions" / EXTENSION_ID
    part_paths = [result_root / "parts" / f"seed_{seed}.csv" for seed in seeds]
    audit_paths = [result_root / "audits" / f"seed_{seed}.json" for seed in seeds]
    missing = [str(path) for path in part_paths + audit_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} support-SVD outputs: {missing[:5]}")
    support = pd.concat([pd.read_csv(path) for path in part_paths], ignore_index=True)
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
    expected_rows = len(seeds) * len(TASK_NAMES) * len(ranks)
    key = ["seed", "task", "method", "rank"]
    checks = {
        "all_seed_audits_pass": all(bool(audit["passed"]) for audit in audits),
        "row_count": len(support) == expected_rows,
        "no_duplicate_conditions": not support.duplicated(key).any(),
        "all_seeds_present": set(support["seed"].astype(int)) == set(seeds),
        "all_tasks_present": set(support["task"]) == set(TASK_NAMES),
        "all_ranks_present": set(support["rank"].astype(int)) == set(ranks),
        "only_support_method_present": set(support["method"]) == {METHOD},
        "all_rank_bounds_satisfied": bool(support["rank_bound_satisfied"].all()),
        "all_key_support_ranks_104": set(support["key_support_rank"].astype(int))
        == {104},
        "all_full_rank_accuracies_recovered": bool(
            (
                support[support["rank"] == 128]["accuracy"]
                - support[support["rank"] == 128]["full_model_accuracy"]
            )
            .abs()
            .le(1e-12)
            .all()
        ),
        "all_full_rank_power_one": bool(
            support[support["rank"] == 128]["svd_cumulative_power"]
            .sub(1.0)
            .abs()
            .le(1e-12)
            .all()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"Aggregate support-SVD audit failed: {checks}")

    old_combined_path = (
        DATA_ROOT
        / "results"
        / RUN_ID
        / "extensions"
        / TRAINED_EXTENSION
        / "tables/analytic_and_trained_reconstruction_results.csv"
    )
    old_combined = pd.read_csv(old_combined_path, low_memory=False)
    raw_svd_count = int((old_combined["method"] == "effective_m_svd").sum())
    if raw_svd_count != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} original-SVD rows, found {raw_svd_count}"
        )
    figure_table = pd.concat(
        [old_combined, support],
        ignore_index=True,
        sort=False,
    )
    checks.update(
        {
            "figure_row_count": len(figure_table)
            == len(old_combined) + expected_rows,
            "figure_retains_original_svd": int(
                (figure_table["method"] == "effective_m_svd").sum()
            )
            == expected_rows,
            "figure_contains_support_svd": int(
                (figure_table["method"] == METHOD).sum()
            )
            == expected_rows,
            "figure_has_no_duplicate_conditions": not figure_table.duplicated(
                ["seed", "task", "method", "rank"]
            ).any(),
        }
    )
    if not all(checks.values()):
        raise AssertionError(f"Figure-table support-SVD audit failed: {checks}")
    threshold_rows = []
    for task, group in support.groupby("task", sort=False):
        threshold_rows.append(
            {
                "task": task,
                "method": METHOD,
                "K95": _threshold_rank(group, 0.95),
                "K99": _threshold_rank(group, 0.99),
                "mean_full_model_accuracy": float(
                    group.groupby("seed")["full_model_accuracy"].first().mean()
                ),
            }
        )
    thresholds = pd.DataFrame(threshold_rows)
    table_root = result_root / "tables"
    atomic_csv_frame(
        support,
        table_root / "support_projected_svd_results.csv",
        replace_existing=replace_derived,
    )
    atomic_csv_frame(
        figure_table,
        table_root / "figure_reconstruction_results.csv",
        replace_existing=replace_derived,
    )
    atomic_csv_frame(
        thresholds,
        table_root / "k95_k99.csv",
        replace_existing=replace_derived,
    )
    aggregate = {
        "extension_id": EXTENSION_ID,
        "parent_run_id": RUN_ID,
        "post_hoc": True,
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "support_rows": len(support),
            "retained_original_svd_rows": raw_svd_count,
            "figure_rows": len(figure_table),
            "seed_audits": len(audits),
        },
        "support_eigenvalue_relative_tolerance": (
            SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE
        ),
    }
    atomic_json(
        aggregate,
        table_root / "aggregate_audit.json",
        replace_existing=replace_derived,
    )
    print(json.dumps({"event": "support_svd_summary_complete", **aggregate}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/experiment.toml"
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--replace-derived", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    with arguments.config.resolve().open("rb") as stream:
        config = tomllib.load(stream)
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    selected = arguments.seed or seeds
    if not set(selected).issubset(set(seeds)):
        raise ValueError("Requested seed is outside the confirmatory seed set")
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "event": "support_svd_runner_validated",
                    "extension_id": EXTENSION_ID,
                    "seeds": selected,
                    "ranks": [int(value) for value in config["sweep"]["ranks"]],
                    "relative_eigenvalue_tolerance": (
                        SUPPORT_EIGENVALUE_RELATIVE_TOLERANCE
                    ),
                },
                indent=2,
            )
        )
        return
    if arguments.summarize:
        if arguments.seed:
            raise ValueError("--summarize cannot be combined with --seed")
        summarize(config, replace_derived=arguments.replace_derived)
        return
    if arguments.replace_derived:
        raise ValueError("--replace-derived requires --summarize")
    for seed in selected:
        run_seed(config, seed)


if __name__ == "__main__":
    main()
