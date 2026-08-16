from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from identifiability_llm.paths import DATA_ROOT  # noqa: E402
from identifiability_llm.ten_task_analysis import evaluate_control_accuracy  # noqa: E402
from identifiability_llm.ten_task_attention import (  # noqa: E402
    TASK_NAMES,
    generate_task_batch,
    initialize_model,
    make_transforms,
    stable_seed,
)
from identifiability_llm.ten_task_effective_score import (  # noqa: E402
    ALL_EFFECTIVE_SCORE_METHODS,
    FOUR_GRAM_METHODS,
    EffectiveScoreBasis,
    build_effective_score_basis,
    collect_effective_score_grams,
    cumulative_svd_power,
    direct_score_audit,
    effective_score_gram_audit,
    effective_score_matrix,
    effective_score_reconstruction_audit,
    evaluate_cached_effective_score,
    gauge_invariance_audit,
    prepare_effective_score_cache,
    random_effective_score_reconstruction,
    reconstruct_effective_score,
)


PRIMARY_COLUMNS = [
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
]

CONTROL_COLUMNS = [
    "seed",
    "evaluation_task",
    "source_task",
    "control",
    "method",
    "rank",
    "accuracy",
    "full_model_accuracy",
    "retention",
]


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def resolve_device(order: list[str]) -> tuple[torch.device, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for name in order:
        available = (
            bool(torch.backends.mps.is_available())
            if name == "mps"
            else bool(torch.cuda.is_available())
            if name == "cuda"
            else name == "cpu"
        )
        attempts.append({"device": name, "available": available})
        if available:
            return torch.device(name), {
                "selected": name,
                "fallback": name != order[0],
                "attempts": attempts,
            }
    raise RuntimeError(f"No requested device is available: {attempts}")


def atomic_csv_create(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite analysis part: {path}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json_create(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch_create(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite basis: {path}")
    torch.save(payload, temporary)
    temporary.replace(path)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    preregistration_path = arguments.preregistration.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if not preregistration.get("locked_before_confirmatory_data"):
        raise RuntimeError("Protocol is not locked")
    if preregistration["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("Config differs from preregistration")
    methods = [str(value) for value in config["sweep"]["methods"]]
    if methods != list(ALL_EFFECTIVE_SCORE_METHODS):
        raise RuntimeError(f"Unexpected method order: {methods}")
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    control_ranks = [int(value) for value in config["sweep"]["control_ranks"]]
    primary_seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    replacement_seeds = [int(value) for value in config["experiment"]["replacement_seeds"]]
    selected_seeds = arguments.seed or primary_seeds
    if not set(selected_seeds).issubset(set(primary_seeds) | set(replacement_seeds)):
        raise ValueError("Requested seed was not preregistered")
    run_id = str(config["experiment"]["run_id"])
    checkpoint_root = DATA_ROOT / "artifacts" / run_id / "confirmatory/checkpoints"
    basis_root = DATA_ROOT / "artifacts" / run_id / "confirmatory/four_gram_bases"
    part_root = DATA_ROOT / "results" / run_id / "confirmatory/parts"
    audit_root = DATA_ROOT / "results" / run_id / "confirmatory/audits"
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "event": "effective_score_analysis_runner_validated",
                    "seeds": selected_seeds,
                    "methods": methods,
                    "ranks": ranks,
                },
                indent=2,
            )
        )
        return
    missing = [
        seed for seed in selected_seeds if not (checkpoint_root / f"seed_{seed}.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing teacher checkpoints for seeds: {missing}")
    device, device_record = resolve_device(
        [str(value) for value in config["experiment"]["device_order"]]
    )
    confirmatory = config["confirmatory"]
    generator_config = config["generator"]
    suite_started = time.monotonic()
    for seed_index, model_seed in enumerate(selected_seeds, start=1):
        checkpoint = torch.load(
            checkpoint_root / f"seed_{model_seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        model, _ = initialize_model(model_seed, float(confirmatory["gamma"]))
        model.load_state_dict(checkpoint["final_state_dict"])
        model.to(device).eval()
        transforms = make_transforms(model_seed)
        matrix = effective_score_matrix(model)
        bases: dict[int, EffectiveScoreBasis] = {}
        task_gram_audits: dict[str, Any] = {}
        for task_id, task_name in enumerate(TASK_NAMES):
            basis_path = basis_root / f"seed_{model_seed}" / f"{task_name}.pt"
            if basis_path.exists():
                payload = torch.load(basis_path, map_location="cpu", weights_only=False)
                bases[task_id] = payload["basis"]
                task_gram_audits[task_name] = payload["gram_audit"]
                continue
            calibration_seed = stable_seed(
                str(config["data"]["confirmatory_calibration_namespace"]),
                model_seed,
                "calibration",
                task_id,
            )
            grams = collect_effective_score_grams(
                model,
                task_id,
                int(confirmatory["calibration_count"]),
                int(confirmatory["evaluation_batch_size"]),
                calibration_seed,
                transforms,
                float(generator_config["rho"]),
                float(generator_config["three_vote_distractor_rho"]),
            )
            gram_audit = effective_score_gram_audit(matrix, grams)
            if not gram_audit["passed"]:
                raise AssertionError(
                    f"Gram audit failed for seed {model_seed}, task {task_name}: {gram_audit}"
                )
            basis = build_effective_score_basis(matrix, grams)
            bases[task_id] = basis
            task_gram_audits[task_name] = gram_audit
            atomic_torch_create(
                {
                    "run_id": run_id,
                    "seed": model_seed,
                    "task_id": task_id,
                    "task": task_name,
                    "calibration_seed": calibration_seed,
                    "grams": grams,
                    "basis": basis,
                    "gram_audit": gram_audit,
                    "config_sha256": file_sha256(config_path),
                    "preregistration_sha256": file_sha256(preregistration_path),
                },
                basis_path,
            )
        audit_batch = generate_task_batch(
            0,
            64,
            stable_seed("effective_score_confirmatory_audit", model_seed),
            transforms,
            float(generator_config["rho"]),
            float(generator_config["three_vote_distractor_rho"]),
        )
        gauge_audit = gauge_invariance_audit(
            model,
            audit_batch.context,
            audit_batch.query,
            stable_seed("effective_score_confirmatory_gauge", model_seed),
        )
        if not gauge_audit["passed"]:
            raise AssertionError(f"Gauge audit failed for seed {model_seed}: {gauge_audit}")
        direct_audits: dict[str, Any] = {}
        for task_id, task_name in enumerate(TASK_NAMES):
            primary_part = part_root / f"seed_{model_seed}" / f"{task_name}.csv"
            control_part = part_root / f"seed_{model_seed}" / f"{task_name}_controls.csv"
            task_started = time.monotonic()
            test_seed = stable_seed(
                str(config["data"]["confirmatory_test_namespace"]),
                model_seed,
                "test",
                task_id,
            )
            audit_example = generate_task_batch(
                task_id,
                64,
                stable_seed(test_seed, "direct_audit"),
                transforms,
                float(generator_config["rho"]),
                float(generator_config["three_vote_distractor_rho"]),
            )
            direct = direct_score_audit(
                model, audit_example.context, audit_example.query, device=device
            )
            direct_audits[task_name] = direct
            if not direct["passed"]:
                raise AssertionError(
                    f"Direct score audit failed for seed {model_seed}, task {task_name}: {direct}"
                )
            if primary_part.exists() and control_part.exists():
                print(
                    json.dumps(
                        {
                            "event": "effective_score_analysis_skip_existing",
                            "seed": model_seed,
                            "task": task_name,
                        }
                    ),
                    flush=True,
                )
                continue
            cache = prepare_effective_score_cache(
                model,
                task_id,
                int(confirmatory["test_count"]),
                int(confirmatory["evaluation_batch_size"]),
                test_seed,
                transforms,
                float(generator_config["rho"]),
                float(generator_config["three_vote_distractor_rho"]),
                device,
            )
            baseline_metrics = evaluate_cached_effective_score(model, cache, matrix)
            expected_accuracy = float(checkpoint["test_accuracy"][task_name])
            if abs(baseline_metrics["accuracy"] - expected_accuracy) > 1e-12:
                raise AssertionError(
                    f"Direct-M test accuracy mismatch for {model_seed}/{task_name}: "
                    f"{baseline_metrics['accuracy']} vs {expected_accuracy}"
                )
            basis = bases[task_id]
            primary_rows: list[dict[str, Any]] = []
            for method in methods:
                for rank in ranks:
                    reconstructed, projector, _ = reconstruct_effective_score(
                        basis, method, rank
                    )
                    numerical = effective_score_reconstruction_audit(
                        matrix, reconstructed, projector, rank
                    )
                    if not numerical["rank_bound_satisfied"]:
                        raise AssertionError(
                            f"Rank audit failed for {model_seed}/{task_name}/{method}/{rank}"
                        )
                    metrics = evaluate_cached_effective_score(model, cache, reconstructed)
                    if rank == 128:
                        if numerical["relative_matrix_error"] > 1e-10:
                            raise AssertionError("Full-rank matrix recovery failed")
                        if abs(metrics["accuracy"] - baseline_metrics["accuracy"]) > 1e-12:
                            raise AssertionError("Full-rank accuracy recovery failed")
                    primary_rows.append(
                        {
                            "seed": model_seed,
                            "task_id": task_id,
                            "task": task_name,
                            "method": method,
                            "rank": rank,
                            "accuracy": metrics["accuracy"],
                            "full_model_accuracy": baseline_metrics["accuracy"],
                            "retention": metrics["accuracy"]
                            / max(baseline_metrics["accuracy"], 1e-12),
                            "svd_cumulative_power": cumulative_svd_power(basis, rank),
                            "mean_attention_kl": metrics["mean_attention_kl"],
                            "centered_score_mse": metrics["centered_score_mse"],
                            "requested_rank": numerical["requested_rank"],
                            "numerical_rank": numerical["numerical_rank"],
                            "rank_bound_satisfied": numerical["rank_bound_satisfied"],
                            "relative_matrix_error": numerical["relative_matrix_error"],
                            "projector_symmetry_error": numerical.get(
                                "projector_symmetry_error", ""
                            ),
                            "projector_idempotence_error": numerical.get(
                                "projector_idempotence_error", ""
                            ),
                        }
                    )
            control_rows: list[dict[str, Any]] = []
            for side in ("left", "right"):
                for rank in control_ranks:
                    reconstructed, _ = random_effective_score_reconstruction(
                        matrix,
                        rank,
                        stable_seed(
                            "effective_score_random_projector",
                            model_seed,
                            task_id,
                            side,
                            rank,
                        ),
                        side,
                    )
                    metrics = evaluate_cached_effective_score(model, cache, reconstructed)
                    control_rows.append(
                        {
                            "seed": model_seed,
                            "evaluation_task": task_name,
                            "source_task": "",
                            "control": f"matched_random_{side}_projector",
                            "method": "",
                            "rank": rank,
                            "accuracy": metrics["accuracy"],
                            "full_model_accuracy": baseline_metrics["accuracy"],
                            "retention": metrics["accuracy"]
                            / max(baseline_metrics["accuracy"], 1e-12),
                        }
                    )
            shuffled_task_id = (task_id + 1) % len(TASK_NAMES)
            shuffled_basis = bases[shuffled_task_id]
            for method in FOUR_GRAM_METHODS:
                for rank in control_ranks:
                    reconstructed, _, _ = reconstruct_effective_score(
                        shuffled_basis, method, rank
                    )
                    metrics = evaluate_cached_effective_score(model, cache, reconstructed)
                    control_rows.append(
                        {
                            "seed": model_seed,
                            "evaluation_task": task_name,
                            "source_task": TASK_NAMES[shuffled_task_id],
                            "control": "shuffled_calibration_task_assignment",
                            "method": method,
                            "rank": rank,
                            "accuracy": metrics["accuracy"],
                            "full_model_accuracy": baseline_metrics["accuracy"],
                            "retention": metrics["accuracy"]
                            / max(baseline_metrics["accuracy"], 1e-12),
                        }
                    )
            for control in (
                "query_task_code_zero",
                "query_task_code_shuffle",
                "context_label_permutation",
                "query_only",
            ):
                accuracy = evaluate_control_accuracy(
                    model,
                    task_id,
                    int(confirmatory["test_count"]),
                    int(confirmatory["evaluation_batch_size"]),
                    test_seed,
                    transforms,
                    float(generator_config["rho"]),
                    device,
                    float(generator_config["three_vote_distractor_rho"]),
                    control,
                )
                control_rows.append(
                    {
                        "seed": model_seed,
                        "evaluation_task": task_name,
                        "source_task": "",
                        "control": control,
                        "method": "",
                        "rank": "",
                        "accuracy": accuracy,
                        "full_model_accuracy": baseline_metrics["accuracy"],
                        "retention": accuracy / max(baseline_metrics["accuracy"], 1e-12),
                    }
                )
            atomic_csv_create(primary_rows, PRIMARY_COLUMNS, primary_part)
            atomic_csv_create(control_rows, CONTROL_COLUMNS, control_part)
            del cache
            clear_device_cache(device)
            completed_parts = len(list(part_root.glob("seed_*/*.csv")))
            print(
                json.dumps(
                    {
                        "event": "effective_score_task_analysis_complete",
                        "seed": model_seed,
                        "task": task_name,
                        "baseline_accuracy": baseline_metrics["accuracy"],
                        "primary_rows": len(primary_rows),
                        "control_rows": len(control_rows),
                        "elapsed_seconds": time.monotonic() - task_started,
                        "completed_csv_parts": completed_parts,
                        "expected_csv_parts": len(selected_seeds) * len(TASK_NAMES) * 2,
                    }
                ),
                flush=True,
            )
        audit_payload = {
            "run_id": run_id,
            "seed": model_seed,
            "passed": gauge_audit["passed"]
            and all(record["passed"] for record in task_gram_audits.values())
            and all(record["passed"] for record in direct_audits.values()),
            "device": device_record,
            "gauge_audit": gauge_audit,
            "gram_audits": task_gram_audits,
            "direct_score_audits": direct_audits,
            "config_sha256": file_sha256(config_path),
            "preregistration_sha256": file_sha256(preregistration_path),
        }
        audit_path = audit_root / f"seed_{model_seed}.json"
        if not audit_path.exists():
            atomic_json_create(audit_payload, audit_path)
        if not audit_payload["passed"]:
            raise AssertionError(f"Seed audit failed: {audit_payload}")
        print(
            json.dumps(
                {
                    "event": "effective_score_seed_analysis_complete",
                    "seed": model_seed,
                    "seed_index": seed_index,
                    "seed_count": len(selected_seeds),
                    "suite_elapsed_seconds": time.monotonic() - suite_started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
