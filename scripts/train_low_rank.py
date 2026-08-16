from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time
import tomllib
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from identifiability_llm.paths import DATA_ROOT  # noqa: E402
from identifiability_llm.ten_task_attention import (  # noqa: E402
    TASK_NAMES,
    initialize_model,
    make_transforms,
    parameter_hashes,
    stable_seed,
)
from identifiability_llm.ten_task_distillation import (  # noqa: E402
    DistillationTrainingConfig,
    EFFECTIVE_SCORE_MATRIX,
    FunctionalDistillationData,
    build_functional_distillation_data,
    evaluate_factor_metrics,
    replacement_from_factors,
    svd_factor_initialization,
    train_rank_group,
)
from identifiability_llm.ten_task_effective_score import (  # noqa: E402
    cumulative_svd_power,
    effective_score_matrix,
    effective_score_reconstruction_audit,
    evaluate_cached_effective_score,
    prepare_effective_score_cache,
)


RESULT_COLUMNS = [
    "seed",
    "task_id",
    "task",
    "matrix",
    "method",
    "rank",
    "optimization_horizon",
    "accuracy",
    "full_model_accuracy",
    "retention",
    "svd_cumulative_power",
    "initial_raw_logit_mse",
    "initial_normalized_logit_mse",
    "initial_teacher_prediction_agreement",
    "final_raw_logit_mse",
    "final_normalized_logit_mse",
    "final_teacher_prediction_agreement",
    "terminal_raw_logit_mse",
    "terminal_normalized_logit_mse",
    "terminal_teacher_prediction_agreement",
    "distillation_loss_improvement",
    "selected_step",
    "terminal_step",
    "requested_rank",
    "numerical_rank",
    "rank_bound_satisfied",
    "terminal_numerical_rank",
    "terminal_rank_bound_satisfied",
    "relative_matrix_error",
    "svd_initialization_relative_error",
    "all_teacher_parameters_frozen",
    "all_teacher_gradients_absent",
    "teacher_parameters_unchanged",
    "calibration_count",
    "calibration_namespace",
    "calibration_seed",
    "test_count",
    "test_namespace",
    "test_seed",
    "teacher_logits_sha256",
    "checkpoint_sha256",
    "bundle_path",
    "training_elapsed_seconds",
    "device",
]


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite trained result: {path}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite trained audit: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite trained bundle: {path}")
    torch.save(payload, temporary)
    temporary.replace(path)


def materialize(bundle_path: Path, result_path: Path, audit_path: Path) -> None:
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not result_path.exists():
        atomic_csv(bundle["result_rows"], result_path)
    if not audit_path.exists():
        atomic_json(bundle["audit"], audit_path)


def resolve_device(order: list[str]) -> tuple[torch.device, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for candidate in order:
        if candidate == "mps":
            available = bool(
                torch.backends.mps.is_built() and torch.backends.mps.is_available()
            )
        elif candidate == "cpu":
            available = True
        else:
            raise ValueError(f"Unsupported device: {candidate}")
        attempts.append({"device": candidate, "available": available})
        if available:
            return torch.device(candidate), {
                "selected": candidate,
                "fallback": candidate != order[0],
                "attempts": attempts,
            }
    raise RuntimeError("No configured device is available")


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def svd_reconstruction(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    matrix = matrix.detach().cpu().to(torch.float64)
    if rank == 0:
        return torch.zeros_like(matrix)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * singular_values[:rank]) @ vh[:rank]


def relative_error(reference: torch.Tensor, observed: torch.Tensor) -> float:
    reference = reference.detach().cpu().to(torch.float64)
    observed = observed.detach().cpu().to(torch.float64)
    denominator = max(float(torch.linalg.norm(reference).item()), 1e-12)
    return float(torch.linalg.norm(reference - observed).item() / denominator)


def validate_config(config: dict[str, Any]) -> None:
    sweep = config["sweep"]
    optimization = config["optimization"]
    ranks = [int(value) for value in sweep["ranks"]]
    trainable = [int(value) for value in sweep["trainable_ranks"]]
    horizons = [int(value) for value in sweep["horizons"]]
    if ranks != [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]:
        raise AssertionError("The prior trained-low-rank rank grid changed")
    if sorted(set(ranks) - {0, 128}) != sorted(trainable):
        raise AssertionError("Trainable ranks do not match the rank grid")
    if horizons != [5000, 10000]:
        raise AssertionError("Required nested 5k/10k horizons changed")
    if int(optimization["max_steps"]) != horizons[-1]:
        raise AssertionError("Maximum steps must equal the 10k horizon")
    if int(optimization["evaluate_every"]) != 50:
        raise AssertionError("Prior full-calibration evaluation interval changed")
    if bool(optimization["force_full_horizon"]) is not True:
        raise AssertionError("The comparison requires a shared full trajectory")
    if str(sweep["matrix"]) != EFFECTIVE_SCORE_MATRIX:
        raise AssertionError("This runner only trains the effective score matrix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/low_rank.toml",
    )
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--task", action="append", type=int)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()

    config_path = arguments.config.resolve()
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    validate_config(config)
    if arguments.protocol_lock is not None:
        lock = json.loads(arguments.protocol_lock.resolve().read_text(encoding="utf-8"))
        if file_sha256(config_path) != lock["config_sha256"]:
            raise RuntimeError("Effective-score distillation config differs from lock")
        for relative_path, expected_hash in {
            **lock["source_files"],
            **lock["teacher_checkpoints"],
        }.items():
            locked_path = ROOT / relative_path
            if file_sha256(locked_path) != expected_hash:
                raise RuntimeError(f"Locked input changed: {relative_path}")

    sweep = config["sweep"]
    allowed_seeds = [int(value) for value in sweep["seeds"]]
    allowed_tasks = [int(value) for value in sweep["tasks"]]
    seeds = arguments.seed or allowed_seeds
    tasks = arguments.task or allowed_tasks
    if not set(seeds).issubset(set(allowed_seeds)):
        raise ValueError("Unknown benchmark seed")
    if not set(tasks).issubset(set(allowed_tasks)):
        raise ValueError("Unknown benchmark task")
    ranks = [int(value) for value in sweep["ranks"]]
    trainable_ranks = [int(value) for value in sweep["trainable_ranks"]]
    horizons = [int(value) for value in sweep["horizons"]]
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "event": "effective_score_distillation_validated",
                    "seeds": seeds,
                    "tasks": tasks,
                    "ranks": ranks,
                    "trainable_ranks": trainable_ranks,
                    "horizons": horizons,
                    "expected_groups": len(seeds) * len(tasks),
                    "expected_rows": len(seeds)
                    * len(tasks)
                    * len(ranks)
                    * len(horizons),
                },
                indent=2,
            )
        )
        return

    extension = config["extension"]
    data_config = config["data"]
    optimization = config["optimization"]
    parent = str(extension["parent_run_id"])
    version = str(extension["version"])
    checkpoint_root = DATA_ROOT / "artifacts" / parent / "confirmatory/checkpoints"
    basis_root = DATA_ROOT / "artifacts" / parent / "confirmatory/four_gram_bases"
    artifact_root = DATA_ROOT / "artifacts" / parent / "extensions" / version
    result_root = DATA_ROOT / "results" / parent / "extensions" / version
    device, device_record = resolve_device(
        [str(value) for value in optimization["device_order"]]
    )
    protocol = DistillationTrainingConfig(
        learning_rate=float(optimization["learning_rate"]),
        batch_size=int(optimization["batch_size"]),
        max_steps=int(optimization["max_steps"]),
        evaluate_every=int(optimization["evaluate_every"]),
        patience_evaluations=int(optimization["patience_evaluations"]),
        gradient_clip_norm=float(optimization["gradient_clip_norm"]),
        epsilon=float(optimization["epsilon"]),
        improvement_tolerance=float(optimization["improvement_tolerance"]),
    )
    total_groups = len(seeds) * len(tasks)
    completed_groups = 0
    suite_started = time.monotonic()

    for seed in seeds:
        checkpoint_path = checkpoint_root / f"seed_{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model, _ = initialize_model(seed, float(checkpoint["config"]["confirmatory"]["gamma"]))
        model.load_state_dict(checkpoint["final_state_dict"])
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.to(device).eval()
        frozen_hashes = parameter_hashes(model)
        transforms = make_transforms(seed)
        matrix = effective_score_matrix(model)
        checkpoint_hash = file_sha256(checkpoint_path)

        for task_number in tasks:
            task_id = task_number - 1
            task_name = TASK_NAMES[task_id]
            task_slug = f"task_{task_number:02d}_{task_name}"
            bundle_path = (
                artifact_root
                / "trained_bundles"
                / f"seed_{seed}"
                / task_slug
                / "effective_score.pt"
            )
            result_path = (
                result_root
                / "parts"
                / f"seed_{seed}"
                / task_slug
                / "effective_score.csv"
            )
            audit_path = (
                result_root
                / "audits"
                / f"seed_{seed}"
                / task_slug
                / "effective_score.json"
            )
            if bundle_path.exists():
                materialize(bundle_path, result_path, audit_path)
                completed_groups += 1
                print(
                    json.dumps(
                        {
                            "event": "effective_score_distillation_group_skip_existing",
                            "seed": seed,
                            "task": task_name,
                            "completed_groups": completed_groups,
                            "total_groups": total_groups,
                        }
                    ),
                    flush=True,
                )
                continue
            if result_path.exists() or audit_path.exists():
                raise RuntimeError("Found result without authoritative trained bundle")

            group_started = time.monotonic()
            calibration_seed = stable_seed(
                str(data_config["calibration_namespace"]),
                seed,
                "calibration",
                task_id,
            )
            test_seed = stable_seed(
                str(data_config["test_namespace"]), seed, "test", task_id
            )
            if calibration_seed == test_seed:
                raise AssertionError("Calibration and test seeds overlap")

            teacher_data = build_functional_distillation_data(
                model,
                matrix=EFFECTIVE_SCORE_MATRIX,
                task_id=task_id,
                total_count=int(data_config["calibration_count"]),
                generation_batch_size=int(data_config["generation_batch_size"]),
                calibration_seed=calibration_seed,
                transforms=transforms,
                rho=float(data_config["rho"]),
                three_vote_distractor_rho=float(data_config["three_vote_distractor_rho"]),
                device=device,
            )
            data_fields = {field.name for field in fields(FunctionalDistillationData)}
            labels_absent = "targets" not in data_fields and "labels" not in data_fields
            full_a, full_b = svd_factor_initialization(matrix, 128)
            full_calibration_metrics = evaluate_factor_metrics(
                teacher_data,
                full_a,
                full_b,
                batch_size=protocol.batch_size,
                epsilon=protocol.epsilon,
                device=device,
            )
            if (
                full_calibration_metrics["normalized_logit_mse"] > 1e-8
                or full_calibration_metrics["teacher_prediction_agreement"] != 1.0
            ):
                raise AssertionError("Full-rank effective-score factors miss teacher")

            test_cache = prepare_effective_score_cache(
                model,
                task_id,
                int(data_config["test_count"]),
                int(data_config["generation_batch_size"]),
                test_seed,
                transforms,
                float(data_config["rho"]),
                float(data_config["three_vote_distractor_rho"]),
                device,
            )
            baseline_metrics = evaluate_cached_effective_score(model, test_cache, matrix)
            expected_baseline = float(checkpoint["test_accuracy"][task_name])
            if abs(baseline_metrics["accuracy"] - expected_baseline) > 1e-12:
                raise AssertionError("Locked test baseline mismatch")

            basis_payload = torch.load(
                basis_root / f"seed_{seed}" / f"{task_name}.pt",
                map_location="cpu",
                weights_only=False,
            )
            basis = basis_payload["basis"]
            if not torch.equal(basis.matrix, matrix):
                raise AssertionError("Stored basis and effective-score matrix differ")
            optimization_seed = stable_seed(
                "trained_low_rank_effective_score_distillation",
                seed,
                task_id,
                0,
            )
            trained = train_rank_group(
                teacher_data,
                matrix,
                trainable_ranks,
                protocol,
                optimization_seed=optimization_seed,
                device=device,
                horizon_steps=horizons,
            )
            if trained["completed_steps"] != protocol.max_steps:
                raise AssertionError("Shared trajectory stopped before 10,000 updates")
            if sorted(trained["horizon_checkpoints"]) != horizons:
                raise AssertionError("Nested 5k/10k checkpoints are incomplete")

            teacher_unchanged = parameter_hashes(model) == frozen_hashes
            all_frozen = all(not parameter.requires_grad for parameter in model.parameters())
            gradients_absent = all(parameter.grad is None for parameter in model.parameters())
            if not teacher_unchanged or not all_frozen or not gradients_absent:
                raise AssertionError("Frozen teacher integrity failed")

            result_rows: list[dict[str, Any]] = []
            condition_audits: list[dict[str, Any]] = []
            rank_payloads: dict[int, dict[int, dict[str, Any]]] = {}
            for horizon in horizons:
                rank_payloads[horizon] = {}
                for rank in ranks:
                    initial_a, initial_b = svd_factor_initialization(matrix, rank)
                    initial_metrics = evaluate_factor_metrics(
                        teacher_data,
                        initial_a,
                        initial_b,
                        batch_size=protocol.batch_size,
                        epsilon=protocol.epsilon,
                        device=device,
                    )
                    if rank in trainable_ranks:
                        checkpoint_row = trained["horizon_checkpoints"][horizon][rank]
                        factor_a = checkpoint_row["factor_a"]
                        factor_b = checkpoint_row["factor_b"]
                        terminal_a = checkpoint_row["terminal_factor_a"]
                        terminal_b = checkpoint_row["terminal_factor_b"]
                        selected_step = int(checkpoint_row["selected_step"])
                        terminal_step = horizon
                        final_metrics = {
                            "raw_logit_mse": float(checkpoint_row["selected_raw_logit_mse"]),
                            "normalized_logit_mse": float(
                                checkpoint_row["selected_normalized_logit_mse"]
                            ),
                            "teacher_prediction_agreement": float(
                                checkpoint_row["selected_teacher_prediction_agreement"]
                            ),
                        }
                        terminal_metrics = {
                            "raw_logit_mse": float(checkpoint_row["terminal_raw_logit_mse"]),
                            "normalized_logit_mse": float(
                                checkpoint_row["terminal_normalized_logit_mse"]
                            ),
                            "teacher_prediction_agreement": float(
                                checkpoint_row["terminal_teacher_prediction_agreement"]
                            ),
                        }
                    else:
                        factor_a, factor_b = initial_a.clone(), initial_b.clone()
                        terminal_a, terminal_b = factor_a.clone(), factor_b.clone()
                        selected_step = 0
                        terminal_step = 0
                        final_metrics = dict(initial_metrics)
                        terminal_metrics = dict(initial_metrics)

                    initial_product = replacement_from_factors(
                        initial_a.to(torch.float64), initial_b.to(torch.float64)
                    )
                    product = replacement_from_factors(
                        factor_a.to(torch.float64), factor_b.to(torch.float64)
                    )
                    terminal_product = replacement_from_factors(
                        terminal_a.to(torch.float64), terminal_b.to(torch.float64)
                    )
                    expected_svd = svd_reconstruction(matrix, rank)
                    initialization_error = relative_error(expected_svd, initial_product)
                    numerical = effective_score_reconstruction_audit(
                        matrix, product, None, rank
                    )
                    terminal_numerical = effective_score_reconstruction_audit(
                        matrix, terminal_product, None, rank
                    )
                    test_metrics = evaluate_cached_effective_score(
                        model, test_cache, product
                    )
                    loss_improvement = float(
                        initial_metrics["normalized_logit_mse"]
                        - final_metrics["normalized_logit_mse"]
                    )
                    checks = {
                        "rank_bound_satisfied": bool(numerical["rank_bound_satisfied"]),
                        "terminal_rank_bound_satisfied": bool(
                            terminal_numerical["rank_bound_satisfied"]
                        ),
                        "initialization_matches_svd": initialization_error <= 1e-6,
                        "selected_loss_not_worse_than_step_zero": loss_improvement
                        >= -protocol.improvement_tolerance,
                        "selected_step_within_horizon": 0 <= selected_step <= horizon,
                        "rank_zero_is_zero": rank != 0
                        or float(torch.linalg.norm(product).item()) == 0.0,
                        "rank_128_recovers_matrix": rank != 128
                        or numerical["relative_matrix_error"] <= 1e-6,
                        "rank_128_recovers_accuracy": rank != 128
                        or test_metrics["accuracy"] == baseline_metrics["accuracy"],
                    }
                    if not all(checks.values()):
                        raise AssertionError(
                            f"Condition audit failed for {seed}/{task_name}/{horizon}/{rank}: {checks}"
                        )
                    method = (
                        "trained_low_rank_5k"
                        if horizon == 5000
                        else "trained_low_rank_10k"
                    )
                    result_rows.append(
                        {
                            "seed": seed,
                            "task_id": task_id,
                            "task": task_name,
                            "matrix": EFFECTIVE_SCORE_MATRIX,
                            "method": method,
                            "rank": rank,
                            "optimization_horizon": horizon,
                            "accuracy": test_metrics["accuracy"],
                            "full_model_accuracy": baseline_metrics["accuracy"],
                            "retention": test_metrics["accuracy"]
                            / max(baseline_metrics["accuracy"], 1e-12),
                            "svd_cumulative_power": cumulative_svd_power(basis, rank),
                            "initial_raw_logit_mse": initial_metrics["raw_logit_mse"],
                            "initial_normalized_logit_mse": initial_metrics[
                                "normalized_logit_mse"
                            ],
                            "initial_teacher_prediction_agreement": initial_metrics[
                                "teacher_prediction_agreement"
                            ],
                            "final_raw_logit_mse": final_metrics["raw_logit_mse"],
                            "final_normalized_logit_mse": final_metrics[
                                "normalized_logit_mse"
                            ],
                            "final_teacher_prediction_agreement": final_metrics[
                                "teacher_prediction_agreement"
                            ],
                            "terminal_raw_logit_mse": terminal_metrics["raw_logit_mse"],
                            "terminal_normalized_logit_mse": terminal_metrics[
                                "normalized_logit_mse"
                            ],
                            "terminal_teacher_prediction_agreement": terminal_metrics[
                                "teacher_prediction_agreement"
                            ],
                            "distillation_loss_improvement": loss_improvement,
                            "selected_step": selected_step,
                            "terminal_step": terminal_step,
                            "requested_rank": numerical["requested_rank"],
                            "numerical_rank": numerical["numerical_rank"],
                            "rank_bound_satisfied": numerical["rank_bound_satisfied"],
                            "terminal_numerical_rank": terminal_numerical["numerical_rank"],
                            "terminal_rank_bound_satisfied": terminal_numerical[
                                "rank_bound_satisfied"
                            ],
                            "relative_matrix_error": numerical["relative_matrix_error"],
                            "svd_initialization_relative_error": initialization_error,
                            "all_teacher_parameters_frozen": all_frozen,
                            "all_teacher_gradients_absent": gradients_absent,
                            "teacher_parameters_unchanged": teacher_unchanged,
                            "calibration_count": int(data_config["calibration_count"]),
                            "calibration_namespace": str(
                                data_config["calibration_namespace"]
                            ),
                            "calibration_seed": calibration_seed,
                            "test_count": int(data_config["test_count"]),
                            "test_namespace": str(data_config["test_namespace"]),
                            "test_seed": test_seed,
                            "teacher_logits_sha256": teacher_data.teacher_logits_sha256,
                            "checkpoint_sha256": checkpoint_hash,
                            "bundle_path": str(bundle_path),
                            "training_elapsed_seconds": trained["elapsed_seconds"],
                            "device": device.type,
                        }
                    )
                    condition_audits.append(
                        {
                            "horizon": horizon,
                            "rank": rank,
                            **checks,
                            "passed": all(checks.values()),
                        }
                    )
                    rank_payloads[horizon][rank] = {
                        "rank": rank,
                        "horizon": horizon,
                        "selected_step": selected_step,
                        "factor_a": factor_a.detach().cpu(),
                        "factor_b": factor_b.detach().cpu(),
                        "terminal_factor_a": terminal_a.detach().cpu(),
                        "terminal_factor_b": terminal_b.detach().cpu(),
                        "selected_product_sha256": tensor_sha256(product),
                        "terminal_product_sha256": tensor_sha256(terminal_product),
                    }

            audit = {
                "passed": labels_absent
                and all_frozen
                and gradients_absent
                and teacher_unchanged
                and all(row["passed"] for row in condition_audits),
                "seed": seed,
                "task_id": task_id,
                "task": task_name,
                "matrix": EFFECTIVE_SCORE_MATRIX,
                "labels_absent_from_distillation_cache": labels_absent,
                "all_teacher_parameters_frozen": all_frozen,
                "all_teacher_gradients_absent": gradients_absent,
                "teacher_parameters_unchanged": teacher_unchanged,
                "full_rank_calibration_metrics": full_calibration_metrics,
                "calibration_and_test_namespaces_distinct": str(
                    data_config["calibration_namespace"]
                )
                != str(data_config["test_namespace"]),
                "calibration_and_test_seeds_distinct": calibration_seed != test_seed,
                "same_trajectory_nested_horizons": True,
                "completed_steps": trained["completed_steps"],
                "horizons": horizons,
                "condition_audits": condition_audits,
                "device": device_record,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "config_sha256": file_sha256(config_path),
                "checkpoint_sha256": checkpoint_hash,
                "teacher_logits_sha256": teacher_data.teacher_logits_sha256,
            }
            if not audit["passed"]:
                raise AssertionError(f"Group audit failed: {audit}")
            bundle = {
                "extension": extension,
                "seed": seed,
                "task_id": task_id,
                "task": task_name,
                "matrix": EFFECTIVE_SCORE_MATRIX,
                "config_path": str(config_path),
                "config_sha256": file_sha256(config_path),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "calibration_seed": calibration_seed,
                "test_seed": test_seed,
                "teacher_logits_sha256": teacher_data.teacher_logits_sha256,
                "rank_payloads": rank_payloads,
                "optimization_history": trained["history"],
                "optimizer_state_dict": trained["optimizer_state_dict"],
                "optimization_seed": optimization_seed,
                "completed_steps": trained["completed_steps"],
                "training_elapsed_seconds": trained["elapsed_seconds"],
                "result_rows": result_rows,
                "audit": audit,
            }
            atomic_torch(bundle, bundle_path)
            materialize(bundle_path, result_path, audit_path)
            completed_groups += 1
            print(
                json.dumps(
                    {
                        "event": "effective_score_distillation_group_complete",
                        "seed": seed,
                        "task": task_name,
                        "rows": len(result_rows),
                        "training_elapsed_seconds": trained["elapsed_seconds"],
                        "group_elapsed_seconds": time.monotonic() - group_started,
                        "completed_groups": completed_groups,
                        "total_groups": total_groups,
                        "suite_elapsed_seconds": time.monotonic() - suite_started,
                    }
                ),
                flush=True,
            )
            del teacher_data, test_cache, trained, rank_payloads, bundle
            clear_device_cache(device)

    print(
        json.dumps(
            {
                "event": "effective_score_distillation_sweep_complete",
                "groups": completed_groups,
                "total_groups": total_groups,
                "elapsed_seconds": time.monotonic() - suite_started,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
