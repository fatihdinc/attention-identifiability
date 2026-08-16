from __future__ import annotations

import argparse
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
    TASK_COUNT,
    TASK_NAMES,
    audit_transforms,
    evaluate_tasks,
    initialize_model,
    make_transforms,
    model_architecture_audit,
    parameter_hashes,
)
from identifiability_llm.ten_task_effective_score import (  # noqa: E402
    effective_score_matrix,
    train_joint_exposure_matched,
)


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
    raise RuntimeError(f"No requested device available: {attempts}")


def to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def atomic_torch_replace(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_torch_create(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite final checkpoint: {path}")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_create(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite final result: {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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
    primary = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    replacements = [int(value) for value in config["experiment"]["replacement_seeds"]]
    selected = arguments.seed or primary
    if not set(selected).issubset(set(primary) | set(replacements)):
        raise ValueError("Requested seed was not preregistered")
    if arguments.validate_only:
        print(
            json.dumps(
                {
                    "event": "effective_score_training_runner_validated",
                    "seeds": selected,
                    "config_sha256": file_sha256(config_path),
                    "preregistration_sha256": file_sha256(preregistration_path),
                },
                indent=2,
            )
        )
        return
    device, device_record = resolve_device(
        [str(value) for value in config["experiment"]["device_order"]]
    )
    confirmatory = config["confirmatory"]
    generator_config = config["generator"]
    run_id = str(config["experiment"]["run_id"])
    checkpoint_root = DATA_ROOT / "artifacts" / run_id / "confirmatory/checkpoints"
    result_root = DATA_ROOT / "results" / run_id / "confirmatory/training"
    suite_started = time.monotonic()
    for seed_index, model_seed in enumerate(selected, start=1):
        final_checkpoint = checkpoint_root / f"seed_{model_seed}.pt"
        partial_checkpoint = checkpoint_root / f"seed_{model_seed}.partial.pt"
        result_path = result_root / f"seed_{model_seed}.json"
        if final_checkpoint.exists() and result_path.exists():
            print(
                json.dumps(
                    {
                        "event": "effective_score_training_skip_existing",
                        "seed": model_seed,
                        "checkpoint": str(final_checkpoint),
                    }
                ),
                flush=True,
            )
            continue
        model, initial_state = initialize_model(
            model_seed, float(confirmatory["gamma"])
        )
        initial_architecture_audit = model_architecture_audit(model)
        if not initial_architecture_audit["passed"]:
            raise AssertionError(
                f"Initial architecture audit failed: {initial_architecture_audit}"
            )
        transforms = make_transforms(model_seed)
        initial_hashes = parameter_hashes(model)
        start_update = 0
        optimizer_state = None
        task_batch_counts = None
        prior_history: list[dict[str, Any]] = []
        if partial_checkpoint.exists():
            partial = torch.load(partial_checkpoint, map_location="cpu", weights_only=False)
            if partial["config_sha256"] != file_sha256(config_path):
                raise RuntimeError("Partial checkpoint config mismatch")
            model.load_state_dict(partial["model_state_dict"])
            start_update = int(partial["outer_update"])
            optimizer_state = partial["optimizer_state_dict"]
            task_batch_counts = partial["task_batch_counts"]
            prior_history = partial["history"]
            print(
                json.dumps(
                    {
                        "event": "effective_score_training_resume",
                        "seed": model_seed,
                        "start_update": start_update,
                    }
                ),
                flush=True,
            )
        checkpoint_every = int(confirmatory["checkpoint_every"])

        def checkpoint_callback(state: dict[str, Any]) -> None:
            update = int(state["outer_update"])
            if update != 1 and update % checkpoint_every and update != int(
                confirmatory["outer_updates"]
            ):
                return
            atomic_torch_replace(
                {
                    "run_id": run_id,
                    "seed": model_seed,
                    "outer_update": update,
                    "model_state_dict": to_cpu(model.state_dict()),
                    "optimizer_state_dict": to_cpu(state["optimizer_state"]),
                    "history": state["history"],
                    "task_batch_counts": state["task_batch_counts"],
                    "validation_accuracy": state["validation_accuracy"],
                    "transforms": {
                        "context": transforms.context,
                        "query": transforms.query,
                        "context_seed": transforms.context_seed,
                        "query_seed": transforms.query_seed,
                    },
                    "config_sha256": file_sha256(config_path),
                    "preregistration_sha256": file_sha256(preregistration_path),
                },
                partial_checkpoint,
            )

        run_started = time.monotonic()
        training = train_joint_exposure_matched(
            model,
            transforms,
            model_seed,
            "confirmatory",
            str(config["data"]["confirmatory_training_namespace"]),
            float(generator_config["rho"]),
            float(generator_config["three_vote_distractor_rho"]),
            device,
            learning_rate=float(confirmatory["learning_rate"]),
            weight_decay=float(confirmatory["weight_decay"]),
            task_batch_size=int(confirmatory["task_batch_size"]),
            outer_updates=int(confirmatory["outer_updates"]),
            evaluate_every=int(confirmatory["evaluate_every"]),
            evaluation_batch_size=int(confirmatory["evaluation_batch_size"]),
            validation_count=int(confirmatory["validation_count"]),
            optimizer_state=optimizer_state,
            start_update=start_update,
            initial_task_batch_counts=task_batch_counts,
            initial_history=prior_history,
            checkpoint_callback=checkpoint_callback,
        )
        test_accuracy = evaluate_tasks(
            model,
            list(range(TASK_COUNT)),
            int(confirmatory["test_count"]),
            int(confirmatory["evaluation_batch_size"]),
            str(config["data"]["confirmatory_test_namespace"]),
            model_seed,
            "test",
            transforms,
            float(generator_config["rho"]),
            device,
            float(generator_config["three_vote_distractor_rho"]),
        )
        finite = all(
            torch.isfinite(torch.tensor(value)).item()
            for value in list(training.validation_accuracy.values())
            + list(test_accuracy.values())
        )
        expected_updates = int(confirmatory["outer_updates"])
        exposure_exact = all(
            value == expected_updates for value in training.task_batch_counts.values()
        )
        included = finite and exposure_exact and training.outer_updates_completed == expected_updates
        result = {
            "run_id": run_id,
            "seed": model_seed,
            "included": included,
            "finite_accuracies": finite,
            "exact_task_exposure": exposure_exact,
            "outer_updates_completed": training.outer_updates_completed,
            "task_batch_counts": training.task_batch_counts,
            "task_example_counts": training.task_example_counts,
            "validation_accuracy": training.validation_accuracy,
            "test_accuracy": test_accuracy,
            "mean_test_accuracy": float(sum(test_accuracy.values()) / len(test_accuracy)),
            "training_elapsed_seconds": training.elapsed_seconds,
            "total_run_elapsed_seconds": time.monotonic() - run_started,
            "training_examples_per_second": training.examples_per_second,
            "device": device_record,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "initial_architecture_audit": initial_architecture_audit,
            "transform_audit": audit_transforms(transforms),
            "initial_parameter_hashes": initial_hashes,
            "final_parameter_hashes": parameter_hashes(model),
            "effective_score_matrix_sha256": sha256(
                effective_score_matrix(model).numpy().tobytes()
            ).hexdigest(),
            "config_sha256": file_sha256(config_path),
            "preregistration_sha256": file_sha256(preregistration_path),
        }
        if not included:
            raise RuntimeError(f"Confirmatory inclusion checks failed: {result}")
        checkpoint = {
            "run_id": run_id,
            "seed": model_seed,
            "initial_state_dict": initial_state,
            "final_state_dict": to_cpu(model.state_dict()),
            "optimizer_state_dict": to_cpu(training.optimizer_state),
            "training_history": training.history,
            "task_batch_counts": training.task_batch_counts,
            "task_example_counts": training.task_example_counts,
            "validation_accuracy": training.validation_accuracy,
            "test_accuracy": test_accuracy,
            "transforms": {
                "context": transforms.context,
                "query": transforms.query,
                "context_seed": transforms.context_seed,
                "query_seed": transforms.query_seed,
            },
            "config": config,
            "result": result,
        }
        atomic_torch_create(checkpoint, final_checkpoint)
        result["checkpoint_path"] = str(final_checkpoint)
        atomic_json_create(result, result_path)
        print(
            json.dumps(
                {
                    "event": "effective_score_training_complete",
                    "seed": model_seed,
                    "test_accuracy": test_accuracy,
                    "seed_index": seed_index,
                    "seed_count": len(selected),
                    "suite_elapsed_seconds": time.monotonic() - suite_started,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
