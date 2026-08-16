from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
import torch

from .ten_task_attention import (
    D_MODEL,
    LABEL_SLICE,
    MATRIX_TO_PARAMETER,
    TASK_COUNT,
    TASK_SLICE,
    OrthogonalTransforms,
    TaskBatch,
    TenTaskAttention,
    generate_task_batch,
    parameter_hashes,
    split_batches,
    stable_seed,
    tensor_sha256,
)


@dataclass(frozen=True)
class MatrixGrams:
    right: torch.Tensor
    left: torch.Tensor
    observation_count: int


@dataclass(frozen=True)
class MatrixBasis:
    weight: torch.Tensor
    svd_u: torch.Tensor
    svd_s: torch.Tensor
    svd_vh: torch.Tensor
    right_values: torch.Tensor
    right_vectors: torch.Tensor
    left_values: torch.Tensor
    left_vectors: torch.Tensor


def _inputs_from_batch(
    batch: TaskBatch, details: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    flat_context = batch.context.reshape(-1, D_MODEL)
    return {
        "Q": batch.query,
        "K": flat_context,
        "V": flat_context,
        "O": details["retrieved"].detach().cpu(),
    }


@torch.inference_mode()
def collect_task_grams(
    model: TenTaskAttention,
    task_id: int,
    total_count: int,
    batch_size: int,
    split_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    device: torch.device,
    three_vote_distractor_rho: float,
) -> dict[str, MatrixGrams]:
    model.eval()
    parameters = dict(model.named_parameters())
    right_sums = {
        matrix: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
        for matrix in MATRIX_TO_PARAMETER
    }
    left_sums = {
        matrix: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
        for matrix in MATRIX_TO_PARAMETER
    }
    counts = {matrix: 0 for matrix in MATRIX_TO_PARAMETER}
    example_count = 0
    for batch in split_batches(
        task_id,
        total_count,
        batch_size,
        split_seed,
        transforms,
        rho,
        three_vote_distractor_rho,
    ):
        _, details = model(batch.context.to(device), batch.query.to(device))
        inputs = _inputs_from_batch(batch, details)
        for matrix, parameter_name in MATRIX_TO_PARAMETER.items():
            observed = inputs[matrix].detach().cpu().to(dtype=torch.float64)
            weight = parameters[parameter_name].detach().cpu().to(dtype=torch.float64)
            outputs = observed @ weight.T
            right_sums[matrix] += observed.T @ observed
            left_sums[matrix] += outputs.T @ outputs
            counts[matrix] += int(observed.shape[0])
        example_count += batch.count
    if example_count != total_count:
        raise AssertionError(f"Expected {total_count} examples, observed {example_count}")
    result = {
        matrix: MatrixGrams(
            right=right_sums[matrix] / counts[matrix],
            left=left_sums[matrix] / counts[matrix],
            observation_count=counts[matrix],
        )
        for matrix in MATRIX_TO_PARAMETER
    }
    expected_token_count = total_count * 64
    if result["Q"].observation_count != total_count:
        raise AssertionError("Q calibration count does not equal example count")
    if result["O"].observation_count != total_count:
        raise AssertionError("O calibration count does not equal example count")
    if result["K"].observation_count != expected_token_count:
        raise AssertionError("K calibration count does not equal token count")
    if result["V"].observation_count != expected_token_count:
        raise AssertionError("V calibration count does not equal token count")
    return result


def gram_audit(
    model: TenTaskAttention, grams: dict[str, MatrixGrams], atol: float = 1e-9
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    matrix_rows: dict[str, Any] = {}
    passed = True
    for matrix, parameter_name in MATRIX_TO_PARAMETER.items():
        record = grams[matrix]
        weight = parameters[parameter_name].detach().cpu().to(dtype=torch.float64)
        right_symmetry = float(torch.linalg.norm(record.right - record.right.T).item())
        left_symmetry = float(torch.linalg.norm(record.left - record.left.T).item())
        expected_left = weight @ record.right @ weight.T
        equivalence = float(
            torch.linalg.norm(record.left - expected_left).item()
            / max(float(torch.linalg.norm(expected_left).item()), 1e-12)
        )
        matrix_passed = (
            right_symmetry <= atol and left_symmetry <= atol and equivalence <= atol
        )
        passed = passed and matrix_passed
        matrix_rows[matrix] = {
            "passed": matrix_passed,
            "right_symmetry_error": right_symmetry,
            "left_symmetry_error": left_symmetry,
            "left_equivalence_relative_error": equivalence,
            "observation_count": record.observation_count,
        }
    return {"passed": passed, "matrices": matrix_rows}


def _descending_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrix.to(torch.float64))
    return values.flip(0), vectors.flip(1)


def build_matrix_basis(weight: torch.Tensor, grams: MatrixGrams) -> MatrixBasis:
    weight = weight.detach().cpu().to(dtype=torch.float64)
    svd_u, svd_s, svd_vh = torch.linalg.svd(weight, full_matrices=False)
    right_values, right_vectors = _descending_eigh(grams.right)
    left_values, left_vectors = _descending_eigh(grams.left)
    return MatrixBasis(
        weight=weight,
        svd_u=svd_u,
        svd_s=svd_s,
        svd_vh=svd_vh,
        right_values=right_values,
        right_vectors=right_vectors,
        left_values=left_values,
        left_vectors=left_vectors,
    )


def build_bases(
    model: TenTaskAttention, grams: dict[str, MatrixGrams]
) -> dict[str, MatrixBasis]:
    parameters = dict(model.named_parameters())
    return {
        matrix: build_matrix_basis(parameters[parameter_name], grams[matrix])
        for matrix, parameter_name in MATRIX_TO_PARAMETER.items()
    }


def equal_weight_mixture_grams(
    task_grams: dict[int, dict[str, MatrixGrams]], task_ids: Iterable[int]
) -> dict[str, MatrixGrams]:
    selected = list(task_ids)
    if not selected:
        raise ValueError("Mixture requires at least one task")
    if len(set(selected)) != len(selected):
        raise ValueError("Mixture task list contains duplicates")
    missing = set(selected) - set(task_grams)
    if missing:
        raise KeyError(f"Missing task Grams: {sorted(missing)}")
    result: dict[str, MatrixGrams] = {}
    for matrix in MATRIX_TO_PARAMETER:
        right = sum(task_grams[task][matrix].right for task in selected) / len(selected)
        left = sum(task_grams[task][matrix].left for task in selected) / len(selected)
        counts = {task_grams[task][matrix].observation_count for task in selected}
        if len(counts) != 1:
            raise AssertionError(
                f"Unequal calibration observation counts for {matrix}: {sorted(counts)}"
            )
        result[matrix] = MatrixGrams(
            right=right,
            left=left,
            observation_count=next(iter(counts)),
        )
    return result


def leave_one_task_out_grams(
    task_grams: dict[int, dict[str, MatrixGrams]], omitted_task: int
) -> dict[str, MatrixGrams]:
    selected = sorted(set(task_grams) - {omitted_task})
    if len(selected) != len(task_grams) - 1:
        raise KeyError(f"Omitted task {omitted_task} is not present")
    return equal_weight_mixture_grams(task_grams, selected)


@torch.inference_mode()
def collect_shuffled_task_grams(
    model: TenTaskAttention,
    task_ids: list[int],
    total_count_per_task: int,
    batch_size: int,
    split_seed: int | dict[int, int],
    transforms: OrthogonalTransforms,
    rho: float,
    device: torch.device,
    three_vote_distractor_rho: float,
    shuffle_seed: int,
) -> dict[int, dict[str, MatrixGrams]]:
    """Pool examples and permute task labels while preserving exact per-task counts."""
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be a nonempty list without duplicates")
    model.eval()
    parameters = dict(model.named_parameters())
    right_sums = {
        task: {
            matrix: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
            for matrix in MATRIX_TO_PARAMETER
        }
        for task in task_ids
    }
    left_sums = {
        task: {
            matrix: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
            for matrix in MATRIX_TO_PARAMETER
        }
        for task in task_ids
    }
    example_counts = {task: 0 for task in task_ids}
    for batch_index, start in enumerate(range(0, total_count_per_task, batch_size)):
        count = min(batch_size, total_count_per_task - start)
        inputs_by_task: dict[int, dict[str, torch.Tensor]] = {}
        for task in task_ids:
            task_split_seed = (
                split_seed[task] if isinstance(split_seed, dict) else split_seed
            )
            batch = generate_task_batch(
                task,
                count,
                stable_seed(task_split_seed, task, batch_index),
                transforms,
                rho,
                three_vote_distractor_rho,
            )
            _, details = model(batch.context.to(device), batch.query.to(device))
            inputs_by_task[task] = {
                "Q": batch.query,
                "K": batch.context,
                "V": batch.context,
                "O": details["retrieved"].detach().cpu(),
            }
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed(shuffle_seed, batch_index)
        )
        pooled_count = len(task_ids) * count
        assignments = torch.randperm(pooled_count, generator=generator).reshape(
            len(task_ids), count
        )
        for matrix, parameter_name in MATRIX_TO_PARAMETER.items():
            weight = parameters[parameter_name].detach().cpu().to(dtype=torch.float64)
            pooled_observed = torch.cat(
                [inputs_by_task[task][matrix] for task in task_ids], dim=0
            )
            for assigned_index, assigned_task in enumerate(task_ids):
                indices = assignments[assigned_index]
                observed = pooled_observed[indices].detach().cpu().to(dtype=torch.float64)
                observed = observed.reshape(-1, D_MODEL)
                outputs = observed @ weight.T
                right_sums[assigned_task][matrix] += observed.T @ observed
                left_sums[assigned_task][matrix] += outputs.T @ outputs
        for task in task_ids:
            example_counts[task] += count
    result: dict[int, dict[str, MatrixGrams]] = {}
    for task in task_ids:
        if example_counts[task] != total_count_per_task:
            raise AssertionError("Shuffled calibration assignment count mismatch")
        result[task] = {}
        for matrix in MATRIX_TO_PARAMETER:
            multiplier = 64 if matrix in ("K", "V") else 1
            observation_count = example_counts[task] * multiplier
            result[task][matrix] = MatrixGrams(
                right=right_sums[task][matrix] / observation_count,
                left=left_sums[task][matrix] / observation_count,
                observation_count=observation_count,
            )
    return result


def reconstruct_weight(
    basis: MatrixBasis, method: str, rank: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    maximum_rank = min(basis.weight.shape)
    if rank < 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [0, {maximum_rank}]")
    if method == "parameter_svd":
        if rank == 0:
            return torch.zeros_like(basis.weight), None
        return (
            (basis.svd_u[:, :rank] * basis.svd_s[:rank]) @ basis.svd_vh[:rank],
            None,
        )
    if method == "right_gram":
        vectors = basis.right_vectors[:, :rank]
        projector = vectors @ vectors.T
        return basis.weight @ projector, projector
    if method == "left_gram":
        vectors = basis.left_vectors[:, :rank]
        projector = vectors @ vectors.T
        return projector @ basis.weight, projector
    raise ValueError(f"Unknown reconstruction method: {method}")


def random_projector_reconstruction(
    weight: torch.Tensor, rank: int, seed: int, side: str = "right"
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.detach().cpu().to(dtype=torch.float64)
    dimension = weight.shape[1] if side == "right" else weight.shape[0]
    if rank < 0 or rank > dimension:
        raise ValueError(f"rank must be in [0, {dimension}]")
    if rank == 0:
        projector = torch.zeros(dimension, dimension, dtype=torch.float64)
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        raw = torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
        vectors, _ = torch.linalg.qr(raw, mode="reduced")
        projector = vectors @ vectors.T
    if side == "right":
        return weight @ projector, projector
    if side == "left":
        return projector @ weight, projector
    raise ValueError("side must be 'right' or 'left'")


def reconstruction_audit(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    projector: torch.Tensor | None,
    requested_rank: int,
) -> dict[str, Any]:
    original = original.detach().cpu().to(dtype=torch.float64)
    reconstructed = reconstructed.detach().cpu().to(dtype=torch.float64)
    singular_values = torch.linalg.svdvals(reconstructed)
    tolerance = max(reconstructed.shape) * torch.finfo(torch.float64).eps * max(
        float(singular_values.max().item()), 1.0
    )
    numerical_rank = int((singular_values > tolerance).sum().item())
    relative_error = float(
        torch.linalg.norm(original - reconstructed).item()
        / max(float(torch.linalg.norm(original).item()), 1e-12)
    )
    result: dict[str, Any] = {
        "requested_rank": requested_rank,
        "numerical_rank": numerical_rank,
        "rank_bound_satisfied": numerical_rank <= requested_rank,
        "numerical_rank_tolerance": tolerance,
        "relative_weight_error": relative_error,
    }
    if projector is not None:
        denominator = max(float(torch.linalg.norm(projector).item()), 1.0)
        result.update(
            {
                "projector_symmetry_error": float(
                    torch.linalg.norm(projector - projector.T).item() / denominator
                ),
                "projector_idempotence_error": float(
                    torch.linalg.norm(projector @ projector - projector).item() / denominator
                ),
            }
        )
    return result


def svd_power_rows(
    state_dict: dict[str, torch.Tensor], ranks: list[int], **metadata: Any
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for matrix, parameter_name in MATRIX_TO_PARAMETER.items():
        weight = state_dict[parameter_name].detach().cpu().to(dtype=torch.float64)
        power = torch.linalg.svdvals(weight).square()
        total = float(power.sum().item())
        if total <= 0:
            raise ValueError(f"Cannot normalize zero SVD power for {matrix}")
        cumulative = torch.cumsum(power, dim=0) / total
        previous = 0.0
        for rank in ranks:
            if rank < 0 or rank > power.numel():
                raise ValueError(f"Invalid rank {rank} for {matrix}")
            fraction = 0.0 if rank == 0 else float(cumulative[rank - 1].item())
            if fraction + 1e-14 < previous:
                raise AssertionError(f"Cumulative SVD power is not monotone for {matrix}")
            previous = fraction
            rows.append(
                {
                    **metadata,
                    "matrix": matrix,
                    "rank": rank,
                    "svd_cumulative_power": fraction,
                }
            )
    return rows


class FrozenMatrixIntervention(AbstractContextManager["FrozenMatrixIntervention"]):
    def __init__(
        self,
        model: TenTaskAttention,
        matrix: str,
        replacement: torch.Tensor,
    ) -> None:
        if matrix not in MATRIX_TO_PARAMETER:
            raise ValueError(f"Unknown matrix: {matrix}")
        self.model = model
        self.matrix = matrix
        self.parameter_name = MATRIX_TO_PARAMETER[matrix]
        self.replacement = replacement
        self.before_hashes: dict[str, str] = {}
        self.original: torch.Tensor | None = None
        self.entry_audit: dict[str, Any] = {}
        self.restoration_audit: dict[str, Any] = {}

    def __enter__(self) -> "FrozenMatrixIntervention":
        parameters = dict(self.model.named_parameters())
        parameter = parameters[self.parameter_name]
        if tuple(parameter.shape) != tuple(self.replacement.shape):
            raise ValueError("Replacement shape does not match selected matrix")
        self.before_hashes = parameter_hashes(self.model)
        self.original = parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(self.replacement.to(device=parameter.device, dtype=parameter.dtype))
        after_hashes = parameter_hashes(self.model)
        unchanged = {
            name: after_hashes[name] == before_hash
            for name, before_hash in self.before_hashes.items()
            if name != self.parameter_name
        }
        replacement_hash = tensor_sha256(
            self.replacement.to(device="cpu", dtype=parameter.dtype)
        )
        self.entry_audit = {
            "selected_parameter": self.parameter_name,
            "all_other_parameters_bit_identical": all(unchanged.values()),
            "other_parameter_checks": unchanged,
            "selected_parameter_matches_replacement": after_hashes[self.parameter_name]
            == replacement_hash,
        }
        if not all(self.entry_audit[key] for key in (
            "all_other_parameters_bit_identical",
            "selected_parameter_matches_replacement",
        )):
            raise AssertionError(f"Intervention isolation failed: {self.entry_audit}")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool | None:
        if self.original is None:
            return None
        parameter = dict(self.model.named_parameters())[self.parameter_name]
        with torch.no_grad():
            parameter.copy_(self.original)
        restored_hashes = parameter_hashes(self.model)
        checks = {
            name: restored_hashes[name] == before_hash
            for name, before_hash in self.before_hashes.items()
        }
        self.restoration_audit = {
            "all_parameters_restored_bit_identically": all(checks.values()),
            "parameter_checks": checks,
        }
        if not self.restoration_audit["all_parameters_restored_bit_identically"]:
            raise AssertionError(f"Exact restoration failed: {self.restoration_audit}")
        return None


def controlled_inputs(
    batch: TaskBatch,
    transforms: OrthogonalTransforms,
    control: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = batch.context
    query = batch.query
    if control == "none":
        return context, query
    if control == "query_task_code_zero":
        query_latent = batch.query_latent.clone()
        query_latent[:, TASK_SLICE] = 0
        return context, query_latent @ transforms.query.T
    if control == "query_task_code_shuffle":
        query_latent = batch.query_latent.clone()
        wrong_task = (batch.task_id + 1 + (seed % (TASK_COUNT - 1))) % TASK_COUNT
        query_latent[:, TASK_SLICE] = 0
        query_latent[:, TASK_SLICE.start + wrong_task] = 1
        return context, query_latent @ transforms.query.T
    if control == "context_label_permutation":
        context_latent = batch.context_latent.clone()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutations = torch.rand(
            batch.count, context_latent.shape[1], generator=generator
        ).argsort(dim=1)
        labels = context_latent[:, :, LABEL_SLICE]
        shuffled = torch.gather(
            labels,
            1,
            permutations.unsqueeze(-1).expand_as(labels),
        )
        context_latent[:, :, LABEL_SLICE] = shuffled
        return context_latent @ transforms.context.T, query
    if control == "query_only":
        return torch.zeros_like(context), query
    raise ValueError(f"Unknown control: {control}")


@torch.inference_mode()
def evaluate_control_accuracy(
    model: TenTaskAttention,
    task_id: int,
    total_count: int,
    batch_size: int,
    split_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    device: torch.device,
    three_vote_distractor_rho: float,
    control: str,
) -> float:
    model.eval()
    correct = 0
    observed = 0
    for batch_index, batch in enumerate(
        split_batches(
            task_id,
            total_count,
            batch_size,
            split_seed,
            transforms,
            rho,
            three_vote_distractor_rho,
        )
    ):
        context, query = controlled_inputs(
            batch,
            transforms,
            control,
            stable_seed(split_seed, control, batch_index),
        )
        logits, _ = model(context.to(device), query.to(device))
        correct += int((logits.argmax(dim=-1) == batch.targets.to(device)).sum().item())
        observed += batch.count
    return correct / observed


def minimum_rank_table(
    conditions: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    required = set(group_columns) | {
        "seed",
        "rank",
        "accuracy",
        "full_model_accuracy",
    }
    missing = required - set(conditions.columns)
    if missing:
        raise KeyError(f"Missing columns for rank summary: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for keys, group in conditions.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean_by_rank = group.groupby("rank")["accuracy"].mean().sort_index()
        baseline_by_seed = group.groupby("seed")["full_model_accuracy"].first()
        baseline = float(baseline_by_seed.mean())
        row = {column: value for column, value in zip(group_columns, keys, strict=True)}
        row["mean_full_model_accuracy"] = baseline
        for label, fraction in (("K95", 0.95), ("K99", 0.99)):
            passing = mean_by_rank[mean_by_rank >= fraction * baseline]
            row[label] = int(passing.index.min()) if not passing.empty else None
        rows.append(row)
    return pd.DataFrame(rows)


def task_specificity_table(conditions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "seed",
        "source_task",
        "evaluation_task",
        "matrix",
        "method",
        "rank",
        "accuracy",
        "full_model_accuracy",
    }
    missing = required - set(conditions.columns)
    if missing:
        raise KeyError(f"Missing columns for task specificity: {sorted(missing)}")
    frame = conditions.copy()
    if (frame["full_model_accuracy"] <= 0).any():
        raise ValueError("Full-model accuracy must be positive for normalized retention")
    frame["retention"] = frame["accuracy"] / frame["full_model_accuracy"]
    rows: list[dict[str, Any]] = []
    group_columns = ["source_task", "matrix", "method", "rank"]
    for keys, group in frame.groupby(group_columns, sort=True):
        source_task, matrix, method, rank = keys
        diagonal = group[group["evaluation_task"] == source_task]["retention"]
        off_diagonal = group[group["evaluation_task"] != source_task]["retention"]
        if diagonal.empty or off_diagonal.empty:
            raise AssertionError(f"Incomplete specificity group: {keys}")
        diagonal_mean = float(diagonal.mean())
        off_diagonal_mean = float(off_diagonal.mean())
        rows.append(
            {
                "source_task": source_task,
                "matrix": matrix,
                "method": method,
                "rank": int(rank),
                "diagonal_retention": diagonal_mean,
                "mean_off_diagonal_retention": off_diagonal_mean,
                "specificity": diagonal_mean - off_diagonal_mean,
                "seed_count": int(group["seed"].nunique()),
                "evaluation_task_count": int(group["evaluation_task"].nunique()),
            }
        )
    return pd.DataFrame(rows)
