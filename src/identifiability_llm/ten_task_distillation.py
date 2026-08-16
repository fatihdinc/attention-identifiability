from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import time
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .ten_task_attention import (
    D_MODEL,
    MATRIX_TO_PARAMETER,
    OrthogonalTransforms,
    TenTaskAttention,
    split_batches,
    tensor_sha256,
)

EFFECTIVE_OUTPUT_MATRIX = "effective_output"
EFFECTIVE_SCORE_MATRIX = "effective_score"


@dataclass(frozen=True)
class FunctionalDistillationData:
    """Label-free cache for one task and one intervened matrix.

    The cache deliberately has no target-label field.  It contains only inputs
    needed for the exact one-matrix functional forward and frozen-teacher
    logits.
    """

    matrix: str
    task_id: int
    calibration_seed: int
    matrix_inputs: torch.Tensor
    attention_other: torch.Tensor | None
    token_logit_contributions: torch.Tensor | None
    downstream_weight: torch.Tensor | None
    readout_bias: torch.Tensor
    teacher_logits: torch.Tensor
    gamma: float
    teacher_mean_square: float
    teacher_logits_sha256: str
    example_count: int
    batch_count: int
    matrix_inputs_secondary: torch.Tensor | None = None


@dataclass(frozen=True)
class DistillationTrainingConfig:
    learning_rate: float
    batch_size: int
    max_steps: int
    evaluate_every: int
    patience_evaluations: int
    gradient_clip_norm: float
    epsilon: float
    improvement_tolerance: float


def _tensor_collection_sha256(tensors: Iterable[torch.Tensor]) -> str:
    digest = sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_functional_distillation_data(
    model: TenTaskAttention,
    *,
    matrix: str,
    task_id: int,
    total_count: int,
    generation_batch_size: int,
    calibration_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    three_vote_distractor_rho: float,
    device: torch.device,
) -> FunctionalDistillationData:
    if matrix not in {
        *MATRIX_TO_PARAMETER,
        EFFECTIVE_OUTPUT_MATRIX,
        EFFECTIVE_SCORE_MATRIX,
    }:
        raise ValueError(f"Unknown matrix: {matrix}")
    model.eval()
    model_parameters = dict(model.named_parameters())
    o_weight = model_parameters["o_proj.weight"]
    readout_weight = model_parameters["readout.weight"]
    readout_bias = model_parameters["readout.bias"].detach().cpu().clone()
    value_to_logits = readout_weight @ o_weight

    matrix_inputs: list[torch.Tensor] = []
    matrix_inputs_secondary: list[torch.Tensor] = []
    attention_other: list[torch.Tensor] = []
    token_contributions: list[torch.Tensor] = []
    teacher_logits: list[torch.Tensor] = []
    observed = 0
    batch_count = 0
    downstream_weight: torch.Tensor | None
    if matrix == "V":
        downstream_weight = value_to_logits.detach().cpu().clone()
    elif matrix == "O":
        downstream_weight = readout_weight.detach().cpu().clone()
    else:
        downstream_weight = None

    for batch in split_batches(
        task_id,
        total_count,
        generation_batch_size,
        calibration_seed,
        transforms,
        rho,
        three_vote_distractor_rho,
    ):
        context = batch.context.to(device)
        query = batch.query.to(device)
        with torch.no_grad():
            logits, details = model(context, query)
        teacher_logits.append(logits.detach().cpu())
        if matrix == "Q":
            matrix_inputs.append(batch.query.detach().cpu())
            attention_other.append(details["k"].detach().cpu())
            token_contributions.append(
                torch.einsum(
                    "bnd,cd->bnc",
                    details["v"],
                    value_to_logits,
                ).detach().cpu()
            )
        elif matrix == "K":
            matrix_inputs.append(batch.context.detach().cpu())
            attention_other.append(details["q"].detach().cpu())
            token_contributions.append(
                torch.einsum(
                    "bnd,cd->bnc",
                    details["v"],
                    value_to_logits,
                ).detach().cpu()
            )
        elif matrix == "V":
            matrix_inputs.append(
                torch.einsum(
                    "bn,bnd->bd", details["attention"], context
                ).detach().cpu()
            )
        elif matrix in {"O", EFFECTIVE_OUTPUT_MATRIX}:
            matrix_inputs.append(details["retrieved"].detach().cpu())
        elif matrix == EFFECTIVE_SCORE_MATRIX:
            matrix_inputs.append(batch.query.detach().cpu())
            matrix_inputs_secondary.append(batch.context.detach().cpu())
            token_contributions.append(
                torch.einsum(
                    "bnd,cd->bnc",
                    details["v"],
                    value_to_logits,
                ).detach().cpu()
            )
        else:
            raise AssertionError(f"Unhandled distillation matrix: {matrix}")
        observed += batch.count
        batch_count += 1

    if observed != total_count:
        raise AssertionError(f"Expected {total_count} examples, observed {observed}")
    combined_logits = torch.cat(teacher_logits, dim=0).contiguous()
    teacher_mean_square = float(combined_logits.square().mean().item())
    if not math.isfinite(teacher_mean_square) or teacher_mean_square <= 0:
        raise ValueError("Teacher-logit normalization is not positive and finite")
    return FunctionalDistillationData(
        matrix=matrix,
        task_id=task_id,
        calibration_seed=calibration_seed,
        matrix_inputs=torch.cat(matrix_inputs, dim=0).contiguous(),
        attention_other=(
            torch.cat(attention_other, dim=0).contiguous()
            if attention_other
            else None
        ),
        token_logit_contributions=(
            torch.cat(token_contributions, dim=0).contiguous()
            if token_contributions
            else None
        ),
        downstream_weight=downstream_weight,
        readout_bias=readout_bias,
        teacher_logits=combined_logits,
        gamma=float(model.gamma),
        teacher_mean_square=teacher_mean_square,
        teacher_logits_sha256=_tensor_collection_sha256(teacher_logits),
        example_count=observed,
        batch_count=batch_count,
        matrix_inputs_secondary=(
            torch.cat(matrix_inputs_secondary, dim=0).contiguous()
            if matrix_inputs_secondary
            else None
        ),
    )


def svd_factor_initialization(
    weight: torch.Tensor,
    rank: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight64 = weight.detach().cpu().to(torch.float64)
    maximum_rank = min(weight64.shape)
    if rank < 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [0, {maximum_rank}]")
    if rank == 0:
        return (
            torch.empty(weight64.shape[0], 0, dtype=dtype),
            torch.empty(0, weight64.shape[1], dtype=dtype),
        )
    u, singular_values, vh = torch.linalg.svd(weight64, full_matrices=False)
    roots = singular_values[:rank].sqrt()
    factor_a = u[:, :rank] * roots
    factor_b = roots[:, None] * vh[:rank]
    return factor_a.to(dtype=dtype), factor_b.to(dtype=dtype)


def replacement_from_factors(factor_a: torch.Tensor, factor_b: torch.Tensor) -> torch.Tensor:
    if factor_a.ndim != 2 or factor_b.ndim != 2:
        raise ValueError("Low-rank factors must be matrices")
    if factor_a.shape[1] != factor_b.shape[0]:
        raise ValueError("Low-rank factor inner dimensions do not match")
    return factor_a @ factor_b


def _low_rank_linear(
    inputs: torch.Tensor, factor_a: torch.Tensor, factor_b: torch.Tensor
) -> torch.Tensor:
    return F.linear(F.linear(inputs, factor_b), factor_a)


def functional_student_logits(
    data: FunctionalDistillationData,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    indices: torch.Tensor | slice,
    device: torch.device,
) -> torch.Tensor:
    matrix_inputs = data.matrix_inputs[indices].to(device)
    readout_bias = data.readout_bias.to(device)
    if data.matrix == "Q":
        if data.attention_other is None or data.token_logit_contributions is None:
            raise AssertionError("Q cache is missing fixed keys or token logits")
        q = _low_rank_linear(matrix_inputs, factor_a, factor_b)
        keys = data.attention_other[indices].to(device)
        attention_logits = data.gamma * torch.einsum("bd,bnd->bn", q, keys)
        attention = torch.softmax(attention_logits, dim=-1)
        token_logits = data.token_logit_contributions[indices].to(device)
        return torch.einsum("bn,bnc->bc", attention, token_logits) + readout_bias
    if data.matrix == "K":
        if data.attention_other is None or data.token_logit_contributions is None:
            raise AssertionError("K cache is missing fixed queries or token logits")
        keys = _low_rank_linear(matrix_inputs, factor_a, factor_b)
        queries = data.attention_other[indices].to(device)
        attention_logits = data.gamma * torch.einsum("bd,bnd->bn", queries, keys)
        attention = torch.softmax(attention_logits, dim=-1)
        token_logits = data.token_logit_contributions[indices].to(device)
        return torch.einsum("bn,bnc->bc", attention, token_logits) + readout_bias
    if data.matrix == EFFECTIVE_SCORE_MATRIX:
        if (
            data.matrix_inputs_secondary is None
            or data.token_logit_contributions is None
        ):
            raise AssertionError(
                "Effective-score cache is missing contexts or token logits"
            )
        # The effective score is M = gamma W_Q^T W_K and is factorized as
        # A @ B.  Scores are therefore (query @ A) dot (context @ B.T).
        query_factors = F.linear(matrix_inputs, factor_a.T)
        contexts = data.matrix_inputs_secondary[indices].to(device)
        context_factors = F.linear(contexts, factor_b)
        attention_logits = torch.einsum(
            "bk,bnk->bn", query_factors, context_factors
        )
        attention = torch.softmax(attention_logits, dim=-1)
        token_logits = data.token_logit_contributions[indices].to(device)
        return torch.einsum("bn,bnc->bc", attention, token_logits) + readout_bias
    transformed = _low_rank_linear(matrix_inputs, factor_a, factor_b)
    if data.matrix == EFFECTIVE_OUTPUT_MATRIX:
        return transformed + readout_bias
    if data.downstream_weight is None:
        raise AssertionError(f"{data.matrix} cache is missing its downstream weight")
    return F.linear(transformed, data.downstream_weight.to(device), readout_bias)


@torch.inference_mode()
def evaluate_factor_metrics(
    data: FunctionalDistillationData,
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    *,
    batch_size: int,
    epsilon: float,
    device: torch.device,
) -> dict[str, float]:
    squared_error = 0.0
    agreement = 0
    value_count = 0
    example_count = 0
    factor_a = factor_a.to(device)
    factor_b = factor_b.to(device)
    for start in range(0, data.example_count, batch_size):
        stop = min(start + batch_size, data.example_count)
        selected = slice(start, stop)
        student = functional_student_logits(data, factor_a, factor_b, selected, device)
        teacher = data.teacher_logits[selected].to(device)
        squared_error += float((student - teacher).square().sum().item())
        agreement += int(
            (student.argmax(dim=-1) == teacher.argmax(dim=-1)).sum().item()
        )
        value_count += int(teacher.numel())
        example_count += int(teacher.shape[0])
    if example_count != data.example_count or value_count == 0:
        raise AssertionError("Incomplete factor evaluation")
    raw_mse = squared_error / value_count
    return {
        "raw_logit_mse": raw_mse,
        "normalized_logit_mse": raw_mse / (data.teacher_mean_square + epsilon),
        "teacher_prediction_agreement": agreement / example_count,
    }


def _factor_norms(factor_a: torch.Tensor, factor_b: torch.Tensor) -> dict[str, float]:
    return {
        "factor_a_norm": float(torch.linalg.norm(factor_a.detach()).item()),
        "factor_b_norm": float(torch.linalg.norm(factor_b.detach()).item()),
        "product_norm": float(
            torch.linalg.norm(replacement_from_factors(factor_a, factor_b).detach()).item()
        ),
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def train_rank_group(
    data: FunctionalDistillationData,
    weight: torch.Tensor,
    ranks: list[int],
    config: DistillationTrainingConfig,
    *,
    optimization_seed: int,
    device: torch.device,
    horizon_steps: list[int] | None = None,
) -> dict[str, Any]:
    trainable_ranks = sorted(set(ranks))
    if not trainable_ranks or trainable_ranks[0] <= 0:
        raise ValueError("train_rank_group requires unique positive ranks")
    if trainable_ranks[-1] >= min(weight.shape):
        raise ValueError("Full-rank and rank-zero conditions are not optimized")
    horizons = sorted(set(horizon_steps or []))
    if any(step <= 0 or step > config.max_steps for step in horizons):
        raise ValueError("Horizon steps must be within the optimization interval")
    if any(
        step % config.evaluate_every != 0 and step != config.max_steps
        for step in horizons
    ):
        raise ValueError("Every horizon must coincide with an evaluation step")

    factor_as = nn.ParameterDict()
    factor_bs = nn.ParameterDict()
    initial_factors: dict[int, dict[str, torch.Tensor]] = {}
    for rank in trainable_ranks:
        factor_a, factor_b = svd_factor_initialization(weight, rank)
        key = f"rank_{rank:03d}"
        factor_as[key] = nn.Parameter(factor_a.to(device))
        factor_bs[key] = nn.Parameter(factor_b.to(device))
        initial_factors[rank] = {
            "factor_a": factor_a.clone(),
            "factor_b": factor_b.clone(),
        }

    optimizer = torch.optim.Adam(
        list(factor_as.parameters()) + list(factor_bs.parameters()),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    best: dict[int, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    horizon_checkpoints: dict[int, dict[int, dict[str, Any]]] = {}
    for rank in trainable_ranks:
        key = f"rank_{rank:03d}"
        metrics = evaluate_factor_metrics(
            data,
            factor_as[key],
            factor_bs[key],
            batch_size=config.batch_size,
            epsilon=config.epsilon,
            device=device,
        )
        best[rank] = {
            "step": 0,
            "factor_a": factor_as[key].detach().cpu().clone(),
            "factor_b": factor_bs[key].detach().cpu().clone(),
            **metrics,
        }
        history.append(
            {
                "step": 0,
                "rank": rank,
                "gradient_norm": 0.0,
                **metrics,
                **_factor_norms(factor_as[key], factor_bs[key]),
            }
        )

    generator = torch.Generator(device="cpu").manual_seed(optimization_seed)
    permutation = torch.randperm(data.example_count, generator=generator)
    cursor = 0
    evaluations_without_any_improvement = 0
    completed_steps = 0
    started = time.monotonic()
    for step in range(1, config.max_steps + 1):
        if cursor + config.batch_size > data.example_count:
            permutation = torch.randperm(data.example_count, generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + config.batch_size]
        cursor += config.batch_size
        optimizer.zero_grad(set_to_none=True)
        current_metrics: dict[int, dict[str, float]] = {}
        for rank in trainable_ranks:
            key = f"rank_{rank:03d}"
            student = functional_student_logits(
                data,
                factor_as[key],
                factor_bs[key],
                indices,
                device,
            )
            teacher = data.teacher_logits[indices].to(device)
            raw_mse = F.mse_loss(student, teacher)
            loss = raw_mse / (data.teacher_mean_square + config.epsilon)
            loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                list(factor_as.parameters()) + list(factor_bs.parameters()),
                config.gradient_clip_norm,
            ).item()
        )
        optimizer.step()
        completed_steps = step

        should_evaluate = step % config.evaluate_every == 0 or step == config.max_steps
        if not should_evaluate:
            continue
        any_improvement = False
        for rank in trainable_ranks:
            key = f"rank_{rank:03d}"
            metrics = evaluate_factor_metrics(
                data,
                factor_as[key],
                factor_bs[key],
                batch_size=config.batch_size,
                epsilon=config.epsilon,
                device=device,
            )
            if (
                metrics["normalized_logit_mse"]
                < best[rank]["normalized_logit_mse"] - config.improvement_tolerance
            ):
                any_improvement = True
                best[rank] = {
                    "step": step,
                    "factor_a": factor_as[key].detach().cpu().clone(),
                    "factor_b": factor_bs[key].detach().cpu().clone(),
                    **metrics,
                }
            current_metrics[rank] = metrics
            history.append(
                {
                    "step": step,
                    "rank": rank,
                    "gradient_norm": gradient_norm,
                    **metrics,
                    **_factor_norms(factor_as[key], factor_bs[key]),
                }
            )
        if step in horizons:
            horizon_checkpoints[step] = {}
            for rank in trainable_ranks:
                key = f"rank_{rank:03d}"
                selected = best[rank]
                horizon_checkpoints[step][rank] = {
                    "rank": rank,
                    "horizon_step": step,
                    "selected_step": int(selected["step"]),
                    "factor_a": selected["factor_a"].detach().cpu().clone(),
                    "factor_b": selected["factor_b"].detach().cpu().clone(),
                    "selected_raw_logit_mse": float(selected["raw_logit_mse"]),
                    "selected_normalized_logit_mse": float(
                        selected["normalized_logit_mse"]
                    ),
                    "selected_teacher_prediction_agreement": float(
                        selected["teacher_prediction_agreement"]
                    ),
                    "terminal_factor_a": factor_as[key].detach().cpu().clone(),
                    "terminal_factor_b": factor_bs[key].detach().cpu().clone(),
                    "terminal_raw_logit_mse": float(
                        current_metrics[rank]["raw_logit_mse"]
                    ),
                    "terminal_normalized_logit_mse": float(
                        current_metrics[rank]["normalized_logit_mse"]
                    ),
                    "terminal_teacher_prediction_agreement": float(
                        current_metrics[rank]["teacher_prediction_agreement"]
                    ),
                }
        if any_improvement:
            evaluations_without_any_improvement = 0
        else:
            evaluations_without_any_improvement += 1
        if evaluations_without_any_improvement >= config.patience_evaluations:
            break

    elapsed = time.monotonic() - started
    results: dict[int, dict[str, Any]] = {}
    for rank in trainable_ranks:
        key = f"rank_{rank:03d}"
        initial = next(
            row for row in history if row["rank"] == rank and row["step"] == 0
        )
        selected = best[rank]
        terminal_factor_a = factor_as[key].detach().cpu().clone()
        terminal_factor_b = factor_bs[key].detach().cpu().clone()
        terminal = next(
            row
            for row in reversed(history)
            if row["rank"] == rank and row["step"] == completed_steps
        )
        if (
            selected["normalized_logit_mse"]
            > initial["normalized_logit_mse"] + config.improvement_tolerance
        ):
            raise AssertionError("Selected trained checkpoint is worse than step zero")
        results[rank] = {
            "rank": rank,
            "initial_factor_a": initial_factors[rank]["factor_a"],
            "initial_factor_b": initial_factors[rank]["factor_b"],
            "factor_a": selected["factor_a"],
            "factor_b": selected["factor_b"],
            "selected_step": int(selected["step"]),
            "terminal_factor_a": terminal_factor_a,
            "terminal_factor_b": terminal_factor_b,
            "terminal_step": completed_steps,
            "initial_raw_logit_mse": float(initial["raw_logit_mse"]),
            "initial_normalized_logit_mse": float(
                initial["normalized_logit_mse"]
            ),
            "initial_teacher_prediction_agreement": float(
                initial["teacher_prediction_agreement"]
            ),
            "final_raw_logit_mse": float(selected["raw_logit_mse"]),
            "final_normalized_logit_mse": float(
                selected["normalized_logit_mse"]
            ),
            "final_teacher_prediction_agreement": float(
                selected["teacher_prediction_agreement"]
            ),
            "terminal_raw_logit_mse": float(terminal["raw_logit_mse"]),
            "terminal_normalized_logit_mse": float(
                terminal["normalized_logit_mse"]
            ),
            "terminal_teacher_prediction_agreement": float(
                terminal["teacher_prediction_agreement"]
            ),
            "distillation_loss_improvement": float(
                initial["normalized_logit_mse"]
                - selected["normalized_logit_mse"]
            ),
            "initial_product_sha256": tensor_sha256(
                replacement_from_factors(
                    initial_factors[rank]["factor_a"],
                    initial_factors[rank]["factor_b"],
                )
            ),
            "trained_product_sha256": tensor_sha256(
                replacement_from_factors(selected["factor_a"], selected["factor_b"])
            ),
            "terminal_product_sha256": tensor_sha256(
                replacement_from_factors(terminal_factor_a, terminal_factor_b)
            ),
        }
    return {
        "ranks": results,
        "history": history,
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "optimization_seed": optimization_seed,
        "completed_steps": completed_steps,
        "elapsed_seconds": elapsed,
        "stopped_early": completed_steps < config.max_steps,
        "horizon_checkpoints": horizon_checkpoints,
    }


def local_output_optimal_residual_fraction(
    left_gram_eigenvalues: torch.Tensor, rank: int
) -> float:
    values = left_gram_eigenvalues.detach().cpu().to(torch.float64).clamp_min(0)
    if rank < 0 or rank > values.numel():
        raise ValueError("Invalid rank")
    total = float(values.sum().item())
    if total <= 0:
        return 0.0
    return float(values[rank:].sum().item() / total)
