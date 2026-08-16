from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .ten_task_attention import (
    D_MODEL,
    N_CONTEXT,
    TASK_COUNT,
    TASK_NAMES,
    OrthogonalTransforms,
    TenTaskAttention,
    evaluate_tasks,
    generate_task_batch,
    split_batches,
    stable_seed,
)


FOUR_GRAM_METHODS = (
    "query_input_gram",
    "query_output_gram",
    "key_input_gram",
    "key_output_gram",
)
ALL_EFFECTIVE_SCORE_METHODS = FOUR_GRAM_METHODS + ("effective_m_svd",)


@dataclass(frozen=True)
class EffectiveScoreGrams:
    query_input: torch.Tensor
    key_input: torch.Tensor
    key_output: torch.Tensor
    query_output: torch.Tensor
    query_observation_count: int
    key_observation_count: int


@dataclass(frozen=True)
class EffectiveScoreBasis:
    matrix: torch.Tensor
    svd_u: torch.Tensor
    svd_s: torch.Tensor
    svd_vh: torch.Tensor
    query_input_values: torch.Tensor
    query_input_vectors: torch.Tensor
    key_input_values: torch.Tensor
    key_input_vectors: torch.Tensor
    key_output_values: torch.Tensor
    key_output_vectors: torch.Tensor
    query_output_values: torch.Tensor
    query_output_vectors: torch.Tensor


@dataclass(frozen=True)
class ExposureMatchedTrainingResult:
    history: list[dict[str, Any]]
    optimizer_state: dict[str, Any]
    outer_updates_completed: int
    task_batch_counts: dict[str, int]
    task_example_counts: dict[str, int]
    validation_accuracy: dict[str, float]
    elapsed_seconds: float
    examples_per_second: float


@dataclass(frozen=True)
class CachedEffectiveScoreBatch:
    context: torch.Tensor
    query: torch.Tensor
    values: torch.Tensor
    targets: torch.Tensor
    original_scores: torch.Tensor
    original_attention: torch.Tensor


def effective_score_matrix(
    model: TenTaskAttention, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Return gamma W_Q^T W_K in the observed input coordinates."""

    q_weight = model.q_proj.weight.detach().to(device="cpu", dtype=dtype)
    k_weight = model.k_proj.weight.detach().to(device="cpu", dtype=dtype)
    return float(model.gamma) * (q_weight.T @ k_weight)


def forward_with_effective_score(
    model: TenTaskAttention,
    context: torch.Tensor,
    query: torch.Tensor,
    matrix: torch.Tensor,
    *,
    precomputed_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the frozen value/output path with scores supplied directly by M."""

    matrix = matrix.to(device=query.device, dtype=query.dtype)
    scores = torch.einsum("bd,de,bne->bn", query, matrix, context)
    attention = torch.softmax(scores, dim=-1)
    values = model.v_proj(context) if precomputed_values is None else precomputed_values
    retrieved = torch.einsum("bn,bnd->bd", attention, values)
    attention_output = model.o_proj(retrieved)
    logits = model.readout(attention_output)
    return logits, {
        "attention_logits": scores,
        "attention": attention,
        "values": values,
        "retrieved": retrieved,
        "attention_output": attention_output,
    }


@torch.inference_mode()
def direct_score_audit(
    model: TenTaskAttention,
    context: torch.Tensor,
    query: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    # Audit the algebraic identity in float64. In float32, the factored path
    # ((q W_Q^T) (c W_K^T)^T) and the direct path (q M c^T) use different
    # reduction orders, so otherwise equivalent models can differ by a few
    # ulps after training has increased the downstream logit scale.
    audit_model = deepcopy(model).eval().to(device=device, dtype=torch.float64)
    context = context.to(device=device, dtype=torch.float64)
    query = query.to(device=device, dtype=torch.float64)
    original_logits, original_details = audit_model(context, query)
    matrix = effective_score_matrix(audit_model, dtype=torch.float64).to(device)
    direct_logits, direct_details = forward_with_effective_score(
        audit_model, context, query, matrix
    )
    score_error = float(
        (original_details["attention_logits"] - direct_details["attention_logits"])
        .abs()
        .max()
        .item()
    )
    attention_error = float(
        (original_details["attention"] - direct_details["attention"]).abs().max().item()
    )
    logit_error = float((original_logits - direct_logits).abs().max().item())
    predictions_match = bool(
        torch.equal(original_logits.argmax(-1), direct_logits.argmax(-1))
    )
    return {
        "passed": score_error <= 2e-4
        and attention_error <= 2e-5
        and logit_error <= 2e-5
        and predictions_match,
        "maximum_score_absolute_error": score_error,
        "maximum_attention_absolute_error": attention_error,
        "maximum_class_logit_absolute_error": logit_error,
        "predictions_match": predictions_match,
    }


@torch.inference_mode()
def collect_effective_score_grams(
    model: TenTaskAttention,
    task_id: int,
    total_count: int,
    batch_size: int,
    split_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    three_vote_distractor_rho: float,
) -> EffectiveScoreGrams:
    matrix = effective_score_matrix(model)
    query_input_sum = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
    key_input_sum = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
    key_output_sum = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
    query_output_sum = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
    query_count = 0
    key_count = 0
    for batch in split_batches(
        task_id,
        total_count,
        batch_size,
        split_seed,
        transforms,
        rho,
        three_vote_distractor_rho,
    ):
        query_inputs = batch.query.to(torch.float64)
        key_inputs = batch.context.reshape(-1, D_MODEL).to(torch.float64)
        key_outputs = key_inputs @ matrix.T
        query_outputs = query_inputs @ matrix
        query_input_sum += query_inputs.T @ query_inputs
        key_input_sum += key_inputs.T @ key_inputs
        key_output_sum += key_outputs.T @ key_outputs
        query_output_sum += query_outputs.T @ query_outputs
        query_count += int(query_inputs.shape[0])
        key_count += int(key_inputs.shape[0])
    if query_count != total_count:
        raise AssertionError(f"Expected {total_count} query observations, got {query_count}")
    if key_count != total_count * N_CONTEXT:
        raise AssertionError(
            f"Expected {total_count * N_CONTEXT} context observations, got {key_count}"
        )
    return EffectiveScoreGrams(
        query_input=query_input_sum / query_count,
        key_input=key_input_sum / key_count,
        key_output=key_output_sum / key_count,
        query_output=query_output_sum / query_count,
        query_observation_count=query_count,
        key_observation_count=key_count,
    )


def effective_score_gram_audit(
    matrix: torch.Tensor,
    grams: EffectiveScoreGrams,
    *,
    atol: float = 1e-9,
) -> dict[str, Any]:
    matrix = matrix.to(torch.float64)
    symmetry_errors = {
        name: float(torch.linalg.norm(value - value.T).item())
        for name, value in (
            ("query_input", grams.query_input),
            ("key_input", grams.key_input),
            ("key_output", grams.key_output),
            ("query_output", grams.query_output),
        )
    }
    expected_key_output = matrix @ grams.key_input @ matrix.T
    expected_query_output = matrix.T @ grams.query_input @ matrix
    key_output_equivalence = float(
        torch.linalg.norm(grams.key_output - expected_key_output).item()
        / max(float(torch.linalg.norm(expected_key_output).item()), 1e-12)
    )
    query_output_equivalence = float(
        torch.linalg.norm(grams.query_output - expected_query_output).item()
        / max(float(torch.linalg.norm(expected_query_output).item()), 1e-12)
    )
    minimum_eigenvalues = {
        name: float(torch.linalg.eigvalsh(value).min().item())
        for name, value in (
            ("query_input", grams.query_input),
            ("key_input", grams.key_input),
            ("key_output", grams.key_output),
            ("query_output", grams.query_output),
        )
    }
    passed = (
        max(symmetry_errors.values()) <= atol
        and key_output_equivalence <= atol
        and query_output_equivalence <= atol
        and min(minimum_eigenvalues.values()) >= -atol
    )
    return {
        "passed": passed,
        "symmetry_errors": symmetry_errors,
        "minimum_eigenvalues": minimum_eigenvalues,
        "key_output_equivalence_relative_error": key_output_equivalence,
        "query_output_equivalence_relative_error": query_output_equivalence,
        "query_observation_count": grams.query_observation_count,
        "key_observation_count": grams.key_observation_count,
    }


def _descending_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh(matrix.to(torch.float64))
    return values.flip(0), vectors.flip(1)


def build_effective_score_basis(
    matrix: torch.Tensor, grams: EffectiveScoreGrams
) -> EffectiveScoreBasis:
    matrix = matrix.detach().cpu().to(torch.float64)
    svd_u, svd_s, svd_vh = torch.linalg.svd(matrix, full_matrices=False)
    query_input_values, query_input_vectors = _descending_eigh(grams.query_input)
    key_input_values, key_input_vectors = _descending_eigh(grams.key_input)
    key_output_values, key_output_vectors = _descending_eigh(grams.key_output)
    query_output_values, query_output_vectors = _descending_eigh(grams.query_output)
    return EffectiveScoreBasis(
        matrix=matrix,
        svd_u=svd_u,
        svd_s=svd_s,
        svd_vh=svd_vh,
        query_input_values=query_input_values,
        query_input_vectors=query_input_vectors,
        key_input_values=key_input_values,
        key_input_vectors=key_input_vectors,
        key_output_values=key_output_values,
        key_output_vectors=key_output_vectors,
        query_output_values=query_output_values,
        query_output_vectors=query_output_vectors,
    )


def _projector(vectors: torch.Tensor, rank: int) -> torch.Tensor:
    selected = vectors[:, :rank]
    return selected @ selected.T


def reconstruct_effective_score(
    basis: EffectiveScoreBasis,
    method: str,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor | None, str | None]:
    if rank < 0 or rank > D_MODEL:
        raise ValueError(f"rank must be in [0, {D_MODEL}]")
    if method == "effective_m_svd":
        if rank == 0:
            return torch.zeros_like(basis.matrix), None, None
        reconstructed = (
            basis.svd_u[:, :rank] * basis.svd_s[:rank]
        ) @ basis.svd_vh[:rank]
        return reconstructed, None, None
    if method == "key_input_gram":
        projector = _projector(basis.key_input_vectors, rank)
        return basis.matrix @ projector, projector, "right"
    if method == "key_output_gram":
        projector = _projector(basis.key_output_vectors, rank)
        return projector @ basis.matrix, projector, "left"
    if method == "query_input_gram":
        projector = _projector(basis.query_input_vectors, rank)
        return projector @ basis.matrix, projector, "left"
    if method == "query_output_gram":
        projector = _projector(basis.query_output_vectors, rank)
        return basis.matrix @ projector, projector, "right"
    raise ValueError(f"Unknown effective-score reconstruction method: {method}")


def random_effective_score_reconstruction(
    matrix: torch.Tensor,
    rank: int,
    seed: int,
    side: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rank < 0 or rank > D_MODEL:
        raise ValueError(f"rank must be in [0, {D_MODEL}]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if rank == 0:
        projector = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
    else:
        raw = torch.randn(D_MODEL, rank, generator=generator, dtype=torch.float64)
        vectors, _ = torch.linalg.qr(raw, mode="reduced")
        projector = vectors @ vectors.T
    matrix = matrix.to(torch.float64)
    if side == "left":
        return projector @ matrix, projector
    if side == "right":
        return matrix @ projector, projector
    raise ValueError("side must be 'left' or 'right'")


def effective_score_reconstruction_audit(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    projector: torch.Tensor | None,
    requested_rank: int,
) -> dict[str, Any]:
    original = original.to(torch.float64)
    reconstructed = reconstructed.to(torch.float64)
    singular_values = torch.linalg.svdvals(reconstructed)
    tolerance = max(reconstructed.shape) * torch.finfo(torch.float64).eps * max(
        float(singular_values.max().item()) if singular_values.numel() else 0.0, 1.0
    )
    numerical_rank = int((singular_values > tolerance).sum().item())
    denominator = max(float(torch.linalg.norm(original).item()), 1e-12)
    result: dict[str, Any] = {
        "requested_rank": requested_rank,
        "numerical_rank": numerical_rank,
        "rank_bound_satisfied": numerical_rank <= requested_rank,
        "numerical_rank_tolerance": tolerance,
        "relative_matrix_error": float(
            torch.linalg.norm(original - reconstructed).item() / denominator
        ),
    }
    if projector is not None:
        projector_denominator = max(float(torch.linalg.norm(projector).item()), 1.0)
        result.update(
            {
                "projector_symmetry_error": float(
                    torch.linalg.norm(projector - projector.T).item()
                    / projector_denominator
                ),
                "projector_idempotence_error": float(
                    torch.linalg.norm(projector @ projector - projector).item()
                    / projector_denominator
                ),
            }
        )
    return result


def cumulative_svd_power(basis: EffectiveScoreBasis, rank: int) -> float:
    if rank == 0:
        return 0.0
    power = basis.svd_s.square()
    return float((power[:rank].sum() / power.sum()).item())


def train_joint_exposure_matched(
    model: TenTaskAttention,
    transforms: OrthogonalTransforms,
    model_seed: int,
    phase: str,
    data_namespace: str,
    rho: float,
    three_vote_distractor_rho: float,
    device: torch.device,
    *,
    learning_rate: float,
    weight_decay: float,
    task_batch_size: int,
    outer_updates: int,
    evaluate_every: int,
    evaluation_batch_size: int,
    validation_count: int,
    optimizer_state: dict[str, Any] | None = None,
    start_update: int = 0,
    initial_task_batch_counts: dict[str, int] | None = None,
    initial_history: list[dict[str, Any]] | None = None,
    checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ExposureMatchedTrainingResult:
    if start_update < 0 or outer_updates <= start_update:
        raise ValueError("outer_updates must be greater than start_update")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    counts = {
        name: int((initial_task_batch_counts or {}).get(name, 0)) for name in TASK_NAMES
    }
    history: list[dict[str, Any]] = list(initial_history or [])
    final_validation: dict[str, float] = {}
    started = time.monotonic()
    segment_examples = 0
    for update in range(start_update + 1, outer_updates + 1):
        contexts: list[torch.Tensor] = []
        queries: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for task_id, task_name in enumerate(TASK_NAMES):
            batch = generate_task_batch(
                task_id,
                task_batch_size,
                stable_seed(
                    data_namespace,
                    phase,
                    "training_example",
                    model_seed,
                    task_id,
                    update,
                ),
                transforms,
                rho,
                three_vote_distractor_rho,
            )
            contexts.append(batch.context)
            queries.append(batch.query)
            targets.append(batch.targets)
            counts[task_name] += 1
            segment_examples += batch.count
        context = torch.cat(contexts, dim=0).to(device)
        query = torch.cat(queries, dim=0).to(device)
        target = torch.cat(targets, dim=0).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(context, query)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        if update == 1 or update % evaluate_every == 0 or update == outer_updates:
            final_validation = evaluate_tasks(
                model,
                list(range(TASK_COUNT)),
                validation_count,
                evaluation_batch_size,
                data_namespace,
                model_seed,
                "validation",
                transforms,
                rho,
                device,
                three_vote_distractor_rho,
            )
            elapsed = time.monotonic() - started
            row = {
                "outer_update": update,
                "train_loss": float(loss.detach().cpu().item()),
                "validation_accuracy": final_validation,
                "mean_validation_accuracy": float(
                    sum(final_validation.values()) / len(final_validation)
                ),
                "minimum_validation_accuracy": float(min(final_validation.values())),
                "elapsed_seconds": elapsed,
                "training_examples_per_second": segment_examples / max(elapsed, 1e-12),
            }
            history.append(row)
            print(
                {
                    "event": "exposure_matched_training_progress",
                    "seed": model_seed,
                    **row,
                },
                flush=True,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "outer_update": update,
                        "optimizer_state": optimizer.state_dict(),
                        "history": history,
                        "task_batch_counts": dict(counts),
                        "validation_accuracy": final_validation,
                    }
                )
    elapsed = time.monotonic() - started
    expected_count = outer_updates
    if any(value != expected_count for value in counts.values()):
        raise AssertionError(
            f"Task exposure mismatch: expected {expected_count}, observed {counts}"
        )
    return ExposureMatchedTrainingResult(
        history=history,
        optimizer_state=optimizer.state_dict(),
        outer_updates_completed=outer_updates,
        task_batch_counts=counts,
        task_example_counts={name: value * task_batch_size for name, value in counts.items()},
        validation_accuracy=final_validation,
        elapsed_seconds=elapsed,
        examples_per_second=segment_examples / max(elapsed, 1e-12),
    )


@torch.inference_mode()
def prepare_effective_score_cache(
    model: TenTaskAttention,
    task_id: int,
    total_count: int,
    batch_size: int,
    split_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    three_vote_distractor_rho: float,
    device: torch.device,
) -> list[CachedEffectiveScoreBatch]:
    model.eval().to(device)
    matrix = effective_score_matrix(model, dtype=torch.float32).to(device)
    cache: list[CachedEffectiveScoreBatch] = []
    for batch in split_batches(
        task_id,
        total_count,
        batch_size,
        split_seed,
        transforms,
        rho,
        three_vote_distractor_rho,
    ):
        context = batch.context.to(device)
        query = batch.query.to(device)
        values = model.v_proj(context)
        original_scores = torch.einsum("bd,de,bne->bn", query, matrix, context)
        original_attention = torch.softmax(original_scores, dim=-1)
        cache.append(
            CachedEffectiveScoreBatch(
                context=context,
                query=query,
                values=values,
                targets=batch.targets.to(device),
                original_scores=original_scores,
                original_attention=original_attention,
            )
        )
    if sum(int(batch.targets.shape[0]) for batch in cache) != total_count:
        raise AssertionError("Evaluation cache count mismatch")
    return cache


@torch.inference_mode()
def evaluate_cached_effective_score(
    model: TenTaskAttention,
    cache: Iterable[CachedEffectiveScoreBatch],
    matrix: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    correct = 0
    observed = 0
    kl_sum = 0.0
    centered_score_squared_sum = 0.0
    score_count = 0
    for batch in cache:
        logits, details = forward_with_effective_score(
            model,
            batch.context,
            batch.query,
            matrix,
            precomputed_values=batch.values,
        )
        attention = details["attention"]
        scores = details["attention_logits"]
        correct += int((logits.argmax(-1) == batch.targets).sum().item())
        observed += int(batch.targets.shape[0])
        kl = batch.original_attention * (
            torch.log(batch.original_attention.clamp_min(1e-12))
            - torch.log(attention.clamp_min(1e-12))
        )
        kl_sum += float(kl.sum(dim=-1).sum().item())
        centered_original = batch.original_scores - batch.original_scores.mean(
            dim=-1, keepdim=True
        )
        centered_reconstructed = scores - scores.mean(dim=-1, keepdim=True)
        centered_score_squared_sum += float(
            (centered_original - centered_reconstructed).square().sum().item()
        )
        score_count += int(scores.numel())
    return {
        "accuracy": correct / observed,
        "mean_attention_kl": kl_sum / observed,
        "centered_score_mse": centered_score_squared_sum / score_count,
    }


@torch.inference_mode()
def gauge_invariance_audit(
    model: TenTaskAttention,
    context: torch.Tensor,
    query: torch.Tensor,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw_left = torch.randn(D_MODEL, D_MODEL, generator=generator, dtype=torch.float64)
    raw_right = torch.randn(D_MODEL, D_MODEL, generator=generator, dtype=torch.float64)
    left, _ = torch.linalg.qr(raw_left)
    right, _ = torch.linalg.qr(raw_right)
    scales = torch.logspace(-0.3, 0.3, D_MODEL, dtype=torch.float64)
    transform = left @ torch.diag(scales) @ right.T
    inverse_transpose = torch.linalg.inv(transform).T
    q_weight = model.q_proj.weight.detach().cpu().to(torch.float64)
    k_weight = model.k_proj.weight.detach().cpu().to(torch.float64)
    q_transformed = transform @ q_weight
    k_transformed = inverse_transpose @ k_weight
    original_matrix = float(model.gamma) * (q_weight.T @ k_weight)
    transformed_matrix = float(model.gamma) * (q_transformed.T @ k_transformed)
    denominator = max(float(torch.linalg.norm(original_matrix).item()), 1e-12)
    matrix_error = float(
        torch.linalg.norm(original_matrix - transformed_matrix).item() / denominator
    )
    context = context.detach().cpu().to(torch.float64)
    query = query.detach().cpu().to(torch.float64)
    original_scores = float(model.gamma) * torch.einsum(
        "bd,bnd->bn", query @ q_weight.T, context @ k_weight.T
    )
    transformed_scores = float(model.gamma) * torch.einsum(
        "bd,bnd->bn", query @ q_transformed.T, context @ k_transformed.T
    )
    score_error = float((original_scores - transformed_scores).abs().max().item())
    original_attention = torch.softmax(original_scores, dim=-1)
    transformed_attention = torch.softmax(transformed_scores, dim=-1)
    attention_error = float((original_attention - transformed_attention).abs().max().item())
    return {
        "passed": matrix_error <= 1e-12
        and score_error <= 1e-10
        and attention_error <= 1e-12,
        "transform_condition_number": float(torch.linalg.cond(transform).item()),
        "effective_matrix_relative_error": matrix_error,
        "maximum_score_absolute_error": score_error,
        "maximum_attention_absolute_error": attention_error,
    }
