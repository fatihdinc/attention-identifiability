from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import time
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


KEY_DIM = 32
LABEL_COUNT = 32
CATEGORY_COUNT = 8
TASK_COUNT = 10
N_CONTEXT = 64
D_MODEL = 128

A_SLICE = slice(0, 32)
B_SLICE = slice(32, 64)
LABEL_SLICE = slice(64, 96)
CATEGORY_SLICE = slice(96, 104)
PRIORITY_INDEX = 104
TASK_SLICE = slice(105, 115)
CONSTANT_INDEX = 115
RESERVED_SLICE = slice(116, 128)

TASK_NAMES = (
    "exact_a_lookup",
    "noisy_a_lookup",
    "partial_a_lookup",
    "b_lookup",
    "two_key_lookup",
    "category_filtered_lookup",
    "highest_priority",
    "lowest_priority",
    "category_majority_vote",
    "three_item_vote",
)

TRAINING_CONFIGURATIONS: dict[str, tuple[int, ...]] = {
    **{f"task_{task_id + 1}": (task_id,) for task_id in range(TASK_COUNT)},
    "retrieval_mixture": (0, 1, 2, 3, 4, 5),
    "rule_mixture": (0, 5, 6, 7, 8),
    "aggregation_mixture": (1, 4, 8, 9),
    "all_tasks": tuple(range(TASK_COUNT)),
}

MATRIX_TO_PARAMETER = {
    "Q": "q_proj.weight",
    "K": "k_proj.weight",
    "V": "v_proj.weight",
    "O": "o_proj.weight",
}


@dataclass(frozen=True)
class OrthogonalTransforms:
    context: torch.Tensor
    query: torch.Tensor
    context_seed: int
    query_seed: int


@dataclass(frozen=True)
class TaskBatch:
    task_id: int
    context_latent: torch.Tensor
    query_latent: torch.Tensor
    context: torch.Tensor
    query: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, torch.Tensor]

    @property
    def count(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, Any]]
    validation_accuracy: dict[str, float]
    optimizer_state: dict[str, Any]
    steps_completed: int
    elapsed_seconds: float
    examples_per_second: float
    qualified: bool


def stable_seed(*parts: Any) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little") % (2**63 - 1)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _orthogonal_matrix(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(D_MODEL, D_MODEL, generator=generator, dtype=torch.float64)
    matrix, triangular = torch.linalg.qr(raw)
    signs = torch.sign(torch.diag(triangular))
    signs[signs == 0] = 1
    matrix = matrix * signs.unsqueeze(0)
    return matrix.to(torch.float32)


def make_transforms(model_seed: int) -> OrthogonalTransforms:
    context_seed = stable_seed("context_transform", model_seed)
    query_seed = stable_seed("query_transform", model_seed)
    return OrthogonalTransforms(
        context=_orthogonal_matrix(context_seed),
        query=_orthogonal_matrix(query_seed),
        context_seed=context_seed,
        query_seed=query_seed,
    )


def audit_transforms(transforms: OrthogonalTransforms, atol: float = 2e-5) -> dict[str, Any]:
    identity = torch.eye(D_MODEL, dtype=torch.float64)
    context = transforms.context.to(torch.float64)
    query = transforms.query.to(torch.float64)
    context_error = float(torch.linalg.norm(context.T @ context - identity).item())
    query_error = float(torch.linalg.norm(query.T @ query - identity).item())
    cross_difference = float(torch.linalg.norm(context - query).item())
    context_rank = int(torch.linalg.matrix_rank(context).item())
    query_rank = int(torch.linalg.matrix_rank(query).item())
    checks = {
        "context_orthogonal": context_error <= atol,
        "query_orthogonal": query_error <= atol,
        "context_full_rank": context_rank == D_MODEL,
        "query_full_rank": query_rank == D_MODEL,
        "transforms_are_distinct": cross_difference > 1e-3,
        "seeds_are_distinct": transforms.context_seed != transforms.query_seed,
    }
    return {
        "passed": all(checks.values()),
        "context_orthogonality_error": context_error,
        "query_orthogonality_error": query_error,
        "context_rank": context_rank,
        "query_rank": query_rank,
        "cross_transform_difference": cross_difference,
        **checks,
    }


def _gather_token(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0])
    return values[batch, indices]


def _set_token(values: torch.Tensor, indices: torch.Tensor, replacements: torch.Tensor) -> None:
    batch = torch.arange(values.shape[0])
    values[batch, indices] = replacements


def _different_labels(
    labels: torch.Tensor, generator: torch.Generator, count: int = 1
) -> torch.Tensor:
    offsets = torch.randint(
        1,
        LABEL_COUNT,
        (labels.shape[0], count),
        generator=generator,
        dtype=torch.long,
    )
    return (labels.unsqueeze(1) + offsets) % LABEL_COUNT


def _permute_tokens(
    a: torch.Tensor,
    b: torch.Tensor,
    labels: torch.Tensor,
    categories: torch.Tensor,
    priorities: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    permutation = torch.rand(a.shape[0], a.shape[1], generator=generator).argsort(dim=1)

    def gather(values: torch.Tensor) -> torch.Tensor:
        index = permutation
        while index.ndim < values.ndim:
            index = index.unsqueeze(-1)
        return torch.gather(values, 1, index.expand_as(values))

    return (
        gather(a),
        gather(b),
        gather(labels),
        gather(categories),
        gather(priorities),
    )


def _base_fields(
    count: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(KEY_DIM)
    a = torch.randn(count, N_CONTEXT, KEY_DIM, generator=generator) * scale
    b = torch.randn(count, N_CONTEXT, KEY_DIM, generator=generator) * scale
    labels = torch.randint(0, LABEL_COUNT, (count, N_CONTEXT), generator=generator)
    categories = torch.randint(0, CATEGORY_COUNT, (count, N_CONTEXT), generator=generator)
    priorities = 2.0 * torch.rand(count, N_CONTEXT, generator=generator) - 1.0
    return a, b, labels, categories, priorities


def _latent_records(
    a: torch.Tensor,
    b: torch.Tensor,
    labels: torch.Tensor,
    categories: torch.Tensor,
    priorities: torch.Tensor,
    query_a: torch.Tensor,
    query_b: torch.Tensor,
    query_categories: torch.Tensor,
    task_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = a.shape[0]
    context = torch.cat(
        [
            a,
            b,
            F.one_hot(labels, LABEL_COUNT).to(torch.float32),
            F.one_hot(categories, CATEGORY_COUNT).to(torch.float32),
            priorities.unsqueeze(-1),
            torch.zeros(count, N_CONTEXT, TASK_COUNT),
            torch.ones(count, N_CONTEXT, 1),
            torch.zeros(count, N_CONTEXT, 12),
        ],
        dim=-1,
    )
    task_code = F.one_hot(
        torch.full((count,), task_id, dtype=torch.long), TASK_COUNT
    ).to(torch.float32)
    query = torch.cat(
        [
            query_a,
            query_b,
            torch.zeros(count, LABEL_COUNT),
            query_categories,
            torch.zeros(count, 1),
            task_code,
            torch.ones(count, 1),
            torch.zeros(count, 12),
        ],
        dim=-1,
    )
    if context.shape[-1] != D_MODEL or query.shape[-1] != D_MODEL:
        raise AssertionError("Latent record has the wrong dimensionality")
    return context, query


def generate_task_batch(
    task_id: int,
    count: int,
    seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    three_vote_distractor_rho: float = 0.75,
) -> TaskBatch:
    if task_id < 0 or task_id >= TASK_COUNT:
        raise ValueError(f"task_id must be in [0, {TASK_COUNT - 1}]")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must be strictly between zero and one")
    if not 0.0 <= three_vote_distractor_rho < 1.0:
        raise ValueError("three_vote_distractor_rho must be in [0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    a, b, labels, categories, priorities = _base_fields(count, generator)
    query_a = torch.zeros(count, KEY_DIM)
    query_b = torch.zeros(count, KEY_DIM)
    query_categories = torch.zeros(count, CATEGORY_COUNT)
    metadata: dict[str, torch.Tensor] = {}

    if task_id in (0, 1, 2, 3, 4, 5):
        target_indices = torch.randint(0, N_CONTEXT, (count,), generator=generator)
        targets = _gather_token(labels, target_indices)
        target_a = _gather_token(a, target_indices).clone()
        target_b = _gather_token(b, target_indices).clone()
        metadata["target_indices"] = target_indices

        if task_id == 0:
            query_a = target_a
        elif task_id == 1:
            noise = torch.randn(count, KEY_DIM, generator=generator) / math.sqrt(KEY_DIM)
            query_a = rho * target_a + math.sqrt(1.0 - rho**2) * noise
        elif task_id == 2:
            observed = torch.rand(count, KEY_DIM, generator=generator).argsort(dim=1)[:, :16]
            mask = torch.zeros(count, KEY_DIM)
            mask.scatter_(1, observed, 1.0)
            query_a = target_a * mask
            metadata["partial_mask"] = mask
        elif task_id == 3:
            query_b = target_b
        elif task_id == 4:
            query_a = target_a
            query_b = target_b
            distractor_a = (target_indices + 1) % N_CONTEXT
            distractor_b = (target_indices + 2) % N_CONTEXT
            _set_token(a, distractor_a, target_a)
            _set_token(b, distractor_b, target_b)
            replacements = _different_labels(targets, generator, 2)
            _set_token(labels, distractor_a, replacements[:, 0])
            _set_token(labels, distractor_b, replacements[:, 1])
            metadata["a_distractor_indices"] = distractor_a
            metadata["b_distractor_indices"] = distractor_b
        else:
            query_a = target_a
            target_categories = _gather_token(categories, target_indices)
            query_categories = F.one_hot(target_categories, CATEGORY_COUNT).to(torch.float32)
            distractor = (target_indices + 1) % N_CONTEXT
            _set_token(a, distractor, target_a)
            category_offsets = torch.randint(
                1, CATEGORY_COUNT, (count,), generator=generator
            )
            _set_token(
                categories,
                distractor,
                (target_categories + category_offsets) % CATEGORY_COUNT,
            )
            _set_token(labels, distractor, _different_labels(targets, generator)[:, 0])
            metadata["category_distractor_indices"] = distractor

    elif task_id in (6, 7):
        query_category = torch.randint(0, CATEGORY_COUNT, (count,), generator=generator)
        excluded_categories = torch.randint(
            0, CATEGORY_COUNT - 1, (count, N_CONTEXT), generator=generator
        )
        excluded_categories += (excluded_categories >= query_category.unsqueeze(1)).long()
        categories = excluded_categories
        categories[:, :8] = query_category.unsqueeze(1)
        distinct_labels = torch.rand(count, LABEL_COUNT, generator=generator).argsort(dim=1)[:, :8]
        labels[:, :8] = distinct_labels
        if task_id == 6:
            priorities[:, :8] = -1.0 + 1.5 * torch.rand(count, 8, generator=generator)
            local_indices = priorities[:, :8].argmax(dim=1)
            priorities[:, 8] = 0.9
        else:
            priorities[:, :8] = -0.5 + 1.5 * torch.rand(count, 8, generator=generator)
            local_indices = priorities[:, :8].argmin(dim=1)
            priorities[:, 8] = -0.9
        targets = _gather_token(labels[:, :8], local_indices)
        query_categories = F.one_hot(query_category, CATEGORY_COUNT).to(torch.float32)
        metadata["query_category"] = query_category
        metadata["pre_permutation_target_indices"] = local_indices
        a, b, labels, categories, priorities = _permute_tokens(
            a, b, labels, categories, priorities, generator
        )

    elif task_id == 8:
        query_category = torch.randint(0, CATEGORY_COUNT, (count,), generator=generator)
        excluded_categories = torch.randint(
            0, CATEGORY_COUNT - 1, (count, N_CONTEXT), generator=generator
        )
        excluded_categories += (excluded_categories >= query_category.unsqueeze(1)).long()
        categories = excluded_categories
        categories[:, :7] = query_category.unsqueeze(1)
        majority = torch.randint(0, LABEL_COUNT, (count,), generator=generator)
        labels[:, :4] = majority.unsqueeze(1)
        nonmajor_offsets = torch.rand(count, LABEL_COUNT - 1, generator=generator).argsort(dim=1)[
            :, :3
        ]
        labels[:, 4:7] = (majority.unsqueeze(1) + 1 + nonmajor_offsets) % LABEL_COUNT
        targets = majority
        query_categories = F.one_hot(query_category, CATEGORY_COUNT).to(torch.float32)
        metadata["query_category"] = query_category
        a, b, labels, categories, priorities = _permute_tokens(
            a, b, labels, categories, priorities, generator
        )

    else:
        majority = torch.randint(0, LABEL_COUNT, (count,), generator=generator)
        labels[:, 0] = majority
        labels[:, 1] = majority
        labels[:, 2] = _different_labels(majority, generator)[:, 0]
        metadata["voter_keys"] = a[:, :3].clone()
        metadata["voter_labels"] = labels[:, :3].clone()
        query_a = (a[:, 0] + a[:, 1] + a[:, 2]) / math.sqrt(3.0)
        hard_rho = three_vote_distractor_rho
        noise = torch.randn(count, 2, KEY_DIM, generator=generator) / math.sqrt(KEY_DIM)
        a[:, 3:5] = (
            hard_rho * query_a.unsqueeze(1)
            + math.sqrt(1.0 - hard_rho**2) * noise
        )
        labels[:, 3:5] = _different_labels(majority, generator, 2)
        targets = majority
        metadata["pre_permutation_voter_indices"] = torch.tensor([0, 1, 2]).repeat(count, 1)
        metadata["pre_permutation_distractor_indices"] = torch.tensor([3, 4]).repeat(count, 1)
        a, b, labels, categories, priorities = _permute_tokens(
            a, b, labels, categories, priorities, generator
        )

    context_latent, query_latent = _latent_records(
        a,
        b,
        labels,
        categories,
        priorities,
        query_a,
        query_b,
        query_categories,
        task_id,
    )
    context = context_latent @ transforms.context.T
    query = query_latent @ transforms.query.T
    return TaskBatch(
        task_id=task_id,
        context_latent=context_latent,
        query_latent=query_latent,
        context=context,
        query=query,
        targets=targets,
        metadata=metadata,
    )


class TenTaskAttention(nn.Module):
    """Exactly one attention head/layer with no residual, MLP, position signal, or norm."""

    def __init__(self, gamma: float) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.readout = nn.Linear(D_MODEL, LABEL_COUNT, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.readout):
            nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(
        self, context: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)
        attention_logits = self.gamma * torch.einsum("bd,bnd->bn", q, k)
        attention = torch.softmax(attention_logits, dim=-1)
        retrieved = torch.einsum("bn,bnd->bd", attention, v)
        attention_output = self.o_proj(retrieved)
        logits = self.readout(attention_output)
        return logits, {
            "q": q,
            "k": k,
            "v": v,
            "attention_logits": attention_logits,
            "attention": attention,
            "retrieved": retrieved,
            "attention_output": attention_output,
        }


def initialize_model(model_seed: int, gamma: float) -> tuple[TenTaskAttention, dict[str, torch.Tensor]]:
    torch.manual_seed(stable_seed("model_initialization", model_seed))
    model = TenTaskAttention(gamma)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return model, state


def model_architecture_audit(model: TenTaskAttention) -> dict[str, Any]:
    linear_modules = [module for module in model.modules() if isinstance(module, nn.Linear)]
    normalization_modules = [
        module
        for module in model.modules()
        if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm))
    ]
    checks = {
        "exactly_five_linear_maps": len(linear_modules) == 5,
        "qkvo_are_128_square": all(
            tuple(getattr(model, f"{name}_proj").weight.shape) == (D_MODEL, D_MODEL)
            for name in ("q", "k", "v", "o")
        ),
        "readout_shape_is_32_by_128": tuple(model.readout.weight.shape)
        == (LABEL_COUNT, D_MODEL),
        "qkvo_have_no_bias": all(
            getattr(model, f"{name}_proj").bias is None for name in ("q", "k", "v", "o")
        ),
        "readout_bias_is_zero_initialized": bool(torch.count_nonzero(model.readout.bias) == 0),
        "no_normalization": not normalization_modules,
        "no_mlp": not any(isinstance(module, nn.Sequential) for module in model.modules()),
        "no_positional_parameter": not any(
            "position" in name.lower() for name, _ in model.named_parameters()
        ),
    }
    return {
        "passed": all(checks.values()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **checks,
    }


def parameter_hashes(model: nn.Module) -> dict[str, str]:
    return {name: tensor_sha256(parameter) for name, parameter in model.named_parameters()}


def split_batches(
    task_id: int,
    total_count: int,
    batch_size: int,
    split_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    three_vote_distractor_rho: float = 0.75,
) -> Iterable[TaskBatch]:
    for batch_index, start in enumerate(range(0, total_count, batch_size)):
        count = min(batch_size, total_count - start)
        yield generate_task_batch(
            task_id,
            count,
            stable_seed(split_seed, task_id, batch_index),
            transforms,
            rho,
            three_vote_distractor_rho,
        )


@torch.inference_mode()
def evaluate_accuracy(
    model: TenTaskAttention,
    task_id: int,
    total_count: int,
    batch_size: int,
    split_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    device: torch.device,
    three_vote_distractor_rho: float = 0.75,
) -> float:
    model.eval()
    correct = 0
    observed = 0
    for batch in split_batches(
        task_id,
        total_count,
        batch_size,
        split_seed,
        transforms,
        rho,
        three_vote_distractor_rho,
    ):
        logits, _ = model(batch.context.to(device), batch.query.to(device))
        targets = batch.targets.to(device)
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        observed += batch.count
    if observed != total_count:
        raise AssertionError(f"Expected {total_count} examples, observed {observed}")
    return correct / observed


def evaluate_tasks(
    model: TenTaskAttention,
    task_ids: list[int],
    total_count: int,
    batch_size: int,
    data_namespace: str,
    model_seed: int,
    split_name: str,
    transforms: OrthogonalTransforms,
    rho: float,
    device: torch.device,
    three_vote_distractor_rho: float = 0.75,
) -> dict[str, float]:
    return {
        TASK_NAMES[task_id]: evaluate_accuracy(
            model,
            task_id,
            total_count,
            batch_size,
            stable_seed(data_namespace, model_seed, split_name, task_id),
            transforms,
            rho,
            device,
            three_vote_distractor_rho,
        )
        for task_id in task_ids
    }


def _qualification_status(
    accuracies: dict[str, float], required_mean: float, required_each: float
) -> bool:
    if not accuracies:
        return False
    values = list(accuracies.values())
    return sum(values) / len(values) >= required_mean and min(values) >= required_each


def train_configuration(
    model: TenTaskAttention,
    task_ids: list[int],
    configuration_name: str,
    phase: str,
    model_seed: int,
    transforms: OrthogonalTransforms,
    rho: float,
    device: torch.device,
    *,
    learning_rate: float,
    batch_size: int,
    evaluation_batch_size: int,
    validation_count: int,
    max_steps: int,
    evaluate_every: int,
    required_mean_accuracy: float,
    required_each_accuracy: float,
    weight_decay: float = 0.0,
    three_vote_distractor_rho: float = 0.75,
    optimizer_state: dict[str, Any] | None = None,
    start_step: int = 0,
    stop_when_qualified: bool = True,
) -> TrainingResult:
    if not task_ids:
        raise ValueError("At least one task is required")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Task list contains duplicates")
    if start_step < 0 or max_steps <= start_step:
        raise ValueError("max_steps must be greater than start_step")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
    schedule_generator = torch.Generator(device="cpu").manual_seed(
        stable_seed(phase, "task_schedule", configuration_name, model_seed)
    )
    if start_step:
        torch.randint(
            0,
            len(task_ids),
            (start_step,),
            generator=schedule_generator,
        )
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    training_examples = 0
    final_validation: dict[str, float] = {}
    qualified = False
    for step in range(start_step + 1, max_steps + 1):
        scheduled_index = int(
            torch.randint(0, len(task_ids), (1,), generator=schedule_generator).item()
        )
        task_id = task_ids[scheduled_index]
        batch = generate_task_batch(
            task_id,
            batch_size,
            stable_seed(
                phase,
                "training_example",
                configuration_name,
                model_seed,
                task_id,
                step,
            ),
            transforms,
            rho,
            three_vote_distractor_rho,
        )
        context = batch.context.to(device)
        query = batch.query.to(device)
        targets = batch.targets.to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(context, query)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        training_examples += batch.count
        if step == 1 or step % evaluate_every == 0 or step == max_steps:
            final_validation = evaluate_tasks(
                model,
                task_ids,
                validation_count,
                evaluation_batch_size,
                f"{phase}_validation",
                model_seed,
                "validation",
                transforms,
                rho,
                device,
                three_vote_distractor_rho,
            )
            qualified = _qualification_status(
                final_validation, required_mean_accuracy, required_each_accuracy
            )
            elapsed = time.monotonic() - started
            row = {
                "step": step,
                "training_task": TASK_NAMES[task_id],
                "train_loss": float(loss.detach().cpu().item()),
                "validation_accuracy": final_validation,
                "mean_validation_accuracy": float(
                    sum(final_validation.values()) / len(final_validation)
                ),
                "minimum_validation_accuracy": float(min(final_validation.values())),
                "qualified": qualified,
                "elapsed_seconds": elapsed,
                "training_examples_per_second": training_examples / max(elapsed, 1e-12),
            }
            history.append(row)
            print({"event": "training_progress", "configuration": configuration_name, **row}, flush=True)
            if qualified and stop_when_qualified:
                break
    elapsed = time.monotonic() - started
    return TrainingResult(
        history=history,
        validation_accuracy=final_validation,
        optimizer_state=optimizer.state_dict(),
        steps_completed=history[-1]["step"] if history else start_step,
        elapsed_seconds=elapsed,
        examples_per_second=training_examples / max(elapsed, 1e-12),
        qualified=qualified,
    )
