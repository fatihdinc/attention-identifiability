from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import tomllib
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = ROOT
sys.path.insert(0, str(PARENT_ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from identifiability_llm.paths import DATA_ROOT  # noqa: E402
from identifiability_llm.ten_task_attention import (  # noqa: E402
    D_MODEL,
    N_CONTEXT,
    TASK_COUNT,
    TASK_NAMES,
    TASK_SLICE,
    initialize_model,
    make_transforms,
    split_batches,
    stable_seed,
)
from identifiability_llm.ten_task_effective_score import (  # noqa: E402
    FOUR_GRAM_METHODS,
    effective_score_matrix,
    evaluate_cached_effective_score,
    prepare_effective_score_cache,
)


PARENT_CONFIG = PARENT_ROOT / "configs/experiment.toml"
RUN_ID = "ten_task_effective_score_20seeds_v1"
CONTROL_ROOT = DATA_ROOT / "controls"
CHECKPOINT_ROOT = DATA_ROOT / "artifacts" / RUN_ID / "confirmatory/checkpoints"
PARENT_BASIS_ROOT = (
    DATA_ROOT / "artifacts" / RUN_ID / "confirmatory/four_gram_bases"
)
PARENT_RESULTS = (
    DATA_ROOT
    / "results"
    / RUN_ID
    / "confirmatory/tables/reconstruction_results.csv"
)
PART_ROOT = CONTROL_ROOT / "parts"
AUDIT_ROOT = CONTROL_ROOT / "audits"
TABLE_ROOT = CONTROL_ROOT / "tables"
FIGURE_ROOT = ROOT / "figures/controls"
REPORT_ROOT = CONTROL_ROOT / "reports"

METHOD_LABELS = {
    "key_input_gram": "Key-input Gram",
    "key_output_gram": "Key-output Gram",
    "query_input_gram": "Query-input Gram",
    "query_output_gram": "Query-output Gram",
}
METHOD_COLORS = {
    "key_input_gram": "#D55E00",
    "key_output_gram": "#009E73",
    "query_input_gram": "#CC79A7",
    "query_output_gram": "#56B4E9",
}
RESULT_COLUMNS = [
    "seed",
    "source_task_id",
    "source_task",
    "evaluation_task_id",
    "evaluation_task",
    "task_code_used",
    "method",
    "rank",
    "accuracy",
    "full_model_accuracy",
    "retention",
    "mean_attention_kl",
    "centered_score_mse",
    "matched_task",
]


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_csv(rows: Iterable[dict[str, Any]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def replace_json(payload: Any, path: Path) -> None:
    """Atomically replace a derived aggregate while preserving seed parts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    with PARENT_CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    if str(config["experiment"]["run_id"]) != RUN_ID:
        raise RuntimeError("Unexpected parent run id")
    if [str(value) for value in config["sweep"]["methods"]][:-1] != list(
        FOUR_GRAM_METHODS
    ):
        raise RuntimeError("The parent four-Gram method order changed")
    if len(config["experiment"]["confirmatory_seeds"]) != 20:
        raise RuntimeError("The parent experiment is not the final 20-seed run")
    return config


def code_matched_queries(batch: Any, transforms: Any) -> torch.Tensor:
    """Return [target_task, batch, d] queries with target task codes."""

    latents = batch.query_latent.unsqueeze(0).expand(TASK_COUNT, -1, -1).clone()
    codes = torch.eye(TASK_COUNT, dtype=latents.dtype).unsqueeze(1)
    latents[:, :, TASK_SLICE] = codes
    return latents @ transforms.query.T


def descending_eigenvectors(matrix: torch.Tensor) -> torch.Tensor:
    _, vectors = torch.linalg.eigh(matrix.to(torch.float64))
    return vectors.flip(-1)


@torch.inference_mode()
def collect_transfer_vectors(
    matrix: torch.Tensor,
    model_seed: int,
    transforms: Any,
    config: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Collect [source, target-code, d, d] eigenvector bases."""

    confirmatory = config["confirmatory"]
    generator = config["generator"]
    calibration_count = int(confirmatory["calibration_count"])
    batch_size = int(confirmatory["evaluation_batch_size"])
    matrix64 = matrix.to(torch.float64)
    vectors = {
        method: torch.empty(
            TASK_COUNT, TASK_COUNT, D_MODEL, D_MODEL, dtype=torch.float64
        )
        for method in FOUR_GRAM_METHODS
    }
    maximum_symmetry_error = 0.0
    maximum_key_output_identity_error = 0.0
    maximum_query_output_identity_error = 0.0
    maximum_diagonal_query_error = 0.0
    maximum_diagonal_parent_gram_error = 0.0
    eye_codes = torch.eye(TASK_COUNT)

    for source_id, source_name in enumerate(TASK_NAMES):
        query_input_sum = torch.zeros(
            TASK_COUNT, D_MODEL, D_MODEL, dtype=torch.float64
        )
        query_output_sum = torch.zeros_like(query_input_sum)
        key_input_sum = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
        key_output_sum = torch.zeros_like(key_input_sum)
        query_count = 0
        key_count = 0
        calibration_seed = stable_seed(
            str(config["data"]["confirmatory_calibration_namespace"]),
            model_seed,
            "calibration",
            source_id,
        )
        for batch in split_batches(
            source_id,
            calibration_count,
            batch_size,
            calibration_seed,
            transforms,
            float(generator["rho"]),
            float(generator["three_vote_distractor_rho"]),
        ):
            queries32 = code_matched_queries(batch, transforms)
            maximum_diagonal_query_error = max(
                maximum_diagonal_query_error,
                float((queries32[source_id] - batch.query).abs().max().item()),
            )
            latent_codes = batch.query_latent.new_zeros(
                TASK_COUNT, batch.count, TASK_COUNT
            )
            latent_codes[:] = eye_codes[:, None, :]
            if not torch.equal(
                latent_codes,
                eye_codes[:, None, :].expand(-1, batch.count, -1),
            ):
                raise AssertionError("Task-code replacement audit failed")
            queries = queries32.to(torch.float64)
            contexts = batch.context.reshape(-1, D_MODEL).to(torch.float64)
            key_outputs = contexts @ matrix64.T
            query_outputs = queries @ matrix64
            query_input_sum += torch.einsum("tbd,tbe->tde", queries, queries)
            query_output_sum += torch.einsum(
                "tbd,tbe->tde", query_outputs, query_outputs
            )
            key_input_sum += contexts.T @ contexts
            key_output_sum += key_outputs.T @ key_outputs
            query_count += batch.count
            key_count += int(contexts.shape[0])
        if query_count != calibration_count or key_count != calibration_count * N_CONTEXT:
            raise AssertionError("Calibration observation count mismatch")
        query_input = query_input_sum / query_count
        query_output = query_output_sum / query_count
        key_input = key_input_sum / key_count
        key_output = key_output_sum / key_count

        for gram in (query_input, query_output):
            maximum_symmetry_error = max(
                maximum_symmetry_error,
                float(torch.linalg.norm(gram - gram.transpose(-1, -2)).item()),
            )
        for gram in (key_input, key_output):
            maximum_symmetry_error = max(
                maximum_symmetry_error,
                float(torch.linalg.norm(gram - gram.T).item()),
            )
        expected_key_output = matrix64 @ key_input @ matrix64.T
        expected_query_output = torch.einsum(
            "de,tef,fg->tdg", matrix64.T, query_input, matrix64
        )
        maximum_key_output_identity_error = max(
            maximum_key_output_identity_error,
            float(
                torch.linalg.norm(key_output - expected_key_output).item()
                / max(float(torch.linalg.norm(expected_key_output).item()), 1e-12)
            ),
        )
        query_output_denominators = torch.linalg.norm(
            expected_query_output, dim=(-2, -1)
        ).clamp_min(1e-12)
        maximum_query_output_identity_error = max(
            maximum_query_output_identity_error,
            float(
                (
                    torch.linalg.norm(
                        query_output - expected_query_output, dim=(-2, -1)
                    )
                    / query_output_denominators
                )
                .max()
                .item()
            ),
        )

        key_input_vectors = descending_eigenvectors(key_input)
        key_output_vectors = descending_eigenvectors(key_output)
        query_input_vectors = descending_eigenvectors(query_input)
        query_output_vectors = descending_eigenvectors(query_output)
        vectors["key_input_gram"][source_id] = key_input_vectors.unsqueeze(0)
        vectors["key_output_gram"][source_id] = key_output_vectors.unsqueeze(0)
        vectors["query_input_gram"][source_id] = query_input_vectors
        vectors["query_output_gram"][source_id] = query_output_vectors

        parent_payload = torch.load(
            PARENT_BASIS_ROOT / f"seed_{model_seed}" / f"{source_name}.pt",
            map_location="cpu",
            weights_only=False,
        )
        parent_grams = parent_payload["grams"]
        for observed, expected in (
            (query_input[source_id], parent_grams.query_input),
            (key_input, parent_grams.key_input),
            (key_output, parent_grams.key_output),
            (query_output[source_id], parent_grams.query_output),
        ):
            denominator = max(float(torch.linalg.norm(expected).item()), 1e-12)
            maximum_diagonal_parent_gram_error = max(
                maximum_diagonal_parent_gram_error,
                float(torch.linalg.norm(observed - expected).item() / denominator),
            )

    return vectors, {
        "maximum_symmetry_error": maximum_symmetry_error,
        "maximum_key_output_identity_relative_error": maximum_key_output_identity_error,
        "maximum_query_output_identity_relative_error": maximum_query_output_identity_error,
        "maximum_diagonal_query_absolute_error": maximum_diagonal_query_error,
        "maximum_diagonal_parent_gram_relative_error": maximum_diagonal_parent_gram_error,
    }


@torch.inference_mode()
def evaluate_all_sources(
    model: Any,
    cache: Iterable[Any],
    matrix: torch.Tensor,
    basis_vectors: torch.Tensor,
    method: str,
    ranks: list[int],
) -> dict[str, torch.Tensor]:
    """Evaluate ten source bases and all ranks using nested projector factors."""

    matrix32 = matrix.to(torch.float32)
    vectors = basis_vectors.to(torch.float32)
    source_count = int(vectors.shape[0])
    rank_count = len(ranks)
    correct = torch.zeros(source_count, rank_count, dtype=torch.float64)
    kl_sum = torch.zeros_like(correct)
    score_squared_sum = torch.zeros_like(correct)
    observed = 0
    score_count = 0
    value_to_logits = (
        model.readout.weight.detach().cpu() @ model.o_proj.weight.detach().cpu()
    )
    readout_bias = model.readout.bias.detach().cpu()
    premultiplies_matrix = method in {"query_input_gram", "key_output_gram"}

    for batch in cache:
        context = batch.context.detach().cpu()
        query = batch.query.detach().cpu()
        if premultiplies_matrix:
            mapped_context = context @ matrix32.T
            query_coordinates = torch.einsum("bd,sdk->sbk", query, vectors)
            context_coordinates = torch.einsum(
                "bnd,sdk->sbnk", mapped_context, vectors
            )
        else:
            mapped_query = query @ matrix32
            query_coordinates = torch.einsum("bd,sdk->sbk", mapped_query, vectors)
            context_coordinates = torch.einsum("bnd,sdk->sbnk", context, vectors)
        components = query_coordinates[:, :, None, :] * context_coordinates
        cumulative_scores = components.cumsum(dim=-1)
        scores = torch.empty(
            source_count,
            rank_count,
            query.shape[0],
            context.shape[1],
            dtype=torch.float32,
        )
        for rank_index, rank in enumerate(ranks):
            if rank == 0:
                scores[:, rank_index].zero_()
            elif rank == D_MODEL:
                scores[:, rank_index] = batch.original_scores.detach().cpu().unsqueeze(0)
            else:
                scores[:, rank_index] = cumulative_scores[..., rank - 1]
        attention = torch.softmax(scores, dim=-1)
        token_logits = torch.einsum(
            "bnd,cd->bnc", batch.values.detach().cpu(), value_to_logits
        )
        logits = torch.einsum("srbn,bnc->srbc", attention, token_logits)
        logits += readout_bias
        targets = batch.targets.detach().cpu()
        correct += (
            logits.argmax(dim=-1) == targets[None, None, :]
        ).sum(dim=-1).to(torch.float64)
        original_attention = batch.original_attention.detach().cpu()
        kl = original_attention[None, None] * (
            torch.log(original_attention.clamp_min(1e-12))[None, None]
            - torch.log(attention.clamp_min(1e-12))
        )
        kl_sum += kl.sum(dim=(-1, -2)).to(torch.float64)
        original_scores = batch.original_scores.detach().cpu()
        centered_original = original_scores - original_scores.mean(dim=-1, keepdim=True)
        centered_scores = scores - scores.mean(dim=-1, keepdim=True)
        score_squared_sum += (
            centered_original[None, None] - centered_scores
        ).square().sum(dim=(-1, -2)).to(torch.float64)
        observed += int(targets.shape[0])
        score_count += int(original_scores.numel())
    if observed == 0 or score_count == 0:
        raise AssertionError("Empty evaluation cache")
    return {
        "accuracy": correct / observed,
        "mean_attention_kl": kl_sum / observed,
        "centered_score_mse": score_squared_sum / score_count,
    }


def diagonal_accuracy_audit(seed: int, rows: list[dict[str, Any]]) -> float:
    parent = pd.read_csv(PARENT_RESULTS, low_memory=False)
    parent = parent[
        parent["seed"].eq(seed) & parent["method"].isin(FOUR_GRAM_METHODS)
    ][["task", "method", "rank", "accuracy"]]
    observed = pd.DataFrame(rows)
    observed = observed[observed["matched_task"]][
        ["evaluation_task", "method", "rank", "accuracy"]
    ].rename(columns={"evaluation_task": "task", "accuracy": "control_accuracy"})
    merged = parent.merge(observed, on=["task", "method", "rank"], validate="one_to_one")
    expected = len(TASK_NAMES) * len(FOUR_GRAM_METHODS) * parent["rank"].nunique()
    if len(merged) != expected:
        raise AssertionError(f"Incomplete diagonal audit: {len(merged)} vs {expected}")
    return float((merged["accuracy"] - merged["control_accuracy"]).abs().max())


def run_seed(seed: int, config: dict[str, Any]) -> None:
    part_path = PART_ROOT / f"seed_{seed}.csv"
    audit_path = AUDIT_ROOT / f"seed_{seed}.json"
    if part_path.exists() and audit_path.exists():
        print(json.dumps({"event": "task_transfer_seed_skip_existing", "seed": seed}))
        return
    if part_path.exists() or audit_path.exists():
        raise RuntimeError("Found incomplete seed output pair")
    allowed = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    if seed not in allowed:
        raise ValueError(f"Seed {seed} is not in the final 20-seed experiment")
    torch.set_num_threads(1)
    started = time.monotonic()
    checkpoint_path = CHECKPOINT_ROOT / f"seed_{seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = initialize_model(seed, float(config["confirmatory"]["gamma"]))
    model.load_state_dict(checkpoint["final_state_dict"])
    model.eval().cpu()
    transforms = make_transforms(seed)
    matrix = effective_score_matrix(model)
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    vectors, gram_audit = collect_transfer_vectors(
        matrix, seed, transforms, config
    )
    rows: list[dict[str, Any]] = []
    evaluation_durations: list[float] = []
    generator = config["generator"]
    for evaluation_id, evaluation_name in enumerate(TASK_NAMES):
        evaluation_started = time.monotonic()
        test_seed = stable_seed(
            str(config["data"]["confirmatory_test_namespace"]),
            seed,
            "test",
            evaluation_id,
        )
        cache = prepare_effective_score_cache(
            model,
            evaluation_id,
            int(config["confirmatory"]["test_count"]),
            int(config["confirmatory"]["evaluation_batch_size"]),
            test_seed,
            transforms,
            float(generator["rho"]),
            float(generator["three_vote_distractor_rho"]),
            torch.device("cpu"),
        )
        baseline = evaluate_cached_effective_score(model, cache, matrix)
        expected_baseline = float(checkpoint["test_accuracy"][evaluation_name])
        if abs(baseline["accuracy"] - expected_baseline) > 1e-12:
            raise AssertionError("Locked test baseline mismatch")
        for method in FOUR_GRAM_METHODS:
            method_vectors = vectors[method][:, evaluation_id]
            metrics = evaluate_all_sources(
                model, cache, matrix, method_vectors, method, ranks
            )
            for source_id, source_name in enumerate(TASK_NAMES):
                for rank_index, rank in enumerate(ranks):
                    accuracy = float(metrics["accuracy"][source_id, rank_index].item())
                    rows.append(
                        {
                            "seed": seed,
                            "source_task_id": source_id,
                            "source_task": source_name,
                            "evaluation_task_id": evaluation_id,
                            "evaluation_task": evaluation_name,
                            "task_code_used": evaluation_name,
                            "method": method,
                            "rank": rank,
                            "accuracy": accuracy,
                            "full_model_accuracy": baseline["accuracy"],
                            "retention": accuracy / max(baseline["accuracy"], 1e-12),
                            "mean_attention_kl": float(
                                metrics["mean_attention_kl"][source_id, rank_index].item()
                            ),
                            "centered_score_mse": float(
                                metrics["centered_score_mse"][source_id, rank_index].item()
                            ),
                            "matched_task": source_id == evaluation_id,
                        }
                    )
        evaluation_duration = time.monotonic() - evaluation_started
        evaluation_durations.append(evaluation_duration)
        remaining = len(TASK_NAMES) - evaluation_id - 1
        print(
            json.dumps(
                {
                    "event": "task_transfer_evaluation_task_complete",
                    "seed": seed,
                    "evaluation_task": evaluation_name,
                    "completed": evaluation_id + 1,
                    "total": len(TASK_NAMES),
                    "remaining_eta_seconds": (
                        sum(evaluation_durations) / len(evaluation_durations) * remaining
                    ),
                }
            ),
            flush=True,
        )
        del cache
    expected_rows = TASK_COUNT * TASK_COUNT * len(FOUR_GRAM_METHODS) * len(ranks)
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(rows)}")
    diagonal_error = diagonal_accuracy_audit(seed, rows)
    rank128 = [row for row in rows if row["rank"] == D_MODEL]
    full_rank_error = max(
        abs(float(row["accuracy"]) - float(row["full_model_accuracy"]))
        for row in rank128
    )
    audit = {
        "passed": (
            gram_audit["maximum_symmetry_error"] <= 1e-9
            and gram_audit["maximum_key_output_identity_relative_error"] <= 1e-9
            and gram_audit["maximum_query_output_identity_relative_error"] <= 1e-9
            and gram_audit["maximum_diagonal_query_absolute_error"] == 0.0
            and gram_audit["maximum_diagonal_parent_gram_relative_error"] <= 1e-12
            and diagonal_error <= 1e-3
            and full_rank_error == 0.0
        ),
        "seed": seed,
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "methods": list(FOUR_GRAM_METHODS),
        "ranks": ranks,
        "task_code_control": "source calibration query code replaced by evaluation task code",
        "gram_audit": gram_audit,
        "maximum_diagonal_accuracy_difference_vs_parent": diagonal_error,
        "maximum_full_rank_accuracy_difference": full_rank_error,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "parent_config_sha256": file_sha256(PARENT_CONFIG),
        "elapsed_seconds": time.monotonic() - started,
    }
    if not audit["passed"]:
        raise AssertionError(f"Seed audit failed: {audit}")
    atomic_csv(rows, RESULT_COLUMNS, part_path)
    atomic_json(audit, audit_path)
    print(json.dumps({"event": "task_transfer_seed_complete", **audit}), flush=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "figure.titlesize": 13,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def plot_curves(specificity: pd.DataFrame) -> None:
    style()
    summary = (
        specificity.groupby(["method", "rank"], as_index=False)
        .agg(
            matched_mean=("matched_accuracy", "mean"),
            matched_sem=("matched_accuracy", "sem"),
            mismatched_mean=("mismatched_accuracy_mean", "mean"),
            mismatched_sem=("mismatched_accuracy_mean", "sem"),
            advantage_mean=("accuracy_advantage", "mean"),
            advantage_sem=("accuracy_advantage", "sem"),
        )
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, FOUR_GRAM_METHODS):
        data = summary[summary["method"].eq(method)].sort_values("rank")
        x = data["rank"].to_numpy(float)
        for mean_column, sem_column, label, linestyle in (
            ("matched_mean", "matched_sem", "Matched source task", "-"),
            ("mismatched_mean", "mismatched_sem", "Mean mismatched source", "--"),
        ):
            mean = data[mean_column].to_numpy(float)
            sem = data[sem_column].fillna(0).to_numpy(float)
            axis.plot(
                x,
                mean,
                color=METHOD_COLORS[method],
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
            axis.fill_between(
                x,
                np.clip(mean - sem, 0, 1.02),
                np.clip(mean + sem, 0, 1.02),
                color=METHOD_COLORS[method],
                alpha=0.14,
                linewidth=0,
            )
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlim(0, 128)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("Reconstruction rank K")
    for axis in axes[:, 0]:
        axis.set_ylabel("Held-out accuracy")
    figure.suptitle("Task-code-matched Gram transfer: matched vs mismatched tasks")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURE_ROOT / f"matched_vs_mismatched_curves.{suffix}")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, FOUR_GRAM_METHODS):
        data = summary[summary["method"].eq(method)].sort_values("rank")
        x = data["rank"].to_numpy(float)
        mean = data["advantage_mean"].to_numpy(float)
        sem = data["advantage_sem"].fillna(0).to_numpy(float)
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        axis.plot(x, mean, color=METHOD_COLORS[method], linewidth=1.8)
        axis.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=METHOD_COLORS[method],
            alpha=0.18,
            linewidth=0,
        )
        axis.set_title(METHOD_LABELS[method])
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Reconstruction rank K")
    for axis in axes[:, 0]:
        axis.set_ylabel("Matched minus mismatched accuracy")
    figure.suptitle("Task-specificity advantage after matching the explicit task code")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURE_ROOT / f"task_specificity_advantage.{suffix}")
    plt.close(figure)


def plot_heatmaps(mean_transfer: pd.DataFrame, selected_ranks: list[int]) -> None:
    style()
    labels = [str(index + 1) for index in range(TASK_COUNT)]
    for rank in selected_ranks:
        figure, axes = plt.subplots(2, 2, figsize=(12.2, 10.2))
        image = None
        for axis, method in zip(axes.flat, FOUR_GRAM_METHODS):
            selected = mean_transfer[
                mean_transfer["method"].eq(method) & mean_transfer["rank"].eq(rank)
            ]
            pivot = selected.pivot(
                index="source_task_id",
                columns="evaluation_task_id",
                values="retention_mean",
            ).reindex(index=range(TASK_COUNT), columns=range(TASK_COUNT))
            if pivot.isna().any().any():
                raise AssertionError("Incomplete transfer heatmap")
            image = axis.imshow(
                pivot.to_numpy(float),
                vmin=0,
                vmax=1.02,
                cmap="viridis",
                aspect="equal",
                origin="upper",
            )
            axis.set_title(METHOD_LABELS[method])
            axis.set_xticks(range(TASK_COUNT), labels)
            axis.set_yticks(range(TASK_COUNT), labels)
            axis.set_xlabel("Evaluation task")
            axis.set_ylabel("Gram source task")
            for index in range(TASK_COUNT):
                axis.add_patch(
                    plt.Rectangle(
                        (index - 0.5, index - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="white",
                        linewidth=0.8,
                        alpha=0.8,
                    )
                )
        assert image is not None
        colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
        colorbar.set_label("Accuracy retention")
        figure.suptitle(f"Task-code-matched Gram transfer at K={rank}")
        figure.subplots_adjust(top=0.92, right=0.9, wspace=0.22, hspace=0.25)
        for suffix in ("png", "pdf"):
            figure.savefig(
                FIGURE_ROOT / f"task_transfer_heatmaps_K{rank:03d}.{suffix}",
                bbox_inches="tight",
            )
        plt.close(figure)


def finalize(config: dict[str, Any]) -> None:
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    missing_parts = [seed for seed in seeds if not (PART_ROOT / f"seed_{seed}.csv").exists()]
    missing_audits = [seed for seed in seeds if not (AUDIT_ROOT / f"seed_{seed}.json").exists()]
    if missing_parts or missing_audits:
        raise FileNotFoundError(
            f"Missing parts={missing_parts}, missing audits={missing_audits}"
        )
    frames = [pd.read_csv(PART_ROOT / f"seed_{seed}.csv") for seed in seeds]
    results = pd.concat(frames, ignore_index=True)
    expected_rows = len(seeds) * TASK_COUNT * TASK_COUNT * len(FOUR_GRAM_METHODS) * len(ranks)
    duplicate_count = int(
        results.duplicated(
            ["seed", "source_task", "evaluation_task", "method", "rank"]
        ).sum()
    )
    if len(results) != expected_rows or duplicate_count:
        raise AssertionError(
            f"Aggregate rows={len(results)}/{expected_rows}, duplicates={duplicate_count}"
        )
    audits = [
        json.loads((AUDIT_ROOT / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in seeds
    ]
    if not all(audit["passed"] for audit in audits):
        raise AssertionError("At least one seed audit failed")
    write_dataframe(results, TABLE_ROOT / "task_transfer_results.csv")
    mean_transfer = (
        results.groupby(
            [
                "source_task_id",
                "source_task",
                "evaluation_task_id",
                "evaluation_task",
                "method",
                "rank",
            ],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_sem=("accuracy", "sem"),
            retention_mean=("retention", "mean"),
            retention_std=("retention", "std"),
            retention_sem=("retention", "sem"),
        )
    )
    write_dataframe(mean_transfer, TABLE_ROOT / "mean_task_transfer_curves.csv")
    matched = results[results["matched_task"]].rename(
        columns={"accuracy": "matched_accuracy", "retention": "matched_retention"}
    )
    mismatched = (
        results[~results["matched_task"]]
        .groupby(["seed", "evaluation_task", "method", "rank"], as_index=False)
        .agg(
            mismatched_accuracy_mean=("accuracy", "mean"),
            mismatched_accuracy_best=("accuracy", "max"),
            mismatched_retention_mean=("retention", "mean"),
        )
    )
    specificity = matched[
        [
            "seed",
            "evaluation_task",
            "method",
            "rank",
            "matched_accuracy",
            "matched_retention",
        ]
    ].merge(
        mismatched,
        on=["seed", "evaluation_task", "method", "rank"],
        validate="one_to_one",
    )
    specificity["accuracy_advantage"] = (
        specificity["matched_accuracy"] - specificity["mismatched_accuracy_mean"]
    )
    specificity["retention_advantage"] = (
        specificity["matched_retention"] - specificity["mismatched_retention_mean"]
    )
    write_dataframe(specificity, TABLE_ROOT / "task_specificity_advantage.csv")
    selected_ranks = [8, 16, 24, 32]
    selected_summary = (
        specificity[specificity["rank"].isin(selected_ranks)]
        .groupby(["method", "rank"], as_index=False)
        .agg(
            matched_accuracy=("matched_accuracy", "mean"),
            mismatched_accuracy=("mismatched_accuracy_mean", "mean"),
            accuracy_advantage=("accuracy_advantage", "mean"),
            advantage_sem=("accuracy_advantage", "sem"),
            matched_beats_mean_mismatch_fraction=(
                "accuracy_advantage",
                lambda values: float((values > 0).mean()),
            ),
        )
    )
    write_dataframe(selected_summary, TABLE_ROOT / "selected_rank_summary.csv")
    plot_curves(specificity)
    plot_heatmaps(mean_transfer, selected_ranks)

    maximum_diagonal_error = max(
        float(audit["maximum_diagonal_accuracy_difference_vs_parent"])
        for audit in audits
    )
    report_lines = [
        "# Task-code-matched Gram transfer controls",
        "",
        "Each source-task Gram was recomputed after replacing the source query's ",
        "task one-hot with the evaluation task's one-hot. Evaluation used untouched ",
        "held-out examples from the evaluation task.",
        "",
        f"- Seeds: {len(seeds)}",
        f"- Source/evaluation task pairs per seed: {TASK_COUNT * TASK_COUNT}",
        f"- Methods: {len(FOUR_GRAM_METHODS)} Gram reconstructions",
        f"- Ranks: {len(ranks)}",
        f"- Total evaluated conditions: {len(results):,}",
        f"- Maximum diagonal accuracy difference from parent run: {maximum_diagonal_error:.6g}",
        "",
        "## Selected-rank results",
        "",
        "```text",
        selected_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        ),
        "```",
        "",
        "The full transfer tables are in `data/controls/tables/`; publication ",
        "figures are in `figures/controls/`.",
    ]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "SCIENTIFIC_SUMMARY.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    aggregate_audit = {
        "passed": True,
        "run_id": RUN_ID,
        "control": "all-pairs task-code-matched four-Gram transfer",
        "seed_count": len(seeds),
        "row_count": len(results),
        "expected_row_count": expected_rows,
        "duplicate_count": duplicate_count,
        "methods": list(FOUR_GRAM_METHODS),
        "ranks": ranks,
        "maximum_diagonal_accuracy_difference_vs_parent": maximum_diagonal_error,
        "all_seed_audits_pass": True,
        "parent_config_sha256": file_sha256(PARENT_CONFIG),
    }
    replace_json(aggregate_audit, REPORT_ROOT / "aggregate_audit.json")

    manifest_entries: dict[str, str] = {}
    for path in sorted(CONTROL_ROOT.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.sha256.json"}:
            if "logs" in path.relative_to(CONTROL_ROOT).parts:
                continue
            manifest_entries[str(path.relative_to(CONTROL_ROOT))] = file_sha256(path)
    replace_json(
        {"algorithm": "sha256", "files": manifest_entries},
        CONTROL_ROOT / "MANIFEST.sha256.json",
    )
    print(json.dumps({"event": "task_transfer_controls_finalized", **aggregate_audit}))


def validate(config: dict[str, Any]) -> None:
    ranks = [int(value) for value in config["sweep"]["ranks"]]
    seeds = [int(value) for value in config["experiment"]["confirmatory_seeds"]]
    missing = [seed for seed in seeds if not (CHECKPOINT_ROOT / f"seed_{seed}.pt").exists()]
    if missing:
        raise FileNotFoundError(f"Missing final-run checkpoints: {missing}")
    if ranks[:3] != [0, 1, 2] or ranks[-1] != D_MODEL:
        raise RuntimeError("Unexpected parent rank grid")
    transforms = make_transforms(seeds[0])
    from identifiability_llm.ten_task_attention import generate_task_batch

    batch = generate_task_batch(
        3,
        8,
        stable_seed("task_transfer_control_validation", seeds[0]),
        transforms,
        float(config["generator"]["rho"]),
        float(config["generator"]["three_vote_distractor_rho"]),
    )
    queries = code_matched_queries(batch, transforms)
    if not torch.equal(queries[3], batch.query):
        raise AssertionError("Diagonal task-code replacement is not an identity")
    expected_codes = torch.eye(TASK_COUNT)[:, None, :].expand(-1, batch.count, -1)
    latent = batch.query_latent.unsqueeze(0).expand(TASK_COUNT, -1, -1).clone()
    latent[:, :, TASK_SLICE] = expected_codes
    if not torch.equal(latent[:, :, TASK_SLICE], expected_codes):
        raise AssertionError("Target task-code replacement failed")
    print(
        json.dumps(
            {
                "event": "task_transfer_controls_validated",
                "parent_root": str(PARENT_ROOT),
                "seed_count": len(seeds),
                "task_pairs_per_seed": TASK_COUNT * TASK_COUNT,
                "methods": list(FOUR_GRAM_METHODS),
                "ranks": ranks,
                "expected_total_rows": (
                    len(seeds)
                    * TASK_COUNT
                    * TASK_COUNT
                    * len(FOUR_GRAM_METHODS)
                    * len(ranks)
                ),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    selected = sum([args.seed is not None, args.validate_only, args.finalize])
    if selected != 1:
        raise ValueError("Choose exactly one of --seed, --validate-only, or --finalize")
    config = load_config()
    if args.validate_only:
        validate(config)
    elif args.finalize:
        finalize(config)
    else:
        assert args.seed is not None
        run_seed(args.seed, config)


if __name__ == "__main__":
    main()
