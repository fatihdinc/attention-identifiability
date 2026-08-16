from dataclasses import fields

import torch

from identifiability_llm.ten_task_analysis import build_bases, collect_task_grams
from identifiability_llm.ten_task_attention import (
    MATRIX_TO_PARAMETER,
    TenTaskAttention,
    make_transforms,
)
from identifiability_llm.ten_task_distillation import (
    DistillationTrainingConfig,
    EFFECTIVE_SCORE_MATRIX,
    FunctionalDistillationData,
    build_functional_distillation_data,
    evaluate_factor_metrics,
    local_output_optimal_residual_fraction,
    replacement_from_factors,
    svd_factor_initialization,
    train_rank_group,
)
from identifiability_llm.ten_task_effective_score import effective_score_matrix


def _model() -> TenTaskAttention:
    torch.manual_seed(17)
    return TenTaskAttention(gamma=0.25).eval()


def _data(model: TenTaskAttention, matrix: str) -> FunctionalDistillationData:
    return build_functional_distillation_data(
        model,
        matrix=matrix,
        task_id=0,
        total_count=16,
        generation_batch_size=8,
        calibration_seed=1234,
        transforms=make_transforms(19),
        rho=0.9,
        three_vote_distractor_rho=0.4,
        device=torch.device("cpu"),
    )


def test_distillation_cache_is_label_free_and_deterministic() -> None:
    model = _model()
    first = _data(model, "Q")
    second = _data(model, "Q")
    names = {field.name for field in fields(FunctionalDistillationData)}
    assert "targets" not in names
    assert "labels" not in names
    assert first.teacher_logits_sha256 == second.teacher_logits_sha256
    assert torch.equal(first.teacher_logits, second.teacher_logits)


def test_full_rank_functional_forward_recovers_teacher_for_every_matrix() -> None:
    model = _model()
    parameters = dict(model.named_parameters())
    for matrix, parameter_name in MATRIX_TO_PARAMETER.items():
        data = _data(model, matrix)
        factor_a, factor_b = svd_factor_initialization(
            parameters[parameter_name], 128
        )
        metrics = evaluate_factor_metrics(
            data,
            factor_a,
            factor_b,
            batch_size=8,
            epsilon=1e-12,
            device=torch.device("cpu"),
        )
        assert metrics["normalized_logit_mse"] < 1e-10
        assert metrics["teacher_prediction_agreement"] == 1.0


def test_effective_score_distillation_is_label_free_and_full_rank_recovers() -> None:
    model = _model()
    data = _data(model, EFFECTIVE_SCORE_MATRIX)
    assert data.matrix_inputs_secondary is not None
    assert tuple(data.matrix_inputs_secondary.shape) == (16, 64, 128)
    factor_a, factor_b = svd_factor_initialization(effective_score_matrix(model), 128)
    metrics = evaluate_factor_metrics(
        data,
        factor_a,
        factor_b,
        batch_size=8,
        epsilon=1e-12,
        device=torch.device("cpu"),
    )
    assert metrics["normalized_logit_mse"] < 1e-10
    assert metrics["teacher_prediction_agreement"] == 1.0


def test_effective_score_training_saves_nested_horizons() -> None:
    model = _model()
    data = _data(model, EFFECTIVE_SCORE_MATRIX)
    result = train_rank_group(
        data,
        effective_score_matrix(model),
        [2, 4],
        DistillationTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=2,
            evaluate_every=1,
            patience_evaluations=3,
            gradient_clip_norm=10.0,
            epsilon=1e-12,
            improvement_tolerance=0.0,
        ),
        optimization_seed=101,
        device=torch.device("cpu"),
        horizon_steps=[1, 2],
    )
    assert result["completed_steps"] == 2
    assert sorted(result["horizon_checkpoints"]) == [1, 2]
    for horizon in (1, 2):
        assert sorted(result["horizon_checkpoints"][horizon]) == [2, 4]
        for row in result["horizon_checkpoints"][horizon].values():
            assert 0 <= row["selected_step"] <= horizon


def test_svd_factors_reproduce_rank_k_parameter_svd() -> None:
    model = _model()
    weight = model.q_proj.weight
    factor_a, factor_b = svd_factor_initialization(weight, 8, dtype=torch.float64)
    observed = replacement_from_factors(factor_a, factor_b)
    u, singular_values, vh = torch.linalg.svd(weight.detach().to(torch.float64))
    expected = (u[:, :8] * singular_values[:8]) @ vh[:8]
    assert torch.allclose(observed, expected, atol=1e-12, rtol=1e-12)
    assert int(torch.linalg.matrix_rank(observed).item()) <= 8


def test_small_training_run_keeps_step_zero_as_loss_upper_bound() -> None:
    model = _model()
    data = _data(model, "O")
    result = train_rank_group(
        data,
        model.o_proj.weight,
        [2, 4],
        DistillationTrainingConfig(
            learning_rate=1e-3,
            batch_size=8,
            max_steps=2,
            evaluate_every=1,
            patience_evaluations=3,
            gradient_clip_norm=10.0,
            epsilon=1e-12,
            improvement_tolerance=0.0,
        ),
        optimization_seed=99,
        device=torch.device("cpu"),
    )
    for row in result["ranks"].values():
        assert row["final_normalized_logit_mse"] <= row[
            "initial_normalized_logit_mse"
        ]
        product = replacement_from_factors(row["factor_a"], row["factor_b"])
        assert int(torch.linalg.matrix_rank(product).item()) <= row["rank"]
        terminal_product = replacement_from_factors(
            row["terminal_factor_a"], row["terminal_factor_b"]
        )
        assert int(torch.linalg.matrix_rank(terminal_product).item()) <= row["rank"]
        assert row["terminal_step"] == 2
        terminal_history = next(
            history
            for history in result["history"]
            if history["rank"] == row["rank"]
            and history["step"] == row["terminal_step"]
        )
        assert (
            row["terminal_normalized_logit_mse"]
            == terminal_history["normalized_logit_mse"]
        )


def test_left_gram_residual_is_local_output_rank_k_optimum() -> None:
    model = _model()
    transforms = make_transforms(19)
    grams = collect_task_grams(
        model,
        0,
        32,
        8,
        1234,
        transforms,
        0.9,
        torch.device("cpu"),
        0.4,
    )
    bases = build_bases(model, grams)
    for matrix in MATRIX_TO_PARAMETER:
        basis = bases[matrix]
        for rank in (0, 4, 16, 128):
            residual = local_output_optimal_residual_fraction(
                basis.left_values, rank
            )
            values = basis.left_values.clamp_min(0)
            expected = (
                float(values[rank:].sum().item() / values.sum().item())
                if float(values.sum().item()) > 0
                else 0.0
            )
            assert abs(residual - expected) < 1e-12
