from __future__ import annotations

import unittest

import pandas as pd
import torch

from identifiability_llm.ten_task_analysis import (
    FrozenMatrixIntervention,
    MatrixGrams,
    build_bases,
    collect_task_grams,
    collect_shuffled_task_grams,
    controlled_inputs,
    equal_weight_mixture_grams,
    gram_audit,
    leave_one_task_out_grams,
    minimum_rank_table,
    random_projector_reconstruction,
    reconstruct_weight,
    reconstruction_audit,
    svd_power_rows,
    task_specificity_table,
)
from identifiability_llm.ten_task_attention import (
    D_MODEL,
    LABEL_SLICE,
    MATRIX_TO_PARAMETER,
    TASK_SLICE,
    generate_task_batch,
    initialize_model,
    make_transforms,
    parameter_hashes,
)


class TenTaskAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, _ = initialize_model(2221, 0.25)
        cls.transforms = make_transforms(2221)
        cls.grams = collect_task_grams(
            cls.model,
            task_id=0,
            total_count=32,
            batch_size=16,
            split_seed=7701,
            transforms=cls.transforms,
            rho=0.9,
            device=torch.device("cpu"),
            three_vote_distractor_rho=0.4,
        )
        cls.bases = build_bases(cls.model, cls.grams)

    def test_online_grams_are_symmetric_equivalent_and_use_expected_counts(self) -> None:
        audit = gram_audit(self.model, self.grams)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(self.grams["Q"].observation_count, 32)
        self.assertEqual(self.grams["O"].observation_count, 32)
        self.assertEqual(self.grams["K"].observation_count, 32 * 64)
        self.assertEqual(self.grams["V"].observation_count, 32 * 64)

    def test_all_reconstruction_methods_obey_rank_and_full_rank_invariants(self) -> None:
        for matrix, basis in self.bases.items():
            for method in ("parameter_svd", "right_gram", "left_gram"):
                for rank in (0, 8, 128):
                    reconstructed, projector = reconstruct_weight(basis, method, rank)
                    audit = reconstruction_audit(
                        basis.weight, reconstructed, projector, rank
                    )
                    self.assertTrue(audit["rank_bound_satisfied"], (matrix, method, rank, audit))
                    if projector is not None:
                        self.assertLess(audit["projector_symmetry_error"], 1e-12)
                        self.assertLess(audit["projector_idempotence_error"], 1e-10)
                    if rank == 0:
                        self.assertEqual(int(torch.count_nonzero(reconstructed)), 0)
                    if rank == 128:
                        self.assertLess(audit["relative_weight_error"], 1e-10)

    def test_parameter_svd_power_is_monotone_and_has_exact_endpoints(self) -> None:
        ranks = [0, 1, 2, 8, 32, 64, 128]
        rows = svd_power_rows(self.model.state_dict(), ranks, seed=2221)
        self.assertEqual(len(rows), len(MATRIX_TO_PARAMETER) * len(ranks))
        for matrix in MATRIX_TO_PARAMETER:
            values = [
                row["svd_cumulative_power"]
                for row in rows
                if row["matrix"] == matrix
            ]
            self.assertEqual(values[0], 0.0)
            self.assertAlmostEqual(values[-1], 1.0, places=12)
            self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))

    def test_intervention_changes_only_selected_matrix_and_restores_everything_exactly(self) -> None:
        before = parameter_hashes(self.model)
        replacement, _ = reconstruct_weight(self.bases["Q"], "parameter_svd", 8)
        intervention = FrozenMatrixIntervention(self.model, "Q", replacement)
        with intervention:
            during = parameter_hashes(self.model)
            self.assertNotEqual(during["q_proj.weight"], before["q_proj.weight"])
            for name in before:
                if name != "q_proj.weight":
                    self.assertEqual(during[name], before[name])
            self.assertTrue(intervention.entry_audit["all_other_parameters_bit_identical"])
        self.assertEqual(parameter_hashes(self.model), before)
        self.assertTrue(intervention.restoration_audit["all_parameters_restored_bit_identically"])

    def test_equal_weight_mixture_does_not_use_observation_count_weights(self) -> None:
        task_grams = {}
        for task, scale in ((0, 1.0), (1, 3.0), (2, 8.0)):
            task_grams[task] = {
                matrix: MatrixGrams(
                    right=torch.eye(D_MODEL, dtype=torch.float64) * scale,
                    left=torch.eye(D_MODEL, dtype=torch.float64) * (scale + 1),
                    observation_count=64,
                )
                for matrix in MATRIX_TO_PARAMETER
            }
        mixture = equal_weight_mixture_grams(task_grams, [0, 1])
        self.assertTrue(torch.equal(mixture["Q"].right, torch.eye(D_MODEL) * 2.0))
        leave_out = leave_one_task_out_grams(task_grams, 2)
        self.assertTrue(torch.equal(leave_out["O"].left, torch.eye(D_MODEL) * 3.0))

    def test_random_projector_is_seed_matched_symmetric_and_rank_bounded(self) -> None:
        weight = self.bases["K"].weight
        first, projector_first = random_projector_reconstruction(weight, 16, 881)
        second, projector_second = random_projector_reconstruction(weight, 16, 881)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(projector_first, projector_second))
        audit = reconstruction_audit(weight, first, projector_first, 16)
        self.assertTrue(audit["rank_bound_satisfied"], audit)
        self.assertLess(audit["projector_symmetry_error"], 1e-12)
        self.assertLess(audit["projector_idempotence_error"], 1e-12)

    def test_shuffled_calibration_labels_preserve_counts_and_pooled_grams(self) -> None:
        task_ids = [0, 1, 2]
        split_seeds = {task: 8800 + task for task in task_ids}
        shuffled = collect_shuffled_task_grams(
            self.model,
            task_ids=task_ids,
            total_count_per_task=12,
            batch_size=4,
            split_seed=split_seeds,
            transforms=self.transforms,
            rho=0.9,
            device=torch.device("cpu"),
            three_vote_distractor_rho=0.4,
            shuffle_seed=9917,
        )
        repeated = collect_shuffled_task_grams(
            self.model,
            task_ids=task_ids,
            total_count_per_task=12,
            batch_size=4,
            split_seed=split_seeds,
            transforms=self.transforms,
            rho=0.9,
            device=torch.device("cpu"),
            three_vote_distractor_rho=0.4,
            shuffle_seed=9917,
        )
        for task in task_ids:
            self.assertEqual(shuffled[task]["Q"].observation_count, 12)
            self.assertEqual(shuffled[task]["K"].observation_count, 12 * 64)
            for matrix in MATRIX_TO_PARAMETER:
                self.assertTrue(
                    torch.equal(
                        shuffled[task][matrix].right,
                        repeated[task][matrix].right,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        shuffled[task][matrix].left,
                        repeated[task][matrix].left,
                    )
                )
        original = {
            task: collect_task_grams(
                self.model,
                task,
                12,
                4,
                split_seeds[task],
                self.transforms,
                0.9,
                torch.device("cpu"),
                0.4,
            )
            for task in task_ids
        }
        for matrix in MATRIX_TO_PARAMETER:
            pooled_original_right = sum(original[t][matrix].right for t in task_ids)
            pooled_shuffled_right = sum(shuffled[t][matrix].right for t in task_ids)
            pooled_original_left = sum(original[t][matrix].left for t in task_ids)
            pooled_shuffled_left = sum(shuffled[t][matrix].left for t in task_ids)
            self.assertTrue(torch.allclose(pooled_original_right, pooled_shuffled_right))
            self.assertTrue(torch.allclose(pooled_original_left, pooled_shuffled_left))
        self.assertFalse(torch.equal(shuffled[0]["Q"].right, shuffled[1]["Q"].right))

    def test_behavioral_controls_modify_only_the_declared_inputs(self) -> None:
        batch = generate_task_batch(6, 16, 991, self.transforms, 0.9, 0.4)
        zero_context, zero_query = controlled_inputs(
            batch, self.transforms, "query_task_code_zero", 10
        )
        recovered_zero_query = zero_query @ self.transforms.query
        self.assertTrue(torch.equal(zero_context, batch.context))
        self.assertLess(float(recovered_zero_query[:, TASK_SLICE].abs().max()), 1e-5)

        shuffled_context, shuffled_query = controlled_inputs(
            batch, self.transforms, "query_task_code_shuffle", 10
        )
        recovered_shuffled_query = shuffled_query @ self.transforms.query
        self.assertTrue(torch.equal(shuffled_context, batch.context))
        self.assertTrue(
            torch.all(recovered_shuffled_query[:, TASK_SLICE].argmax(dim=1) != batch.task_id)
        )

        permuted_context, permuted_query = controlled_inputs(
            batch, self.transforms, "context_label_permutation", 11
        )
        recovered_context = permuted_context @ self.transforms.context
        original_label_counts = batch.context_latent[:, :, LABEL_SLICE].sum(dim=1)
        permuted_label_counts = recovered_context[:, :, LABEL_SLICE].sum(dim=1)
        self.assertTrue(torch.allclose(original_label_counts, permuted_label_counts, atol=1e-5))
        self.assertTrue(torch.equal(permuted_query, batch.query))

        query_only_context, query_only_query = controlled_inputs(
            batch, self.transforms, "query_only", 12
        )
        self.assertEqual(int(torch.count_nonzero(query_only_context)), 0)
        self.assertTrue(torch.equal(query_only_query, batch.query))

    def test_k95_k99_and_specificity_use_only_classification_accuracy(self) -> None:
        rank_rows = []
        for seed, baseline in ((1, 1.0), (2, 0.98)):
            for rank, accuracy in ((0, 0.03), (8, 0.94), (16, 0.97), (32, 0.99)):
                rank_rows.append(
                    {
                        "task": "task_a",
                        "matrix": "Q",
                        "method": "right_gram",
                        "seed": seed,
                        "rank": rank,
                        "accuracy": accuracy,
                        "full_model_accuracy": baseline,
                    }
                )
        summary = minimum_rank_table(
            pd.DataFrame(rank_rows), ["task", "matrix", "method"]
        )
        self.assertEqual(int(summary.loc[0, "K95"]), 16)
        self.assertEqual(int(summary.loc[0, "K99"]), 32)

        specificity_rows = []
        for seed in (1, 2):
            for evaluation_task, accuracy in (("task_a", 0.9), ("task_b", 0.5), ("task_c", 0.7)):
                specificity_rows.append(
                    {
                        "seed": seed,
                        "source_task": "task_a",
                        "evaluation_task": evaluation_task,
                        "matrix": "Q",
                        "method": "right_gram",
                        "rank": 8,
                        "accuracy": accuracy,
                        "full_model_accuracy": 1.0,
                    }
                )
        specificity = task_specificity_table(pd.DataFrame(specificity_rows))
        self.assertAlmostEqual(float(specificity.loc[0, "diagonal_retention"]), 0.9)
        self.assertAlmostEqual(float(specificity.loc[0, "mean_off_diagonal_retention"]), 0.6)
        self.assertAlmostEqual(float(specificity.loc[0, "specificity"]), 0.3)


if __name__ == "__main__":
    unittest.main()
