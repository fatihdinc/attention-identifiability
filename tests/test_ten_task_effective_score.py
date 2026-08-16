from __future__ import annotations

import unittest

import torch

from identifiability_llm.ten_task_attention import (
    TASK_NAMES,
    generate_task_batch,
    initialize_model,
    make_transforms,
    parameter_hashes,
)
from identifiability_llm.ten_task_effective_score import (
    ALL_EFFECTIVE_SCORE_METHODS,
    build_effective_score_basis,
    collect_effective_score_grams,
    cumulative_svd_power,
    direct_score_audit,
    effective_score_gram_audit,
    effective_score_matrix,
    effective_score_reconstruction_audit,
    forward_with_effective_score,
    gauge_invariance_audit,
    reconstruct_effective_score,
    train_joint_exposure_matched,
)


class TenTaskEffectiveScoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.seed = 3181
        cls.model, _ = initialize_model(cls.seed, 0.25)
        cls.transforms = make_transforms(cls.seed)
        cls.batch = generate_task_batch(0, 32, 9181, cls.transforms, 0.9, 0.4)
        cls.grams = collect_effective_score_grams(
            cls.model,
            0,
            32,
            16,
            7711,
            cls.transforms,
            0.9,
            0.4,
        )
        cls.matrix = effective_score_matrix(cls.model)
        cls.basis = build_effective_score_basis(cls.matrix, cls.grams)

    def test_direct_effective_matrix_matches_original_qk_path(self) -> None:
        audit = direct_score_audit(
            self.model,
            self.batch.context,
            self.batch.query,
            device=torch.device("cpu"),
        )
        self.assertTrue(audit["passed"], audit)
        original, original_details = self.model(self.batch.context, self.batch.query)
        direct, direct_details = forward_with_effective_score(
            self.model, self.batch.context, self.batch.query, self.matrix
        )
        self.assertTrue(torch.equal(original.argmax(-1), direct.argmax(-1)))
        self.assertTrue(
            torch.allclose(
                original_details["attention"], direct_details["attention"], atol=2e-6
            )
        )

    def test_four_grams_have_correct_counts_and_two_sided_equivalences(self) -> None:
        self.assertEqual(self.grams.query_observation_count, 32)
        self.assertEqual(self.grams.key_observation_count, 32 * 64)
        audit = effective_score_gram_audit(self.matrix, self.grams)
        self.assertTrue(audit["passed"], audit)

    def test_all_five_reconstructions_obey_rank_and_full_rank_invariants(self) -> None:
        for method in ALL_EFFECTIVE_SCORE_METHODS:
            for rank in (0, 8, 128):
                reconstructed, projector, _ = reconstruct_effective_score(
                    self.basis, method, rank
                )
                audit = effective_score_reconstruction_audit(
                    self.matrix, reconstructed, projector, rank
                )
                self.assertTrue(audit["rank_bound_satisfied"], (method, rank, audit))
                if rank == 0:
                    self.assertEqual(int(torch.count_nonzero(reconstructed)), 0)
                if rank == 128:
                    self.assertLess(audit["relative_matrix_error"], 1e-10)
                if projector is not None:
                    self.assertLess(audit["projector_symmetry_error"], 1e-12)
                    self.assertLess(audit["projector_idempotence_error"], 1e-10)

    def test_cumulative_power_is_monotone_with_exact_endpoints(self) -> None:
        ranks = [0, 1, 2, 8, 32, 64, 128]
        values = [cumulative_svd_power(self.basis, rank) for rank in ranks]
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 1.0, places=12)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_gauge_transform_preserves_effective_scores(self) -> None:
        audit = gauge_invariance_audit(
            self.model, self.batch.context, self.batch.query, 5519
        )
        self.assertTrue(audit["passed"], audit)

    def test_small_exposure_matched_training_counts_every_task_equally(self) -> None:
        seed = 61
        model, _ = initialize_model(seed, 0.25)
        transforms = make_transforms(seed)
        before = parameter_hashes(model)
        result = train_joint_exposure_matched(
            model,
            transforms,
            seed,
            "unit_test",
            "unit_test_effective_score",
            0.9,
            0.4,
            torch.device("cpu"),
            learning_rate=0.003,
            weight_decay=0.0,
            task_batch_size=2,
            outer_updates=2,
            evaluate_every=2,
            evaluation_batch_size=4,
            validation_count=4,
        )
        self.assertTrue(all(value == 2 for value in result.task_batch_counts.values()))
        self.assertTrue(all(value == 4 for value in result.task_example_counts.values()))
        self.assertEqual(set(result.task_batch_counts), set(TASK_NAMES))
        self.assertNotEqual(before, parameter_hashes(model))


if __name__ == "__main__":
    unittest.main()
