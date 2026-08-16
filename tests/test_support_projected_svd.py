from __future__ import annotations

import unittest

import torch

from identifiability_llm.support_projected_svd import (
    build_support_projected_svd,
    reconstruct_support_projected_svd,
    support_projected_svd_power,
)
from identifiability_llm.ten_task_attention import (
    generate_task_batch,
    initialize_model,
    make_transforms,
)
from identifiability_llm.ten_task_effective_score import (
    build_effective_score_basis,
    collect_effective_score_grams,
    effective_score_matrix,
)


class SupportProjectedSVDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        seed = 3181
        cls.model, _ = initialize_model(seed, 0.25)
        cls.transforms = make_transforms(seed)
        cls.batch = generate_task_batch(0, 32, 9181, cls.transforms, 0.9, 0.4)
        grams = collect_effective_score_grams(
            cls.model,
            0,
            64,
            16,
            7711,
            cls.transforms,
            0.9,
            0.4,
        )
        cls.matrix = effective_score_matrix(cls.model)
        cls.basis = build_effective_score_basis(cls.matrix, grams)

    def test_uses_only_non_null_input_bases_and_obeys_rank_bound(self) -> None:
        supported = build_support_projected_svd(self.basis)
        self.assertEqual(supported.query_support_rank, 33)
        self.assertEqual(supported.key_support_rank, 104)
        rank_eight = reconstruct_support_projected_svd(supported, 8)
        self.assertLessEqual(int(torch.linalg.matrix_rank(rank_eight)), 8)

    def test_full_lift_equals_two_sided_projection_and_preserves_scores(self) -> None:
        supported = build_support_projected_svd(self.basis)
        full_supported = reconstruct_support_projected_svd(supported, 128)
        query_projector = supported.query_vectors @ supported.query_vectors.T
        key_projector = supported.key_vectors @ supported.key_vectors.T
        expected = query_projector @ self.matrix @ key_projector
        self.assertTrue(torch.allclose(full_supported, expected, atol=1e-12, rtol=1e-12))

        query = self.batch.query.to(torch.float64)
        context = self.batch.context.to(torch.float64)
        original_scores = torch.einsum(
            "bd,de,bne->bn", query, self.matrix, context
        )
        supported_scores = torch.einsum(
            "bd,de,bne->bn", query, full_supported, context
        )
        self.assertTrue(
            torch.allclose(original_scores, supported_scores, atol=2e-6, rtol=1e-6)
        )

    def test_projected_power_is_monotone_with_exact_endpoints(self) -> None:
        supported = build_support_projected_svd(self.basis)
        ranks = [0, 1, 8, 32, 64, 128]
        values = [support_projected_svd_power(supported, rank) for rank in ranks]
        self.assertEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 1.0, places=12)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        self.assertEqual(values[-2], 1.0)


if __name__ == "__main__":
    unittest.main()
