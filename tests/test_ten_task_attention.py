from __future__ import annotations

import math
import unittest

import torch

from identifiability_llm.ten_task_attention import (
    A_SLICE,
    B_SLICE,
    CATEGORY_COUNT,
    CATEGORY_SLICE,
    CONSTANT_INDEX,
    D_MODEL,
    LABEL_COUNT,
    LABEL_SLICE,
    N_CONTEXT,
    PRIORITY_INDEX,
    RESERVED_SLICE,
    TASK_COUNT,
    TASK_SLICE,
    TenTaskAttention,
    TRAINING_CONFIGURATIONS,
    audit_transforms,
    generate_task_batch,
    initialize_model,
    make_transforms,
    model_architecture_audit,
    parameter_hashes,
    train_configuration,
)


class TenTaskAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transforms = make_transforms(1701)

    def generate(self, task_id: int, count: int = 128):
        return generate_task_batch(task_id, count, 901_000 + task_id, self.transforms, 0.9)

    def test_transforms_are_dense_distinct_orthogonal_and_full_rank(self) -> None:
        audit = audit_transforms(self.transforms)
        self.assertTrue(audit["passed"], audit)
        self.assertGreater(float((self.transforms.context.abs() > 1e-8).float().mean()), 0.99)
        self.assertGreater(float((self.transforms.query.abs() > 1e-8).float().mean()), 0.99)

    def test_exact_fourteen_training_configurations_and_mixture_memberships(self) -> None:
        self.assertEqual(len(TRAINING_CONFIGURATIONS), 14)
        self.assertEqual(TRAINING_CONFIGURATIONS["retrieval_mixture"], (0, 1, 2, 3, 4, 5))
        self.assertEqual(TRAINING_CONFIGURATIONS["rule_mixture"], (0, 5, 6, 7, 8))
        self.assertEqual(TRAINING_CONFIGURATIONS["aggregation_mixture"], (1, 4, 8, 9))
        self.assertEqual(TRAINING_CONFIGURATIONS["all_tasks"], tuple(range(10)))
        for task_id in range(10):
            self.assertEqual(TRAINING_CONFIGURATIONS[f"task_{task_id + 1}"], (task_id,))

    def test_all_tasks_have_exact_common_record_structure(self) -> None:
        for task_id in range(TASK_COUNT):
            batch = self.generate(task_id, 32)
            self.assertEqual(tuple(batch.context_latent.shape), (32, N_CONTEXT, D_MODEL))
            self.assertEqual(tuple(batch.query_latent.shape), (32, D_MODEL))
            self.assertEqual(tuple(batch.context.shape), (32, N_CONTEXT, D_MODEL))
            self.assertEqual(tuple(batch.query.shape), (32, D_MODEL))
            self.assertEqual(tuple(batch.targets.shape), (32,))
            self.assertTrue(torch.equal(batch.query_latent[:, LABEL_SLICE], torch.zeros(32, 32)))
            self.assertTrue(
                torch.equal(
                    batch.context_latent[:, :, TASK_SLICE], torch.zeros(32, N_CONTEXT, TASK_COUNT)
                )
            )
            expected_code = torch.nn.functional.one_hot(
                torch.full((32,), task_id), TASK_COUNT
            ).float()
            self.assertTrue(torch.equal(batch.query_latent[:, TASK_SLICE], expected_code))
            self.assertTrue(torch.equal(batch.context_latent[:, :, CONSTANT_INDEX], torch.ones(32, 64)))
            self.assertTrue(torch.equal(batch.query_latent[:, CONSTANT_INDEX], torch.ones(32)))
            self.assertEqual(int(torch.count_nonzero(batch.context_latent[:, :, RESERVED_SLICE])), 0)
            self.assertEqual(int(torch.count_nonzero(batch.query_latent[:, RESERVED_SLICE])), 0)
            self.assertGreaterEqual(int(batch.targets.min()), 0)
            self.assertLess(int(batch.targets.max()), LABEL_COUNT)

    def test_tasks_one_through_six_follow_declared_lookup_rules(self) -> None:
        for task_id in range(6):
            batch = self.generate(task_id)
            rows = torch.arange(batch.count)
            targets = batch.metadata["target_indices"]
            target_labels = batch.context_latent[rows, targets, LABEL_SLICE].argmax(dim=-1)
            self.assertTrue(torch.equal(target_labels, batch.targets))
            target_a = batch.context_latent[rows, targets, A_SLICE]
            target_b = batch.context_latent[rows, targets, B_SLICE]
            if task_id == 0:
                self.assertTrue(torch.equal(batch.query_latent[:, A_SLICE], target_a))
            elif task_id == 1:
                self.assertFalse(torch.equal(batch.query_latent[:, A_SLICE], target_a))
            elif task_id == 2:
                mask = batch.metadata["partial_mask"]
                self.assertTrue(torch.equal(mask.sum(dim=1), torch.full((batch.count,), 16.0)))
                self.assertTrue(torch.equal(batch.query_latent[:, A_SLICE], target_a * mask))
            elif task_id == 3:
                self.assertTrue(torch.equal(batch.query_latent[:, B_SLICE], target_b))
            elif task_id == 4:
                da = batch.metadata["a_distractor_indices"]
                db = batch.metadata["b_distractor_indices"]
                self.assertTrue(torch.equal(batch.context_latent[rows, da, A_SLICE], target_a))
                self.assertTrue(torch.equal(batch.context_latent[rows, db, B_SLICE], target_b))
                self.assertTrue(
                    torch.all((batch.context_latent[rows, da, B_SLICE] != target_b).any(dim=1))
                )
                self.assertTrue(
                    torch.all((batch.context_latent[rows, db, A_SLICE] != target_a).any(dim=1))
                )
            else:
                distractor = batch.metadata["category_distractor_indices"]
                target_category = batch.context_latent[rows, targets, CATEGORY_SLICE].argmax(dim=-1)
                distractor_category = batch.context_latent[
                    rows, distractor, CATEGORY_SLICE
                ].argmax(dim=-1)
                self.assertTrue(torch.equal(batch.context_latent[rows, distractor, A_SLICE], target_a))
                self.assertTrue(torch.all(distractor_category != target_category))
                self.assertTrue(
                    torch.equal(
                        batch.query_latent[:, CATEGORY_SLICE],
                        torch.nn.functional.one_hot(target_category, CATEGORY_COUNT).float(),
                    )
                )

    def test_highest_and_lowest_priority_tasks_have_exact_category_extrema(self) -> None:
        for task_id in (6, 7):
            batch = self.generate(task_id)
            labels = batch.context_latent[:, :, LABEL_SLICE].argmax(dim=-1)
            categories = batch.context_latent[:, :, CATEGORY_SLICE].argmax(dim=-1)
            priorities = batch.context_latent[:, :, PRIORITY_INDEX]
            query_category = batch.query_latent[:, CATEGORY_SLICE].argmax(dim=-1)
            mask = categories == query_category.unsqueeze(1)
            self.assertTrue(torch.equal(mask.sum(dim=1), torch.full((batch.count,), 8)))
            masked = priorities.masked_fill(~mask, -torch.inf if task_id == 6 else torch.inf)
            target_indices = masked.argmax(dim=1) if task_id == 6 else masked.argmin(dim=1)
            self.assertTrue(torch.equal(labels.gather(1, target_indices[:, None])[:, 0], batch.targets))
            outside = priorities.masked_fill(mask, -torch.inf if task_id == 6 else torch.inf)
            if task_id == 6:
                self.assertTrue(torch.all(outside.max(dim=1).values > masked.max(dim=1).values))
            else:
                self.assertTrue(torch.all(outside.min(dim=1).values < masked.min(dim=1).values))

    def test_category_majority_is_unique_four_of_seven(self) -> None:
        batch = self.generate(8)
        labels = batch.context_latent[:, :, LABEL_SLICE].argmax(dim=-1)
        categories = batch.context_latent[:, :, CATEGORY_SLICE].argmax(dim=-1)
        query_category = batch.query_latent[:, CATEGORY_SLICE].argmax(dim=-1)
        for row in range(batch.count):
            selected = labels[row, categories[row] == query_category[row]]
            self.assertEqual(selected.numel(), 7)
            counts = torch.bincount(selected, minlength=LABEL_COUNT)
            self.assertEqual(int(counts.max()), 4)
            self.assertEqual(int(counts.argmax()), int(batch.targets[row]))
            self.assertEqual(int((counts == 1).sum()), 3)

    def test_three_item_vote_query_and_majority_are_exact(self) -> None:
        batch = self.generate(9)
        voter_keys = batch.metadata["voter_keys"]
        voter_labels = batch.metadata["voter_labels"]
        expected_query = voter_keys.sum(dim=1) / math.sqrt(3.0)
        self.assertTrue(torch.equal(batch.query_latent[:, A_SLICE], expected_query))
        self.assertTrue(torch.equal(voter_labels[:, 0], batch.targets))
        self.assertTrue(torch.equal(voter_labels[:, 1], batch.targets))
        self.assertTrue(torch.all(voter_labels[:, 2] != batch.targets))

    def test_model_has_no_bypass_path_and_initialization_is_seed_paired(self) -> None:
        first, first_state = initialize_model(41, 0.25)
        second, second_state = initialize_model(41, 0.25)
        third, _ = initialize_model(42, 0.25)
        audit = model_architecture_audit(first)
        self.assertTrue(audit["passed"], audit)
        self.assertEqual(parameter_hashes(first), parameter_hashes(second))
        self.assertNotEqual(parameter_hashes(first), parameter_hashes(third))
        self.assertEqual(first_state.keys(), second_state.keys())
        context = torch.randn(4, N_CONTEXT, D_MODEL)
        query_a = torch.randn(4, D_MODEL)
        query_b = torch.randn(4, D_MODEL)
        logits_a, _ = first(context, query_a)
        logits_b, _ = first(context, query_b)
        zero_context = torch.zeros_like(context)
        zero_a, _ = first(zero_context, query_a)
        zero_b, _ = first(zero_context, query_b)
        self.assertFalse(torch.equal(logits_a, logits_b))
        self.assertTrue(torch.equal(zero_a, zero_b))

    def test_checkpoint_resume_matches_uninterrupted_training_exactly(self) -> None:
        seed = 71
        transforms = make_transforms(seed)
        device = torch.device("cpu")
        common = dict(
            task_ids=[0],
            configuration_name="resume_test",
            phase="pilot_resume_test",
            model_seed=seed,
            transforms=transforms,
            rho=0.9,
            device=device,
            learning_rate=0.003,
            batch_size=8,
            evaluation_batch_size=8,
            validation_count=16,
            evaluate_every=2,
            required_mean_accuracy=1.1,
            required_each_accuracy=1.1,
            three_vote_distractor_rho=0.4,
            stop_when_qualified=False,
        )
        uninterrupted, _ = initialize_model(seed, 0.25)
        train_configuration(uninterrupted, max_steps=6, **common)

        resumed, _ = initialize_model(seed, 0.25)
        first = train_configuration(resumed, max_steps=3, **common)
        second = train_configuration(
            resumed,
            max_steps=6,
            optimizer_state=first.optimizer_state,
            start_step=3,
            **common,
        )
        self.assertEqual(second.steps_completed, 6)
        self.assertEqual(parameter_hashes(resumed), parameter_hashes(uninterrupted))


if __name__ == "__main__":
    unittest.main()
