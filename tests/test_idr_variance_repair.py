"""Exact covariance and finite-sum control-variate correctness checks."""

import copy
import unittest

import torch
from test_idr_exact_diagnostic import diagnostic, fixture
from test_idr_update_diagnostic import conditioning, load_script, updates

from stability_gfn.idr.context_experiment import DirectContextHead

variance = load_script(
    "audit_idr_gradient_variance_cpu",
    {"condition_idr_exact_cpu": conditioning, "diagnose_idr_updates_cpu": updates},
)
repair = load_script(
    "fit_idr_variance_reduced_cpu",
    {"condition_idr_exact_cpu": conditioning, "diagnose_idr_updates_cpu": updates},
)


class VarianceRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rows, cache = fixture()
        self.tree = diagnostic.FiniteBankTree(rows, cache)
        self.tree.log_reward = self.tree.log_prior + torch.tensor([0.0, -2.0, 0.0])
        self.tree.target = self.tree.log_reward.softmax(0)
        self.tree.fixed_weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        torch.manual_seed(21)
        self.head = conditioning.ConditionedHead(
            DirectContextHead(4, 6, 20, 8), [self.tree]
        )
        self.offset = torch.nn.Parameter(torch.zeros(1))

    def test_covariance_matches_centered_second_moment(self):
        gradients = torch.tensor([[1.0, 2.0], [3.0, -1.0], [0.0, 4.0]])
        weights = self.tree.fixed_weights
        mean, trace = variance.moments(gradients, weights)
        direct = sum(
            weights[i] * (gradients[i] - mean).square().sum() for i in range(3)
        )
        self.assertAlmostEqual(trace, float(direct), places=12)
        self.assertEqual(variance.moments(gradients[:1], torch.ones(1))[1], 0)

    def test_control_variate_unbiased_and_exact_at_snapshot(self):
        trees = [self.tree]
        _, snapshot_mean = repair.loss_gradient(trees, self.head, self.offset)
        current = copy.deepcopy(self.head)
        with torch.no_grad():
            current.output.weight.add_(0.01)
        _, current_mean = repair.loss_gradient(trees, current, self.offset)
        average = [torch.zeros_like(g) for g in current_mean]
        for leaf, probability in enumerate(self.tree.fixed_weights):
            indices = [torch.tensor([leaf])]
            _, old = repair.loss_gradient(trees, self.head, self.offset, indices)
            _, new = repair.loss_gradient(trees, current, self.offset, indices)
            corrected = repair.correct_gradient(new, old, snapshot_mean)
            at_snapshot = repair.correct_gradient(old, old, snapshot_mean)
            for total, gradient, exact, reference in zip(
                average, corrected, at_snapshot, snapshot_mean
            ):
                total.add_(probability * gradient)
                torch.testing.assert_close(exact, reference, atol=0, rtol=0)
        for value, reference in zip(average, current_mean):
            torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-5)

    def test_same_indices_no_oracle_training_and_unchanged_source(self):
        config = dict(
            steps=4,
            learning_rate=1e-4,
            normalizer_learning_rate=0.01,
            gradient_clip=1.0,
            decay_fraction=0.75,
            final_lr_factor=0.1,
        )
        before = {key: value.clone() for key, value in self.head.state_dict().items()}
        _, _, _, ordinary = repair.fit_fixed(
            [self.tree], self.head, self.offset, config, 0, "plain"
        )
        first_result, first, _, corrected = repair.fit_fixed(
            [self.tree], self.head, self.offset, config, 0, "svrg"
        )
        self.assertEqual(
            [s["indices"] for s in ordinary], [s["indices"] for s in corrected]
        )
        self.tree.target = self.tree.target.flip(0)
        _, second, _, _ = repair.fit_fixed(
            [self.tree], self.head, self.offset, config, 0, "svrg"
        )
        self.assertEqual(first_result["sampled_occurrences"], 128)
        self.assertEqual(first_result["full_snapshot_occurrences"], 3)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(
                before[key], self.head.state_dict()[key], rtol=0, atol=0
            )
        self.assertFalse(torch.cuda.is_initialized())
