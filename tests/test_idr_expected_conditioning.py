"""Validate exact averaging of the sampled TB gradient, not teacher training."""

import unittest

import torch
from test_idr_exact_diagnostic import diagnostic, fixture
from test_idr_update_diagnostic import conditioning, load_script, updates

from stability_gfn.idr.context_experiment import DirectContextHead

expected = load_script(
    "fit_idr_expected_conditioning_cpu",
    {"condition_idr_exact_cpu": conditioning, "diagnose_idr_updates_cpu": updates},
)


class ExpectedConditioningTests(unittest.TestCase):
    def test_gradient_equals_enumeration_of_all_sample_pairs(self):
        # Compare through normalized logits so accidental differentiation of proposal
        # weights would produce an extra gradient and fail this test.
        logits = torch.tensor([0.3, -0.4, 0.9], dtype=torch.float64, requires_grad=True)
        log_q = logits.log_softmax(0)
        normalizer = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        reward = torch.tensor([-1.0, -2.5, -0.8], dtype=torch.float64)
        for stratum in (
            torch.tensor([True, False, True]),
            torch.tensor([False, True, False]),
        ):
            loss = expected.expected_loss(log_q, reward, normalizer, stratum)
            actual = torch.autograd.grad(loss, (logits, normalizer), retain_graph=True)
            totals = [torch.zeros_like(value) for value in actual]
            residual = normalizer + log_q - reward
            for fresh in range(3):
                for stored in torch.where(stratum)[0].tolist():
                    pair_loss = (
                        residual[fresh].square() + residual[stored].square()
                    ) / 2
                    gradients = torch.autograd.grad(
                        pair_loss, (logits, normalizer), retain_graph=True
                    )
                    weight = log_q.detach().exp()[fresh] / stratum.sum()
                    for total, gradient in zip(totals, gradients):
                        total.add_(weight * gradient)
            for value, total in zip(actual, totals):
                torch.testing.assert_close(value, total, atol=1e-12, rtol=1e-12)

    def test_fit_preserves_source_and_does_not_train_on_target_probabilities(self):
        torch.set_num_threads(1)
        rows, cache = fixture()
        tree = diagnostic.FiniteBankTree(rows, cache)
        tree.log_reward = tree.log_prior + torch.tensor([0.0, -2.0, 0.0])
        tree.target = tree.log_reward.softmax(0)
        torch.manual_seed(21)
        head = conditioning.ConditionedHead(DirectContextHead(4, 6, 20, 8), [tree])
        before = {key: value.clone() for key, value in head.state_dict().items()}
        offsets = torch.zeros(1)
        config = dict(
            steps=8,
            learning_rate=1e-4,
            normalizer_learning_rate=0.01,
            gradient_clip=1.0,
            decay_fraction=0.75,
            final_lr_factor=0.1,
        )
        _, first, first_z, trace = expected.fit_expected([tree], head, offsets, config)
        tree.target = tree.target.flip(0)
        _, second, second_z, repeat = expected.fit_expected(
            [tree], head, offsets, config
        )
        self.assertEqual(trace, repeat)
        torch.testing.assert_close(first_z, second_z, atol=0, rtol=0)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
            torch.testing.assert_close(
                head.state_dict()[key], before[key], atol=0, rtol=0
            )
        self.assertFalse(torch.cuda.is_initialized())
