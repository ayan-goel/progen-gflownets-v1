"""Check sampled residual gradients and matching against the existing CPU trainer."""

import unittest

import torch
from test_idr_exact_diagnostic import diagnostic, fixture
from test_idr_update_diagnostic import conditioning, load_script, updates

from stability_gfn.idr.context_experiment import DirectContextHead

sampled = load_script(
    "compare_idr_sampled_conditioning_cpu",
    {"condition_idr_exact_cpu": conditioning, "diagnose_idr_updates_cpu": updates},
)


class SampledConditioningTests(unittest.TestCase):
    def test_loss_differentiates_only_the_two_sampled_residuals(self):
        log_q = torch.tensor([-0.3, -1.5, -3.0], requires_grad=True)
        reward = torch.tensor([-1.0, -3.0, -4.0])
        normalizer = torch.tensor(0.4, requires_grad=True)
        loss, fresh = sampled.sampled_loss(
            log_q, reward, normalizer, 2, torch.Generator().manual_seed(9)
        )
        gradient, normalizer_gradient = torch.autograd.grad(loss, (log_q, normalizer))
        residual = normalizer.detach() + log_q.detach() - reward
        expected = torch.zeros_like(log_q)
        expected[fresh] += residual[fresh]
        expected[2] += residual[2]
        torch.testing.assert_close(gradient, expected)
        torch.testing.assert_close(normalizer_gradient, expected.sum())

    def test_matches_legacy_control_and_pairs_replay_without_mutating_start(self):
        torch.set_num_threads(1)
        rows, cache = fixture()
        tree = diagnostic.FiniteBankTree(rows, cache)
        tree.log_reward = tree.log_prior + torch.tensor([0.0, -2.0, 0.0])
        tree.target = tree.log_reward.softmax(0)
        torch.manual_seed(21)
        head = DirectContextHead(4, 6, 20, 8)
        before = {k: v.clone() for k, v in head.state_dict().items()}
        delta = tree.log_reward - tree.log_probabilities(head).detach().double()
        offsets = torch.stack(
            [(delta[tree.passed].mean() + delta[~tree.passed].mean()) / 2]
        ).float()
        config = dict(
            steps=8,
            learning_rate=1e-4,
            normalizer_learning_rate=0.01,
            gradient_clip=1.0,
            decay_fraction=0.75,
            final_lr_factor=0.1,
        )
        legacy = diagnostic.fit_condition([tree], head, config, "sampled_tb")
        ordinary, _, _, ordinary_samples = sampled.fit_sampled(
            [tree], head, offsets, config, 0
        )
        self.assertEqual(legacy["mean_tv"], ordinary["mean_tv"])
        conditioned = conditioning.ConditionedHead(head, [tree])
        _, _, _, conditioned_samples = sampled.fit_sampled(
            [tree], conditioned, offsets, config, 0
        )
        self.assertEqual(
            [r["replay"] for r in ordinary_samples],
            [r["replay"] for r in conditioned_samples],
        )
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        # Exact targets are evaluation-only: changing them cannot change fitted weights.
        tree.target = tree.target.flip(0)
        _, first, _, samples = sampled.fit_sampled([tree], head, offsets, config, 1)
        tree.target = tree.target.flip(0)
        _, second, _, repeated = sampled.fit_sampled([tree], head, offsets, config, 1)
        self.assertEqual(samples, repeated)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())
