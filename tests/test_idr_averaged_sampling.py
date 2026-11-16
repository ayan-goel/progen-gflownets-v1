"""Verify averaged residual gradients and backward compatibility at one draw."""

import unittest

import torch
from test_idr_exact_diagnostic import diagnostic, fixture
from test_idr_sampled_conditioning import sampled
from test_idr_update_diagnostic import conditioning, load_script, updates

from stability_gfn.idr.context_experiment import DirectContextHead

averaged = load_script(
    "fit_idr_averaged_sampling_cpu",
    {"condition_idr_exact_cpu": conditioning, "diagnose_idr_updates_cpu": updates},
)


class AveragedSamplingTests(unittest.TestCase):
    def test_gradient_counts_duplicate_draws_and_averages_before_update(self):
        log_q = torch.tensor(
            [-0.3, -1.5, -3.0], dtype=torch.float64, requires_grad=True
        )
        reward = torch.tensor([-1.0, -3.0, -4.0], dtype=torch.float64)
        normalizer = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        stratum = torch.tensor([True, False, True])
        loss, fresh, stored = averaged.averaged_loss(
            log_q,
            reward,
            normalizer,
            stratum,
            torch.Generator().manual_seed(9),
            torch.Generator().manual_seed(5),
            16,
        )
        self.assertEqual(len(fresh), 16)
        self.assertEqual(len(stored), 16)
        self.assertTrue(all(stratum[index] for index in stored))
        residual = normalizer.detach() + log_q.detach() - reward
        counts = torch.bincount(torch.tensor(fresh + stored), minlength=3).double()
        expected = counts * residual / 16
        actual, actual_z = torch.autograd.grad(loss, (log_q, normalizer))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual_z, expected.sum(), atol=1e-12, rtol=1e-12)

    def test_single_draw_matches_original_trainer_and_preserves_source(self):
        torch.set_num_threads(1)
        rows, cache = fixture()
        tree = diagnostic.FiniteBankTree(rows, cache)
        tree.log_reward = tree.log_prior + torch.tensor([0.0, -2.0, 0.0])
        tree.target = tree.log_reward.softmax(0)
        torch.manual_seed(21)
        head = conditioning.ConditionedHead(DirectContextHead(4, 6, 20, 8), [tree])
        before = {key: value.clone() for key, value in head.state_dict().items()}
        config = dict(
            steps=8,
            learning_rate=1e-4,
            normalizer_learning_rate=0.01,
            gradient_clip=1.0,
            decay_fraction=0.75,
            final_lr_factor=0.1,
        )
        original, original_head, _, _ = sampled.fit_sampled(
            [tree], head, torch.zeros(1), config, 0
        )
        result, trained, _, _ = averaged.fit_averaged(
            [tree], head, torch.zeros(1), config, 0, draws=1
        )
        self.assertEqual(original["mean_tv"], result["mean_tv"])
        self.assertEqual(result["trajectory_draws"], 16)
        for key, value in trained.state_dict().items():
            torch.testing.assert_close(
                value, original_head.state_dict()[key], rtol=0, atol=0
            )
            torch.testing.assert_close(
                before[key], head.state_dict()[key], rtol=0, atol=0
            )
        self.assertFalse(torch.cuda.is_initialized())
