"""Probability identities and tiny CPU optimization checks, not real experiments."""

import unittest

import torch

from stability_gfn.backbone import FrozenProGen, cache_tree
from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.guidance import CachedTask, trajectory_balance
from stability_gfn.metrics import distribution_metrics
from stability_gfn.training import train_supervised, train_tb


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny()
        context = SequenceContext("AA", 2, "AG")
        tree = PrefixTree(context, context.enumerate_sequences())
        features, logits = cache_tree(cls.backbone, tree)
        cls.task = CachedTask(tree, features, logits, [-1, 0, 0, -1])
        cls.config = {
            "hidden_size": 32,
            "steps": 80,
            "batch_size": 32,
            "learning_rate": 0.01,
            "normalizer_learning_rate": 0.03,
            "beta": 1.0,
            "gradient_clip": 10.0,
            "log_every": 40,
            "rollouts": 64,
        }

    def test_zero_guidance_normalization_and_forced_probabilities(self):
        task = self.task
        log_q = task.all_log_probabilities(task.head())
        torch.testing.assert_close(log_q, task.base_log_probabilities, atol=0, rtol=0)
        self.assertAlmostEqual(log_q.exp().sum().item(), 1.0, places=6)
        loss = trajectory_balance(
            log_q, task.base_log_probabilities, task.scores, torch.tensor(0.0), 0
        )
        self.assertEqual(loss.item(), 0)
        tree = PrefixTree(SequenceContext("AA", 1, "AG"), ["GA"])
        features, logits = cache_tree(self.backbone, tree)
        forced = CachedTask(tree, features, logits, [0])
        self.assertEqual(forced.all_log_probabilities(forced.head()).item(), 0)

    def test_exact_target_has_zero_tb_residual(self):
        task = self.task
        log_weights = task.base_log_probabilities + task.scores
        log_z = log_weights.logsumexp(0)
        loss = trajectory_balance(
            log_weights - log_z, task.base_log_probabilities, task.scores, log_z, 1
        )
        self.assertLess(loss.item(), 1e-12)

    def test_cpu_tb_changes_head_but_not_backbone(self):
        before = {
            name: value.clone()
            for name, value in self.backbone.model.state_dict().items()
        }
        head, log_z, history, cost = train_tb(self.task, self.config, seed=0)
        result = distribution_metrics(
            self.task.all_log_probabilities(head),
            self.task.base_log_probabilities,
            self.task.scores,
            1,
            log_z,
        )
        self.assertLess(result["total_variation"], 0.1)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        for name, value in self.backbone.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        leaves = torch.arange(4)
        torch.testing.assert_close(
            self.task.log_probabilities(head, leaves),
            self.task.all_log_probabilities(head),
        )

    def test_supervised_baseline_runs(self):
        head, history, cost = train_supervised(self.task, self.config, seed=0)
        self.assertLess(history[-1]["loss"], history[0]["loss"])
        self.assertAlmostEqual(
            self.task.all_log_probabilities(head).exp().sum().item(), 1, places=6
        )

    def test_sample_frequencies_and_beta_zero_training(self):
        task = self.task
        leaves = task.sample(None, 5000, torch.Generator().manual_seed(13))
        empirical = torch.bincount(leaves, minlength=4).float() / 5000
        self.assertLess(
            (empirical - task.base_log_probabilities.exp()).abs().max().item(), 0.04
        )
        config = {**self.config, "beta": 0, "steps": 3}
        head, log_z, history, _ = train_tb(task, config, seed=1)
        torch.testing.assert_close(
            task.all_log_probabilities(head),
            task.base_log_probabilities,
            atol=0,
            rtol=0,
        )
        self.assertEqual(log_z.item(), 0)


if __name__ == "__main__":
    unittest.main()
