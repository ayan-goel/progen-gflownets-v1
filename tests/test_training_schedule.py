"""CPU checks for the rate boundary and observational exact monitoring."""

import copy
import json
import unittest

import torch

from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.guidance import CachedTask
from stability_gfn.metrics import distribution_metrics
from stability_gfn.pipeline import ROOT, validate_config
from stability_gfn.training import train_supervised, train_tb, validate_training_options


class TrainingScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        context = SequenceContext("AA", 2, "AG")
        tree = PrefixTree(context, context.enumerate_sequences())
        generator = torch.Generator().manual_seed(7)
        self.task = CachedTask(
            tree,
            torch.randn(len(tree.prefixes), 8, generator=generator),
            torch.randn(len(tree.prefixes), 2, generator=generator),
            [-1, 0, 0, -1],
        )
        self.config = {
            "steps": 10,
            "hidden_size": 8,
            "batch_size": 8,
            "learning_rate": 0.01,
            "normalizer_learning_rate": 0.02,
            "beta": 1.0,
            "gradient_clip": 1.0,
            "log_every": 4,
            "rollouts": 16,
            "lr_decay": {"after_fraction": 0.8, "factor": 0.1},
        }

    def test_both_optimizers_use_original_rate_through_boundary(self):
        for train in (train_tb, train_supervised):
            with self.subTest(method=train.__name__):
                result = train(self.task, self.config, seed=0)
                history = result[-2]
                self.assertEqual([row["step"] for row in history], [1, 4, 8, 9, 10])
                for row in history:
                    self.assertAlmostEqual(
                        row["learning_rate"], 0.01 if row["step"] <= 8 else 0.001
                    )
                    if train is train_tb:
                        self.assertAlmostEqual(
                            row["normalizer_learning_rate"],
                            0.02 if row["step"] <= 8 else 0.002,
                        )
                # The schedule actually affects updates, not just reported rates.
                constant = {
                    key: value
                    for key, value in self.config.items()
                    if key != "lr_decay"
                }
                constant_result = train(self.task, constant, seed=0)
                self.assertTrue(
                    any(
                        not torch.equal(value, constant_result[0].state_dict()[name])
                        for name, value in result[0].state_dict().items()
                    )
                )

    def test_monitoring_preserves_weights_normalizer_losses_and_rng(self):
        for train in (train_tb, train_supervised):
            with self.subTest(method=train.__name__):
                plain = train(self.task, self.config, seed=2)
                plain_rng = torch.random.get_rng_state().clone()
                emitted = []
                observed = train(
                    self.task,
                    {**self.config, "monitor_exact_distribution": True},
                    seed=2,
                    progress=emitted.append,
                )
                self.assertTrue(torch.equal(plain_rng, torch.random.get_rng_state()))
                for name, value in plain[0].state_dict().items():
                    torch.testing.assert_close(
                        observed[0].state_dict()[name], value, atol=0, rtol=0
                    )
                if train is train_tb:
                    torch.testing.assert_close(observed[1], plain[1], atol=0, rtol=0)
                self.assertEqual(
                    [row["loss"] for row in plain[-2]],
                    [row["loss"] for row in observed[-2] if row["step"] > 0],
                )
                self.assertEqual(emitted, observed[-2])
                self.assertEqual([row["step"] for row in emitted], [0, 1, 4, 8, 9, 10])
                self.assertIsNone(emitted[0]["loss"])
                expected = distribution_metrics(
                    self.task.all_log_probabilities(observed[0]),
                    self.task.base_log_probabilities,
                    self.task.scores,
                    1,
                    observed[1] if train is train_tb else None,
                )
                self.assertEqual(emitted[-1]["distribution"], expected)
                cost = observed[-1]
                self.assertEqual(cost["exact_monitoring_calls"], 6)
                self.assertEqual(cost["exact_monitoring_cached_score_reads"], 24)
                self.assertEqual(
                    cost["cached_terminal_reward_lookups"],
                    plain[-1]["cached_terminal_reward_lookups"],
                )
                self.assertGreater(cost["exact_monitoring_seconds"], 0)
                self.assertAlmostEqual(
                    cost["training_seconds"],
                    cost["training_seconds_excluding_exact_monitoring"]
                    + cost["exact_monitoring_seconds"],
                )
                self.assertEqual(plain[-1]["exact_monitoring_calls"], 0)
                self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_schedules_are_rejected(self):
        for schedule in (
            {},
            {"after_fraction": 0.8},
            {"after_fraction": 0.8, "factor": 0},
            {"after_fraction": 0.8, "factor": 1},
            {"after_fraction": float("nan"), "factor": 0.1},
            {"after_fraction": 0.8, "factor": float("inf")},
            {"after_fraction": 0.01, "factor": 0.1},
            {"after_fraction": True, "factor": 0.1},
        ):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                validate_training_options({**self.config, "lr_decay": schedule})
        with self.assertRaises(ValueError):
            validate_training_options(
                {**self.config, "monitor_exact_distribution": "true"}
            )

    def test_exact_monitoring_cannot_enable_biological_training_target_access(self):
        config = json.loads((ROOT / "configs/biological.json").read_text())
        for section in ("training", "supervised"):
            modified = copy.deepcopy(config)
            modified[section]["monitor_exact_distribution"] = True
            with self.assertRaises(ValueError):
                validate_config(modified)
        validate_config(json.loads((ROOT / "configs/exact-decay.json").read_text()))


if __name__ == "__main__":
    unittest.main()
