"""CPU correctness checks for reward invariants and isolated predictor fitting."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from stability_gfn.reward_study import (
    compare_predictors,
    reward_distribution,
    target_metrics,
)


class RewardStudyTests(unittest.TestCase):
    def test_reward_invariants(self):
        prior = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        prediction = torch.tensor([-1.0, 1.0, 3.0], dtype=torch.float64)
        torch.testing.assert_close(
            reward_distribution(prior.log(), prediction, 0.0, 2.0, 0.0), prior
        )
        linear = reward_distribution(prior.log(), prediction, 0.0, 2.0, 1.0)
        torch.testing.assert_close(
            linear, reward_distribution(prior.log(), prediction, 7.0, 2.0, 1.0)
        )
        torch.testing.assert_close(
            linear, reward_distribution(prior.log(), prediction, 0.0, 4.0, 2.0)
        )
        capped = reward_distribution(prior.log(), prediction, 0.0, 2.0, 1.0, "capped")
        self.assertAlmostEqual(float(capped[1] / capped[2]), float(prior[1] / prior[2]))
        self.assertTrue(bool(torch.all(capped > 0)))
        metrics = target_metrics(
            torch.tensor([1.0, 0.0, 0.0]), prediction, prediction, 0.0
        )
        self.assertAlmostEqual(metrics["expected_unique_in_256"], 1.0)
        self.assertAlmostEqual(metrics["effective_support"], 1.0)

    def test_tiny_comparison_and_evaluation_label_isolation(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(20)
        features = {
            split: torch.randn(n, 4, generator=generator)
            for split, n in [("train", 12), ("validation", 6), ("test", 6)]
        }
        rows = {
            split: [
                {"sequence": f"{split}-{i}", "stability": float(2 * x[0] + 4)}
                for i, x in enumerate(values)
            ]
            for split, values in features.items()
        }
        settings = {
            "seeds": [0],
            "train_sizes": [4, 12],
            "betas": [0.0, 1.0],
            "rewards": ["linear", "capped"],
            "top_k": 2,
        }
        scorer = {
            "ridge_penalties": [0.1, 1.0],
            "hidden_size": 4,
            "steps": 3,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip": 1.0,
            "validate_every": 1,
            "minimum_spearman": 0.2,
        }
        with tempfile.TemporaryDirectory() as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            first.mkdir()
            second.mkdir()
            results = compare_predictors(
                features, rows, 0, torch.zeros(6), settings, scorer, first, "cpu"
            )
            self.assertEqual(len(results), 6)
            for row in rows["test"]:
                row["stability"] += 100
            compare_predictors(
                features, rows, 0, torch.zeros(6), settings, scorer, second, "cpu"
            )
            for path in first.glob("*-predictions.json"):
                self.assertEqual(
                    json.loads(path.read_text())["predictions"],
                    json.loads((second / path.name).read_text())["predictions"],
                )
            subsets = json.loads((first / "subsets.json").read_text())
            self.assertTrue(
                set(subsets[0]["sequences"]) <= set(subsets[1]["sequences"])
            )
            for family in ["ridge", "mlp"]:
                candidates = json.loads(
                    (first / f"seed-0-n-4-{family}-fit.json").read_text()
                )["candidates"]
                self.assertTrue(all(c["model"].startswith(family) for c in candidates))
