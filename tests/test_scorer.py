"""Synthetic scorer gates and reward transformations, using tiny CPU fixtures."""

import unittest

import torch

from stability_gfn.scorer import fit_scorer, regression_metrics, stability_scores


class ScorerTests(unittest.TestCase):
    def test_fit_scaling_and_frozen_reward(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(3)
        features = torch.randn(40, 4, generator=generator)
        labels = 3 * features[:, 0] + 2
        config = {
            "ridge_penalties": [0.01, 1.0],
            "hidden_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "steps": 10,
            "batch_size": 8,
            "gradient_clip": 1.0,
            "validate_every": 5,
            "minimum_spearman": 0.2,
        }
        model, report = fit_scorer(
            features[:30], labels[:30], features[30:], labels[30:], config
        )
        self.assertTrue(report["gate_passed"])
        torch.testing.assert_close(model.feature_mean, features[:30].mean(0))
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.parameters())
        )
        predictions, scores = stability_scores(model, features[:2], features[:1])
        self.assertAlmostEqual(scores[0].item(), 0)
        self.assertAlmostEqual(
            scores[1].item(),
            ((predictions[1] - predictions[0]) / model.label_scale).item(),
        )
        self.assertIsNone(regression_metrics([1, 1], [1, 2])["spearman"])

    def test_tied_spearman(self):
        self.assertAlmostEqual(regression_metrics([1, 1, 3], [2, 2, 4])["spearman"], 1)


if __name__ == "__main__":
    unittest.main()
