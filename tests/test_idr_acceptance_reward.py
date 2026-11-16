"""Acceptance boundaries, reward ordering, and the TB integration convention."""

import math
import unittest
from dataclasses import replace

import torch

from stability_gfn.guidance import trajectory_balance
from stability_gfn.idr.acceptance_reward import (
    AcceptanceRewardConfig,
    joint_acceptance_reward,
)
from stability_gfn.idr.charge import charge_reward


def tail(first_positive, second_negative):
    """Simple canonical 64-residue examples with independent block charges."""
    return (
        "K" * first_positive
        + "A" * (32 - first_positive)
        + "E" * second_negative
        + "A" * (32 - second_negative)
    )


class AcceptanceRewardTests(unittest.TestCase):
    def test_thresholds_and_plateau_including_disorder_cutoff(self):
        sequences = [tail(2, 6), tail(4, 4), tail(2, 6), tail(2, 6)]
        disorder = torch.ones(4, 64, dtype=torch.float64)
        disorder[2, :13] = 0.5  # Only 51 pass the strict > 0.5 test.
        disorder[3, :12] = 0.5  # 52/64 meets the >= 0.8 fraction.
        result = joint_acceptance_reward(sequences, disorder)
        self.assertEqual(result["charge_pass"].tolist(), [True] * 4)
        self.assertEqual(result["joint_pass"].tolist(), [True, True, False, True])
        self.assertEqual(result["log_factor"][[0, 1, 3]].tolist(), [0, 0, 0])
        self.assertLess(result["log_factor"][2], -math.log(20))

    def test_historical_inversion_removed(self):
        # Reconstruct the actual descriptor inversion without using saved data.
        passing = "K" * 3 + "E" + "A" * 28 + "E" * 6 + "A" * 26
        failing = "K" * 5 + "E" * 4 + "A" * 23 + "K" + "E" * 5 + "A" * 26
        old, _, _, success = charge_reward(
            [passing, failing],
            {
                "block_targets": [0.125, -0.125],
                "block_scale": 0.125,
                "fcr_target": 0.25,
                "fcr_scale": 0.25,
                "block_tolerance": 0.0625,
                "fcr_tolerance": 0.125,
            },
        )
        self.assertEqual(success.tolist(), [True, False])
        self.assertGreater(old[1], old[0])
        new = joint_acceptance_reward([passing, failing], torch.ones(2, 64))
        self.assertGreater(new["log_factor"][0], new["log_factor"][1])

    def test_failure_proximity_and_bounded_factor(self):
        sequences = [tail(0, 0), tail(1, 6), tail(2, 6), "K" * 64]
        result = joint_acceptance_reward(sequences, torch.ones(4, 64))
        factors = result["log_factor"].exp()
        self.assertLess(factors[0], factors[1])
        self.assertLess(factors[1], factors[2])
        self.assertTrue((factors >= 1 / 80).all())
        self.assertTrue((factors[~result["joint_pass"]] <= 1 / 20).all())
        self.assertTrue((result["violations"] >= 0).all())

    def test_exact_target_zero_tb_with_beta_one_and_prior_once(self):
        sequences = [tail(2, 6), tail(1, 6), tail(0, 0), tail(4, 4)]
        predicted = torch.ones(4, 64)
        predicted[-1] = 0.0
        score = joint_acceptance_reward(sequences, predicted)["log_factor"]
        prior = torch.tensor([0.1, 0.5, 0.3, 0.1], dtype=torch.float64)
        log_reward = prior.log() + score
        log_z = log_reward.logsumexp(0)
        log_q = log_reward - log_z
        loss = trajectory_balance(log_q, prior.log(), score, log_z, 1.0)
        self.assertLess(loss.item(), 1e-25)
        wrong_beta = trajectory_balance(log_q, prior.log(), score, log_z, 2.0)
        self.assertGreater(wrong_beta.item(), 1)
        # The property factor does not cancel the prior within the passing set.
        self.assertFalse(score.requires_grad)

    def test_invalid_scores_lengths_and_settings_fail_closed(self):
        valid_tail = tail(4, 4)
        for invalid in (
            torch.ones(1, 63),
            torch.full((1, 64), float("nan")),
            torch.full((1, 64), 1.1),
        ):
            with self.assertRaises(ValueError):
                joint_acceptance_reward([valid_tail], invalid)
        with self.assertRaises(ValueError):
            joint_acceptance_reward(["A" * 62], torch.ones(1, 62))
        settings = AcceptanceRewardConfig()
        for changes in (
            {"failure_floor_fraction": 0},
            {"pass_to_near_miss_ratio": 1},
            {"disorder_deficit_scale": 0},
        ):
            with self.assertRaises(ValueError):
                replace(settings, **changes)


if __name__ == "__main__":
    unittest.main()
