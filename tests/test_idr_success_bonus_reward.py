"""Small CPU checks of the bonus target and identical fresh/replay weights."""

import copy
import math
import unittest
from unittest.mock import patch

import torch

from stability_gfn.idr.continuous_reward import continuous_reward
from stability_gfn.idr.success_bonus_reward import (
    REWARD_VERSION,
    apply_success_bonus_rows,
    success_bonus_reward,
    validate_success_bonus_rows,
)
from stability_gfn.idr.success_bonus_sampling import refresh_success_bonus_pool


def tail(first_positive, second_negative):
    return (
        "K" * first_positive
        + "A" * (32 - first_positive)
        + "E" * second_negative
        + "A" * (32 - second_negative)
    )


def continuous_rows():
    sequences = [tail(2, 6), tail(1, 6), tail(0, 0), tail(4, 4)]
    scores = torch.ones(4, 64, dtype=torch.float64)
    scores[-1, :13] = 0.5
    values = continuous_reward(sequences, scores, None)
    return [
        dict(
            id=f"path-{index}",
            tail=sequence,
            split="train",
            joint_pass=bool(values["joint_pass"][index]),
            violations=values["violations"][index].tolist(),
            log_factor=float(values["log_factor"][index]),
            isolated_tail_v3_scores=scores[index].tolist(),
        )
        for index, sequence in enumerate(sequences)
    ]


class SuccessBonusRewardTests(unittest.TestCase):
    def test_success_odds_identity_and_failure_ratios_preserved(self):
        rows = continuous_rows()
        scores = torch.tensor(
            [row["isolated_tail_v3_scores"] for row in rows], requires_grad=True
        )
        values = success_bonus_reward([row["tail"] for row in rows], scores)
        prior = torch.tensor([0.1, 0.5, 0.3, 0.1], dtype=torch.float64)
        passed = values["joint_pass"]
        old_target = (prior.log() + values["base_log_factor"]).softmax(0)
        new_target = (prior.log() + values["log_factor"]).softmax(0)
        old_success = old_target[passed].sum()
        expected = 9 * old_success / (1 - old_success + 9 * old_success)
        self.assertAlmostEqual(
            float(new_target[passed].sum()), float(expected), places=14
        )
        self.assertEqual(passed.tolist(), [True, False, False, False])
        torch.testing.assert_close(
            values["log_factor"][~passed],
            values["base_log_factor"][~passed],
            atol=0,
            rtol=0,
        )
        self.assertAlmostEqual(float(values["log_factor"][0]), math.log(9))
        self.assertGreater(
            float(values["log_factor"][1]), float(values["log_factor"][2])
        )
        self.assertTrue(all(not value.requires_grad for value in values.values()))
        self.assertIsNone(scores.grad)

    def test_inclusive_acceptance_and_strict_per_residue_disorder_cutoff(self):
        sequences = [tail(2, 6)] * 3
        scores = torch.ones(3, 64)
        scores[0, :12] = 0.5
        scores[1, :13] = 0.5
        scores[2] = 0.5
        values = success_bonus_reward(sequences, scores)
        self.assertEqual(values["joint_pass"].tolist(), [True, False, False])
        self.assertEqual(values["success_log_bonus"].tolist(), [math.log(9), 0, 0])
        for multiplier in (True, None, "9", 0, 1, -1, float("nan"), float("inf")):
            with self.subTest(multiplier=multiplier), self.assertRaises(ValueError):
                success_bonus_reward(sequences, scores, pass_multiplier=multiplier)
        for scores_bad in (torch.ones(3, 63), torch.full((3, 64), float("nan"))):
            with self.assertRaises(ValueError):
                success_bonus_reward(sequences, scores_bad)

    def test_row_conversion_clones_sources_and_rejects_corrupt_or_double_bonus(self):
        rows = continuous_rows()
        old = copy.deepcopy(rows)
        converted = apply_success_bonus_rows(rows)
        validate_success_bonus_rows(converted)
        direct = success_bonus_reward(
            [row["tail"] for row in rows],
            [row["isolated_tail_v3_scores"] for row in rows],
        )
        for index, row in enumerate(converted):
            self.assertEqual(row["reward_version"], REWARD_VERSION)
            self.assertEqual(row["log_factor"], float(direct["log_factor"][index]))
        converted[0]["isolated_tail_v3_scores"][0] = 0
        self.assertEqual(rows, old)
        with self.assertRaisesRegex(ValueError, "already versioned"):
            apply_success_bonus_rows(converted)
        for override in (
            dict(joint_pass=False),
            dict(joint_pass=1),
            dict(log_factor=math.log(9)),
            dict(log_factor=float("nan")),
            dict(violations=[0, 0, -1, 0]),
            dict(violations=[0, 0, 0]),
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                apply_success_bonus_rows([dict(rows[0], **override)])
        for override in (
            dict(log_factor=2 * math.log(9)),
            dict(success_log_bonus=0),
            dict(reward_version="continuous_acceptance_v1"),
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_success_bonus_rows([dict(converted[0], **override)])

    def test_refresh_adds_bonus_only_to_fresh_paths(self):
        fresh = continuous_rows()
        replay = apply_success_bonus_rows(fresh[:1])
        old_fresh, old_replay = copy.deepcopy(fresh), copy.deepcopy(replay)
        settings = dict(pass_multiplier=9.0, penalty=2.5)
        with patch(
            "stability_gfn.idr.success_bonus_sampling.refresh_objective_pool",
            return_value=(fresh + replay, fresh, dict(scoring_seconds=0.1)),
        ):
            pool, refreshed, costs = refresh_success_bonus_pool(
                None,
                None,
                None,
                None,
                None,
                settings,
                None,
                replay,
                None,
                None,
                1,
                "replicate-1",
                None,
            )
        self.assertEqual(pool[-1], old_replay[0])
        self.assertEqual(pool[-1]["log_factor"], math.log(9))
        self.assertEqual(refreshed[0]["log_factor"], math.log(9))
        self.assertEqual(fresh, old_fresh)
        self.assertEqual(replay, old_replay)
        self.assertGreater(costs["scoring_seconds"], 0.1)
        validate_success_bonus_rows(pool)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
