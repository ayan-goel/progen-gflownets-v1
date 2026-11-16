"""Reward semantics, matched TB continuation, and self-contained CPU recovery."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_replay import fixture_predictor, fixture_source

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.continuous_data import check_continuous_inputs
from stability_gfn.idr.continuous_evaluation import progression_decision, target_shape
from stability_gfn.idr.continuous_experiment import (
    reevaluate_continuous,
    run_continuous,
)
from stability_gfn.idr.continuous_reward import continuous_reward
from stability_gfn.idr.replay_experiment import run_replay
from stability_gfn.idr.warmstart_experiment import run_warmstart


class ContinuousRewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_acceptance_plateau_and_unfloored_log_penalties(self):
        reward = replace(
            AcceptanceRewardConfig(), tail_length=4, block_targets=(0, 0), fcr_target=0
        )
        tails = ["AAAA", "GGGG", "KAAA", "KKAA"]
        scores = torch.full((4, 4), 0.9, requires_grad=True)
        values = continuous_reward(tails, scores, reward)
        self.assertEqual(values["joint_pass"].tolist(), [True, True, False, False])
        self.assertEqual(values["log_factor"].tolist(), [0, 0, -20, -45])
        self.assertFalse(values["log_factor"].requires_grad)
        self.assertGreater(float(values["current_log_factor"][3]), -5)
        for penalty in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                continuous_reward(tails, scores, reward, penalty)
        disordered_fail = continuous_reward(["AAAA"], [[0.4] * 4], reward)
        self.assertFalse(disordered_fail["joint_pass"][0])
        self.assertEqual(float(disordered_fail["log_factor"][0]), -10)

    def test_decision_requires_improvement_over_start_and_control(self):
        arms = {
            arm: {
                "joint_success_rate": value,
                "joint_success_without_exact_training_copy": value,
                "unique_tail_fraction": 1,
            }
            for arm, value in [
                ("unchanged", 0.15),
                ("current_tb", 0.07),
                ("continuous_tb", 0.14),
            ]
        }
        self.assertFalse(progression_decision(arms)["passes_practical_screen"])
        arms["continuous_tb"].update(
            joint_success_rate=0.23, joint_success_without_exact_training_copy=0.23
        )
        self.assertTrue(progression_decision(arms)["passes_practical_screen"])
        arms["continuous_tb"]["joint_success_without_exact_training_copy"] = 0.16
        self.assertFalse(progression_decision(arms)["passes_practical_screen"])

    def test_cross_target_fit_excludes_union_of_exposed_holdouts(self):
        bank = [
            {"id": str(i), "log_factor": -i, "violations": [i], "split": "holdout"}
            for i in range(3)
        ]
        fits = [
            dict(
                id=str(i),
                arm=arm,
                stage=stage,
                split="holdout",
                prompt_id="a",
                log_policy=-10 - i + (3 if arm == "current_tb" else 0),
                log_prior=-10,
                fresh_training_overlap=arm == "continuous_tb" and i == 2,
            )
            for arm in ("unchanged", "current_tb", "continuous_tb")
            for stage in ("initial", "final")
            for i in range(3)
        ]
        values = target_shape(fits, bank, 2.5)
        for arm in ("unchanged", "current_tb", "continuous_tb"):
            self.assertEqual(
                values[f"{arm}/final/current/holdout"]["mean_prompt_centered_rmse"], 0
            )
            self.assertEqual(
                values[f"{arm}/final/continuous/holdout"]["mean_prompt_centered_rmse"],
                0.75,
            )
            self.assertEqual(
                values[f"{arm}/final/current/holdout"]["excluded_exposed_holdout_rows"],
                1,
            )

    def test_cpu_reward_only_fit_and_snapshot_recovery(self):
        backbone = FrozenProGen.tiny(11)
        frozen = {
            key: value.clone() for key, value in backbone.model.state_dict().items()
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained execution"),
            ),
        ):
            root = Path(directory)
            replay_settings, _, _, _ = fixture_source(root, backbone)
            run_replay(
                replay_settings, root / "replay", "cpu", backbone, fixture_predictor
            )
            warm = dict(
                source_run=str(root / "replay"),
                source_generation_sha256=file_hash(root / "replay/generation.json"),
                training_steps=3,
                evaluation_samples_per_prompt=2,
                training_seed=92000,
                evaluation_seed=93000,
                bank_sampling_seed=95000,
            )
            run_warmstart(warm, root / "warm", "cpu", backbone, fixture_predictor)
            settings = dict(
                warm,
                control_run=str(root / "warm"),
                control_generation_sha256=file_hash(root / "warm/generation.json"),
                evaluation_seed=103000,
                penalty=2.5,
            )
            before = {
                p: file_hash(p)
                for folder in (root / "replay", root / "warm")
                for p in folder.rglob("*")
                if p.is_file()
            }
            check_continuous_inputs(settings, False)
            bad = dict(settings, training_seed=92001)
            with self.assertRaises(ValueError):
                check_continuous_inputs(bad, False)
            with self.assertRaises(ValueError):
                check_continuous_inputs(dict(settings, penalty=3), False)
            run_continuous(settings, root / "run", "cpu", backbone, fixture_predictor)
            for path, digest in before.items():
                self.assertEqual(file_hash(path), digest)
            for key, value in backbone.model.state_dict().items():
                torch.testing.assert_close(value, frozen[key], atol=0, rtol=0)
            self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))
            start = torch.load(
                root / "run/initial.pt", map_location="cpu", weights_only=True
            )
            final = torch.load(
                root / "run/continuous_tb/step-3.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertTrue(
                any(
                    not torch.equal(start["head"][k], v)
                    for k, v in final["head"].items()
                )
            )
            self.assertEqual(final["reward_version"], "continuous_acceptance_v1")
            logs = {}
            for arm in ("current_tb", "continuous_tb"):
                logs[arm] = [
                    json.loads(line)
                    for line in (root / f"run/{arm}/training.jsonl")
                    .read_text()
                    .splitlines()
                ]
            self.assertEqual(
                [r["replay_ids"] for r in logs["current_tb"]],
                [r["replay_ids"] for r in logs["continuous_tb"]],
            )
            self.assertEqual(
                [r["rates"] for r in logs["current_tb"]],
                [r["rates"] for r in logs["continuous_tb"]],
            )
            # Reward differences affect updates, not the identical initial policy.
            fresh = {
                arm: [
                    json.loads(line)
                    for line in (root / f"run/{arm}/fresh_training.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for arm in logs
            }
            self.assertEqual(
                [r["tail"] for r in fresh["current_tb"][:2]],
                [r["tail"] for r in fresh["continuous_tb"][:2]],
            )
            initial_fits = {}
            for arm in ("unchanged", "continuous_tb"):
                rows = json.loads((root / f"run/{arm}/bank_fit.json").read_text())
                initial_fits[arm] = [
                    r["log_policy"] for r in rows if r["stage"] == "initial"
                ]
            self.assertEqual(initial_fits["unchanged"], initial_fits["continuous_tb"])
            with self.assertRaises(FileExistsError):
                run_continuous(
                    settings, root / "run", "cpu", backbone, fixture_predictor
                )
            for folder in ("replay", "warm", "data"):
                (root / folder).rename(root / f"hidden-{folder}")
            reevaluate_continuous(root / "run", root / "recovered", fixture_predictor)
            for name in (
                "metrics.json",
                "decision.json",
                "target_shape.json",
                "training_summary.json",
            ):
                self.assertEqual(
                    json.loads((root / "run" / name).read_text()),
                    json.loads((root / "recovered" / name).read_text()),
                )
            with (root / "run/samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_continuous(root / "run", root / "corrupt", fixture_predictor)
