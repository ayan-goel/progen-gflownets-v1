"""Tiny CPU execution, control integrity, scorer inputs, and evaluation recovery."""

import copy
import itertools
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_idr_branch_rate import fixture_predictor
from test_idr_charge import configuration

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.acceptance_evaluation import (
    prior_target_estimates,
    summarize_acceptance,
)
from stability_gfn.idr.acceptance_experiment import (
    check_acceptance_inputs,
    reevaluate_acceptance,
    run_acceptance,
)
from stability_gfn.idr.acceptance_reward import (
    AcceptanceRewardConfig,
    joint_acceptance_reward,
)
from stability_gfn.idr.charge import charge_reward
from stability_gfn.idr.charge_experiment import run_charge
from stability_gfn.idr.context_experiment import run_context
from stability_gfn.idr.subtb_experiment import run_subtb


class AcceptanceExperimentTests(unittest.TestCase):
    def test_small_frozen_run_and_evaluation_recovery(self):
        torch.set_num_threads(1)
        backbone = FrozenProGen.tiny(11)
        before = {k: v.clone() for k, v in backbone.model.state_dict().items()}
        references = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKQEKAG"},
            {"id": "fixture2", "role": "prompt", "sequence": "AQPSEKGS"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            write_json(data / "references.json", references)
            write_json(
                data / "manifest.json",
                {
                    "count": 2,
                    "prompt_count": 2,
                    "references_sha256": file_hash(data / "references.json"),
                },
            )
            config = configuration()
            config.update(
                data=str(data), manifest_sha256=file_hash(data / "manifest.json")
            )
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained execution"),
            ):
                run_charge(config, root / "pilot", "cpu", backbone, fixture_predictor)
                run_context(
                    {
                        "source_run": str(root / "pilot"),
                        "source_config_sha256": file_hash(root / "pilot/config.json"),
                        "arms": ["standard", "direct_context"],
                    },
                    root / "context",
                    "cpu",
                    backbone,
                    fixture_predictor,
                )
                run_subtb(
                    {
                        "source_run": str(root / "context"),
                        "source_files": {
                            name: file_hash(root / "context" / name)
                            for name in (
                                "generation.json",
                                "standard/metrics.json",
                                "direct_context/metrics.json",
                            )
                        },
                        "branch_rate_multiplier": 10,
                        "training_steps": 3,
                        "evaluation_samples_per_prompt": 2,
                        "subtb_lambda": 1.0,
                        "flow_learning_rate": 0.001,
                    },
                    root / "control",
                    "cpu",
                    backbone,
                    fixture_predictor,
                )
                original_hashes = {
                    p: file_hash(p)
                    for p in (root / "control").rglob("*")
                    if p.is_file()
                }
                reward = replace(AcceptanceRewardConfig(), tail_length=4)
                settings = {
                    "source_run": str(root / "control"),
                    "source_generation_sha256": file_hash(
                        root / "control/generation.json"
                    ),
                    "reward_version": "joint_acceptance_v1",
                    "reward": asdict(reward),
                }
                check_acceptance_inputs(settings, False)
                bad = copy.deepcopy(settings)
                bad["reward"]["pass_to_near_miss_ratio"] = 200
                with self.assertRaises(ValueError):
                    check_acceptance_inputs(bad, False)
                with self.assertRaises(ValueError):
                    check_acceptance_inputs(settings, True)
                calls = []

                def tracked_predictor(sequences, version):
                    calls.append((list(sequences), version))
                    return fixture_predictor(sequences, version)

                run_acceptance(
                    settings, root / "new", "cpu", backbone, tracked_predictor
                )
                for sequences, version in calls[:3]:
                    self.assertEqual(version, "V3")
                    self.assertEqual([len(s) for s in sequences], [4, 4])
                for path, digest in original_hashes.items():
                    self.assertEqual(file_hash(path), digest)
                with self.assertRaises(FileExistsError):
                    run_acceptance(
                        settings, root / "new", "cpu", backbone, fixture_predictor
                    )
                # CPU evaluation must not need the original controls or old data.
                (root / "control").rename(root / "hidden-control")
                reevaluate_acceptance(root / "new", root / "replay", fixture_predictor)
            for name in (
                "metrics.json",
                "decision.json",
                "target_estimates.json",
                "training_diagnostics.json",
                "sequence_metrics.json",
            ):
                self.assertEqual(
                    json.loads((root / "new" / name).read_text()),
                    json.loads((root / "replay" / name).read_text()),
                )
            self.assertEqual(
                (root / "new/report.md").read_text(),
                (root / "replay/report.md").read_text(),
            )
            raw = [
                json.loads(line)
                for line in (root / "new/samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(raw), 16)
            self.assertEqual(len({r["id"] for r in raw}), 16)
            logs = [
                json.loads(line)
                for line in (root / "new/seed-0/training.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([r["step"] for r in logs], [1, 2, 3])
            self.assertAlmostEqual(logs[0]["rates"]["task_branch"], 1e-3)
            self.assertAlmostEqual(logs[-1]["rates"]["task_branch"], 1e-4)
            checkpoint = torch.load(
                root / "new/seed-0/step-3.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["config"]["beta"], 1)
            self.assertGreater(checkpoint["head"]["task_weight"].norm().item(), 0)
            costs = json.loads((root / "new/costs.json").read_text())["0"]
            self.assertEqual(costs["new_tb_training"]["trajectories"], 6)
            self.assertEqual(costs["new_tb_training"]["v3_tail_predictions"], 6)
            self.assertEqual(costs["old_tb_training"]["trajectories"], 0)
            inputs = [
                json.loads(line)
                for line in (root / "new/seed-0/reward_inputs.jsonl")
                .read_text()
                .splitlines()
            ]
            recomputed = joint_acceptance_reward(
                [r["tail"] for r in inputs],
                [r["isolated_tail_v3_scores"] for r in inputs],
                reward,
            )
            np.testing.assert_array_equal(
                recomputed["log_factor"].numpy(), [r["log_factor"] for r in inputs]
            )
            first_scores = charge_reward(
                [r["tail"] for r in inputs[:2]], config["reward"]
            )[0]
            old_log = json.loads(
                (root / "hidden-control/tb/seed-0/training.jsonl")
                .read_text()
                .splitlines()[0]
            )
            self.assertAlmostEqual(float(first_scores.mean()), old_log["mean_score"])
            scored = json.loads((root / "new/sequence_metrics.json").read_text())
            for row in scored:
                if row["arm"] in ("new_tb", "old_tb"):
                    factor = (
                        row["acceptance"]["log_factor"]
                        if row["arm"] == "new_tb"
                        else 2 * row["old_charge_score"]
                    )
                    expected = (
                        row["log_z"] + row["log_policy"] - row["log_prior"] - factor
                    )
                    self.assertEqual(row["own_reward_tb_residual"], expected)
            for arm in ("prior", "select"):
                self.assertEqual(sum(r["arm"] == arm for r in scored), 4)
            self.assertTrue(
                all("log_policy" not in r for r in scored if r["arm"] == "select")
            )
            # Check efficient exact Hamming calculation against explicit pairs.
            group = [r for r in scored if r["arm"] == "prior_pool"]
            tails = [r["sequence"][4:] for r in group]
            expected = np.mean(
                [
                    sum(a != b for a, b in zip(left, right)) / 4
                    for left, right in itertools.combinations(tails, 2)
                ]
            )
            self.assertAlmostEqual(
                summarize_acceptance(group)["mean_pairwise_tail_hamming_fraction"],
                expected,
            )
            (root / "new/samples.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_acceptance(root / "new", root / "bad", fixture_predictor)
        for key, value in backbone.model.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))
        self.assertFalse(torch.cuda.is_initialized())

    def test_importance_weights_do_not_include_prior_twice_and_keep_empty_success(self):
        rows = []
        for prompt, passed in (("a", True), ("b", False)):
            for index, factor in enumerate((1, 0.05)):
                rows.append(
                    {
                        "seed": 0,
                        "prompt_id": prompt,
                        "log_prior": -1000 * index,
                        "old_charge_score": np.log(factor) / 2,
                        "acceptance": {
                            "log_factor": np.log(factor),
                            "joint_pass": passed and index == 0,
                            "charge_pass": index == 0,
                            "disorder_pass": passed,
                        },
                    }
                )
        results = prior_target_estimates(rows, 2)
        self.assertEqual(results["old_target"], results["new_target"])
        self.assertAlmostEqual(
            results["new_target"]["summary"]["joint_pass"], (1 / 1.05) / 2
        )
        self.assertEqual(
            results["new_target"]["summary"]["groups_without_observed_joint_pass"], 1
        )


if __name__ == "__main__":
    unittest.main()
