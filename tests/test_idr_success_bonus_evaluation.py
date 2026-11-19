"""Saved-data CPU checks for three-repeat success-bonus evaluation and its gate."""

import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from test_idr_objective_evaluation import evaluation_fixture, read_jsonl, write_jsonl
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import write_json
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.success_bonus_evaluation import (
    evaluate_success_bonus,
    success_bonus_decision,
)
from stability_gfn.idr.success_bonus_reward import REWARD_VERSION, success_bonus_reward


def success_bonus_evaluation_fixture(root):
    """Extend the historical synthetic fixture, preserving its source helper."""
    output = evaluation_fixture(root)
    settings = json.loads((root / "config.json").read_text())
    settings["fit"]["pass_multiplier"] = 9.0
    settings["replicates"].append(
        dict(
            name="replicate-3",
            training_seed=3,
            evaluation_seed=13,
            bank_sampling_seed=23,
        )
    )
    write_json(root / "config.json", settings)
    definition = dict(version=REWARD_VERSION, penalty=2.5, pass_multiplier=9.0)
    write_json(root / "reward_definition.json", definition)
    shutil.copytree(root / "replicate-1", root / "replicate-3")
    for path in (root / "replicate-3").rglob("*.jsonl"):
        path.write_text(path.read_text().replace("replicate-1", "replicate-3"))
    # A first-repeat output appears in training only during the third repeat.
    # The common copy union must include it for every arm and earlier repeat.
    path = root / "replicate-3/training_rows.jsonl"
    fresh = read_jsonl(path)
    for row in fresh:
        if row["generator_arm"] == "gfn":
            row["tail"] = "TTTTTT"
            row["sequence"] = row["sequence"][:4] + row["tail"]
    write_jsonl(path, fresh)
    reward = AcceptanceRewardConfig(**json.loads((root / "reward.json").read_text()))
    for path in [root / "bank.json", *root.glob("replicate-*/training_rows.jsonl")]:
        rows = (
            json.loads(path.read_text()) if path.suffix == ".json" else read_jsonl(path)
        )
        values = success_bonus_reward(
            [row["tail"] for row in rows], [[0.9] * 6 for row in rows], reward
        )
        for index, row in enumerate(rows):
            row.update(
                isolated_tail_v3_scores=[0.9] * 6,
                reward_version=REWARD_VERSION,
                joint_pass=bool(values["joint_pass"][index]),
                **{
                    key: float(values[key][index])
                    for key in ("base_log_factor", "success_log_bonus", "log_factor")
                },
            )
        if path.suffix == ".json":
            write_json(path, rows)
        else:
            write_jsonl(path, rows)
    all_samples = []
    for replicate in settings["replicates"]:
        folder = root / replicate["name"]
        write_json(
            folder / "config.json",
            {
                **settings["fit"],
                **{key: value for key, value in replicate.items() if key != "name"},
            },
        )
        path = folder / "supervised/training.jsonl"
        logs = read_jsonl(path)
        for row in logs:
            for prompt in row["weighting"]["prompts"]:
                prompt.update(
                    minimum_log_factor=math.log(9), maximum_log_factor=math.log(9)
                )
        write_jsonl(path, logs)
        for arm in ("gfn", "supervised"):
            torch.save(
                dict(
                    head={"weight": torch.ones(2, 2)},
                    last_block={"weight": torch.ones(2, 2)},
                    log_z=torch.zeros(2) if arm == "gfn" else None,
                    prompt_ids=["a", "b"],
                    architecture="lastblock_direct_context",
                    arm=arm,
                    step=2,
                    objective="trajectory_balance"
                    if arm == "gfn"
                    else "weighted_token_nll",
                    reward_version=REWARD_VERSION,
                    reward_definition=definition,
                ),
                folder / arm / "checkpoint.pt",
            )
        all_samples.extend(read_jsonl(folder / "samples.jsonl"))
    write_jsonl(root / "samples.jsonl", all_samples)
    return output


class SuccessBonusEvaluationTests(unittest.TestCase):
    def test_three_repeat_copy_union_reward_residual_and_cached_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = success_bonus_evaluation_fixture(root)
            before = {
                path: path.read_bytes() for path in root.rglob("*") if path.is_file()
            }
            report = evaluate_success_bonus(root, output, fixture_predictor)
            self.assertTrue(report.exists())
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(len(metrics["by_replicate"]), 3)
            self.assertEqual(metrics["arms"]["gfn"]["count"], 24)
            first = metrics["by_replicate"]["replicate-1"]["gfn"]
            self.assertEqual(first["joint_success_rate"], 0.75)
            self.assertEqual(first["joint_success_without_common_training_copy"], 0)
            self.assertEqual(first["exact_common_training_copy_fraction"], 0.75)
            self.assertEqual(first["exact_own_training_copy_fraction"], 0.25)
            records = json.loads((output / "sequence_metrics.json").read_text())
            passing = next(
                row
                for row in records
                if row["arm"] == "gfn" and row["acceptance"]["joint_pass"]
            )
            self.assertAlmostEqual(passing["acceptance"]["base_log_factor"], 0)
            self.assertAlmostEqual(
                passing["acceptance"]["success_log_bonus"], math.log(9)
            )
            self.assertAlmostEqual(passing["own_reward_tb_residual"], 1 - math.log(9))
            training = json.loads((output / "training_summary.json").read_text())
            self.assertIn("ALL THREE", training["copy_definition"])
            self.assertEqual(training["reward_definition"]["pass_multiplier"], 9)
            self.assertEqual(len(training["replicates"]), 3)
            cache = {"V3": {}, "V2": {}}
            for row in records:
                for version, values in row["disorder"].items():
                    cache[version][row["sequence"]] = values["full_scores"]
                    cache[version][row["tail"]] = values["isolated_tail_scores"]

            def cached_predictor(sequences, version):
                return [np.asarray(cache[version][sequence]) for sequence in sequences]

            recovered = root / "recovered"
            evaluate_success_bonus(root, recovered, cached_predictor)
            for filename in (
                "metrics.json",
                "sequence_metrics.json",
                "decision.json",
                "training_summary.json",
                "report.md",
            ):
                self.assertEqual(
                    (output / filename).read_bytes(),
                    (recovered / filename).read_bytes(),
                )
            self.assertTrue(
                all(path.read_bytes() == value for path, value in before.items())
            )

    def test_screen_requires_each_repeat_raw_gain_distinct_yield_and_quality(self):
        arms = {
            arm: dict(
                count=100,
                joint_success_rate=rate,
                joint_success_without_common_training_copy=0.2,
                distinct_noncopy_successful_tail_count=count,
                distinct_noncopy_successful_tail_fraction=count / 100,
                unique_tail_fraction=0.9,
                v3_disorder_success_rate=0.9,
            )
            for arm, rate, count in (
                ("gfn", 0.5, 20),
                ("supervised", 0.4, 19),
                ("prior", 0.1, 10),
            )
        }
        repeats = {f"replicate-{index}": copy.deepcopy(arms) for index in (1, 2, 3)}
        # No additional ten-point noncopy-rate requirement was declared.
        self.assertTrue(success_bonus_decision(repeats)["passes_practical_screen"])
        for metric, value in (
            ("joint_success_rate", 0.499),
            ("distinct_noncopy_successful_tail_count", 19),
            ("unique_tail_fraction", 0.849),
            ("v3_disorder_success_rate", 0.849),
        ):
            with self.subTest(metric=metric):
                changed = copy.deepcopy(repeats)
                changed["replicate-3"]["gfn"][metric] = value
                decision = success_bonus_decision(changed)
                self.assertFalse(decision["passes_practical_screen"])
                self.assertTrue(decision["replicates"]["replicate-1"]["passes"])
        with self.assertRaises(ValueError):
            success_bonus_decision(
                {key: value for key, value in repeats.items() if key != "replicate-3"}
            )
        repeats["replicate-3"]["supervised"]["count"] = 99
        with self.assertRaises(ValueError):
            success_bonus_decision(repeats)

    def test_rejects_wrong_bonus_bank_or_fresh_labels_and_repeat_count(self):
        corruptions = (
            "definition",
            "fit",
            "replicates",
            "disorder_threshold",
            "charge_threshold",
            "bank_factor",
            "bank_bonus",
            "bank_label",
            "fresh_base",
            "fresh_factor",
            "fresh_bonus",
            "fresh_label",
            "fresh_version",
            "bank_missing_scores",
            "weighting",
            "checkpoint_version",
            "checkpoint_bonus",
            "checkpoint_step",
            "supervised_normalizer",
        )
        for corruption in corruptions:
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = success_bonus_evaluation_fixture(root)
                folder = root / "replicate-1"
                if corruption == "definition":
                    path = root / "reward_definition.json"
                    value = json.loads(path.read_text())
                    value["pass_multiplier"] = 8
                    write_json(path, value)
                elif corruption in ("fit", "replicates"):
                    path = root / "config.json"
                    value = json.loads(path.read_text())
                    if corruption == "fit":
                        value["fit"]["pass_multiplier"] = 8
                    else:
                        value["replicates"].pop()
                    write_json(path, value)
                elif corruption.endswith("threshold"):
                    path = root / "pilot_config.json"
                    value = json.loads(path.read_text())
                    if corruption == "disorder_threshold":
                        value["minimum_disordered_fraction"] = 0.7
                    else:
                        value["reward"]["block_tolerance"] = 0.1
                    write_json(path, value)
                elif (
                    corruption.startswith("checkpoint")
                    or corruption == "supervised_normalizer"
                ):
                    arm = (
                        "supervised" if corruption == "supervised_normalizer" else "gfn"
                    )
                    path = folder / arm / "checkpoint.pt"
                    value = torch.load(path, map_location="cpu", weights_only=True)
                    if corruption == "checkpoint_version":
                        value["reward_version"] = "continuous_acceptance_v1"
                    elif corruption == "checkpoint_bonus":
                        value["reward_definition"]["pass_multiplier"] = 1
                    elif corruption == "checkpoint_step":
                        value["step"] = 1
                    else:
                        value["log_z"] = torch.zeros(2)
                    torch.save(value, path)
                elif corruption == "weighting":
                    path = folder / "supervised/training.jsonl"
                    rows = read_jsonl(path)
                    rows[0]["weighting"]["prompts"][0]["maximum_log_factor"] = 0
                    write_jsonl(path, rows)
                else:
                    bank = corruption.startswith("bank")
                    path = (
                        root / "bank.json" if bank else folder / "training_rows.jsonl"
                    )
                    rows = json.loads(path.read_text()) if bank else read_jsonl(path)
                    if corruption.endswith("label"):
                        rows[0]["joint_pass"] = not rows[0]["joint_pass"]
                    elif corruption.endswith("version"):
                        rows[0]["reward_version"] = "continuous_acceptance_v1"
                    elif corruption == "bank_missing_scores":
                        rows[0].pop("isolated_tail_v3_scores")
                    else:
                        field = (
                            "base_log_factor"
                            if corruption.endswith("base")
                            else "success_log_bonus"
                            if corruption.endswith("bonus")
                            else "log_factor"
                        )
                        rows[0][field] -= 0.5
                    if bank:
                        write_json(path, rows)
                    else:
                        write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    evaluate_success_bonus(root, output, fixture_predictor)

    def test_rejects_shared_data_and_evaluation_corruption(self):
        for corruption in (
            "heldout_replay",
            "unpaired",
            "evaluation_count",
            "evaluation_copy",
            "reference_drift",
            "gfn_normalizer",
        ):
            with (
                self.subTest(corruption=corruption),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = success_bonus_evaluation_fixture(root)
                folder = root / "replicate-3"
                path = folder / "samples.jsonl"
                if corruption == "heldout_replay":
                    path = folder / "shared_training.jsonl"
                elif corruption == "unpaired":
                    path = folder / "supervised/training.jsonl"
                rows = read_jsonl(path)
                if corruption == "heldout_replay":
                    rows[0]["example_ids"][-1] = "b-KKKKKK"
                elif corruption == "unpaired":
                    rows[0]["example_ids"].reverse()
                elif corruption == "evaluation_count":
                    rows.pop()
                elif corruption == "evaluation_copy":
                    rows[1]["id"] = rows[0]["id"]
                elif corruption == "gfn_normalizer":
                    next(row for row in rows if row["arm"] == "gfn")["log_z"] = 1
                else:
                    next(row for row in rows if row["arm"] == "prior")[
                        "log_policy"
                    ] += 1
                write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    evaluate_success_bonus(root, output, fixture_predictor)


if __name__ == "__main__":
    unittest.main()
