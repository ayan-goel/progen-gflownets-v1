"""CPU-only synthetic checks of matched data, holdout isolation and raw endpoints."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from test_idr_charge import configuration
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import write_json
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.continuous_reward import continuous_reward
from stability_gfn.idr.lastblock_evaluation import (
    ARMS,
    evaluate_lastblock,
    progression_decision,
)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def evaluation_fixture(root):
    """No pretrained execution: probabilities are explicitly synthetic scalars."""
    config = configuration()
    config.update(steps=2, samples_per_prompt=2)
    config["reward"] = dict(config["reward"], block_targets=[0, 0], fcr_target=0)
    settings = dict(
        penalty=2.5,
        block_gradient_clip=1.0,
        training_steps=2,
        refresh_every=2,
        shared_batch={"frozen_tb": 2, "lastblock_tb": 2, "prior": 2, "replay": 2},
    )
    reward = replace(
        AcceptanceRewardConfig(), tail_length=4, block_targets=[0, 0], fcr_target=0
    )
    prompts = [
        dict(id="a", role="prompt", sequence="GSKQAAAA"),
        dict(id="b", role="prompt", sequence="AQPSGGGG"),
    ]
    write_json(root / "config.json", settings)
    write_json(root / "pilot_config.json", config)
    write_json(root / "reward.json", asdict(reward))
    write_json(root / "references.json", prompts)
    bank = []
    for prompt_index, prompt in enumerate(prompts):
        for tail, split in (
            ("AAAA", "train"),
            ("KKKK", "holdout"),
            ("DDDD", "holdout"),
        ):
            values = continuous_reward([tail], [[0.9] * 4], reward, 2.5)
            bank.append(
                dict(
                    id=f"{prompt['id']}-{tail}",
                    prompt_id=prompt["id"],
                    prompt_index=prompt_index,
                    tail=tail,
                    sequence=prompt["sequence"][:4] + tail,
                    split=split,
                    joint_pass=bool(values["joint_pass"][0]),
                    log_factor=float(values["log_factor"][0]),
                    log_prior=-10.0,
                )
            )
    write_json(root / "bank.json", bank)
    inherited = [dict(step=1, replay_ids=["a-AAAA", "b-AAAA"])]
    write_jsonl(root / "initial_training.jsonl", inherited)
    write_jsonl(root / "supervised_training.jsonl", inherited)
    fresh = []
    for source, tail in (
        ("frozen_tb", "GGGG"),
        ("lastblock_tb", "KKKK"),
        ("prior", "SSSS"),
    ):
        for prompt_index, prompt in enumerate(prompts):
            values = continuous_reward([tail], [[0.9] * 4], reward, 2.5)
            fresh.append(
                dict(
                    id=f"fresh-{source}-{prompt['id']}",
                    generator_arm=source,
                    refresh_round=0,
                    prompt_id=prompt["id"],
                    prompt_index=prompt_index,
                    tail=tail,
                    sequence=prompt["sequence"][:4] + tail,
                    isolated_tail_v3_scores=[0.9] * 4,
                    log_factor=float(values["log_factor"][0]),
                    log_prior=-10.0,
                )
            )
    write_jsonl(root / "training_rows.jsonl", fresh)
    shared = [
        dict(
            step=step,
            refresh_round=0,
            example_ids=[row["id"] for row in fresh] + ["a-AAAA", "b-AAAA"],
            counts=settings["shared_batch"],
        )
        for step in (1, 2)
    ]
    write_jsonl(root / "shared_training.jsonl", shared)
    for arm in ARMS:
        folder = root / arm
        folder.mkdir()
        fits = []
        for row in bank:
            fits.append(
                dict(
                    row,
                    stage="final",
                    log_policy=-10.5,
                    log_z=None if arm in ARMS[:2] else 0.0,
                    tb_residual=None if arm in ARMS[:2] else -0.5 - row["log_factor"],
                    fresh_training_overlap=row["tail"] == "KKKK",
                )
            )
        write_json(folder / "bank_fit.json", fits)
        if arm in ARMS[2:]:
            write_jsonl(
                folder / "training.jsonl",
                [
                    dict(
                        row,
                        rates={"head": 0.0001, "direct": 0.001, "normalizer": 0.01},
                        loss=0.1,
                    )
                    for row in shared
                ],
            )
    records = []
    for arm in ARMS:
        for prompt in prompts:
            for index, tail in enumerate(("GGGG", "TTTT")):
                prefix = prompt["sequence"][:4]
                records.append(
                    dict(
                        id=f"{arm}-{prompt['id']}-{index}",
                        arm=arm,
                        seed=0,
                        prompt_id=prompt["id"],
                        prefix=prefix,
                        sequence=prefix + tail,
                        evaluation_start=4,
                        log_policy=-9.0,
                        log_prior=-10.0,
                        log_z=None if arm in ARMS[:2] else 0.0,
                    )
                )
    write_jsonl(root / "samples.jsonl", records)
    output = root / "evaluation"
    output.mkdir()
    return output


class LastblockEvaluationTests(unittest.TestCase):
    def test_screen_checks_disorder_diversity_and_supervised_control(self):
        arms = {
            arm: dict(
                joint_success_rate=rate,
                joint_success_without_common_training_copy=rate,
                unique_tail_fraction=1.0,
                v3_disorder_success_rate=0.8,
            )
            for arm, rate in zip(ARMS, (0.1, 0.24, 0.13, 0.25))
        }
        self.assertTrue(progression_decision(arms)["passes_progression_screen"])
        for metric, value in (
            ("joint_success_without_common_training_copy", 0.15),
            ("unique_tail_fraction", 0.9),
            ("v3_disorder_success_rate", 0.7),
        ):
            changed = copy.deepcopy(arms)
            changed["lastblock_tb"][metric] = value
            self.assertFalse(progression_decision(changed)["passes_primary_screen"])
        arms["continued_supervised"]["joint_success_rate"] = 0.28
        decision = progression_decision(arms)
        self.assertTrue(decision["passes_primary_screen"])
        self.assertFalse(decision["passes_progression_screen"])
        for metric in ("unique_tail_fraction", "v3_disorder_success_rate"):
            changed = copy.deepcopy(arms)
            changed["continued_supervised"]["joint_success_rate"] = 0.24
            changed["unchanged"][metric] = 0.7
            changed["frozen_tb"][metric] = 0.7
            changed["lastblock_tb"][metric] = 0.7
            changed["continued_supervised"][metric] = 0.8
            decision = progression_decision(changed)
            self.assertTrue(decision["passes_primary_screen"])
            self.assertFalse(decision["passes_progression_screen"])

    def test_evaluation_common_copy_denominator_and_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = evaluation_fixture(root)
            report = evaluate_lastblock(root, output, fixture_predictor)
            self.assertTrue(report.exists())
            metrics = json.loads((output / "metrics.json").read_text())
            for arm in ARMS:
                row = metrics["arms"][arm]
                self.assertEqual(row["joint_success_rate"], 1.0)
                self.assertEqual(row["joint_success_without_common_training_copy"], 0.5)
                self.assertEqual(row["mean_log_policy_minus_frozen_log_prior"], 1.0)
                self.assertEqual(row["exact_common_training_copy_fraction"], 0.5)
            self.assertEqual(
                metrics["arms"]["unchanged"]["exact_own_training_copy_fraction"], 0
            )
            self.assertEqual(
                metrics["arms"]["lastblock_tb"]["exact_own_training_copy_fraction"], 0.5
            )
            self.assertNotIn(
                "own_reward_tb_residual_rmse", metrics["arms"]["unchanged"]
            )
            self.assertEqual(
                metrics["arms"]["lastblock_tb"]["own_reward_tb_residual_rmse"], 1.0
            )
            diagnostics = json.loads((output / "bank_diagnostics.json").read_text())
            for arm in ARMS:
                self.assertEqual(diagnostics[f"{arm}/holdout/all"]["count"], 2)
            training = json.loads((output / "training_summary.json").read_text())
            self.assertEqual(training["accidental_holdout_overlap_count"], 2)
            self.assertEqual(training["fresh_trajectories"], 6)
            self.assertEqual(training["examples_per_trained_arm"], 16)

    def test_explicit_holdout_replay_and_unpaired_examples_are_rejected(self):
        for change in ("holdout", "unpaired", "counts", "fit_prior"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                output = evaluation_fixture(root)
                if change == "fit_prior":
                    path = root / "lastblock_tb/bank_fit.json"
                    rows = json.loads(path.read_text())
                    rows[0]["log_prior"] += 0.1
                    write_json(path, rows)
                else:
                    path = root / (
                        "lastblock_tb/training.jsonl"
                        if change == "unpaired"
                        else "shared_training.jsonl"
                    )
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    if change == "holdout":
                        rows[0]["example_ids"][-1] = "b-DDDD"
                    elif change == "unpaired":
                        rows[0]["example_ids"].reverse()
                    else:
                        rows[0]["counts"]["prior"] = 5
                    write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    evaluate_lastblock(root, output, fixture_predictor)


if __name__ == "__main__":
    unittest.main()
