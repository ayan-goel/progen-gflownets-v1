"""Synthetic checks for the bounded 1600-versus-800 endpoint and copy history."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from test_idr_confirmation_evaluation import seal_generation
from test_idr_lastblock_evaluation import evaluation_fixture, write_jsonl
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.idr.budget1600_evaluation import (
    BUDGET1600_ARMS,
    budget1600_decision,
    evaluate_budget1600,
)
from stability_gfn.idr.confirmation_evaluation import evaluate_prior
from stability_gfn.idr.lastblock_evaluation import evaluate_lastblock, read_jsonl


def extend_fixture_training(folder, steps, streams):
    """Expand a two-update arithmetic fixture without fitting a model."""
    settings = json.loads((folder / "config.json").read_text())
    settings.update(training_steps=steps, **streams)
    write_json(folder / "config.json", settings)
    config = json.loads((folder / "pilot_config.json").read_text())
    config["steps"] = steps
    write_json(folder / "pilot_config.json", config)
    original_fresh = read_jsonl(folder / "training_rows.jsonl")
    fresh = []
    for round_index in range(steps // 2):
        fresh.extend(
            dict(row, id=f"{row['id']}-round{round_index}", refresh_round=round_index)
            for row in original_fresh
        )
    write_jsonl(folder / "training_rows.jsonl", fresh)
    for filename in (
        "shared_training.jsonl",
        "frozen_tb/training.jsonl",
        "lastblock_tb/training.jsonl",
    ):
        templates = read_jsonl(folder / filename)
        rows = []
        for step in range(1, steps + 1):
            round_index = (step - 1) // 2
            row = copy.deepcopy(templates[(step - 1) % 2])
            row.update(step=step, refresh_round=round_index)
            row["example_ids"] = [
                f"{identity}-round{round_index}"
                if identity.startswith("fresh-")
                else identity
                for identity in row["example_ids"]
            ]
            rows.append(row)
        write_jsonl(folder / filename, rows)


def evaluation1600_fixture(directory):
    """Eight-update saved controls, sixteen-update candidates and old-only tails."""
    root = directory / "run"
    root.mkdir()
    source = directory / "source-budget"
    source.mkdir()
    plans, combined_raw, source_files = [], [], {}
    for index, name in enumerate(("replicate-1", "replicate-2"), 1):
        old_fit = source / name / "fit"
        old_fit.mkdir(parents=True)
        evaluation_fixture(old_fit)
        fit = root / name / "fit"
        shutil.copytree(old_fit, fit)
        streams = dict(
            training_seed=1000 * index,
            bank_sampling_seed=1000 * index + 1,
            evaluation_seed=1000 * index + 2,
        )
        extend_fixture_training(old_fit, 8, streams)
        (old_fit / "lastblock_tb/checkpoint.pt").write_bytes(b"synthetic checkpoint")
        seal_generation(old_fit)
        source_files[f"{name}/fit/generation.json"] = file_hash(
            old_fit / "generation.json"
        )

        history = read_jsonl(old_fit / "training_rows.jsonl")[:12]
        for row in history:
            if row["tail"] == "SSSS":
                row["tail"] = "TTTT"
                row["sequence"] = row["sequence"][:-4] + "TTTT"
        old_history = source / name / "control400/training_rows.jsonl"
        old_history.parent.mkdir()
        write_jsonl(old_history, history)
        source_files[f"{name}/control400/training_rows.jsonl"] = file_hash(old_history)
        shutil.copyfile(old_history, root / name / "history400_training_rows.jsonl")

        evaluation_seed = 1000 * index + 100
        plans.append(dict(name=name, evaluation_seed=evaluation_seed))
        extend_fixture_training(fit, 16, dict(streams, evaluation_seed=evaluation_seed))
        seal_generation(fit)
        evaluate_lastblock(fit, fit, fixture_predictor)
        prior = root / name / "prior"
        prior.mkdir()
        prior_rows = []
        for row in read_jsonl(fit / "samples.jsonl"):
            if row["arm"] == "unchanged":
                row = dict(
                    row,
                    id="prior-" + row["id"],
                    arm="prior",
                    log_policy=row["log_prior"],
                )
                del row["log_z"]
                prior_rows.append(row)
        write_jsonl(prior / "samples.jsonl", prior_rows)
        write_json(
            prior / "config.json",
            dict(
                arm="prior",
                evaluation_seed=evaluation_seed,
                fit_settings=json.loads((fit / "config.json").read_text()),
            ),
        )
        seal_generation(prior)
        evaluate_prior(fit, prior, predictor=fixture_predictor)
        control = root / name / "control800"
        control.mkdir()
        for old, new in (
            ("config.json", "source_settings.json"),
            ("generation.json", "source_generation.json"),
            ("bank.json", "bank.json"),
            ("training_rows.jsonl", "training_rows.jsonl"),
            ("lastblock_tb/checkpoint.pt", "checkpoint.pt"),
        ):
            shutil.copyfile(old_fit / old, control / new)
        write_json(
            control / "config.json",
            dict(
                arm="lastblock_800",
                evaluation_seed=evaluation_seed,
                source_checkpoint_sha256=file_hash(control / "checkpoint.pt"),
            ),
        )
        control_rows = [
            dict(row, arm="lastblock_800", id="old-" + row["id"])
            for row in read_jsonl(old_fit / "samples.jsonl")
            if row["arm"] == "lastblock_tb"
        ]
        write_jsonl(control / "samples.jsonl", control_rows)
        combined_raw.extend(
            dict(row, id=f"{name}/{row['id']}", replicate=name)
            for row in read_jsonl(fit / "samples.jsonl") + control_rows + prior_rows
        )
    write_json(source / "config.json", dict(training_steps=8, replicates=plans))
    source_files["config.json"] = file_hash(source / "config.json")
    write_json(source / "generation.json", dict(complete=True, files=source_files))
    shutil.copyfile(source / "config.json", root / "budget_config.json")
    shutil.copyfile(source / "generation.json", root / "budget_generation.json")
    write_json(
        root / "config.json",
        dict(
            budget_run="source-budget",
            budget_generation_sha256=file_hash(root / "budget_generation.json"),
            training_steps=16,
            replicates=plans,
        ),
    )
    write_json(root / "provenance.json", dict(packages={}, files={}, fixture=True))
    write_jsonl(root / "samples.jsonl", combined_raw)
    seal_generation(root)
    return root


class Budget1600EvaluationTests(unittest.TestCase):
    def test_historical_only_copies_count_for_all_arms_with_raw_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = evaluation1600_fixture(Path(temporary))
            evaluate_budget1600(root, root, fixture_predictor)
            metrics = json.loads((root / "metrics.json").read_text())
            for replicate in metrics["replicates"].values():
                self.assertEqual(set(replicate["arms"]), set(BUDGET1600_ARMS))
                self.assertEqual(len(replicate["by_prompt"]), 12)
                for row in replicate["arms"].values():
                    self.assertEqual(row["count"], 4)
                    self.assertEqual(row["joint_success_rate"], 1)
                    self.assertEqual(row["exact_common_training_copy_fraction"], 1)
                    self.assertEqual(
                        row["joint_success_without_common_training_copy"], 0
                    )
                    self.assertNotIn("exact_own_training_copy_fraction", row)
            child = json.loads((root / "replicate-1/fit/metrics.json").read_text())
            self.assertEqual(
                child["arms"]["lastblock_tb"][
                    "joint_success_without_common_training_copy"
                ],
                0.5,
            )
            records = json.loads((root / "sequence_metrics.json").read_text())
            self.assertEqual(len(records), 48)
            self.assertEqual(len({row["id"] for row in records}), 48)
            self.assertNotIn("lastblock_400", {row["arm"] for row in records})

    def test_decision_boundaries_require_both_seeds_without_secondary_gate(self):
        arms = {
            arm: dict(
                joint_success_rate=0.1,
                joint_success_without_common_training_copy=0.1,
                unique_tail_fraction=0.95,
                v3_disorder_success_rate=0.90,
            )
            for arm in BUDGET1600_ARMS
        }
        arms["lastblock_tb"].update(
            joint_success_rate=0.35,
            joint_success_without_common_training_copy=0.35,
            unique_tail_fraction=0.90,
            v3_disorder_success_rate=0.85,
        )
        arms["lastblock_800"].update(
            joint_success_rate=0.30, joint_success_without_common_training_copy=0.30
        )
        replicates = {name: dict(arms=copy.deepcopy(arms)) for name in ("one", "two")}
        self.assertTrue(budget1600_decision(replicates)["passes_budget_screen"])
        for metric in (
            "joint_success_rate",
            "joint_success_without_common_training_copy",
            "unique_tail_fraction",
            "v3_disorder_success_rate",
        ):
            changed = copy.deepcopy(replicates)
            changed["one"]["arms"]["lastblock_tb"][metric] -= 0.0001
            decision = budget1600_decision(changed)
            self.assertFalse(decision["passes_budget_screen"])
            self.assertTrue(
                decision["replicates"]["two"]["paired_800_control"]["passes"]
            )
        replicates["one"]["arms"]["continued_supervised"]["joint_success_rate"] = 0.9
        decision = budget1600_decision(replicates)
        self.assertTrue(decision["passes_budget_screen"])
        self.assertFalse(
            decision["replicates"]["one"]["inherited_1600_screen_secondary"][
                "passes_progression_screen"
            ]
        )

    def test_recovery_needs_no_external_ancestor_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = evaluation1600_fixture(Path(temporary))
            evaluate_budget1600(root, root, fixture_predictor)
            shutil.rmtree(Path(temporary) / "source-budget")
            before = {
                str(path): file_hash(path) for path in root.rglob("*") if path.is_file()
            }
            recovered = Path(temporary) / "recovered"
            for name in ("replicate-1", "replicate-2"):
                fit, prior = root / name / "fit", root / name / "prior"
                out_fit, out_prior = (
                    recovered / name / "fit",
                    recovered / name / "prior",
                )
                out_fit.mkdir(parents=True)
                out_prior.mkdir()
                evaluate_lastblock(fit, out_fit, fixture_predictor)
                evaluate_prior(fit, prior, out_prior, fixture_predictor)
            evaluate_budget1600(root, recovered, fixture_predictor)
            for filename in (
                "sequence_metrics.json",
                "metrics.json",
                "decision.json",
                "copy_definition.json",
                "report.md",
            ):
                self.assertEqual(
                    (root / filename).read_bytes(), (recovered / filename).read_bytes()
                )
            self.assertEqual(
                before,
                {
                    str(path): file_hash(path)
                    for path in root.rglob("*")
                    if path.is_file()
                },
            )

    def test_history_hash_and_raw_control_budget_are_checked(self):
        for change, error in (
            ("history", "400-step training history differs"),
            ("count", "counts or identities"),
            ("root", "Root generation does not match"),
        ):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = evaluation1600_fixture(Path(temporary))
                if change == "history":
                    path = root / "replicate-1/history400_training_rows.jsonl"
                    write_jsonl(path, read_jsonl(path)[:-1])
                elif change == "count":
                    path = root / "replicate-1/control800/samples.jsonl"
                    write_jsonl(path, read_jsonl(path)[:-1])
                else:
                    path = root / "samples.jsonl"
                    rows = read_jsonl(path)
                    rows[0]["id"] = rows[1]["id"]
                    write_jsonl(path, rows)
                seal_generation(root)
                with self.assertRaisesRegex(ValueError, error):
                    evaluate_budget1600(root, root, fixture_predictor)


if __name__ == "__main__":
    unittest.main()
