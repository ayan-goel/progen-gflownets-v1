"""Synthetic CPU checks of fresh-control scoring and cross-budget copy controls."""

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
from stability_gfn.idr.budget_evaluation import (
    BUDGET_ARMS,
    budget_decision,
    evaluate_budget,
)
from stability_gfn.idr.confirmation_evaluation import evaluate_prior
from stability_gfn.idr.lastblock_evaluation import evaluate_lastblock, read_jsonl


def budget_fixture(directory):
    """Two-step controls and four-step reruns exercise the doubled-budget logic."""
    root = directory / "run"
    root.mkdir()
    replicates = []
    combined_raw = []
    confirmation_files = {}
    for index, name in enumerate(("replicate-1", "replicate-2"), 1):
        old = directory / f"old-{name}"
        old.mkdir()
        evaluation_fixture(old)
        settings = json.loads((old / "config.json").read_text())
        settings.update(
            training_seed=index * 1000,
            bank_sampling_seed=index * 1000 + 1,
            evaluation_seed=index * 1000 + 2,
        )
        write_json(old / "config.json", settings)
        (old / "lastblock_tb/checkpoint.pt").write_bytes(
            b"synthetic preserved checkpoint"
        )
        fit = root / name / "fit"
        shutil.copytree(old, fit)
        # This successful old-training tail occurs in evaluation, but not in the
        # new fit's bank/fresh data: parent copy checks must still flag it.
        old_fresh = read_jsonl(old / "training_rows.jsonl")
        for row in old_fresh:
            if row["tail"] == "SSSS":
                row["tail"] = "TTTT"
                row["sequence"] = row["sequence"][:-4] + "TTTT"
        write_jsonl(old / "training_rows.jsonl", old_fresh)
        seal_generation(old)
        confirmation_files[f"{name}/fit/generation.json"] = file_hash(
            old / "generation.json"
        )

        evaluation_seed = index * 1000 + 100
        replicates.append(dict(name=name, evaluation_seed=evaluation_seed))
        new_settings = dict(settings, training_steps=4, evaluation_seed=evaluation_seed)
        write_json(fit / "config.json", new_settings)
        config = json.loads((fit / "pilot_config.json").read_text())
        config["steps"] = 4
        write_json(fit / "pilot_config.json", config)
        fresh = read_jsonl(fit / "training_rows.jsonl")
        fresh += [dict(row, id=row["id"] + "-next", refresh_round=1) for row in fresh]
        write_jsonl(fit / "training_rows.jsonl", fresh)
        for filename in (
            "shared_training.jsonl",
            "frozen_tb/training.jsonl",
            "lastblock_tb/training.jsonl",
        ):
            rows = read_jsonl(fit / filename)
            for row in copy.deepcopy(rows):
                row["step"] += 2
                row["refresh_round"] = 1
                row["example_ids"] = [
                    identity + "-next" if identity.startswith("fresh-") else identity
                    for identity in row["example_ids"]
                ]
                rows.append(row)
            write_jsonl(fit / filename, rows)
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
                arm="prior", evaluation_seed=evaluation_seed, fit_settings=new_settings
            ),
        )
        seal_generation(prior)
        evaluate_prior(fit, prior, predictor=fixture_predictor)
        control = root / name / "control400"
        control.mkdir()
        for source, destination in (
            ("config.json", "source_settings.json"),
            ("generation.json", "source_generation.json"),
            ("bank.json", "bank.json"),
            ("training_rows.jsonl", "training_rows.jsonl"),
            ("lastblock_tb/checkpoint.pt", "checkpoint.pt"),
        ):
            shutil.copyfile(old / source, control / destination)
        write_json(
            control / "config.json",
            dict(
                arm="lastblock_400",
                evaluation_seed=evaluation_seed,
                source_checkpoint_sha256=file_hash(control / "checkpoint.pt"),
            ),
        )
        control_rows = [
            dict(row, arm="lastblock_400", id="old-" + row["id"])
            for row in read_jsonl(old / "samples.jsonl")
            if row["arm"] == "lastblock_tb"
        ]
        write_jsonl(control / "samples.jsonl", control_rows)
        combined_raw += [
            dict(row, id=f"{name}/{row['id']}", replicate=name)
            for row in read_jsonl(fit / "samples.jsonl") + control_rows + prior_rows
        ]
    write_json(
        root / "confirmation_generation.json",
        dict(complete=True, files=confirmation_files),
    )
    write_json(
        root / "config.json",
        dict(
            confirmation_run="synthetic",
            confirmation_generation_sha256=file_hash(
                root / "confirmation_generation.json"
            ),
            training_steps=4,
            replicates=replicates,
        ),
    )
    write_json(root / "provenance.json", dict(packages={}, files={}, fixture=True))
    write_jsonl(root / "samples.jsonl", combined_raw)
    seal_generation(root)
    return root


class BudgetEvaluationTests(unittest.TestCase):
    def test_union_includes_old_only_tails_and_keeps_raw_denominators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = budget_fixture(Path(temporary))
            evaluate_budget(root, root, fixture_predictor)
            metrics = json.loads((root / "metrics.json").read_text())
            for replicate in metrics["replicates"].values():
                self.assertEqual(set(replicate["arms"]), set(BUDGET_ARMS))
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
            rows = json.loads((root / "sequence_metrics.json").read_text())
            self.assertEqual(len(rows), 48)
            self.assertEqual(len({row["id"] for row in rows}), 48)
            self.assertTrue(all("own_training_copy" not in row for row in rows))

    def test_both_replicates_and_both_success_endpoints_are_required(self):
        arms = {
            arm: dict(
                joint_success_rate=0.1,
                joint_success_without_common_training_copy=0.1,
                unique_tail_fraction=0.95,
                v3_disorder_success_rate=0.9,
            )
            for arm in BUDGET_ARMS
        }
        arms["lastblock_tb"].update(
            joint_success_rate=0.27, joint_success_without_common_training_copy=0.27
        )
        arms["lastblock_400"].update(
            joint_success_rate=0.22, joint_success_without_common_training_copy=0.22
        )
        replicates = {name: dict(arms=copy.deepcopy(arms)) for name in ("one", "two")}
        self.assertTrue(budget_decision(replicates)["passes_budget_screen"])
        for metric, value in (
            ("joint_success_rate", 0.26),
            ("joint_success_without_common_training_copy", 0.26),
            ("unique_tail_fraction", 0.89),
            ("v3_disorder_success_rate", 0.84),
        ):
            changed = copy.deepcopy(replicates)
            changed["one"]["arms"]["lastblock_tb"][metric] = value
            self.assertFalse(budget_decision(changed)["passes_budget_screen"])
        # An inherited secondary screen cannot silently become an extra gate.
        replicates["one"]["arms"]["continued_supervised"]["joint_success_rate"] = 0.9
        decision = budget_decision(replicates)
        self.assertTrue(decision["passes_budget_screen"])
        self.assertFalse(
            decision["replicates"]["one"]["inherited_800_screen_secondary"][
                "passes_progression_screen"
            ]
        )

    def test_recovery_is_identical_and_does_not_modify_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = budget_fixture(Path(temporary))
            evaluate_budget(root, root, fixture_predictor)
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
            evaluate_budget(root, recovered, fixture_predictor)
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

    def test_rejects_corrupted_control_and_unmatched_settings(self):
        for change, expected in (
            ("manifest", "Captured artifact changed"),
            ("settings", "settings differ beyond"),
            ("support", "Invalid evaluation support"),
            ("count", "counts or identities"),
            ("nonfinite", "Nonfinite generated"),
            ("seed", "evaluation settings differ"),
            ("root", "Root generation does not match"),
        ):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = budget_fixture(Path(temporary))
                control = root / "replicate-1/control400"
                if change == "manifest":
                    (control / "checkpoint.pt").write_bytes(b"changed")
                elif change == "settings":
                    fit = root / "replicate-1/fit"
                    settings = json.loads((fit / "config.json").read_text())
                    settings["penalty"] = 5
                    write_json(fit / "config.json", settings)
                    seal_generation(fit)
                elif change == "seed":
                    settings = json.loads((control / "config.json").read_text())
                    settings["evaluation_seed"] += 1
                    write_json(control / "config.json", settings)
                elif change == "root":
                    rows = read_jsonl(root / "samples.jsonl")
                    rows[0]["id"] = rows[1]["id"]
                    write_jsonl(root / "samples.jsonl", rows)
                else:
                    rows = read_jsonl(control / "samples.jsonl")
                    if change == "support":
                        rows[0]["prefix"] = "AAAA"
                    elif change == "count":
                        rows.pop()
                    else:
                        rows[0]["log_policy"] = float("nan")
                    write_jsonl(control / "samples.jsonl", rows)
                if change != "manifest":
                    seal_generation(root)
                with self.assertRaisesRegex(ValueError, expected):
                    evaluate_budget(root, root, fixture_predictor)


if __name__ == "__main__":
    unittest.main()
