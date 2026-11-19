"""Synthetic CPU checks for plain-reference scoring and confirmation decisions."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from test_idr_lastblock_evaluation import evaluation_fixture, write_jsonl
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.idr.confirmation_evaluation import (
    CONFIRMATION_ARMS,
    confirmation_decision,
    evaluate_prior,
    summarize_confirmation,
)
from stability_gfn.idr.lastblock_evaluation import evaluate_lastblock, read_jsonl


def seal_generation(folder):
    """Fixture manifest pins raw inputs exactly as the real runner does."""
    files = {
        str(path.relative_to(folder)): file_hash(path)
        for path in folder.rglob("*")
        if path.is_file() and path.name != "generation.json"
    }
    write_json(
        folder / "generation.json",
        dict(
            complete=True,
            count=len(read_jsonl(folder / "samples.jsonl")),
            files=files,
        ),
    )


def confirmation_fixture(root):
    replicates = []
    for index, name in enumerate(("replicate-1", "replicate-2"), 1):
        fit = root / name / "fit"
        prior = root / name / "prior"
        fit.mkdir(parents=True)
        prior.mkdir()
        evaluation_fixture(fit)
        streams = dict(
            name=name,
            training_seed=1000 * index,
            evaluation_seed=1000 * index + 100,
            bank_sampling_seed=1000 * index + 200,
        )
        replicates.append(streams)
        settings = json.loads((fit / "config.json").read_text())
        settings.update({key: value for key, value in streams.items() if key != "name"})
        write_json(fit / "config.json", settings)
        seal_generation(fit)
        evaluate_lastblock(fit, fit, fixture_predictor)
        rows = []
        for row in read_jsonl(fit / "samples.jsonl"):
            if row["arm"] != "unchanged":
                continue
            row = dict(row, id="prior-" + row["id"], arm="prior")
            row["log_policy"] = row["log_prior"]
            del row["log_z"]
            rows.append(row)
        write_jsonl(prior / "samples.jsonl", rows)
        write_json(
            prior / "config.json",
            dict(
                arm="prior",
                evaluation_seed=streams["evaluation_seed"],
                fit_settings=settings,
            ),
        )
        seal_generation(prior)
        evaluate_prior(fit, prior, predictor=fixture_predictor)
    write_json(root / "config.json", dict(replicates=replicates))
    return root


class ConfirmationEvaluationTests(unittest.TestCase):
    def test_plain_reference_uses_same_common_copy_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = confirmation_fixture(Path(temporary))
            prior = root / "replicate-1/prior"
            metrics = json.loads((prior / "metrics.json").read_text())["arms"]["prior"]
            self.assertEqual(metrics["count"], 4)
            self.assertEqual(metrics["joint_success_rate"], 1)
            self.assertEqual(metrics["exact_common_training_copy_fraction"], 0.5)
            self.assertEqual(metrics["joint_success_without_common_training_copy"], 0.5)
            self.assertEqual(metrics["exact_own_training_copy_fraction"], 0)
            self.assertEqual(metrics["mean_log_policy_minus_frozen_log_prior"], 0)
            self.assertNotIn("own_reward_tb_residual_rmse", metrics)
            self.assertEqual(
                len(json.loads((prior / "metrics.json").read_text())["by_prompt"]), 2
            )

    def test_plain_rejects_guided_probability_invalid_budget_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = confirmation_fixture(Path(temporary))
            fit, prior = root / "replicate-1/fit", root / "replicate-1/prior"
            original = read_jsonl(prior / "samples.jsonl")
            cases = []
            changed = copy.deepcopy(original)
            changed[0]["log_policy"] += 1
            cases.append((changed, "policy must equal"))
            changed = copy.deepcopy(original)
            changed[0]["log_z"] = 0
            cases.append((changed, "no fitted TB normalizer"))
            changed = copy.deepcopy(original)
            changed[0]["id"] = changed[1]["id"]
            cases.append((changed, "counts or identities"))
            cases.append((original[:-1], "counts or identities"))
            changed = copy.deepcopy(original)
            changed[0]["prefix"] = "AAAA"
            cases.append((changed, "Invalid evaluation support"))
            for rows, message in cases:
                with self.subTest(message=message):
                    write_jsonl(prior / "samples.jsonl", rows)
                    seal_generation(prior)
                    with self.assertRaisesRegex(ValueError, message):
                        evaluate_prior(fit, prior, predictor=fixture_predictor)

    def test_one_failed_seed_cannot_be_hidden_and_plain_screen_is_separate(self):
        arms = {
            arm: dict(
                joint_success_rate=rate,
                joint_success_without_common_training_copy=rate,
                unique_tail_fraction=1.0,
                v3_disorder_success_rate=0.9,
            )
            for arm, rate in zip(CONFIRMATION_ARMS, (0.02, 0.08, 0.14, 0.12, 0.21))
        }
        replicates = {name: dict(arms=copy.deepcopy(arms)) for name in ("one", "two")}
        self.assertTrue(confirmation_decision(replicates)["passes_confirmation_screen"])
        changed = copy.deepcopy(replicates)
        changed["one"]["arms"]["lastblock_tb"]["joint_success_rate"] = 0.13
        changed["two"]["arms"]["lastblock_tb"]["joint_success_rate"] = 0.6
        result = confirmation_decision(changed)
        self.assertFalse(result["passes_confirmation_screen"])
        self.assertTrue(result["replicates"]["two"]["passes_confirmation_screen"])
        changed = copy.deepcopy(replicates)
        for arm in CONFIRMATION_ARMS[1:]:
            changed["one"]["arms"][arm]["unique_tail_fraction"] = 0.9
        result = confirmation_decision(changed)
        self.assertTrue(result["passes_inherited_screen_all_replicates"])
        self.assertFalse(result["passes_plain_screen_all_replicates"])
        self.assertFalse(result["passes_confirmation_screen"])

    def test_recovery_uses_saved_sources_and_preserves_all_prompt_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = confirmation_fixture(Path(temporary) / "source")
            original = root / "summary"
            original.mkdir()
            summarize_confirmation(root, original)
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
                evaluate_prior(fit, prior, out_prior, predictor=fixture_predictor)
            summarize_confirmation(root, recovered, evaluations=recovered)
            for filename in ("metrics.json", "decision.json", "report.md"):
                self.assertEqual(
                    (original / filename).read_bytes(),
                    (recovered / filename).read_bytes(),
                )
            result = json.loads((recovered / "metrics.json").read_text())
            self.assertEqual(len(result["replicates"]), 2)
            for value in result["replicates"].values():
                self.assertEqual(set(value["arms"]), set(CONFIRMATION_ARMS))
                self.assertEqual(len(value["by_prompt"]), 10)
            self.assertNotIn("pooled", result)
            (root / "replicate-1/fit/config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Captured artifact changed"):
                summarize_confirmation(root, recovered)

    def test_summary_rejects_duplicate_random_streams(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = confirmation_fixture(Path(temporary))
            settings = json.loads((root / "config.json").read_text())
            settings["replicates"][1]["training_seed"] = settings["replicates"][0][
                "training_seed"
            ]
            write_json(root / "config.json", settings)
            with self.assertRaisesRegex(ValueError, "reuse training_seed"):
                summarize_confirmation(root, root)


if __name__ == "__main__":
    unittest.main()
