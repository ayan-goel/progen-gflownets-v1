"""CPU-only provenance and scope checks for the final bounded budget increase."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_budget_data import budget_fixture
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.budget1600_data import (
    MAX_SEED,
    budget1600_preflight,
    check_budget1600_inputs,
)
from stability_gfn.idr.budget_experiment import run_budget


def budget1600_fixture(root, backbone):
    """Build a tiny completed 8-update source; the supplied root must exist."""
    settings = budget_fixture(root, backbone)
    output = root / "budget"
    run_budget(settings, output, "cpu", backbone, fixture_predictor)
    return dict(
        budget_run=str(output),
        budget_generation_sha256=file_hash(output / "generation.json"),
        training_steps=16,
        replicates=[
            dict(name=f"replicate-{index}", evaluation_seed=533000 + 100000 * index)
            for index in (1, 2)
        ],
    )


def repin1600(settings, omit=None):
    """Update synthetic hashes so semantic checks are tested independently."""
    root = Path(settings["budget_run"])
    for folder in [root / "replicate-1/fit", root]:
        path = folder / "generation.json"
        manifest = json.loads(path.read_text())
        manifest["files"] = {
            name: file_hash(folder / name)
            for name in manifest["files"]
            if folder != root or name != omit
        }
        write_json(path, manifest)
    settings["budget_generation_sha256"] = file_hash(root / "generation.json")


class Budget1600InputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with patch.object(
            FrozenProGen,
            "from_checkpoint",
            side_effect=AssertionError("No real model execution"),
        ):
            cls.settings = budget1600_fixture(cls.root, FrozenProGen.tiny(11))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "budget"
        shutil.copytree(self.root / "budget", self.output)
        self.settings = copy.deepcopy(self.settings)
        self.settings["budget_run"] = str(self.output)

    def test_only_steps_and_evaluation_change_and_originals_are_retained(self):
        source, plans, config, references, provenance = check_budget1600_inputs(
            self.settings, False
        )
        self.assertEqual(source, self.output)
        self.assertEqual(config["steps"], 16)
        self.assertEqual(len(references), 2)
        self.assertTrue(provenance["fixture"])
        for plan, replica in zip(plans, self.settings["replicates"]):
            self.assertEqual(plan["source_fit"], source / replica["name"] / "fit")
            old = json.loads((plan["source_fit"] / "config.json").read_text())
            self.assertEqual(
                plan["fit_settings"],
                dict(
                    old, training_steps=16, evaluation_seed=replica["evaluation_seed"]
                ),
            )
        plans[0]["fit_settings"]["shared_batch"]["prior"] = 99
        self.assertEqual(plans[1]["fit_settings"]["shared_batch"]["prior"], 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_schema_fixed_budget_and_safe_replicates(self):
        for override in (
            dict(penalty=5),
            dict(training_steps=True),
            dict(training_steps=8),
            dict(training_steps=32),
            dict(replicates=[]),
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                check_budget1600_inputs(dict(self.settings, **override), False)
        for override in (
            dict(name="../escape"),
            dict(name="replicate-2"),
            dict(name="another"),
            dict(evaluation_seed=True),
            dict(evaluation_seed=-1),
            dict(evaluation_seed=MAX_SEED + 1),
            dict(training_seed=999),
        ):
            settings = copy.deepcopy(self.settings)
            settings["replicates"][0].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                check_budget1600_inputs(settings, False)

    def test_evaluation_seeds_exclude_all_prior_budget_and_sampler_streams(self):
        for seed in (
            123000,
            123001,
            123002,
            133000,
            125000,
            223000,
            223001,
            223002,
            225000,
            233000,
            323002,
            325000,
            333000,
            433000,
            533000,
            92000,
            93000,
            95000,
            103000,
            733000,
        ):
            settings = copy.deepcopy(self.settings)
            settings["replicates"][0]["evaluation_seed"] = seed
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(ValueError, "fresh and disjoint"),
            ):
                check_budget1600_inputs(settings, False)

    def test_requires_complete_real_sources_and_protected_history(self):
        with self.assertRaisesRegex(ValueError, "fixture inputs"):
            check_budget1600_inputs(self.settings, True)
        status = self.output / "replicate-1/fit/status.json"
        previous = json.loads(status.read_text())
        write_json(status, dict(status="failed"))
        with self.assertRaisesRegex(ValueError, "not complete"):
            check_budget1600_inputs(self.settings, False)
        write_json(status, previous)
        repin1600(self.settings, omit="replicate-1/control400/training_rows.jsonl")
        with self.assertRaisesRegex(ValueError, "not protected"):
            check_budget1600_inputs(self.settings, False)

    def test_detects_child_settings_drift_even_with_updated_hashes(self):
        path = self.output / "replicate-1/fit/config.json"
        settings = json.loads(path.read_text())
        write_json(path, dict(settings, penalty=5))
        repin1600(self.settings)
        with self.assertRaisesRegex(ValueError, "settings differ from its derivation"):
            check_budget1600_inputs(self.settings, False)

    def test_rejects_changed_initial_bank_and_historical_training(self):
        path = self.output / "replicate-1/fit/bank.json"
        saved = path.read_text()
        bank = json.loads(saved)
        bank[0]["log_factor"] = 999
        write_json(path, bank)
        repin1600(self.settings)
        with self.assertRaisesRegex(ValueError, "initialization or bank changed"):
            check_budget1600_inputs(self.settings, False)
        path.write_text(saved)
        history = self.output / "replicate-1/control400/training_rows.jsonl"
        history.write_text(history.read_text() + "\n")
        repin1600(self.settings)
        with self.assertRaisesRegex(
            ValueError, "Historical 400-update copy input differs"
        ):
            check_budget1600_inputs(self.settings, False)

    def test_rejects_malformed_final_checkpoint_and_reference(self):
        path = self.output / "replicate-1/fit/lastblock_tb/checkpoint.pt"
        original = torch.load(path, map_location="cpu", weights_only=True)
        cases = []
        altered = copy.deepcopy(original)
        altered["step"] = 4
        cases.append((altered, "metadata"))
        altered = copy.deepcopy(original)
        altered["last_block"]["mlp.fc_in.weight"][0, 0] = float("nan")
        cases.append((altered, "tensors"))
        altered = copy.deepcopy(original)
        altered["head"]["output.weight"] = torch.zeros(2, 2)
        cases.append((altered, "head shapes"))
        altered = copy.deepcopy(original)
        altered["last_block"]["attn.out_proj.weight"] = torch.zeros(2, 2)
        cases.append((altered, "last-block shapes"))
        altered = copy.deepcopy(original)
        altered["reference"] = "adapted checkpoint"
        cases.append((altered, "metadata"))
        for checkpoint, message in cases:
            torch.save(checkpoint, path)
            repin1600(self.settings)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                check_budget1600_inputs(self.settings, False)

    def test_preflight_counts_bounded_budget_and_six_final_arms(self):
        inputs = check_budget1600_inputs(self.settings, False)
        with patch(
            "stability_gfn.idr.budget1600_data.check_budget1600_inputs",
            return_value=inputs,
        ):
            result = budget1600_preflight(self.settings)
        self.assertTrue(result["ready"])
        self.assertEqual(result["updates_per_training_arm"], 16)
        self.assertEqual(result["reference_updates_per_training_arm"], 8)
        self.assertEqual(result["learning_rate_decay_after_update"], 12)
        self.assertEqual(result["fresh_training_trajectories_per_replicate"], 48)
        self.assertEqual(result["trajectory_presentations_per_training_arm"], 128)
        self.assertEqual(result["evaluation_arms_per_replicate"], 6)
        self.assertEqual(result["total_final_evaluation_samples"], 48)
        self.assertFalse(result["model_or_scorer_executed"])
        self.assertFalse(result["cuda_initialized"])


if __name__ == "__main__":
    unittest.main()
