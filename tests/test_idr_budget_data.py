"""CPU-only lineage, schedule and checkpoint checks for the budget comparison."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_confirmation import confirmation_fixture
from test_idr_replay import fixture_predictor

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.budget_data import (
    MAX_SEED,
    budget_preflight,
    check_budget_inputs,
)
from stability_gfn.idr.confirmation_experiment import run_confirmation


def budget_fixture(root, backbone):
    """Build real artifact formats using only a tiny injected CPU backbone."""
    confirmation = confirmation_fixture(root, backbone)
    output = root / "confirmation"
    run_confirmation(confirmation, output, "cpu", backbone, fixture_predictor)
    return dict(
        confirmation_run=str(output),
        confirmation_generation_sha256=file_hash(output / "generation.json"),
        training_steps=8,
        replicates=[
            dict(name=f"replicate-{index}", evaluation_seed=333000 + 100000 * index)
            for index in (1, 2)
        ],
    )


def repin(settings, omit=None):
    """Re-pin synthetic hashes to exercise semantic guards rather than hash guards."""
    root = Path(settings["confirmation_run"])
    for folder in [root / "replicate-1/fit", root]:
        path = folder / "generation.json"
        manifest = json.loads(path.read_text())
        manifest["files"] = {
            name: file_hash(folder / name)
            for name in manifest["files"]
            if folder != root or name != omit
        }
        write_json(path, manifest)
    settings["confirmation_generation_sha256"] = file_hash(root / "generation.json")


class BudgetInputsTests(unittest.TestCase):
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
            cls.settings = budget_fixture(cls.root, FrozenProGen.tiny(11))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.directory_copy = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory_copy.cleanup)
        self.output = Path(self.directory_copy.name) / "confirmation"
        shutil.copytree(self.root / "confirmation", self.output)
        self.settings = copy.deepcopy(self.settings)
        self.settings["confirmation_run"] = str(self.output)

    def test_only_budget_and_evaluation_seed_change(self):
        source, plans, config, references, provenance = check_budget_inputs(
            self.settings, check_checkpoint=False
        )
        self.assertEqual(source, self.output)
        self.assertEqual(config["steps"], 8)
        self.assertEqual(len(references), 2)
        self.assertTrue(provenance["fixture"])
        for plan, replica in zip(plans, self.settings["replicates"]):
            old = json.loads((plan["source_fit"] / "config.json").read_text())
            expected = dict(
                old, training_steps=8, evaluation_seed=replica["evaluation_seed"]
            )
            self.assertEqual(plan["fit_settings"], expected)
            self.assertEqual(plan["name"], replica["name"])
        plans[0]["fit_settings"]["shared_batch"]["prior"] = 99
        self.assertEqual(plans[1]["fit_settings"]["shared_batch"]["prior"], 2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_exact_schema_and_bounded_budget(self):
        overrides = [
            dict(penalty=5),
            dict(training_steps=True),
            dict(training_steps=9),
            dict(training_steps=4),
            dict(replicates=[]),
        ]
        for override in overrides:
            with self.subTest(override=override), self.assertRaises(ValueError):
                check_budget_inputs(dict(self.settings, **override), False)
        for override in (
            dict(name="../escape"),
            dict(name="replicate-2"),
            dict(name="other-name"),
            dict(evaluation_seed=True),
            dict(evaluation_seed=-1),
            dict(evaluation_seed=MAX_SEED + 1),
            dict(training_seed=999),
        ):
            settings = copy.deepcopy(self.settings)
            settings["replicates"][0].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                check_budget_inputs(settings, False)

    def test_evaluation_streams_are_disjoint_from_all_historical_streams(self):
        for seed in (
            123000,
            123001,
            223002,
            233000,
            325000,
            92000,
            93000,
            95000,
            103000,
            533000,
        ):
            settings = copy.deepcopy(self.settings)
            settings["replicates"][0]["evaluation_seed"] = seed
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(ValueError, "fresh and disjoint"),
            ):
                check_budget_inputs(settings, False)

    def test_requires_complete_nonfixture_real_sources_and_protected_children(self):
        with self.assertRaisesRegex(ValueError, "fixture inputs"):
            check_budget_inputs(self.settings, True)
        status_path = self.output / "replicate-1/fit/status.json"
        old_status = json.loads(status_path.read_text())
        write_json(status_path, dict(status="failed"))
        with self.assertRaisesRegex(ValueError, "not complete"):
            check_budget_inputs(self.settings, False)
        write_json(status_path, old_status)
        repin(self.settings, omit="replicate-1/fit/lastblock_tb/checkpoint.pt")
        with self.assertRaisesRegex(ValueError, "not protected"):
            check_budget_inputs(self.settings, False)

    def test_rejects_settings_drift_even_if_manifest_repinned(self):
        path = self.output / "replicate-1/fit/config.json"
        old = json.loads(path.read_text())
        write_json(path, dict(old, last_block_learning_rate=1e-3))
        repin(self.settings)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            check_budget_inputs(self.settings, False)

    def test_rejects_changed_initialization_or_bank(self):
        path = self.output / "replicate-1/fit/bank.json"
        bank = json.loads(path.read_text())
        bank[0]["log_factor"] = 999
        write_json(path, bank)
        repin(self.settings)
        with self.assertRaisesRegex(ValueError, "initial bank/source differs"):
            check_budget_inputs(self.settings, False)

    def test_rejects_wrong_final_step_and_nonfinite_or_wrong_shape_tensors(self):
        path = self.output / "replicate-1/fit/lastblock_tb/checkpoint.pt"
        original = torch.load(path, map_location="cpu", weights_only=True)
        cases = []
        altered = copy.deepcopy(original)
        altered["step"] = 3
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
        altered["log_z"] = torch.zeros(1)
        cases.append((altered, "normalizers"))
        altered = copy.deepcopy(original)
        altered["reference"] = "adapted ProGen"
        cases.append((altered, "metadata"))
        for checkpoint, message in cases:
            torch.save(checkpoint, path)
            repin(self.settings)
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                check_budget_inputs(self.settings, False)

    def test_preflight_counts_extra_budget_and_all_six_final_arms(self):
        inputs = check_budget_inputs(self.settings, False)
        with patch(
            "stability_gfn.idr.budget_data.check_budget_inputs", return_value=inputs
        ):
            result = budget_preflight(self.settings)
        self.assertEqual(result["updates_per_training_arm"], 8)
        self.assertEqual(result["reference_updates_per_training_arm"], 4)
        self.assertEqual(result["learning_rate_decay_after_update"], 6)
        self.assertEqual(result["fresh_training_trajectories_per_replicate"], 24)
        self.assertEqual(result["trajectory_presentations_per_training_arm"], 64)
        self.assertEqual(result["evaluation_arms_per_replicate"], 6)
        self.assertEqual(result["total_final_evaluation_samples"], 48)
        self.assertFalse(result["model_or_scorer_executed"])
        self.assertFalse(result["cuda_initialized"])


if __name__ == "__main__":
    unittest.main()
