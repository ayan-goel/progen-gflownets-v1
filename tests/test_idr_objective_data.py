"""CPU-only input checks for the matched GFN and reward-weighted SFT experiment."""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_replay import fixture_predictor, fixture_source

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.continuous_experiment import run_continuous
from stability_gfn.idr.objective_data import (
    MAX_SEED,
    REAL_FIT_SETTINGS,
    SEED_KEYS,
    check_objective_inputs,
    objective_preflight,
)
from stability_gfn.idr.replay_experiment import run_replay
from stability_gfn.idr.warmstart_experiment import run_warmstart


def objective_fixture(root, backbone):
    """Create protected tiny source artifacts without a real model or scorer."""
    replay, _, _, _ = fixture_source(root, backbone)
    run_replay(replay, root / "replay", "cpu", backbone, fixture_predictor)
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
    continuous = dict(
        warm,
        control_run=str(root / "warm"),
        control_generation_sha256=file_hash(root / "warm/generation.json"),
        evaluation_seed=103000,
        penalty=2.5,
    )
    run_continuous(continuous, root / "continuous", "cpu", backbone, fixture_predictor)
    fit = dict(
        REAL_FIT_SETTINGS,
        source_run=str(root / "continuous"),
        source_generation_sha256=file_hash(root / "continuous/generation.json"),
        control_run=str(root / "warm"),
        control_generation_sha256=file_hash(root / "warm/generation.json"),
        control_checkpoint="continued_supervised/step-3.pt",
        control_checkpoint_sha256=file_hash(
            root / "warm/continued_supervised/step-3.pt"
        ),
        training_steps=4,
        refresh_every=2,
        microbatch_size=3,
        shared_batch={name: 2 for name in ("gfn", "supervised", "prior", "replay")},
        evaluation_samples_per_prompt=2,
        training_seed=123000,
        evaluation_seed=133000,
        bank_sampling_seed=125000,
    )
    return dict(
        fit=fit,
        replicates=[
            dict(
                name=f"replicate-{index}",
                training_seed=723000 + index * 100000,
                evaluation_seed=733000 + index * 100000,
                bank_sampling_seed=725000 + index * 100000,
            )
            for index in (1, 2)
        ],
    )


def repin_fixture(settings, folder_key, omit=None):
    """Re-pin synthetic mutations to exercise semantic and protection checks."""
    folder = Path(settings["fit"][folder_key])
    path = folder / "generation.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {
        name: file_hash(folder / name) for name in manifest["files"] if name != omit
    }
    write_json(path, manifest)
    digest_key = folder_key.replace("_run", "_generation_sha256")
    settings["fit"][digest_key] = file_hash(path)


class ObjectiveInputsTests(unittest.TestCase):
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
            cls.template = objective_fixture(cls.root, FrozenProGen.tiny(11))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.settings = copy.deepcopy(self.template)
        for key in ("source_run", "control_run"):
            original = Path(self.settings["fit"][key])
            copied = self.output / original.name
            shutil.copytree(original, copied)
            self.settings["fit"][key] = str(copied)

    def test_original_inputs_shared_by_independent_plans_and_validation_adapter(self):
        from stability_gfn.idr.lastblock_data import check_lastblock_inputs

        with patch(
            "stability_gfn.idr.objective_data.check_lastblock_inputs",
            wraps=check_lastblock_inputs,
        ) as inherited:
            inputs, plans = check_objective_inputs(self.settings, False)
        self.assertEqual(len(inputs), 10)
        self.assertEqual(inputs[2]["steps"], 4)
        self.assertEqual(len(inputs[3]), 2)
        self.assertTrue(inputs[4]["fixture"])
        inherited.assert_called_once()
        adapter = inherited.call_args.args[0]
        self.assertEqual(
            adapter["shared_batch"],
            {name: 2 for name in ("frozen_tb", "lastblock_tb", "prior", "replay")},
        )
        self.assertNotIn("sft_weighting", adapter)
        for plan, replica in zip(plans, self.settings["replicates"]):
            self.assertEqual(plan["name"], replica["name"])
            self.assertEqual(
                plan["fit_settings"],
                dict(self.settings["fit"], **{key: replica[key] for key in SEED_KEYS}),
            )
        plans[0]["fit_settings"]["shared_batch"]["gfn"] = 99
        self.assertEqual(plans[1]["fit_settings"]["shared_batch"]["gfn"], 2)
        self.assertEqual(self.settings["fit"]["shared_batch"]["gfn"], 2)

    def test_exact_schema_fixed_weighting_and_safe_replicates(self):
        bad_settings = [
            dict(self.settings, penalty=5),
            dict(self.settings, replicates=[]),
            dict(self.settings, replicates=self.settings["replicates"][:1]),
        ]
        for override in (
            dict(name="../escape"),
            dict(name="replicate-2"),
            dict(training_steps=800),
            dict(training_seed=True),
            dict(evaluation_seed=-1),
            dict(bank_sampling_seed=MAX_SEED + 1),
        ):
            bad = copy.deepcopy(self.settings)
            bad["replicates"][0].update(override)
            bad_settings.append(bad)
        for override in (
            dict(sft_weighting="uniform"),
            dict(penalty=5),
            dict(training_steps=3),
            dict(training_steps=32),
            dict(training_seed=True),
            dict(shared_batch={"gfn": 2, "supervised": 2, "prior": 2, "replay": 3}),
            dict(
                shared_batch={
                    "frozen_tb": 2,
                    "lastblock_tb": 2,
                    "prior": 2,
                    "replay": 2,
                }
            ),
        ):
            bad_settings.append(
                dict(self.settings, fit=dict(self.settings["fit"], **override))
            )
        for bad in bad_settings:
            with self.subTest(settings=bad), self.assertRaises(ValueError):
                check_objective_inputs(bad, False)

    def test_all_old_streams_and_adjacent_cross_replicate_streams_are_excluded(self):
        old_seeds = (
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
            323000,
            323001,
            323002,
            325000,
            333000,
            433000,
            533000,
            633000,
            733000,
            92000,
            92001,
            92002,
            93000,
            95000,
            103000,
            923000,
            923001,
            923002,
            933000,
            925000,
        )
        for seed in old_seeds:
            bad = copy.deepcopy(self.settings)
            bad["replicates"][0]["evaluation_seed"] = seed
            with (
                self.subTest(seed=seed),
                self.assertRaisesRegex(ValueError, "fresh and disjoint"),
            ):
                check_objective_inputs(bad, False)
        for key, value in (
            ("training_seed", 922999),
            ("evaluation_seed", 823001),
            ("bank_sampling_seed", 833000),
        ):
            bad = copy.deepcopy(self.settings)
            bad["replicates"][0][key] = value
            with (
                self.subTest(key=key, value=value),
                self.assertRaisesRegex(ValueError, "fresh and disjoint"),
            ):
                check_objective_inputs(bad, False)

    def test_small_budget_requires_both_manifest_protected_fixture_flags(self):
        with self.assertRaisesRegex(ValueError, "fixture inputs"):
            check_objective_inputs(self.settings, True)
        source = Path(self.settings["fit"]["source_run"])
        provenance_path = source / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        write_json(provenance_path, dict(provenance, fixture=False))
        repin_fixture(self.settings, "source_run")
        with self.assertRaisesRegex(ValueError, "fixture provenance differs"):
            check_objective_inputs(self.settings, False)
        control = Path(self.settings["fit"]["control_run"])
        path = control / "provenance.json"
        write_json(path, dict(json.loads(path.read_text()), fixture=False))
        repin_fixture(self.settings, "control_run")
        with self.assertRaisesRegex(ValueError, "fixed 1600-update protocol"):
            check_objective_inputs(self.settings, False)
        repin_fixture(self.settings, "source_run", omit="provenance.json")
        with self.assertRaisesRegex(ValueError, "must be protected"):
            check_objective_inputs(self.settings, False)

    def test_inherited_hashes_initialization_and_reward_are_still_checked(self):
        for key in (
            "source_generation_sha256",
            "control_generation_sha256",
            "control_checkpoint_sha256",
        ):
            bad = copy.deepcopy(self.settings)
            bad["fit"][key] = "incorrect"
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_objective_inputs(bad, False)
        source = Path(self.settings["fit"]["source_run"])
        initial = source / "initial.pt"
        checkpoint = torch.load(initial, map_location="cpu", weights_only=True)
        checkpoint["architecture"] = "adapted_previous_run"
        torch.save(checkpoint, initial)
        repin_fixture(self.settings, "source_run")
        with self.assertRaisesRegex(ValueError, "disagree: initial.pt"):
            check_objective_inputs(self.settings, False)

    def test_preflight_has_exact_costs_no_model_execution_and_preserves_inputs(self):
        previous = {
            path: file_hash(path) for path in self.output.rglob("*") if path.is_file()
        }
        with (
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("model load"),
            ),
            patch(
                "stability_gfn.idr.scoring.predict_disorder",
                side_effect=AssertionError("scorer execution"),
            ),
        ):
            report = objective_preflight(self.settings, False)
        self.assertTrue(report["ready"])
        self.assertEqual(report["updates_per_training_arm"], 4)
        self.assertEqual(report["learning_rate_decay_after_update"], 3)
        self.assertEqual(report["fresh_training_trajectories_per_replicate"], 12)
        self.assertEqual(report["replay_draws_per_replicate"], 4)
        self.assertEqual(report["trajectory_presentations_per_training_arm"], 32)
        self.assertEqual(report["evaluation_arms"], ["gfn", "supervised", "prior"])
        self.assertEqual(report["total_final_evaluation_samples"], 24)
        self.assertFalse(report["model_or_scorer_executed"])
        self.assertFalse(report["cuda_initialized"])
        self.assertEqual(previous, {path: file_hash(path) for path in previous})


if __name__ == "__main__":
    unittest.main()
