"""Protected three-repeat inputs and immutable reward relabeling on tiny CPU data."""

import copy
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_objective_data import objective_fixture, repin_fixture

from stability_gfn.artifacts import write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.continuous_reward import continuous_reward
from stability_gfn.idr.objective_data import check_objective_inputs
from stability_gfn.idr.success_bonus_data import (
    MAX_SEED,
    SEED_KEYS,
    check_success_bonus_inputs,
    success_bonus_preflight,
)
from stability_gfn.idr.success_bonus_reward import validate_success_bonus_rows


def success_bonus_fixture(root, backbone):
    settings = objective_fixture(root, backbone)
    # The old tiny fixture omits these saved diagnostics; real source rows have
    # them. Enrich only the new temporary inputs from their existing score arrays.
    for folder_key in ("source_run", "control_run"):
        folder = Path(settings["fit"][folder_key])
        bank = json.loads((folder / "bank.json").read_text())
        reward = AcceptanceRewardConfig(
            **json.loads((folder / "reward.json").read_text())
        )
        values = continuous_reward(
            [row["tail"] for row in bank],
            [row["isolated_tail_v3_scores"] for row in bank],
            reward,
        )
        for index, row in enumerate(bank):
            row["violations"] = values["violations"][index].tolist()
        write_json(folder / "bank.json", bank)
        repin_fixture(settings, folder_key)
    settings["fit"]["pass_multiplier"] = 9.0
    settings["replicates"] = [
        dict(
            name=f"replicate-{index + 1}",
            training_seed=1023000 + index * 100000,
            evaluation_seed=1033000 + index * 100000,
            bank_sampling_seed=1025000 + index * 100000,
        )
        for index in range(3)
    ]
    return settings


class SuccessBonusInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        with patch.object(
            FrozenProGen,
            "from_checkpoint",
            side_effect=AssertionError("No real models"),
        ):
            cls.template = success_bonus_fixture(cls.root, FrozenProGen.tiny(11))

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

    def test_three_independent_plans_and_cloned_bank_without_source_changes(self):
        source = Path(self.settings["fit"]["source_run"])
        old_bank = (source / "bank.json").read_bytes()
        with patch(
            "stability_gfn.idr.success_bonus_data.check_objective_inputs",
            wraps=check_objective_inputs,
        ) as inherited:
            inputs, plans = check_success_bonus_inputs(self.settings, False)
        inherited.assert_called_once()
        adapter = inherited.call_args.args[0]
        self.assertNotIn("pass_multiplier", adapter["fit"])
        self.assertEqual(len(adapter["replicates"]), 2)
        self.assertEqual(len(inputs), 10)
        self.assertEqual(len(plans), 3)
        self.assertTrue(inputs[4]["fixture"])
        self.assertEqual(inputs[2]["steps"], 4)
        validate_success_bonus_rows(inputs[6])
        for row in inputs[6]:
            if row["joint_pass"]:
                self.assertAlmostEqual(row["log_factor"], math.log(9))
            else:
                self.assertEqual(row["log_factor"], row["base_log_factor"])
        self.assertEqual((source / "bank.json").read_bytes(), old_bank)
        for plan, replica in zip(plans, self.settings["replicates"]):
            self.assertEqual(
                plan["fit_settings"],
                dict(self.settings["fit"], **{key: replica[key] for key in SEED_KEYS}),
            )
        plans[0]["fit_settings"]["shared_batch"]["gfn"] = 99
        self.assertEqual(plans[2]["fit_settings"]["shared_batch"]["gfn"], 2)
        self.assertEqual(self.settings["fit"]["shared_batch"]["gfn"], 2)
        workload = success_bonus_preflight(self.settings, False)
        self.assertEqual(workload["total_final_evaluation_samples"], 36)
        self.assertEqual(workload["updates_per_training_arm"], 4)
        self.assertEqual(workload["pass_multiplier"], 9)
        self.assertFalse(workload["model_or_scorer_executed"])
        self.assertFalse(workload["cuda_initialized"])

    def test_schema_and_all_three_streams_fail_closed(self):
        for overrides in (
            dict(name="../escape"),
            dict(name="replicate-1"),
            dict(training_seed=True),
            dict(evaluation_seed=-1),
            dict(bank_sampling_seed=MAX_SEED + 1),
            dict(evaluation_seed=1023001),
            dict(training_seed=1122999),
            dict(bank_sampling_seed=1033000),
            # Historic objective streams and the earlier protected fixture.
            dict(training_seed=823000),
            dict(evaluation_seed=823002),
            dict(bank_sampling_seed=833000),
            dict(evaluation_seed=825000),
            dict(training_seed=923000),
            dict(evaluation_seed=923001),
            dict(bank_sampling_seed=933000),
            dict(evaluation_seed=925000),
            dict(evaluation_seed=92000),
            dict(evaluation_seed=103000),
        ):
            settings = copy.deepcopy(self.settings)
            settings["replicates"][2].update(overrides)
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                check_success_bonus_inputs(settings, False)
        for changes in (
            dict(pass_multiplier=1),
            dict(pass_multiplier=True),
            dict(pass_multiplier=float("nan")),
            dict(pass_multiplier=10),
            dict(training_steps=32),
            dict(penalty=5),
        ):
            settings = copy.deepcopy(self.settings)
            settings["fit"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check_success_bonus_inputs(settings, False)
        for count in (0, 1, 2):
            with self.assertRaises(ValueError):
                check_success_bonus_inputs(
                    dict(self.settings, replicates=self.settings["replicates"][:count]),
                    False,
                )

    def test_small_budget_requires_manifest_protected_fixture_provenance(self):
        with self.assertRaisesRegex(ValueError, "fixture inputs"):
            check_success_bonus_inputs(self.settings, True)
        for key in ("source_run", "control_run"):
            folder = Path(self.settings["fit"][key])
            path = folder / "provenance.json"
            write_json(path, dict(json.loads(path.read_text()), fixture=False))
            repin_fixture(self.settings, key)
        with self.assertRaisesRegex(ValueError, "1600-update protocol"):
            check_success_bonus_inputs(self.settings, False)


if __name__ == "__main__":
    unittest.main()
