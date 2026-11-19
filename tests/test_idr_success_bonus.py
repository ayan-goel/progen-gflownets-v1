"""Tiny CPU integration for the three-repeat, matched success-bonus experiment."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_idr_replay import fixture_predictor
from test_idr_success_bonus_data import success_bonus_fixture

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.lastblock_evaluation import read_jsonl
from stability_gfn.idr.success_bonus_experiment import (
    _save_policy,
    reevaluate_success_bonus,
    run_success_bonus,
)
from stability_gfn.idr.success_bonus_reward import REWARD_VERSION
from stability_gfn.idr.warmstart_data import initialize_normalizer


class SuccessBonusIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            path.write_bytes(b"previous complete checkpoint")

            def fail_after_partial_write(checkpoint, destination):
                self.assertEqual(checkpoint["reward_version"], REWARD_VERSION)
                self.assertEqual(checkpoint["reward_definition"]["pass_multiplier"], 9)
                destination.write_bytes(b"partial replacement")
                raise OSError("Synthetic disk write failure")

            with patch(
                "stability_gfn.idr.success_bonus_experiment.torch.save",
                fail_after_partial_write,
            ):
                with self.assertRaises(OSError):
                    _save_policy(
                        path,
                        "supervised",
                        SimpleNamespace(last_block=torch.nn.Linear(2, 2)),
                        torch.nn.Linear(2, 2),
                        None,
                        [{"id": "prompt"}],
                        100,
                    )
            self.assertEqual(path.read_bytes(), b"previous complete checkpoint")

    def test_three_matched_repeats_bonus_everywhere_and_source_free_recovery(self):
        backbone = FrozenProGen.tiny(11)
        original = {
            key: value.clone() for key, value in backbone.model.state_dict().items()
        }
        checked_offsets = []

        def check_normalizer(fit_rows, train_bank, prompt_ids):
            # Equal pass/fail weight means the ninefold pass bonus shifts each
            # least-squares offset by log(9)/2, not log(9). No held-out row enters.
            self.assertTrue(all(row["split"] == "train" for row in train_bank))
            self.assertTrue(all(row["split"] == "train" for row in fit_rows))
            old_bank = [
                dict(row, log_factor=row["base_log_factor"]) for row in train_bank
            ]
            before, _ = initialize_normalizer(fit_rows, old_bank, prompt_ids)
            after, diagnostics = initialize_normalizer(fit_rows, train_bank, prompt_ids)
            torch.testing.assert_close(
                after - before,
                torch.full_like(after, math.log(9) / 2),
                rtol=0,
                atol=1e-5,
            )
            checked_offsets.append(diagnostics)
            return after, diagnostics

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real ProGen construction"),
            ),
            patch(
                "metapredict.predict_disorder",
                side_effect=AssertionError("No real disorder scorer execution"),
            ),
        ):
            root = Path(directory)
            history = root / "history"
            history.mkdir()
            settings = success_bonus_fixture(history, backbone)
            historical_hashes = {
                path: file_hash(path) for path in history.rglob("*") if path.is_file()
            }
            output = root / "success-bonus"
            with patch(
                "stability_gfn.idr.success_bonus_experiment.initialize_normalizer",
                check_normalizer,
            ):
                run_success_bonus(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(len(checked_offsets), 1)
            self.assertEqual(
                json.loads((output / "initialization.json").read_text())["normalizer"],
                checked_offsets[0],
            )
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertTrue(status["fixture"])
            definition = json.loads((output / "reward_definition.json").read_text())
            self.assertEqual(
                definition,
                dict(version=REWARD_VERSION, penalty=2.5, pass_multiplier=9.0),
            )
            bank = json.loads((output / "bank.json").read_text())
            records = read_jsonl(output / "samples.jsonl")
            self.assertEqual(len(records), 36)
            self.assertEqual(len({row["id"] for row in records}), 36)
            self.assertEqual(
                {row["arm"] for row in records}, {"gfn", "supervised", "prior"}
            )
            self.assertEqual(
                {row["replicate"] for row in records},
                {plan["name"] for plan in settings["replicates"]},
            )
            for plan in settings["replicates"]:
                folder = output / plan["name"]
                gfn = read_jsonl(folder / "gfn/training.jsonl")
                supervised = read_jsonl(folder / "supervised/training.jsonl")
                self.assertEqual(len(gfn), 4)
                self.assertEqual(len(supervised), 4)
                self.assertEqual(
                    [row["example_ids"] for row in gfn],
                    [row["example_ids"] for row in supervised],
                )
                self.assertEqual(gfn[0]["rates"]["base"], gfn[2]["rates"]["base"])
                self.assertAlmostEqual(
                    gfn[3]["rates"]["base"], gfn[0]["rates"]["base"] * 0.1
                )
                fresh = read_jsonl(folder / "training_rows.jsonl")
                self.assertEqual(len(fresh), 12)
                for row in bank + fresh:
                    self.assertEqual(row["reward_version"], REWARD_VERSION)
                    self.assertAlmostEqual(
                        row["success_log_bonus"],
                        math.log(9) if row["joint_pass"] else 0,
                    )
                    self.assertAlmostEqual(
                        row["log_factor"],
                        row["base_log_factor"] + row["success_log_bonus"],
                    )
                for arm in ("gfn", "supervised"):
                    checkpoint = torch.load(
                        folder / arm / "checkpoint.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.assertEqual(checkpoint["step"], 4)
                    self.assertEqual(checkpoint["arm"], arm)
                    self.assertEqual(checkpoint["reward_version"], REWARD_VERSION)
                    self.assertEqual(checkpoint["reward_definition"], definition)
                    self.assertEqual(checkpoint["log_z"] is None, arm == "supervised")
                    self.assertFalse((folder / arm / "latest.pt").exists())
                    self.assertFalse(list((folder / arm).glob("step-*.pt")))
                    self.assertTrue(
                        any(
                            not torch.equal(value, original[f"transformer.h.1.{key}"])
                            for key, value in checkpoint["last_block"].items()
                        )
                    )
            scored = json.loads((output / "sequence_metrics.json").read_text())
            self.assertEqual(len(scored), 36)
            for row in scored:
                acceptance = row["acceptance"]
                self.assertAlmostEqual(
                    acceptance["log_factor"],
                    acceptance["base_log_factor"]
                    + (math.log(9) if acceptance["joint_pass"] else 0),
                )
                if row["arm"] == "gfn":
                    self.assertAlmostEqual(
                        row["own_reward_tb_residual"],
                        row["log_z"]
                        + row["log_policy"]
                        - row["log_prior"]
                        - acceptance["log_factor"],
                    )
            decision = json.loads((output / "decision.json").read_text())
            self.assertEqual(len(decision["replicates"]), 3)
            self.assertEqual(decision["required_joint_gain"], 0.1)
            self.assertTrue(
                decision["requires_strictly_more_distinct_successful_noncopy_tails"]
            )
            for key, value in backbone.model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(
                all(parameter.grad is None for parameter in backbone.model.parameters())
            )
            for path, digest in historical_hashes.items():
                self.assertEqual(file_hash(path), digest)
            with self.assertRaises(FileExistsError):
                run_success_bonus(settings, output, "cpu", backbone, fixture_predictor)

            # Final evaluation is self-contained after all prior run folders move.
            history.rename(root / "hidden-history")
            recovery = root / "recovery"
            reevaluate_success_bonus(output, recovery, fixture_predictor)
            for name in (
                "metrics.json",
                "decision.json",
                "training_summary.json",
                "sequence_metrics.json",
                "report.md",
            ):
                self.assertEqual(
                    (output / name).read_bytes(), (recovery / name).read_bytes()
                )
            with (output / "replicate-1/samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_success_bonus(output, root / "corrupt", fixture_predictor)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
