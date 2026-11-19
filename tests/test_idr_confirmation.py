"""Tiny CPU end-to-end confirmation, including fresh starts and saved recovery."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_replay import fixture_predictor, fixture_source

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.confirmation_data import check_confirmation_inputs
from stability_gfn.idr.confirmation_experiment import (
    reevaluate_confirmation,
    run_confirmation,
)
from stability_gfn.idr.continuous_experiment import run_continuous
from stability_gfn.idr.lastblock_evaluation import read_jsonl
from stability_gfn.idr.lastblock_experiment import run_lastblock
from stability_gfn.idr.replay_experiment import run_replay
from stability_gfn.idr.warmstart_experiment import run_warmstart


def confirmation_fixture(root, backbone):
    """Build the actual artifact lineage with two prompts and four-token tails."""
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
    pilot = dict(
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
        shared_batch={
            arm: 2 for arm in ("frozen_tb", "lastblock_tb", "prior", "replay")
        },
        evaluation_samples_per_prompt=2,
        training_seed=123000,
        evaluation_seed=133000,
        bank_sampling_seed=125000,
        penalty=2.5,
        last_block_learning_rate=1e-5,
        block_gradient_clip=1.0,
    )
    run_lastblock(pilot, root / "pilot", "cpu", backbone, fixture_predictor)
    return dict(
        pilot_run=str(root / "pilot"),
        pilot_generation_sha256=file_hash(root / "pilot/generation.json"),
        replicates=[
            dict(
                name=f"replicate-{index}",
                training_seed=123000 + index * 100000,
                evaluation_seed=133000 + index * 100000,
                bank_sampling_seed=125000 + index * 100000,
            )
            for index in (1, 2)
        ],
    )


class ConfirmationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_two_fresh_fits_plain_baseline_and_self_contained_recovery(self):
        backbone = FrozenProGen.tiny(11)
        original = {
            key: value.clone() for key, value in backbone.model.state_dict().items()
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real model execution"),
            ),
        ):
            root = Path(directory)
            settings = confirmation_fixture(root, backbone)
            check_confirmation_inputs(settings, False)
            with self.assertRaises(ValueError):
                check_confirmation_inputs(settings, True)
            historical = {
                path: file_hash(path)
                for folder in (
                    "pilot",
                    "continuous",
                    "warm",
                    "replay",
                    "source",
                    "data",
                )
                for path in (root / folder).rglob("*")
                if path.is_file()
            }
            output = root / "confirmation"
            run_confirmation(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "complete"
            )
            self.assertEqual(len(read_jsonl(output / "samples.jsonl")), 40)
            records = read_jsonl(output / "samples.jsonl")
            self.assertEqual(len({row["id"] for row in records}), 40)
            self.assertEqual(
                {row["replicate"] for row in records}, {"replicate-1", "replicate-2"}
            )
            fit_traces = []
            for replicate in settings["replicates"]:
                fit = output / replicate["name"] / "fit"
                baseline = output / replicate["name"] / "prior"
                fit_settings = json.loads((fit / "config.json").read_text())
                for key in ("training_seed", "evaluation_seed", "bank_sampling_seed"):
                    self.assertEqual(fit_settings[key], replicate[key])
                # Every run starts from the same original head, not the previous
                # replicate's trained weights. Both histories still contain all steps.
                self.assertEqual(
                    file_hash(fit / "initial.pt"), file_hash(root / "pilot/initial.pt")
                )
                initial = torch.load(
                    fit / "initial.pt", map_location="cpu", weights_only=True
                )
                unchanged = torch.load(
                    fit / "unchanged/checkpoint.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                for key, value in initial["head"].items():
                    torch.testing.assert_close(
                        value, unchanged["head"][key], rtol=0, atol=0
                    )
                self.assertEqual(
                    [row["step"] for row in read_jsonl(fit / "shared_training.jsonl")],
                    [1, 2, 3, 4],
                )
                fit_traces.append((fit / "training_rows.jsonl").read_text())
                prior = read_jsonl(baseline / "samples.jsonl")
                self.assertEqual(len(prior), 4)
                for row in prior:
                    self.assertEqual(row["arm"], "prior")
                    self.assertEqual(row["log_policy"], row["log_prior"])
                    self.assertNotIn("log_z", row)
                # Baseline evaluation must not simply reuse the prior rows drawn
                # during training; row IDs and RNG streams are separate.
                self.assertTrue(
                    {row["id"] for row in prior}.isdisjoint(
                        row["id"] for row in read_jsonl(fit / "training_rows.jsonl")
                    )
                )
            self.assertNotEqual(fit_traces[0], fit_traces[1])
            for key, value in backbone.model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(
                all(parameter.grad is None for parameter in backbone.model.parameters())
            )
            for path, digest in historical.items():
                self.assertEqual(file_hash(path), digest)
            with self.assertRaises(FileExistsError):
                run_confirmation(settings, output, "cpu", backbone, fixture_predictor)
            # Recovery needs only the new output snapshot and unchanged software.
            for folder in ("pilot", "continuous", "warm", "replay", "source", "data"):
                (root / folder).rename(root / f"hidden-{folder}")
            reevaluate_confirmation(output, root / "recovered", fixture_predictor)
            for name in ("metrics.json", "decision.json", "report.md"):
                self.assertEqual(
                    (output / name).read_text(), (root / "recovered" / name).read_text()
                )
            with (output / "replicate-2/prior/samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_confirmation(output, root / "corrupt", fixture_predictor)
            self.assertFalse(torch.cuda.is_initialized())

    def test_final_scoring_failure_is_recoverable_without_retraining(self):
        backbone = FrozenProGen.tiny(11)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real model execution"),
            ),
        ):
            root = Path(directory)
            settings = confirmation_fixture(root, backbone)
            output = root / "interrupted"
            with patch(
                "stability_gfn.idr.confirmation_experiment.evaluate_prior",
                side_effect=RuntimeError("Injected scoring interruption"),
            ):
                with self.assertRaisesRegex(RuntimeError, "scoring interruption"):
                    run_confirmation(
                        settings, output, "cpu", backbone, fixture_predictor
                    )
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "failed"
            )
            self.assertTrue((output / "generation.json").is_file())
            # Recovery cannot accidentally depend on source runs or retrain.
            for folder in ("pilot", "continuous", "warm", "replay", "source", "data"):
                (root / folder).rename(root / f"hidden-{folder}")
            with patch(
                "stability_gfn.idr.confirmation_experiment.run_lastblock",
                side_effect=AssertionError("Recovery must not retrain"),
            ):
                report = reevaluate_confirmation(
                    output, root / "recovered", fixture_predictor
                )
            self.assertTrue(report.is_file())
            self.assertEqual(
                json.loads((root / "recovered/status.json").read_text())["status"],
                "complete",
            )
            self.assertFalse(torch.cuda.is_initialized())
