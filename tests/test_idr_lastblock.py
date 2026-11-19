"""Tiny CPU integration: paired refresh, immutable reference, snapshot recovery."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_replay import fixture_predictor, fixture_source

from stability_gfn.artifacts import file_hash
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.continuous_experiment import run_continuous
from stability_gfn.idr.lastblock_data import check_lastblock_inputs
from stability_gfn.idr.lastblock_experiment import reevaluate_lastblock, run_lastblock
from stability_gfn.idr.replay_experiment import run_replay
from stability_gfn.idr.warmstart_experiment import run_warmstart


class LastBlockIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cpu_paired_fit_and_self_contained_recovery(self):
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
            replay_settings, _, _, _ = fixture_source(root, backbone)
            run_replay(
                replay_settings, root / "replay", "cpu", backbone, fixture_predictor
            )
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
            run_continuous(
                continuous, root / "continuous", "cpu", backbone, fixture_predictor
            )
            settings = dict(
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
            check_lastblock_inputs(settings, False)
            for bad in (
                dict(settings, training_steps=3),
                dict(settings, penalty=3),
                dict(settings, control_checkpoint_sha256="wrong"),
            ):
                with self.assertRaises(ValueError):
                    check_lastblock_inputs(bad, False)
            with self.assertRaises(ValueError):
                check_lastblock_inputs(settings, True)
            prior_files = {
                p: file_hash(p)
                for folder in ("warm", "continuous")
                for p in (root / folder).rglob("*")
                if p.is_file()
            }
            output = root / "lastblock"
            run_lastblock(settings, output, "cpu", backbone, fixture_predictor)
            self.assertEqual(
                json.loads((output / "status.json").read_text())["status"], "complete"
            )
            for key, value in backbone.model.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in backbone.model.parameters()))
            for path, digest in prior_files.items():
                self.assertEqual(file_hash(path), digest)
            last = torch.load(
                output / "lastblock_tb/checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            initial_block = backbone.model.transformer.h[-1].state_dict()
            self.assertTrue(
                any(
                    not torch.equal(v, initial_block[k])
                    for k, v in last["last_block"].items()
                )
            )
            unchanged = torch.load(
                output / "unchanged/checkpoint.pt",
                map_location="cpu",
                weights_only=True,
            )
            initial_head = torch.load(
                output / "initial.pt", map_location="cpu", weights_only=True
            )
            for key, value in unchanged["head"].items():
                torch.testing.assert_close(
                    value, initial_head["head"][key], rtol=0, atol=0
                )
            self.assertGreater(
                json.loads((output / "adaptation.json").read_text())[
                    "last_block_delta_l2"
                ],
                0,
            )
            self.assertEqual(
                len((output / "training_rows.jsonl").read_text().splitlines()), 12
            )
            self.assertEqual(
                len((output / "samples.jsonl").read_text().splitlines()), 16
            )
            with self.assertRaises(FileExistsError):
                run_lastblock(settings, output, "cpu", backbone, fixture_predictor)
            # Evaluation recovery has no dependency on old run folders or data.
            for folder in ("continuous", "warm", "replay", "data"):
                (root / folder).rename(root / f"hidden-{folder}")
            reevaluate_lastblock(output, root / "recovered", fixture_predictor)
            for name in (
                "metrics.json",
                "decision.json",
                "bank_diagnostics.json",
                "training_summary.json",
            ):
                self.assertEqual(
                    json.loads((output / name).read_text()),
                    json.loads((root / "recovered" / name).read_text()),
                )
            with (output / "samples.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_lastblock(output, root / "corrupt", fixture_predictor)
            self.assertFalse(torch.cuda.is_initialized())
