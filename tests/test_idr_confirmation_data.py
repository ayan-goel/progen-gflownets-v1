"""CPU-only checks that confirmation changes RNG streams, not the experiment."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.idr.acceptance_reward import AcceptanceRewardConfig
from stability_gfn.idr.confirmation_data import (
    MAX_SEED,
    SEED_KEYS,
    check_confirmation_inputs,
    confirmation_preflight,
)

MODULE = "stability_gfn.idr.confirmation_data"


def refresh_manifest(pilot, settings, omit=None):
    """Re-pin synthetic files so tests distinguish missing guards from bad hashes."""
    files = {
        path.name: file_hash(path)
        for path in pilot.iterdir()
        if path.is_file() and path.name not in {"generation.json", "status.json", omit}
    }
    write_json(pilot / "generation.json", dict(complete=True, count=8, files=files))
    settings["pilot_generation_sha256"] = file_hash(pilot / "generation.json")


def input_fixture(root):
    """Synthetic snapshots; the inherited validator is mocked, never a real model."""
    source, control, pilot = (root / name for name in ("source", "control", "pilot"))
    for folder in (source, control, pilot, control / "continued_supervised"):
        folder.mkdir()
    historical = dict(
        training_seed=92000, evaluation_seed=93000, bank_sampling_seed=95000
    )
    for folder in (source, control):
        write_json(folder / "config.json", historical)
        write_json(folder / "generation.json", {"synthetic": folder.name})
    fit = dict(
        source_run=str(source),
        control_run=str(control),
        source_generation_sha256=file_hash(source / "generation.json"),
        control_generation_sha256=file_hash(control / "generation.json"),
        control_checkpoint="continued_supervised/step-400.pt",
        training_seed=123000,
        evaluation_seed=133000,
        bank_sampling_seed=125000,
        training_steps=400,
        refresh_every=4,
        shared_batch={
            name: 1 for name in ("frozen_tb", "lastblock_tb", "prior", "replay")
        },
        evaluation_samples_per_prompt=2,
        penalty=2.5,
    )
    config = dict(steps=400, samples_per_prompt=2, length=80, prefix_length=16)
    references = [dict(id="prompt", role="prompt", sequence="A" * 80)]
    provenance = dict(files={}, packages={}, fixture=False)
    reward, bank = AcceptanceRewardConfig(), [{"id": "train", "split": "train"}]
    for name, value in (
        ("config.json", fit),
        ("pilot_config.json", config),
        ("references.json", references),
        ("provenance.json", provenance),
        ("reward.json", asdict(reward)),
        (
            "reward_definition.json",
            dict(version="continuous_acceptance_v1", penalty=2.5),
        ),
        ("bank.json", bank),
        ("status.json", dict(status="complete", fixture=False)),
    ):
        write_json(pilot / name, value)
    for folder, filename, snapshot in (
        (source, "initial.pt", "initial.pt"),
        (source, "initial_training.jsonl", "initial_training.jsonl"),
        (control, fit["control_checkpoint"], "supervised.pt"),
        (control, "continued_supervised/training.jsonl", "supervised_training.jsonl"),
    ):
        (folder / filename).write_text(snapshot)
        (pilot / snapshot).write_bytes((folder / filename).read_bytes())
    for folder, name in (
        (source, "source_generation.json"),
        (control, "control_generation.json"),
    ):
        (pilot / name).write_bytes((folder / "generation.json").read_bytes())
    (pilot / "samples.jsonl").write_text("{}\n" * 8)
    settings = dict(
        pilot_run=str(pilot),
        replicates=[
            dict(
                name=f"replicate-{index}",
                training_seed=base + 3000,
                evaluation_seed=base + 13000,
                bank_sampling_seed=base + 5000,
            )
            for index, base in ((1, 220000), (2, 320000))
        ],
    )
    refresh_manifest(pilot, settings)
    inputs = (source, control, config, references, provenance, reward, bank, {}, {}, {})
    return settings, fit, inputs


class ConfirmationInputsTests(unittest.TestCase):
    def test_only_seeds_change_and_large_checkpoint_is_hashed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, template, inputs = input_fixture(Path(directory))
            with (
                patch(f"{MODULE}.check_lastblock_inputs", return_value=inputs) as check,
                patch(f"{MODULE}.verify_implementation") as implementation,
            ):
                pilot, replicas, config, references, provenance = (
                    check_confirmation_inputs(settings)
                )
            self.assertEqual(pilot, Path(settings["pilot_run"]))
            self.assertEqual(
                (config, references, provenance), (inputs[2], inputs[3], inputs[4])
            )
            self.assertEqual(check.call_count, 2)
            self.assertEqual(
                [call.kwargs["check_checkpoint"] for call in check.call_args_list],
                [True, False],
            )
            implementation.assert_called_once()
            for index, (name, fit) in enumerate(replicas):
                self.assertEqual(name, settings["replicates"][index]["name"])
                self.assertEqual(
                    {k: v for k, v in fit.items() if k not in SEED_KEYS},
                    {k: v for k, v in template.items() if k not in SEED_KEYS},
                )
                self.assertEqual(
                    {k: fit[k] for k in SEED_KEYS},
                    {k: settings["replicates"][index][k] for k in SEED_KEYS},
                )
            replicas[0][1]["shared_batch"]["prior"] = 99
            self.assertEqual(replicas[1][1]["shared_batch"]["prior"], 1)

    def test_rejects_hyperparameter_overrides_and_unsafe_replicate_names(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, _ = input_fixture(Path(directory))
            bad_settings = [
                dict(settings, training_steps=800),
                dict(settings, replicates=[]),
            ]
            for override in (
                {"name": "../escape"},
                {"name": "replicate-2"},
                {"penalty": 3},
                {"training_seed": True},
                {"evaluation_seed": -1},
                {"training_seed": MAX_SEED + 1},
            ):
                bad = copy.deepcopy(settings)
                bad["replicates"][0].update(override)
                bad_settings.append(bad)
            for bad in bad_settings:
                with self.subTest(settings=bad), self.assertRaises(ValueError):
                    check_confirmation_inputs(bad)

    def test_rejects_historical_and_cross_replicate_stream_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, inputs = input_fixture(Path(directory))
            for index, key, value in (
                (0, "training_seed", 123000),
                (0, "evaluation_seed", 123001),
                (0, "bank_sampling_seed", 95000),
                (0, "evaluation_seed", 223002),
                (1, "training_seed", 222999),
                (1, "evaluation_seed", 233000),
            ):
                bad = copy.deepcopy(settings)
                bad["replicates"][index][key] = value
                with (
                    self.subTest(index=index, key=key, value=value),
                    patch(f"{MODULE}.check_lastblock_inputs", return_value=inputs),
                    patch(f"{MODULE}.verify_implementation"),
                    self.assertRaisesRegex(ValueError, "fresh and disjoint"),
                ):
                    check_confirmation_inputs(bad)

    def test_requires_completed_protected_nonfixture_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, inputs = input_fixture(Path(directory))
            pilot = Path(settings["pilot_run"])
            with (
                patch(f"{MODULE}.check_lastblock_inputs", return_value=inputs),
                patch(f"{MODULE}.verify_implementation"),
            ):
                write_json(pilot / "status.json", dict(status="running"))
                with self.assertRaisesRegex(ValueError, "not complete"):
                    check_confirmation_inputs(settings)
                write_json(pilot / "status.json", dict(status="complete", fixture=True))
                with self.assertRaisesRegex(ValueError, "fixture pilot"):
                    check_confirmation_inputs(settings)
                check_confirmation_inputs(settings, check_checkpoint=False)
                write_json(
                    pilot / "status.json", dict(status="complete", fixture=False)
                )
                refresh_manifest(pilot, settings, omit="initial.pt")
                with self.assertRaisesRegex(ValueError, "not protected"):
                    check_confirmation_inputs(settings)

    def test_rejects_changed_initialization_even_with_updated_pilot_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, inputs = input_fixture(Path(directory))
            pilot = Path(settings["pilot_run"])
            (pilot / "initial.pt").write_text("would warm-start from adapted pilot")
            refresh_manifest(pilot, settings)
            with (
                patch(f"{MODULE}.check_lastblock_inputs", return_value=inputs),
                patch(f"{MODULE}.verify_implementation"),
                self.assertRaisesRegex(ValueError, "snapshot differs: initial.pt"),
            ):
                check_confirmation_inputs(settings)

    def test_rejects_changed_support_even_with_updated_pilot_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, inputs = input_fixture(Path(directory))
            pilot = Path(settings["pilot_run"])
            config = json.loads((pilot / "pilot_config.json").read_text())
            write_json(pilot / "pilot_config.json", dict(config, length=120))
            refresh_manifest(pilot, settings)
            with (
                patch(f"{MODULE}.check_lastblock_inputs", return_value=inputs),
                patch(f"{MODULE}.verify_implementation"),
                self.assertRaisesRegex(ValueError, "support/configuration"),
            ):
                check_confirmation_inputs(settings)

    def test_preflight_counts_all_five_arms_without_model_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            settings, _, inputs = input_fixture(Path(directory))
            with (
                patch(f"{MODULE}.check_lastblock_inputs", return_value=inputs),
                patch(f"{MODULE}.verify_implementation"),
            ):
                result = confirmation_preflight(settings)
            self.assertEqual(result["total_final_evaluation_samples"], 20)
            self.assertEqual(result["updates_per_training_arm"], 400)
            self.assertEqual(result["fresh_training_trajectories_per_replicate"], 300)
            self.assertEqual(result["trajectory_presentations_per_training_arm"], 1600)
            self.assertFalse(result["cuda_initialized"])
            self.assertFalse(result["model_or_scorer_executed"])


if __name__ == "__main__":
    unittest.main()
