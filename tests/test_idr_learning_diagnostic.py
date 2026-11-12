"""CPU fixtures for real-state diagnostic capture, gradient averaging and replay."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.charge import (
    charge_reward,
    make_head,
    path_log_probabilities,
    rollout,
    update_head,
)
from stability_gfn.idr.learning_diagnostic import (
    batch_gradients,
    behavior_probe,
    check_source,
    gradient_geometry,
    optimizer_for,
    replay_learning_diagnostic,
    run_learning_diagnostic,
)


def config_fixture():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/idr-charge.json").read_text())
    config.update(
        length=8,
        prefix_length=4,
        batch_size=2,
        samples_per_prompt=2,
        seeds=[0],
        steps=1,
        hidden_size=16,
    )
    return config


class LearningDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny(23)
        cls.prompts = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKQEKAG"},
            {"id": "fixture2", "role": "prompt", "sequence": "AQPSEKGS"},
        ]

    def test_averaged_gradients_equal_combined_balanced_batch(self):
        config = config_fixture()
        head = make_head(self.backbone, 2, 16)
        log_z = torch.nn.Parameter(torch.zeros(2))
        batches = [
            rollout(
                self.backbone,
                head,
                self.prompts,
                [0, 1],
                config,
                torch.Generator().manual_seed(s),
                retain=True,
            )
            for s in (11, 12)
        ]
        gradients = [batch_gradients(head, log_z, batch, config) for batch in batches]
        combined = {
            key: torch.cat([b[key] for b in batches])
            if isinstance(batches[0][key], torch.Tensor)
            else [item for b in batches for item in b[key]]
            for key in batches[0]
        }
        scores = charge_reward(
            [s[4:] for s in combined["sequences"]], config["reward"]
        )[0]
        residual = (
            log_z.repeat(2)
            + path_log_probabilities(head, combined)
            - combined["log_prior"]
            - config["beta"] * scores
        )
        exact = torch.autograd.grad(
            residual.square().mean(), [*head.parameters(), log_z]
        )
        expected = torch.cat([g.flatten() for g in exact[:-1]])
        torch.testing.assert_close((gradients[0][0] + gradients[1][0]) / 2, expected)
        torch.testing.assert_close((gradients[0][1] + gradients[1][1]) / 2, exact[-1])
        for _, _, stats in gradients:
            self.assertAlmostEqual(
                stats["post_clip_head_norm"],
                stats["head_norm"] * stats["clip_multiplier"],
            )

    def test_zero_output_has_no_token_or_context_effect(self):
        config = config_fixture()
        head = make_head(self.backbone, 2, 16)
        batch = rollout(
            self.backbone,
            head,
            self.prompts,
            [0, 1],
            config,
            torch.Generator().manual_seed(2),
            retain=True,
        )
        stats = behavior_probe(head, batch)
        self.assertEqual(stats["first_half_expected_charge_change"], [0, 0])
        self.assertEqual(stats["second_half_expected_charge_change"], [0, 0])
        self.assertEqual(stats["counterfactual_max_centered_logit_contrast"], 0)
        self.assertEqual(stats["mean_token_kl_to_prior"], 0)

    def test_geometry_uses_separate_reference_half(self):
        vectors = torch.tensor(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result = gradient_geometry(vectors, [1, 2])
        self.assertEqual(result["groups"]["1"]["cosines_to_reference"], [0.0] * 4)
        self.assertEqual(result["groups"]["2"]["cosines_to_reference"], [None, None])
        self.assertEqual(result["reference_mean_norm"], 1.0)

    def source_fixture(self, root):
        config = config_fixture()
        data, source = root / "data", root / "source"
        data.mkdir()
        (source / "seed-0").mkdir(parents=True)
        write_json(data / "references.json", self.prompts)
        write_json(
            data / "manifest.json",
            {
                "count": 2,
                "prompt_count": 2,
                "references_sha256": file_hash(data / "references.json"),
            },
        )
        config.update(data=str(data), manifest_sha256=file_hash(data / "manifest.json"))
        write_json(source / "config.json", config)
        write_json(source / "references.json", self.prompts)
        write_json(source / "status.json", {"status": "complete", "fixture": True})
        write_json(source / "provenance.json", {"files": {}})
        head = make_head(self.backbone, 2, 16)
        log_z = torch.nn.Parameter(torch.zeros(2))
        optimizer = optimizer_for(head, log_z, config)
        batch = rollout(
            self.backbone,
            head,
            self.prompts,
            [0, 1],
            config,
            torch.Generator().manual_seed(13),
            retain=True,
        )
        update_head(head, log_z, optimizer, batch, [0, 1], config)
        checkpoint = source / "seed-0/step-1.pt"
        torch.save(
            {
                "head": head.state_dict(),
                "log_z": log_z.detach(),
                "optimizer": optimizer.state_dict(),
                "step": 1,
                "prompt_ids": [p["id"] for p in self.prompts],
                "config": config,
            },
            checkpoint,
        )
        return {
            "source_run": str(source),
            "source_config_sha256": file_hash(source / "config.json"),
            "checkpoint_step": 1,
            "checkpoint_sha256": {"0": file_hash(checkpoint)},
            "seeds": [0],
            "batches_per_case": 8,
            "gradient_group_sizes": [1, 2],
            "capture_seed_base": 80000,
        }

    def test_tiny_capture_replay_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = self.source_fixture(root)
            source = Path(settings["source_run"])
            hashes = {str(p): file_hash(p) for p in source.rglob("*") if p.is_file()}
            before = {k: v.clone() for k, v in self.backbone.model.state_dict().items()}
            check_source(settings, False)
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real model"),
            ):
                run_learning_diagnostic(settings, root / "run", "cpu", self.backbone)
                replay_learning_diagnostic(root / "run", root / "replay")

            def read(path):
                return json.loads(path.read_text())

            self.assertEqual(
                read(root / "run/metrics.json"), read(root / "replay/metrics.json")
            )
            self.assertEqual(read(root / "run/status.json")["status"], "complete")
            self.assertTrue(read(root / "run/status.json")["fixture"])
            self.assertEqual(read(root / "run/capture.json")["trajectories"], 32)
            for path, digest in hashes.items():
                self.assertEqual(file_hash(Path(path)), digest)
            for name, value in self.backbone.model.state_dict().items():
                self.assertTrue(torch.equal(value, before[name]))
                self.assertIsNone(
                    dict(self.backbone.model.named_parameters()).get(name, value).grad
                )
            with self.assertRaises(FileExistsError):
                run_learning_diagnostic(settings, root / "run", "cpu", self.backbone)
            (root / "run/seed-0-final/batch-00.pt").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                replay_learning_diagnostic(root / "run", root / "corrupt-replay")
            self.assertFalse(torch.cuda.is_initialized())

    def test_source_tampering_and_invalid_group_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            settings = self.source_fixture(Path(temp))
            bad = dict(settings, gradient_group_sizes=[1, 4])
            with self.assertRaises(ValueError):
                check_source(bad, False)
            checkpoint = Path(settings["source_run"]) / "seed-0/step-1.pt"
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                check_source(settings, False)
