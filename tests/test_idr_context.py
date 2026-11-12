"""CPU controls for matched initialization, direct routing and recoverable fits."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_charge import configuration

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.charge import rollout, update_head
from stability_gfn.idr.charge_experiment import run_charge
from stability_gfn.idr.context_experiment import (
    check_context_inputs,
    make_context_head,
    reevaluate_context,
    run_context,
)
from stability_gfn.idr.data import ALPHABET


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny(11)
        cls.prompts = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKQEKAG"},
            {"id": "fixture2", "role": "prompt", "sequence": "AQPSEKGS"},
        ]

    def test_initialization_matches_and_does_not_consume_rng(self):
        before = torch.random.get_rng_state().clone()
        standard = make_context_head(self.backbone, 2, configuration(), 0, "standard")
        direct = make_context_head(
            self.backbone, 2, configuration(), 0, "direct_context"
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for key, value in standard.state_dict().items():
            self.assertTrue(torch.equal(value, direct.state_dict()[key]))
        self.assertEqual(direct.task_weight.numel(), 100)
        features = torch.randn(2, 3, self.backbone.hidden_size)
        context = torch.randn(2, 3, 7)
        self.assertTrue(
            torch.equal(standard(features, context), direct(features, context))
        )
        self.assertEqual(float(direct(features, context).abs().sum()), 0)

    def test_branch_can_steer_opposite_charge_without_feature_changes(self):
        head = make_context_head(self.backbone, 2, configuration(), 0, "direct_context")
        charges = torch.tensor(
            [1 if a in "KR" else -1 if a in "DE" else 0 for a in ALPHABET]
        )
        with torch.no_grad():
            head.task_weight[:, 4] = charges * 4.0
        features = torch.zeros(2, self.backbone.hidden_size)
        context = torch.zeros(2, 7)
        context[:, 4] = torch.tensor([0.125, -0.125])
        probabilities = head(features, context).softmax(-1)
        means = (probabilities * charges).sum(-1)
        self.assertGreater(float(means[0]), 0)
        self.assertLess(float(means[1]), 0)
        # Prompt one-hot values do not leak into the shortcut.
        context[:, 5:] = 1
        torch.testing.assert_close(probabilities, head(features, context).softmax(-1))

    def test_tb_updates_branch_and_keeps_backbone_frozen(self):
        config = configuration()
        head = make_context_head(self.backbone, 2, config, 0, "direct_context")
        log_z = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([*head.parameters(), log_z], lr=1e-4)
        batch = rollout(
            self.backbone,
            head,
            self.prompts,
            [0, 1],
            config,
            torch.Generator().manual_seed(20),
            retain=True,
        )
        update_head(head, log_z, optimizer, batch, [0, 1], config)
        self.assertGreater(float(head.task_weight.grad.norm()), 0)
        self.assertGreater(float(head.task_weight.norm()), 0)
        self.assertTrue(
            all(
                p.grad is None and not p.requires_grad
                for p in self.backbone.model.parameters()
            )
        )

    def test_tiny_paired_run_matches_legacy_and_replays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            write_json(data / "references.json", self.prompts)
            write_json(
                data / "manifest.json",
                {
                    "count": 2,
                    "prompt_count": 2,
                    "references_sha256": file_hash(data / "references.json"),
                },
            )
            config = configuration()
            config.update(
                data=str(data), manifest_sha256=file_hash(data / "manifest.json")
            )
            original_weights = {
                k: v.clone() for k, v in self.backbone.model.state_dict().items()
            }
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real weights"),
            ):
                run_charge(config, root / "legacy", "cpu", backbone=self.backbone)
                settings = {
                    "source_run": str(root / "legacy"),
                    "source_config_sha256": file_hash(root / "legacy/config.json"),
                    "arms": ["standard", "direct_context"],
                }
                check_context_inputs(settings, False)
                run_context(
                    settings, root / "comparison", "cpu", backbone=self.backbone
                )
                reevaluate_context(root / "comparison", root / "replay")
            run = root / "comparison"
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"], "complete"
            )
            self.assertEqual(
                json.loads((run / "metrics.json").read_text()),
                json.loads((root / "replay/metrics.json").read_text()),
            )
            legacy = torch.load(
                root / "legacy/seed-0/step-3.pt", map_location="cpu", weights_only=True
            )
            standard = torch.load(
                run / "standard-seed-0/step-3.pt", map_location="cpu", weights_only=True
            )
            for key, value in legacy["head"].items():
                self.assertTrue(torch.equal(value, standard["head"][key]), key)
            self.assertTrue(torch.equal(legacy["log_z"], standard["log_z"]))
            old_samples = [
                json.loads(s)
                for s in (root / "legacy/samples.jsonl").read_text().splitlines()
            ]
            new_samples = [
                json.loads(s) for s in (run / "samples.jsonl").read_text().splitlines()
            ]
            for row in old_samples:
                expected_id = (
                    row["id"] if row["arm"] == "prior_pool" else f"standard-{row['id']}"
                )
                candidate = next(r for r in new_samples if r["id"] == expected_id)
                for key in row:
                    if key != "id":
                        self.assertEqual(row[key], candidate[key], key)
            self.assertEqual(len(new_samples), 16)  # 8 shared pool + 4 per GFN.
            self.assertEqual(len({r["id"] for r in new_samples}), len(new_samples))
            for key, value in self.backbone.model.state_dict().items():
                self.assertTrue(torch.equal(value, original_weights[key]))
            for arm in settings["arms"]:
                metrics = json.loads((run / arm / "metrics.json").read_text())
                self.assertEqual(metrics["arms"]["prior"]["count"], 4)
                self.assertEqual(metrics["arms"]["gfn"]["count"], 4)
            with self.assertRaises(FileExistsError):
                run_context(settings, run, "cpu", backbone=self.backbone)
            (run / "samples.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_context(run, root / "tampered")
            settings["arms"] = ["direct_context"]
            with self.assertRaises(ValueError):
                check_context_inputs(settings, False)
            self.assertFalse(torch.cuda.is_initialized())
