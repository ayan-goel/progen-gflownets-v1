"""CPU controls for a branch-only rate change and immutable reused comparators."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_idr_charge import configuration

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.idr.branch_rate import (
    branch_optimizer,
    check_rate_inputs,
    check_shared_controls,
    reevaluate_rate,
    run_rate,
    set_rates,
    train_rate_head,
)
from stability_gfn.idr.charge_experiment import run_charge
from stability_gfn.idr.context_experiment import (
    make_context_head,
    run_context,
    train_head,
)


def fixture_predictor(sequences, version):
    return [np.full(len(sequence), 0.8) for sequence in sequences]


class BranchRateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny(11)
        cls.prompts = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKQEKAG"},
            {"id": "fixture2", "role": "prompt", "sequence": "AQPSEKGS"},
        ]

    def test_only_branch_update_is_scaled_and_parameters_are_not_duplicated(self):
        config = configuration()
        heads = [
            make_context_head(self.backbone, 2, config, 0, "direct_context")
            for _ in range(2)
        ]
        normalizers = [torch.nn.Parameter(torch.zeros(2)) for _ in range(2)]
        before = {name: value.clone() for name, value in heads[0].state_dict().items()}
        for multiplier, head, log_z in zip((1, 10), heads, normalizers):
            optimizer = branch_optimizer(head, log_z, config, multiplier)
            parameters = [p for g in optimizer.param_groups for p in g["params"]]
            self.assertEqual(len(parameters), len({id(p) for p in parameters}))
            self.assertEqual(
                {id(p) for p in parameters},
                {id(p) for p in [*head.parameters(), log_z]},
            )
            for parameter in parameters:
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        for name, value in heads[0].state_dict().items():
            if name == "task_weight":
                torch.testing.assert_close(
                    heads[1].state_dict()[name] - before[name],
                    10 * (value - before[name]),
                )
            else:
                self.assertTrue(torch.equal(value, heads[1].state_dict()[name]), name)
        self.assertTrue(torch.equal(normalizers[0], normalizers[1]))

    def test_decay_keeps_each_parameter_group_at_its_declared_rate(self):
        config = configuration()
        config["steps"] = 400
        head = make_context_head(self.backbone, 2, config, 0, "direct_context")
        optimizer = branch_optimizer(
            head, torch.nn.Parameter(torch.zeros(2)), config, 10
        )
        for step, rates in [
            (1, [1e-4, 1e-3, 1e-2]),
            (300, [1e-4, 1e-3, 1e-2]),
            (301, [1e-5, 1e-4, 1e-3]),
            (400, [1e-5, 1e-4, 1e-3]),
        ]:
            set_rates(optimizer, config, 10, step)
            for group, expected in zip(optimizer.param_groups, rates):
                self.assertAlmostEqual(group["lr"], expected)

    def test_control_check_tolerates_roundoff_but_not_changed_success(self):
        saved = {
            "arms": {
                arm: {
                    "count": 384,
                    "charge_success_rate": 0.125,
                    "v3_mean_context_tail_disorder": 0.7,
                }
                for arm in ("prior", "prior_pool", "select", "swap")
            }
        }
        current = copy.deepcopy(saved)
        current["arms"]["prior"]["v3_mean_context_tail_disorder"] += 1e-8
        check_shared_controls(current, saved)
        current["arms"]["prior"]["charge_success_rate"] += 1 / 384
        with self.assertRaises(ValueError):
            check_shared_controls(current, saved)
        current = copy.deepcopy(saved)
        current["arms"]["prior"]["v3_mean_context_tail_disorder"] += 0.01
        with self.assertRaises(ValueError):
            check_shared_controls(current, saved)

    def test_unit_multiplier_reproduces_original_training(self):
        with tempfile.TemporaryDirectory() as temp:
            original, separate = Path(temp) / "original", Path(temp) / "separate"
            original.mkdir()
            separate.mkdir()
            config = configuration()
            head, log_z, _ = train_head(
                self.backbone, self.prompts, config, 0, "direct_context", original
            )
            other, other_z, _ = train_rate_head(
                self.backbone, self.prompts, config, 0, 1, separate
            )
            for key, value in head.state_dict().items():
                self.assertTrue(torch.equal(value, other.state_dict()[key]), key)
            self.assertTrue(torch.equal(log_z, other_z))

    def test_tiny_run_reuses_controls_and_replays_without_source(self):
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
            before = {
                key: value.clone()
                for key, value in self.backbone.model.state_dict().items()
            }
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained model"),
            ):
                run_charge(
                    config, root / "pilot", "cpu", self.backbone, fixture_predictor
                )
                context_settings = {
                    "source_run": str(root / "pilot"),
                    "source_config_sha256": file_hash(root / "pilot/config.json"),
                    "arms": ["standard", "direct_context"],
                }
                run_context(
                    context_settings,
                    root / "control",
                    "cpu",
                    self.backbone,
                    fixture_predictor,
                )
                names = [
                    "generation.json",
                    "standard/metrics.json",
                    "direct_context/metrics.json",
                ]
                settings = {
                    "source_run": str(root / "control"),
                    "source_files": {n: file_hash(root / "control" / n) for n in names},
                    "branch_rate_multiplier": 10,
                }
                source_before = {
                    str(p): file_hash(p)
                    for p in (root / "control").rglob("*")
                    if p.is_file()
                }
                check_rate_inputs(settings, False)
                run_rate(
                    settings, root / "rate", "cpu", self.backbone, fixture_predictor
                )
                for path, digest in source_before.items():
                    self.assertEqual(file_hash(Path(path)), digest)
                with self.assertRaises(FileExistsError):
                    run_rate(
                        settings, root / "rate", "cpu", self.backbone, fixture_predictor
                    )
                (root / "control").rename(root / "hidden-control")
                reevaluate_rate(root / "rate", root / "replay", fixture_predictor)
            self.assertEqual(
                json.loads((root / "rate/comparison.json").read_text()),
                json.loads((root / "replay/comparison.json").read_text()),
            )
            self.assertEqual(
                json.loads((root / "rate/status.json").read_text())["status"],
                "complete",
            )
            samples = [
                json.loads(s)
                for s in (root / "rate/samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(samples), 12)
            self.assertEqual(len({r["id"] for r in samples}), 12)
            costs = json.loads((root / "rate/costs.json").read_text())["0"]
            self.assertEqual(costs["prior_pool"]["trajectories"], 0)
            self.assertEqual(costs["prior_pool"]["reused_trajectories"], 8)
            self.assertEqual(costs["training"]["trajectories"], 6)
            for key, value in self.backbone.model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))
            (root / "rate/samples.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_rate(root / "rate", root / "bad", fixture_predictor)
            (root / "hidden-control").rename(root / "control")
            (root / "control/direct_context/metrics.json").write_text("{}\n")
            with self.assertRaises(ValueError):
                check_rate_inputs(settings, False)
            self.assertFalse(torch.cuda.is_initialized())
