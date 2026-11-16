"""Exact tiny-tree checks plus frozen-backbone CPU execution and artifact replay."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from test_idr_branch_rate import fixture_predictor
from test_idr_charge import configuration

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.guidance import trajectory_balance
from stability_gfn.idr.charge_experiment import run_charge
from stability_gfn.idr.context_experiment import run_context
from stability_gfn.idr.subtb import segment_residuals, subtrajectory_balance
from stability_gfn.idr.subtb_experiment import reevaluate_subtb, run_subtb


def tiny_tree():
    """Eight leaves, nonuniform prior and a nonadditive terminal score."""
    actions = torch.tensor(list(itertools.product(range(2), repeat=3)))
    paths = []
    for row in actions.tolist():
        node, path = 0, []
        for action in row:
            path.append(node)
            node = 2 * node + 1 + action
        paths.append(path)
    paths = torch.tensor(paths)
    base_logits = torch.arange(14, dtype=torch.float64).reshape(7, 2) / 8
    base_logits[:, 0] *= -1
    base_logps = base_logits.log_softmax(-1)
    path_base = base_logps[paths, actions]
    score = torch.tensor([-2, 0, -1, 1, -3, 2, -1, 0], dtype=torch.float64)
    log_reward = path_base.sum(-1) + score
    log_flow = [None] * 15
    log_flow[7:] = list(log_reward)
    for node in reversed(range(7)):
        log_flow[node] = torch.logsumexp(
            torch.stack(log_flow[2 * node + 1 : 2 * node + 3]), 0
        )
    log_flow = torch.stack(log_flow)
    return actions, paths, base_logps, path_base, score, log_reward, log_flow


class SubTBMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_flows_make_every_segment_zero_and_preserve_target(self):
        actions, paths, base_logps, path_base, score, log_reward, log_flow = tiny_tree()
        children = 2 * paths + 1 + actions
        log_policy = log_flow[children] - log_flow[paths]
        prefix_prior = torch.cat([torch.zeros(8, 1), path_base.cumsum(1)], 1)
        nodes = torch.cat([paths, children[:, -1:]], 1)
        factored_flow = log_flow[nodes] - prefix_prior
        residuals, lengths = segment_residuals(log_policy - path_base, factored_flow)
        self.assertEqual(residuals.shape, (8, 6))
        self.assertEqual(lengths.tolist(), [1, 2, 3, 1, 2, 1])
        self.assertLess(residuals.abs().max().item(), 1e-12)
        torch.testing.assert_close(factored_flow[:, -1], score)
        torch.testing.assert_close(log_policy.sum(1).exp(), log_reward.softmax(0))
        for weighting in (0.5, 1, 2):
            self.assertLess(
                subtrajectory_balance(
                    log_policy - path_base, factored_flow, weighting
                ).item(),
                1e-24,
            )
        # No tilt: normalized prior, root Z=1, all prior-factored flows equal zero.
        self.assertEqual(
            subtrajectory_balance(
                torch.zeros_like(path_base), torch.zeros_like(factored_flow)
            ).item(),
            0,
        )

    def test_full_segment_matches_tb_value_and_gradients(self):
        generator = torch.Generator().manual_seed(7)
        ratios = torch.randn(
            2, 4, generator=generator, dtype=torch.float64, requires_grad=True
        )
        root = torch.randn(
            2, generator=generator, dtype=torch.float64, requires_grad=True
        )
        scores = torch.tensor([-2.0, -0.5], dtype=torch.float64)
        flows = torch.cat([root[:, None], torch.zeros(2, 3), 2 * scores[:, None]], 1)
        residuals, lengths = segment_residuals(ratios, flows)
        full = residuals[:, lengths == 4].square().mean()
        expected = trajectory_balance(ratios.sum(1), torch.zeros(2), scores, root, 2)
        torch.testing.assert_close(full, expected)
        first = torch.autograd.grad(full, (ratios, root), retain_graph=True)
        second = torch.autograd.grad(expected, (ratios, root))
        for actual, target in zip(first, second):
            torch.testing.assert_close(actual, target)
        for weighting in (0.5, 1.0, 3.0):
            weights = weighting ** lengths.to(torch.float64)
            manual = (residuals.square() * weights / weights.sum()).sum(1).mean()
            torch.testing.assert_close(
                subtrajectory_balance(ratios, flows, weighting), manual
            )

    def test_tiny_exact_optimization_learns_nonuniform_target(self):
        actions, paths, base_logps, path_base, scores, log_reward, _ = tiny_tree()
        correction = torch.nn.Parameter(torch.zeros_like(base_logps))
        flows = torch.nn.Parameter(torch.zeros(7, dtype=torch.float64))
        optimizer = torch.optim.Adam([correction, flows], lr=0.05)
        for _ in range(300):
            log_policy = (base_logps + correction).log_softmax(-1)[paths, actions]
            path_flows = torch.cat([flows[paths], scores[:, None]], 1)
            loss = subtrajectory_balance(log_policy - path_base, path_flows)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probabilities = (
                (base_logps + correction).log_softmax(-1)[paths, actions].sum(1).exp()
            )
            tv = 0.5 * (probabilities - log_reward.softmax(0)).abs().sum()
        self.assertLess(tv.item(), 0.005)
        self.assertLess(loss.item(), 1e-5)


class SubTBRunTests(unittest.TestCase):
    def test_frozen_tiny_run_matched_initialization_counts_rates_and_replay(self):
        torch.set_num_threads(1)
        backbone = FrozenProGen.tiny(11)
        prompts = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKQEKAG"},
            {"id": "fixture2", "role": "prompt", "sequence": "AQPSEKGS"},
        ]
        before = {
            key: value.clone() for key, value in backbone.model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            write_json(data / "references.json", prompts)
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
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No pretrained models"),
            ):
                run_charge(config, root / "pilot", "cpu", backbone, fixture_predictor)
                run_context(
                    {
                        "source_run": str(root / "pilot"),
                        "source_config_sha256": file_hash(root / "pilot/config.json"),
                        "arms": ["standard", "direct_context"],
                    },
                    root / "control",
                    "cpu",
                    backbone,
                    fixture_predictor,
                )
                names = (
                    "generation.json",
                    "standard/metrics.json",
                    "direct_context/metrics.json",
                )
                settings = {
                    "source_run": str(root / "control"),
                    "source_files": {
                        name: file_hash(root / "control" / name) for name in names
                    },
                    "branch_rate_multiplier": 10,
                    "training_steps": 3,
                    "evaluation_samples_per_prompt": 2,
                    "subtb_lambda": 1.0,
                    "flow_learning_rate": 0.001,
                }
                original_hashes = {
                    str(p): file_hash(p)
                    for p in (root / "control").rglob("*")
                    if p.is_file()
                }
                run_subtb(settings, root / "run", "cpu", backbone, fixture_predictor)
                for filename, digest in original_hashes.items():
                    self.assertEqual(file_hash(Path(filename)), digest)
                with self.assertRaises(FileExistsError):
                    run_subtb(
                        settings, root / "run", "cpu", backbone, fixture_predictor
                    )
                (root / "control").rename(root / "hidden-control")
                reevaluate_subtb(root / "run", root / "replay", fixture_predictor)
            for name in (
                "comparison.json",
                "training_curves.json",
                "tb/sequence_metrics.json",
                "subtb/sequence_metrics.json",
            ):
                self.assertEqual(
                    json.loads((root / "run" / name).read_text()),
                    json.loads((root / "replay" / name).read_text()),
                )
            rows = [
                json.loads(line)
                for line in (root / "run/samples.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 16)
            self.assertEqual(len({row["id"] for row in rows}), 16)
            costs = json.loads((root / "run/costs.json").read_text())
            logs = {}
            for method in ("tb", "subtb"):
                self.assertEqual(costs[method]["0"]["training"]["trajectories"], 6)
                logs[method] = [
                    json.loads(line)
                    for line in (root / f"run/{method}/seed-0/training.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(logs[method][-1]["step"], 3)
                self.assertAlmostEqual(logs[method][-1]["rates"]["task_branch"], 1e-4)
            self.assertEqual(
                logs["tb"][0]["mean_score"], logs["subtb"][0]["mean_score"]
            )
            self.assertAlmostEqual(
                logs["tb"][0]["tb_loss"], logs["subtb"][0]["tb_loss"], places=4
            )
            self.assertAlmostEqual(logs["subtb"][-1]["rates"]["flow"], 1e-4)
            self.assertGreater(logs["subtb"][0]["flow_gradient_norm_before_clip"], 0)
            self.assertEqual(
                costs["tb"]["0"]["training"]["backbone_work"],
                costs["subtb"]["0"]["training"]["backbone_work"],
            )
            checkpoint = torch.load(
                root / "run/subtb/seed-0/step-3.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertGreater(checkpoint["flow"]["output.weight"].norm().item(), 0)
            self.assertGreater(checkpoint["head"]["task_weight"].norm().item(), 0)
            (root / "run/samples.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_subtb(root / "run", root / "bad", fixture_predictor)
        for key, value in backbone.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
            self.assertIsNone(
                backbone.model.get_parameter(key).grad
                if key in dict(backbone.model.named_parameters())
                else None
            )
        self.assertFalse(torch.cuda.is_initialized())
