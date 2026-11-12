"""Small CPU tests for open-sequence TB, reward semantics and comparison integrity."""

import itertools
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.guidance import trajectory_balance
from stability_gfn.idr.charge import (
    charge_reward,
    context_features,
    make_head,
    path_log_probabilities,
    rollout,
    update_head,
)
from stability_gfn.idr.charge_experiment import (
    check_charge_inputs,
    derive_comparators,
    reevaluate_charge,
    run_charge,
)
from stability_gfn.idr.data import ALPHABET


def configuration():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/idr-charge.json").read_text())
    config.update(
        length=8,
        prefix_length=4,
        batch_size=2,
        samples_per_prompt=2,
        seeds=[0],
        steps=3,
        log_every=1,
        checkpoint_every=3,
        selection_factor=2,
        swap_proposals=4,
        hidden_size=16,
    )
    return config


class ChargeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.backbone = FrozenProGen.tiny(11)
        cls.prompts = [
            {"id": "fixture1", "role": "prompt", "sequence": "GSKQEKAG"},
            {"id": "fixture2", "role": "prompt", "sequence": "AQPSEKGS"},
        ]

    def test_reward_is_order_sensitive_and_independently_calculable(self):
        reward = configuration()["reward"]
        first = "K" * 6 + "E" * 2 + "A" * 24
        second = "K" * 2 + "E" * 6 + "A" * 24
        tails = [first + second, second + first, "A" * 64]
        score, blocks, fcr, passes = charge_reward(tails, reward)
        self.assertEqual(score.tolist(), [0.0, -4.0, -2.0])
        self.assertEqual(blocks[0].tolist(), [0.125, -0.125])
        self.assertEqual(fcr[0].item(), 0.25)
        self.assertEqual(passes.tolist(), [True, False, False])
        self.assertEqual(Counter(tails[0]), Counter(tails[1]))
        with self.assertRaises(ValueError):
            charge_reward(["AX"], reward)

    def test_rollout_likelihoods_match_independent_teacher_forcing(self):
        config = configuration()
        head = make_head(self.backbone, 2, 16)
        with torch.no_grad():
            head.output.weight.fill_(0.03)
            head.output.bias.copy_(torch.linspace(-0.2, 0.2, 20))
        batch = rollout(
            self.backbone,
            head,
            self.prompts,
            [0, 1],
            config,
            torch.Generator().manual_seed(8),
            retain=True,
        )
        cached = path_log_probabilities(head, batch)
        torch.testing.assert_close(
            cached.detach(), batch["log_policy"], atol=2e-6, rtol=1e-6
        )
        states, logits = self.backbone.forward(batch["sequences"])
        ids = self.backbone.residue_ids(ALPHABET)
        expected_q = torch.zeros(2)
        expected_p = torch.zeros(2)
        for position in range(4, 8):
            context = context_features(
                [s[4:position] for s in batch["sequences"]],
                [0, 1],
                2,
                4,
                config["reward"]["block_targets"],
                "cpu",
            )
            base = logits[:, position, ids]
            policy = base + head(states[:, position], context)
            actions = torch.tensor(
                [ALPHABET.index(s[position]) for s in batch["sequences"]]
            )[:, None]
            expected_q += policy.log_softmax(-1).gather(1, actions).squeeze(1).detach()
            expected_p += base.log_softmax(-1).gather(1, actions).squeeze(1)
        torch.testing.assert_close(cached.detach(), expected_q, atol=3e-6, rtol=1e-6)
        torch.testing.assert_close(batch["log_prior"], expected_p, atol=3e-6, rtol=1e-6)
        self.assertFalse(batch["features"].requires_grad)

    def test_zero_beta_recovers_prior_and_backbone_stays_frozen(self):
        config = configuration()
        config["beta"] = 0.0
        head = make_head(self.backbone, 2, 16)
        normalizer = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([*head.parameters(), normalizer], lr=1e-3)
        before = {k: v.clone() for k, v in head.state_dict().items()}
        batch = rollout(
            self.backbone,
            head,
            self.prompts,
            [0, 1],
            config,
            torch.Generator().manual_seed(9),
            retain=True,
        )
        self.assertTrue(torch.equal(batch["log_prior"], batch["log_policy"]))
        stats = update_head(head, normalizer, optimizer, batch, [0, 1], config)
        self.assertLess(stats["tb_loss"], 1e-10)
        # Recomputed reductions may differ slightly, so check numerical near-identity.
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=1e-3, rtol=0)
        self.assertTrue(
            all(
                p.grad is None and not p.requires_grad
                for p in self.backbone.model.parameters()
            )
        )

    def test_exact_two_residue_target_satisfies_tb(self):
        tails = ["".join(t) for t in itertools.product(ALPHABET, repeat=2)]
        sequences = ["GSKQ" + t for t in tails]
        _, logits = self.backbone.forward(sequences)
        ids = self.backbone.residue_ids(ALPHABET)
        log_prior = torch.zeros(len(tails))
        for index in range(2):
            action = torch.tensor([ALPHABET.index(t[index]) for t in tails])[:, None]
            log_prior += (
                logits[:, 4 + index, ids].log_softmax(-1).gather(1, action).squeeze(1)
            )
        self.assertAlmostEqual(log_prior.exp().sum().item(), 1.0, places=5)
        config = configuration()
        scores = charge_reward(tails, config["reward"])[0]
        weights = log_prior.double() + config["beta"] * scores.double()
        log_z = weights.logsumexp(0)
        loss = trajectory_balance(
            weights - log_z, log_prior.double(), scores.double(), log_z, config["beta"]
        )
        self.assertLess(loss.item(), 1e-20)

    def test_tiny_run_recovery_selection_and_editing(self):
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
            check_charge_inputs(config, False)
            backbone_before = {
                k: v.clone() for k, v in self.backbone.model.state_dict().items()
            }
            with patch.object(
                FrozenProGen,
                "from_checkpoint",
                side_effect=AssertionError("No real weights"),
            ):
                run_charge(config, root / "run", "cpu", backbone=self.backbone)
                reevaluate_charge(root / "run", root / "evaluation")

            def read(name):
                return json.loads((root / "run" / name).read_text())

            self.assertEqual(read("status.json")["status"], "complete")
            self.assertTrue(read("status.json")["fixture"])
            self.assertEqual(
                read("metrics.json"),
                json.loads((root / "evaluation/metrics.json").read_text()),
            )
            samples = [
                json.loads(line)
                for line in (root / "run/samples.jsonl").read_text().splitlines()
            ]
            derived = derive_comparators(samples, config)
            by_id = {r["id"]: r for r in derived}
            self.assertEqual(len(by_id), len(derived))
            for row in derived:
                if row["arm"] in {"swap", "shuffle"}:
                    source = by_id[row["source_id"]]
                    self.assertEqual(
                        Counter(row["sequence"][4:]), Counter(source["sequence"][4:])
                    )
                    self.assertEqual(row["sequence"][:4], source["sequence"][:4])
                    self.assertNotIn("log_prior", row)
                    if row["arm"] == "swap":
                        scores = charge_reward(
                            [row["sequence"][4:], source["sequence"][4:]],
                            config["reward"],
                        )[0]
                        self.assertGreaterEqual(float(scores[0]), float(scores[1]))
            for prompt in self.prompts:
                pool = [
                    r
                    for r in derived
                    if r["arm"] == "prior_pool" and r["prompt_id"] == prompt["id"]
                ]
                expected = sorted(
                    pool,
                    key=lambda r: -float(
                        charge_reward([r["sequence"][4:]], config["reward"])[0][0]
                    ),
                )[:2]
                actual = [
                    r["source_id"]
                    for r in derived
                    if r["arm"] == "select" and r["prompt_id"] == prompt["id"]
                ]
                self.assertEqual(actual, [r["id"] for r in expected])
            checkpoint = torch.load(
                root / "run/seed-0/step-3.pt", map_location="cpu", weights_only=True
            )
            self.assertGreater(
                checkpoint["head"]["output.weight"].abs().sum().item(), 0
            )
            for key, value in self.backbone.model.state_dict().items():
                self.assertTrue(torch.equal(value, backbone_before[key]))
            with self.assertRaises(FileExistsError):
                run_charge(config, root / "run", "cpu", backbone=self.backbone)
            (root / "run/samples.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                reevaluate_charge(root / "run", root / "bad")
            self.assertFalse(torch.cuda.is_initialized())

    def test_invalid_reward_and_batch_fail_preflight(self):
        config = configuration()
        # Input checking is isolated here; the real-data check is exercised by CLI.
        with patch(
            "stability_gfn.idr.charge_experiment.check_inputs",
            return_value=(self.prompts, {}),
        ):
            config["reward"]["fcr_target"] = 0.01
            with self.assertRaises(ValueError):
                check_charge_inputs(config, False)
            config = configuration()
            config["batch_size"] = 1
            with self.assertRaises(ValueError):
                check_charge_inputs(config, False)
