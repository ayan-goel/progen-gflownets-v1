"""Independent finite-tree checks for the saved-state IDR diagnostic."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stability_gfn.idr.context_experiment import DirectContextHead

spec = importlib.util.spec_from_file_location(
    "idr_exact_diagnostic",
    Path(__file__).resolve().parents[1] / "scripts/diagnose_idr_exact_cpu.py",
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def fixture():
    tails = ["KA", "KE", "AA"]
    alphabet = diagnostic.ALPHABET
    features = torch.randn(3, 4, generator=torch.Generator().manual_seed(4))
    cache = {
        "features": torch.stack([features[[0, 1]], features[[0, 1]], features[[0, 2]]]),
        "context": torch.zeros(3, 2, 6),
        "logits": torch.full((3, 2, 20), -100.0),
    }
    for leaf in range(3):
        for token, probability in {"A": 0.5, "K": 0.4, "E": 0.1}.items():
            cache["logits"][leaf, 0, alphabet.index(token)] = torch.tensor(
                probability
            ).log()
        distribution = (
            {"A": 0.1, "K": 0.1, "E": 0.8}
            if leaf < 2
            else {"A": 0.01, "K": 0.98, "E": 0.01}
        )
        for token, probability in distribution.items():
            cache["logits"][leaf, 1, alphabet.index(token)] = torch.tensor(
                probability
            ).log()
    rows = [
        dict(tail=tail, joint_pass=i != 1, split="train", prompt_id="test")
        for i, tail in enumerate(tails)
    ]
    return rows, cache


class ExactDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_local_normalization_oracle_and_conditioning_distinction(self):
        rows, cache = fixture()
        tree = diagnostic.FiniteBankTree(rows, cache)
        expected = torch.tensor([4 / 81, 32 / 81, 5 / 9], dtype=torch.float64)
        torch.testing.assert_close(tree.log_prior.exp(), expected, atol=1e-7, rtol=1e-6)
        global_conditioned = torch.tensor([0.04, 0.32, 0.005], dtype=torch.float64)
        global_conditioned /= global_conditioned.sum()
        self.assertGreater(float((expected - global_conditioned).abs().sum() / 2), 0.5)
        target = (tree.log_prior + torch.tensor([0.0, -2.0, -1.0])).softmax(0)
        torch.testing.assert_close(tree.oracle_log_probabilities(target).exp(), target)
        self.assertEqual(tree.summary["branching_nodes"], 2)
        self.assertAlmostEqual(tree.summary["mean_decisions_per_leaf"], 5 / 3)

    def test_shared_prefix_cache_corruption_rejected(self):
        rows, cache = fixture()
        cache["features"][1, 0, 0] += 0.5
        with self.assertRaisesRegex(ValueError, "Inconsistent shared prefix"):
            diagnostic.FiniteBankTree(rows, cache)

    def test_expected_tb_gradient_equals_weighted_per_leaf_gradients(self):
        logits = torch.tensor([0.2, -0.7, 0.4], dtype=torch.float64, requires_grad=True)
        log_q = logits.log_softmax(0)
        reward = torch.tensor([-1.0, -3.0, -2.0], dtype=torch.float64)
        weights = 0.5 * log_q.detach().exp() + torch.tensor([0.25, 0.0, 0.25])
        residuals = log_q - reward
        analytic = torch.autograd.grad(
            (weights * residuals.square()).sum(), logits, retain_graph=True
        )[0]
        enumerated = sum(
            weights[i]
            * torch.autograd.grad(residuals[i].square(), logits, retain_graph=True)[0]
            for i in range(3)
        )
        torch.testing.assert_close(analytic, enumerated)

    def test_all_fit_paths_on_tiny_fixture_preserve_initial_weights(self):
        rows, cache = fixture()
        tree = diagnostic.FiniteBankTree(rows, cache)
        tree.log_reward = tree.log_prior + torch.tensor([0.0, -2.0, 0.0])
        tree.target = tree.log_reward.softmax(0)
        head = DirectContextHead(4, 6, 20, 8)
        initial = {key: value.clone() for key, value in head.state_dict().items()}
        config = dict(
            steps=10,
            learning_rate=0.001,
            normalizer_learning_rate=0.01,
            gradient_clip=1.0,
            decay_fraction=0.75,
            final_lr_factor=0.1,
        )
        for condition in ("exact_teacher", "expected_tb", "sampled_tb"):
            result = diagnostic.fit_condition([tree], head, config, condition)
            self.assertTrue(0 <= result["mean_tv"] <= 1)
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_algebraic_capacity_witness_in_existing_architecture(self):
        capacity_spec = importlib.util.spec_from_file_location(
            "idr_exact_capacity",
            Path(__file__).resolve().parents[1] / "scripts/audit_idr_exact_capacity.py",
        )
        capacity = importlib.util.module_from_spec(capacity_spec)
        with patch.dict(sys.modules, {"diagnose_idr_exact_cpu": diagnostic}):
            capacity_spec.loader.exec_module(capacity)
        rows, cache = fixture()
        tree = diagnostic.FiniteBankTree(rows, cache)
        tree.target = (tree.log_prior + torch.tensor([0.0, -2.0, 0.0])).softmax(0)
        initial = DirectContextHead(4, 6, 20, 128)
        weights = {key: value.clone() for key, value in initial.state_dict().items()}
        constructed, result = capacity.construct_head(initial, [tree])
        self.assertEqual(result["numerical_rank"], 2)
        with torch.no_grad():
            metrics = diagnostic.distribution_metrics(
                tree.log_probabilities(constructed), tree.target, tree.passed
            )
        self.assertLess(metrics["tv"], 1e-5)
        for key, value in initial.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
