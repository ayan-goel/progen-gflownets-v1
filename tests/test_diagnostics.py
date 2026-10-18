"""Hand-checkable exact-oracle and diagnostic reporting checks on CPU fixtures."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from stability_gfn.diagnostics import (
    conditional_errors,
    exact_conditionals,
    fit_exact_conditionals,
    inspect_saved_fit,
)
from stability_gfn.environment import PrefixTree, SequenceContext
from stability_gfn.evaluation import load_head
from stability_gfn.guidance import CachedTask
from stability_gfn.metrics import distribution_metrics


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        # An irregular tree includes both forced and branching actions.
        self.context = SequenceContext("AAA", 3, "AG")
        self.tree = PrefixTree(self.context, ["AAA", "AGA", "GAA", "GAG"])
        generator = torch.Generator().manual_seed(3)
        self.task = CachedTask(
            self.tree,
            torch.randn(len(self.tree.prefixes), 8, generator=generator),
            torch.randn(len(self.tree.prefixes), 2, generator=generator),
            [-2, 0, 1, -1],
        )

    def test_oracle_and_chain_rule_on_irregular_tree(self):
        task = self.task
        for beta in (0, 1.3):
            conditional, visitation = exact_conditionals(task, beta)
            log_target = (
                task.base_log_probabilities.double() + beta * task.scores.double()
            ).log_softmax(0)
            torch.testing.assert_close(
                conditional[task.paths, task.actions].sum(-1), log_target
            )
            # Check marginal mass independently by enumerating matching strings.
            for index, prefix in enumerate(task.tree.prefixes):
                mass = sum(
                    log_target[leaf].exp()
                    for leaf, sequence in enumerate(task.tree.sequences)
                    if sequence.startswith(prefix)
                )
                torch.testing.assert_close(visitation[index], mass)
            errors = conditional_errors(task, None, conditional, visitation)
            metrics = distribution_metrics(
                task.base_log_probabilities,
                task.base_log_probabilities,
                task.scores,
                beta,
            )
            self.assertAlmostEqual(
                errors["chain_rule_kl_target_to_q"], metrics["kl_target_to_q"], places=6
            )
            self.assertAlmostEqual(
                sum(row["sequence_kl_contribution"] for row in errors["by_depth"]),
                errors["chain_rule_kl_target_to_q"],
                places=12,
            )
            self.assertGreater(errors["conditional_entropy_floor"], 0)

    def test_logging_audit_and_original_update_parity(self):
        task = self.task
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "source-run"
            run.mkdir()
            (run / "config.json").write_text(
                json.dumps(
                    {
                        "experiment": "exact",
                        "smoke": True,
                        "training": {"beta": 1.0, "hidden_size": 8},
                    }
                )
            )
            torch.save(
                {
                    "context": {
                        "reference": "AAA",
                        "mutation_budget": 3,
                        "alphabet": "AG",
                    },
                    "sequences": task.tree.sequences,
                    "features": task.features,
                    "logits": task.logits,
                    "scores": task.scores,
                },
                run / "task.pt",
            )
            fit = root / "fit"
            metrics = fit_exact_conditionals(run, fit, steps=5)
            actual, _ = load_head(task, fit / "head.pt")
            # Independently repeat the original five updates: logging must not
            # change parameters, targets, weighting, or the final checkpoint.
            torch.manual_seed(0)
            expected = task.head(8)
            optimizer = torch.optim.Adam(expected.parameters(), lr=0.001)
            targets, _ = exact_conditionals(task, 1)
            targets = targets.exp().float()
            nodes = torch.where(task.legal.sum(-1) > 1)[0]
            legal = task.legal[nodes]
            for _ in range(5):
                loss = -(
                    targets[nodes][legal] * task.conditionals(expected, nodes)[legal]
                ).sum() / len(nodes)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0)
                optimizer.step()
            for name, value in expected.state_dict().items():
                torch.testing.assert_close(
                    actual.state_dict()[name], value, atol=0, rtol=0
                )
            rows = [
                json.loads(line)
                for line in (fit / "progress.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["step"] for row in rows], [0, 1, 5])
            self.assertEqual(rows[-1]["total_variation"], metrics["total_variation"])
            output = root / "audit"
            report = inspect_saved_fit(run, fit, output)
            self.assertLess(report["oracle_reconstruction_tv"], 1e-12)
            self.assertLess(
                report["independent_oracle_max_log_conditional_difference"], 1e-12
            )
            self.assertLess(report["chain_rule_kl_difference"], 1e-6)
            self.assertEqual(report["saved_tv_reproduction_difference"], 0)
            self.assertFalse(report["training_performed"])
            self.assertTrue((output / "report.md").is_file())
            with self.assertRaises(FileExistsError):
                inspect_saved_fit(run, fit, output)
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
