"""Exhaustive small-space tests of constraints, prefix paths and run protection."""

import tempfile
import unittest
from pathlib import Path

from stability_gfn.artifacts import new_run
from stability_gfn.environment import PrefixTree, SequenceContext


class EnvironmentTests(unittest.TestCase):
    def test_mutation_budget_and_forced_residues(self):
        context = SequenceContext("AAA", 1, "AG")
        self.assertEqual(context.enumerate_sequences(), ["AAA", "AAG", "AGA", "GAA"])
        self.assertEqual(context.legal_actions("G"), ["A"])
        self.assertEqual(context.legal_actions("GAA"), [])
        with self.assertRaises(ValueError):
            context.legal_actions("GG")
        with self.assertRaises(ValueError):
            context.enumerate_sequences(maximum=3)

    def test_tree_support_and_unique_paths(self):
        context = SequenceContext("AAAA", 4, "AGLS")
        tree = PrefixTree(context, context.enumerate_sequences())
        self.assertEqual(len(tree.sequences), 256)
        self.assertEqual(len(tree.prefixes), 85)
        for leaf, sequence in enumerate(tree.sequences):
            node = 0
            for position, residue in enumerate(sequence):
                action = context.alphabet.index(residue)
                self.assertEqual(tree.paths[leaf, position], node)
                self.assertTrue(tree.legal[node, action])
                if position == 3:
                    self.assertEqual(tree.leaves[node, action], leaf)
                else:
                    node = tree.children[node, action]

    def test_finite_support_removes_dead_ends(self):
        tree = PrefixTree(SequenceContext("AAA", 1, "AG"), ["AAA", "AGA"])
        self.assertEqual(tree.legal[0].tolist(), [True, False])
        self.assertEqual(tree.context_features()[tree.node_index["AG"], 1], 0)
        with self.assertRaises(ValueError):
            PrefixTree(tree.context, ["AAA", "AAA"])
        with self.assertRaises(ValueError):
            PrefixTree(tree.context, ["GGA"])

    def test_existing_runs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "run"
            new_run(destination, {"seed": 0})
            with self.assertRaises(FileExistsError):
                new_run(destination, {"seed": 1})


if __name__ == "__main__":
    unittest.main()
