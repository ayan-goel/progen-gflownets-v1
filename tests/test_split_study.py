"""CPU synthetic checks for partition integrity, cache alignment and full resplit runs."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from stability_gfn.artifacts import file_hash, write_json
from stability_gfn.backbone import FrozenProGen
from stability_gfn.environment import SequenceContext
from stability_gfn.pipeline import ROOT
from stability_gfn.split_study import (
    mutation_positions,
    position_split_indices,
    prepare_split_inputs,
    run_split_study,
    split_indices,
    summarize_split_results,
)


class SplitStudyTests(unittest.TestCase):
    def test_position_groups_disjoint_complete_and_order_invariant(self):
        reference = "AAAAAA"
        sequences = SequenceContext(reference, 1).enumerate_sequences()
        # Uneven group sizes must not cause variants to be moved between groups.
        sequences.remove("CAAAAA")
        partition = position_split_indices(sequences, reference, 43)
        positions = mutation_positions(sequences, reference)
        groups = {
            split: {positions[index] for index in indices} - {None}
            for split, indices in partition.items()
        }
        self.assertEqual([len(groups[s]) for s in partition], [4, 1, 1])
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        self.assertEqual(
            sorted(i for indices in partition.values() for i in indices),
            list(range(len(sequences))),
        )
        self.assertEqual(sequences[partition["train"][0]], reference)
        reversed_sequences = list(reversed(sequences))
        reordered = position_split_indices(reversed_sequences, reference, 43)
        for split in partition:
            self.assertEqual(
                [sequences[i] for i in partition[split]],
                [reversed_sequences[i] for i in reordered[split]],
            )

    def test_position_holdout_rejects_ambiguous_groups(self):
        for sequences in [
            ["AAAA", "CAAA", "CCAA"],
            ["AAAA", "CAAA", "AAA"],
            ["AAAA", "CAAA", "CAAA"],
            ["AAAA", "CAAA", "DAAA"],
        ]:
            with self.assertRaises(ValueError):
                position_split_indices(sequences, "AAAA", 43)

    def test_position_holdout_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            config["split_unit"] = "position"
            ready = prepare_split_inputs(config)[-1]
            self.assertEqual(ready["split_unit"], "position")
            run = root / "run"
            run_split_study(config, run, "cpu", smoke=True)
            comparison = json.loads((run / "comparison.json").read_text())
            self.assertEqual(len(comparison), 9)
            self.assertEqual(
                comparison, summarize_split_results(run, config["split_seeds"])
            )
            for seed in config["split_seeds"]:
                directory = run / f"split-{seed}"
                groups = json.loads((directory / "mutation_positions.json").read_text())
                self.assertEqual(groups, ready["mutation_positions"][str(seed)])
                self.assertFalse(set(groups["train"]) & set(groups["test"]))
                self.assertFalse(set(groups["validation"]) & set(groups["test"]))
                saved_rows = json.loads((directory / "partitions.json").read_text())
                for split, rows in saved_rows.items():
                    actual = set(
                        mutation_positions([r["sequence"] for r in rows], "AAAA")
                    ) - {None}
                    self.assertEqual(actual, set(groups[split]))
                prior = torch.load(
                    directory / "prior.pt", map_location="cpu", weights_only=True
                )
                self.assertEqual(
                    prior["sequences"], [r["sequence"] for r in saved_rows["test"]]
                )
                self.assertAlmostEqual(
                    float(prior["log_prior"].double().exp().sum()), 1, places=5
                )
            for result in comparison:
                self.assertTrue(0 < result["prior_top_one_mass"] <= 1)
                self.assertTrue(0 < result["target_top_one_mass"] <= 1)
                if result["family"] == "constant":
                    self.assertAlmostEqual(
                        result["measured_gain_over_own_progen"], 0, places=10
                    )
                    self.assertEqual(
                        result["prior_top_one_mass"], result["target_top_one_mass"]
                    )
            self.assertIn("Concentration", (run / "report.md").read_text())
            self.assertIn("Mutation-position", (run / "protocol.md").read_text())
            for path, expected in config["input_sha256"].items():
                self.assertEqual(file_hash(Path(path)), expected)
            self.assertFalse(torch.cuda.is_initialized())
            with self.assertRaises(FileExistsError):
                run_split_study(config, run, "cpu", smoke=True)

    def fixture(self, root):
        torch.set_num_threads(1)
        pilot, study = root / "pilot", root / "study"
        (pilot / "data_manifest").mkdir(parents=True)
        study.mkdir()
        original = json.loads((ROOT / "configs/biological-pilot.json").read_text())
        original["smoke"] = True
        write_json(pilot / "config.json", original)
        reference = "AAAA"
        sequences = SequenceContext(reference, 1).enumerate_sequences()
        features = FrozenProGen.tiny().sequence_features(sequences)
        rows = [
            {"sequence": s, "stability": float(4 + 10 * features[i, 0])}
            for i, s in enumerate(sequences)
        ]
        partition = split_indices(sequences, reference, 42)
        torch.save(
            {
                "rows": {s: [rows[i] for i in v] for s, v in partition.items()},
                "features": {s: features[v] for s, v in partition.items()},
            },
            study / "features.pt",
        )
        write_json(
            study / "provenance.json",
            {
                "feature_cache_sha256": file_hash(study / "features.pt"),
                "checkpoint_sha256": {},
            },
        )
        write_json(
            pilot / "data_manifest/manifest.json",
            {
                "reference": reference,
                "parent": "synthetic",
                "mutation_budget": 1,
                "split_seed": 42,
                "split_counts": {s: len(v) for s, v in partition.items()},
            },
        )
        paths = [
            pilot / "config.json",
            pilot / "data_manifest/manifest.json",
            study / "features.pt",
            study / "provenance.json",
        ]
        return {
            "pilot_run": str(pilot),
            "predictor_run": str(study),
            "split_seeds": [43, 44, 45],
            "predictor_seed": 0,
            "input_sha256": {str(p): file_hash(p) for p in paths},
        }

    def test_identity_based_disjoint_deterministic_partitions(self):
        sequences = ["reference"] + [f"variant-{i:04}" for i in range(1357)]
        partition = split_indices(sequences, "reference", 43)
        self.assertEqual(
            [len(partition[s]) for s in ["train", "validation", "test"]],
            [950, 204, 204],
        )
        self.assertEqual(partition["train"][0], 0)
        flat = [i for indices in partition.values() for i in indices]
        self.assertEqual(sorted(flat), list(range(1358)))
        reversed_sequences = list(reversed(sequences))
        reordered = split_indices(reversed_sequences, "reference", 43)
        for split in partition:
            self.assertEqual(
                [sequences[i] for i in partition[split]],
                [reversed_sequences[i] for i in reordered[split]],
            )
        self.assertNotEqual(
            partition["test"], split_indices(sequences, "reference", 44)["test"]
        )
        with self.assertRaises(ValueError):
            split_indices(sequences + ["reference"], "reference", 43)

    def test_tiny_end_to_end_and_report_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            _, _, rows, features, _, ready = prepare_split_inputs(config)
            for row, feature in zip(rows, features):
                self.assertAlmostEqual(
                    row["stability"], float(4 + 10 * feature[0]), places=6
                )
            self.assertEqual(ready["new_gfn_fits"], 0)
            run = root / "run"
            run_split_study(config, run, "cpu", smoke=True)
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"],
                "smoke_complete",
            )
            first = json.loads((run / "comparison.json").read_text())
            self.assertEqual(first, summarize_split_results(run, config["split_seeds"]))
            self.assertEqual(len(first), 9)
            for seed in config["split_seeds"]:
                directory = run / f"split-{seed}"
                partitions = json.loads((directory / "partitions.json").read_text())
                prior = torch.load(
                    directory / "prior.pt", map_location="cpu", weights_only=True
                )
                self.assertEqual(
                    prior["sequences"], [r["sequence"] for r in partitions["test"]]
                )
                self.assertAlmostEqual(
                    float(prior["log_prior"].double().exp().sum()), 1.0, places=5
                )
                control = next(
                    r
                    for r in first
                    if r["split_seed"] == seed and r["family"] == "constant"
                )
                self.assertAlmostEqual(
                    control["measured_gain_over_own_progen"], 0.0, places=10
                )
            for p, expected in config["input_sha256"].items():
                self.assertEqual(file_hash(Path(p)), expected)
            self.assertFalse(torch.cuda.is_initialized())
            with self.assertRaises(FileExistsError):
                run_split_study(config, run, "cpu", smoke=True)

    def test_original_split_and_real_cpu_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            with self.assertRaises(ValueError):
                run_split_study(config, root / "bad", "cpu")
            self.assertFalse((root / "bad").exists())
            config["split_seeds"] = [42, 43, 44]
            with self.assertRaisesRegex(ValueError, "exclude the original"):
                prepare_split_inputs(config)

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            (root / "study/provenance.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                prepare_split_inputs(config)
