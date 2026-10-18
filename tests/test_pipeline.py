"""Small end-to-end CLI-equivalent runs and reload parity on CPU only."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from stability_gfn.backbone import FrozenProGen
from stability_gfn.data import REQUIRED_COLUMNS, prepare_local_task
from stability_gfn.diagnostics import fit_exact_conditionals
from stability_gfn.environment import SequenceContext
from stability_gfn.evaluation import evaluate_run
from stability_gfn.pipeline import ROOT, run_experiment, validate_config
from stability_gfn.scorer import fit_scorer


class PipelineTests(unittest.TestCase):
    def test_exact_smoke_and_saved_evaluation_parity(self):
        config = json.loads((ROOT / "configs/exact.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            self.assertTrue(run_experiment(config, run, "cpu", smoke=True))
            self.assertEqual(
                json.loads((run / "status.json").read_text())["status"],
                "smoke_complete",
            )
            self.assertFalse(json.loads((run / "gate.json").read_text())["passed"])
            replay = Path(directory) / "evaluation"
            evaluate_run(run, replay)
            original = json.loads((run / "evaluation/summary.json").read_text())
            repeated = json.loads((replay / "summary.json").read_text())
            self.assertEqual(original, repeated)
            for method in ["progen", "gfn", "supervised"]:
                first = json.loads(
                    (run / f"evaluation/seed-0-{method}.json").read_text()
                )
                second = json.loads((replay / f"seed-0-{method}.json").read_text())
                self.assertEqual(first["raw_sequences"], second["raw_sequences"])
                self.assertEqual(first["selected_indices"], second["selected_indices"])
            self.assertFalse(torch.cuda.is_initialized())
            diagnostic = Path(directory) / "capacity"
            metrics = fit_exact_conditionals(run, diagnostic, steps=5)
            self.assertGreaterEqual(metrics["total_variation"], 0)
            self.assertTrue(
                json.loads((diagnostic / "diagnostics.json").read_text())["privileged"]
            )
            with self.assertRaises(FileExistsError):
                run_experiment(config, run, "cpu", smoke=True)

    def test_scheduled_exact_smoke_persists_metrics_and_transition(self):
        config = json.loads((ROOT / "configs/exact-decay.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scheduled-smoke"
            self.assertTrue(run_experiment(config, output, "cpu", smoke=True))
            for method in ("gfn", "supervised"):
                history = json.loads(
                    (output / f"seed-0/{method}_history.json").read_text()
                )
                progress = [
                    json.loads(line)
                    for line in (output / f"seed-0/{method}_progress.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(history, progress)
                self.assertEqual(
                    [row["step"] for row in history], [0, 1, 10, 16, 17, 20]
                )
                self.assertAlmostEqual(history[-3]["learning_rate"], 0.001)
                self.assertAlmostEqual(history[-2]["learning_rate"], 0.0001)
                self.assertGreaterEqual(
                    history[-1]["distribution"]["total_variation"], 0
                )
            gate = json.loads((output / "gate.json").read_text())
            self.assertFalse(gate["passed"])
            gfn_history = json.loads((output / "seed-0/gfn_history.json").read_text())
            self.assertEqual(
                gfn_history[-1]["distribution"]["total_variation"],
                gate["seeds"][0]["total_variation"],
            )
            self.assertFalse(torch.cuda.is_initialized())

    def test_real_cpu_experiments_are_not_accidental(self):
        config = json.loads((ROOT / "configs/exact.json").read_text())
        with self.assertRaises(ValueError):
            run_experiment(config, "/tmp/not-created-stability-gfn", "cpu", smoke=False)
        config["supervised"]["beta"] = 2
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_biological_pipeline_on_synthetic_stability_fixture(self):
        """Exercise the biological branch without fitting on real protein measurements."""
        torch.set_num_threads(1)
        context = SequenceContext("AAAA", 1)
        sequences = context.enumerate_sequences()
        backbone = FrozenProGen.tiny()
        # Known linear synthetic property keeps the gate meaningful and reproducible.
        labels = backbone.sequence_features(sequences)[:, 0] * 10
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.csv"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
                writer.writeheader()
                for sequence, label in zip(sequences, labels.tolist()):
                    writer.writerow(
                        {
                            "name": sequence,
                            "aa_seq": sequence,
                            "WT_name": "synthetic",
                            "WT_cluster": "synthetic",
                            "mut_type": "wt" if sequence == "AAAA" else "mut",
                            "dG_ML": label,
                            "ddG_ML": "-",
                        }
                    )
            task_dir = Path(directory) / "task"
            prepare_local_task(
                source, task_dir, min_variants=10, min_length=4, synthetic=True
            )
            config = json.loads((ROOT / "configs/biological.json").read_text())
            config["data_task"] = str(task_dir)
            output = Path(directory) / "run"
            self.assertTrue(run_experiment(config, output, "cpu", smoke=True))
            self.assertTrue(
                json.loads((output / "scorer_validation.json").read_text())[
                    "gate_passed"
                ]
            )
            metrics = json.loads((output / "evaluation/summary.json").read_text())[
                "metrics"
            ]
            self.assertTrue(all(row["measurement_coverage"] == 1 for row in metrics))
            self.assertTrue(all(row["constraint_violations"] == 0 for row in metrics))
            self.assertTrue(
                all(row["fraction_absent_from_training"] == 1 for row in metrics)
            )

            def failing_gate(*args, **kwargs):
                model, report = fit_scorer(*args, **kwargs)
                report["gate_passed"] = False
                return model, report

            failed_run = Path(directory) / "failed-gate"
            with patch("stability_gfn.pipeline.fit_scorer", side_effect=failing_gate):
                self.assertFalse(run_experiment(config, failed_run, "cpu", smoke=True))
            self.assertEqual(
                json.loads((failed_run / "status.json").read_text())["status"],
                "scorer_gate_failed",
            )
            self.assertFalse((failed_run / "seed-0").exists())
            self.assertFalse((failed_run / "evaluation_labels.json").exists())
            self.assertFalse((failed_run / "data_manifest/test_labels.json").exists())


if __name__ == "__main__":
    unittest.main()
