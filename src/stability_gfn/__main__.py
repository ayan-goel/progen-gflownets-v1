"""Small command-line interface for preparation, experiments and saved-run evaluation."""

import argparse
import json
import os

# Set before importing Torch; deterministic CUDA configuration matters for user runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .confirmation import run_confirmation
from .confirmation_data import prepare_confirmation_support
from .confirmation_evaluation import evaluate_confirmation
from .data import prepare_local_task
from .diagnostics import fit_exact_conditionals, inspect_saved_fit
from .diversity import diagnose_diversity, inspect_saved_unique_selection
from .diversity_experiment import (
    check_diversity_inputs,
    evaluate_diversity,
    run_diversity,
)
from .evaluation import evaluate_run
from .panel import prepare_panel
from .pipeline import ROOT, project_path, run_experiment
from .target_experiment import evaluate_target_run, run_target_diagnostic


def main():
    """Resolve all relative paths from v1 and keep CPU smoke runs explicit."""
    parser = argparse.ArgumentParser(description="Frozen ProGen2 stability guidance")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Audit Mega-scale and create a local task; no model execution"
    )
    prepare.add_argument(
        "--csv", default="data/raw/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    )
    prepare.add_argument("--output", default="data/processed/local-v1")
    prepare.add_argument("--parent")
    prepare.add_argument("--split-seed", type=int, default=42)
    run = commands.add_parser(
        "run", help="Train and evaluate; explicitly choose real CUDA or tiny CPU smoke"
    )
    run.add_argument("--config", default="configs/exact.json")
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--smoke", action="store_true")
    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate saved checkpoints without retraining or loading ProGen",
    )
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--output", required=True)
    doctor = commands.add_parser(
        "doctor", help="CPU-only environment/file readiness checks"
    )
    doctor.add_argument("--config", default="configs/exact.json")
    diagnostic = commands.add_parser(
        "diagnose-capacity",
        help="Privileged exact conditional fit on a saved finite cache",
    )
    diagnostic.add_argument("--run", required=True)
    diagnostic.add_argument("--output", required=True)
    diagnostic.add_argument("--steps", type=int, default=2000)
    diagnostic.add_argument("--learning-rate", type=float, default=0.001)
    diagnostic.add_argument("--seed", type=int, default=0)
    audit = commands.add_parser(
        "inspect-fit", help="Read-only CPU audit of a saved exact conditional fit"
    )
    audit.add_argument("--run", required=True)
    audit.add_argument("--fit", required=True)
    audit.add_argument("--output", required=True)
    panel = commands.add_parser(
        "prepare-panel",
        help="Freeze development/confirmation identities and splits; no models",
    )
    panel.add_argument("--config", default="configs/target-diagnostic.json")
    panel.add_argument(
        "--csv", default="data/raw/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    )
    panel.add_argument("--output")
    target = commands.add_parser(
        "diagnose-target",
        help="Fit development scorers and evaluate exact targets; no GFN training",
    )
    target.add_argument("--config", default="configs/target-diagnostic.json")
    target.add_argument("--output", required=True)
    target.add_argument("--device", default="cpu")
    target.add_argument("--smoke", action="store_true")
    target_evaluation = commands.add_parser(
        "evaluate-target",
        help="Recompute saved development diagnostics on CPU; no fitting",
    )
    target_evaluation.add_argument("--run", required=True)
    target_evaluation.add_argument("--output", required=True)
    confirmation_prepare = commands.add_parser(
        "prepare-confirmation",
        help="Reconstruct reserved candidate identities without opening test-label files",
    )
    confirmation_prepare.add_argument("--config", default="configs/confirmation.json")
    confirmation_run = commands.add_parser(
        "confirm",
        help="Fit all reserved parents, then evaluate; real GPU runs are user initiated",
    )
    confirmation_run.add_argument("--config", default="configs/confirmation.json")
    confirmation_run.add_argument("--output", required=True)
    confirmation_run.add_argument("--device", default="cpu")
    confirmation_run.add_argument("--smoke", action="store_true")
    confirmation_eval = commands.add_parser(
        "evaluate-confirmation",
        help="Verify completed fits and evaluate confirmation outcomes on CPU",
    )
    confirmation_eval.add_argument("--run", required=True)
    confirmation_eval.add_argument("--output", required=True)
    diversity_diag = commands.add_parser(
        "diagnose-diversity",
        help="CPU development concentration and unique-selection audit",
    )
    diversity_diag.add_argument("--run", default="runs/target-diagnostic-v1")
    diversity_diag.add_argument("--output", required=True)
    diversity_run = commands.add_parser(
        "diversity", help="User-run two-prior development ablation"
    )
    diversity_run.add_argument("--config", default="configs/diversity.json")
    diversity_run.add_argument("--output", required=True)
    diversity_run.add_argument("--device", default="cpu")
    diversity_run.add_argument("--smoke", action="store_true")
    diversity_eval = commands.add_parser(
        "evaluate-diversity", help="CPU replay of saved development prior ablation"
    )
    diversity_eval.add_argument("--run", required=True)
    diversity_eval.add_argument("--output", required=True)
    diversity_check = commands.add_parser(
        "check-diversity",
        help="CPU-only metadata, checkpoint and frozen-input integrity check",
    )
    diversity_check.add_argument("--config", default="configs/diversity.json")
    unique_review = commands.add_parser(
        "inspect-unique",
        help="Post-hoc unique selection from completed confirmation proposals",
    )
    unique_review.add_argument("--run", required=True)
    unique_review.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "inspect-unique":
        inspect_saved_unique_selection(
            project_path(args.run), project_path(args.output)
        )
        print(project_path(args.output) / "report.md")
    elif args.command == "diagnose-diversity":
        diagnose_diversity(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command == "diversity":
        config = json.loads(project_path(args.config).read_text())
        run_diversity(config, project_path(args.output), args.device, args.smoke)
    elif args.command == "evaluate-diversity":
        evaluate_diversity(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command == "check-diversity":
        import torch

        config = json.loads(project_path(args.config).read_text())
        loaded, _ = check_diversity_inputs(config)
        print(
            json.dumps(
                {
                    "parents": [entry["parent"] for entry, _, _ in loaded],
                    "inputs_verified": True,
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "gpu_readiness": "Not queried",
                },
                indent=2,
            )
        )
    elif args.command == "prepare-confirmation":
        config = json.loads(project_path(args.config).read_text())
        manifest = prepare_confirmation_support(
            project_path(config["panel"]), project_path(config["candidate_support"])
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "confirm":
        config = json.loads(project_path(args.config).read_text())
        run_confirmation(config, project_path(args.output), args.device, args.smoke)
    elif args.command == "evaluate-confirmation":
        evaluate_confirmation(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command == "prepare-panel":
        config = json.loads(project_path(args.config).read_text())
        manifest = prepare_panel(
            project_path(args.csv),
            project_path(args.output or config["panel"]),
            config["panel_preparation"],
        )
        print(
            json.dumps(
                {
                    "parents": [
                        {"parent": entry["parent"], "role": entry["role"]}
                        for entry in manifest["parents"]
                    ]
                },
                indent=2,
            )
        )
    elif args.command == "diagnose-target":
        config = json.loads(project_path(args.config).read_text())
        run_target_diagnostic(
            config, project_path(args.output), args.device, args.smoke
        )
    elif args.command == "evaluate-target":
        evaluate_target_run(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command == "prepare":
        manifest = prepare_local_task(
            project_path(args.csv),
            project_path(args.output),
            args.parent,
            args.split_seed,
        )
        print(json.dumps(manifest, indent=2))
    elif args.command == "run":
        config = json.loads(project_path(args.config).read_text())
        if not run_experiment(
            config, project_path(args.output), args.device, args.smoke
        ):
            raise SystemExit(2)
    elif args.command == "evaluate":
        evaluate_run(project_path(args.run), project_path(args.output), device="cpu")
        print(project_path(args.output) / "report.md")
    elif args.command == "inspect-fit":
        inspect_saved_fit(
            project_path(args.run), project_path(args.fit), project_path(args.output)
        )
        print(project_path(args.output) / "report.md")
    elif args.command == "diagnose-capacity":
        print(
            json.dumps(
                fit_exact_conditionals(
                    project_path(args.run),
                    project_path(args.output),
                    args.steps,
                    args.learning_rate,
                    args.seed,
                ),
                indent=2,
            )
        )
    else:
        import sys

        import tokenizers
        import torch
        import transformers

        from .backbone import TOKENIZER_PATH
        from .pipeline import validate_config

        config = json.loads(project_path(args.config).read_text())
        validate_config(config)
        checkpoint = project_path(config["checkpoint"])
        missing = [
            str(checkpoint / name)
            for name in ["config.json", "pytorch_model.bin"]
            if not (checkpoint / name).is_file()
        ]
        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        }
        compatible = (
            torch.__version__.split("+")[0] == "2.6.0"
            and transformers.__version__ == "4.27.4"
            and tokenizers.__version__ == "0.13.3"
        )
        print(
            json.dumps(
                {
                    "root": str(ROOT),
                    "python": sys.version,
                    "versions": versions,
                    "pinned_versions_match": compatible,
                    "checkpoint": str(checkpoint),
                    "missing_files": missing,
                    "tokenizer_exists": TOKENIZER_PATH.is_file(),
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "gpu_readiness": "Not queried; must be tested by the user's first GPU run",
                },
                indent=2,
            )
        )
        if missing or not compatible:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
