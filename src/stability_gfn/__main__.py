"""Small command-line interface for preparation, experiments and saved-run evaluation."""

import argparse
import json
import os

# Set before importing Torch; deterministic CUDA configuration matters for user runs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .data import prepare_local_task
from .diagnostics import fit_exact_conditionals, inspect_saved_fit
from .evaluation import evaluate_run
from .pipeline import ROOT, project_path, run_experiment


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
    study = commands.add_parser(
        "reward-study", help="Matched predictor comparison and exact reward evaluation"
    )
    study.add_argument("--config", default="configs/predictor-reward-study.json")
    study.add_argument("--output", required=True)
    study.add_argument("--device", default="cuda")
    reward_audit = commands.add_parser(
        "audit-reward", help="CPU analysis of the original saved reward"
    )
    reward_audit.add_argument("--run", default="runs/biological-first")
    reward_audit.add_argument("--output", required=True)
    ridge = commands.add_parser(
        "ridge-gfn", help="Train guidance on the saved ridge reward and evaluate"
    )
    ridge.add_argument("--config", default="configs/ridge-gfn.json")
    ridge.add_argument("--output")
    ridge.add_argument("--device", default="cuda")
    ridge.add_argument(
        "--check-only",
        action="store_true",
        help="CPU input verification only; no training",
    )
    noise = commands.add_parser(
        "reward-noise", help="Train and evaluate fixed ridge-score noise sensitivity"
    )
    noise.add_argument("--config", default="configs/reward-noise.json")
    noise.add_argument("--output")
    noise.add_argument("--device", default="cuda")
    noise.add_argument("--check-only", action="store_true")
    noise_eval = commands.add_parser(
        "evaluate-noise", help="CPU evaluation of saved noise-study heads"
    )
    noise_eval.add_argument("--run", required=True)
    noise_eval.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "reward-noise":
        from .reward_noise import check_noise_inputs, run_noise_study

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            _, readiness = check_noise_inputs(config)
            print(json.dumps(readiness, indent=2))
        else:
            if args.output is None:
                parser.error("reward-noise requires --output unless --check-only")
            run_noise_study(config, project_path(args.output), args.device)
    elif args.command == "evaluate-noise":
        from .reward_noise import evaluate_noise_study

        evaluate_noise_study(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command == "ridge-gfn":
        from .ridge_guidance import prepare_ridge_task, run_ridge_guidance

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            import torch

            torch.set_num_threads(1)
            _, _, metadata = prepare_ridge_task(config)
            print(json.dumps(metadata, indent=2))
        else:
            if args.output is None:
                parser.error("ridge-gfn requires --output unless --check-only")
            run_ridge_guidance(config, project_path(args.output), args.device)
    elif args.command == "reward-study":
        from .reward_study import run_reward_study

        run_reward_study(
            json.loads(project_path(args.config).read_text()),
            project_path(args.output),
            args.device,
        )
    elif args.command == "audit-reward":
        from .reward_study import audit_pilot

        print(
            json.dumps(
                audit_pilot(project_path(args.run), project_path(args.output)), indent=2
            )
        )
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
