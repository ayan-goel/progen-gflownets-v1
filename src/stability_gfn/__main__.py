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
    splits = commands.add_parser(
        "split-study",
        help="Three new local splits: fit predictors and evaluate exact targets",
    )
    splits.add_argument("--config", default="configs/split-robustness.json")
    splits.add_argument("--output")
    splits.add_argument("--device", default="cuda")
    splits.add_argument("--check-only", action="store_true")
    positions = commands.add_parser(
        "position-study",
        help="Hold out whole mutation positions; fit predictors and evaluate targets",
    )
    positions.add_argument("--config", default="configs/position-holdout.json")
    positions.add_argument("--output")
    positions.add_argument("--device", default="cuda")
    positions.add_argument("--check-only", action="store_true")
    position_gfn = commands.add_parser(
        "position-gfn", help="Train GFN on all saved position-held-out ridge targets"
    )
    position_gfn.add_argument("--config", default="configs/position-gfn.json")
    position_gfn.add_argument("--output")
    position_gfn.add_argument("--device", default="cuda")
    position_gfn.add_argument("--check-only", action="store_true")
    position_eval = commands.add_parser(
        "evaluate-position-gfn", help="CPU reevaluation of saved position-GFN heads"
    )
    position_eval.add_argument("--run", required=True)
    position_eval.add_argument("--output", required=True)
    multi_prepare = commands.add_parser(
        "prepare-multiprotein", help="CPU audit and grouped multi-protein split"
    )
    multi_prepare.add_argument("--config", default="configs/multiprotein-data.json")
    multi_prepare.add_argument(
        "--csv", default="data/raw/Tsuboyama2023_Dataset2_Dataset3_20230416.csv"
    )
    multi_prepare.add_argument(
        "--exposure", default="data/metadata/prior-exposure-2026-11-10.json"
    )
    multi_prepare.add_argument("--output", required=True)
    multi = commands.add_parser(
        "multiprotein",
        help="Fit and evaluate stability predictors across protein groups",
    )
    multi.add_argument("--config", default="configs/multiprotein.json")
    multi.add_argument("--output")
    multi.add_argument("--device", default="cuda")
    multi.add_argument("--check-only", action="store_true")
    multi_eval = commands.add_parser(
        "evaluate-multiprotein", help="CPU replay of multi-protein predictor metrics"
    )
    multi_eval.add_argument("--run", required=True)
    multi_eval.add_argument("--output", required=True)
    regularization = commands.add_parser(
        "regularization",
        help="Grouped ridge-strength comparison on saved protein features",
    )
    regularization.add_argument("--config", default="configs/regularization.json")
    regularization.add_argument("--output")
    regularization.add_argument("--device", default="cuda")
    regularization.add_argument("--check-only", action="store_true")
    coverage_prepare = commands.add_parser(
        "prepare-coverage",
        help="Add training proteins while preserving existing holdouts",
    )
    coverage_prepare.add_argument("--config", default="configs/coverage-data.json")
    coverage_prepare.add_argument("--output", required=True)
    coverage = commands.add_parser(
        "coverage", help="Compare fixed-ridge training coverage"
    )
    coverage.add_argument("--config", default="configs/coverage.json")
    coverage.add_argument("--output")
    coverage.add_argument("--device", default="cuda")
    coverage.add_argument("--check-only", action="store_true")
    idr = commands.add_parser(
        "idr-baseline", help="Frozen-ProGen IDR generation and automatic evaluation"
    )
    idr.add_argument("--config", default="configs/idr-baseline.json")
    idr.add_argument("--output")
    idr.add_argument("--device", default="cuda")
    idr.add_argument("--check-only", action="store_true")
    idr_prepare = commands.add_parser(
        "prepare-idr",
        help="Prepare an experimental IDR reference panel from a local DisProt snapshot",
    )
    idr_prepare.add_argument("--source", default="data/raw/disprot-2026-11-12.json")
    idr_prepare.add_argument("--output", required=True)
    idr_eval = commands.add_parser(
        "evaluate-idr",
        help="CPU disorder evaluation of saved IDR samples; no ProGen loading",
    )
    idr_eval.add_argument("--run", required=True)
    idr_eval.add_argument("--output", required=True)
    charge = commands.add_parser(
        "idr-charge",
        help="Train frozen-ProGen charge guidance and evaluate comparisons",
    )
    charge.add_argument("--config", default="configs/idr-charge.json")
    charge.add_argument("--output")
    charge.add_argument("--device", default="cuda")
    charge.add_argument("--check-only", action="store_true")
    charge_eval = commands.add_parser(
        "evaluate-idr-charge", help="CPU evaluation of completed charge-pilot samples"
    )
    charge_eval.add_argument("--run", required=True)
    charge_eval.add_argument("--output", required=True)
    learning = commands.add_parser(
        "idr-diagnose", help="Capture real states and diagnose charge-learning updates"
    )
    learning.add_argument("--config", default="configs/idr-learning-diagnostic.json")
    learning.add_argument("--output")
    learning.add_argument("--device", default="cuda")
    learning.add_argument("--check-only", action="store_true")
    learning_replay = commands.add_parser(
        "analyze-idr-diagnostic",
        help="CPU analysis of saved diagnostic feature batches",
    )
    learning_replay.add_argument("--run", required=True)
    learning_replay.add_argument("--output", required=True)
    context = commands.add_parser(
        "idr-context", help="Compare standard and direct charge-context guidance"
    )
    context.add_argument("--config", default="configs/idr-context.json")
    context.add_argument("--output")
    context.add_argument("--device", default="cuda")
    context.add_argument("--check-only", action="store_true")
    context_eval = commands.add_parser(
        "evaluate-idr-context", help="Replay context-comparison scoring on CPU"
    )
    context_eval.add_argument("--run", required=True)
    context_eval.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "idr-context":
        from .idr.context_experiment import check_context_inputs, run_context

        settings = json.loads(project_path(args.config).read_text())
        if args.check_only:
            config, _, manifest = check_context_inputs(settings)
            print(
                json.dumps(
                    {
                        "arms": settings["arms"],
                        "seeds": config["seeds"],
                        "steps_per_fit": config["steps"],
                        "training_trajectories": 2
                        * len(config["seeds"])
                        * config["steps"]
                        * manifest["prompt_count"],
                        "added_parameters": 100,
                        "model_execution": False,
                    },
                    indent=2,
                )
            )
        else:
            if not args.output:
                parser.error("idr-context requires --output unless --check-only")
            print(run_context(settings, project_path(args.output), args.device))
    elif args.command == "evaluate-idr-context":
        from .idr.context_experiment import reevaluate_context

        print(reevaluate_context(project_path(args.run), project_path(args.output)))
    elif args.command == "idr-diagnose":
        from .idr.learning_diagnostic import check_source, run_learning_diagnostic

        settings = json.loads(project_path(args.config).read_text())
        if args.check_only:
            _, _, references = check_source(settings)
            prompts = sum(r["role"] == "prompt" for r in references)
            print(
                json.dumps(
                    {
                        "seeds": settings["seeds"],
                        "cases": 2 * len(settings["seeds"]),
                        "trajectories": 2
                        * len(settings["seeds"])
                        * settings["batches_per_case"]
                        * prompts,
                        "model_execution": False,
                    },
                    indent=2,
                )
            )
        else:
            if not args.output:
                parser.error("idr-diagnose requires --output unless --check-only")
            print(
                run_learning_diagnostic(
                    settings, project_path(args.output), args.device
                )
            )
    elif args.command == "analyze-idr-diagnostic":
        from .idr.learning_diagnostic import replay_learning_diagnostic

        print(
            replay_learning_diagnostic(
                project_path(args.run), project_path(args.output)
            )
        )
    elif args.command == "idr-charge":
        from .idr.charge_experiment import check_charge_inputs, run_charge

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            _, manifest = check_charge_inputs(config)
            print(
                json.dumps(
                    {
                        "prompts": manifest["prompt_count"],
                        "seeds": config["seeds"],
                        "steps_per_seed": config["steps"],
                        "reward": config["reward"],
                        "model_execution": False,
                    },
                    indent=2,
                )
            )
        else:
            if not args.output:
                parser.error("idr-charge requires --output unless --check-only")
            print(run_charge(config, project_path(args.output), args.device))
    elif args.command == "evaluate-idr-charge":
        from .idr.charge_experiment import reevaluate_charge

        print(reevaluate_charge(project_path(args.run), project_path(args.output)))
    elif args.command == "idr-baseline":
        from .idr.baseline import check_inputs, run_baseline

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            references, manifest = check_inputs(config)
            print(
                json.dumps(
                    {
                        "references": len(references),
                        "manifest": manifest,
                        "model_execution": False,
                    },
                    indent=2,
                )
            )
        else:
            if not args.output:
                parser.error("idr-baseline requires --output unless --check-only")
            print(run_baseline(config, project_path(args.output), args.device))
    elif args.command == "prepare-idr":
        from .idr.data import prepare_references

        print(
            prepare_references(
                project_path(args.source),
                project_path(args.output),
                "https://disprot.org/api/search?format=json",
            )
        )
    elif args.command == "evaluate-idr":
        from .idr.baseline import reevaluate

        print(reevaluate(project_path(args.run), project_path(args.output)))
    elif args.command == "prepare-multiprotein":
        from .multiprotein_data import prepare_multiprotein

        manifest = prepare_multiprotein(
            project_path(args.csv),
            project_path(args.output),
            project_path(args.exposure),
            json.loads(project_path(args.config).read_text()),
        )
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in [
                        "eligible_parents",
                        "components",
                        "unexposed_components",
                        "split_counts",
                        "parent_counts",
                    ]
                },
                indent=2,
            )
        )
    elif args.command == "prepare-coverage":
        from .coverage_data import prepare_coverage

        config = json.loads(project_path(args.config).read_text())
        manifest = prepare_coverage(config, project_path(args.output))
        print(
            json.dumps(
                {
                    key: manifest[key]
                    for key in [
                        "training_parent_counts",
                        "additional_parents",
                        "additional_sequences",
                        "validation_and_test_unchanged",
                    ]
                },
                indent=2,
            )
        )
    elif args.command == "coverage":
        from .coverage import check_coverage, run_coverage

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            _, _, manifest, files, _, _, _ = check_coverage(config)
            print(
                json.dumps(
                    {
                        "training_parent_counts": manifest["training_parent_counts"],
                        "additional_sequences": manifest["additional_sequences"],
                        "validation_sequences": len(files["validation.json"]),
                        "test_evaluated": False,
                    },
                    indent=2,
                )
            )
        else:
            if args.output is None:
                parser.error("coverage requires --output unless --check-only")
            run_coverage(config, project_path(args.output), args.device)
    elif args.command == "regularization":
        from .regularization import check_regularization, run_regularization

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            _, _, _, rows = check_regularization(config)
            print(
                json.dumps(
                    {
                        "rows": {
                            split: len(partition) for split, partition in rows.items()
                        },
                        "folds": config["folds"],
                        "ridge_penalties": config["ridge_penalties"],
                        "test_evaluated": False,
                    },
                    indent=2,
                )
            )
        else:
            if args.output is None:
                parser.error("regularization requires --output unless --check-only")
            run_regularization(config, project_path(args.output), args.device)
    elif args.command == "multiprotein":
        from .multiprotein import check_multiprotein, run_multiprotein

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            manifest, _ = check_multiprotein(config)
            print(
                json.dumps(
                    {
                        key: manifest[key]
                        for key in [
                            "split_counts",
                            "parent_counts",
                            "maximum_cross_partition_reference_identity",
                            "scope",
                        ]
                    },
                    indent=2,
                )
            )
        else:
            if args.output is None:
                parser.error("multiprotein requires --output unless --check-only")
            run_multiprotein(config, project_path(args.output), args.device)
    elif args.command == "evaluate-multiprotein":
        from .multiprotein import evaluate_multiprotein

        evaluate_multiprotein(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command == "position-gfn":
        from .position_guidance import prepare_position_guidance, run_position_guidance

        config = json.loads(project_path(args.config).read_text())
        if args.check_only:
            import torch

            torch.set_num_threads(1)
            print(json.dumps(prepare_position_guidance(config)[-1], indent=2))
        else:
            if args.output is None:
                parser.error("position-gfn requires --output unless --check-only")
            run_position_guidance(config, project_path(args.output), args.device)
    elif args.command == "evaluate-position-gfn":
        import torch

        from .position_guidance import evaluate_position_guidance

        torch.set_num_threads(1)
        evaluate_position_guidance(project_path(args.run), project_path(args.output))
        print(project_path(args.output) / "report.md")
    elif args.command in {"split-study", "position-study"}:
        from .split_study import prepare_split_inputs, run_split_study

        config = json.loads(project_path(args.config).read_text())
        if args.command == "position-study" and config.get("split_unit") != "position":
            parser.error("position-study requires a position-split configuration")
        if args.check_only:
            import torch

            torch.set_num_threads(1)
            print(json.dumps(prepare_split_inputs(config)[-1], indent=2))
        else:
            if args.output is None:
                parser.error(f"{args.command} requires --output unless --check-only")
            run_split_study(config, project_path(args.output), args.device)
    elif args.command == "reward-noise":
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
