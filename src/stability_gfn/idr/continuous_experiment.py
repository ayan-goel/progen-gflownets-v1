"""Reward-only GFlowNet comparison, preserving all completed implementations."""

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .charge_experiment import generate_records, work_delta
from .context_experiment import make_context_head
from .continuous_data import check_continuous_inputs
from .continuous_evaluation import ARMS, evaluate_continuous
from .continuous_training import train_continuous
from .scoring import predict_disorder


def run_continuous(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """One new TB fit and fresh generation from the candidate and two controls."""
    inputs, control, control_checkpoint = check_continuous_inputs(
        settings, backbone is None
    )
    source, config, references, provenance, reward, bank, cache, initial = inputs
    output = new_run(output, settings)
    fixture, started = backbone is not None, time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(output / "pilot_config.json", config)
        write_json(output / "reward.json", asdict(reward))
        write_json(
            output / "reward_definition.json",
            {
                "version": "continuous_acceptance_v1",
                "penalty": settings["penalty"],
                "log_factor": "-penalty * sum(outside_tolerance_violations)",
                "acceptance_criteria": "unchanged",
                "effective_beta": 1,
            },
        )
        write_json(output / "references.json", references)
        write_json(output / "bank.json", bank)
        if "reward_audit" in settings:
            shutil.copyfile(
                project_path(settings["reward_audit"]), output / "reward_audit.json"
            )
        for origin, name in (
            (source / "bank_cache.pt", "bank_cache.pt"),
            (source / "generation.json", "source_generation.json"),
            (control / "generation.json", "control_generation.json"),
            (control / "costs.json", "control_costs.json"),
            (control / "initial.pt", "initial.pt"),
            (control / "initial_training.jsonl", "initial_training.jsonl"),
        ):
            shutil.copyfile(origin, output / name)
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            fixture=fixture,
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        for path in Path(__file__).parent.glob("*.py"):
            provenance["files"][str(path.resolve())] = file_hash(path)
        write_json(output / "provenance.json", provenance)
        if backbone is None:
            print(
                "Loading frozen ProGen for the continuous-reward GFlowNet comparison",
                flush=True,
            )
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [row for row in references if row["role"] == "prompt"]
        costs = {"training": {}, "generation": {}, "cache": "reused pinned cache"}
        with (output / "samples.jsonl").open("w") as samples:
            for arm in ARMS:
                folder = output / arm
                folder.mkdir()
                if arm == "continuous_tb":
                    head, normalizer, costs["training"][arm] = train_continuous(
                        backbone,
                        prompts,
                        config,
                        settings,
                        reward,
                        bank,
                        cache,
                        initial,
                        folder,
                        predictor,
                    )
                else:
                    checkpoint = initial if arm == "unchanged" else control_checkpoint
                    head = make_context_head(
                        backbone, len(prompts), config, 0, "direct_context"
                    )
                    head.load_state_dict(checkpoint["head"])
                    normalizer = checkpoint["log_z"]
                    if normalizer is not None:
                        normalizer = normalizer.to(backbone.device)
                    origin = control / (
                        "unchanged" if arm == "unchanged" else "warm_tb"
                    )
                    checkpoint_path = origin / (
                        "checkpoint.pt"
                        if arm == "unchanged"
                        else f"step-{config['steps']}.pt"
                    )
                    shutil.copyfile(checkpoint_path, folder / "checkpoint.pt")
                    fits = json.loads((origin / "bank_fit.json").read_text())
                    write_json(
                        folder / "bank_fit.json", [dict(row, arm=arm) for row in fits]
                    )
                    if arm == "current_tb":
                        for name in (
                            "training.jsonl",
                            "fresh_training.jsonl",
                            "initialization.json",
                        ):
                            shutil.copyfile(origin / name, folder / name)
                    costs["training"][arm] = {
                        "seconds": 0,
                        "new_updates": 0,
                        "reused": True,
                    }
                generation_start, before = time.monotonic(), dict(backbone.work)
                print(f"{arm}: fresh evaluation generation", flush=True)
                records = generate_records(
                    backbone,
                    head,
                    prompts,
                    config,
                    0,
                    config["samples_per_prompt"],
                    arm,
                    torch.Generator(device=backbone.device).manual_seed(
                        settings["evaluation_seed"]
                    ),
                    normalizer,
                )
                for row in records:
                    samples.write(json.dumps(row, allow_nan=False) + "\n")
                samples.flush()
                costs["generation"][arm] = {
                    "seconds": time.monotonic() - generation_start,
                    "trajectories": len(records),
                    "backbone_work": work_delta(backbone, before),
                }
                write_json(output / "costs.json", costs)
                del head, normalizer
        captured = [
            path
            for path in output.rglob("*")
            if path.is_file() and path.name != "status.json"
        ]
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": 3 * len(prompts) * config["samples_per_prompt"],
                "files": {
                    str(path.relative_to(output)): file_hash(path) for path in captured
                },
            },
        )
        evaluate_continuous(output, output, predictor)
        write_json(
            output / "status.json",
            {
                "status": "complete",
                "fixture": fixture,
                "seconds": time.monotonic() - started,
            },
        )
    except Exception as error:
        write_json(
            output / "status.json",
            {"status": "failed", "fixture": fixture, "error": repr(error)},
        )
        raise
    return output / "report.md"


def reevaluate_continuous(run, output, predictor=predict_disorder):
    """Recover scoring on CPU from the self-contained snapshot, with no ProGen."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    output = new_run(output, {"source_run": str(run.resolve())})
    torch.set_num_threads(
        json.loads((run / "pilot_config.json").read_text())["cpu_threads"]
    )
    try:
        report = evaluate_continuous(run, output, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
