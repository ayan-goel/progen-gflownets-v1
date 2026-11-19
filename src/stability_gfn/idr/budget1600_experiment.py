"""Bounded 1,600-update reruns with fresh, preserved 800-update controls.

The original paired trainer remains unchanged. New orchestration preserves the
completed 800-update experiment's live source hashes and scoring recovery while
carrying forward both earlier training histories for symmetric copy controls.
"""

import json
import os
import shutil
import time
from pathlib import Path

import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .budget1600_data import check_budget1600_inputs
from .budget1600_evaluation import evaluate_budget1600
from .confirmation_evaluation import evaluate_prior
from .confirmation_experiment import _capture_generation, _generate_prior
from .context_experiment import DirectContextHead
from .lastblock_backbone import LastBlockProGen
from .lastblock_evaluation import evaluate_lastblock, read_jsonl
from .lastblock_experiment import run_lastblock
from .lastblock_sampling import sample_records
from .scoring import predict_disorder


def _generate_controls(source_fit, fit, folder, device, backbone):
    """Sample the unchanged 800-update policy and p0 on a fresh common stream."""
    config = json.loads((fit / "pilot_config.json").read_text())
    settings = json.loads((fit / "config.json").read_text())
    references = json.loads((fit / "references.json").read_text())
    prompts = [row for row in references if row["role"] == "prompt"]
    checkpoint_path = source_fit / "lastblock_tb/checkpoint.pt"
    control = new_run(
        folder / "control800",
        dict(
            arm="lastblock_800",
            evaluation_seed=settings["evaluation_seed"],
            source_fit=str(source_fit.resolve()),
            source_checkpoint_sha256=file_hash(checkpoint_path),
        ),
    )
    started = time.monotonic()
    for old, new in (
        ("lastblock_tb/checkpoint.pt", "checkpoint.pt"),
        ("config.json", "source_settings.json"),
        ("bank.json", "bank.json"),
        ("training_rows.jsonl", "training_rows.jsonl"),
        ("generation.json", "source_generation.json"),
    ):
        shutil.copy2(source_fit / old, control / new)
    if backbone is None:
        print("Loading original ProGen for fresh saved-800 controls", flush=True)
        backbone = FrozenProGen.from_checkpoint(
            project_path(config["checkpoint"]), device, config["precision"]
        )
    checkpoint = torch.load(
        control / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    policy = LastBlockProGen(backbone)
    policy.last_block.load_state_dict(checkpoint["last_block"], strict=True)
    policy.last_block.requires_grad_(False)
    head = DirectContextHead(
        backbone.hidden_size, len(prompts) + 5, 20, config["hidden_size"]
    ).to(backbone.device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.requires_grad_(False)
    head.eval()
    normalizer = checkpoint["log_z"].to(backbone.device)
    generator = torch.Generator(device=backbone.device).manual_seed(
        settings["evaluation_seed"]
    )
    with (control / "samples.jsonl").open("w") as handle:
        for repeat in range(config["samples_per_prompt"]):
            records = sample_records(
                policy,
                head,
                prompts,
                config,
                "lastblock_800",
                generator,
                repeat,
                normalizer,
            )
            for row in records:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            if (repeat + 1) % 16 == 0 or repeat + 1 == config["samples_per_prompt"]:
                print(
                    f"saved 800-update control: {repeat + 1}/{config['samples_per_prompt']} per prompt",
                    flush=True,
                )
    for name, value in policy.last_block.state_dict().items():
        if not torch.equal(value.cpu(), checkpoint["last_block"][name]):
            raise ValueError("Saved 800-update control changed during generation")
    if any(parameter.grad is not None for parameter in backbone.model.parameters()):
        raise ValueError("The original reference received gradients")
    write_json(
        control / "costs.json",
        dict(
            seconds=time.monotonic() - started,
            trajectories=len(prompts) * config["samples_per_prompt"],
        ),
    )
    write_json(control / "status.json", dict(status="generated"))
    del policy, head, normalizer, checkpoint
    _generate_prior(fit, folder / "prior", device, backbone)


def run_budget1600(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """Rerun both paired seeds from their original supervised initialization.

    The preserved 800-update checkpoints are evaluation comparators only. Tiny
    injected CPU fixtures exercise this pipeline without pretrained inference.
    """
    source, plans, _, _, provenance = check_budget1600_inputs(
        settings, check_checkpoint=backbone is None
    )
    output = new_run(output, settings)
    started = time.monotonic()
    fixture = backbone is not None
    try:
        snapshot = output / "source"
        snapshot.mkdir()
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            fixture=fixture,
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        for path in sorted(Path(__file__).parent.glob("budget1600_*.py")):
            provenance["files"][str(path.resolve())] = file_hash(path)
            shutil.copy2(path, snapshot / path.name)
        shutil.copy2(
            Path(__file__).parents[1] / "__main__.py", snapshot / "__main__.py"
        )
        shutil.copy2(
            project_path("docs/experiments/idr_lastblock_1600_2026-11-18.md"),
            snapshot / "protocol.md",
        )
        for name in ("config.json", "generation.json"):
            shutil.copy2(source / name, output / f"budget_{name}")
        write_json(output / "provenance.json", provenance)
        write_json(
            output / "plan.json",
            [dict(plan, source_fit=str(plan["source_fit"])) for plan in plans],
        )
        costs = {}
        for index, plan in enumerate(plans, 1):
            name, fit_settings = plan["name"], plan["fit_settings"]
            folder = output / name
            folder.mkdir()
            # This older exposure must survive even if ancestor folders are later
            # unavailable. It contributes to copy detection, never new training.
            shutil.copy2(
                source / name / "control400/training_rows.jsonl",
                folder / "history400_training_rows.jsonl",
            )
            print(
                f"1600-budget comparison {index}/{len(plans)}: {name}, "
                f"{fit_settings['training_steps']} updates from original initialization",
                flush=True,
            )
            write_json(
                output / "status.json",
                dict(
                    status="running",
                    replicate=name,
                    stage="paired_fit",
                    fixture=fixture,
                ),
            )
            tick = time.monotonic()
            run_lastblock(fit_settings, folder / "fit", device, backbone, predictor)
            fit_seconds = time.monotonic() - tick
            write_json(
                output / "status.json",
                dict(
                    status="running",
                    replicate=name,
                    stage="fresh_controls",
                    fixture=fixture,
                ),
            )
            tick = time.monotonic()
            _generate_controls(
                plan["source_fit"], folder / "fit", folder, device, backbone
            )
            costs[name] = dict(
                paired_fit_and_evaluation_seconds=fit_seconds,
                fresh_control_generation_seconds=time.monotonic() - tick,
            )
            write_json(output / "costs.json", dict(replicates=costs))

        count = 0
        with (output / "samples.jsonl").open("w") as handle:
            for plan in plans:
                name = plan["name"]
                for part in ("fit", "control800", "prior"):
                    for row in read_jsonl(output / name / part / "samples.jsonl"):
                        row = dict(row, id=f"{name}/{row['id']}", replicate=name)
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        count += 1
        verify_manifest(source)
        # Seal all generations and inherited histories before extra scoring.
        # A scoring failure can then recover without fitting or generating again.
        _capture_generation(output, count)
        scoring_costs = {}
        for plan in plans:
            folder = output / plan["name"]
            tick = time.monotonic()
            evaluate_prior(folder / "fit", folder / "prior", predictor=predictor)
            scoring_costs[plan["name"]] = time.monotonic() - tick
        tick = time.monotonic()
        report = evaluate_budget1600(output, output, predictor=predictor)
        write_json(
            output / "evaluation_cost.json",
            dict(
                prior_scoring_seconds=scoring_costs,
                saved_control_scoring_and_comparison_seconds=time.monotonic() - tick,
            ),
        )
        for plan in plans:
            for part in ("control800", "prior"):
                write_json(
                    output / plan["name"] / part / "status.json",
                    dict(status="complete"),
                )
        write_json(
            output / "status.json",
            dict(
                status="complete", fixture=fixture, seconds=time.monotonic() - started
            ),
        )
        return report
    except Exception as error:
        write_json(
            output / "status.json",
            dict(status="failed", fixture=fixture, error=repr(error)),
        )
        raise


def reevaluate_budget1600(run, output, predictor=predict_disorder):
    """Score sealed generation using only its snapshots, without ProGen or fitting."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    settings = json.loads((run / "config.json").read_text())
    output = new_run(output, dict(source_run=str(run.resolve())))
    try:
        for replica in settings["replicates"]:
            name = replica["name"]
            source, destination = run / name, output / name
            (destination / "fit").mkdir(parents=True)
            (destination / "prior").mkdir()
            config = json.loads((source / "fit/pilot_config.json").read_text())
            torch.set_num_threads(config["cpu_threads"])
            evaluate_lastblock(source / "fit", destination / "fit", predictor)
            evaluate_prior(
                source / "fit", source / "prior", destination / "prior", predictor
            )
        report = evaluate_budget1600(run, output, predictor=predictor)
        write_json(
            output / "status.json",
            dict(status="complete", source_fixture=provenance["fixture"]),
        )
        return report
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise
