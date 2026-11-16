"""One-seed full-support transfer of the cached-bank GFlowNet variance repair."""

import copy
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
from .context_experiment import DirectContextHead
from .replay_experiment import bank_fit
from .scoring import predict_disorder
from .transfer_data import check_transfer_inputs
from .transfer_evaluation import ARMS, evaluate_transfer
from .transfer_head import GenerationMonitor, amplification_summary, condition_head
from .transfer_training import fold_check, make_proposal, train_transfer
from .warmstart_data import initialize_normalizer


def run_transfer(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    inputs = check_transfer_inputs(settings, check_checkpoint=backbone is None)
    source, config, references, provenance, reward, bank, cache, checkpoint = inputs
    fixture = backbone is not None
    output = new_run(output, settings)
    started = time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prompts = [r for r in references if r["role"] == "prompt"]
        for name, value in (
            ("pilot_config.json", config),
            ("reward.json", asdict(reward)),
            ("bank.json", bank),
            ("references.json", references),
        ):
            write_json(output / name, value)
        torch.save(cache, output / "bank_cache.pt")
        shutil.copy2(source / "initial.pt", output / "initial.pt")
        shutil.copy2(
            source / "initial_training.jsonl", output / "initial_training.jsonl"
        )
        shutil.copy2(source / "generation.json", output / "source_generation.json")
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            fixture=fixture,
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        snapshot = output / "source"
        snapshot.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            provenance["files"][str(path.resolve())] = file_hash(path)
            if path.name.startswith("transfer_"):
                shutil.copy2(path, snapshot / path.name)
        protocol = project_path("docs/experiments/idr_transfer_2026-11-16.md")
        shutil.copy2(protocol, snapshot / "protocol.md")
        write_json(output / "provenance.json", provenance)
        original = DirectContextHead(
            cache["features"].shape[-1],
            cache["context"].shape[-1],
            20,
            config["hidden_size"],
        )
        original.load_state_dict(checkpoint["head"])
        print("Preparing training-only full-trajectory conditioning", flush=True)
        conditioned, geometry = condition_head(
            original, cache, settings["conditioning_maximum_gain"]
        )
        write_json(output / "conditioning.json", geometry)
        initial_fit = bank_fit(
            original, None, bank, cache, config, "unchanged", "initial", set()
        )
        offsets, normalizer_diagnostics = initialize_normalizer(
            initial_fit, bank, [p["id"] for p in prompts]
        )
        weights, groups = make_proposal(bank, initial_fit, len(prompts))
        write_json(
            output / "initialization.json",
            dict(
                normalizer=normalizer_diagnostics,
                proposal_weights=weights.tolist(),
                proposal_ids=[r["id"] for r in bank],
                rule="half initial full-path policy normalized on bank, quarter each replay stratum",
            ),
        )
        costs = dict(
            training={},
            generation={},
            cache="reused full-path training-only cache; no new rollouts",
        )
        initial_fit = bank_fit(
            original, offsets, bank, cache, config, "unchanged", "initial", set()
        )
        for arm in ARMS:
            folder = output / arm
            folder.mkdir()
            if arm == "unchanged":
                head = copy.deepcopy(conditioned).to(device)
                normalizer = offsets.to(device)
                costs["training"][arm] = dict(
                    seconds=0, updates=0, fresh_training_rollouts=0
                )
            else:
                head, normalizer, costs["training"][arm] = train_transfer(
                    conditioned,
                    offsets,
                    bank,
                    cache,
                    weights,
                    groups,
                    config,
                    settings,
                    arm,
                    folder,
                    device,
                )
            folded, checks = fold_check(head, bank, cache, settings["microbatch_size"])
            if arm == "unchanged":
                folded = copy.deepcopy(original).to(device)
            fits = [dict(r, arm=arm) for r in initial_fit]
            fits += bank_fit(
                folded, normalizer, bank, cache, config, arm, "final", set()
            )
            write_json(folder / "bank_fit.json", fits)
            write_json(folder / "fold_check.json", checks)
            torch.save(
                dict(
                    head=folded.state_dict(),
                    log_z=normalizer.detach(),
                    architecture="direct_context",
                    arm=arm,
                    prompt_ids=[p["id"] for p in prompts],
                ),
                folder / "checkpoint.pt",
            )
            torch.save(
                dict(
                    head=head.state_dict(),
                    log_z=normalizer.detach(),
                    architecture="full_path_conditioned",
                ),
                folder / "conditioned.pt",
            )
            write_json(output / "costs.json", costs)
            del head, normalizer, folded
        if backbone is None:
            print("Loading frozen ProGen for fresh unrestricted evaluation", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        generation_diagnostics = {}
        with (output / "samples.jsonl").open("w") as samples:
            for arm in ARMS:
                saved = torch.load(
                    output / arm / "checkpoint.pt",
                    map_location=device,
                    weights_only=True,
                )
                folded = copy.deepcopy(original).to(device)
                folded.load_state_dict(saved["head"])
                head = copy.deepcopy(conditioned).to(device)
                head.load_state_dict(
                    torch.load(
                        output / arm / "conditioned.pt",
                        map_location=device,
                        weights_only=True,
                    )["head"]
                )
                monitor = GenerationMonitor(folded, head)
                before, generation_start = dict(backbone.work), time.monotonic()
                records = generate_records(
                    backbone,
                    monitor,
                    prompts,
                    config,
                    0,
                    config["samples_per_prompt"],
                    arm,
                    torch.Generator(device=backbone.device).manual_seed(
                        settings["evaluation_seed"]
                    ),
                    saved["log_z"],
                )
                for row in records:
                    samples.write(json.dumps(row, allow_nan=False) + "\n")
                samples.flush()
                generation_diagnostics[arm] = amplification_summary(monitor.gains)
                if len(monitor.gains) != len(records) * reward.tail_length:
                    raise ValueError("Missing fresh-state conditioning observations")
                costs["generation"][arm] = dict(
                    seconds=time.monotonic() - generation_start,
                    trajectories=len(records),
                    backbone_work=work_delta(backbone, before),
                )
                write_json(
                    output / "generation_diagnostics.json", generation_diagnostics
                )
                write_json(output / "costs.json", costs)
                del monitor, head, folded
        verify_manifest(source)
        captured = [
            p for p in output.rglob("*") if p.is_file() and p.name != "status.json"
        ]
        write_json(
            output / "generation.json",
            dict(
                complete=True,
                count=3 * len(prompts) * config["samples_per_prompt"],
                files={str(p.relative_to(output)): file_hash(p) for p in captured},
            ),
        )
        report = evaluate_transfer(output, output, predictor)
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


def reevaluate_transfer(run, output, predictor=predict_disorder):
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    output = new_run(output, dict(source_run=str(run.resolve())))
    try:
        report = evaluate_transfer(run, output, predictor)
        write_json(
            output / "status.json",
            dict(status="complete", source_fixture=provenance["fixture"]),
        )
        return report
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise
