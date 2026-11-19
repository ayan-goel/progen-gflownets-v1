"""Matched GFlowNets with frozen versus adapted final ProGen representations."""

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
from .context_experiment import DirectContextHead
from .lastblock_backbone import LastBlockProGen
from .lastblock_data import check_lastblock_inputs
from .lastblock_evaluation import ARMS, evaluate_lastblock
from .lastblock_sampling import refresh_pool, sample_records
from .lastblock_training import bank_fits, make_optimizer, paired_update
from .replay_experiment import bank_fit
from .scoring import predict_disorder
from .warmstart_data import initialize_normalizer


def _write_rows(handle, rows):
    for row in rows:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()


def _save_policy(folder, name, head, normalizer, prompts, step, adapted=None):
    """Analysis checkpoints contain all changed policy weights, not a full PLM copy."""
    checkpoint = dict(
        head={k: v.detach().cpu() for k, v in head.state_dict().items()},
        log_z=normalizer.detach().cpu(),
        prompt_ids=[p["id"] for p in prompts],
        architecture="lastblock_direct_context" if adapted else "direct_context",
        step=step,
        arm=folder.name,
        reward_version="continuous_acceptance_v1",
        reference="immutable original checkpoint in provenance.json",
    )
    if adapted is not None:
        checkpoint["last_block"] = {
            k: v.detach().cpu() for k, v in adapted.last_block.state_dict().items()
        }
    torch.save(checkpoint, folder / name)


def run_lastblock(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    inputs = check_lastblock_inputs(settings, check_checkpoint=backbone is None)
    (
        source,
        control,
        config,
        references,
        provenance,
        reward,
        bank,
        cache,
        initial,
        supervised,
    ) = inputs
    fixture = backbone is not None
    output = new_run(output, settings)
    started = time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(settings["training_seed"])
        prompts = [r for r in references if r["role"] == "prompt"]
        for name, value in (
            ("pilot_config.json", config),
            ("references.json", references),
            ("reward.json", asdict(reward)),
            ("bank.json", bank),
            (
                "reward_definition.json",
                dict(
                    version="continuous_acceptance_v1",
                    penalty=settings["penalty"],
                    formula="log R = log p0 - penalty * sum(violations); p0 is always original frozen ProGen",
                ),
            ),
        ):
            write_json(output / name, value)
        for folder, old, new in (
            (source, "initial.pt", "initial.pt"),
            (source, "initial_training.jsonl", "initial_training.jsonl"),
            (control, settings["control_checkpoint"], "supervised.pt"),
            (
                control,
                "continued_supervised/training.jsonl",
                "supervised_training.jsonl",
            ),
            (source, "generation.json", "source_generation.json"),
            (control, "generation.json", "control_generation.json"),
        ):
            shutil.copy2(folder / old, output / new)
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
            if path.name.startswith("lastblock_"):
                shutil.copy2(path, snapshot / path.name)
        shutil.copy2(
            project_path("docs/experiments/idr_lastblock_2026-11-16.md"),
            snapshot / "protocol.md",
        )
        write_json(output / "provenance.json", provenance)

        # Only inherited training rows initialize offsets; holdout outcomes never
        # determine optimizer settings, normalizers or checkpoint selection.
        selected = [i for i, row in enumerate(bank) if row["split"] == "train"]
        train_bank = [bank[i] for i in selected]
        train_cache = {k: v[selected] for k, v in cache.items()}
        original = DirectContextHead(
            cache["features"].shape[-1],
            cache["context"].shape[-1],
            20,
            config["hidden_size"],
        )
        original.load_state_dict(initial["head"])
        supervised_head = copy.deepcopy(original)
        supervised_head.load_state_dict(supervised["head"])
        offsets, diagnostics = {}, {}
        for arm, head in (
            ("unchanged", original),
            ("continued_supervised", supervised_head),
        ):
            fit = bank_fit(
                head, None, train_bank, train_cache, config, arm, "initial", set()
            )
            offsets[arm], diagnostics[arm] = initialize_normalizer(
                fit, train_bank, [p["id"] for p in prompts]
            )
        write_json(
            output / "initialization.json",
            dict(
                offsets=diagnostics,
                rule="training-only equal pass/fail least-squares offsets, not true log partition estimates",
                conditioning="none; original DirectContextHead parameterization in both arms",
            ),
        )
        del cache, train_cache
        if backbone is None:
            print(
                "Loading original ProGen; adapting a copy of its final block only",
                flush=True,
            )
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        adapted = LastBlockProGen(backbone)
        heads = dict(
            unchanged=original.to(backbone.device),
            continued_supervised=supervised_head.to(backbone.device),
            frozen_tb=copy.deepcopy(original).to(backbone.device),
            lastblock_tb=copy.deepcopy(original).to(backbone.device),
        )
        normalizers = {
            arm: torch.nn.Parameter(offsets["unchanged"].clone().to(backbone.device))
            for arm in ("frozen_tb", "lastblock_tb")
        }
        optimizers = {
            arm: make_optimizer(
                heads[arm],
                normalizers[arm],
                config,
                settings,
                adapted.last_block if arm == "lastblock_tb" else None,
            )
            for arm in normalizers
        }
        generators = {
            arm: torch.Generator(device=backbone.device).manual_seed(
                settings["training_seed"] + index
            )
            for index, arm in enumerate(("frozen_tb", "lastblock_tb", "prior"))
        }
        bank_generator = torch.Generator().manual_seed(settings["bank_sampling_seed"])
        for arm in ARMS:
            (output / arm).mkdir()
        write_json(
            output / "trainable_parameters.json",
            dict(
                original_reference=sum(p.numel() for p in backbone.model.parameters()),
                reference_trainable=sum(
                    p.numel() for p in backbone.model.parameters() if p.requires_grad
                ),
                head=sum(p.numel() for p in original.parameters()),
                adapted_block=sum(p.numel() for p in adapted.parameters()),
                frozen_lower_blocks=len(backbone.model.transformer.h) - 1,
                output_projection="frozen",
                final_normalization="frozen",
            ),
        )
        costs = dict(
            training=dict(refresh_seconds=0.0, paired_update_seconds=0.0), generation={}
        )
        fresh_seen = set()
        training_start = time.monotonic()
        with (
            (output / "shared_training.jsonl").open("w") as shared_log,
            (output / "training_rows.jsonl").open("w") as fresh_log,
            (output / "frozen_tb/training.jsonl").open("w") as frozen_log,
            (output / "lastblock_tb/training.jsonl").open("w") as adapted_log,
        ):
            logs = {"frozen_tb": frozen_log, "lastblock_tb": adapted_log}
            for step in range(1, config["steps"] + 1):
                refresh_round = (step - 1) // settings["refresh_every"]
                if (step - 1) % settings["refresh_every"] == 0:
                    tick = time.monotonic()
                    pool, fresh = refresh_pool(
                        backbone,
                        adapted,
                        heads,
                        prompts,
                        config,
                        settings,
                        reward,
                        train_bank,
                        generators,
                        bank_generator,
                        refresh_round,
                        predictor,
                    )
                    costs["training"]["refresh_seconds"] += time.monotonic() - tick
                    fresh_seen.update(r["tail"] for r in fresh)
                    _write_rows(fresh_log, fresh)
                tick = time.monotonic()
                stats = paired_update(
                    adapted,
                    heads,
                    normalizers,
                    optimizers,
                    pool,
                    config,
                    settings,
                    step,
                )
                costs["training"]["paired_update_seconds"] += time.monotonic() - tick
                _write_rows(
                    shared_log,
                    [
                        dict(
                            step=step,
                            refresh_round=refresh_round,
                            example_ids=[r["id"] for r in pool],
                            counts=settings["shared_batch"],
                        )
                    ],
                )
                for arm in normalizers:
                    stats[arm]["seconds"] = time.monotonic() - training_start
                    _write_rows(logs[arm], [stats[arm]])
                if (
                    step == 1
                    or step % config["log_every"] == 0
                    or step == config["steps"]
                ):
                    print(
                        f"shared step {step}/{config['steps']}: frozen TB {stats['frozen_tb']['tb_loss']:.4g}, last-block TB {stats['lastblock_tb']['tb_loss']:.4g}; fresh joint {sum(r['joint_pass'] for r in fresh)}/{len(fresh)}",
                        flush=True,
                    )
                    write_json(output / "costs.json", costs)
                if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                    for arm in normalizers:
                        _save_policy(
                            output / arm,
                            f"step-{step}.pt",
                            heads[arm],
                            normalizers[arm],
                            prompts,
                            step,
                            adapted if arm == "lastblock_tb" else None,
                        )
        costs["training"].update(
            seconds=time.monotonic() - training_start,
            updates_per_arm=config["steps"],
            examples_per_arm=config["steps"] * sum(settings["shared_batch"].values()),
            fresh_trajectories=3
            * len(prompts)
            * config["steps"]
            // settings["refresh_every"],
        )
        # Discard optimizer states before evaluation; no fitting uses heldout results.
        del optimizers
        delta_sq = initial_sq = 0.0
        with torch.no_grad():
            for current, original_parameter in zip(
                adapted.parameters(), backbone.model.transformer.h[-1].parameters()
            ):
                delta_sq += float(
                    (current - original_parameter).double().square().sum()
                )
                initial_sq += float(original_parameter.double().square().sum())
        write_json(
            output / "adaptation.json",
            dict(
                last_block_delta_l2=delta_sq**0.5,
                last_block_relative_delta_l2=(delta_sq / initial_sq) ** 0.5,
                reference_has_gradients=any(
                    p.grad is not None for p in backbone.model.parameters()
                ),
            ),
        )
        with (output / "samples.jsonl").open("w") as samples:
            for arm in ARMS:
                policy = adapted if arm == "lastblock_tb" else backbone
                log_z = (
                    normalizers[arm]
                    if arm in normalizers
                    else offsets[arm].to(backbone.device)
                )
                _save_policy(
                    output / arm,
                    "checkpoint.pt",
                    heads[arm],
                    log_z,
                    prompts,
                    config["steps"] if arm in normalizers else 0,
                    adapted if arm == "lastblock_tb" else None,
                )
                tick = time.monotonic()
                fits = bank_fits(
                    policy,
                    heads[arm],
                    log_z,
                    bank,
                    config,
                    prompts,
                    arm,
                    fresh_seen,
                    settings["microbatch_size"],
                )
                write_json(output / arm / "bank_fit.json", fits)
                costs.setdefault("diagnostics", {})[arm] = dict(
                    seconds=time.monotonic() - tick, trajectories=len(bank)
                )
                generator = torch.Generator(device=backbone.device).manual_seed(
                    settings["evaluation_seed"]
                )
                tick = time.monotonic()
                for repeat in range(config["samples_per_prompt"]):
                    records = sample_records(
                        policy,
                        heads[arm],
                        prompts,
                        config,
                        arm,
                        generator,
                        repeat,
                        log_z,
                    )
                    _write_rows(samples, records)
                    if (repeat + 1) % 16 == 0 or repeat + 1 == config[
                        "samples_per_prompt"
                    ]:
                        print(
                            f"{arm}: {repeat + 1}/{config['samples_per_prompt']} fresh outputs per prompt",
                            flush=True,
                        )
                costs["generation"][arm] = dict(
                    seconds=time.monotonic() - tick,
                    trajectories=len(prompts) * config["samples_per_prompt"],
                )
                write_json(output / "costs.json", costs)
        costs["backbone_work"] = dict(
            original_reference=backbone.work, adapted_shared=adapted.work
        )
        write_json(output / "costs.json", costs)
        verify_manifest(source)
        verify_manifest(control)
        captured = [
            p for p in output.rglob("*") if p.is_file() and p.name != "status.json"
        ]
        write_json(
            output / "generation.json",
            dict(
                complete=True,
                count=len(ARMS) * len(prompts) * config["samples_per_prompt"],
                files={str(p.relative_to(output)): file_hash(p) for p in captured},
            ),
        )
        tick = time.monotonic()
        report = evaluate_lastblock(output, output, predictor)
        write_json(
            output / "evaluation_cost.json", dict(seconds=time.monotonic() - tick)
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


def reevaluate_lastblock(run, output, predictor=predict_disorder):
    """Recover scoring from snapshots only; no ProGen load or training restart."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    config = json.loads((run / "pilot_config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    output = new_run(output, dict(source_run=str(run.resolve())))
    try:
        report = evaluate_lastblock(run, output, predictor)
        write_json(
            output / "status.json",
            dict(status="complete", source_fixture=provenance["fixture"]),
        )
        return report
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise
