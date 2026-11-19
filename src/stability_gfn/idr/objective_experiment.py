"""Capacity- and data-matched GFN versus reward-weighted supervised training."""

import copy
import json
import os
import shutil
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .context_experiment import DirectContextHead
from .lastblock_backbone import LastBlockProGen
from .lastblock_sampling import sample_records
from .objective_data import check_objective_inputs
from .objective_evaluation import evaluate_objective
from .objective_sampling import refresh_objective_pool
from .objective_training import ARMS, make_objective_optimizer, objective_update
from .replay_experiment import bank_fit
from .scoring import predict_disorder
from .warmstart_data import initialize_normalizer


def _write_rows(handle, rows):
    for row in rows:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()


def _save_policy(path, arm, policy, head, normalizer, prompts, step):
    """Atomically replace one analysis checkpoint; do not accumulate step copies.

    Only changed weights are saved. These are generation/analysis checkpoints,
    not optimizer/RNG resumptions. A failed write cannot destroy the prior file.
    """
    checkpoint = dict(
        head={key: value.detach().cpu() for key, value in head.state_dict().items()},
        last_block={
            key: value.detach().cpu()
            for key, value in policy.last_block.state_dict().items()
        },
        log_z=None if normalizer is None else normalizer.detach().cpu(),
        prompt_ids=[prompt["id"] for prompt in prompts],
        architecture="lastblock_direct_context",
        arm=arm,
        step=step,
        objective="trajectory_balance" if arm == "gfn" else "weighted_token_nll",
        reward_version="continuous_acceptance_v1",
    )
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _run_pair(
    folder,
    replica,
    config,
    settings,
    reference,
    original_head,
    offset,
    prompts,
    reward,
    train_bank,
    predictor,
):
    """Train two independent copied blocks on an identical refreshed pool."""
    folder.mkdir()
    write_json(folder / "config.json", settings)
    torch.manual_seed(settings["training_seed"])
    policies = {arm: LastBlockProGen(reference) for arm in ARMS}
    heads = {arm: copy.deepcopy(original_head).to(reference.device) for arm in ARMS}
    normalizer = torch.nn.Parameter(offset.clone().to(reference.device))
    optimizers = {
        arm: make_objective_optimizer(
            heads[arm],
            config,
            settings,
            policies[arm].last_block,
            normalizer if arm == "gfn" else None,
        )
        for arm in ARMS
    }
    generators = {
        arm: torch.Generator(device=reference.device).manual_seed(
            settings["training_seed"] + index
        )
        for index, arm in enumerate((*ARMS, "prior"))
    }
    bank_generator = torch.Generator().manual_seed(settings["bank_sampling_seed"])
    for arm in ARMS:
        (folder / arm).mkdir()
    write_json(
        folder / "trainable_parameters.json",
        dict(
            policies={
                arm: dict(
                    head=sum(
                        parameter.numel() for parameter in heads[arm].parameters()
                    ),
                    last_block=sum(
                        parameter.numel() for parameter in policies[arm].parameters()
                    ),
                    auxiliary_normalizers=len(prompts) if arm == "gfn" else 0,
                )
                for arm in ARMS
            },
            reference_trainable=sum(
                parameter.numel()
                for parameter in reference.model.parameters()
                if parameter.requires_grad
            ),
            final_normalization="frozen",
            output_projection="frozen",
        ),
    )
    costs = dict(
        training=dict(
            refresh_seconds=0.0,
            paired_update_seconds=0.0,
            proposal_generation_seconds={arm: 0.0 for arm in (*ARMS, "prior")},
            scoring_seconds=0.0,
            scorer_calls=0,
            scored_trajectories=0,
        ),
        generation={},
        interpretation="Coupled shared-data objective comparison, not independent pipelines",
    )
    started = time.monotonic()
    with ExitStack() as stack:
        logs = {
            arm: stack.enter_context((folder / arm / "training.jsonl").open("w"))
            for arm in ARMS
        }
        shared_log = stack.enter_context((folder / "shared_training.jsonl").open("w"))
        fresh_log = stack.enter_context((folder / "training_rows.jsonl").open("w"))
        for step in range(1, config["steps"] + 1):
            refresh_round = (step - 1) // settings["refresh_every"]
            if (step - 1) % settings["refresh_every"] == 0:
                tick = time.monotonic()
                pool, fresh, refresh_cost = refresh_objective_pool(
                    reference,
                    policies,
                    heads,
                    prompts,
                    config,
                    settings,
                    reward,
                    train_bank,
                    generators,
                    bank_generator,
                    refresh_round,
                    replica,
                    predictor,
                )
                costs["training"]["refresh_seconds"] += time.monotonic() - tick
                for arm, seconds in refresh_cost["generation_seconds"].items():
                    costs["training"]["proposal_generation_seconds"][arm] += seconds
                for key in ("scoring_seconds", "scorer_calls", "scored_trajectories"):
                    costs["training"][key] += refresh_cost[key]
                _write_rows(fresh_log, fresh)
            tick = time.monotonic()
            stats = objective_update(
                policies, heads, normalizer, optimizers, pool, config, settings, step
            )
            costs["training"]["paired_update_seconds"] += time.monotonic() - tick
            _write_rows(
                shared_log,
                [
                    dict(
                        step=step,
                        refresh_round=refresh_round,
                        example_ids=[row["id"] for row in pool],
                        counts=settings["shared_batch"],
                    )
                ],
            )
            for arm in ARMS:
                stats[arm]["seconds"] = time.monotonic() - started
                _write_rows(logs[arm], [stats[arm]])
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"{replica} step {step}/{config['steps']}: "
                    f"GFN TB {stats['gfn']['tb_loss']:.4g}, "
                    f"supervised NLL {stats['supervised']['weighted_nll']:.4g}; "
                    f"fresh joint {sum(row['joint_pass'] for row in fresh)}/{len(fresh)}",
                    flush=True,
                )
                write_json(folder / "costs.json", costs)
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                for arm in ARMS:
                    _save_policy(
                        folder / arm / "latest.pt",
                        arm,
                        policies[arm],
                        heads[arm],
                        normalizer if arm == "gfn" else None,
                        prompts,
                        step,
                    )
    costs["training"].update(
        seconds=time.monotonic() - started,
        updates_per_arm=config["steps"],
        examples_per_arm=config["steps"] * sum(settings["shared_batch"].values()),
        fresh_trajectories=3
        * len(prompts)
        * config["steps"]
        // settings["refresh_every"],
    )
    del optimizers
    adaptation = {}
    with torch.no_grad():
        for arm in ARMS:
            delta_squared, original_squared = 0.0, 0.0
            for current, original in zip(
                policies[arm].parameters(),
                reference.model.transformer.h[-1].parameters(),
            ):
                delta_squared += float((current - original).double().square().sum())
                original_squared += float(original.double().square().sum())
            adaptation[arm] = dict(
                last_block_delta_l2=delta_squared**0.5,
                last_block_relative_delta_l2=(delta_squared / original_squared) ** 0.5,
            )
            (folder / arm / "latest.pt").replace(folder / arm / "checkpoint.pt")
    adaptation["reference_has_gradients"] = any(
        parameter.grad is not None for parameter in reference.model.parameters()
    )
    if adaptation["reference_has_gradients"]:
        raise ValueError("Original reference acquired gradients")
    write_json(folder / "adaptation.json", adaptation)
    with (folder / "samples.jsonl").open("w") as handle:
        for arm in (*ARMS, "prior"):
            generator = torch.Generator(device=reference.device).manual_seed(
                settings["evaluation_seed"]
            )
            tick = time.monotonic()
            for repeat in range(config["samples_per_prompt"]):
                records = sample_records(
                    reference if arm == "prior" else policies[arm],
                    None if arm == "prior" else heads[arm],
                    prompts,
                    config,
                    arm,
                    generator,
                    repeat,
                    normalizer if arm == "gfn" else None,
                )
                for row in records:
                    row.update(id=f"{replica}-{row['id']}", replicate=replica)
                _write_rows(handle, records)
                if (repeat + 1) % 16 == 0 or repeat + 1 == config["samples_per_prompt"]:
                    print(
                        f"{replica} {arm}: {repeat + 1}/{config['samples_per_prompt']} outputs per prompt",
                        flush=True,
                    )
            costs["generation"][arm] = dict(
                seconds=time.monotonic() - tick,
                trajectories=len(prompts) * config["samples_per_prompt"],
            )
            write_json(folder / "costs.json", costs)
    costs["adapted_backbone_work"] = {arm: dict(policies[arm].work) for arm in ARMS}
    write_json(folder / "costs.json", costs)


def run_objective(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """Fit both objectives afresh, then score only final-checkpoint generation."""
    started = time.monotonic()
    inputs, plans = check_objective_inputs(settings, check_checkpoint=backbone is None)
    source, control, config, references, provenance, reward, bank, cache, initial, _ = (
        inputs
    )
    del inputs
    fixture = backbone is not None
    output = new_run(output, settings)
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        prompts = [row for row in references if row["role"] == "prompt"]
        for name, value in (
            ("pilot_config.json", config),
            ("references.json", references),
            ("reward.json", asdict(reward)),
            ("bank.json", bank),
            (
                "reward_definition.json",
                dict(version="continuous_acceptance_v1", penalty=2.5),
            ),
        ):
            write_json(output / name, value)
        for name in ("initial.pt", "initial_training.jsonl", "generation.json"):
            shutil.copy2(
                source / name,
                output
                / ("source_generation.json" if name == "generation.json" else name),
            )
        selected = [index for index, row in enumerate(bank) if row["split"] == "train"]
        train_bank = [bank[index] for index in selected]
        train_cache = {key: value[selected] for key, value in cache.items()}
        original_head = DirectContextHead(
            cache["features"].shape[-1],
            cache["context"].shape[-1],
            20,
            config["hidden_size"],
        )
        original_head.load_state_dict(initial["head"])
        initial_fit = bank_fit(
            original_head,
            None,
            train_bank,
            train_cache,
            config,
            "gfn",
            "initial",
            set(),
        )
        offset, diagnostics = initialize_normalizer(
            initial_fit, train_bank, [prompt["id"] for prompt in prompts]
        )
        write_json(
            output / "initialization.json",
            dict(
                normalizer=diagnostics,
                rule="GFN-only training-bank least-squares offsets; not exact log partitions",
                policy_initialization="Same original supervised head and original ProGen block for both arms",
            ),
        )
        del cache, train_cache, initial_fit
        provenance = copy.deepcopy(provenance)
        provenance.update(
            fixture=fixture,
            device=str(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        snapshot = output / "source"
        snapshot.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            provenance["files"][str(path.resolve())] = file_hash(path)
            shutil.copy2(path, snapshot / path.name)
        shutil.copy2(
            project_path("docs/experiments/idr_objective_2026-11-18.md"),
            snapshot / "protocol.md",
        )
        write_json(output / "provenance.json", provenance)
        if backbone is None:
            print(
                "Loading original ProGen; training matched GFN and supervised final-block copies",
                flush=True,
            )
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        reference_versions = [
            parameter._version for parameter in backbone.model.parameters()
        ]
        costs = dict(
            setup_seconds=time.monotonic() - started,
            inherited_setup="Original supervised initialization and scored bank are common sunk preparation; their acquisition is not included in this run's clock",
            comparison="Coupled shared examples; no independent pipeline efficiency claim",
        )
        for plan in plans:
            write_json(
                output / "status.json",
                dict(status="running", replicate=plan["name"], fixture=fixture),
            )
            _run_pair(
                output / plan["name"],
                plan["name"],
                config,
                plan["fit_settings"],
                backbone,
                original_head,
                offset,
                prompts,
                reward,
                train_bank,
                predictor,
            )
        if reference_versions != [
            parameter._version for parameter in backbone.model.parameters()
        ]:
            raise ValueError("Original reference parameters were modified")
        with (output / "samples.jsonl").open("w") as combined:
            for plan in plans:
                with (output / plan["name"] / "samples.jsonl").open() as child:
                    shutil.copyfileobj(child, combined)
        costs["generation_complete_seconds"] = time.monotonic() - started
        costs["original_backbone_work"] = dict(backbone.work)
        write_json(output / "costs.json", costs)
        verify_manifest(source)
        verify_manifest(control)
        verify_implementation(provenance)
        captured = [
            path
            for path in output.rglob("*")
            if path.is_file() and path.name != "status.json"
        ]
        write_json(
            output / "generation.json",
            dict(
                complete=True,
                count=len(plans) * 3 * len(prompts) * config["samples_per_prompt"],
                files={
                    str(path.relative_to(output)): file_hash(path) for path in captured
                },
            ),
        )
        tick = time.monotonic()
        report = evaluate_objective(output, output, predictor)
        write_json(
            output / "evaluation_cost.json", dict(seconds=time.monotonic() - tick)
        )
        write_json(
            output / "status.json",
            dict(
                status="complete",
                fixture=fixture,
                seconds=time.monotonic() - started,
            ),
        )
        return report
    except Exception as error:
        write_json(
            output / "status.json",
            dict(status="failed", fixture=fixture, error=repr(error)),
        )
        raise


def reevaluate_objective(run, output, predictor=predict_disorder):
    """Recover final scoring from sealed snapshots without ProGen or ancestor runs."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    config = json.loads((run / "pilot_config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    output = new_run(output, dict(source_run=str(run.resolve())))
    try:
        report = evaluate_objective(run, output, predictor)
        write_json(
            output / "status.json",
            dict(status="complete", source_fixture=provenance["fixture"]),
        )
        return report
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error=repr(error)))
        raise
