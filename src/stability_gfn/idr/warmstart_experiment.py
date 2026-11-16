"""Supervised-start TB versus continued supervision, with frozen ProGen.

Historical experiment modules stay unchanged so their pinned source hashes remain
valid. Reuse their update, caching and sampling primitives; this driver owns only
continuation initialization, budgets and its self-contained result snapshot.
"""

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..pipeline import project_path
from .acceptance_experiment import (
    BRANCH_MULTIPLIER,
    REWARD_VERSION,
    update_acceptance_head,
    verify_implementation,
    verify_manifest,
)
from .acceptance_reward import joint_acceptance_reward
from .branch_rate import branch_optimizer, set_rates
from .charge import rollout
from .charge_experiment import generate_records, work_delta
from .context_experiment import make_context_head
from .replay_data import CACHE_KEYS, bank_indices, cached_batch
from .replay_experiment import bank_fit, supervised_update
from .scoring import predict_disorder
from .warmstart_data import check_warmstart_inputs, initialize_normalizer
from .warmstart_evaluation import evaluate_warmstart

ARMS = ("unchanged", "continued_supervised", "warm_tb")


def train_continuation(
    backbone,
    prompts,
    config,
    settings,
    reward,
    bank,
    cache,
    arm,
    output,
    predictor,
    initial_checkpoint,
):
    """Match update and trajectory-example counts; fresh proposal costs differ."""
    seed = 0
    head = make_context_head(backbone, len(prompts), config, seed, "direct_context")
    head.load_state_dict(initial_checkpoint["head"])
    # Reset Adam for BOTH branches: source supervised moments are not appropriate
    # for the TB loss. Both restart the same head-rate schedule.
    initial_fit = bank_fit(head, None, bank, cache, config, arm, "initial", set())
    offsets, diagnostics = initialize_normalizer(
        initial_fit, bank, [r["id"] for r in prompts]
    )
    normalizer = torch.nn.Parameter(offsets.to(backbone.device))
    write_json(
        output / "initialization.json",
        {
            "optimizer": "fresh_adam",
            "initial_head_sha256": file_hash(output.parent / "initial.pt"),
            "normalizer": diagnostics if arm == "warm_tb" else None,
            "normalizer_rule": "equal_pass_fail_training_bank_least_squares"
            if arm == "warm_tb"
            else None,
        },
    )
    optimizer = branch_optimizer(head, normalizer, config, BRANCH_MULTIPLIER)
    # Preserve inherited head rates; the supervised objective has no log Z.
    if arm == "continued_supervised":
        optimizer.param_groups.pop()
    generator = torch.Generator(device=backbone.device).manual_seed(
        settings["training_seed"]
    )
    bank_generator = torch.Generator().manual_seed(settings["bank_sampling_seed"])
    started, before = time.monotonic(), dict(backbone.work)
    fresh_seen = set()
    fresh_count = 0
    scorer_seconds = 0.0
    fit_rows = bank_fit(
        head,
        None if arm == "continued_supervised" else normalizer,
        bank,
        cache,
        config,
        arm,
        "initial",
        fresh_seen,
    )
    with (
        (output / "training.jsonl").open("w") as log,
        (output / "fresh_training.jsonl").open("w") as fresh_log,
    ):
        for step in range(1, config["steps"] + 1):
            set_rates(optimizer, config, BRANCH_MULTIPLIER, step)
            selected = []
            fresh_passes = 0
            if arm == "continued_supervised":
                selected = bank_indices(
                    bank, len(prompts), bank_generator, step, supervised=True
                )
                batch = cached_batch(cache, selected, backbone.device)
                stats = supervised_update(head, optimizer, batch, config)
            else:
                prompt_indices = list(range(len(prompts)))
                fresh = rollout(
                    backbone,
                    head,
                    prompts,
                    prompt_indices,
                    config,
                    generator,
                    retain=True,
                )
                tails = [s[config["prefix_length"] :] for s in fresh["sequences"]]
                score_start = time.monotonic()
                predictions = np.asarray(predictor(tails, "V3"), dtype=float)
                values = joint_acceptance_reward(tails, predictions, reward)
                scorer_seconds += time.monotonic() - score_start
                fresh_count += len(tails)
                fresh_seen.update(tails)
                fresh_passes = int(values["joint_pass"].sum())
                for index, tail in enumerate(tails):
                    fresh_log.write(
                        json.dumps(
                            {
                                "step": step,
                                "prompt_id": prompts[prompt_indices[index]]["id"],
                                "tail": tail,
                                "isolated_tail_v3_scores": predictions[index].tolist(),
                                "log_prior": float(fresh["log_prior"][index]),
                                "behavior_log_policy": float(
                                    fresh["log_policy"][index]
                                ),
                                "joint_pass": bool(values["joint_pass"][index]),
                                "log_factor": float(values["log_factor"][index]),
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                fresh_log.flush()
                batch = {key: fresh[key] for key in CACHE_KEYS}
                factors = values["log_factor"].to(backbone.device)
                if arm == "warm_tb":
                    selected = bank_indices(bank, len(prompts), bank_generator, step)
                    stored = cached_batch(cache, selected, backbone.device)
                    batch = {
                        key: torch.cat([batch[key], stored[key]]) for key in CACHE_KEYS
                    }
                    factors = torch.cat(
                        [
                            factors,
                            torch.tensor(
                                [bank[i]["log_factor"] for i in selected],
                                device=backbone.device,
                            ),
                        ]
                    )
                    prompt_indices += [bank[i]["prompt_index"] for i in selected]
                stats = update_acceptance_head(
                    head, normalizer, optimizer, batch, prompt_indices, config, factors
                )
                del fresh
            stats.update(
                step=step,
                examples=len(batch["actions"]),
                fresh_examples=0 if arm == "continued_supervised" else len(prompts),
                fresh_joint_passes=fresh_passes,
                replay_ids=[bank[i]["id"] for i in selected],
                stored_successes=sum(bank[i]["joint_pass"] for i in selected),
                rates={g["name"]: g["lr"] for g in optimizer.param_groups},
                seconds=time.monotonic() - started,
            )
            if arm != "continued_supervised":
                stats["log_z"] = normalizer.detach().cpu().tolist()
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            del batch
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                loss_key = "token_nll" if arm == "continued_supervised" else "tb_loss"
                print(
                    f"{arm} step {step}/{config['steps']}: {loss_key} {stats[loss_key]:.4g}, fresh passes {fresh_passes}, stored successes {stats['stored_successes']}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    {
                        "arm": arm,
                        "architecture": "direct_context",
                        "reward_version": REWARD_VERSION,
                        "head": head.state_dict(),
                        "log_z": None
                        if arm == "continued_supervised"
                        else normalizer.detach().cpu(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": config,
                        "settings": settings,
                        "prompt_ids": [r["id"] for r in prompts],
                        "generator_state": generator.get_state(),
                        "bank_generator_state": bank_generator.get_state(),
                    },
                    output / f"step-{step}.pt",
                )
    fit_rows += bank_fit(
        head,
        None if arm == "continued_supervised" else normalizer,
        bank,
        cache,
        config,
        arm,
        "final",
        fresh_seen,
    )
    write_json(output / "bank_fit.json", fit_rows)
    costs = {
        "seconds": time.monotonic() - started,
        "scorer_seconds": scorer_seconds,
        "updates": config["steps"],
        "trajectory_examples": config["steps"] * 2 * len(prompts),
        "fresh_trajectories": fresh_count,
        "backbone_work": work_delta(backbone, before),
        "heldout_tails_seen_in_fresh_training": len(
            fresh_seen.intersection(r["tail"] for r in bank if r["split"] == "holdout")
        ),
    }
    return head, None if arm == "continued_supervised" else normalizer.detach(), costs


def run_warmstart(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """Two bounded continuations and fresh evaluation of all three heads."""
    source, config, references, provenance, reward, bank, cache, initial = (
        check_warmstart_inputs(settings, backbone is None)
    )
    output = new_run(output, settings)
    fixture, started = backbone is not None, time.monotonic()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(output / "pilot_config.json", config)
        write_json(output / "reward.json", asdict(reward))
        write_json(output / "references.json", references)
        write_json(output / "bank.json", bank)
        shutil.copyfile(source / "bank_cache.pt", output / "bank_cache.pt")
        shutil.copyfile(source / "generation.json", output / "source_generation.json")
        shutil.copyfile(source / "costs.json", output / "source_costs.json")
        shutil.copyfile(
            source / "supervised/training.jsonl", output / "initial_training.jsonl"
        )
        shutil.copyfile(
            source / f"supervised/step-{initial['step']}.pt", output / "initial.pt"
        )
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
            print("Loading frozen ProGen for supervised-start comparison", flush=True)
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [r for r in references if r["role"] == "prompt"]
        costs = {
            "training": {},
            "generation": {},
            "cache": "reused pinned source cache",
        }
        with (output / "samples.jsonl").open("w") as samples:
            for arm in ARMS:
                arm_dir = output / arm
                arm_dir.mkdir()
                if arm == "unchanged":
                    head = make_context_head(
                        backbone, len(prompts), config, 0, "direct_context"
                    )
                    head.load_state_dict(initial["head"])
                    log_z = None
                    # Persist the actual evaluated baseline, even though it is also initial.pt.
                    shutil.copyfile(output / "initial.pt", arm_dir / "checkpoint.pt")
                    fits = bank_fit(
                        head, None, bank, cache, config, arm, "initial", set()
                    )
                    write_json(
                        arm_dir / "bank_fit.json",
                        fits + [dict(row, stage="final") for row in fits],
                    )
                    costs["training"][arm] = {
                        "updates": 0,
                        "trajectory_examples": 0,
                        "fresh_trajectories": 0,
                        "seconds": 0,
                    }
                else:
                    head, log_z, costs["training"][arm] = train_continuation(
                        backbone,
                        prompts,
                        config,
                        settings,
                        reward,
                        bank,
                        cache,
                        arm,
                        arm_dir,
                        predictor,
                        initial,
                    )
                generation_start, before = time.monotonic(), dict(backbone.work)
                print(f"{arm}: generating fresh evaluation samples", flush=True)
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
                    log_z,
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
                del head, log_z
        captured = [
            str(p.relative_to(output))
            for p in output.rglob("*")
            if p.is_file() and p.name != "status.json"
        ]
        write_json(
            output / "generation.json",
            {
                "complete": True,
                "count": 3 * len(prompts) * config["samples_per_prompt"],
                "files": {name: file_hash(output / name) for name in captured},
            },
        )
        evaluate_warmstart(output, output, predictor)
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


def reevaluate_warmstart(run, output, predictor=predict_disorder):
    """CPU scoring recovery from pinned artifacts, without ProGen or source runs."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    output = new_run(output, {"source_run": str(run.resolve())})
    torch.set_num_threads(
        json.loads((run / "pilot_config.json").read_text())["cpu_threads"]
    )
    try:
        report = evaluate_warmstart(run, output, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
