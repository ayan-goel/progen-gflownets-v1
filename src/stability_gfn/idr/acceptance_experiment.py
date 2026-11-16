"""Joint-acceptance TB fits with frozen ProGen and fresh comparator sampling.

Only the reward changes relative to the completed TB control. Its fitted heads
are reused, never retrained. Historical experiment modules remain unchanged.
"""

import importlib.metadata
import json
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_hash, new_run, write_json
from ..backbone import FrozenProGen
from ..guidance import trajectory_balance
from ..pipeline import project_path
from .acceptance_evaluation import evaluate_acceptance
from .acceptance_reward import AcceptanceRewardConfig, joint_acceptance_reward
from .baseline import scorer_files
from .branch_rate import branch_optimizer, set_rates
from .charge import path_log_probabilities, rollout
from .charge_experiment import check_charge_inputs, generate_records, work_delta
from .context_experiment import make_context_head
from .scoring import predict_disorder

REWARD_VERSION = "joint_acceptance_v1"
BRANCH_MULTIPLIER = 10
TRAIN_SEED_OFFSET = 30000
PRIOR_SEED_OFFSET = 60000
EVALUATION_SEED_OFFSET = 70000


def verify_manifest(run):
    """Check a completed generation snapshot, including all captured inputs."""
    manifest = json.loads((run / "generation.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Generation is incomplete")
    for name, digest in manifest["files"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Captured artifact changed: {name}")
    if len((run / "samples.jsonl").read_text().splitlines()) != manifest["count"]:
        raise ValueError("Sample count differs from generation manifest")
    return manifest


def verify_implementation(provenance):
    """Verify software/scorer identity without loading a model or querying CUDA."""
    for name, version in provenance["packages"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Package differs from source: {name}")
    scorer_paths = set(scorer_files())
    for filename, digest in provenance["files"].items():
        path = Path(filename)
        if path.suffix == ".py" or path in scorer_paths:
            if file_hash(path) != digest:
                raise ValueError(f"Implementation/scorer changed: {path}")


def check_acceptance_inputs(settings, check_checkpoint=True):
    """Pin controls, support, acceptance criteria and inherited training budget."""
    source = project_path(settings["source_run"])
    if file_hash(source / "generation.json") != settings["source_generation_sha256"]:
        raise ValueError("Completed TB control manifest changed")
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("TB control is not complete")
    verify_manifest(source)
    provenance = json.loads((source / "provenance.json").read_text())
    verify_implementation(provenance)
    if check_checkpoint and provenance["fixture"]:
        raise ValueError("Real runs cannot use fixture controls")
    config = json.loads((source / "pilot_config.json").read_text())
    references, _ = check_charge_inputs(config, check_checkpoint)
    if references != json.loads((source / "references.json").read_text()):
        raise ValueError("Reference panel differs from the control")
    control_settings = json.loads((source / "config.json").read_text())
    if (
        config["beta"] != 2
        or control_settings["branch_rate_multiplier"] != BRANCH_MULTIPLIER
    ):
        raise ValueError("Need the completed beta-2, tenfold-branch TB control")
    if settings["reward_version"] != REWARD_VERSION:
        raise ValueError("Unknown acceptance reward version")
    reward = AcceptanceRewardConfig(**settings["reward"])
    expected = AcceptanceRewardConfig()
    if not check_checkpoint:
        expected = replace(
            expected, tail_length=config["length"] - config["prefix_length"]
        )
    # JSON changes tuples to lists. Compare canonical JSON instead of container types.
    if json.dumps(asdict(reward), sort_keys=True) != json.dumps(
        asdict(expected), sort_keys=True
    ):
        raise ValueError("This bounded experiment fixes the audited reward constants")
    if reward.tail_length != config["length"] - config["prefix_length"]:
        raise ValueError("Reward and generation lengths differ")
    for key in ("block_targets", "block_tolerance", "fcr_target", "fcr_tolerance"):
        if json.dumps(getattr(reward, key)) != json.dumps(config["reward"][key]):
            raise ValueError(f"Charge pass criterion differs from control: {key}")
    for key in ("disorder_cutoff", "minimum_disordered_fraction"):
        if getattr(reward, key) != config[key]:
            raise ValueError(f"Disorder criterion differs from control: {key}")
    prompts = [r["id"] for r in references if r["role"] == "prompt"]
    for seed in config["seeds"]:
        checkpoint = torch.load(
            source / f"tb/seed-{seed}/step-{config['steps']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        if (
            checkpoint["method"] != "tb"
            or checkpoint["architecture"] != "direct_context"
            or checkpoint["step"] != config["steps"]
            or checkpoint["config"] != config
            or checkpoint["prompt_ids"] != prompts
            or checkpoint["settings"] != control_settings
        ):
            raise ValueError(f"TB checkpoint/protocol mismatch for seed {seed}")
    if check_checkpoint:
        for name in ("config.json", "pytorch_model.bin"):
            path = project_path(config["checkpoint"]) / name
            if file_hash(path) != provenance["files"].get(str(path.resolve())):
                raise ValueError("Frozen model differs from completed control")
    return source, config, references, provenance


def update_acceptance_head(head, log_z, optimizer, batch, indices, config, log_factor):
    """TB with log R = log p0 + log factor; prior once, effective beta one."""
    log_q = path_log_probabilities(head, batch)
    factor = log_factor.detach().to(device=log_q.device, dtype=log_q.dtype)
    loss = trajectory_balance(log_q, batch["log_prior"], factor, log_z[indices], 1.0)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite acceptance TB loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [*head.parameters(), log_z], config["gradient_clip"], error_if_nonfinite=True
    )
    optimizer.step()
    return {
        "tb_loss": float(loss.detach()),
        "gradient_norm_before_clip": float(norm),
        "mean_log_q_minus_log_p0": float((log_q.detach() - batch["log_prior"]).mean()),
    }


def train_acceptance_head(backbone, prompts, config, reward, seed, output, predictor):
    """Fresh on-policy trajectories, one update per batch, no replay or fine-tuning."""
    head = make_context_head(backbone, len(prompts), config, seed, "direct_context")
    log_z = torch.nn.Parameter(torch.zeros(len(prompts), device=backbone.device))
    optimizer = branch_optimizer(head, log_z, config, BRANCH_MULTIPLIER)
    generator = torch.Generator(device=backbone.device).manual_seed(
        TRAIN_SEED_OFFSET + seed
    )
    indices = list(range(len(prompts)))
    started, before = time.monotonic(), dict(backbone.work)
    scorer_seconds = 0.0
    with (
        (output / "training.jsonl").open("w") as log,
        (output / "reward_inputs.jsonl").open("w") as inputs,
    ):
        for step in range(1, config["steps"] + 1):
            set_rates(optimizer, config, BRANCH_MULTIPLIER, step)
            batch = rollout(
                backbone, head, prompts, indices, config, generator, retain=True
            )
            tails = [s[config["prefix_length"] :] for s in batch["sequences"]]
            scoring_start = time.monotonic()
            # Real adapter explicitly uses CPU, unrounded metapredict 3.0.2 V3.
            with torch.no_grad():
                predictions = np.asarray(predictor(tails, "V3"), dtype=float)
                values = joint_acceptance_reward(tails, predictions, reward)
            scoring_seconds = time.monotonic() - scoring_start
            scorer_seconds += scoring_seconds
            stats = update_acceptance_head(
                head, log_z, optimizer, batch, indices, config, values["log_factor"]
            )
            for index, tail in enumerate(tails):
                row = {
                    "step": step,
                    "seed": seed,
                    "prompt_id": prompts[index]["id"],
                    "tail": tail,
                    "isolated_tail_v3_scores": predictions[index].tolist(),
                    "log_prior": float(batch["log_prior"][index]),
                    "behavior_log_policy": float(batch["log_policy"][index]),
                    "log_factor": float(values["log_factor"][index]),
                    "violations": values["violations"][index].tolist(),
                    "joint_pass": bool(values["joint_pass"][index]),
                }
                inputs.write(json.dumps(row, allow_nan=False) + "\n")
            inputs.flush()
            stats.update(
                step=step,
                mean_log_factor=float(values["log_factor"].mean()),
                charge_success=float(values["charge_pass"].double().mean()),
                disorder_success=float(values["disorder_pass"].double().mean()),
                joint_success=float(values["joint_pass"].double().mean()),
                joint_count=int(values["joint_pass"].sum()),
                mean_violations=values["violations"].mean(0).tolist(),
                scoring_seconds=scoring_seconds,
                seconds=time.monotonic() - started,
                rates={g["name"]: g["lr"] for g in optimizer.param_groups},
                log_z=log_z.detach().cpu().tolist(),
                branch_weight_norm=float(head.task_weight.detach().norm()),
            )
            log.write(json.dumps(stats, allow_nan=False) + "\n")
            log.flush()
            del batch
            if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
                print(
                    f"acceptance seed {seed} step {step}/{config['steps']}: TB {stats['tb_loss']:.4g}, charge {stats['charge_success']:.1%}, joint {stats['joint_success']:.1%}",
                    flush=True,
                )
            if step % config["checkpoint_every"] == 0 or step == config["steps"]:
                torch.save(
                    {
                        "architecture": "direct_context",
                        "reward_version": REWARD_VERSION,
                        "reward_settings": asdict(reward),
                        "head": head.state_dict(),
                        "log_z": log_z.detach(),
                        "optimizer": optimizer.state_dict(),
                        "generator_state": generator.get_state(),
                        "step": step,
                        "prompt_ids": [r["id"] for r in prompts],
                        "config": config,
                    },
                    output / f"step-{step}.pt",
                )
    return (
        head,
        log_z,
        {
            "seconds": time.monotonic() - started,
            "scorer_seconds": scorer_seconds,
            "trajectories": config["steps"] * len(prompts),
            "v3_tail_predictions": config["steps"] * len(prompts),
            "backbone_work": work_delta(backbone, before),
        },
    )


def run_acceptance(
    settings, output, device="cuda", backbone=None, predictor=predict_disorder
):
    """Train three new heads; sample old heads and prior afresh; evaluate on CPU."""
    source, original_config, references, provenance = check_acceptance_inputs(
        settings, backbone is None
    )
    output = new_run(output, settings)
    started, fixture = time.monotonic(), backbone is not None
    try:
        config = dict(original_config, beta=1.0, reward_version=REWARD_VERSION)
        reward = AcceptanceRewardConfig(**settings["reward"])
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(output / "pilot_config.json", config)
        write_json(output / "source_config.json", original_config)
        write_json(output / "references.json", references)
        for name in ("generation.json", "costs.json"):
            shutil.copyfile(source / name, output / f"source_{name}")
        (output / "controls").mkdir()
        for seed in config["seeds"]:
            shutil.copyfile(
                source / f"tb/seed-{seed}/step-{config['steps']}.pt",
                output / f"controls/seed-{seed}.pt",
            )
        provenance = dict(
            provenance,
            files=dict(provenance["files"]),
            device=str(device),
            fixture=fixture,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
        for path in Path(__file__).parent.glob("*.py"):
            provenance["files"][str(path.resolve())] = file_hash(path)
        write_json(output / "provenance.json", provenance)
        if backbone is None:
            print(
                "Loading frozen ProGen for the joint-acceptance reward comparison",
                flush=True,
            )
            backbone = FrozenProGen.from_checkpoint(
                project_path(config["checkpoint"]), device, config["precision"]
            )
        prompts = [r for r in references if r["role"] == "prompt"]
        costs = {}
        with (output / "samples.jsonl").open("w") as handle:
            for seed in config["seeds"]:
                seed_dir = output / f"seed-{seed}"
                seed_dir.mkdir()
                head, log_z, training = train_acceptance_head(
                    backbone, prompts, config, reward, seed, seed_dir, predictor
                )
                costs[str(seed)] = {
                    "new_tb_training": training,
                    "old_tb_training": {
                        "seconds": 0,
                        "trajectories": 0,
                        "reused_checkpoint": f"controls/seed-{seed}.pt",
                        "original_cost_record": "source_costs.json",
                    },
                }
                for arm in ("new_tb", "old_tb", "prior_pool"):
                    stage_start, before = time.monotonic(), dict(backbone.work)
                    if arm == "old_tb":
                        checkpoint = torch.load(
                            output / f"controls/seed-{seed}.pt",
                            map_location=backbone.device,
                            weights_only=True,
                        )
                        head.load_state_dict(checkpoint["head"])
                        log_z = checkpoint["log_z"]
                    is_prior = arm == "prior_pool"
                    rows = generate_records(
                        backbone,
                        None if is_prior else head,
                        prompts,
                        config,
                        seed,
                        config["samples_per_prompt"]
                        * (config["selection_factor"] if is_prior else 1),
                        arm,
                        torch.Generator(device=backbone.device).manual_seed(
                            (PRIOR_SEED_OFFSET if is_prior else EVALUATION_SEED_OFFSET)
                            + seed
                        ),
                        None if is_prior else log_z,
                    )
                    costs[str(seed)][arm] = {
                        "seconds": time.monotonic() - stage_start,
                        "trajectories": len(rows),
                        "backbone_work": work_delta(backbone, before),
                    }
                    for row in rows:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
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
                "count": len(config["seeds"])
                * len(prompts)
                * config["samples_per_prompt"]
                * (2 + config["selection_factor"]),
                "files": {name: file_hash(output / name) for name in captured},
                "seconds_before_evaluation": time.monotonic() - started,
            },
        )
        evaluate_acceptance(output, output, predictor)
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


def reevaluate_acceptance(run, output, predictor=predict_disorder):
    """Recover CPU scoring from captured generation; never reload ProGen or train."""
    run = Path(run)
    verify_manifest(run)
    provenance = json.loads((run / "provenance.json").read_text())
    verify_implementation(provenance)
    output = new_run(output, {"source_run": str(run.resolve())})
    config = json.loads((run / "pilot_config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    try:
        report = evaluate_acceptance(run, output, predictor)
        write_json(
            output / "status.json",
            {"status": "complete", "source_fixture": provenance["fixture"]},
        )
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": repr(error)})
        raise
    return report
