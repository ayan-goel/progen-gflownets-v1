"""Read-only inputs for the bounded 400-versus-800 last-block comparison."""

import copy
import json
import re

import torch

from ..artifacts import file_hash
from ..pipeline import project_path
from .acceptance_experiment import verify_implementation, verify_manifest
from .confirmation_data import MAX_SEED, _stream_seeds, check_confirmation_inputs
from .lastblock_data import check_lastblock_inputs


def _completed_snapshot(folder, check_checkpoint):
    """Check saved generation and its software without executing model code."""
    status = json.loads((folder / "status.json").read_text())
    if status["status"] != "complete":
        raise ValueError(f"Budget source is not complete: {folder}")
    manifest = verify_manifest(folder)
    provenance = json.loads((folder / "provenance.json").read_text())
    verify_implementation(provenance)
    if check_checkpoint and (provenance["fixture"] or status.get("fixture", False)):
        raise ValueError("Real budget comparison cannot use fixture inputs")
    return manifest, provenance


def _check_final_policy(fit, config, references, check_checkpoint):
    """Validate the saved short-budget policy, loading tensors on CPU only.

    This checkpoint is an evaluation comparator, never the longer fit's start.
    Shape checks use the original head and ProGen configuration, not a model load.
    Tiny fixtures omit a real ProGen configuration and use internally consistent
    synthetic block dimensions instead.
    """
    initial = torch.load(fit / "initial.pt", map_location="cpu", weights_only=True)
    checkpoint = torch.load(
        fit / "lastblock_tb/checkpoint.pt", map_location="cpu", weights_only=True
    )
    expected_metadata = dict(
        step=config["steps"],
        arm="lastblock_tb",
        architecture="lastblock_direct_context",
        reward_version="continuous_acceptance_v1",
        reference="immutable original checkpoint in provenance.json",
        prompt_ids=[row["id"] for row in references if row["role"] == "prompt"],
    )
    if any(checkpoint.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Short-budget final checkpoint metadata differs")
    for group in ("head", "last_block"):
        values = checkpoint.get(group)
        if (
            not isinstance(values, dict)
            or not values
            or any(
                not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in values.values()
            )
        ):
            raise ValueError(f"Invalid short-budget checkpoint {group} tensors")
    if {name: tuple(value.shape) for name, value in checkpoint["head"].items()} != {
        name: tuple(value.shape) for name, value in initial["head"].items()
    }:
        raise ValueError("Short-budget checkpoint head shapes differ")
    normalizers = checkpoint.get("log_z")
    if (
        not isinstance(normalizers, torch.Tensor)
        or normalizers.shape != (len(expected_metadata["prompt_ids"]),)
        or not torch.isfinite(normalizers).all()
    ):
        raise ValueError("Invalid short-budget checkpoint normalizers")

    hidden_size = initial["head"]["normalization.weight"].numel()
    block = checkpoint["last_block"]
    if check_checkpoint:
        model_config = json.loads(
            (project_path(config["checkpoint"]) / "config.json").read_text()
        )
        if hidden_size != model_config["n_embd"]:
            raise ValueError("Original ProGen and guidance dimensions differ")
        inner_size = model_config.get("n_inner") or 4 * hidden_size
        positions = model_config["n_positions"]
    else:
        inner_size = block.get("mlp.fc_in.bias", torch.empty(0)).numel()
        attention_mask = block.get("attn.bias", torch.empty(0))
        positions = attention_mask.shape[-1] if attention_mask.ndim == 4 else 0
    expected_shapes = {
        "ln_1.weight": (hidden_size,),
        "ln_1.bias": (hidden_size,),
        "attn.bias": (1, 1, positions, positions),
        "attn.masked_bias": (),
        "attn.qkv_proj.weight": (3 * hidden_size, hidden_size),
        "attn.out_proj.weight": (hidden_size, hidden_size),
        "mlp.fc_in.weight": (inner_size, hidden_size),
        "mlp.fc_in.bias": (inner_size,),
        "mlp.fc_out.weight": (hidden_size, inner_size),
        "mlp.fc_out.bias": (hidden_size,),
    }
    if (
        positions < 1
        or inner_size < 1
        or {name: tuple(value.shape) for name, value in block.items()}
        != expected_shapes
    ):
        raise ValueError("Short-budget checkpoint last-block shapes differ")
    adaptation = json.loads((fit / "adaptation.json").read_text())
    if adaptation.get("reference_has_gradients") is not False:
        raise ValueError("Short-budget fit did not retain the frozen reference")


def check_budget_inputs(settings, check_checkpoint=True):
    """Derive longer fits with only the budget and evaluation stream replaced."""
    if set(settings) != {
        "confirmation_run",
        "confirmation_generation_sha256",
        "training_steps",
        "replicates",
    }:
        raise ValueError(
            "Budget comparison accepts only source, doubled steps and seeds"
        )
    replicas = settings["replicates"]
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ValueError("Budget comparison requires both confirmation replicates")
    if type(settings["training_steps"]) is not int or settings["training_steps"] < 1:
        raise ValueError("Invalid training_steps")
    names = set()
    for replica in replicas:
        if not isinstance(replica, dict) or set(replica) != {"name", "evaluation_seed"}:
            raise ValueError(
                "Each budget replicate accepts only name and evaluation_seed"
            )
        name, seed = replica["name"], replica["evaluation_seed"]
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is None
            or name in names
        ):
            raise ValueError("Replicate names must be safe and distinct")
        names.add(name)
        if type(seed) is not int or not 0 <= seed <= MAX_SEED:
            raise ValueError("Invalid evaluation_seed")

    confirmation = project_path(settings["confirmation_run"])
    if (
        file_hash(confirmation / "generation.json")
        != settings["confirmation_generation_sha256"]
    ):
        raise ValueError("Pinned confirmation manifest differs")
    manifest, provenance = _completed_snapshot(confirmation, check_checkpoint)
    required = {
        "config.json",
        "pilot_config.json",
        "pilot_generation.json",
        "provenance.json",
    }
    if not required.issubset(manifest["files"]):
        raise ValueError("Confirmation source snapshots are not protected")
    source_settings = json.loads((confirmation / "config.json").read_text())
    pilot, old_plans, old_config, references, pilot_provenance = (
        check_confirmation_inputs(source_settings, check_checkpoint=check_checkpoint)
    )
    if [name for name, _ in old_plans] != [replica["name"] for replica in replicas]:
        raise ValueError("Use both original confirmation replicate identities in order")
    for filename, snapshot in (
        ("config.json", "pilot_config.json"),
        ("generation.json", "pilot_generation.json"),
    ):
        if file_hash(pilot / filename) != file_hash(confirmation / snapshot):
            raise ValueError("Confirmation pilot snapshot differs")
    for filename, digest in pilot_provenance["files"].items():
        if provenance["files"].get(filename) != digest:
            raise ValueError("Confirmation and pilot provenance differ")
    if provenance["packages"] != pilot_provenance["packages"]:
        raise ValueError("Confirmation and pilot package versions differ")
    if settings["training_steps"] != 2 * old_config["steps"]:
        raise ValueError("This comparison requires exactly double the completed budget")

    # Retain the paired training/replay streams, but never reuse an evaluation
    # stream, including the three adjacent sampler streams used during training.
    used_seeds = set()
    historical_settings = [fit for _, fit in old_plans]
    historical_settings.append(json.loads((pilot / "config.json").read_text()))
    for key in ("source_run", "control_run"):
        folder = project_path(historical_settings[0][key])
        historical_settings.append(json.loads((folder / "config.json").read_text()))
    for historical in historical_settings:
        used_seeds.update(_stream_seeds(historical))
    for replica in replicas:
        if replica["evaluation_seed"] in used_seeds:
            raise ValueError("Budget evaluation streams must be fresh and disjoint")
        used_seeds.add(replica["evaluation_seed"])

    plans = []
    config = None
    for replica, (name, old_fit_settings) in zip(replicas, old_plans):
        fit = confirmation / name / "fit"
        fit_manifest, fit_provenance = _completed_snapshot(fit, check_checkpoint)
        _completed_snapshot(confirmation / name / "prior", check_checkpoint)
        child_required = {
            "config.json",
            "pilot_config.json",
            "references.json",
            "provenance.json",
            "initial.pt",
            "initial_training.jsonl",
            "bank.json",
            "reward.json",
            "reward_definition.json",
            "adaptation.json",
            "lastblock_tb/checkpoint.pt",
        }
        if not child_required.issubset(fit_manifest["files"]) or any(
            f"{name}/fit/{filename}" not in manifest["files"]
            for filename in child_required | {"generation.json"}
        ):
            raise ValueError("Confirmation child inputs are not protected")
        if json.loads((fit / "config.json").read_text()) != old_fit_settings:
            raise ValueError("Confirmation child settings differ from its derivation")
        if json.loads((fit / "pilot_config.json").read_text()) != old_config:
            raise ValueError("Confirmation child support/configuration differs")
        if json.loads((fit / "references.json").read_text()) != references:
            raise ValueError("Confirmation child reference panel differs")
        for filename in (
            "initial.pt",
            "initial_training.jsonl",
            "bank.json",
            "reward.json",
            "reward_definition.json",
            "supervised.pt",
            "supervised_training.jsonl",
            "source_generation.json",
            "control_generation.json",
        ):
            if file_hash(fit / filename) != file_hash(pilot / filename):
                raise ValueError(
                    f"Confirmation initial bank/source differs: {filename}"
                )
        for filename, digest in fit_provenance["files"].items():
            if provenance["files"].get(filename) != digest:
                raise ValueError("Confirmation parent/child provenance differs")
        if fit_provenance["packages"] != provenance["packages"]:
            raise ValueError("Confirmation parent/child packages differ")
        _check_final_policy(fit, old_config, references, check_checkpoint)

        fit_settings = copy.deepcopy(old_fit_settings)
        fit_settings.update(
            training_steps=settings["training_steps"],
            evaluation_seed=replica["evaluation_seed"],
        )
        # The full original-checkpoint hash was already checked above. Reuse the
        # unchanged input validator for the derived budget without hashing it again.
        inputs = check_lastblock_inputs(fit_settings, check_checkpoint=False)
        expected_config = dict(old_config, steps=settings["training_steps"])
        if inputs[2] != expected_config or inputs[3] != references:
            raise ValueError("Derived longer fit changes support or other settings")
        config = inputs[2]
        plans.append(dict(name=name, source_fit=fit, fit_settings=fit_settings))
    prompt_count = sum(row["role"] == "prompt" for row in references)
    if (
        manifest["count"]
        != len(plans) * 5 * prompt_count * old_config["samples_per_prompt"]
    ):
        raise ValueError("Confirmation generation count differs from its protocol")
    return confirmation, plans, config, references, copy.deepcopy(provenance)


def budget_preflight(settings):
    """Describe user-run work after CPU-only validation of saved inputs."""
    confirmation, plans, config, references, _ = check_budget_inputs(settings)
    fit = plans[0]["fit_settings"]
    prompt_count = sum(row["role"] == "prompt" for row in references)
    per_arm = prompt_count * config["samples_per_prompt"]
    fresh_per_refresh = sum(
        count for name, count in fit["shared_batch"].items() if name != "replay"
    )
    return dict(
        ready=True,
        confirmation_run=str(confirmation),
        replicates=[
            dict(
                name=plan["name"],
                evaluation_seed=plan["fit_settings"]["evaluation_seed"],
            )
            for plan in plans
        ],
        training_arms_per_replicate=2,
        updates_per_training_arm=fit["training_steps"],
        reference_updates_per_training_arm=fit["training_steps"] // 2,
        learning_rate_decay_after_update=int(
            config["decay_fraction"] * fit["training_steps"]
        ),
        fresh_training_trajectories_per_replicate=(
            fit["training_steps"] // fit["refresh_every"] * fresh_per_refresh
        ),
        trajectory_presentations_per_training_arm=(
            fit["training_steps"] * sum(fit["shared_batch"].values())
        ),
        evaluation_arms_per_replicate=6,
        evaluation_samples_per_arm=per_arm,
        total_final_evaluation_samples=len(plans) * 6 * per_arm,
        cuda_initialized=torch.cuda.is_initialized(),
        model_or_scorer_executed=False,
    )
