"""Bounded CPU test averaging 16 fresh/replay draws before each TB update."""

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import torch
from condition_idr_exact_cpu import ConditionedHead, equivalence
from diagnose_idr_updates_cpu import load_task, metrics

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import verify_manifest
from stability_gfn.idr.branch_rate import branch_optimizer, set_rates


def averaged_loss(log_q, log_reward, normalizer, stratum, fresh_rng, replay_rng, draws):
    """Average sampled residuals before clipping; retain duplicate-leaf counts."""
    if draws < 1:
        raise ValueError("Need at least one draw per group")
    candidates = torch.where(stratum)[0]
    if not len(candidates):
        raise ValueError("Replay stratum is empty")
    fresh = torch.multinomial(
        log_q.detach().exp(), draws, replacement=True, generator=fresh_rng
    )
    stored = candidates[torch.randint(len(candidates), (draws,), generator=replay_rng)]
    residual = normalizer + log_q - log_reward
    loss = (residual[fresh].square().mean() + residual[stored].square().mean()) / 2
    return loss, fresh.tolist(), stored.tolist()


def fit_averaged(trees, initial, offsets, config, seed, draws=16):
    """One Adam step per averaged batch, with the inherited replay strata."""
    head = copy.deepcopy(initial)
    normalizer = torch.nn.Parameter(offsets.detach().clone())
    parameters = [*head.parameters(), normalizer]
    optimizer = branch_optimizer(head, normalizer, config, 10)
    fresh_rng = torch.Generator().manual_seed(92000 + seed)
    replay_rng = torch.Generator().manual_seed(95000 + seed)
    history, samples = [], []
    clipped = 0
    started = time.monotonic()
    for step in range(1, config["steps"] + 1):
        set_rates(optimizer, config, 10, step)
        losses, fresh_indices, stored_indices = [], [], []
        for prompt, tree in enumerate(trees):
            loss, fresh, stored = averaged_loss(
                tree.log_probabilities(head).double(),
                tree.log_reward,
                normalizer[prompt],
                tree.passed == ((prompt + step) % 2 == 0),
                fresh_rng,
                replay_rng,
                draws,
            )
            losses.append(loss)
            fresh_indices.append(fresh)
            stored_indices.append(stored)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_clip"], error_if_nonfinite=True
        )
        clipped += float(norm) > config["gradient_clip"]
        optimizer.step()
        samples.append(
            dict(
                step=step,
                fresh=fresh_indices,
                replay=stored_indices,
                loss=float(loss.detach()),
                gradient_norm=float(norm),
            )
        )
        if step == 1 or step % 100 == 0:
            history.append(dict(step=step, **metrics(trees, head)))
    result = metrics(trees, head)
    target_joint = sum(float(t.target[t.passed].sum()) for t in trees) / len(trees)
    result.update(
        seed=seed,
        steps=config["steps"],
        draws_per_group_per_prompt=draws,
        trajectory_draws=config["steps"] * len(trees) * draws * 2,
        max_prompt_tv=max(r["tv"] for r in result["by_prompt"].values()),
        target_joint_success=target_joint,
        clipped_fraction=clipped / config["steps"],
        elapsed_seconds=time.monotonic() - started,
        history=history,
    )
    result["passes_gate"] = (
        result["mean_tv"] < 0.05
        and result["max_prompt_tv"] < 0.05
        and abs(result["mean_joint_success"] - target_joint) < 0.02
    )
    for prompt, tree in enumerate(trees):
        result["by_prompt"][tree.rows[0]["prompt_id"]]["normalizer_error"] = float(
            normalizer[prompt].detach() - tree.log_reward.logsumexp(0)
        )
    return result, head, normalizer, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    trees, initial, offsets, config = load_task(args.run)
    if config["steps"] != 400:
        raise ValueError("This diagnostic fixes a 400-update budget")
    candidate = ConditionedHead(initial, trees)
    initial_check = equivalence(trees, initial, candidate)
    if initial_check["maximum_prompt_tv_between_policies"] > 1e-4:
        raise ValueError("Initial policy changed")
    output = new_run(
        args.output,
        dict(
            source=str(args.run.resolve()),
            seeds=[0, 1, 2],
            steps=400,
            arms=["conditioned"],
            draws_per_group_per_prompt=16,
            diagnostic_only=True,
            support="locally_masked_training_bank",
            optimizer="inherited sampled Adam",
        ),
    )
    source = output / "source"
    source.mkdir()
    for name in (
        Path(__file__).name,
        "condition_idr_exact_cpu.py",
        "fit_idr_exact_optimizer_cpu.py",
        "diagnose_idr_updates_cpu.py",
        "diagnose_idr_exact_cpu.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    root = Path(__file__).resolve().parents[1]
    shutil.copy2(
        root / "docs/experiments/idr_averaged_sampling_2026-11-16.md",
        source / "protocol.md",
    )
    inputs = {
        name: file_hash(args.run / name)
        for name in (
            "generation.json",
            "bank.json",
            "bank_cache.pt",
            "initial.pt",
            "pilot_config.json",
            "reward.json",
            "provenance.json",
        )
    }
    source_hashes = {p.name: file_hash(p) for p in source.iterdir()}
    controls = {}
    for directory in ("idr-sampled-conditioning-v1", "idr-expected-conditioning-v1"):
        control = root / "runs" / directory
        provenance = json.loads((control / "provenance.json").read_text())
        if provenance["input_hashes"] != inputs:
            raise ValueError("Control training inputs differ")
        initialization = json.loads((control / "initialization.json").read_text())
        if (
            initialization["config"] != config
            or initialization["conditioning"] != candidate.conditioning
        ):
            raise ValueError("Control configuration or conditioning differs")
        controls[directory] = {
            name: file_hash(control / name)
            for name in (
                "results.json",
                "initialization.json",
                "provenance.json",
                "status.json",
            )
        }
        if json.loads((control / "status.json").read_text())["status"] != "complete":
            raise ValueError("Control incomplete")
    write_json(output / "controls.json", controls)
    write_json(
        output / "initialization.json",
        dict(
            equivalence=initial_check,
            conditioning=candidate.conditioning,
            initial_metrics=metrics(trees, initial),
            config=config,
            leaf_ids=[[r["id"] for r in t.rows] for t in trees],
        ),
    )
    results, checks = {}, {}
    for seed in range(3):
        key = f"conditioned-{seed}"
        result, head, normalizer, samples = fit_averaged(
            trees, candidate, offsets, config, seed
        )
        results[key] = result
        write_json(output / f"{key}-samples.json", samples)
        torch.save(
            dict(
                head=head.state_dict(),
                log_z=normalizer.detach(),
                diagnostic_only=True,
            ),
            output / f"{key}.pt",
        )
        # Reconstruct reported distributions from the saved checkpoint.
        saved = torch.load(output / f"{key}.pt", map_location="cpu", weights_only=True)
        reconstructed = copy.deepcopy(candidate)
        reconstructed.load_state_dict(saved["head"])
        check = metrics(trees, reconstructed)
        if check["mean_tv"] != result["mean_tv"]:
            raise ValueError("Saved checkpoint does not reproduce evaluation")
        checks[key] = dict(saved_metrics_match=True)
        folded = reconstructed.fold()
        fold_check = equivalence(trees, reconstructed, folded)
        if fold_check["maximum_prompt_tv_between_policies"] > 1e-4:
            raise ValueError("Folding changes policy")
        checks[key]["fold_equivalence"] = fold_check
        torch.save(
            dict(
                head=folded.state_dict(),
                log_z=saved["log_z"],
                diagnostic_only=True,
            ),
            output / f"{key}-folded.pt",
        )
        write_json(output / "results.json", results)
        print(
            key,
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "mean_tv",
                        "max_prompt_tv",
                        "mean_joint_success",
                        "passes_gate",
                    )
                }
            ),
            flush=True,
        )
    verify_manifest(args.run)
    if inputs != {name: file_hash(args.run / name) for name in inputs}:
        raise ValueError("Source inputs changed during fitting")
    assert not torch.cuda.is_initialized()
    write_json(
        output / "verification.json",
        dict(
            fits=checks,
            draws_per_fit=102400,
            source_inputs_unchanged=True,
            cuda_initialized=False,
        ),
    )
    write_json(
        output / "provenance.json",
        dict(
            input_hashes=inputs,
            source_files=source_hashes,
            torch=torch.__version__,
            cuda_initialized=False,
        ),
    )
    write_json(output / "status.json", dict(status="complete", diagnostic_only=True))


if __name__ == "__main__":
    main()
