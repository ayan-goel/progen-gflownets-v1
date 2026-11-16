"""Fixed-bank TB with an SVRG-style control variate; cached finite task only."""

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


def fixed_tb_loss(trees, head, normalizer, indices=None):
    """Full fixed-proposal mean or sampled mean; terminal TB rewards only."""
    losses = []
    for prompt, tree in enumerate(trees):
        residual = (
            normalizer[prompt] + tree.log_probabilities(head).double() - tree.log_reward
        )
        if indices is None:
            weights = tree.fixed_weights.detach() / tree.fixed_weights.sum()
            losses.append((weights * residual.square()).sum())
        else:
            losses.append(residual[indices[prompt]].square().mean())
    return torch.stack(losses).mean()


def loss_gradient(trees, head, normalizer, indices=None):
    loss = fixed_tb_loss(trees, head, normalizer, indices)
    gradients = torch.autograd.grad(loss, [*head.parameters(), normalizer])
    return float(loss.detach()), [g.detach() for g in gradients]


def correct_gradient(current, snapshot_sample, snapshot_mean):
    """A control variate with zero mean under the fixed sampling proposal."""
    return [
        g - old + mean for g, old, mean in zip(current, snapshot_sample, snapshot_mean)
    ]


def fit_fixed(trees, initial, offsets, config, seed, method):
    head = copy.deepcopy(initial)
    normalizer = torch.nn.Parameter(offsets.detach().clone())
    parameters = [*head.parameters(), normalizer]
    optimizer = branch_optimizer(head, normalizer, config, 10)
    generator = torch.Generator().manual_seed(98000 + seed)
    history, samples = [], []
    clipped, snapshots = 0, 0
    started = time.monotonic()
    for step in range(1, config["steps"] + 1):
        set_rates(optimizer, config, 10, step)
        if method == "svrg" and (step - 1) % 20 == 0:
            snapshot = copy.deepcopy(head)
            snapshot_z = torch.nn.Parameter(normalizer.detach().clone())
            _, snapshot_mean = loss_gradient(trees, snapshot, snapshot_z)
            snapshots += 1
        indices = [
            torch.multinomial(
                t.fixed_weights, 32, replacement=True, generator=generator
            )
            for t in trees
        ]
        loss, gradients = loss_gradient(trees, head, normalizer, indices)
        if method == "svrg":
            _, snapshot_sample = loss_gradient(trees, snapshot, snapshot_z, indices)
            gradients = correct_gradient(gradients, snapshot_sample, snapshot_mean)
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, config["gradient_clip"], error_if_nonfinite=True
        )
        clipped += float(norm) > config["gradient_clip"]
        optimizer.step()
        samples.append(
            dict(
                step=step,
                indices=[index.tolist() for index in indices],
                loss=loss,
                gradient_norm=float(norm),
            )
        )
        if step == 1 or step % 100 == 0:
            history.append(dict(step=step, **metrics(trees, head)))
    result = metrics(trees, head)
    target_joint = sum(float(t.target[t.passed].sum()) for t in trees) / len(trees)
    count = config["steps"] * len(trees) * 32
    result.update(
        seed=seed,
        method=method,
        steps=config["steps"],
        sampled_occurrences=count,
        snapshot_sample_occurrences=count if method == "svrg" else 0,
        full_snapshot_occurrences=snapshots * sum(len(t.rows) for t in trees),
        snapshots=snapshots,
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
    parser.add_argument("--method", choices=("plain", "svrg"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
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
            seeds=args.seeds,
            method=args.method,
            refresh_steps=20,
            steps=400,
            arms=["conditioned"],
            draws_per_prompt=32,
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
        root / "docs/experiments/idr_variance_repair_2026-11-16.md",
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
            fixed_weights=[t.fixed_weights.tolist() for t in trees],
            leaf_ids=[[r["id"] for r in t.rows] for t in trees],
        ),
    )
    results, checks = {}, {}
    for seed in args.seeds:
        key = f"{args.method}-{seed}"
        result, head, normalizer, samples = fit_fixed(
            trees, candidate, offsets, config, seed, args.method
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
