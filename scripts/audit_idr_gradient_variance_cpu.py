"""Exact raw-gradient noise attribution on saved finite IDR trees, CPU only."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import torch
from condition_idr_exact_cpu import ConditionedHead
from diagnose_idr_updates_cpu import load_task

from stability_gfn.artifacts import file_hash, new_run, write_json


def moments(gradients, probabilities):
    """Mean and trace covariance; columns are actual trainable coordinates."""
    probabilities = probabilities.double() / probabilities.double().sum()
    gradients = gradients.double()
    mean = probabilities @ gradients
    second_moment = probabilities @ gradients.square().sum(1)
    variance = (second_moment - mean.square().sum()).clamp_min(0)
    return mean, float(variance)


def audit(trees, head, normalizer):
    parameters = [*head.parameters(), normalizer]
    dimension = sum(p.numel() for p in parameters)
    prompt_count = len(trees)
    means = [torch.zeros(dimension, dtype=torch.float64) for _ in range(2)]
    variances = [dict(fresh=0.0, replay=0.0) for _ in range(2)]
    details = {}
    for prompt, tree in enumerate(trees):
        log_q = tree.log_probabilities(head).double()
        residual = normalizer[prompt] + log_q - tree.log_reward
        gradients = []
        for loss in residual.square():
            gradient = torch.autograd.grad(loss, parameters, retain_graph=True)
            gradients.append(torch.cat([g.detach().flatten() for g in gradient]))
        gradients = torch.stack(gradients)
        distributions = dict(
            fresh=log_q.detach().exp(), passed=tree.passed, failed=~tree.passed
        )
        statistics = {
            name: moments(gradients, weights) for name, weights in distributions.items()
        }
        details[tree.rows[0]["prompt_id"]] = {
            name: dict(variance_trace=variance, mean_gradient_norm=float(mean.norm()))
            for name, (mean, variance) in statistics.items()
        }
        for parity in range(2):
            replay = "passed" if (prompt + parity) % 2 == 0 else "failed"
            means[parity] += (statistics["fresh"][0] + statistics[replay][0]) / (
                2 * prompt_count
            )
            variances[parity]["fresh"] += (
                statistics["fresh"][1] / (2 * prompt_count) ** 2
            )
            variances[parity]["replay"] += (
                statistics[replay][1] / (2 * prompt_count) ** 2
            )
    result = {}
    for parity, groups in enumerate(variances):
        total = sum(groups.values())
        result[str(parity)] = dict(
            one_draw_variance=groups,
            fresh_fraction=groups["fresh"] / total,
            sixteen_draw_variance=total / 16,
            mean_gradient_norm=float(means[parity].norm()),
            sixteen_draw_noise_rms=(total / 16) ** 0.5,
        )
    return dict(parity=result, by_prompt=details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = Path("runs/idr-continuous-v1")
    trees, initial, offsets, _ = load_task(source)
    candidate = ConditionedHead(initial, trees)
    output = new_run(
        args.output, dict(diagnostic_only=True, kind="raw_gradient_variance")
    )
    snapshot = output / "source"
    snapshot.mkdir()
    for name in (
        Path(__file__).name,
        "condition_idr_exact_cpu.py",
        "diagnose_idr_updates_cpu.py",
        "diagnose_idr_exact_cpu.py",
        "fit_idr_exact_optimizer_cpu.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    shutil.copy2(
        "docs/experiments/idr_variance_repair_2026-11-16.md", snapshot / "protocol.md"
    )
    inputs = {
        str(source / name): file_hash(source / name)
        for name in (
            "generation.json",
            "bank_cache.pt",
            "initial.pt",
            "bank.json",
            "reward.json",
            "pilot_config.json",
        )
    }
    results = {}
    for name in ("initial", "conditioned-0", "conditioned-1", "conditioned-2"):
        head = copy.deepcopy(candidate)
        normalizer = torch.nn.Parameter(offsets.detach().clone())
        if name != "initial":
            path = Path("runs/idr-averaged-sampling-v1") / f"{name}.pt"
            inputs[str(path)] = file_hash(path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            head.load_state_dict(saved["head"])
            normalizer = torch.nn.Parameter(saved["log_z"].clone())
        results[name] = audit(trees, head, normalizer)
        write_json(output / "results.json", results)
        print(name, json.dumps(results[name]["parity"]), flush=True)
    assert all(file_hash(Path(name)) == digest for name, digest in inputs.items())
    assert not torch.cuda.is_initialized()
    write_json(
        output / "provenance.json",
        dict(
            inputs=inputs,
            source_files={p.name: file_hash(p) for p in snapshot.iterdir()},
            cuda_initialized=False,
        ),
    )
    write_json(output / "status.json", dict(status="complete", diagnostic_only=True))


if __name__ == "__main__":
    main()
