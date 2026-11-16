"""Function-preserving input conditioning on the fixed cached IDR diagnostic."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from diagnose_idr_updates_cpu import load_task, metrics
from fit_idr_exact_optimizer_cpu import fit_fixed_tb
from torch.nn import functional as F

from stability_gfn.artifacts import file_hash, new_run, write_json
from stability_gfn.idr.acceptance_experiment import verify_manifest
from stability_gfn.idr.context_experiment import DirectContextHead


class ConditionedHead(DirectContextHead):
    """Invertible fixed change of hidden-input coordinates; same model class."""

    def __init__(self, initial, trees):
        feature_size = initial.normalization.normalized_shape[0]
        context_size = initial.hidden.in_features - feature_size
        super().__init__(feature_size, context_size, 20, initial.hidden.out_features)
        self.load_state_dict(initial.state_dict())
        with torch.no_grad():
            inputs = []
            for tree in trees:
                combined = torch.cat(
                    [self.normalization(tree.features), tree.context], dim=-1
                )
                inputs.append(
                    torch.cat([combined, torch.ones(len(combined), 1)], dim=-1)
                )
            design = torch.cat(inputs).double()
            _, singular, right = torch.linalg.svd(design, full_matrices=False)
            if singular.min() <= 1e-12 * singular.max():
                raise ValueError(
                    "Require full row rank for the fixed conditioning check"
                )
            basis = right.T
            scales = math.sqrt(len(design)) / singular
            weight = torch.cat(
                [initial.hidden.weight, initial.hidden.bias[:, None]], dim=1
            ).double()
            reparameterized = weight + ((weight @ basis) * (1 / scales - 1)) @ basis.T
            self.hidden = torch.nn.Linear(
                design.shape[1], initial.hidden.out_features, bias=False
            )
            self.hidden.weight.copy_(reparameterized.float())
            self.register_buffer("basis", basis.float())
            self.register_buffer("scales", scales.float())
            transformed = design + ((design @ basis) * (scales - 1)) @ basis.T
            spectrum = torch.linalg.svdvals(transformed)
            self.conditioning = {
                "states": len(design),
                "input_columns_including_bias": design.shape[1],
                "original_condition_number": float(singular.max() / singular.min()),
                "transformed_condition_number_double": float(
                    spectrum.max() / spectrum.min()
                ),
                "minimum_scale": float(scales.min()),
                "maximum_scale": float(scales.max()),
                "label_usage": "None: matrix computed from frozen features/context and initial LayerNorm only.",
            }

    def forward(self, features, context):
        combined = torch.cat([self.normalization(features), context], dim=-1)
        augmented = torch.cat(
            [combined, combined.new_ones(*combined.shape[:-1], 1)], dim=-1
        )
        transformed = (
            augmented + ((augmented @ self.basis) * (self.scales - 1)) @ self.basis.T
        )
        return self.output(F.silu(self.hidden(transformed))) + F.linear(
            context[..., :5], self.task_weight
        )

    def fold(self):
        """Return an equivalent ordinary head; no conditioning machinery at inference."""
        feature_size = self.normalization.normalized_shape[0]
        context_size = self.hidden.in_features - feature_size - 1
        head = DirectContextHead(
            feature_size, context_size, 20, self.hidden.out_features
        )
        head.normalization.load_state_dict(self.normalization.state_dict())
        head.output.load_state_dict(self.output.state_dict())
        with torch.no_grad():
            weight = self.hidden.weight.double()
            basis = self.basis.double()
            folded = weight + ((weight @ basis) * (self.scales.double() - 1)) @ basis.T
            head.hidden.weight.copy_(folded[:, :-1].float())
            head.hidden.bias.copy_(folded[:, -1].float())
            head.task_weight.copy_(self.task_weight)
        return head


def equivalence(trees, first, second):
    with torch.no_grad():
        return {
            "maximum_cached_logit_difference": max(
                float(
                    (first(t.features, t.context) - second(t.features, t.context))
                    .abs()
                    .max()
                )
                for t in trees
            ),
            "maximum_prompt_tv_between_policies": max(
                float(
                    (
                        t.log_probabilities(first).double().exp()
                        - t.log_probabilities(second).double().exp()
                    )
                    .abs()
                    .sum()
                    / 2
                )
                for t in trees
            ),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/idr-continuous-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    trees, initial, offsets, config = load_task(args.run)
    candidate = ConditionedHead(initial, trees)
    check = equivalence(trees, initial, candidate)
    if check["maximum_prompt_tv_between_policies"] > 1e-4:
        raise ValueError("Initial policy not preserved accurately")
    # Off-bank algebra check: this is not a biological/generalization evaluation.
    generator = torch.Generator().manual_seed(16112026)
    features = torch.randn(
        16, initial.normalization.normalized_shape[0], generator=generator
    )
    context_size = initial.hidden.in_features - features.shape[1]
    context = torch.randn(16, context_size, generator=generator)
    with torch.no_grad():
        check["maximum_random_input_logit_difference"] = float(
            (initial(features, context) - candidate(features, context)).abs().max()
        )
    if check["maximum_random_input_logit_difference"] > 0.001:
        raise ValueError("Input reparameterization changes the initial function")
    output = new_run(
        args.output,
        {
            "source": str(args.run.resolve()),
            "optimizer": "same bounded L-BFGS as idr-exact-optimizer-v1",
            "gradient_budget": 400,
            "transform": "reward-blind invertible row-span whitening of augmented hidden inputs",
            "diagnostic_only": True,
        },
    )
    source = output / "source"
    source.mkdir()
    for name in (
        "condition_idr_exact_cpu.py",
        "fit_idr_exact_optimizer_cpu.py",
        "diagnose_idr_updates_cpu.py",
        "diagnose_idr_exact_cpu.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), source / name)
    write_json(
        output / "initialization.json",
        {
            "conditioning": candidate.conditioning,
            "equivalence": check,
            "initial_metrics": metrics(trees, candidate),
        },
    )
    result, trained, normalizer = fit_fixed_tb(
        trees, candidate, offsets, config, "lbfgs"
    )
    folded = trained.fold()
    result["fold_equivalence"] = equivalence(trees, trained, folded)
    if result["fold_equivalence"]["maximum_prompt_tv_between_policies"] > 1e-4:
        raise ValueError("Folded head does not preserve the final policy")
    write_json(output / "results.json", result)
    torch.save(
        {
            "head": folded.state_dict(),
            "log_z": normalizer.detach(),
            "diagnostic_only": True,
        },
        output / "folded.pt",
    )
    torch.save(
        {
            "head": trained.state_dict(),
            "log_z": normalizer.detach(),
            "diagnostic_only": True,
        },
        output / "conditioned.pt",
    )
    verify_manifest(args.run)
    assert not torch.cuda.is_initialized()
    write_json(
        output / "provenance.json",
        {
            "source_manifest_sha256": file_hash(args.run / "generation.json"),
            "source_files": {p.name: file_hash(p) for p in source.iterdir()},
            "torch": torch.__version__,
            "cuda_initialized": False,
        },
    )
    write_json(output / "status.json", {"status": "complete", "diagnostic_only": True})
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("by_prompt", "history")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
