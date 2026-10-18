# Implementation architecture

## Data flow

```text
Published Mega-scale CSV → audited task and immutable splits
                                  ↓
Frozen ProGen full-sequence features → validated ridge/MLP scorer → fixed rewards
                                  ↓
Candidate identities → prefix tree → frozen prefix features/base logits
                                  ↓
          residual guidance head + locally legal normalization
                                  ↓
       TB or supervised future-reward training → saved heads
                                  ↓
       raw sampling / declared selection → held-out evaluation
```

## Module responsibilities

| Module | Contract |
|---|---|
| `data.py` | Filter ML labels, map references, collapse duplicates, split unique variants |
| `environment.py` | Substitution budget, finite support, unique paths and legal edges |
| `backbone.py` | Frozen local ProGen loading, token checks, causal/KV features and complete-sequence features |
| `guidance.py` | Residual head, locally constrained policy, sampling, likelihoods and TB loss |
| `scorer.py` | Train-only normalization, ridge/MLP validation and frozen reward transform |
| `training.py` | On-policy TB and baseline-rollout future-reward regression |
| `metrics.py` | Explicit distribution, diversity, novelty, coverage and stability metrics |
| `evaluation.py` | Load cached inputs and heads; produce samples and reports without retraining |
| `diagnostics.py` | Privileged exact-target head fit when diagnosing capacity |
| `pipeline.py` | Stage gates, immutable outputs, provenance and costs |
| `artifacts.py` | Strict JSON, streaming hashes and no-overwrite run creation |
| `__main__.py` | Preparation, doctor, run, evaluation and diagnostic commands |

The upstream ProGen files and license are isolated in `vendor/progen/`; they are
unmodified. Their [provenance](../src/stability_gfn/vendor/progen/provenance.md)
records the source commit. No old test-project scripts are imported.

## Probability and feature invariants

The literal direction marker `1` differs from special-token ID 1. Residues must
round-trip as individual tokens. Fixed length means no sampled stop action.
Forced actions have conditional probability one under both policies; their residues
remain in causal prefixes and complete-sequence feature extraction.

The prefix head sees only the current causal state plus known context. The scorer
sees pooled final-layer residue states from the complete sequence, excluding the
direction marker. Scorer features never become future candidate inputs to the head.

A zero-initialized output projection reproduces the constrained prior exactly.
Frozen tensors are detached. No ProGen or scorer parameter belongs to the guidance
optimizer. TB uses log prior plus beta times the fixed score, never raw positive
rewards subject to underflow.

## Finite support and caching

The initial experiments have explicit supports: 256 synthetic completions or 204
held-out measured variants. Compile a prefix tree, cache each distinct prefix's
features/logits, and evaluate each completed sequence's score once. This implements
the specified fixed-backbone policy exactly on that support; it is not a separate
standalone sequence generator. Prefix extraction currently uses equal-length,
full-prefix batches; cached incremental decoding is separately verified.

Cache extraction is shared setup work. Head training and saved-run evaluation do
not execute ProGen. Costs distinguish frozen-model work, new scorer computation
and terminal-score lookups. This speedup is specific to enumerable supports and
must not be presented as unrestricted-generation deployment speed.

## Reuse of BioSeq-GFN-AL

The supplied [repository](https://github.com/MJ10/BioSeq-GFN-AL) remains the
standalone design reference. Its `TBGFlowNetGenerator` in
`lib/generator/gfn.py` sums trajectory log probabilities, learns log Z and
minimizes squared log-reward residuals with Adam and gradient clipping. These
mechanics inform this implementation. Its DNA MLP, padding conventions, proxy
training and reward clamping are not transplanted into frozen protein guidance.
Here the terminal log reward explicitly includes the constrained ProGen prior,
invalid numbers fail, and prefix legality follows the declared protein support.
No original BioSeq reproduction result is changed or claimed for this integration.

## Artifacts and failure behavior

Every run has its own directory; reuse raises an error. Save configuration, package
versions, checkpoint/tokenizer hashes, source snapshots, model configuration, data
manifests, frozen task cache, scorer/head checkpoints, progress and final curves,
raw/selected samples, selection indices, metrics, timing and status.

Scorer-gate failure retains its report and setup costs but does not start guidance
or open test-label values. Exact-gate failure retains all results but prevents the
configured biological run. Interruptions preserve written artifacts; there is no
automatic restart or overwrite. `running` after a hard kill is not proof of a live
process or completion. Training-resume support is outside this initial handoff.

## Confirmation modules — 16 October 2026

- `confirmation_data.py`: reconstructs label-free candidate identity files and reads only allowed fitting partitions.
- `confirmation.py`: validates the fixed protocol, caches frozen features/rewards, reuses existing scorer/learner functions, and seals all-parent fit artifacts.
- `confirmation_evaluation.py`: checks that seal before releasing outcomes, evaluates actual policy samples plus an exact-target reference, and supports CPU-only replay.

The existing backbone, scorer, guidance and training implementations are unchanged
relative to the completed development source snapshot. Confirmation has separate
artifacts and CLI commands rather than altering historical run evaluation behavior.

## Development diversity ablation

`diversity.py` implements label-free descendant counts, path-prior diagnostics,
deduplication-before-ranking, exact-law categorical evaluation and a clearly
post-hoc review of old confirmation draws. `diversity_experiment.py` freezes and
verifies development inputs, caches each parent's prefixes once, writes two explicit
prior arms, reuses the unchanged TB/supervised learners and seals final fits for
CPU replay. Existing confirmation code and original-prior behavior are unchanged.

The alternate arm stores logits plus log descendant counts in its task payload;
thus the existing zero-head baseline, sampling, reward and supervised targets all
use the same corrected probabilities. The correction uses finite candidate
identities only. It is not sequence-temperature scaling or original-ProGen global
conditioning. The [spec](../spec.md) defines the changed prior explicitly.

Selection reports keep the existing repeated-slot rule alongside unique designs
from the same fixed proposal budget. Categorical repeated evaluation is separated
from timed trajectory sampling. Known development labels never enter rewards or
selection, and no reserved panel test file is needed for the new ablation.
