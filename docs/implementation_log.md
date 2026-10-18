# Implementation decisions

Substantive implementation choices and the problems they address. Scientific
rationale and parameter choices are in the [decision record](decisions.md);
protocols and outcomes are in the [experiment index](experiments/README.md).

## 2026-10-14 — Finite sequence environment and frozen backbone

Represent each candidate set as a prefix tree. Every sequence has one construction
path, constraints exclude dead ends, and forced suffix residues remain explicit.
This enables exact target/learned-distribution comparisons before attempting larger
spaces. The four-position toy has 85 internal prefixes and 256 leaves. Cache frozen
prefix features once and charge that work to setup; this does not establish
scalability beyond enumerable supports.

Reuse the working Python/ProGen package versions and vendor upstream model code,
tokenizer and license to preserve checkpoint compatibility. Use the literal `1`
direction marker, float32 logits, causal prefix features for guidance and pooled
complete-sequence features for the stability predictor. Tiny CPU fixtures use
64 hidden dimensions and eight heads because upstream attention packs eight ways.
Cached/full-forward and shared-prefix comparisons verify causal alignment.
See [architecture](architecture.md) and [backbone decisions](decisions.md#2026-10-14--backbone-and-feature-definitions).

## 2026-10-14 — Guidance and comparison methods

Initialize the residual head's output to zero so generation starts exactly at the
locally constrained prior. Use log-space trajectory balance with a learned log
normalizer. Implement Monte Carlo future-reward supervision as the cheaper guidance
comparison, with the same frozen backbone and reward. Verify zero-guidance,
beta-zero, forced-action and likelihood identities before real training.

Save frozen numerical inputs and final heads so evaluation can be reproduced on
CPU without reloading ProGen or refitting. Preserve raw proposals and selection
indices to distinguish generation from selection. Use immutable run directories,
source snapshots and hashes to keep comparisons reproducible.
See [learning mechanics](decisions.md#2026-10-14--learning-mechanics).

## 2026-10-14 — Stability data and predictor

Separate candidate identities from held-out labels so support construction need
not consume evaluation outcomes. Apply canonical-sequence and label filters,
resolve reference identities, average sequence-level replicates and preserve
seeded 70/15/15 splits. The real release audit selected 1O6X by coverage, yielding
950/204/204 train/validation/test sequences. See [data](data.md).

Fit scaling on training data only. Compare ridge and a small MLP, select by
validation, and require the predictor to beat a training-mean baseline and meet
the declared ranking screen before guidance. Construct rewards from frozen
predicted improvement relative to the reference. Biological smoke tests require
explicitly synthetic data; screen failure blocks guidance and held-out evaluation.
See [scorer/data decisions](decisions.md#2026-10-14--mega-scale-preprocessing-and-scorer).

## 2026-10-14 — Diagnose exact-task optimization before changing capacity

Initial exact-task and privileged conditional fits missed the TV criterion.
Independent leaf-mass calculations and a prefix-depth KL decomposition localized
most error at length-two prefixes. Add conditional-fit diagnostics and gradient,
TV/KL and entropy-floor logging to distinguish insufficient optimization from
limited head capacity. Preserve the architecture and compare longer fits at two
rates. The longer original-rate conditional fit passed, motivating a longer TB run
rather than a larger model. See [diagnostics](experiments/exact_diagnostics_2026-10-14.md).

Longer constant-rate TB passed only one seed. Add a deterministic late rate drop
and observational exact-distribution monitoring to test late settling. Monitoring
must not change RNG, updates or checkpoint selection, and its cost is separate.
CPU comparisons establish parity with the original constant-rate implementation.
The scheduled run passed two seeds; extend the low-rate phase to 20,000 total
updates while preserving the drop after 8,000 and the final-checkpoint criterion.
See [scheduled experiment](experiments/exact_decay_2026-10-14.md).

## 2026-10-15 — Transfer the verified schedule to biology

All three final exact-task seeds passed. Transfer the tested schedule to both
biological guidance learners, require matching checkpoint hashes and remove
training-time exact monitoring from the biological run. Keep predictor validation
before guidance. This verifies mechanics before biology without treating toy
convergence as evidence of improved stability.
See [gate and pilot setup](experiments/exact_pass_and_biological_pilot_2026-10-15.md).

## 2026-10-15 — Separate target quality from guidance accuracy

The first biological pilot fit its target accurately without improving measured
stability. Implement an exact development-target diagnostic so scorer/prior utility
can be assessed before further guidance training. Reuse the scorer fitter; add
separate fit/inner/development partitions, three development parents and five
reserved confirmation parents. Cache teacher-forced causal logits and verify that
they reproduce the existing prefix-tree prior.

Compare ridge, MLP-0 and an ensemble, alongside declared prior-tempering and uniform
controls. Keep failed screens visible and exclude reserved outcomes from analysis.
This separates an unhelpful target from difficulty learning a useful target.
See [development protocol](experiments/target_diagnostic_2026-10-15.md) and
[results](experiments/target_diagnostic_results_2026-10-15.md).

## 2026-10-16 — Seal all confirmation fits before releasing outcomes

Reconstruct reserved candidate identities separately from numerical labels.
Train the fixed MLP-0/original-prior procedure on all five parents with the existing
learners. Require every parent's fitting phase to complete or explicitly fail its
screen before a hashed fit seal permits evaluation. An interrupted fit cannot
silently release outcomes or omit an unfavorable seed.

Use matched proposal budgets, retain duplicates and preserve every final seed.
Replay verifies sealed artifacts and uses saved inputs on CPU. Synthetic tests
exercise outcome isolation, failed screens and tampering rejection.
See [confirmation protocol](experiments/confirmation_plan_2026-10-16.md) and
[completed results](experiments/confirmation_results_2026-10-17.md).

## 2026-10-17 — Isolate prior construction and distinct-design utility

Confirmation established strong target accuracy but limited diversity and higher
fitting cost than supervised guidance. Development controls showed that local
branching induces early-mutation concentration even with uniform logits. Add an
explicit descendant-count prior arm alongside the original prior; keep scorer,
support, reward strength and learners fixed. Store corrected logits in each arm's
task so initialization, sampling, TB rewards and supervised rollouts all use the
same prior. Reuse prefix features across arms to avoid duplicate backbone work.

Add a separate selection rule that deduplicates the full proposal batch before
ranking and never refills. Retain repeated-slot metrics and report actual design
count alongside quality. Exact-law sampling repeats measure selection variability;
time actual trajectory decoding separately. Keep uniform selection and exact-target
sampling as finite-support controls, with setup and fitting costs explicit.

Freeze development source/scorer hashes and seal final artifacts for CPU replay.
Synthetic tests verify uniform-leaf correction, zero-head behavior, causal cache
parity, label-independent selection, failed-screen handling and replay integrity.
The full suite has 41 passing CPU tests; these do not establish real convergence.
See [protocol](experiments/diversity_plan_2026-10-17.md),
[findings](experiments/diversity_findings_2026-10-17.md) and
[verification](verification/diversity-tests.txt).
