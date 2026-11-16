# Exact cached-IDR diagnosis

**16 November 2026 — completed CPU diagnosis.** Protocol below was written before
the three bounded fits; the subsequent algebraic capacity check is explicitly post hoc. The user authorized
separating target preference, guidance-head representability and TB optimization.
No new pretrained/scorer execution or GPU work is needed. This is a bounded CPU
correctness/learning diagnostic, not a new full-generation training experiment.

## Why this check differs from previous exact tests

The earlier nine-leaf fixture used Gaussian features and a two-residue synthetic
reward. Here we reuse actual frozen ProGen states, the current direct-context head,
64-residue IDR tails, saved disorder predictions and the unchanged continuous reward.
Use only the 408 **training** rows of `runs/idr-continuous-v1/bank.json` and its
cache. Do not add diagnostic holdout rows or generated evaluation outcomes to fits.
These training rows were selected for successes/near misses; they are not a sample
of the prior. Each prompt defines a separate finite prefix tree of its saved tails.

**Critical limitation:** locally mask actions to this tree and renormalize at each
branch. This defines a new constrained prior, not original ProGen conditioned on
eventually reaching a bank sequence. Forced suffixes have probability one. Report
branch depths/counts, the global-conditioned alternative and the bank's original
prior probability mass. This isolates a finite implementation/optimization problem;
it cannot establish capacity, exploration or convergence for open generation.
No new proteins are generated. Sharing cached features at identical prefixes is
checked for numerical consistency.

## Exact quantities and controlled checks

For each prompt, `p_tree(x)` is the product of locally masked ProGen probabilities,
`w(x)=exp(-2.5 V(x))` is the unchanged property factor, and
`pi(x)=p_tree(x) w(x)/Z`, with `Z=sum_x p_tree(x) w(x)` over that tree's leaves.
`x` is a stored 64-residue tail; `V` is the total normalized violation from the
[continuous reward protocol](idr_continuous_2026-11-15.md). Compute all probabilities
and joint-success expectations exactly. Also report the previous reward on the
same support, without fitting or choosing a reward by the outcome.

Construct exact tabular conditional probabilities by summing target mass below
each edge. Reconstructing leaf probabilities must recover `pi`; this checks tree
arithmetic and establishes a reference independent of the neural head.

From identical saved supervised-start weights, compare three bounded head fits:

1. **Exact target cross-entropy:** minimize `-sum_x pi(x) log q(x)` averaged equally
   over prompts. This is a privileged representability/optimization control, not
   a replacement generator or achievable training data for the full problem.
2. **Exact expected TB/replay gradient:** same per-step conditional sampling mixture
   as the next arm, but sum every leaf's squared TB residual with detached weights.
3. **Sampled TB/replay:** one on-policy leaf plus one stored leaf per prompt;
   replay alternates positive/negative strata by prompt and step, as production.

All use the same production head, initial checkpoint and 400-update budget;
Adam rates 1e-4 head / 1e-3 direct branch / 1e-2 normalizer, decay by 0.1 after
300, clip norm 1. TB prompt normalizers use the same balanced-stratum least-squares
initialization rule on this constrained task; the teacher has no normalizer loss.
Sampling seeds 92000 and 95000 match existing roles. No sweeps, extensions or
best-checkpoint selection. Exact expectations supply privileged enumeration;
matching updates does not match compute or information.

Record exact per-prompt and mean TV, target/prior/policy joint rates, loss history,
clipping and final normalizer error. TV below 0.05 is an engineering diagnostic
threshold, not statistical significance. A successful teacher fit demonstrates
adequate representation on this restricted task; a failed teacher fit cannot
prove insufficient capacity because its optimizer/budget may also fail. If teacher
fits but TB fails, investigate the objective/optimization; if expected TB fits but
sampled TB fails, stochastic gradient estimation is implicated on this task. If
all fit, do not declare full-generation solved: investigate state coverage and
long open branching with a separately designed short exhaustive ProGen tree.

Preserve source/config/checkpoint hashes, script and imported implementation hashes,
CPU package versions and source integrity. Write analysis only to a new output
folder. Reuse existing tests for the reward/head; add independent hand-computable
checks of finite-tree normalization, conditioning distinction and exact gradients.


## Results

Sources: [exact diagnostic](../../runs/idr-exact-diagnosis-v1/target_audit.json),
[bounded fits](../../runs/idr-exact-diagnosis-v1/fits.json), and
[algebraic capacity check](../../runs/idr-exact-capacity-v1/capacity.json).
All use the continuous reward; the previous reward is reported only in the target
audit. No completed production code, checkpoint or run output was changed.

| Distribution / fitting method | Mean exact joint success | Mean TV to exact target |
|---|---:|---:|
| Locally constrained ProGen prior | 32.97% | — |
| Shared supervised starting head | 34.17% | 0.32547 |
| Exact continuous-reward target | 62.02% | 0 |
| Exact-target cross-entropy, 400 updates | 46.73% | 0.18536 |
| Expected TB/replay, 400 updates | 40.58% | 0.25593 |
| Sampled TB/replay, 400 updates | 36.03% | 0.30865 |
| Algebraically constructed same-architecture head | 62.02% | 0.00000190 |

Means weight the eight prompts equally. TV is half the sum of absolute probability
differences over all leaves; zero means exact distribution matching. These numbers
are **not new full-generation success rates** and must not be compared directly to
the earlier 9.57%/9.86% rates. Every fitted method remains above the diagnostic
mean-TV threshold 0.05. The exact teacher improves most; expected TB is better than
this one sampled run but still substantially underfits. This is not a multiple-seed
estimate of stochastic optimization performance.

### Post-hoc capacity witness

Because even the privileged teacher failed to fit closely, we added a deterministic
algebraic check rather than claiming a capacity bottleneck or trying more training.
There are 168 branching states and 1,550 input columns (normalized frozen features,
context and a constant). The design matrix has row rank 168; singular-value ratio
is 2,937.14. For each state, exact descendant target masses determine the required
legal next-token log probabilities. Subtract the base logits and center the legal
corrections; softmax ignores that constant.

Use 20 of the existing 128 hidden units. Add one fixed offset so desired hidden
outputs are positive, invert the positive branch of SiLU by 60 bisection iterations,
and solve the hidden affine map by double-precision least squares (`rcond=1e-12`).
The output layer selects those 20 units and removes the offset; the direct branch
is zero. This is the **same model class**, with the saved LayerNorm unchanged.
No extra inputs, state lookup layer or changed activation are used. The original
head is copied, never modified or deployed; constructed weights are not saved.

The linear solve's maximum residual is 6.04e-14. After casting weights back to the
production float32 dtype, mean TV is 1.90e-6 and maximum prompt TV is 2.94e-6.
The hidden-weight norm is 20.15 versus 8.14 initially. Thus the architecture can
represent this finite target, but this construction is a memorization witness,
not evidence of good behavior outside the 168 supplied states or of an easy
optimization path from the saved initialization. No inference about full-space
head capacity follows.

### What this isolates

- **Target:** on this explicitly restricted task, the unchanged reward improves
  exact expected joint success from 32.97% to 62.02%. All eight prompts improve.
  An entirely useless target is not the explanation for the reduced-task failure.
- **Representation:** the same neural head can fit this target to numerical
  precision. Insufficient model-class capacity does not explain this particular
  finite-task failure.
- **Learning:** the inherited update procedure fails to reach that fit. Removing
  trajectory sampling noise helps but does not solve it; changing to exact target
  cross-entropy helps more but also underfits at this budget. Every TB update clips,
  versus 0.5% of teacher updates. That is evidence to inspect, not proof clipping
  is causal or a prescription to remove it.
- **Schedule:** teacher mean TV at steps 100/200/300/400 is
  0.2790/0.2373/0.1899/0.1854. Expected TB gives
  0.3081/0.2803/0.2578/0.2559. Progress slows after the inherited rate decay.
  This does not establish that a longer run or higher rate fixes open generation.

### Boundary of the conclusion

The 408 leaves are selected training examples. Seven trees average only 2.16–3.11
actual choices per sequence; DP00181 averages 9.94. All other residues are forced.
This makes it especially important not to blame *only* long-horizon credit for a
failure already present on these short branching trees. It does not rule out an
additional credit/coverage problem in unrestricted generation.

The locally masked prior is materially different from original ProGen conditioned
on this bank: globally conditioned target success is near zero for five prompts,
near one for two, and 0.008 for DP01538. Whole-sequence likelihoods dominate on that
selected finite set. The total bank probability under original ProGen is at most
about 8.6e-6 (DP00181); other prompts have log mass from −67.5 to −134.6. Neither
conditioning calculation is an estimator of the unrestricted target success rate.
These differences expose why finite-bank matching must not be advertised as a
validated full-generation target or a biological advance.

## Decision

**Pause new full-length GPU experiments.** Before implementing prefix flows or
another reward change, use this fixed CPU problem and its exact reference to make
the current learning procedure reliably fit a representable target. Investigate
update scaling/conditioning and the behavior of head versus normalizer gradients;
a future change must address a measured problem and beat the unchanged procedure
on the same task. No optimizer sweep or automatic extra steps were run here.

Once that passes, a short *fully branching* ProGen capture is the next bridge to
open generation; that would require a separate user-run GPU command and new
protocol. This result does not yet justify unfreezing, larger batches or another
loss swap on the full task. It provides a tractable optimization failure rather
than declaring a unique cause for all previous IDR failures.

## Reproduction and verification

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/diagnose_idr_exact_cpu.py \
  --output runs/idr-exact-diagnosis-reproduction
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/audit_idr_exact_capacity.py \
  --output runs/idr-exact-capacity-reproduction
```

These are reproduction commands, not requested GPU work. Source manifests verify
before/after both analyses and CUDA remains uninitialized. The first script uses
three bounded fits on 168 cached branching states; the second performs only linear
algebra and cached-head evaluation. No ProGen or disorder model executes.

Five [CPU tests](../verification/idr-exact-diagnostic-tests-2026-11-16.txt) check
hand-calculated tree probabilities, global/local conditioning differences, oracle
reconstruction, expected-gradient arithmetic, cache inconsistency rejection,
all fit paths and the algebraic witness while preserving source weights.
The [executed source snapshot](../verification/idr-exact-executed-source-2026-11-16.zip)
preserves the scripts matching recorded run hashes before formatting/import cleanup;
current scripts retain identical executable function logic. The
[verification audit](../verification/idr-exact-analysis-audit-2026-11-16.json)
records that comparison and key result artifacts. No additional literature claims
were needed: this is an algebraic/empirical diagnosis of the existing implementation.
