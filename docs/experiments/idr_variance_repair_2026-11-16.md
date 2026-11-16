# CPU variance diagnosis and adaptive repair — 16 November 2026

## Authorization, scope and fixed success criterion

The user authorized continued CPU diagnosis and mechanism-based repairs until we
obtain useful results. This supersedes the earlier requirement to stop after one
CPU comparison. All work stays on the small saved task; no GPU, ProGen/scorer
inference, real generation, held-out data or production deployment. Preserve
completed runs and scripts. Log each intervention before execution, retain failed
attempts, and stop escalating an approach when evidence does not support it.

Keep the current continuous reward, eight local trees (408 leaves/168 branching
states), original supervised initialization and conditioned head unless a separately
recorded control explicitly tests a change. The target remains unchanged. A useful
local fit must satisfy the existing gate: mean and worst-prompt TV <0.05 and joint
mass within 0.02 of target. Require three fixed seeds, then three fresh sampling
seeds for confirmation of the selected repair; no best-checkpoint selection.
This remains a development task with repeatedly inspected outcomes, not biological
confirmation or proof of generalization. Report computational/sample costs honestly.

## Stage A: attribution before further training

At the shared initial conditioned head and each of the three saved 16-pair final
heads, enumerate the raw squared-TB-residual gradient for each finite leaf. Include
all trainable head parameters and the eight log-normalizers, in the actual
conditioned parameter coordinates. For each prompt compute the trace of gradient
covariance under (a) current policy, (b) uniform passing replay leaves, (c) uniform
failing replay leaves. Normalize each distribution explicitly.

For each replay parity, sum independent prompt/group variance contributions with
squared averaging weights 1/(2P)^2, where P=8 prompts. Divide by K=16 for the
16-pair estimator. Also report the norm of the exact aggregate mean gradient.
This identifies raw fresh/replay variance at fixed checkpoints; it is not a
prediction of Adam trajectories or a decomposition after nonlinear clipping.
No fits or external inference occur in this stage. Analytic checks on tiny vectors
verify the variance identity before interpreting the real cached task.

Follow-ups will be specified here **before** running them, based on this attribution.
The motivating evidence is the [averaged-sampling result](idr_averaged_sampling_2026-11-16.md)
and the successful [exact expectation](idr_expected_conditioning_2026-11-16.md).

## Stage A results and Stage B pre-run choice

`runs/idr-gradient-variance-v1` attributes fresh draws to 25–64% of initial
raw variance (depending on replay parity). At all three final 16-pair heads,
fresh draws contribute 30–49% and replay 51–70%. Total 16-pair noise RMS is
1.27–1.79, exceeding mean gradient norms 0.71–1.00. Thus neither source is
negligible, and replay-only averaging is not a demonstrated sufficient fix.
These are analytic fixed-checkpoint measurements in conditioned coordinates.

### Fixed-proposal TB with a gradient control variate

Test a finite-bank variance-reduction method, motivated by
[Johnson & Zhang (2013), SVRG](https://proceedings.neurips.cc/paper_files/paper/2013/file/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Paper.pdf).
The paper's estimator subtracts a snapshot gradient on the same sampled examples
and adds the full snapshot mean. Its strongly-convex guarantees do not apply to
our nonlinear head, Adam or clipping. This is an engineering adaptation, not a
published protein-generation result.

Use the previous full-support **fixed** proposal per prompt: half the original
policy plus one-quarter uniform passing and one-quarter uniform failing leaves.
The proposal stays fixed as parameters change. This changes training visitation,
not the reward/zero-residual TB solution. It supports all 408 leaves of this task.
It does **not** support every unrestricted generated IDR; coverage is a major
limitation of any deployment using only a fixed bank.

Compare two matched conditions: ordinary sampled TB/Adam under this proposal,
and the same estimator with the snapshot correction. Draw 32 leaves per prompt
per update (256 total), with replacement; use seed 98000 + fit seed. Both share
exact sampled indices. At steps 1,21,...,381, copy current head/normalizers and
compute the proposal-weighted full-bank raw gradient. In the corrected arm use
current sampled gradient minus snapshot sampled gradient plus snapshot full mean;
then clip once and apply the inherited Adam update. All terms include normalizers.
The correction is unbiased for the fixed-proposal raw gradient before clipping;
it is not an on-policy estimator or an expected Adam parameter update.

Run seeds 0/1/2 for both arms, 400 updates each. Keep initialization, conditioning,
terminal reward, rates and decay unchanged. Refresh every 20 steps is a preselected
engineering tradeoff (20 full-bank gradients), not a tuned result. Stop at 400;
no checkpoint selection. Each arm samples 102,400 leaf occurrences. The correction
also evaluates 102,400 snapshot occurrences and 8,160 full-bank occurrences per
fit (20 times 408). This costs extra head work but no new ProGen rollouts because
all states are cached. Record those counts; short CPU runtime is not a production
cost estimate. Current implementation evaluates shared tree states together.

If all three corrected fits pass the unchanged gate, run three fresh sampling
seeds 3/4/5 with exactly the same method/settings. If they fail, inspect evidence
before selecting another mechanism; do not silently increase budget or weaken
the criterion. The ordinary proposal control distinguishes a proposal-only benefit
from the effect of the correction. Outcome interpretation remains finite-task only.

## Stage B outcome and predeclared confirmation

The fixed-proposal ordinary control fails in all three seeds: mean TV
0.11491 and joint mass 55.57%. Corrected fits pass in all three: mean TV
0.02032, worst-prompt TV 0.03575–0.03693, joint mass 61.66% versus 62.02% target.
Thus the fixed-proposal change alone does not explain the gain. Corrected updates
clip on 9.25–10.25% of steps, versus 99.5–100% for the matched control.

Proceed with the already specified seeds 3/4/5, unchanged method and budget,
in `runs/idr-fixed-svrg-confirm-v1`. This confirms sampling robustness on the same
known task, not new-state or biological generalization. No settings are changed
based on these results, and no additional favorable seed search is authorized
by the observed success.

## Confirmation outcome: a working local sampled-TB repair

All three fresh-seed fits pass with unchanged settings:

| Stage | Seed | Mean TV | Worst prompt TV | Joint-pass probability |
|---|---:|---:|---:|---:|
| Development | 0 | 0.02009 | 0.03575 | 61.67% |
| Development | 1 | 0.02000 | 0.03624 | 61.67% |
| Development | 2 | 0.02087 | 0.03693 | 61.63% |
| Confirmation | 3 | 0.01917 | 0.03482 | 61.68% |
| Confirmation | 4 | 0.01946 | 0.03488 | 61.67% |
| Confirmation | 5 | 0.01972 | 0.03560 | 61.67% |

The exact target is 62.019% joint-pass probability. This is a six-seed local
sampling-robustness result on the same task, not six independent biological
replications. All 48 prompt/fit combinations have TV below 0.05. No checkpoint,
seed, reward parameter or accuracy threshold was selected after viewing outcomes.
All corrected fits end at step 400; failed preceding approaches remain recorded.

The matched ordinary fixed-proposal fits have mean TV 0.11491; corrected
development fits have 0.02032 (about 82% lower). The matched controls use identical
sampled leaf indices and initial parameters. Changing visitation alone does not
explain the result. Correction also costs additional gradient evaluations, so
this is not an equal-compute or optimized sample-efficiency comparison.

### What changed, plainly

We still train a GFlowNet with terminal trajectory-balance rewards. Every 20
updates we calculate a reliable reference gradient using the cached bank. Each
sampled update then measures how the model's gradient has changed since that
reference, and adds that change to the reliable reference direction. Subtracting
the reference gradient on exactly the same sampled leaves cancels much of the
sampling variation. This does not train a next-token teacher, change the reward,
or replace the GFlowNet with a supervised classifier.

For precision, the raw estimator is

\[
\widehat g = g_\theta(B)-g_{\widetilde\theta}(B)+G_{\widetilde\theta}.
\]

Here \(\theta\) denotes the current head and log-normalizer parameters;
\(\widetilde\theta\) their fixed snapshot; \(B\) the sampled bank trajectories;
\(g_\theta(B)\) their averaged TB-loss gradient at current parameters;
\(g_{\widetilde\theta}(B)\) the gradient on the **same** trajectories at the snapshot;
\(G_{\widetilde\theta}\) the full proposal-weighted snapshot gradient; and
\(\widehat g\) the corrected raw gradient. Clipping and Adam are applied afterward.
The expectation identity holds for the fixed proposal; no convex convergence
or exact Adam-update equivalence is claimed.

### Cost and limits

Each corrected fit samples 102,400 occurrences, evaluates those again at the
snapshot and performs 8,160 full-bank gradient occurrences: **212,960 total**
leaf-gradient occurrences by this accounting. The ordinary matched fit uses
102,400. On this tiny task, full enumeration at every step would use 163,200,
so the corrected method is **not cheaper than exact averaging here**. Its potential
advantage is amortizing reference passes over a larger cached bank; that scaling
has not been measured. Corrected fits take about ten seconds of CPU fitting.
All data/rewards/features were already cached; no new pretrained inference occurs.

Success depends on the conditioned coordinates and selected finite-bank support.
The method has not been tested without conditioning, on unseen states or a large
bank. The fixed proposal differs from the previous moving on-policy/replay mixture,
although the target and zero-TB-residual solution on these trees are unchanged.
There is no evidence that this finite replay bank covers the unrestricted target.
Most residues are forced by the local masks. Charge/disorder success remains a
computed property criterion, not experimental biological validation.

## Verification and reproduction

Nineteen focused [CPU tests](../verification/idr-variance-repair-tests-2026-11-16.txt)
pass. They include covariance identities, exact enumeration of control-variate
unbiasedness, zero estimator noise at the snapshot, matched sampled indices,
source-weight preservation, and independence from oracle target probabilities.
The [saved-artifact audit](../verification/idr-variance-repair-audit-2026-11-16.json)
reconstructs all nine fits' metrics/normalizers/gates, checks folded policies,
validates paired draws and source/input hashes, and verifies the variance-audit
inputs. No GPU was initialized. Production modules and previous outputs are intact.

Run from the project root using **new output directories** when reproducing:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src
.venv/bin/python scripts/audit_idr_gradient_variance_cpu.py \
  --output runs/idr-gradient-variance-reproduction
.venv/bin/python scripts/fit_idr_variance_reduced_cpu.py --method plain \
  --output runs/idr-fixed-plain-reproduction
.venv/bin/python scripts/fit_idr_variance_reduced_cpu.py --method svrg \
  --output runs/idr-fixed-svrg-reproduction
.venv/bin/python scripts/fit_idr_variance_reduced_cpu.py --method svrg --seeds 3 4 5 \
  --output runs/idr-fixed-svrg-confirm-reproduction
```

The verification script checks the completed canonical directories:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  .venv/bin/python scripts/verify_idr_variance_repair_cpu.py \
  --output /tmp/idr-variance-repair-verification.json
```

## Next boundary

Stop CPU tuning on this task: the fixed local criterion is met on all six seeds.
Retain the conditioned fixed-proposal TB/control-variate implementation as the
candidate repair. The next meaningful work is a separately specified **short
fully branching transfer test** with new frozen-ProGen states and an explicit
cache/coverage and generation-cost budget. It must compare fresh generated samples,
not just bank fit, and assess conditioning off-bank. Do not deploy these diagnostic
checkpoints or claim unrestricted generation is solved. Any needed GPU capture
or training remains user-run. No GPU command is prepared as part of this CPU study.
