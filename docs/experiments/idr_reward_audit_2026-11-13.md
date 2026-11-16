# Charge reward audit before replay — 13 November 2026

**Stage:** post-hoc CPU analysis of saved outputs; revised reward implemented and
unit-tested. The subsequent [TB comparison](idr_acceptance_2026-11-13.md) now
tested it in training: joint success did not improve meaningfully. See its
completed-results section for diagnosis and the proposed replay pilot. Historical
rewards/runs are unchanged. The user subsequently asked
us to investigate and fix the reward; the adopted candidate is specified below.

## Question and conclusion

Would better learning of our existing target satisfy the intended objective of
producing IDRs with the requested charge profile? The answer is not automatically.
There is evidence of **both** imperfect target fitting and an imperfect match
between the chosen target and the operational success criteria. Replay remains a
plausible learning intervention, not an established fix or the next automatic run.

The original exact nominal-charge calculation is not a fitted, noisy stability
predictor. The concern is how exact descriptors are turned into utility, how strong
the tilt is, and which requirements the objective omits. No sign or arithmetic
bug was identified in the previous independent recomputation.

## What the implemented score rewards

For a 64-residue continuation, the first/second 32-residue blocks have signed mean
charges `b1` and `b2`. `f` is the overall fraction of K/R/D/E residues. The score is

\[
S=-\tfrac12\left[((b_1-0.125)/0.125)^2+
((b_2+0.125)/0.125)^2\right]-((f-0.25)/0.25)^2.
\]

`S` is zero at the requested center and negative elsewhere. The denominators are
chosen score scales, not measured biological uncertainties. The terminal reward is
`R(x|c) = p0(x|c) exp(beta S(x))`: `x` is the continuation, `c` the supplied prompt,
`p0` the fixed canonical-residue ProGen prior, `beta=2` the tilt strength, and `R`
the unnormalized positive terminal reward. GFlowNet training seeks the normalized
distribution proportional to `R`, not a sampler that always returns the best `S`.

The charge **pass test** requires all three conditions:

- first block within 0.0625 of +0.125;
- second block within 0.0625 of -0.125;
- charged fraction within 0.125 of 0.25.

The smooth score sums deviations from the exact centers. It does not prioritize
crossing every pass threshold, and it penalizes deviations even within acceptable
ranges. This is a legitimate smooth preference, but it is not equivalent to
maximizing the frequency of satisfying all bounds.

### Concrete ranking mismatch

In the unselected prior pool, **21 of 24 prompt/seed groups** contain at least one
failing sequence with a better score than a passing sequence. This is an existence
check, not the fraction of all pairwise rankings that are inverted.

One same-prompt, same-seed example:

| Saved sample | First block | Second block | Charged fraction | Score | Pass? |
|---|---:|---:|---:|---:|---|
| prior_pool-0-0-247 | +0.06250 | -0.18750 | 0.15625 | -0.390625 | Yes |
| prior_pool-0-0-7 | +0.03125 | -0.12500 | 0.234375 | -0.28515625 | No |

The failure misses the positive-block requirement but is closer on the other
terms, so its **property factor** is higher. This comparison does not assert its
full reward or probability is higher: those also depend on its ProGen probability.
Increasing beta preserves this score ordering; it cannot remove the mismatch.

## Target behavior using the existing pool

Use 6,144 unselected ProGen proposals, split into 24 prompt/seed groups of 256.
Within each group normalize weights `exp(beta*S)`, then average group expectations
equally. No extra factor of `p0` belongs in the weights because proposals already
come from `p0`. All variants below are **post-hoc sensitivity analyses**, not
predeclared experiments or selected production settings.

| Beta | Estimated charge pass | Estimated joint charge / V3 disorder pass | Estimated disorder pass | Mean effective sample size / 256 | Minimum ESS |
|---|---:|---:|---:|---:|---:|
| 0 | 4.39% | 2.10% | 66.32% | 256.0 | 256.0 |
| 1 | 13.06% | 6.07% | 60.84% | 164.6 | 137.3 |
| **2, current** | **24.65%** | **11.35%** | **56.43%** | **79.6** | **48.0** |
| 4 | 48.69% | 22.30% | 50.08% | 29.1 | 4.7 |
| 8 | 76.37% | 35.64% | 47.50% | 11.3 | 1.4 |
| 16 | 91.98% | 44.46% | 47.92% | 4.9 | 1.0 |

Beta-zero here uses the **whole pool**, so it differs from the report's first-quarter
raw-prior baseline (4.23% charge / 1.82% joint). The current TB outputs achieve
8.27% charge / 4.30% joint: an apparent gap to the beta-2 target estimate remains.
That gap motivates learning work, but the target itself still does not concentrate
most probability on passing sequences.

These are self-normalized finite-pool estimates, not exact target rates, upper
bounds, expected trained-model performance or evidence of new biological function.
At large beta a few observed samples dominate, especially within some prompts;
rare unseen sequences could change conclusions. Do not select beta 8/16 simply
because the estimated pass percentage is higher. Harder concentration may make
training harder. No uncertainty interval or independent validation is claimed.

### Disorder is absent from the reward

The current reward does not use metapredict or any disorder term. Retaining
predicted disorder is an evaluation requirement and a hope based on the prior,
not something the objective guarantees. On this pool, stronger charge weighting
reduces estimated disorder yield. Improved charge fitting can therefore reveal a
tradeoff rather than automatically produce better IDRs.

For a diagnostic of explicit acceptance, additionally examine
`exp(2*S + a*I_charge + b*I_joint)`. `I_charge` is one for a charge pass, zero
otherwise; `I_joint` is one for both a charge pass and the existing V3 isolated-tail
screen. `a` and `b` are diagnostic log-weight bonuses. This is our proposed
sensitivity calculation, not an algorithm prescribed by the attached book.

| Diagnostic bonus, base beta 2 | Estimated charge | Estimated joint | Estimated disorder | Mean / minimum ESS |
|---|---:|---:|---:|---:|
| Charge pass +2 | 67.11% | 30.57% | 51.37% | 20.3 / 5.4 |
| Charge pass +4 | 92.73% | 41.93% | 46.80% | 11.9 / 1.4 |
| Joint pass +2 | 51.35% | 42.80% | 71.63% | 24.6 / 6.8 |
| Joint pass +4 | 80.18% | 77.15% | 90.02% | 11.1 / 1.5 |

These illustrate how changing the objective changes its implied outcomes; they do
not show that a neural sampler can learn it or that high-bonus estimates are
reliable. Including V3 in a future reward would end its status as evaluation-only.
It would establish optimization of a predicted-disorder proxy, not independently
verified disorder. A new assessment plan would be required; V2 is a sensitivity
check, not fully independent biological validation.

## What the attached RL monograph contributes

See the [targeted reading note](../../literature/RL_THEORY_REWARD_AUDIT_2026-11-13.md).
Its KL-regularized formulation helps separate target choice from target learning.
For a fixed prompt and positive beta, our target also maximizes

\[
J(q)=\mathbb E_q[S]-\frac{1}{\beta}D_{\mathrm{KL}}(q\Vert p_0).
\]

`q` is a candidate normalized sequence distribution, `E_q` averages over its
outputs, `J` is the regularized utility, and `D_KL` is Kullback–Leibler divergence
from the frozen prior. This is a characterization of the ideal target, **not** a
claim that our TB loss is this objective or its gradient. The book uses beta as
the KL coefficient, inverse to our guidance-strength convention.

Replay can change data use and help target fitting; it cannot change this target
at convergence. Multiplying all terminal rewards by a common positive constant
only changes the normalizer. Adding a constant to `S` likewise does not change
the normalized target. Scaling score differences changes the effective beta.
Simply moving scores into [0,1] is not a generic solution, and negative `S` is
valid because `R` is positive.

## Adopted candidate: joint_acceptance_v1

The user's follow-up authorizes a reward fix. The objective is to satisfy **all
three existing charge bounds and the existing isolated-tail disorder screen**.
We adopt a new, versioned candidate rather than changing historical scores. This
is our engineering design, not a formula prescribed by a paper.

### Definition and every parameter

Let `[u]+ = max(u, 0)`. Define the total outside-tolerance violation

\[
V(x)=\sum_{j=1}^{2}\left[\frac{|b_j-t_j|}{\tau_b}-1\right]_+
 +\left[\frac{|f-f_0|}{\tau_f}-1\right]_+
 +\left[\frac{d_0-d}{s_d}\right]_+.
\]

Then set

\[
w(x)=\begin{cases}
1 & \text{if } A(x)=1,\\
\frac{\rho+(1-\rho)\exp[-V(x)]}{M} & \text{otherwise},
\end{cases}
\qquad R_{\rm new}(x\mid c)=p_0(x\mid c)w(x).
\]

| Symbol | Meaning and fixed setting |
|---|---|
| `x`, `c`, `p0` | Generated 64-residue tail, supplied prompt/design context, and its frozen canonical-residue ProGen probability. |
| `b_j`, `t_j` | Actual and requested signed mean nominal charge in block `j`; targets `(+0.125, -0.125)`. Each block has 32 residues. |
| `tau_b` | Existing allowed block error, `0.0625`. |
| `f`, `f0`, `tau_f` | Actual charged fraction, requested fraction `0.25`, and existing allowed error `0.125`. |
| `d` | Fraction of isolated-tail residues with metapredict V3 score **strictly greater than 0.5**. |
| `d0` | Existing required fraction, `0.8`; with 64 residues, at least 52 must pass. |
| `s_d` | Disorder-deficit scale, `0.2`; falling 20 percentage points short contributes one violation unit. |
| `V` | Sum of nonnegative violations; constraints already inside tolerance contribute zero. |
| `A` | One if all three charge tests and `d >= d0` pass; zero otherwise. |
| `M` | Minimum pass/near-failure property-factor ratio, `20`. |
| `rho` | Fractional floor for failing sequences, `0.25`. |
| `w` | Positive property multiplier, `1` for passes and between `0.0125` and `0.05` for failures. |
| `R_new` | Unnormalized terminal reward; the desired distribution is `R_new / Z_new`, where `Z_new` sums it over allowed tails for the prompt. |

In the earlier notation, use `S_new = log(w)` and **beta = 1**. In TB, the
residual is `log Z_new + log q_theta - log p0 - log w`, where `q_theta` is the
learned sampling policy and `theta` its head parameters. Do not apply historical
beta 2 to this already scaled log factor. Keep all reward calculations in log
space; the full sequence probability can be extremely small.

### Why these choices

- **Acceptable ranges have a plateau.** No extra reward for moving toward an
  arbitrary center after a requirement is met. Every joint pass outranks every
  failure in property factor. This does not guarantee its full prior-weighted
  reward exceeds every failing sequence's reward.
- **Disorder is explicit.** A charge-perfect but predicted-ordered sequence fails
  the joint objective. V3 now becomes part of training, not an independent check.
  Use pinned metapredict 3.0.2, V3, unrounded isolated-tail scores. No new predictor
  training is needed. V2 remains same-package sensitivity; neither establishes
  experimental disorder or biological function.
- **Positive partial credit.** Outside violations still distinguish failures by
  proximity, without compensating a missed constraint using excess performance
  elsewhere. The 20-fold acceptance step makes reaching every requirement matter.
  This is graded terminal feedback, not intermediate supervision or a fully smooth
  reward: the pass boundary and residue threshold are discontinuous. TB does not
  differentiate through either, but discovering rare passes remains a challenge.
- **Bounded range.** `rho=0.25` allows at most fourfold variation among failures;
  together with `M=20`, the largest property-factor ratio is 80 and the log range
  is at most `log(80)=4.382`. This bounds reward-induced extreme residuals, not
  policy gradients or training loss in general. Among passes, the target retains
  the prior's relative probabilities rather than favoring the exact center.
- **Defaults are declared engineering choices.** `M=20` gives a clear incentive
  at a roughly 2% prior joint rate without a 100-fold near-miss cliff. In a pure
  binary 20-fold tilt, a 2% prior rate would become about 29%; graded failure
  penalties can increase this further. It does not promise majority success.
  `rho=0.25` bounds failure-score variation instead of using an unbounded quadratic;
  `s_d=0.2` sets an interpretable deficit unit. These values were fixed before
  evaluating this candidate; prior beta/bonus analyses were already inspected.
  They are not validated optima or literature-derived constants.

We rejected a zero-reward hard gate because it removes the finite log reward for
failures. A proximity-only version is retained as an offline diagnostic ablation,
not another proposed GPU sweep. Larger beta on the old score preserves its ranking
mismatch and its missing disorder term. Adding only a charge bonus omits disorder.

### Saved-pool results for the chosen candidate

Recompute descriptors and screen flags from all 6,144 existing unselected prior
proposals. Use the same 24-group self-normalized importance calculation above.
Only one new candidate and its no-acceptance-gap ablation were assessed; no search
over `M`, `rho` or disorder scales was performed.

| Target | Charge | Joint charge / V3 | V3 disorder | V2 disorder sensitivity | Median / minimum ESS per 256 |
|---|---:|---:|---:|---:|---:|
| Prior | 4.39% | 2.10% | 66.32% | 65.61% | 256 / 256 |
| Historical beta-2 score | 24.65% | 11.35% | 56.43% | 57.67% | 78.7 / 48.0 |
| Proximity-only ablation | 8.55% | 5.91% | 70.96% | 69.45% | 220.6 / 203.3 |
| **joint_acceptance_v1** | **49.20%** | **47.86%** | **84.34%** | **80.02%** | **16.8 / 14.3** |

The ablation uses `rho + (1-rho)*exp(-V)` everywhere. Bounded proximity alone is
too weak on this pool; the acceptance step supplies most of the new target's
joint success. The old observed log-factor range is `[-51.625, 0]`, versus
`[-4.382, 0]` for the new design. Mean within-group standard deviation of log
factors under prior proposals decreases from 2.36 to 0.58. This motivates testing
whether extreme reward signals contributed to weak fitting; it does not establish
the cause or imply easier learning of rare passing trajectories.

**Coverage limitation is substantial.** Group joint-pass counts range from 0 to
19; two DP00181 pools contain none and a third contains just one. Reweighting
cannot recover an unseen pass. The new mean ESS is 37.2, inflated by the two
zero-pass groups; the median of 16.8 is more representative. One proposal can
hold 21.6% of a group's estimated probability. Unlike the old score, the candidate
concentrates on a rare accepted set even while its log-factor range is smaller.

A paired, within-group proposal bootstrap (2,000 resamples, seed 20261113) gives
40.05–50.33% for the new target's joint rate and +30.27–37.60 percentage points
for its difference from the old target. These conditional percentile intervals
omit unseen modes, prompt-selection uncertainty and reward-design selection.
They are descriptive development checks, **not independent confirmation or a
forecast of trained GFN performance**. Weighted repeated-trimer fraction is 13.3%
(prior 15.8%, old target 10.8%); diversity/repetition still need raw-output checks.

### Implementation and bounded next step

[Reward implementation](../../src/stability_gfn/idr/acceptance_reward.py) accepts
complete tails and supplied per-residue V3 scores, validates inputs, and returns
detached log factors plus every component. It never runs a model. The
[audit script](../../scripts/audit_idr_acceptance_reward.py) verifies the pinned
saved pool and reproduces the table and bootstrap. New tests cover the historical
inversion, inclusive charge bounds, strict disorder cutoff, the 51/52-residue
boundary, failure proximity, bounded weights, invalid inputs and exact TB target
arithmetic with the prior included once. All **90 CPU tests pass**.

The reward fix is implemented as a scorer; existing training commands intentionally
retain their historical reward. **The subsequent [TB runner](idr_acceptance_2026-11-13.md)
has now completed without a useful generation gain.** The protocol below records
the motivation before that test; its result record owns the subsequent decision.
It connects this scorer to one bounded TB comparison using the same frozen
ProGen/direct-context head and inherited 800 updates, three seeds, rates and
clipping. Score complete tails during training without backpropagating through V3;
account for its extra CPU scoring time. Do not combine replay, SubTB or unfreezing.

Before execution, pin fresh evaluation RNG and the comparator provenance; report
raw charge and joint rates per prompt/seed, V2 sensitivity, duplicates/repetition,
target-estimate coverage, and TB residuals **under each policy's own reward**.
Existing target calculations are inspected development data. New output sampling
does not make these eight selected prompts an independent biological benchmark.

Use a provisional practical screen of at least **+5 percentage points in aggregate
raw joint success over the old TB control**, improvement in at least two of three
seeds, and no greater than 5-point drop in unique-tail fraction. These are engineering
decision thresholds to fix in that protocol, not significance tests. If the change
still yields only a few extra passes, do not extend training or sweep reward
constants automatically: distinguish failure to discover accepted trajectories
from failure to fit those observed, then consider replay as a separate intervention.
No GPU experiment or performance improvement has been established in this audit.

## Evidence and reproducibility

[Machine-readable audit](../verification/idr-reward-audit-2026-11-13.json) includes
source hashes, all beta/bonus values, every group estimate and the observed ranking
examples. The source is `runs/idr-subtb-v1/tb/sequence_metrics.json`; only
`arm=prior_pool` enters reweighting. Use saved charge scores, charge flags and V3
isolated-tail pass flags, normalize log weights with a max-subtraction per group,
and calculate ESS as `1/sum(normalized_weights**2)`. No new model execution or
training was performed. Existing runs, code and configurations remain unchanged.

The follow-up [acceptance-reward audit](../verification/idr-acceptance-reward-2026-11-13.json)
adds the new settings, script/module hashes, per-group estimates and bootstrap.
Historical source files/configurations and completed run artifacts are unchanged;
only the separate candidate implementation is new. The targeted literature update
is [review section 11](../../literature/IDP_IDR_REVIEW_2026-11-12.md#11-bounded-joint-acceptance-reward--13-november-2026).
