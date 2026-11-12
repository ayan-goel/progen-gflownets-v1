# IDR charge-profile guidance pilot

**12 November 2026 — user run complete; results and diagnostics below.**
No real training, generation or GPU execution was performed during implementation.
The [completed baseline](idr_baseline_2026-11-12.md#completed-results--12-november-2026)
supports using IDR-prefix continuation as the starting prior.

## Question and limits

Can a trajectory-balance GFlowNet head steer frozen ProGen toward a simple charge
pattern while retaining predicted disorder and diversity? Compare against unguided
sampling, charge-based selection, and a cheap composition-preserving editing control.
This is an open continuation task (20 choices at each generated position), not the
finite measured-variant tree used in our stability studies.

Use all eight existing prefixes. Do not pick the best prompt after observing the
baseline; RPB1 remains a repetition stress case. The same contexts appear during
training and evaluation: fresh draws test sampling, not generalization to new
protein families or prompts. No experimental function or physical charge is claimed.

## Target, reward and interpretation

Keep the supplied 16 residues unchanged and generate a 64-residue tail. For this
first pilot the charge profile has **two 32-residue blocks**, not 64 separately
specified target values. The existing centered nine-residue profile remains an
evaluation descriptor. Coarse blocks isolate the ability to put charge in the right
part of a sequence before imposing finer patterns.

For each tail residue assign K/R +1, D/E −1 and everything else zero. Ignore
histidine ionization, termini and modifications. Let:

- `x` be the generated tail; `c` its fixed 16-residue prompt and task settings.
- `m1(x)`, `m2(x)` be the signed mean nominal charge in its first and second blocks.
- `f(x)` be its fraction of charged residues (K/R/D/E).

The desired block means are +0.125 and −0.125, and desired charged fraction is 0.25:

\[
S(x)=-\frac12\left[\left(\frac{m_1(x)-0.125}{0.125}\right)^2+
\left(\frac{m_2(x)+0.125}{0.125}\right)^2\right]
-\left(\frac{f(x)-0.25}{0.25}\right)^2.
\]

`S` is a dimensionless, nonpositive score: zero is a perfect match, more negative
is worse. Block scale 0.125 gives a completely neutral block a unit squared error;
FCR scale 0.25 similarly penalizes absence/excess of charged residues without
requiring fixed composition. Both terms are soft and equally weighted as written.
A feasible perfect example has 6 positive and 2 negative residues in the first
block, 2 positive and 6 negative in the second, with 24 neutral residues in each.
Many sequences realize those counts; residue order within each block is unconstrained.

\[
R(x\mid c)=p_0(x\mid c)e^{\beta S(x)},\qquad
p^*(x\mid c)=R(x\mid c)/Z(c),\quad \beta=2.
\]

- `p0` is the frozen-ProGen conditional probability of the entire tail: each step
  normalizes over the 20 canonical residues, with temperature 1 and fixed termination.
- `R` is its positive unnormalized target weight, computed in log space.
- `beta` is the fixed guidance strength. A score difference of one multiplies
  relative target weights by exp(2), about 7.39, relative to their prior odds.
- `Z(c)` sums target weights over every possible tail for this prompt.
- `p*` is the intended normalized distribution; `q_theta` is the trained policy,
  where `theta` denotes only the guidance-head parameters.

At beta zero the target equals the prior. Since S is nonpositive, log Z is at most
zero; zero is the initial estimate, not its assumed true value. The target retains
prior preferences; even successful optimization need not produce perfect profiles.

**Charge pass**, separate from the smooth training score, requires both block
means within 0.0625 of their targets and FCR within 0.125 of 0.25. Equivalently,
first-block net charge is +2 through +6; second-block net charge is −6 through −2;
the tail has 8 through 24 charged residues. Boundaries are inclusive.

**Rationale and outcome access:** these are engineering settings, not values
prescribed by a paper. The existing development baseline has prompted block-charge
10th/50th/90th percentiles −0.09375/0/0.125 for each block and FCR
0/0.15625/0.34375; natural references have median FCR 0.265625. Thus modest signed
blocks and a 0.25 FCR target test an attainable descriptor regime, rather than an
almost fully charged chain. After fixing the candidate rule, a descriptive check
found 11/192 prompted baseline tails and 1/16 references pass. These already
inspected samples inform development; they are not independent confirmation.
The new experiment uses fresh draws, not those saved sequences as a training set.

## Training mechanics and hyperparameters

Reuse the existing LayerNorm → Linear → SiLU → Linear residual guidance head and
TB loss. At each generation step add head corrections to frozen canonical-token
logits, then normalize. Head inputs are the frozen last-layer prefix representation,
prompt one-hot identity, tail progress, accumulated signed charge in each block
(divided by final block length), charged count divided by final tail length, and
the current block target. These are available prefix statistics, not future labels.

For a sampled tail, minimize the squared residual:

\[
\mathcal L=\operatorname{mean}_{x,c}
[\log Z_\phi(c)+\log q_\theta(x\mid c)-\log p_0(x\mid c)-\beta S(x)]^2.
\]

`L` is the loss; the mean is over the current batch. `phi` denotes eight learned
log-normalizers, one per prompt; the other symbols are defined above. Appending
residues gives each sequence a unique trajectory and a unique predecessor, so
backward transition probability is one. Prompt likelihood is excluded from both
path sums. A single shared Z across different prompts would be incorrect.

| Setting | Value and reason |
|---|---|
| Backbone/support | Same frozen ProGen2-medium, float32, 80 total/16 supplied/64 generated; all 20 canonical actions, no EOS/top-k/top-p/filters |
| Training seeds | 0, 1, 2; separate heads, not independent biological datasets |
| Head width | 128, inherited small architecture; zero final projection starts exactly at the prior |
| Normalizers | One per prompt, initialized to zero; no normalizer network needed for eight fixed contexts |
| Updates | 400 per seed; bounded initial convergence probe, not a claimed sufficient budget |
| Batch | Eight trajectories, exactly one per prompt at every update; prevents frequent contexts dominating |
| Head Adam rate | 1e-4, lower than historical 1e-3 because all 64 positions now branch and on-policy states change |
| Normalizer Adam rate | 0.01; inherit faster scalar normalization adaptation |
| Adam defaults | betas (0.9, 0.999), epsilon 1e-8, weight decay 0, AMSGrad false |
| Schedule | First 300 updates at initial rates; final 100 at one tenth; fixed, no outcome-driven scheduling |
| Gradient clipping | Joint global norm 1; nonfinite loss/gradients raise errors |
| Policy/exploration | Fresh on-policy rollouts only; no replay, extra exploration mixture, dropout, clipping of rewards or correction logits |
| Checkpoints | Every 100 steps and final; evaluate step 400, no best-checkpoint selection |
| Progress | Every ten steps plus first/final; all step metrics saved |
| Determinism | Explicit RNG streams, deterministic Torch algorithms, TF32 disabled; same-environment replay only |
| CPU evaluation | One thread, metapredict 3.0.2 V3/V2 inherited from baseline |

Each seed sees 3,200 fresh trajectories (400 per prompt); total training is 9,600.
Sampling uses incremental KV caching. Retain only the current rollout's detached
features/logits/actions, recompute head likelihoods with gradients, update once,
then discard. This avoids a second backbone pass without introducing stale-head
replay. ProGen never has gradients. Real throughput/memory remains to be tested
by the user's GPU run; there is no giant precomputed sequence tree.

Head initialization uses seed s on CPU. Sampling streams are 1000+10s for training,
1001+10s for prior pools and 1002+10s for final GFN evaluation. CPU controls use
NumPy seed 100000+s. Batch size, ordering and RNG conventions are part of replay.

## Comparators and budget accounting

For each seed and each prefix:

1. **Prior pool:** generate 64 fresh unguided tails; keep all proposals in artifacts.
2. **Prior:** first 16 pool draws, chosen before looking at scores.
3. **Select:** best 16 of the 64 by S, stable sample-order tie handling. No disorder
   selection or duplicate removal. Four generation proposals per retained output.
4. **GFN:** 16 fresh raw samples from the final head. No post-selection.
5. **Swap control:** start from each of the 16 prior samples, propose 128 random
   tail-residue swaps, accept only strictly better S. Fixed composition and length,
   prefix unchanged. This cheap direct optimization is not a prior-target sampler.
6. **Shuffle control:** one composition-preserving random permutation of each GFN
   tail. Tests whether order, rather than composition alone, explains the score.

Prior/select/GFN each produce 384 evaluated outputs across seeds. Total new
model-generated evaluation proposals are 1,920 (1,536 prior + 384 GFN).
Controls add no ProGen generation; swapping makes 49,536 score calls including
initial scores. The 3,456 scored records include the complete pool and overlapping
subsets/derived sequences, not 3,456 independent ProGen draws.

**These are equal retained-output counts, not equal total compute.** Selection
uses 4x generation per output; GFN additionally pays training cost. Report separate
measured training/generation seconds, trajectory counts and backbone input/forward
counts, plus CPU evaluation time and swap-score counts. Loading/hashing belongs to
overall time, not model-stage time. Selection's saved prior log likelihood is the
original sequence likelihood, not its selected-output sampling probability. Do not
claim an efficiency advantage without including costs and an amortization analysis.

GOOSE/IDiom remain broader comparators; this first pilot uses the transparent swap
control without adding dependencies. It cannot claim to beat those tools or to
be a full benchmark of biological IDR design.

## Evaluation and next decisions

Primary comparison: raw GFN versus raw prior **joint charge pass and isolated-tail
V3 disorder pass** (at least 80% of 64 tail residues score >0.5). Also report charge
pass/error alone, full-context and isolated-tail disorder, V2 sensitivity, mean
block charges/FCR, uniqueness, pairwise distances, composition entropy, repeated
trimers, homopolymers and limited-reference similarity. Every result is broken
down by seed, prompt, and seed/prompt. Inspect RPB1 explicitly; no aggregate pass
can conceal its motif repetition. All outputs count, including duplicates.

Keep disorder out of the reward, checkpoint choice and selection rule. This tests
whether charge guidance preserves it; optimizing the same disorder tool would
obscure that question. V2 is related evidence, not independent confirmation.

Fresh final-head samples also report TB residual mean/RMSE using saved conditional
path probabilities and log Z. Low sampled residual is not exact TV or proof of
correct distribution sampling over 20^64 possibilities. A small CPU enumeration
checks the formula, not convergence of the real model. No held-out physical labels
or calibrated uncertainty intervals are available in this pilot.

- If charge improves but disorder or diversity degrades, diagnose the tradeoff
  before adding a functional objective; keep all raw outcomes.
- If charge barely moves and residuals remain large/unstable, inspect training
  curves, prompt-level normalizers and length/optimization before changing losses.
- If joint success consistently improves across seeds and is not carried by one
  repetitive prompt, test stronger/finer profiles or held-out prompts separately.
- If selection or swaps do as well for lower total cost, report that: charge is
  deliberately a simple control task, not evidence that a GFN is necessary.

These are interpretation branches, not automated promotion or post-hoc tuning.
No biological-function claim follows from charge success.

## Commands and artifacts

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0  # choose the available GPU yourself
scripts/run.sh idr-charge --output runs/idr-charge-v1 --device cuda
```

No-inference preflight: `scripts/run.sh idr-charge --check-only`.
CPU-only scoring recovery after all generation finishes:

```bash
scripts/run.sh evaluate-idr-charge --run runs/idr-charge-v1 --output runs/idr-charge-v1-evaluation
```

New output directories are required. The runner saves config/references/provenance,
per-seed training logs/checkpoints (head, optimizer, normalizers, RNG), full raw
samples/FASTA, model-stage costs, scored samples, metrics and report. Generation
completion includes input/output hashes. Evaluation can recover without rerunning
training; partial-training automatic resume is not implemented. No completed
baseline artifact is edited.

## Evidence and verification

[Trajectory balance, NeurIPS 2022](https://arxiv.org/abs/2201.13259) motivates the
training objective. [Das and Pappu, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3746876/)
motivates examining charge arrangement as well as charged content, not our numeric
target or a universal function prediction. The [IDR review](../../literature/IDP_IDR_REVIEW_2026-11-12.md)
records descriptor/scorer sources and limitations. The coarse-block score,
context features and pilot hyperparameters are our explicit design choices.

CPU verification includes analytic score/order controls, cached-policy likelihood
comparison with independent teacher forcing, zero-guidance/prior identity,
400-sequence two-residue target normalization/TB identity, invalid-input rejection,
and tiny end-to-end training/evaluation/recovery with real metapredict V3/V2.
The end-to-end test verifies frozen backbone weights, changed head weights,
selection within prompts, fixed-composition controls, immutable outputs and
corruption rejection. Real training convergence is left to the user-run experiment.

Verification before handoff: all **67 CPU tests pass**, including six charge tests.
The real-input `idr-charge --check-only` succeeds without model execution. These
are correctness checks, not real-model convergence or biological results.


## Completed results — 12 November 2026

The user completed `runs/idr-charge-v1` in **4,441.48 s (74 min 1 s)**. All three
heads reached step 400; status is complete and fixture=false. Training consumed
9,600 trajectories and 3,666.02 seconds (61 min 6 s). Saved evaluation contains
1,920 model-generated proposals plus the declared overlapping/derived controls.
[Original report](../../runs/idr-charge-v1/report.md) and
[metrics](../../runs/idr-charge-v1/metrics.json) are preserved unchanged.

**Outcome: no useful aggregate charge-steering improvement from the GFN.**
The learned heads preserve predicted disorder but remain close to the prior.
Selection and composition-preserving swaps outperform this trained guidance setup.

| Method | Charge pass | Isolated V3 disorder pass | Both pass | Mean charge score |
|---|---:|---:|---:|---:|
| Prior | 20/384 (5.2%) | 247/384 (64.3%) | 10/384 (2.6%) | −1.935 |
| Select best quarter | 79/384 (20.6%) | 207/384 (53.9%) | 33/384 (8.6%) | −0.741 |
| GFN | 20/384 (5.2%) | 254/384 (66.1%) | 11/384 (2.9%) | −1.882 |
| Composition-preserving swaps | 142/384 (37.0%) | 266/384 (69.3%) | 66/384 (17.2%) | −0.973 |
| Shuffled GFN | 20/384 (5.2%) | 270/384 (70.3%) | 6/384 (1.6%) | −1.774 |

There is just one additional joint-success output for GFN versus prior; the
small difference is not evidence of a reproducible improvement. The smooth score
also changes little. Mean block charges are (+0.01294,+0.00488) for prior and
(+0.00765,+0.00781) for GFN, compared with the requested (+0.125,−0.125).
Selection reaches (+0.06038,−0.05583); swaps reach (+0.09871,−0.08089).
The GFN has not learned the intended positive-to-negative arrangement on average.

### Seed and prompt checks

| Seed | Prior charge / joint passes | GFN charge / joint passes | GFN fresh-sample TB residual RMSE |
|---|---:|---:|---:|
| 0 | 5/128 / 4/128 | 8/128 / 6/128 | 3.884 |
| 1 | 5/128 / 2/128 | 5/128 / 2/128 | 2.597 |
| 2 | 10/128 / 4/128 | 7/128 / 3/128 | 2.912 |

The direction does not repeat across seeds. GFN uniqueness falls to 365/384 (95.1%),
from prior 373/384 (97.1%). All GFN duplicate tails occur in the RPB1 context:
29/48 unique (60.4%), versus 37/48 (77.1%) for prior. RPB1 contributes zero charge
passes to all main methods. Excluding it in a post-hoc sensitivity calculation
leaves prior and GFN at the same 20/336 charge passes and 10 versus 11 joint passes;
it does not explain away the failure. Keep the full-panel primary result.

Using isolated-tail V2 instead of V3 gives joint pass counts 10/384 prior, 36/384
selection, 11/384 GFN and 66/384 swaps. This related-model sensitivity does not
reverse the conclusion. Neither scorer provides experimental confirmation.

### Training diagnosis from saved artifacts (post-hoc)

Mean TB loss over updates 1–50 versus 351–400 decreases from 17.83→7.68,
16.15→7.66 and 17.98→11.44 for seeds 0/1/2. Mean charge scores in those same
windows move only −1.852→−1.764, −1.760→−1.765 and −1.825→−1.766.
All 1,200 updates exceed the global gradient-clipping threshold of one. Typical
window-mean pre-clip norms range roughly 43–137. This suggests an optimization
issue to investigate, but does not establish that clipping caused the failure;
Adam dynamics and sample variance also matter.

The final head's output weights are nonzero (norms 0.0659/0.0701/0.0749), so updates
were applied. Learned prompt log-normalizers move from zero to roughly −1.84 to
−2.54. On fresh GFN samples, mean log(q/p0) is 0.0113, 0.0069 and −0.0097 per seed,
with standard deviations 0.176/0.207/0.165. Their pooled mean is 0.00285 nats.
These are finite-sample diagnostics; the negative estimate is sampling variation,
not a negative true KL. They support that policy changes are small on sampled paths.

To separate normalizer shifts from path corrections, recompute the saved final
residual on exactly the same GFN samples, removing either term algebraically:

| Seed | Actual final TB MSE | Set log(q/p0)=0 | Set log Z=0 |
|---|---:|---:|---:|
| 0 | 15.082 | 15.260 | 28.051 |
| 1 | 6.743 | 6.775 | 16.915 |
| 2 | 8.482 | 8.586 | 19.325 |

Thus the normalizer shift makes a large contribution to the lower residual, while
the learned path correction contributes little on these samples. This is an
algebraic diagnostic, not a separately trained ablation; it must not be presented
as proof that learning log Z prevented policy learning. Lower loss alone was
insufficient evidence of charge guidance.

### Is the intended reward tilt itself too weak? (approximate diagnostic)

Using all 192 saved prior-pool proposals per prompt across seeds, assign each
proposal weight exp(2*S), normalize **within that prompt**, then average its
weighted metrics equally across the eight prompts. This self-normalized importance
estimate gives a target mean score about **−0.795**, charge success **27.8%**, and
joint success **11.7%**, versus the trained GFN's −1.882, 5.2% and 2.9%.
Effective sample sizes range **47.8–75.5 of 192** per prompt. Estimated log Z ranges
−3.640 to −1.694; these are sample estimates, not exact normalizers.

This is a post-hoc approximation from a limited prior pool, not exact distribution
validation, a new sampler, or a promised attainable measured yield. Unseen regions
of sequence space remain unassessed. Together with selection's real outputs, it
suggests that the specified tilt has useful headroom and that the GFN has not
learned it; simply declaring the reward too weak is unsupported.

### Decision

Do not move to a biological-function target or claim successful charge guidance.
Keep this as the first open-generation failure, distinct from the earlier successful
finite-candidate stability GFNs. The direct charge calculation is verified; unlike
the stability stage, fitting a noisy reward predictor is not the demonstrated issue.

Next work should diagnose policy optimization on the **same charge objective**:
separate head-update behavior from normalizer fitting, inspect gradient scaling,
and establish a small controlled learning/convergence check before spending on
another full run. Change one factor at a time (optimization or horizon), keeping
fixed all other comparisons and preserving the full baseline. No new training
configuration is selected or executed by this analysis. SubTB, stronger rewards,
unfreezing ProGen and biological objectives are not justified automatic next steps.

Selection spends four prior proposals per retained output; swaps spend extra cheap
score evaluations. They are different output distributions, not proofs of superior
target sampling. Nevertheless this GFN provides no practical score advantage to
justify its roughly one-hour training cost in the current pilot.

### Verification of completed outputs

CPU-only checks matched saved sample/config/cost checksums and current implementation
hashes against provenance; checked all 1,920 raw records against the 3,456 scored
records; independently reconstructed charge scores/thresholds from residue counts
and disorder thresholds from saved V3/V2 arrays; and confirmed primary arm success
counts. All three training logs contain exactly steps 1–400; final checkpoint step
and prompt log-normalizers match the logs. No real ProGen/scorer inference,
training or GPU execution was performed. Original run artifacts are unchanged.


Further investigation: [checkpoint/gradient checks and tiny exact learning controls](idr_charge_diagnosis_2026-11-12.md)
find weak context response and noise-sensitive learning, without an identified
missing-gradient or reward-sign bug. The next recommendation is a short real-feature
diagnostic; no new full training setting has been chosen.
