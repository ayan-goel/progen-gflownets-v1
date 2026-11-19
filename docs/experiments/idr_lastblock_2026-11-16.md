# Last-block adaptation for IDR GFlowNet generation — 16 November 2026

## Question and rationale

Can adapting ProGen's last transformer block with the GFlowNet objective improve
fresh IDR generation over training the same guidance head on frozen features?
The project remains generation with a GFlowNet toward the existing charge and
predicted-disorder requirements. This is a bounded architecture comparison, not
a switch to population search or a new biological property.

The [full-support transfer experiment](idr_transfer_2026-11-16.md#completed-results--16-november-2026)
fit 408 stored paths almost exactly but did not improve generation: joint success
was 8.50% unchanged, 6.45% ordinary TB and 6.93% corrected TB. Its common holdout
error improved without approaching training error. This supports testing
generalization beyond stored paths; it does not establish that frozen features
are the cause. Earlier [continued supervision](idr_warmstart_2026-11-15.md#completed-results--15-november-2026)
reached 16.11% with the same frozen representation and head, so that saved model
must remain a useful-performance comparator.

Changing the last transformer block lets the policy learn how to represent the
sequence prefix. Updating only the LM output head would primarily change another
readout of fixed features. Increasing capacity while repeatedly fitting the same
408 paths would not address the observed failure, so both trained arms receive
the same refreshed trajectories. Their direct comparison tests adaptation under
this shared training curriculum; comparing against historical fixed-bank results
does not isolate one change.

## Four arms and immutable reference

| Arm | Learning in this experiment |
|---|---|
| Unchanged | Original supervised starting head; no updates |
| Continued supervision | Saved completed supervised continuation; no new fit |
| Frozen-head TB | Train the original direct-context head and prompt normalizers |
| Last-block TB | Train that same head, prompt normalizers and the final ProGen transformer block |

Both TB arms start from the identical original supervised head and original ProGen
weights. All earlier transformer blocks, token embeddings, final layer normalization
(`ln_f`) and LM output head remain frozen. The adapted policy uses its changed
last-block states for both the frozen LM readout and the direct-context correction.
The original last block remains available separately for the fixed reference
probabilities. No gradient enters that reference or the disorder scorer.

Use the ordinary `DirectContextHead` parameterization in both arms. Do not apply
the transfer experiment's conditioning transform or gradient correction. Ordinary
TB already fit its full-path bank, and the present question concerns representation
and fresh generation. Adding those mechanisms would complicate interpretation.

**What is carried forward:** the model/tokenizer, prompts and support, existing
charge/disorder reward, ordinary TB, the supervised starting weights, and a small
training-only replay contribution. Earlier SubTB, feature conditioning, SVRG and
restricted-bank sampler implementations are not used by this trainer. Their
results motivate the comparison; they are not stacked into its architecture.

The unchanged target and trajectory-balance loss are

\[
R(x\mid c)=p_0(x\mid c)\exp[-2.5V(x)],\qquad
\mathcal L_{\rm TB}
=\bigl[z_c+\log q_\theta(x\mid c)-\log p_0(x\mid c)+2.5V(x)\bigr]^2.
\]

- `c`: one of eight supplied 16-residue IDR prefixes; `x`: its complete generated
  64-residue tail.
- `p0`: original, immutable ProGen probability, conditional on the prefix and
  normalized over all 20 canonical residues at every position, at temperature one.
- `V`: the existing sum of normalized outside-tolerance violations for the two
  charge blocks, fraction charged and predicted disorder; `R`: unnormalized
  terminal reward. The penalty 2.5 is inherited unchanged.
- `q_theta`: the current generation policy; `theta`: trainable guidance-head
  parameters, plus final-block parameters in the adapted arm. Its sequence
  probability is the product of its next-residue probabilities.
- `z_c`: a trainable log-normalizer for prompt `c`; `L_TB`: the squared trajectory
  balance residual. All logarithms are natural.

The [continuous-reward record](idr_continuous_2026-11-15.md#reward-and-acceptance-criteria)
owns exact tolerances, scorer conventions and the penalty's calibration limits.
Keep all 20 residues available throughout generation, without selection, rejection
or restrictions to bank membership. Reference `p0` must not be replaced with the
adapted policy when computing rewards. Frozen final-layer feature caches cannot
serve as the adapted model's inputs; recompute the changed block from frozen
lower-block states.

Initialize each prompt normalizer from the same original training-bank offsets,
with equal total success/failure weight, as in the
[supervised-start protocol](idr_warmstart_2026-11-15.md#tb-normalizer-initialization).
This is an initialization heuristic, not a true partition-function estimate.

## Shared refreshed training data

Run 100 refresh rounds, each followed by four optimizer updates per arm. At the
start of a round, form one shared 32-trajectory batch:

| Source | Trajectories per round | Purpose |
|---|---:|---|
| Current frozen-head policy | One per prompt, eight total | Cover states visited by the frozen comparator |
| Current last-block policy | One per prompt, eight total | Cover states visited by the adapted policy |
| Original unguided ProGen reference | One per prompt, eight total | Retain broader reference-policy coverage |
| Original training replay bank | One per prompt, eight total | Retain known successes and near misses |

Choose replay from the inherited 408 training rows, balancing pass/fail exposure
across rounds. The 95 diagnostic rows never supply replay or normalization data.
Score each new complete trajectory with the unchanged reward. Both arms receive
the identical 32 trajectory IDs and target values for all four updates before
the next refresh; neither arm receives extra examples from its own policy.
Record source, round, prompt and sequence for every draw. Do not select the fresh
examples by success or add a separate search procedure.

This produces 2,400 fresh training trajectories and 800 replay presentations at
collection, with 12,800 trajectory presentations per fitted arm over 400 updates.
Repeating each batch four times limits generation cost while keeping examples
recent. It does not establish an optimal update-to-generation ratio. The shared
pool couples the two arms: this is a comparison under a common, adaptive curriculum,
not independent benchmarking of two complete data-collection systems.

Track any diagnostic-tail overlap with all training trajectories. Exclude exposed
tails from common holdout summaries for every arm/stage and report the resulting
counts. Do not use diagnostic membership to filter training draws. Holdout data
have already been inspected in prior work and remain development diagnostics.

## Fixed optimization budget and rationale

| Choice | Value and rationale |
|---|---|
| Training run | One seed, seed 0; bounded development comparison |
| Updates | 400 per trained arm, matching the recent continuation budget |
| Batch / microbatch | Shared 32 trajectories; accumulate microbatches of four with global mean weights |
| Head learning rate | `1e-4`, inherited |
| Direct task-branch rate | `1e-3`, inherited tenfold head multiplier |
| Prompt-normalizer rate | `1e-2`, inherited |
| Last-block rate | `1e-5`, a conservative engineering default tenfold below the head rate; not literature-optimized |
| Optimizer | Fresh Adam, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay zero; inherited defaults |
| Decay | All rates multiplied by 0.1 after update 300 |
| Clipping | Head plus normalizers clipped together at norm 1 in both arms; adapted block separately clipped at norm 1 |
| Selection | Final update-400 models only; no outcome-selected checkpoint |
| Training / replay RNG seeds | `123000` / `125000`, separate from previous experiments |
| Evaluation RNG seed | `133000`, reset for each arm; independent of training |

Inherit the 128-unit head, eight prompts, 16-residue prefixes, 64-residue tails,
temperature one and float32 model parameters. Disable dropout and TF32; deterministic
PyTorch operations are required. TB residual arithmetic uses float64 after float32
path likelihoods to limit additional cancellation when subtracting large path logs;
this numerical convention is shared by both new arms.
Checkpoint every 100 steps and at the final step; only the fixed final model is
evaluated. Save modified weights and normalizers, without automatic optimizer resume.

Separate clipping prevents the much larger block's gradient norm from shrinking
the head update merely because more parameters participate. Record pre-clip norms
and clipping frequencies for each group, plus the last block's parameter change.
These diagnose negligible adaptation or drift; they do not authorize automatic
learning-rate tuning. Microbatching must accumulate the full shared-batch loss
before clipping and stepping, not clip each microbatch separately.

The two arms match update count and training examples, not gradient computation
or wall time. Report frozen-model inference, generation and adapted-block fitting
costs. No reward sweep, loss switch, curriculum over property difficulty, extra
supervised fit or automatic extension belongs to this run.

## Fresh evaluation and decision rules

Generate 128 raw tails per prompt for each of four arms: 1,024 per arm and 4,096
total. Use the same fresh evaluation seed, while recognizing that trajectories
diverge as policy probabilities differ. Do not compare fresh measurements directly
with the old 16.11% estimate as if they used the same draws.

Primary outputs are raw joint charge/V3 predicted-disorder success and joint
success excluding exact copies of a common training collection. That collection
must include the original replay/training history, the saved supervised control's
consumed examples and all new training draws. The noncopy metric keeps all 1,024
outputs in its denominator. Report charge and V3 separately, V2 sensitivity,
unique-tail fraction, repetition, near-bank identity, per-prompt results and cost.

**Architecture screen:** last-block TB must improve both raw and common-training-
noncopy joint success by at least five percentage points over **each** of frozen-head
TB and unchanged initialization. Its predicted-disorder and unique-tail rates
must each fall by no more than five points against either comparator.

**Useful-performance screen:** in addition, last-block TB must remain within two
percentage points of the freshly evaluated saved supervised model on both raw and
noncopy joint success, and within five points on disorder and uniqueness. This is
a practical competitiveness requirement, not a statistical noninferiority test,
and passing it does not establish superiority over supervision. Report the direct
differences against every comparator rather than only a combined pass flag.

These are predeclared engineering screens, not significance tests. A second seed
is a subsequent decision, not part of this command. Interpret outcomes separately:

- If adapted TB improves materially over frozen TB and remains competitive with
  the supervised comparator, adaptation merits confirmation.
- If both new TB arms improve similarly, refreshed training may be useful under
  this curriculum, but do not credit the improvement to unfreezing.
- If adaptation beats frozen TB while still trailing the saved supervised model
  materially, there is an architectural signal without a competitive final method.
- If neither improves, stop this bounded approach. Do not automatically train
  longer or unfreeze additional layers. A single conservative-rate failure would
  not prove all partial fine-tuning impossible.

Report training and common held-out TB residuals as diagnostics, separately from
fresh generation success. Neither a selected-bank fit nor a finite-pool softmax
is an exact global target-distance measurement. Predicted disorder remains an
optimized proxy; no result establishes experimentally validated IDR behavior.

## Inputs, provenance and implementation boundary

Use `runs/idr-continuous-v1` for the original head, training bank, references and
continuous-reward conventions. Its generation-manifest SHA-256 is
`2f4dfc3296b4d72b8eba7f8bb57a5f1a9832677d488bcbdbe6923d1ff85c0c11`.
The starting `initial.pt` SHA-256 is
`7d2256b7e5ec2150762f85ea458d7470dfddaaaf0757824da2861be76e52e584`.

The saved supervised comparator is exactly
`runs/idr-warmstart-v1/continued_supervised/step-400.pt`, SHA-256
`e0db233fa73caaee03a9b7f7076d87ce595236ec0a69b2bd58350cc4f9a7a999`.
Its generation manifest has SHA-256
`eab5e8066d85ebe8244aa26b5b3d87ac0def8fd3dd4aa3d3990cf9f9473b2312`.
These hashes were checked against actual files before implementation; preserve the
source artifacts and inherited implementation provenance.

Save the effective configuration, source hashes, exact common training draws,
reference likelihoods/rewards, checkpoints, update logs and fresh outputs. Source
snapshots must distinguish the immutable reference block from the trainable copy.
The assistant may run synthetic CPU fixtures and read-only preflight. All real
training, ProGen inference, generation and GPU execution are user-run. Verification
uses tiny random models and an injected synthetic scorer. The
[verification record](../verification.md#last-block-adaptation--16-november-2026)
owns CPU test evidence and the real-input read-only preflight.

From the activated environment, after the user selects `CUDA_VISIBLE_DEVICES`, run:

```bash
scripts/run.sh idr-lastblock --output runs/idr-lastblock-v1 --device cuda
```

Training, fresh generation and scoring run together. See the
[runbook](../runbook.md#last-block-idr-gflownet-adaptation--completed) for preflight,
artifacts and scoring recovery. **The user completed this run; results follow below.**

## Literature and limits

[Hu et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/bc667ac84ef58f2b5022da97a465cbab-Paper-Conference.pdf)
provide a direct precedent for an adapted autoregressive language-model GFlowNet,
using LoRA and a mixture of current-policy, tempered-policy and replay trajectories.
The reviewed material is section 3.3, the LoRA footnote and relevant appendix
settings. Their work supports adaptation as a legitimate GFN design, not this
exact final-block restriction, our optimizer values or expected IDR performance.

The [existing local-credit review](../../literature/RL_REWARD_DESIGN_2026-11-15.md)
records the distinct prefix-flow alternative. It remains a possible subsequent
structural direction, not an additional intervention in this comparison. GPA can
remain background or a later reference method; it is not the generator trained
or evaluated here.

## Completed results — 16 November 2026

The user completed `runs/idr-lastblock-v1` in **45 min 6 s**. Both arms performed
400 paired updates, each seeing the same 12,800 trajectory presentations. The
shared curriculum collected 2,400 fresh training trajectories. All four policies
produced 1,024 fresh evaluation tails each; no holdout tail was encountered during
training. The predeclared architecture and useful-performance screens both pass.
The [saved-artifact audit](../verification/idr-lastblock-results-2026-11-16.json)
owns manifest, reconstruction and pairing checks; no new model/scorer run is
needed to inspect these results.

| Policy | Joint successes / 1,024 | Joint rate | Charge pass | V3 disorder pass | Unique tails | Exact common-training copies |
|---|---:|---:|---:|---:|---:|---:|
| Unchanged starting head | 90 | 8.79% | 13.18% | 77.25% | 99.90% | 3 |
| Saved continued-supervised head | 142 | 13.87% | 19.43% | 78.12% | 100.00% | 1 |
| Frozen-head TB | 127 | 12.40% | 14.45% | 89.06% | 96.88% | 34 |
| Last-block TB | **209** | **20.41%** | **21.97%** | **93.85%** | **97.36%** | 27 |

Joint success means meeting **both** the charge criteria and isolated-tail V3
predicted-disorder threshold. Last-block TB gains **8.01 percentage points** over
the matched frozen TB arm, **6.54 points** over the freshly evaluated saved
supervised model, and **11.62 points** over initialization. All exact common-training
copies fail the joint criterion, so all four noncopy joint rates equal their raw
joint rates. The gain is not explained by copying successful training tails.

Charge and disorder each improve against frozen TB. Against supervision, much of
the joint advantage accompanies better disorder retention; charge alone improves
by only 2.54 points. Related-scorer V2 joint success is 20.02% adapted, 12.40%
frozen and 14.06% supervised. This is sensitivity evidence, not an independent
biological validation. The saved supervised model's fresh 13.87% differs from its
old 16.11% because the evaluation draws changed, not because it was retrained.

### Plain-ProGen comparison — post-hoc clarification

The 8.79% unchanged arm includes the previously supervised guidance head; it is
**not plain ProGen**. The current four-arm final evaluation omitted a dedicated
unguided arm. Earlier `runs/idr-replay-v1` contains an unguided evaluation with
17/1,024 joint passes (**1.66%**) and 667/1,024 disorder passes (**65.14%**), versus
209/1,024 (**20.41%**) and 961/1,024 (**93.85%**) for the new adapted GFlowNet.
The original checkpoint, eight prompts, 16-residue prefixes, 64-residue tails,
temperature, residue support and acceptance thresholds match. Historical reward
weights changed between studies, but those binary acceptance criteria did not.
Counts were recomputed from saved disorder arrays without scorer/model execution.

The current run also collected 800 unselected original-ProGen paths during
training: 19 joint passes (2.375%) and 533 disorder passes (66.625%). This is
auxiliary development evidence because both trainers consumed those trajectories.
Neither collection is a fresh, dedicated final-evaluation control for this run.
Do not present the historical-to-current ratio as a newly replicated precise effect.

For the project's main claim, compare plain ProGen and guided ProGen on equal
numbers of raw outputs under the same conditions. Retain the frozen GFN to isolate
the value of last-block adaptation. The saved supervised comparator is head-only;
it does not isolate the GFN objective against ordinary fine-tuning of the same
last block. Such a capacity-matched non-GFN control would support a later
method-specific claim, rather than merely showing useful guidance over ProGen.

### Prompt consistency and repetition

These are the predeclared per-prompt outcomes, with 128 raw tails per cell:

| Prompt | Unchanged joint count | Supervised | Frozen TB | Last-block TB |
|---|---:|---:|---:|---:|
| DP01299 | 8 | 17 | 26 | 40 |
| DP00187 | 16 | 23 | 26 | 39 |
| DP00181 (RPB1) | 2 | 10 | 1 | 6 |
| DP00092 | 13 | 15 | 20 | 30 |
| DP00510 | 9 | 12 | 6 | 14 |
| DP01076 | 9 | 15 | 10 | 15 |
| DP01538 | 12 | 17 | 7 | 20 |
| DP01109 | 21 | 33 | 31 | 45 |

Adaptation beats frozen TB on all eight prompts. It beats supervision on six,
ties on one and loses on RPB1. A **post-hoc** leave-one-prompt-out check leaves
gains of 7.59–8.59 points over frozen TB and 4.91–7.92 points over supervision;
the aggregate improvement does not depend on one prompt. This is a sensitivity
check on inspected development data, not additional held-out evidence.

RPB1 remains a weakness: all 27 adapted common-training copies occur there,
and only 101/128 outputs are unique. Its mean repeated-trimer fraction is 53.34%.
The other seven adapted prompt groups have 128/128 unique tails and no exact
training copies. Overall repetition is lower than frozen TB (10.92% versus
16.41%) but higher than supervision (8.40%). Keep RPB1 in confirmation; dropping
it now would change the task after inspecting outcomes.

### Target fitting and implementation checks

Useful generation improved, but not every target-fitting diagnostic improved:

| Diagnostic | Unchanged | Supervised | Frozen TB | Last-block TB |
|---|---:|---:|---:|---:|
| Original 408-row training-bank TB RMSE | 3.770 | 6.161 | 2.909 | 2.567 |
| Common 95-row holdout TB RMSE | 3.981 | 6.114 | 3.016 | 3.528 |
| Holdout within-prompt centered log-ratio RMSE | 1.997 | 2.616 | 1.637 | 2.106 |
| Fresh own-policy TB RMSE | 6.829 | 7.158 | 3.860 | 2.981 |

The adapted model has worse common-holdout residuals than frozen TB. Fresh
own-policy residuals use different generated sequences, so they do not override
that common-data comparison or prove global target matching. Control offsets
are training-only diagnostic initializations. The adapted own-policy mean
`log q - log p0` is 2.268 nats per tail versus 1.340 for frozen TB and 3.352 for
supervision; these are sampled reference-drift estimates, not target KL distances.

The 28,322,304-parameter copied block changes by 1.217% in relative L2 norm.
The original 764,803,616-parameter reference has zero trainable parameters and no
recorded gradients. All 400 head/normalizer updates in both arms and all adapted
block updates clip at their declared norm limit; training remains finite, but one
successful run does not establish robust optimization across seeds. No earlier
blocks or original output weights are optimized.

Paired training took 18 min 59 s, including 14 min 45 s collecting/scoring refreshed
examples and 4 min 14 s for paired updates. Four-arm generation took 25 min 4 s;
common-bank diagnostics took 33 s and final scoring 22 s. These costs include a
shared lower-layer computation and are not independent training-time comparisons.

### Interpretation and next decision

This is a positive result for **GFlowNet-trained generation with limited ProGen
adaptation under the common refreshed-data curriculum**. It supports retaining
the last-block design for confirmation. Both the starting head and original prior
remain the same as the controls. Improvements relative to earlier static-bank
experiments cannot all be attributed to unfreezing because the training curriculum
also changed; the matched frozen control supplies the direct comparison here.

The result is still one fitted seed on eight familiar development prompts and one
charge profile. Around 80% of raw outputs fail the joint screen, RPB1 repetition
persists, common-holdout target fit remains imperfect, and disorder is predicted
rather than measured. No claim of general biological function, global novelty,
unseen-prompt generalization or exact full-space reward sampling follows.

**Proposed next step:** confirm the same matched comparison with two fresh
training/replay/evaluation seed sets, restarting from the original supervised
head. Keep prompts, reward, rates, budgets and final-checkpoint selection fixed;
report every run and per-prompt outcome. Add a fresh, equal-size **plain-ProGen
evaluation arm**, without changing the training or original progression gates,
to measure the end-goal baseline directly. This would test reproducibility of a
material effect before changing the task or architecture. It is a proposal,
not an automatically launched extension or an implemented multi-run command.
