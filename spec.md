# Stability-guided protein generation with frozen ProGen2

**Scientific specification revised: 17 October 2026.** Execution progress is maintained
in [current status](docs/status.md), commands in the [runbook](docs/runbook.md), and
dated evidence in the [experiment index](docs/experiments/README.md).

## 1. Project objective

Guide a pretrained protein language model toward stable proteins by learning an additional guidance model with a GFlowNet objective. Keep ProGen2 fixed and use completed-sequence stability feedback to train guidance over its generation decisions.

Use **Mega-scale folding-stability data** as the biological foundation. Begin with variants near measured reference proteins, then investigate how performance changes as generation moves farther from those references. The scope is general protein stability, starting within the small-domain regime represented by the dataset.

The central research question is:

> Can GFlowNet-trained guidance produce diverse, more stable protein variants from frozen ProGen2, within a controlled mutation neighborhood, more accurately or economically than alternative guidance and selection methods?

Success requires credible stability evaluation, useful sequence diversity and a fair accounting of training and generation costs. Improved predicted rewards alone are insufficient.

## 2. Current foundation

- **ProGen2-medium runs independently.** The existing checkpoint has approximately 765 million parameters and 1,536-dimensional hidden representations. Repetition and termination behavior have been investigated; successful decoding does not establish biological quality.
- **Standalone GFlowNet training and evaluation are complete for three TF-Bind-8 seeds.** Mean selected-batch score was 0.690, compared with 0.653 for random proposals with the same learned-score selection and 0.471 for unranked random proposals. The paper reports 0.840. This is a working adapted pipeline, with a remaining numerical reproduction gap. [Run report](../progen-gflownets-test/gflownet/runs/tfbind8/report.md)
- **The integrated exact-task gate passed on 15 October 2026.** Frozen ProGen2 plus TB-trained guidance reached final TV 0.04090 / 0.04686 / 0.04472 across three seeds on the 256-sequence synthetic task. This clears the engineering milestone; the subsequent biological result is recorded below. [Milestone record](docs/experiments/exact_pass_and_biological_pilot_2026-10-15.md)
- **The first biological pilot completed on 15 October 2026.** The scorer passed validation and GFN accurately fit its predicted target, but measured stability did not improve over the constrained ProGen baseline. Reward ranking and support concentration require analysis before further expansion. [Pilot results](docs/experiments/biological_first_results_2026-10-15.md)
- **Literature review is available.** It identifies relevant objectives, architectures and comparisons. [Review archive](literature/README.md)

Reuse the existing ProGen adapter knowledge and the [BioSeq-GFN-AL implementation](https://github.com/MJ10/BioSeq-GFN-AL) as references. The TF-Bind-8 generator operates on DNA and is not a pretrained protein guidance head. Preserve the source snapshots, checkpoints and results in `../progen-gflownets-test`.

## 3. Biological task and data

Use Tsuboyama et al.'s **Mega-scale experimental analysis of protein folding stability in biology and design**. The curated data contain roughly 776,000 stability measurements from natural and designed domains of 40–72 residues, including single mutants and selected double mutants. These are assay-derived thermodynamic folding-stability estimates, not melting-temperature measurements. [Paper](https://www.nature.com/articles/s41586-023-06328-6), [data release](https://zenodo.org/records/7992926)

### Data contract

The data preparation stage must record:

- Release URL, version, checksums, filtering decisions and retained row counts.
- Full domain sequence, parent/reference identity, mutation positions, sequence length and measurement-quality fields.
- The exact stability-label column, units, sign convention and missing/censored-value handling.
- Duplicate handling and immutable training, validation and test manifests.

Prefer the authors' machine-learning-filtered measurements. Verify the distinction between absolute stability, **ΔG**, and stability change relative to a reference, **ΔΔG**. In this dataset's convention, positive ΔΔG denotes stabilization. Missing measurements must not become zero rewards or negative examples. [Data filtering and label conventions](https://www.nature.com/articles/s41586-023-06328-6)

### Initial generation space

Choose one reference protein with suitable measurements. Generate its full domain sequence from left to right, retaining its length and allowing only canonical amino-acid substitutions. Start with at most **one substitution**, then examine a budget of two when the data and predictor support that experiment. Insertions, deletions and variable-length generation are outside the first implementation.

An action is legal only if the resulting prefix can still complete a sequence within the mutation budget. Once the budget is exhausted, remaining positions must match the reference. ProGen still processes these residues so subsequent representations remain correct.

Use two clearly labeled evaluation settings:

| Setting | Candidate support | Interpretation |
|---|---|---|
| Measured-candidate benchmark | A declared set of measured variants, with evaluation labels withheld from learning and selection | Retrospective design on a finite candidate set |
| Mutation-neighborhood generation | All legal variants within the declared budget | Broader generation; some candidates may have no measured outcome |

For the finite benchmark, candidate sequence identities may define legal prefix completions; their held-out stability values may not influence training. All methods must use the same candidate support. For broader generation, report measurement coverage and predicted outcomes separately; performance on the measured subset alone cannot establish performance on all outputs.

## 4. Model architecture

```text
Reference protein and mutation budget
                    |
Generated prefix -> frozen ProGen2 -> base logits and prefix features
                                           |
                                  trainable guidance head
                                           |
                               corrected legal-token logits
                                           |
                               sample the next amino acid
                                           |
                                  completed sequence
                                           |
                                fixed stability predictor
                                           |
                           whole-sequence GFlowNet training
```

| Component | Responsibility | Trainable during guidance training? |
|---|---|---|
| ProGen2-medium | Prefix representations and next-token predictions | No |
| Stability predictor | Estimate stability of completed sequences | No; fit and validate beforehand |
| Guidance head | Produce one correction per next-amino-acid action | Yes |
| Normalization estimate | Support the trajectory-balance objective | Yes |
| Mutation constraints | Enforce the candidate space | No; deterministic rules |

Start with a small head reading the frozen prefix representation, current position, reference residue and remaining mutation budget. One reference per initial run keeps conditioning simple. A shared model across references may later need additional reference representations and a context-dependent normalizer.

Initialize the head's output corrections to zero. At each position, add corrections to ProGen logits, mask illegal actions and normalize over legal tokens. Zero guidance must exactly recover the constrained baseline.

The head may use the known reference sequence. It must not access future tokens of the generated candidate. ProGen parameters, original output head and evaluation mode remain fixed.

## 5. Target distribution and reward

Define the baseline by normalizing frozen ProGen logits over legal actions at each prefix, using temperature one and no top-k or top-p truncation. Multiply those conditional probabilities to obtain its sequence probability.

This **locally constrained baseline** is a deliberate modeling choice. It generally differs from globally conditioning unrestricted ProGen on the final mutation constraint. Sampling, reward calculation and evaluation must all use the same definition.

The target is:

$$
R(x\mid c)=p_0(x\mid c)\exp\!\left(\beta\,\hat{s}(x,c)\right),
\qquad
p^*(x\mid c)=\frac{R(x\mid c)}{Z(c)},
\qquad
q_\theta(x\mid c)\approx p^*(x\mid c).
$$

| Symbol | Meaning |
|---|---|
| \(x\) | Full completed candidate domain sequence. |
| \(c\) | Fixed design context: reference protein, mutation budget and any declared candidate-support restriction. |
| \(\mid\) | “Given”; the distribution is conditional on the design context. |
| \(p_0(x\mid c)\) | Sequence probability under the locally constrained frozen-ProGen baseline; subscript zero denotes that baseline. |
| \(\hat{s}(x,c)\) | Dimensionless, scaled predicted stability improvement relative to the reference; the hat denotes an estimate. |
| \(\beta\) | Nonnegative guidance strength, fixed for a run and selected using validation. |
| \(\exp\) | Exponential function, converting the property score into a positive reward. |
| \(R(x\mid c)\) | Unnormalized combined sequence weight. |
| \(Z(c)\) | Sum of combined weights over all legal completed sequences. |
| \(p^*(x\mid c)\) | Desired normalized sequence distribution; the star identifies the target. |
| \(q_\theta(x\mid c)\) | Sequence probability under ProGen plus learned guidance. |
| \(\theta\) | Trainable guidance-model parameters. |
| \(\approx\) | Approximate equality: the goal of learning, not a claim of exact convergence. |

The earlier notation \(r(x)^\beta\) expresses the same property factor if **\(r\)** is the positive reward obtained by exponentiating the scaled score. This specification makes the stability score explicit and allows it to depend on the reference.

### Stability predictor and score construction

First fit a small regression model on frozen full-sequence representations, with a linear baseline and a small multilayer model as initial candidates. Select the predictor through validation before using it as a reward source. Its full-sequence features are separate from the causal prefix features used by the guidance head.

Predict absolute stability in the verified dataset convention, subtract the predicted reference stability and divide by a positive scale fixed from training data. This produces the dimensionless improvement score. Subtracting a constant reference score helps interpretation but does not change the normalized target for that reference.

Any clipping bounds must be chosen from training/validation evidence, fixed before final evaluation and stored in the run configuration. Start with a fixed predictor and fixed reward throughout a guidance run. Do not initially add a changing uncertainty bonus or jointly update the scorer and generator.

At guidance strength zero, the target equals the baseline and the normalization is one. Increasing the strength favors predicted stability more strongly, but can also reduce diversity or expose scorer errors. Including the prior does not guarantee folding or function.

## 6. GFlowNet training

For left-to-right construction, each completed sequence has one construction path. Use the trajectory-balance loss:

$$
\mathcal L(x,c)=
\left[
\log Z_\phi(c)+\log q_\theta(x\mid c)
-\log p_0(x\mid c)-\beta\hat{s}(x,c)
\right]^2.
$$

The symbols from the target formula retain their meanings. **\(\mathcal L\)** is the per-sequence training loss; **\(Z_\phi\)** is a learned normalization estimate; **\(\phi\)** denotes its parameters; **\(\log\)** is the natural logarithm; and **the square** penalizes either sign of discrepancy. For one fixed context and strength, learn a single scalar log-normalization. Different contexts, supports or strengths generally require different normalization values.

Training loop:

1. Sample complete legal sequences using the current guided policy.
2. Score each completion with the fixed stability predictor.
3. Compute baseline and guided log probabilities, summing over all generated positions.
4. Compute the loss and update only the guidance parameters and log-normalization.
5. Evaluate at scheduled checkpoints and preserve raw proposals and diagnostics.

Deterministically forced residues have probability one under both constrained policies and contribute zero log probability. They still require backbone processing. Fixed-length termination is deterministic; there is no omitted sampled stop probability.

Begin with on-policy training. Add exploratory sampling or replay only after correctness checks, keeping the behavior policy distinct from the probability used in the loss. If replay is used, its sequences must satisfy the current support and use the same fixed reward definition.

Zero TB loss over all trajectories implies reward-proportional sampling under the appropriate assumptions; low loss on visited examples alone does not establish distribution accuracy. [Trajectory balance](https://arxiv.org/abs/2201.13259)

## 7. Evaluation and comparisons

### Data boundaries

- **Local variant design:** hold out mutations within the selected domain and document which reference/domain observations the predictor receives.
- **Generalization to new proteins:** group related parent proteins before splitting; exclude held-out groups from scorer fitting and hyperparameter selection.
- Fit data preprocessing, reward scaling and model-selection rules using training/validation data only. Keep test stability values out of guidance rewards, replay prioritization, filtering and checkpoint selection.

Do not describe within-domain mutation prediction as generalization to unseen protein families. A short mutation distance is a controllable setting, not proof that the predictor is reliable.

### Metrics

| Goal | Required measurements |
|---|---|
| Stability | Predicted and held-out measured scores separately; improvement over reference; fraction stabilized; predictor error on selected candidates |
| Diversity | Unique fraction, duplicate rate and mean pairwise Hamming distance within each reference's fixed-length batches |
| Novelty | Distance to the stated training/reference set; fraction absent from that set |
| Locality | Mutation-count distribution and constraint violations |
| Coverage | Fraction with valid held-out measurements; uncertainty about unmeasured candidates |
| Distribution correctness | Exact total variation distance, both KL directions and partition error on enumerable tasks |
| Efficiency | Feature extraction, scorer fitting, guidance training, scorer calls, backbone work, sampling time and peak memory |

Hamming distance counts differing residue positions. For comparisons across lengths, also report distance divided by sequence length. Total variation and KL divergence measure differences between probability distributions; the exact test must specify their directions and conventions.

Save raw samples and selected batches separately. Deduplication and top-candidate selection change the output distribution and must not be hidden. Use diversity and novelty as concepts from PG-AbD, with explicit definitions appropriate to this task; do not assume its numerical metrics are interchangeable with ours. Perplexity is not a headline outcome.

### Baselines

1. **Constrained ProGen:** zero guidance, identical support and decoding rules.
2. **Generate and select:** larger baseline pools ranked by the same stability predictor.
3. **Supervised guidance:** a comparable head trained to estimate future terminal reward under baseline continuations; label this a FUDGE-style adaptation.
4. **GFlowNet guidance:** the proposed trajectory-balance learner.

Use matched raw-pool and selected-batch sizes where appropriate, and also compare total cost at comparable stability and diversity. Count all setup costs when assessing whether learned guidance pays off over repeated generation.

Before a broader method claim, add a correctly matched KL-regularized RL comparison. Here KL means Kullback–Leibler divergence, and RL means reinforcement learning. The reward and regularization must target the same distribution. TB/RTB has close connections to maximum-entropy RL, and some formulations or expected on-policy updates are equivalent. Compare actual estimator, exploration and replay differences rather than algorithm names. [RTB and Trust-PCL](https://arxiv.org/abs/2509.01632)

Particle sampling and a separate small GFlowNet generator are later comparisons for inference cost and architectural tradeoffs.

## 8. Implementation deliverables

Implement these components in v1, with explicit configurations and documented interfaces:

| Component | Deliverable |
|---|---|
| Data preparation | Versioned Mega-scale manifest, cleaned tables, reference mapping, immutable splits and coverage report |
| ProGen adapter | Frozen checkpoint loading, tokenizer validation, legal-action probabilities, cached decoding and teacher-forced likelihoods |
| Sequence environment | Reference-conditioned fixed-length construction, mutation accounting and optional finite-support prefix constraints |
| Stability model | Training/validation pipeline, frozen scorer checkpoint, prediction API and reward-transform configuration |
| Guidance model | Zero-initialized correction head with prefix/context features |
| GFlowNet trainer | Stable log-space TB objective, normalization estimation, checkpoints and reproducible seeds |
| Baselines | Unguided sampling, generate-and-select and supervised guidance using shared task definitions |
| Evaluation | Exact-distribution checks, measured/predicted stability separation, diversity/novelty/locality and cost reports |
| Run interface | One documented entry point for a configured training-and-evaluation run, plus evaluation from saved checkpoints |

Each run must save its configuration, data/scorer/checkpoint identities, seeds, training curves, model checkpoint, raw sequences, selection decisions, metrics, timing and completion status. Evaluation must be repeatable without retraining. Do not overwrite existing runs.

## 9. Current implementation sequence

### A. Audit data and establish the task

Inspect Mega-scale labels and coverage, select an initial reference, define splits and candidate support, and document the label convention. Verify checkpoint/tokenizer identity and repair environment paths affected by the previous folder rename.

**Deliverable:** a reproducible task manifest and data-quality report. Specific references, numerical reward scaling and compute caps are set from this audit and a small runtime pilot.

### B. Verify the guidance mechanism exactly

Use a four-token alphabet `{A,G,L,S}` and four generated positions, giving 256 completions. Validate the boundary prefix `1`. Use the positive toy reward whose log is the negative squared difference between glycine count and two. This task checks mechanics, not protein stability.

**Acceptance criteria:** zero guidance recovers the baseline; distributions normalize; likelihood implementations agree; backbone weights remain unchanged; and beta-one training reaches a proposed exact total variation below 0.05 in each of three seeds. Also report both KL directions and log-partition error. The threshold is an engineering criterion, not a literature standard.

If learning fails, fit the same head to exact conditional targets as a privileged capacity diagnostic. Do not count that diagnostic as a fair sample-efficiency baseline.

### C. Validate the stability scorer

Fit and compare simple predictors under the declared split. Inspect errors and ranking near the initial reference before freezing the scorer and reward transform. If it cannot provide useful held-out guidance, improve the scorer or narrow the task before optimizing generation against it.

### D. Run the first biological comparison

Train guidance for one reference and one-substitution budget. Compare all four initial methods on the measured-candidate benchmark, preserving held-out labels until evaluation. Use three seeds for the pilot and report each result.

**Deliverable:** one report showing stability, diversity, novelty, locality, distribution diagnostics where tractable, and total cost. A benefit over unguided ProGen alone does not establish a benefit over alternative guidance.

### E. Test the useful operating range

Expand to additional domains, then two-substitution neighborhoods and larger distances when evaluation remains credible. Separate within-domain and unseen-domain results. Increase replication for substantive comparisons as compute permits, and investigate the tradeoff between reward strength, diversity and prediction reliability.

## 10. Position in the literature

| Reference | Relevance to this project |
|---|---|
| [Tsuboyama et al., 2023](https://www.nature.com/articles/s41586-023-06328-6) | Experimental folding-stability dataset and measurement limitations |
| [ProGen2, Nijkamp et al.](https://arxiv.org/abs/2206.13517) | Pretrained autoregressive protein backbone |
| [Biological Sequence Design with GFlowNets, Jain et al.](https://arxiv.org/abs/2203.04115) | Standalone sequence-design training reference |
| [Trajectory balance, Malkin et al.](https://arxiv.org/abs/2201.13259) | Whole-trajectory training objective |
| [PG-AbD, Yin et al., 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34370) | Frozen ProGen2 reward for a separate GFlowNet antibody generator; implementation and evaluation reference |
| [Amortizing intractable inference in large language models, Hu et al.](https://arxiv.org/abs/2310.04363) | GFlowNet adaptation of pretrained language-model distributions |
| [FUDGE, Yang and Klein](https://aclanthology.org/2021.naacl-main.276/) | Supervised prediction of future desirability for generation guidance |
| [DIRECTOR, Arora et al.](https://aclanthology.org/2022.aacl-main.39/) | Shared representations and next-token attribute scoring |
| [RTB–Trust-PCL equivalence, Deleu et al.](https://arxiv.org/abs/2509.01632) | Relationship to regularized RL and fair baseline design |

PG-AbD's generator is separate from ProGen2; our generator retains frozen ProGen2 and learns corrections to its token decisions, with the property signal supplied by stability data. This architecture is a research configuration to evaluate, not an established novelty claim. The contribution sought is evidence about when the guidance improves measured outcomes, coverage or cost.

## 11. Scope and execution

This specification is the scientific contract. The [README](README.md), [literature synthesis](literature/REVIEW.md) and [experiment plan](literature/EXPERIMENT_PLAN.md) describe the same active Mega-scale stability project. [Implementation documentation](docs/README.md) records concrete decisions and verification; [prior standalone results](docs/prior_results.md) preserve historical evidence.

The first implementation excludes antibody-specific optimization, Potts-model training, full-backbone fine-tuning, unconstrained long-protein generation and conformational-ensemble design. No wet-lab outcome is implied by a computational score.

All GPU execution and real experiments remain user initiated. Tiny CPU runs may verify implementation mechanics. Implement in v1, document decisions and hyperparameters, and verify each atomic slice before proceeding. Passing CPU checks does not satisfy the real-checkpoint convergence or biological-validation gates.

## 12. Current post-pilot development diagnostic

The user authorized the [post-pilot plan](docs/next_steps_2026-10-15.md). The completed
development experiment assesses the target before further guidance training:
three development parents, with five additional parents reserved for confirmation.
Coverage-based selection, cluster/alignment screening, immutable splits and every
numerical choice are recorded in the [implementation record](docs/experiments/target_diagnostic_2026-10-15.md).

Separate inner scorer selection from outer development assessment. Compare ridge,
a small MLP and a five-member MLP ensemble using the existing frozen features.
Compute exact stability expectations and matched sampling references at beta=1.
All parent outcomes and scorer-screen failures remain visible. No confirmation
labels or guidance fits enter this diagnostic.

The accepted diagnostic ablation tempers **sequence** probabilities:

    p_alpha(x | c) = p0(x | c)^alpha / sum_u p0(u | c)^alpha
    R_alpha(x | c) = p_alpha(x | c) * exp(beta * s_hat(x,c))

Here x is a full candidate; c specifies its reference/support; u ranges over that
support; p0 is the locally constrained prior defined above; alpha is a nonnegative
sequence-prior exponent; p_alpha is its normalized tempered distribution;
s_hat is the dimensionless predicted stability improvement defined in section 5;
beta is guidance strength; exp is the exponential function; R_alpha is the positive
unnormalized target weight. Normalize R_alpha over the support to obtain the exact
target. Test alpha=1, 0.5, 0.25; alpha=0 is a uniform-candidate diagnostic only.

Alpha=1 preserves the existing generation contract. These ablations do not modify
the trained generation policy. Adoption of another prior in guidance requires
consistent baseline sampling, reward and policy probabilities. A tokenwise
logit temperature is not an equivalent implementation. Stronger scorers, fresh
confirmation training and larger mutation neighborhoods remain subsequent stages,
conditioned on the diagnostic evidence. The current command stops for review.

### 15 October 2026 — Development result and proposed confirmation primary

The [diagnostic results](docs/experiments/target_diagnostic_results_2026-10-15.md)
show positive exact-target gains on all three development supports at alpha=1.
The recommended confirmation primary is single-MLP seed zero, original prior
alpha=1 and beta=1, retaining inner checkpoint selection. Ensemble gains are nearly
identical on average, while tempering trades raw stability for coverage. This is a
development-based recommendation, not a confirmed optimum. The user authorized this procedure; the confirmation runner and configuration are
now implemented as described below. Preserve all five reserved
parents, matched budgets and duplicate accounting when implementing it.

### 16 October 2026 — Confirmation protocol ready

The [fixed protocol](docs/experiments/confirmation_plan_2026-10-16.md) and
[configuration](configs/confirmation.json) implement all five reserved parents.
Preserve existing fit/inner/test identities, one MLP seed zero, alpha=1/beta=1,
20,000 guidance updates with the established late-rate schedule, and guidance
seeds 0–2. No test-adaptive checkpoint or scorer selection is allowed.

Candidate identities are reconstructed separately using original source label
availability and the frozen split. Fit/inner labels and predicted candidate rewards
are the only outcomes used during learning. All parents must finish fitting or
have an explicitly reported scorer-screen failure before a hashed fit seal permits
test-label evaluation. Failed screens skip guidance and remain in the denominator
and report; any partial numerical training failure blocks outcome release.

Compare raw ProGen, GFN, supervised guidance and a privileged exact-target sampler;
compare score-selected batches with generate-and-select using 256 proposals and
32 slots for every method. Retain duplicates, all seeds, exact expectation/TV/KL,
per-parent effects and setup/training/evaluation cost. There is no automatic
biological success threshold. A positive development result or a passing CPU test
is not confirmation evidence. The agent has not run the real experiment.

### 17 October 2026 — Development diversity and prior-construction ablation

The [completed confirmation](docs/experiments/confirmation_results_2026-10-17.md)
establishes accurate target fitting and positive measured raw gains on all five
supports, with substantial duplication and unresolved advantage over cheaper
baselines. The [new protocol](docs/experiments/diversity_plan_2026-10-17.md) tests one
explicit prior change on the three original **development** parents only. Existing
experiments retain the original locally normalized prior and reward unchanged.

For the new `count_corrected` arm, define the conditional baseline as

\[
p_C(a\mid h,c)=
\frac{\exp(\ell_0(a\mid h,c))\,N(ha,c)}
{\sum_{b\in\mathcal A(h,c)}\exp(\ell_0(b\mid h,c))\,N(hb,c)},
\qquad
R_C(x\mid c)=p_C(x\mid c)\exp(\beta\hat{s}(x,c)).
\]

| Symbol | Meaning |
|---|---|
| \(C\) | Identifier for the descendant-count correction arm. |
| \(h\) | Current generated prefix, including the empty prefix. |
| \(a\), \(b\) | Candidate next residue; \(b\) is the summation variable. |
| \(ha\), \(hb\) | Prefix with that residue appended. |
| \(c\) | Reference sequence, mutation budget and declared candidate support. |
| \(\mathcal A(h,c)\) | Actions with at least one supported completion after prefix \(h\). |
| \(\ell_0(a\mid h,c)\) | Frozen ProGen next-residue logit; zero denotes the pretrained model. |
| \(N(ha,c)\) | Number of distinct supported complete sequences descending from prefix \(ha\); depends only on identities, never labels. |
| \(p_C(a\mid h,c)\) | Corrected, locally normalized next-residue probability. |
| \(x\) | Complete candidate sequence. |
| \(p_C(x\mid c)\) | Product of the corrected next-residue probabilities along \(x\). |
| \(R_C(x\mid c)\) | Unnormalized corrected-prior stability reward. |
| \(\hat{s}(x,c)\) | Frozen predicted stability improvement, divided by the fitting-label scale, as defined above. |
| \(\beta\) | Guidance strength, fixed to one in this ablation. |
| \(\exp\), \(\sum\), \(\mid\) | Exponential, sum, and “given,” respectively. |

Normalize the reward over candidates to define its target. With equal logits,
child-to-parent count ratios telescope, giving each complete candidate equal
probability. With actual ProGen logits, that uniformity is not guaranteed. This is
our label-free construction control and proposed prior, not an established optimal
protein prior, original ProGen conditioned on support, or a scalability result.
Counts require the finite support. The `original` arm remains the explicit control.

Use the same frozen MLP-0 predictions, beta=1, heads, schedules, seed set and
single-mutant support in both arms. Add log counts to baseline token logits before
residual guidance; use those same corrected probabilities for sampling, TB reward,
supervised rollouts and evaluation. A zero residual recovers that arm's baseline.
No new scorer or real experiment is fitted by the agent.

Keep the old repeated-slot metrics. Add a separate practical selection protocol:
256 proposals, deduplicate the entire proposal batch, then select up to 32 distinct
sequences by predicted score, with deterministic first-occurrence tie handling.
Never refill. Report count/fill rate alongside quality and diversity. Compare
matched-prior selection, GFN, supervised guidance, exact target and uniform
candidate selection. Record total costs; equal update counts do not match compute.

The 100 repeated categorical batches per final policy are finite-law sampling
references, not independent proteins or online decoding measurements. Report actual
trajectory timings separately. Known development labels are used for evaluation,
not as guidance rewards or selection criteria. The five inspected confirmation
parents remain historical evidence. Fresh evaluation identities and outcome-access
rules must be frozen before a future confirmation or larger-support claim.
