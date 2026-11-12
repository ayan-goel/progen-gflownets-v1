> Historical stability specification preserved at the 12 November 2026 pivot.
> Its then-current next steps are superseded by the [active specification](../spec.md).
> Completed-run definitions remain unchanged. The previously deleted external archive remains deleted.

# Stability-guided protein generation with frozen ProGen2

**Scope update: 10 November 2026.** The local mechanism studies are complete;
the user authorized a grouped multi-protein stability-predictor benchmark.
[Current status](../docs/status.md) owns execution progress.

## 1. Project objective and immediate scope

Learn an additional guidance head on frozen ProGen2 using whole-sequence
folding-stability feedback and a GFlowNet objective. The long-term motivation is
useful, diverse stable proteins; the immediate task is to assess stability rewards
across protein groups before expanding generation.

Focus on one or two controlled questions: how well a linear versus neural
predictor generalizes, and whether the chosen reward produces a useful target.
Keep the original trajectory-balance implementation as the baseline.
The [multi-protein protocol](../docs/experiments/multiprotein_2026-11-10.md) now defines
a bounded 64-group predictor comparison with frozen ProGen features. It performs
no GFN fitting or generation. Broader generation, folding pipelines and restored
retired scorer-scaling workflows remain outside the active implementation.

## 2. Retained foundation

- Standalone ProGen2 and the adapted TF-Bind-8 GFlowNet run independently.
  [Prior results](../docs/prior_results.md) document the reproduction gap.
- The 256-sequence synthetic integration test passed across three final seeds.
- The original biological pilot on 1O6X.pdb used 950 training, 204 validation
  and 204 evaluation sequences. GFlowNet learned the predicted target, but
  measured expected stability was 4.26975 versus 4.28005 for constrained ProGen.
  [Pilot results](../docs/experiments/biological_first_results_2026-10-15.md).
- Keep original numerical definitions and artifacts unchanged while investigating
  their consequences. The completed predictor/reward study identifies full-data ridge as a provisional baseline;
  [results and limitations](../docs/experiments/predictor_reward_study_2026-11-03.md)
  motivate the [controlled ridge-reward GFN rerun](../docs/experiments/ridge_gfn_2026-11-09.md).
  The completed rerun realizes the ridge target's modest local gain; controlled
  reward-noise sensitivity is complete: accurate fitting persists, but some corrupted
  targets lose biological utility. [Results and next decision](../docs/experiments/reward_noise_2026-11-09.md)
  motivated the completed [three-split reward-generalization study](../docs/experiments/split_robustness_2026-11-09.md).
  Both predictor families improve target means there; ridge is not consistently
  better. The completed [position-held-out study](../docs/experiments/position_holdout_2026-11-09.md)
  finds useful targets for both families, with larger ridge gains on its three
  different supports. The completed GFN follow-up matches these targets across all nine fits and
  realizes their gains; the evidence still concerns one parent and known candidates.
- Existing evaluation outcomes are inspected development evidence for future
  choices, not a fresh test set.

The following sections retain the original scientific and implementation contract.
Proposed changes require an explicit new protocol rather than retroactive edits
to the completed experiment.

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

## 9. Current next steps

The local predictor/reward investigations and GFN follow-up are complete; see
[current status](../docs/status.md). All nine position-held-out-support GFNs closely
match their fixed targets and realize their measured stability gains. This
establishes local sampling, not broad generation of new stable proteins.

1. Run the [fixed multi-protein benchmark](../docs/experiments/multiprotein_2026-11-10.md).
   Data/exposure review and grouping are complete: 44 training, 10 validation and
   10 evaluation parent groups, 65 measured sequences each. Grouping, selection,
   budgets and metrics are fixed before user-run fitting.
2. Retain simple ridge/MLP baselines without assuming the local winner generalizes.
   Assess prediction quality and useful stability selection separately.
3. Before generating beyond known candidates, define the sequence space, reference
   probability convention and independent stability assessment. Do not silently
   extend the locally masked candidate-tree prior to unrestricted generation.
4. Keep ProGen frozen and TB unchanged unless a broader task demonstrates a fitting
   limitation. Partial unfreezing and SubTB remain separate proposed experiments.

Previously inspected outcomes, including retired results, remain development
evidence. A new evaluation protocol requires an exposure audit rather than calling
reused data independent. The multi-protein benchmark is implemented and ready for user execution. Expanded
generation is not implemented; original scientific definitions and completed runs
stay fixed.

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

The [README](../README.md), [status](../docs/status.md), [literature plan](../literature/EXPERIMENT_PLAN.md)
and this specification describe the same local stability investigation.
The user runs all real experiments and every GPU operation. Development may use
small synthetic CPU runs and saved-artifact analysis.

Keep ProGen2 frozen. Exclude antibody-specific optimization, Potts training,
full-backbone fine-tuning, unconstrained generation, folding assessment pipelines,
SMC and speculative decoding from this active implementation.

The later work is recoverable outside this project; it is not an active dependency
or evidence required to execute this task. See the
[scope decision](../docs/decisions.md#2026-11-03--return-to-the-first-biological-pilot).
