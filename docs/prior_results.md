**Historical evidence from the test workspace.** These results precede v1 stability guidance. The active project is defined in [spec.md](../spec.md). No section below is a new implementation milestone.

# Prior standalone results

### ProGen2

The available backbone is **ProGen2-medium**, with 764,803,616 parameters, from the official upstream checkout at commit `c27a419c234a0997923761e1fe7daffcebf0eaf5`. Its checkpoint is in:

`/usr/scratch/agoel320/progen-gflownets-test/progen/checkpoints/progen2-medium`

The recorded environment uses Python 3.10, PyTorch 2.6.0+cu124, Transformers 4.27.4, and tokenizers 0.13.3. The model's final hidden representation has width 1,536; the LM head produces logits over 32 vocabulary tokens. Tokenizer IDs, canonical amino acids, sequence boundaries, and special EOS tokens must be distinguished explicitly.

Generation and decoding have run successfully, but sequence quality remains an issue:

| Saved experiment | Completed samples | Hit length cap | Low-entropy-window flag | Empty sequences |
|---|---:|---:|---:|---:|
| Original baseline, temperature 0.8 / top-p 0.9 | 200 | 185 | 127 | 0 |
| Small original-settings comparison | 20 | 18 | 10 | 0 |
| Small temperature 1.0 / top-p 1.0 comparison | 20 | 14 | 2 | 3 |

These are descriptive results from the saved runs, not estimates of biological function. The two 20-sample runs share the starting seed and batch size but their trajectories diverge when decoding changes. They change both temperature and top-p, so they do not isolate the separate effect of each setting. Unfiltered sampling reduced the recorded repetition flags in this small comparison, but did not eliminate capped sequences and introduced empty outputs.

CPU and user-run GPU checks of selected prefixes found agreement between cached and full-prefix predictions within the recorded numerical tolerances. The inspected repetitive prefixes strongly favored continuation of repeats even before filtering. These checks narrow possible explanations; they do not establish that every decoding path is correct or that the generated sequences are useful proteins.

The saved investigation narrative predates the small sampling comparison and still describes it as future work. The completed run artifacts take precedence for current status.

Evidence:

- [Completed baseline summary](../../progen-gflownets-test/progen/runs/baseline_200_seed42/summary.json)
- [Original-settings comparison](../../progen-gflownets-test/progen/runs/compare_original/summary.json)
- [Unfiltered comparison](../../progen-gflownets-test/progen/runs/compare_unfiltered/summary.json)
- [Decoding investigation](../../progen-gflownets-test/progen/investigations/findings.md)
- [Model and interface notes](../../progen-gflownets-test/progen/docs/model_notes.md)

### Standalone GFlowNet

The TF-Bind-8 pipeline has completed **three full GPU seeds, 0–2**, with training, candidate generation, and evaluation. It uses the supplied [BioSeq-GFN-AL repository](https://github.com/MJ10/BioSeq-GFN-AL) with documented adaptations and corrections, rather than executing the original runner unchanged.

Each seed generated 640 candidate slots and selected 128 using the proxy acquisition score. The full benchmark is an exact lookup over DNA 8-mers; the starting dataset contains 32,898 low-scoring rows.

| Method | Mean oracle score ± sample SD | Diversity | Novelty |
|---|---:|---:|---:|
| GFlowNet + proxy selection | 0.690 ± 0.013 | 4.836 | 1.096 |
| Random proposals + the same proxy selection | 0.653 ± 0.015 | 4.935 | 1.042 |
| Unranked random proposals | 0.471 ± 0.021 | 5.385 | 0.542 |

Diversity and novelty use Levenshtein distance. About 99.2% of GFlowNet-selected slots are unique within their batch, and 96.4% are outside the initial dataset. The GFlowNet batch mean exceeds both baselines in every completed seed.

This establishes a working standalone pipeline and useful baseline improvement. It **does not establish full numerical reproduction** of the paper's reported mean score 0.84 and novelty 2.12. Differences in ensemble training, stopping, reward parameterization, sampling, and evaluation need controlled investigation.

An after-the-fact diagnostic found that choosing the best 128 by true score from our existing 640-candidate pools would average approximately 0.775. That is an optimistic diagnostic of selection within those particular pools, not a valid benchmark result, a global optimum, or a ceiling on another generator. No original result was replaced by this diagnostic.

The prior proposals to audit sampling against the exact proxy target and compare reward exponents 3 and 8 have **not been implemented or run**. They remain useful standalone follow-ups. Matching 0.84 is not logically required to begin a tiny integration correctness experiment, but the incomplete reproduction must remain explicit.

Evidence: [run report](../../progen-gflownets-test/gflownet/runs/tfbind8/report.md), [completion record](../../progen-gflownets-test/gflownet/runs/tfbind8/completion.json), and [implementation differences](../../progen-gflownets-test/gflownet/README.md).

TF-Bind-8 is a DNA benchmark. Its trained generator cannot simply be attached to ProGen as a protein guidance head.

