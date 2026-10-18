# Biological pilot: interim low-loss check — 15 October 2026

**Subsequent result:** the biological pilot is complete. See the
[final result record](biological_first_results_2026-10-15.md); it fits the target
accurately but does not improve measured stability over ProGen. Planning/interim
sections below preserve the state before final evaluation.

The user asked about small and fluctuating seed-0 TB losses while the biological
pilot was running. This is an interim saved-input analysis, not a final result.
The agent read the saved scorer report, early progress and CPU task cache only;
no training, backbone execution, GPU use or run modification occurred. Held-out
measured stability labels were not inspected.

The first recorded sampled loss was 0.094078, falling to 0.000064 at update 100.
The loss is a sampled trajectory-balance residual squared, not a stability-prediction
error or a measured biological score. Different batches and changing parameters
produce fluctuations; very small values make ordinary spikes look large in ratio.
The displayed values alone do not establish either a bug or full-distribution fit.

The cached biological task has 204 candidates and 7,061 prefixes. At beta one:

| Quantity | Saved-input value |
|---|---:|
| Scaled predicted-score minimum / maximum | −3.360712 / 0.639853 |
| Unweighted score standard deviation | 0.893251 |
| Baseline highest sequence probability | 0.385167 |
| Baseline mass of its five highest-probability sequences | 0.914152 |
| Target highest sequence probability | 0.344118 |
| Target mass of its five highest-probability sequences | 0.898712 |
| Baseline TV from the target | 0.091711 |
| Initial expected TB loss | 0.091524 |

The initial expectation is computed directly as E_p0[score²], since beta=1,
zero guidance gives q=p0, and log Z starts at zero. Its agreement with the first
sampled loss is consistent with correct initial loss scale. The scores are not
constant, but most sampling mass is concentrated in a handful of candidates and
the baseline distribution is already fairly close to the tilted target. This
helps explain why sampled TB loss can become small quickly. It also means rare
candidates may be poorly checked by an individual batch. Concentration already
exists in p0 and the target; it is not by itself evidence of learned mode collapse.

The scorer passed its validation gate: selected MLP validation MAE 0.317013 versus
0.853492 for the training-mean baseline, and Spearman 0.864726. These are validation
metrics used for model selection, not independent held-out design success.

Decision: allow the user-run job to finish with its declared settings. Do not
change active training from these loss values. At completion inspect whole-support
TV/KL, raw probability concentration/diversity and held-out measured stability.
Low TB loss alone is not sufficient evidence of a useful or diverse biological
result. All live run files and source remain unchanged by this check.
