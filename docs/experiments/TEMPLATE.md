# Record title

Date: YYYY-MM-DD, America/New_York. Type: protocol / implementation / run / analysis / result.
Status: proposed / prepared / running / completed / failed / interrupted / superseded.
Operator: user or agent. Execution: real GPU / real CPU / synthetic CPU smoke / read-only analysis / documentation only.

## Question and motivating evidence

What are we testing or changing, and why now? Link the relevant primary literature,
release/upstream source and local results next to the claims they support. State
reading limits when relevant. Label an engineering default, hypothesis or original
derivation explicitly; a citation is not proof of our outcome.

## Decisions and fixed setup

| Choice or change | Exact value / configuration link | Reason and supporting source | Tradeoff / alternative |
|---|---|---|---|
| Fill in | Fill in | Fill in | Fill in |

Record relevant data/split identities, feature/reward scaling, model/optimizer
settings, seeds, budgets, stopping/checkpoint selection, comparison fairness,
outcome access and failure handling. Link inherited settings and list differences.
For documentation-only work, state the affected documents and why no scientific
settings change rather than filling irrelevant model fields.

## Execution and provenance

- Exact command(s) or reproducible read-only procedure; explicitly mark planned versus executed.
- Date/operator, environment/device or smoke status; user-selected GPU when known.
- Configuration, source/data/checkpoint hashes or links to machine-generated provenance.
- Input and output/log paths, observed status, elapsed time/cost if available.
- Unexpected warnings/errors, retries and deviations. Unknown details stay unknown.

## Verification and findings

What checks actually ran, and what did they establish? Record failed checks and
fixes when material. For experiments, compare the declared baselines and distinguish
exact expectations, sampled/selected results, predicted scores and measured labels.
Retain all parents/seeds, failures and duplication/coverage information. Link raw
outputs and a numerical audit rather than duplicating them here.

## Interpretation, limits and next decision

What is supported, what remains uncertain, and what do we do next? Identify post-hoc
analysis and any reused development/test outcomes. State any superseded interpretation
with a dated link. Do not turn an implementation check into a scientific claim.

## Documentation updates

Link this record from the experiment index. Update current status after a verified
transition, decisions for changed reasoning, implementation log for actions/checks,
and literature records when new sources were inspected. Update the scientific spec
only if the scientific contract changed. Preserve active-run artifacts and paths.
