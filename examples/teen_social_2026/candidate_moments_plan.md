# Candidate synthetic moments for richer teen social-media twins

This capability probe asks whether frontier-model consensus can provide
explicitly synthetic calibration targets for questions that were not observed
in the source survey.

The proposed battery adds seven candidate questions to the observed daily-time
item:

1. posting frequency;
2. late-night use;
3. sleep interference;
4. negative social comparison;
5. effect on connection to friends;
6. strictness of parental rules; and
7. primary platform.

The first pass should ask several frontier models for each marginal
independently. A candidate should be eligible for calibration only under a
predeclared agreement rule. A useful starting rule is:

- every model returns a valid probability vector;
- the maximum pairwise total-variation distance is at most 0.10;
- no option differs by more than 0.10 across models; and
- the arithmetic-mean model distribution is retained without silently repairing or
  renormalizing any model response.

Model predictions and their dispersion must remain labeled as synthetic
priors, never as observed survey estimates.

## Candidate cross-tabs

The most substantively useful two-question targets are:

- daily time × sleep interference;
- late-night use × sleep interference;
- daily time × strictness of parental rules;
- posting frequency × negative social comparison; and
- primary platform × daily time.

Each cross-tab must be elicited as one joint probability table. Multiplying
separately elicited marginals would impose independence and would not add a
new joint moment.

Before accepting a joint target:

- all cells must be nonnegative and the table must sum to one;
- row and column sums must be compared with the separately elicited
  marginals;
- model disagreement must be reported at both the cell and induced-marginal
  levels; and
- inconsistent targets must be rejected or reconciled explicitly, never
  silently projected onto a coherent table.

## Current umriss capability gaps exposed by the probe

1. Baseline jobs can elicit per-item marginal vectors from multiple models,
   but parsed outputs do not retain a first-class model identifier suitable
   for consensus analysis.
2. There is no consensus command, agreement diagnostic, acceptance policy,
   or aggregate artifact with uncertainty and provenance.
3. Battery truth stores exactly one vector per item. It cannot distinguish
   observed targets from model-synthetic targets or attach source/model
   provenance and confidence.
4. Calibration concatenates independent item vectors. There is no schema for
   joint cells and no optimizer path for pairwise cross-tab moments.
5. There is no cross-tab prompt builder, strict joint-table parser, marginal
   consistency audit, or reconciliation policy.
6. An existing support bank cannot be enriched with probabilities for newly
   added questions. The expanded battery must currently be regenerated and
   remeasured, or matrices must be assembled outside umriss.
7. Checkbox questions are not representable as multiselect outcomes in the
   battery schema. They must be decomposed into binary indicators or modeled
   as a large joint response space.
8. Fit diagnostics do not distinguish observed and synthetic constraints,
   weight moments by confidence, or show tension attributable to each target
   source.
9. `umriss next` has no workflow states for model-consensus targets,
   cross-tab elicitation, bank enrichment, or fit completion.
10. Targets have no population-slice dimension. The current teen example
    duplicates the battery metadata for college-parent and non-college-parent
    marginals. A richer target store should identify population, subgroup
    predicate, and estimand without duplicating the question battery.
11. A population cross-tab cannot generally be reconstructed from two
    persona-level marginal vectors. Using the outer product of each persona's
    vectors assumes conditional independence within persona. Umriss needs
    either directly measured persona-level joint probabilities or an explicit,
    versioned independence assumption before it can construct the cross-tab
    columns of the calibration matrix.

## Minimal useful implementation

The smallest coherent extension would add:

- a versioned `targets` artifact supporting marginal and joint targets,
  population slice, provenance, target kind (`observed` or
  `model_synthetic`), model panel, dispersion, acceptance rule, and status;
- `umriss prior build/export/register-results/parse/consensus`;
- `umriss prior build-cross-tabs` and strict joint-table parsing;
- `umriss targets audit` for probability validity, agreement, duplicate
  implications, and row/column consistency;
- `umriss support extend-items` to measure new questions on an existing,
  stable persona roster; and
- a generalized linear moment design consumed by `umriss fit`, with
  constraint-level residual diagnostics and optional declared confidence
  weights. Joint constraints must state whether their persona-level features
  were elicited directly or derived under conditional independence.

## Probe result

The panel used GPT-5.5 through OpenAI, Claude Opus 5 through Deep Infra, and
Gemini 3.1 Pro. The predeclared agreement threshold was a maximum pairwise
total-variation distance of 0.10 and a maximum option-level difference of
0.10. Three of eight candidate marginals passed:

| Candidate | Max pairwise TV | Max option difference | Decision |
|---|---:|---:|---|
| Posting frequency | 0.07 | 0.06 | Accept |
| Sleep interference | 0.07 | 0.07 | Accept |
| Negative social comparison | 0.06 | 0.05 | Accept |
| Daily social-media time | 0.12 | 0.12 | Reject |
| Late-night use | 0.16 | 0.16 | Reject |
| Connection effect | 0.20 | 0.20 | Reject |
| Parental rules | 0.12 | 0.10 | Reject |
| Primary platform | 0.16 | 0.12 | Reject |

The accepted panel-mean vectors are:

- posting frequency: `[0.203333, 0.26, 0.32, 0.216667]`;
- sleep interference: `[0.09, 0.21, 0.41, 0.29]`; and
- negative social comparison: `[0.15, 0.29, 0.396667, 0.163333]`.

These remain synthetic priors. They have not been copied into battery
`truth`, because doing that would erase their model provenance and make them
indistinguishable from observed marginals.

The observed daily-time item is a useful negative control. The model-panel
mean was `[0.11, 0.393333, 0.496667]`, whereas the published college-parent target
was `[0.40, 0.37, 0.23]` and the non-college-parent target was
`[0.35, 0.29, 0.35]`. Even frontier-model agreement would not establish
empirical validity; here the panel also failed the agreement rule.

### Execution and failure audit

- The initial package scheduled 24 calls and cost `$0.340546`.
- GPT-5.5 and Claude returned all eight vectors.
- The Google Gemini backend returned eight empty rows, but `ep run` reported
  24 results and `umriss baseline parse` returned `ok` with 16 predictions.
- A retry-only Google package again returned eight empty rows and cost `$0`.
  Registration correctly reported `incomplete_results` on this isolated
  retry, and `umriss support audit-results` preserved an eight-job missing-ID
  audit.
- A retry-only package for the same Gemini model through Deep Infra returned
  all eight vectors and cost `$0.4789785`.
- Total inference cost was `$0.8195245`.

The parser's successful status on the incomplete multi-model attempt is a
release-blocking issue for any consensus workflow: completeness must be
checked over `(job_id, model, service)`, and parsed rows must retain those
fields.
