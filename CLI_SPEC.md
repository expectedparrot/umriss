# umriss CLI specification

## Contract

Every command prints one JSON envelope with `schema_version`, `command`,
`status`, `argv`, `data`, `warnings`, `errors`, and `next_steps`. Errors are
nonzero exits. Model calls are executed explicitly with `ep run`, not inside
`umriss`. `umriss capabilities` describes this contract machine-readably.

Expected parser and validation failures use exit status 1. Unexpected internal
failures use exit status 2 and remain JSON-enveloped. The `command` field names
the canonical command that failed.

## Battery metadata

Battery metadata must declare `schema_version: 1`.

Each item must have an identifier, question stem, item text, option labels, and
option codes. Scale semantics are explicit:

```yaml
scale:
  type: ordinal
  direction: high_to_low
```

Nominal scales omit `direction`. Option order is preserved for output vectors,
but is never treated as substantive scale meaning without metadata.

## Workspace id resolution

Wherever a command accepts `--metadata FILE` or `--design FILE`, it also
accepts a workspace id: a bare token (no path separator) resolves against the
active `.umriss` project's imported batteries or designs. An existing file
path always wins, so explicit paths behave exactly as before. A bare token
matching nothing fails closed with the known ids. Commands may omit the flag
entirely once a default is set with `umriss battery use <id>` or
`umriss design use <id>`: the active default is applied, echoed in the
envelope under `data.resolved_defaults`, and shown by `umriss status`.
Explicit flags always override the default; with neither, commands fail
closed with `missing_battery` / `missing_design`.

Pipeline commands are store-backed the same way, keyed by `--tag`: when
`--out` (and stage inputs such as `--prompts`, `--raw`, `--support`,
`--derived`) are omitted, artifacts are written to and read from the tag's
run directory in the active project (`.umriss/projects/<p>/runs/<tag>/`)
using the conventional `<tag>_*` filenames. A missing stage input fails
closed naming the command that produces it. Everything implicitly resolved is
echoed under `data.resolved_defaults.run`; `umriss status` lists store runs
with their stage, and `umriss export --tag <tag> --out <dir>` copies a run's
artifacts out for a replication package. Import artifacts with
`umriss battery import --metadata <file>` and
`umriss design import --design <file>`; list them with `umriss battery list`
and `umriss design list`.

## Designs

```text
umriss design create (--metadata FILE | --battery ID)
  --preset (pattern-coverage | uniform-patterns | balanced-blueprints)
  [--size N] [--seed N] --out FILE

umriss design validate (--metadata FILE | --battery ID) --design FILE
```

Only `schema_version: 1` is accepted. The schema supports:

| Component | Purpose |
| --- | --- |
| `option_coverage` | Require declared item-option regions |
| `pattern_anchors` | Supply complete response-pattern scaffolds |
| `profiles` | Supply expert-authored substantive profiles |
| `uniform_patterns` | Replicate the full response-pattern grid with unique prompt identities |
| `balanced_blueprints` | Construct unique complete response vectors with item marginals balanced to within one row |

`coherence` is one of `global`, `item_specific`, `grouped`, or `explicit`.
Complete coverage is checked before prompts are generated. If the declared
components need 14 rows and `size` is 12, validation returns
`DESIGN_TOO_SMALL`. Partial coverage requires `coverage.mode: partial`.

## Support generation

```text
umriss support build (--metadata FILE | --battery ID)
  (--preset pattern-coverage | --preset uniform-patterns |
   --preset balanced-blueprints | --design FILE)
  --tag TAG [--n-support N] [--seed N] --out DIR [--force]
```

`--n-support` and `--seed` are explicit overrides; all feasibility checks still
apply. A successful build writes:

| Artifact | Meaning |
| --- | --- |
| `<tag>_resolved_design.yaml` | Exact design after defaults and overrides |
| `<tag>_support_plan.csv` | One row per support point and its reason |
| `<tag>_coverage.csv` | Every requested item-option cell and coverage count |
| `<tag>_prompts.jsonl` | Executable prompt rows |
| `<tag>_prompts.html` | Human-readable prompt review |

Custom templates are declared with:

```yaml
prompt:
  template: custom_prompt.jinja2
  validation: strict
```

They must mention every item, probability output, the sum-to-one constraint,
and the configured summary field.

## EP boundary

```text
umriss support export --prompts FILE --path FILE.jobs.ep
  [--model MODEL] [--service-name NAME] [--temperature FLOAT]
  [--max-tokens N] [--limit N] [--tag TAG]
  [--registration-out DIR] [--job-ids MISSING.csv] [--force]

ep run --jobs FILE.jobs.ep --output FILE.results.ep

umriss support register-results --results FILE.results.ep
  --prompts FILE.jsonl --tag TAG --out DIR [--force]
```

Export writes `<tag>_manifest.json` with input hashes, package version,
parameters, model-call count, output hash, and directly executable `ep run`
and registration commands. An identical rerun reuses the verified artifacts.
Reusing an output for different inputs fails with `output_conflict` unless
`--force` is explicit.

Registration refuses Results that lack a valid answer for any prompt job. Audit
and merge separately preserved attempts with:

```text
umriss support audit-results --results ATTEMPT1.results.ep
  [--results ATTEMPT2.results.ep ...] --prompts FILE.jsonl
  --tag TAG --out DIR [--force]
```

The audit writes merged raw rows with source-attempt attribution, per-attempt
coverage, and a retry-only `job_id` CSV. Pass that CSV to
`support export --job-ids` to create retry Jobs.

Direct-prediction baselines use the same explicit EP boundary:

```text
umriss baseline build --metadata FILE
  [--respondents FILE] [--mode (one_shot | conditioned_direct | both)]
  --tag TAG --out DIR

umriss baseline export --prompts FILE --path FILE.jobs.ep [--model MODEL]
  [--tag TAG] [--registration-out DIR] [--job-ids MISSING.csv] [--force]
ep run --jobs FILE.jobs.ep --output FILE.results.ep
umriss baseline register-results --results FILE.results.ep
  --prompts FILE.jsonl --tag TAG --out DIR [--force]
umriss baseline parse --raw FILE --prompts FILE.jsonl
  --metadata FILE --tag TAG --out DIR
```

Conditioned-direct prompts contain every held-in real marginal and exclude the
held-out marginal by construction.

### Synthetic priors and generalized targets

Model-panel priors use a model-aware workflow. Completeness is checked over
`(job_id, model, service_name)`, not merely the prompt job ID:

```text
umriss prior build-marginals --metadata FILE --tag TAG --out DIR
umriss prior build-joints --metadata FILE --pair ITEM_A:ITEM_B
  [--pair ITEM_C:ITEM_D ...] --population ID --tag TAG --out DIR
umriss prior export --prompts FILE --path FILE.jobs.ep
  [--model SERVICE:MODEL ...] [--tag TAG] [--registration-out DIR]
umriss prior register-results --results FILE.results.ep
  --prompts FILE.jsonl --tag TAG --out DIR
umriss prior parse --raw FILE [--raw RETRY.csv ...]
  --prompts FILE.jsonl --metadata FILE --tag TAG --out DIR
umriss prior consensus --predictions FILE [--predictions RETRY.csv ...]
  --metadata FILE --population ID [--minimum-models N]
  [--max-total-variation FLOAT] [--max-option-difference FLOAT]
  [--confidence-weight FLOAT] --tag TAG --out DIR
```

Consensus writes a schema-v1 `umriss_targets` artifact containing accepted and
rejected targets, population identity, model/service provenance, dispersion,
the declared acceptance rule, simplex-preserving aggregation method, and
confidence weight.
Model-synthetic targets are never relabeled as observed truth.

Observed targets can be converted and combined without losing their source:

```text
umriss targets from-metadata --metadata FILE --population ID
  [--confidence-weight FLOAT] --out TARGETS.json
umriss targets merge --targets OBSERVED.json --targets SYNTHETIC.json
  --out MERGED.json
umriss targets audit --targets TARGETS.json --metadata FILE
  [--consistency-tolerance FLOAT] [--out AUDIT.csv]
umriss targets feasibility --targets TARGETS.json --support PROBABILITIES.csv
  --metadata FILE [--tolerance FLOAT] --tag TAG --out DIR
```

The audit validates marginal vectors, joint-table shape and probability mass,
source metadata, population identity, and agreement between accepted joint
tables and accepted marginal targets.

Existing personas can be measured on new items without regenerating their
identities:

```text
umriss support extend-items --points POINTS.csv --metadata FILE
  [--item ITEM ...] [--joint ITEM_A:ITEM_B ...] --tag TAG --out DIR
umriss support export --prompts TAG_extension_prompts.jsonl
  --path TAG.jobs.ep [--model SERVICE:MODEL ...]
umriss support parse-extension --raw FILE --prompts FILE
  --base-support PROBABILITIES.csv --metadata FILE --tag TAG --out DIR
```

Direct joint elicitation produces a separate joint-feature bank. Generalized
fitting consumes accepted marginal, checkbox-marginal, and joint targets:

```text
umriss targets fit --targets TARGETS.json --support PROBABILITIES.csv
  [--joint-features JOINT.csv] [--allow-conditional-independence]
  [--minimum-effective-support FLOAT] [--maximum-weight FLOAT]
  [--require-convergence]
  --metadata FILE --tag TAG --out DIR
```

Without directly elicited persona-level joint features, a joint target fails
closed. `--allow-conditional-independence` is an explicit authorization to
derive each persona's joint cells as the outer product of its two marginal
vectors. Constraint diagnostics record which method was used and report
source-specific residuals.

## Parsing and estimation

```text
umriss support parse --raw FILE --metadata FILE --tag TAG --out DIR
  [--allow-legacy-persona]

umriss support validate-blueprints --support FILE --plan FILE
  --metadata FILE --tag TAG [--minimum-match-fraction FLOAT]
  [--minimum-intended-probability FLOAT] --out DIR

umriss support uniformity --support FILE --metadata FILE
  [--tolerance FLOAT] [--max-duplicate-fraction FLOAT]
  [--min-joint-pattern-fraction FLOAT] --out FILE

umriss support augment-uniform --support FILE --metadata FILE
  --tag TAG [--n-add N] [--tolerance FLOAT] [--seed N] --out DIR

umriss support augment-targets --support FILE --targets TARGETS.json
  --metadata FILE --tag TAG [--n-add N] [--seed N] --out DIR

umriss support merge --base FILE --additions FILE --tag TAG --out DIR

umriss fit --support FILE --metadata FILE
  [--include-item ITEM] [--exclude-item ITEM] --tag TAG --out DIR

umriss validate marginals (--raw FILE | --support FILE) --metadata FILE
  [--uniform-tolerance FLOAT] [--max-duplicate-fraction FLOAT]
  [--min-joint-pattern-fraction FLOAT] [--allow-nonuniform-support]
  --tag TAG --out DIR

umriss predict --support FILE --weights FILE --metadata FILE
  --item ITEM --out FILE

umriss plot validation --derived DIR --tag TAG --out DIR
  [--format (svg | png | pdf)] [--top-personas N]

umriss twins export-edsl --points FILE [--points FILE ...]
  --weights FILE --persona-trait NAME
  [--holdout ITEM] [--minimum-weight FLOAT]
  --path FILE.agents.ep
```

Parsing never repairs invalid responses silently. Fitting and design selection
remain separate: held-out performance may compare declared designs, but any
search over designs must be reported.

Support generation returns a second-person synthesis plus one explicit
`persona_details` statement per battery item. Parsing requires complete detail
coverage, assembles the statements into the visible `persona`, and records the
summary, detail JSON, count, and coverage in the points artifact. The
`--allow-legacy-persona` escape hatch is only for reparsing results produced
under the older summary-only contract. EDSL export stores the assembled text
in the visible trait selected by `--persona-trait`;
umriss does not set a custom agent instruction. The normalized coefficient is
stored as hidden `_weight` metadata and repeated in a CSV sidecar. EDSL
serializes hidden traits but does not interpret `_weight` as a sampling rule.

## Methodological guardrails

- Target marginals are not inserted into support prompts.
- Complete coverage is never silently truncated.
- Custom prompt changes are preserved in the resolved design.
- Generated profiles are synthetic support points, not recovered respondents.
- Support designs and seeds are declared before held-out evaluation.

## Command reference

Every registered command (options and defaults live in `umriss <command> --help`;
`tests/test_contract_sync.py` keeps this table and the CLI from drifting apart).

| Command | Purpose |
|---|---|
| `umriss baseline build` |  |
| `umriss baseline export` |  |
| `umriss baseline parse` |  |
| `umriss baseline register-results` |  |
| `umriss prior build-joints` | Build population joint-table prior prompts for declared item pairs. |
| `umriss prior build-marginals` | Build independent marginal prior prompts for every battery item. |
| `umriss prior consensus` | Apply a declared multi-model agreement rule and write provenance-bearing targets. |
| `umriss prior export` | Export model-panel prior prompts as an externally executed Jobs package. |
| `umriss prior parse` | Strictly parse model-aware marginal or joint priors. |
| `umriss prior register-results` | Register an externally executed prior Results package. |
| `umriss battery compile` |  |
| `umriss battery create` |  |
| `umriss battery export-edsl` |  |
| `umriss battery import` |  |
| `umriss battery list` | List batteries imported into the active project. |
| `umriss battery use` | Set the active battery default for commands that omit --metadata. |
| `umriss export` | Copy a store run's artifacts to a plain directory for replication packages. |
| `umriss battery inspect` |  |
| `umriss capabilities` |  |
| `umriss compare` |  |
| `umriss design create` |  |
| `umriss design import` | Store a design file in the active project under an id. |
| `umriss design list` | List imported designs. |
| `umriss design use` | Set the active design default for commands that omit --design. |
| `umriss design validate` |  |
| `umriss fit` |  |
| `umriss guide` |  |
| `umriss init` |  |
| `umriss marginal add` |  |
| `umriss marginals import` |  |
| `umriss next` |  |
| `umriss plot validation` |  |
| `umriss predict` |  |
| `umriss project create` |  |
| `umriss project current` |  |
| `umriss project list` |  |
| `umriss project show` |  |
| `umriss project use` |  |
| `umriss question add` |  |
| `umriss report` |  |
| `umriss report-data build` |  |
| `umriss status` |  |
| `umriss support audit-results` |  |
| `umriss support augment-uniform` |  |
| `umriss support augment-targets` | Build complete repair blueprints whose declared addition allocation offsets measured target gaps. |
| `umriss support build` |  |
| `umriss support export` |  |
| `umriss support extend-items` | Build measurement prompts for new marginal and direct-joint features on stable personas. |
| `umriss support inspect` |  |
| `umriss support merge` |  |
| `umriss support parse` |  |
| `umriss support parse-extension` | Strictly merge newly measured item probabilities and write direct joint features. |
| `umriss support register-results` |  |
| `umriss support uniformity` |  |
| `umriss support validate-blueprints` | Compare generated response probabilities with complete intended blueprints and emit accepted support plus retry-only job IDs. |
| `umriss targets audit` | Validate target probabilities, provenance, and joint/marginal consistency. |
| `umriss targets feasibility` | Diagnose whether accepted marginal targets lie in the support bank's convex hull and write a witness fit. |
| `umriss targets fit` | Fit generalized marginal and joint constraints with per-target diagnostics. |
| `umriss targets from-metadata` | Convert observed metadata truth into a provenance-bearing target artifact. |
| `umriss targets merge` | Merge nonoverlapping target artifacts for one population. |
| `umriss twins analyze-logprobs` |  |
| `umriss twins analyze-probabilistic-survey` |  |
| `umriss twins analyze-resolution` |  |
| `umriss twins build-resolution-experiment` |  |
| `umriss twins build-survey-jobs` |  |
| `umriss twins compare-survey` |  |
| `umriss twins embed-probabilities` |  |
| `umriss twins export-edsl` |  |
| `umriss twins plot-survey` |  |
| `umriss validate marginals` |  |
| `umriss version` |  |
