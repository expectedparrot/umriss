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
  --preset (pattern-coverage | uniform-patterns) [--size N] [--seed N] --out FILE

umriss design validate (--metadata FILE | --battery ID) --design FILE
```

Only `schema_version: 1` is accepted. The schema supports:

| Component | Purpose |
| --- | --- |
| `option_coverage` | Require declared item-option regions |
| `pattern_anchors` | Supply complete response-pattern scaffolds |
| `profiles` | Supply expert-authored substantive profiles |
| `uniform_patterns` | Replicate the full response-pattern grid with unique prompt identities |

`coherence` is one of `global`, `item_specific`, `grouped`, or `explicit`.
Complete coverage is checked before prompts are generated. If the declared
components need 14 rows and `size` is 12, validation returns
`DESIGN_TOO_SMALL`. Partial coverage requires `coverage.mode: partial`.

## Support generation

```text
umriss support build (--metadata FILE | --battery ID)
  (--preset pattern-coverage | --preset uniform-patterns | --design FILE)
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

## Parsing and estimation

```text
umriss support parse --raw FILE --metadata FILE --tag TAG --out DIR

umriss support uniformity --support FILE --metadata FILE
  [--tolerance FLOAT] [--max-duplicate-fraction FLOAT]
  [--min-joint-pattern-fraction FLOAT] --out FILE

umriss support augment-uniform --support FILE --metadata FILE
  --tag TAG [--n-add N] [--tolerance FLOAT] [--seed N] --out DIR

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

Support generation returns a `persona` written in the second person. EDSL
export stores that text in the visible trait selected by `--persona-trait`;
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
| `umriss support build` |  |
| `umriss support export` |  |
| `umriss support inspect` |  |
| `umriss support merge` |  |
| `umriss support parse` |  |
| `umriss support register-results` |  |
| `umriss support uniformity` |  |
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
