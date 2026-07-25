# umriss CLI specification

## Contract

Every command prints one JSON envelope with `command`, `status`, `data`,
`warnings`, `errors`, and `next_steps`. Errors are nonzero exits. Model calls
are executed explicitly with `ep run`, not inside `umriss`.

## Battery metadata

Each item must have an identifier, question stem, item text, option labels, and
option codes. Scale semantics are explicit:

```yaml
scale:
  type: ordinal
  direction: high_to_low
```

Nominal scales omit `direction`. Option order is preserved for output vectors,
but is never treated as substantive scale meaning without metadata.

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
  --tag TAG [--n-support N] [--seed N] --out DIR
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
  [--max-tokens N] [--limit N]

ep run --jobs FILE.jobs.ep --output FILE.results.ep

umriss support register-results --results FILE.results.ep
  --prompts FILE.jsonl --tag TAG --out DIR
```

Direct-prediction baselines use the same explicit EP boundary:

```text
umriss baseline build --metadata FILE
  [--respondents FILE] [--mode (one_shot | conditioned_direct | both)]
  --tag TAG --out DIR

umriss baseline export --prompts FILE --path FILE.jobs.ep [--model MODEL]
umriss baseline run --jobs FILE.jobs.ep --output FILE.results.ep
umriss baseline register-results --results FILE.results.ep
  --prompts FILE.jsonl --tag TAG --out DIR
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
  --weights FILE [--holdout ITEM] [--minimum-weight FLOAT]
  --path FILE.agents.ep
```

Parsing never repairs invalid responses silently. Fitting and design selection
remain separate: held-out performance may compare declared designs, but any
search over designs must be reported.

The EDSL export uses each profile summary as the complete agent instruction.
Its normalized coefficient is stored as hidden `_weight` metadata and repeated
in a CSV sidecar. EDSL serializes hidden traits but does not interpret
`_weight` as a sampling rule.

## Methodological guardrails

- Target marginals are not inserted into support prompts.
- Complete coverage is never silently truncated.
- Custom prompt changes are preserved in the resolved design.
- Generated profiles are synthetic support points, not recovered respondents.
- Support designs and seeds are declared before held-out evaluation.
