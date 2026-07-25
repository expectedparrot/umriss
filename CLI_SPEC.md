# umriss CLI Spec

`umriss` is a Python CLI for constructing marginal-matching digital twin sets.
The name stands for **Marginally Informed Mixture Calibration**.

The tool turns aggregate survey marginals into fitted mixtures over a reusable
support bank. It is designed to replace paper-specific scripts in this
directory while remaining a general package in the parent `llm-survey-priors`
project.

The implementation should closely follow `zwill`'s design style:

- stdlib `argparse` command surface
- explicit command groups and subcommands
- JSON envelope outputs for machine use
- table output where useful for humans
- project state under a hidden workspace directory
- active project pointer via `HEAD`, overridable by an environment variable
- append-only or versioned artifacts rather than silent mutation
- object handoff for model execution: create `.jobs.ep` files for the EP/EDSL
  runtime to run, then register externally produced `.results.ep` files
- small command modules behind one `cli.py` entrypoint
- tests that call command functions and inspect written artifacts

## Package Contract

Install from the parent package as a real console script:

```toml
[project.scripts]
umriss = "umriss.cli:main"
```

Suggested package layout:

```text
umriss/
  __init__.py
  cli.py
  cli_parser.py
  errors.py
  state.py
  metadata.py
  support_designs.py
  support_commands.py
  ep_commands.py
  parsing.py
  calibration.py
  evaluation.py
  comparison.py
  reporting.py
  jsonlio.py
```

The user-facing command should be `umriss`. The import package should remain
descriptive: `umriss`.

## Relationship To Existing Tools

`umriss` is not `survey-prior`.

- `survey-prior` elicits direct LLM prior marginals and combines them with real
  responses through Dirichlet updating.
- `umriss` elicits or imports a bank of synthetic support points, then fits a
  mixture over that bank to match observed aggregate marginals.

`umriss` is adjacent to `zwill`, but not a replacement for it.

- `zwill` builds and validates respondent-level digital twin studies.
- `umriss` works when respondent-level microdata are unavailable or deliberately
  held back, and only aggregate moments/toplines are observed.

## Core Estimator

The central object is a support-bank mixture:

```text
B: support bank, with rows as support points and columns as item-option probabilities
y: observed aggregate marginals for held-in item-options
pi: nonnegative simplex weights over support rows
B_h' pi: predicted marginal vector for held-out item h
```

Default fit:

```text
min_pi ||B_heldin' pi - y_heldin||^2 + rho * KL(pi || base)
subject to pi >= 0, sum(pi) = 1
```

Version 1 should implement entropy-regularized calibration using the stable
softmax parameterization already used in `analysis/marginal_weighting_pilot.py`.

Default `rho` grid:

```text
0.0003, 0.001, 0.003, 0.01, 0.03
```

Default model-selection rule for leave-one-out:

```text
held_in_residual + 0.002 / max(effective_support, 1.0)
```

This rule is pre-outcome for each held-out item because it uses only held-in fit
and weight concentration.

## EP Object Boundary

`umriss` should use the same execution boundary as `zwicky`: it prepares EP
objects and records returned results, but it does not run jobs.

The canonical workflow is:

```text
design prompts -> create .jobs.ep -> external EP/EDSL run creates .results.ep -> register .results.ep -> parse support bank
```

This keeps prompt design, model execution, result registration, and parsing as
separate inspectable steps. `umriss` owns bookkeeping and deterministic analysis;
the calling agent or EP runtime owns execution, waiting, retries, API keys,
offloading, and failures during model calls.

Canonical sequence:

```sh
umriss support build ...
umriss support export --prompts <tag>.jsonl --path <tag>.jobs.ep
umriss support register-results --results <tag>.results.ep --metadata battery.json --tag <tag>
umriss support parse --raw <tag>_raw.csv --metadata battery.json --tag <tag>
```

`umriss support export` should return a run contract:

```json
{
  "run_contract": {
    "owner": "agent",
    "jobs_path": "<tag>.jobs.ep",
    "expected_results_path": "<tag>.results.ep",
    "run_command": "ep run --jobs <tag>.jobs.ep --output <tag>.results.ep"
  },
  "next_steps": [
    "Run `ep run --jobs <tag>.jobs.ep --output <tag>.results.ep`.",
    "Run `umriss support register-results --results <tag>.results.ep --tag <tag>`."
  ]
}
```

There should be no `umriss ep-run`, `umriss edsl-run`, `umriss support run`, or
other command that performs model execution. No command should hide a model call
behind a parser, fitter, evaluator, report renderer, or comparison command.
Parsing and rendering consume registered objects; they do not call frontier
models.

## Command Output Envelope

Like `zwill`, every command should be able to emit one JSON envelope. Commands
that have natural human summaries may default to `table`, but `--format json`
must return this shape:

```json
{
  "command": "umriss support parse",
  "status": "ok",
  "data": {},
  "warnings": [],
  "errors": [],
  "next_steps": []
}
```

Failure:

```json
{
  "command": "umriss support parse",
  "status": "error",
  "data": {},
  "warnings": [],
  "errors": [
    {
      "code": "probability_length_mismatch",
      "message": "Item h expected 5 probabilities and received 4.",
      "context": {
        "job_id": "w157_pattern_coverage_support_n96_043",
        "item": "h"
      },
      "hint": "Inspect the raw model response or rerun this support point."
    }
  ],
  "next_steps": []
}
```

Implementation notes:

- Define `UmrissError` with `code`, `message`, `hint`, `context`, and
  `next_steps`, mirroring `ZwillError`.
- `main()` catches `UmrissError` and returns exit code `1`.
- Successful commands return exit code `0`.
- Unexpected exceptions should not be swallowed during tests; for CLI use they
  may be wrapped as `internal_error` after printing a concise message.

Common error codes:

- `not_initialized`
- `invalid_project_id`
- `metadata_invalid`
- `item_not_found`
- `marginal_missing`
- `unsupported_format`
- `probability_json_invalid`
- `probability_length_mismatch`
- `probability_all_zero`
- `support_empty`
- `fit_not_converged`
- `path_exists`
- `edsl_unavailable`

## Workspace And Project State

`umriss init` creates `.umriss/`, following the shape of `.zwill/`.

```text
.umriss/
  config.json
  HEAD
  projects/
    default/
      project.json
      batteries.json
      batteries/
      support_prompts/
      support_raw/
      support_banks/
      fits/
      evaluations/
      comparisons/
      reports/
      workflows/
```

`UMRISS_PROJECT=<project_id>` overrides `.umriss/HEAD` for one command.

State should be useful but not mandatory. Every artifact-producing command
should also accept explicit `--metadata`, `--raw`, `--support`, `--out`, etc.,
so paper Makefiles can run without relying on active project state.

Path resolution rule:

1. If an explicit path flag is supplied, use it exactly as given.
2. If no explicit path is supplied and a project is active, read/write the
   corresponding artifact directory under `.umriss/projects/<project_id>/`.
3. If neither applies, fail with `not_initialized` or `invalid_input` and a
   concrete next command.

This differs slightly from paper scripts, which use hard-coded `data/` paths.
The CLI should make those paths explicit in Makefiles rather than infer them.

Config fields:

```json
{
  "schema_version": 1,
  "created_at": "2026-07-13T00:00:00Z",
  "output_dir": "umriss_work",
  "default_model": "gpt-5.5",
  "default_rho": [0.0003, 0.001, 0.003, 0.01, 0.03]
}
```

Project fields:

```json
{
  "project_id": "default",
  "title": "Default umriss project",
  "created_at": "2026-07-13T00:00:00Z",
  "schema_version": 1
}
```

## File Contracts

### Battery Metadata

Required:

```json
{
  "wave": "W157",
  "battery": "SKILLIMP",
  "topic": "importance of skills for success in today's economy",
  "context": "A nationally representative Pew survey...",
  "items": {
    "a": {
      "variable": "SKILLIMP_a_W157",
      "item_text": "Interpersonal skills...",
      "question_stem": "Now thinking about workers..."
    }
  },
  "option_codes": [1, 2, 3, 4, 5],
  "option_labels": [
    "Extremely important",
    "Very important",
    "Somewhat important",
    "Not too important",
    "Not at all important"
  ]
}
```

Optional:

- `survey_key`
- `respondent_id`, default `respondent_id`
- `weight`, default `weight`
- `covariates`
- `truth`
- `n_respondents`

Item-level `option_codes` and `option_labels` override battery-level values.

### Target Marginals

Supported sources:

1. Battery metadata with `truth`.
2. Long CSV.
3. Wide CSV, converted by `umriss marginals import`.

Canonical long CSV:

```text
battery,item,option_index,option_code,option_label,proportion
W157_SKILLIMP,a,0,1,Extremely important,0.47
```

Option index is zero-based internally. Option codes preserve source coding.

### Respondent Microdata

Only required for retrospective evaluation.

Expected columns:

- respondent id column from metadata, default `respondent_id`
- weight column from metadata, default `weight`
- item columns named `item_<item_id>`, such as `item_a`

### Support Prompt JSONL

Each row:

```json
{
  "job_id": "w157_pattern_coverage_support_n96_001",
  "battery": "W157_SKILLIMP",
  "support_id": 1,
  "variant": "pattern_anchor",
  "prompt": "..."
}
```

All design metadata should be preserved.

### Support Design CSV

Minimum columns:

```text
support_id,job_id,variant,moment_dimension
```

Strategy-specific columns are allowed:

- `persona`
- `coverage_item`
- `coverage_option`
- `coverage_style`
- one column per item for scaffolded response patterns

### Raw Support Responses

Version 1 reads existing EDSL CSV output for backward compatibility with this
paper's already-computed artifacts:

- `answer.resp`
- `scenario.job_id`
- `scenario.prompt`

New model-call workflows should prefer externally produced `.results.ep` files
plus `umriss support register-results`; the raw CSV is a registered/derived
artifact, not the primary model-call object. It should preserve model and
token/cost columns when present.

Future-compatible JSONL form:

```json
{
  "job_id": "w157_pattern_coverage_support_n96_001",
  "response": "{\"persona\": \"...\", \"probabilities\": {\"a\": [...]}}"
}
```

### Parsed Support Bank

Points:

```text
support_id,job_id,variant,persona,valid
```

Probabilities:

```text
support_id,job_id,item,option_index,option_code,option_label,probability
```

Parse diagnostics:

```text
job_id,support_id,status,code,message,item,raw_sum,min_probability,max_probability
```

### Fit Artifacts

Weights:

```text
support_id,job_id,weight
```

Fit diagnostics:

```text
tag,fit_id,heldout_item,selected_rho,held_in_residual,effective_support,max_weight,top10_weight_share,n_support_valid,converged
```

Predictions:

```text
tag,fit_id,item,option_index,option_code,option_label,prediction
```

### Evaluation Artifacts

For compatibility with this paper, `umriss loo` should write:

```text
<tag>_generated_support_detail.csv
<tag>_generated_support_summary.csv
<tag>_generated_support_diagnostics.csv
<tag>_generated_support_points.csv
```

The detail file should include the current columns from
`analysis/score_generated_support.py`:

```text
tag,battery,holdout,item_text,method,rmse,prediction,truth,selected_rho,held_in_residual,effective_support,n_support_valid
```

## Command Groups

Top-level help should be modeled after `zwill`:

```text
command groups:

  getting started   init, status, project, guide, next
  batteries         battery, marginals
  support banks     support
  model jobs        support export, support register-results
  calibration       fit, predict, loo
  comparisons       compare, report
  misc              version
```

### `umriss init`

Create workspace state.

```sh
umriss init
umriss init --output-dir umriss_work
```

Returns:

```json
{
  "command": "umriss init",
  "status": "ok",
  "data": {
    "path": ".umriss",
    "schema_version": 1,
    "active_project": "default",
    "project_path": ".umriss/projects/default"
  },
  "warnings": [],
  "errors": [],
  "next_steps": [
    "umriss battery import --metadata <metadata.json>",
    "umriss support build --metadata <metadata.json> --strategy pattern-coverage --tag <tag>"
  ]
}
```

### `umriss status`

Summarize active project and artifact counts.

```sh
umriss status
umriss status --format json
```

Table output should show:

- active project
- known batteries
- support prompt sets
- parsed support banks
- completed LOO evaluations

### `umriss project`

Mirror `zwill project`.

```sh
umriss project create paper --use
umriss project current
umriss project list
umriss project show paper
umriss project use default
```

Project IDs follow `^[A-Za-z0-9][A-Za-z0-9_.-]*$`.

### `umriss battery inspect`

Validate and summarize a metadata file.

```sh
umriss battery inspect data/source/normalized/W157_SKILLIMP_metadata.json
```

Output:

- wave and battery
- item count
- option counts
- moment dimension, computed as `sum(k_item - 1)`
- whether embedded truth exists
- warnings for inconsistent option scales

### `umriss battery import`

Copy or register a battery metadata file in project state.

```sh
umriss battery import \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --battery-id w157_skillimp \
  --title "Pew W157 SKILLIMP"
```

State written:

```text
.umriss/projects/default/batteries/w157_skillimp/battery.json
```

### `umriss battery create`

Create a battery incrementally inside the active project. This is the
`zwill`-style authoring path when the user does not already have a complete
metadata JSON file.

```sh
umriss battery create \
  --battery-id w157_skillimp \
  --wave W157 \
  --battery SKILLIMP \
  --topic "importance of skills for success in today's economy" \
  --context "A nationally representative Pew survey of U.S. adults..."
```

State written:

```text
.umriss/projects/default/batteries/w157_skillimp/battery.json
.umriss/projects/default/batteries/w157_skillimp/questions.jsonl
.umriss/projects/default/batteries/w157_skillimp/marginals.jsonl
```

### `umriss question add`

Add one item/question to a project battery.

```sh
umriss question add \
  --battery w157_skillimp \
  --item a \
  --variable SKILLIMP_a_W157 \
  --question-stem "Now thinking about workers in general..." \
  --item-text "Interpersonal skills, such as getting along with people and resolving conflicts" \
  --option-code 1 --option "Extremely important" \
  --option-code 2 --option "Very important" \
  --option-code 3 --option "Somewhat important" \
  --option-code 4 --option "Not too important" \
  --option-code 5 --option "Not at all important"
```

Append to:

```text
.umriss/projects/default/batteries/w157_skillimp/questions.jsonl
```

The compiled `battery.json` should be regenerated or marked stale after every
question edit.

### `umriss marginal add`

Add one answer distribution for a battery item.

```sh
umriss marginal add \
  --battery w157_skillimp \
  --item a \
  --proportion 0.4743 \
  --proportion 0.3710 \
  --proportion 0.1224 \
  --proportion 0.0232 \
  --proportion 0.0091 \
  --source "published topline"
```

Append to:

```text
.umriss/projects/default/batteries/w157_skillimp/marginals.jsonl
```

The number of proportions must match the option count for the item. Values must
be nonnegative and sum to one within tolerance, or the command fails unless an
explicit `--normalize` flag is supplied.

### `umriss battery compile`

Compile incremental project records into a metadata JSON file compatible with
path-based paper workflows.

```sh
umriss battery compile \
  --battery w157_skillimp \
  --path data/source/normalized/W157_SKILLIMP_metadata.json
```

This lets users move between interactive `zwill`-style authoring and explicit
paper Makefile paths without maintaining two schemas by hand.

### `umriss marginals import`

Normalize aggregate marginals into canonical long form.

```sh
umriss marginals import \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --truth-from metadata \
  --out data/derived/w157_truth_marginals.csv
```

Other forms:

```sh
umriss marginals import --long toplines.csv --out marginals.csv
umriss marginals import --wide toplines_wide.csv --metadata battery.json --out marginals.csv
```

### `umriss support build`

Build support-generation prompts and a design CSV.

Battery input has two mutually exclusive forms:

- `--battery <id>` reads a battery already authored in the active `.umriss`
  project.
- `--metadata <path.json>` reads a complete metadata file directly for a
  stateless run.

Support construction also has two mutually exclusive forms:

- `--strategy pattern-coverage|archetype` uses a built-in recipe.
- `--design <path.json|path.yaml>` uses a declared generic design.

```sh
umriss support build \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --strategy pattern-coverage \
  --tag w157_pattern_coverage_support_n96 \
  --n-support 96 \
  --seed 20260625 \
  --out data/computed_objects/support_prompts
```

Generic design files can replace named strategies:

```sh
umriss support build \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --design designs/w157_skillimp_latent.json \
  --tag w157_designed_support_n96 \
  --n-support 96 \
  --out data/computed_objects/support_prompts
```

Design files may be JSON or YAML. The core schema is a list of support-row
components:

```json
{
  "name": "w157_skillimp_latent",
  "guidance": "Use the response scale literally.",
  "components": [
    {
      "type": "axes",
      "name": "latent_axis_maximin",
      "sampler": {"method": "maximin", "n": 96, "seed": 20260625},
      "axes": {
        "technology_orientation": [
          "skeptical of workplace technology",
          "pragmatic about workplace technology",
          "enthusiastic about workplace technology"
        ],
        "response_style": [
          "reserves extreme options",
          "uses the full response scale"
        ]
      }
    },
    {
      "type": "option-coverage",
      "name": "coverage",
      "n": 12
    },
    {
      "type": "patterns",
      "name": "anchors",
      "patterns": [
        {"a": "Very important", "b": "Somewhat important"}
      ]
    }
  ]
}
```

Supported component types:

- `axes`: sample latent, demographic, or response-style axis combinations.
- `patterns`: compile explicit answer-pattern scaffolds.
- `option-coverage`: add item-option coverage rows.

Supported axis samplers:

- `maximin`: greedily maximize one-hot/Hamming spread over the axis product.
- `full-factorial`: enumerate axis combinations in product order.
- `random`: sample axis combinations, with replacement when needed.

Strategies for version 1:

- `archetype`: generic demographic/persona grid.
- `pattern-coverage`: response-pattern anchors plus item-option coverage.

Strategies soon after version 1:

- `battery-anchor`: topic-specific latent anchors.
- `response-style`: extremity, acquiescence, and uncertainty support.
- `hybrid`: combine multiple design families into one bank.

Outputs:

```text
<out>/<tag>.jsonl
<out>/<tag>_design.csv
```

The `pattern-coverage` strategy should reproduce the logic currently in
`analysis/build_pattern_coverage_support_jobs.py`.

The `archetype` strategy should reproduce the logic currently in
`analysis/build_archetype_support_jobs.py`.

### `umriss support inspect`

Inspect support prompts, raw responses, or parsed support banks.

```sh
umriss support inspect --prompts data/computed_objects/support_prompts/w157_pattern_coverage_support_n96.jsonl
umriss support inspect --raw data/computed_objects/support_raw_responses/w157_pattern_coverage_support_n96_raw.csv
umriss support inspect --bank data/computed_objects/support_banks/w157_pattern_coverage_support_n96_probabilities.csv
```

Table output should include row counts and parse validity.

### `umriss support export`

Export support prompts as a `.jobs.ep` object file. This is the preferred
handoff boundary before external model execution.

```sh
umriss support export \
  --prompts data/computed_objects/support_prompts/w157_pattern_coverage_support_n96.jsonl \
  --model gpt-5.5 \
  --temperature 1.0 \
  --max-tokens 2200 \
  --path data/computed_objects/support_prompts/w157_pattern_coverage_support_n96.jobs.ep
```

Options:

- `--prompts`
- `--model`, repeatable
- `--models`, comma-separated
- `--service-name`
- `--temperature`
- `--max-tokens`
- `--model-param`, repeatable `key=value`
- `--limit`
- `--path`
- `--format json`

Output:

```text
<path>
```

The exported EP object should contain:

- one free-text question named `resp`
- scenarios with `job_id` and `prompt`
- model specs and parameters
- enough metadata to reconstruct the source prompt file and tag

### `umriss support register-results`

Register an externally produced `.results.ep` object file and derive raw CSV
artifacts that downstream support parsing can consume.

```sh
umriss support register-results \
  --results data/computed_objects/support_raw_responses/w157_pattern_coverage_support_n96.results.ep \
  --prompts data/computed_objects/support_prompts/w157_pattern_coverage_support_n96.jsonl \
  --tag w157_pattern_coverage_support_n96 \
  --out data/computed_objects/support_raw_responses
```

Outputs:

```text
<out>/<tag>_raw.csv
<out>/<tag>_jobs.csv
```

This command should store provenance linking the imported raw CSV back to the
`.results.ep` path and prompt JSONL path.

### `umriss support parse`

Parse raw model outputs into a support bank.

```sh
umriss support parse \
  --raw data/computed_objects/support_raw_responses/w157_pattern_coverage_support_n96_raw.csv \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --tag w157_pattern_coverage_support_n96 \
  --out data/computed_objects/support_banks
```

Validation behavior:

- Extract JSON from free text using a conservative first-`{` to last-`}` rule.
- Require a top-level `probabilities` object.
- Require every item in metadata.
- Require correct vector length for each item.
- Clip negative probabilities to zero and warn.
- Normalize positive vectors and record the raw sum.
- Mark invalid support rows instead of silently dropping them.

Outputs:

```text
<out>/<tag>_points.csv
<out>/<tag>_probabilities.csv
<out>/<tag>_parse_diagnostics.csv
```

### `umriss fit`

Fit mixture weights to target marginals.

```sh
umriss fit \
  --support data/computed_objects/support_banks/w157_pattern_coverage_support_n96_probabilities.csv \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --marginals data/source/normalized/W157_SKILLIMP_metadata.json \
  --exclude-item a \
  --rho 0.0003 0.001 0.003 0.01 0.03 \
  --tag w157_pattern_coverage_support_n96_holdout_a \
  --out data/derived
```

Options:

- `--support`
- `--metadata`
- `--marginals`
- `--include-item`, repeatable
- `--exclude-item`, repeatable
- `--rho`, one or more floats
- `--base-weights`, optional CSV
- `--select-rho`, default `heldin-plus-effective-support`
- `--tag`
- `--out`

Outputs:

```text
<out>/<tag>_weights.csv
<out>/<tag>_fit_diagnostics.csv
<out>/<tag>_predictions.csv
```

### `umriss predict`

Apply fitted weights to one or more target items.

```sh
umriss predict \
  --support data/computed_objects/support_banks/w157_pattern_coverage_support_n96_probabilities.csv \
  --weights data/derived/w157_pattern_coverage_support_n96_holdout_a_weights.csv \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --item a \
  --out data/derived/w157_pattern_coverage_support_n96_holdout_a_prediction.csv
```

### `umriss loo`

Run leave-one-item-out evaluation for one support bank.

```sh
umriss loo \
  --raw data/computed_objects/support_raw_responses/w157_pattern_coverage_support_n96_raw.csv \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --respondents data/source/normalized/W157_SKILLIMP_respondents.csv \
  --one-shot data/source/one_shot/PEW_W157_SKILLIMP_UNCONDITIONED_ONE_SHOT_gpt-5.5.csv \
  --two-step data/source/two_step/PEW_W157_TWO_STEP_UNCONDITIONED_gpt-5.5.csv \
  --tag w157_pattern_coverage_support_n96 \
  --out data/derived
```

This command intentionally combines parse, fit, predict, and score because it
is the paper's main empirical loop.

Inputs:

- `--raw` raw support responses, or `--support` parsed support probabilities
- `--metadata`
- `--marginals`, optional; defaults to metadata `truth` or computed from
  respondents
- `--respondents`, optional when `--marginals` is supplied
- `--one-shot`, optional baseline
- `--two-step`, optional baseline
- `--rho`
- `--tag`
- `--out`

Methods included by default:

- `generated support mixture`
- `unweighted archetype bank`
- `uniform`
- `unconditioned one-shot`, when supplied
- `conditioned one-shot`, when supplied

Outputs:

```text
<tag>_generated_support_detail.csv
<tag>_generated_support_summary.csv
<tag>_generated_support_diagnostics.csv
<tag>_generated_support_points.csv
```

This should reproduce `analysis/score_generated_support.py` for existing paper
inputs.

### `umriss compare`

Combine multiple LOO summaries into paper-ready comparison tables.

```sh
umriss compare \
  --run w154_archetype_support_n96="W154 DIFF1:Generic demographic" \
  --run w154_pattern_coverage_support_n96="W154 DIFF1:Pattern coverage" \
  --run w154_battery_anchor_support_n96="W154 DIFF1:Battery-designed" \
  --derived data/derived \
  --out data/derived/pattern_coverage_cross_battery_summary.csv
```

Version 1 can implement only a generic comparison table:

```text
comparison_group,battery,bank,tag,method,mean_rmse,median_rmse,max_rmse,items
```

Later versions can add named comparison recipes:

- `battery-designed`
- `pattern-coverage`
- `generic-overlay`
- `negative-controls`

Current implementation includes:

```sh
umriss compare \
  --recipe battery-designed \
  --derived data/derived \
  --out data/derived/battery_designed_support_comparison_summary.csv

umriss compare \
  --recipe pattern-coverage \
  --derived data/derived \
  --out data/derived/pattern_coverage_cross_battery_summary.csv
```

The battery-designed recipe also writes sibling `_detail.csv` and `_wide.csv`
files. The pattern-coverage recipe also writes a sibling `_plot_data.csv` file.

### `umriss report`

Produce an HTML or Markdown diagnostics report for a support bank or LOO run.

Version 1 writes a compact Markdown report from generated-support summary,
diagnostics, and support-point artifacts:

```sh
umriss report \
  --tag w157_archetype_support_n96 \
  --derived data/derived \
  --out output/reports/w157_archetype_support_n96.md
```

The report is intentionally artifact-based; it does not rerun fitting or EP
jobs.

### `umriss report-data build`

Build a zwill-style report-writing data bundle from already-derived artifacts.
This command scans a derived directory; it does not rerun parsing, fitting, LOO,
or EP jobs.

```sh
umriss report-data build \
  --derived data/derived \
  --out output/report_data
```

Outputs:

```text
manifest.json
support_bank_summary.csv
method_comparison.csv
holdout_detail.csv
diagnostics_flags.csv
support_points.csv
prose_facts.json
prose_facts.md
```

Intended use:

- `support_bank_summary.csv`: one generated-support row per support bank
- `method_comparison.csv`: normalized method summary rows across all banks
- `holdout_detail.csv`: item-level LOO detail rows
- `diagnostics_flags.csv`: support-bank diagnostics for report triage
- `support_points.csv`: support-point validity/provenance rows when available
- `prose_facts.{json,md}`: compact numbers and rankings for paper prose
- `manifest.json`: input/output inventory for reproducibility

### `umriss guide` And `umriss next`

Mirror `zwill guide` and `zwill next`: provide concise operational guidance
and compute the next artifact-producing command without running EP jobs.

Guides are built into the package and emitted as structured JSON:

```sh
umriss guide
umriss guide designs
umriss guide ep-boundary
umriss guide paper-rewrite
umriss guide diagnostics
```

Topics:

- `workflow`: support-bank lifecycle from metadata to report
- `designs`: generic design-file approach for replacing named prompt builders
- `ep-boundary`: explicit contract that `umriss` creates `.jobs.ep` files and
  registers `.results.ep` files, but does not run jobs
- `paper-rewrite`: guidance for replacing bespoke scripts with Makefile command
  patterns
- `diagnostics`: parse, inspect, fit, LOO, and report checks

`umriss next` has two modes. Without a tag it inspects a local `umriss` workspace
and recommends project-level setup steps. With `--tag`, it inspects concrete
paper artifacts and recommends the next command in the support-bank lifecycle:

```sh
umriss next \
  --tag w157_designed_support_n96 \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --design designs/w157_skillimp_latent.json
```

Artifact-aware `next` checks, in order:

- metadata file
- prompt JSONL
- `.jobs.ep` job file
- externally produced `.results.ep` file
- registered raw CSV
- fitted probabilities
- generated-support summary

It returns a `stage`, a shell-ready `recommendation`, and an `exists` map. The
recommended command is always a `umriss` command except at the EP execution
boundary, where it says to run the `.jobs.ep` file externally and then come back
with `umriss support register-results`.

## Paper Rewrite Plan

The immediate goal is to rewrite this paper so empirical construction uses
`umriss` commands instead of bespoke scripts.

Recommended Makefile pattern:

```make
UMRISS=UV_CACHE_DIR=$(CURDIR)/.uv-cache uv run umriss
UMRISS_DESIGN=$(UMRISS) support build --metadata $(1) --design $(2) --tag $(3) --n-support $(4) --out data/computed_objects/support_prompts

data/computed_objects/support_prompts/w157_pattern_coverage_support_n96.jsonl:
	$(UMRISS) support build \
	  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
	  --strategy pattern-coverage \
	  --tag w157_pattern_coverage_support_n96 \
	  --n-support 96 \
	  --out data/computed_objects/support_prompts

data/computed_objects/support_prompts/w157_designed_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W157_SKILLIMP_metadata.json,designs/w157_skillimp_latent.json,w157_designed_support_n96,96)

data/computed_objects/support_prompts/w157_latent_anchor_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W157_SKILLIMP_metadata.json,designs/w157_skillimp_latent_anchor.json,w157_latent_anchor_support_n96,96)

data/computed_objects/support_prompts/w157_response_style_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W157_SKILLIMP_metadata.json,designs/w157_skillimp_response_style.json,w157_response_style_support_n96,96)

data/computed_objects/support_prompts/w157_response_style_support_n192.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W157_SKILLIMP_metadata.json,designs/w157_skillimp_response_style_n192.json,w157_response_style_support_n192,192)

data/computed_objects/support_prompts/w157_hybrid_support_n144.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W157_SKILLIMP_metadata.json,designs/w157_skillimp_hybrid.json,w157_hybrid_support_n144,144)

data/computed_objects/support_prompts/w154_battery_anchor_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W154_DIFF1_metadata.json,designs/w154_diff1_battery_anchor.json,w154_battery_anchor_support_n96,96)

data/computed_objects/support_prompts/w158_battery_anchor_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W158_CCPOLICY_metadata.json,designs/w158_ccpolicy_battery_anchor.json,w158_battery_anchor_support_n96,96)

data/computed_objects/support_prompts/w163_battery_anchor_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/W163_SM9_metadata.json,designs/w163_sm9_battery_anchor.json,w163_battery_anchor_support_n96,96)

data/computed_objects/support_prompts/gallup_wellbeing_battery_anchor_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/GALLUP_WELLBEING_gallup_wellbeing_metadata.json,designs/gallup_wellbeing_battery_anchor.json,gallup_wellbeing_battery_anchor_support_n96,96)

data/computed_objects/support_prompts/gallup_remote_covid_battery_anchor_support_n96.jsonl:
	$(call UMRISS_DESIGN,data/source/normalized/GALLUP_REMOTE_COVID_gallup_remote_covid_metadata.json,designs/gallup_remote_covid_battery_anchor.json,gallup_remote_covid_battery_anchor_support_n96,96)

data/derived/w157_pattern_coverage_support_n96_generated_support_summary.csv:
	$(UMRISS) loo \
	  --raw data/computed_objects/support_raw_responses/w157_pattern_coverage_support_n96_raw.csv \
	  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
	  --respondents data/source/normalized/W157_SKILLIMP_respondents.csv \
	  --one-shot data/source/one_shot/PEW_W157_SKILLIMP_UNCONDITIONED_ONE_SHOT_gpt-5.5.csv \
	  --two-step data/source/two_step/PEW_W157_TWO_STEP_UNCONDITIONED_gpt-5.5.csv \
	  --tag w157_pattern_coverage_support_n96 \
	  --out data/derived
```

Scripts that should shrink or disappear:

- `analysis/build_pattern_coverage_support_jobs.py`
- `analysis/build_archetype_support_jobs.py`
- `edsl_jobs/run_support_generation.py`
- `analysis/score_generated_support.py`
- portions of `analysis/build_battery_designed_support_comparison.py`
- portions of `analysis/build_pattern_coverage_cross_battery_summary.py`

Scripts that likely remain paper-specific:

- R figure builders
- appendix table formatters
- highly specific cross-battery narrative tables

The writeup should describe the methods as `umriss` stages:

1. Design support bank.
2. Generate support point response distributions.
3. Parse and validate support bank.
4. Fit entropy-regularized marginal mixture weights.
5. Predict held-out marginals.
6. Compare to one-shot and conditioned direct priors.
7. Run negative controls and support-repair diagnostics.

## Implementation Priorities

### Phase 1: Deterministic Core

Implement without live model calls:

1. `umriss battery inspect`
2. `umriss support build --strategy archetype`
3. `umriss support build --strategy pattern-coverage`
4. `umriss support parse`
5. `umriss fit`
6. `umriss predict`
7. `umriss loo` using existing raw support CSVs

This phase should already rewrite much of the paper Makefile.

### Phase 2: EDSL Execution

Add:

1. `umriss support export`
2. `umriss support register-results`
3. run-contract metadata and next steps that tell the calling agent how to run
   the `.jobs.ep` file externally

This replaces `edsl_jobs/run_support_generation.py` with an agent-owned
execution boundary. `umriss` creates the job package and records the returned
results, but does not run the job.

### Phase 3: Project State And Guides

Add:

1. `umriss init`
2. `umriss status`
3. `umriss project`
4. `umriss guide`
5. `umriss next`

Project state should not block paper use; it makes the CLI feel complete and
consistent with `zwill`.

### Phase 4: Comparisons And Reports

Add:

1. `umriss compare`
2. named comparison recipes
3. support-bank diagnostics report
4. LOO HTML/Markdown report

## Test Plan

Follow `zwill`'s testing style: construct small fixtures, call command functions
or `main(argv)`, and assert files and envelope fields.

Minimum tests:

1. `test_metadata_inspect_accepts_w157_fixture`
2. `test_pattern_coverage_build_writes_jsonl_and_design_csv`
3. `test_archetype_build_writes_expected_number_of_prompts`
4. `test_parse_support_extracts_json_from_edsl_csv`
5. `test_parse_support_records_invalid_probability_length`
6. `test_entropy_fit_recovers_known_two_point_mixture`
7. `test_fit_writes_weights_diagnostics_predictions`
8. `test_support_export_writes_jobs_ep_and_run_contract`
9. `test_register_results_records_results_ep_provenance`
10. `test_loo_reproduces_current_w157_summary_shape`
11. `test_init_creates_workspace_default_project_and_head`
12. `test_project_use_and_env_override`

Fixtures should be tiny and local:

```text
tests/fixtures/
  mini_metadata.json
  mini_respondents.csv
  mini_support_raw.csv
  mini_one_shot.csv
  mini_two_step.csv
```

Default tests must not run EDSL jobs or make network calls. They may assert
that `.jobs.ep` files and run contracts are created, and they may register a
fixture `.results.ep` or a compatibility raw CSV.

## Coding Conventions

Match `zwill` where practical:

- `cli.py` owns shared helpers, envelope, workspace paths, and command shims.
- `cli_parser.py` builds the parser and imports `from .cli import *`.
- Command-heavy modules can import shared helpers from `cli.py` only if that
  does not create circular imports; otherwise put neutral logic in small
  modules like `metadata.py`, `calibration.py`, and `parsing.py`.
- Use `Path` everywhere.
- Use CSV/JSON/JSONL helpers rather than ad hoc repeated file handling.
- Use `rich` only for optional table output, matching parent dependencies.
- Keep scipy/numpy/pandas dependencies aligned with the parent project.
- No model calls in import time.
- No writing outside explicit output paths or `.umriss/`.

## Version 1 Acceptance Criteria

The first useful version is done when:

1. `umriss support build --strategy pattern-coverage` reproduces current support
   prompt/design files modulo row ordering and timestamp-free metadata.
2. `umriss support build --strategy archetype` reproduces current archetype
   prompt files.
3. `umriss support parse` parses existing EDSL raw CSVs and writes valid support
   bank artifacts.
4. `umriss support export` writes a reusable `.jobs.ep` file and a run contract
   for the calling agent; `umriss` itself does not run the job.
5. `umriss support register-results` records an externally produced
   `.results.ep` file and derives compatibility CSVs for parsing.
6. `umriss loo` reproduces `analysis/score_generated_support.py` output shape
   and materially identical W157 summary values for one existing support bank.
7. The parent package installs a working `umriss` console script.
8. The paper Makefile can replace at least three Python analysis commands with
   `umriss` commands without changing downstream figure scripts.

## Later Extensions

- Aggregate-only external marginal ingestion from published toplines.
- Cross-battery pooled support banks.
- Augmented moment matching for covariates or published crosstabs.
- Negative-control recipes: permuted marginals, unrelated batteries, uniform
  response banks.
- Bootstrap or posterior uncertainty over fitted weights.
- Support-bank repair suggestions when held-in residuals are high.
- HTML diagnostics report for a support bank and fitted mixture.
- Registration of remote/offloaded EP results produced outside `umriss`.
- Workflow YAML runner modeled on `zwill workflow run`.
