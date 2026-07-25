# umriss

**A Python CLI for constructing marginal-matching digital twin support banks.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![EDSL](https://img.shields.io/badge/built%20on-EDSL-brightgreen.svg)](https://github.com/expectedparrot/edsl)

`umriss` builds and evaluates synthetic support sets whose answer distributions
can be mixed to match known survey marginals. It is designed for experiments
where you have item-level marginal distributions, want to construct plausible
digital twin support points, and need an auditable path from prompt design to
leave-one-item-out evaluation.

The CLI follows the same operational boundary as `zwill` and `zwicky`:

- `umriss` creates prompt JSONL artifacts and `.jobs.ep` files.
- An external EP/EDSL runner executes those jobs and writes `.results.ep` files.
- `umriss` registers results, parses model probabilities, fits mixture weights,
  and writes diagnostics, comparison tables, and reports.

It does not run model jobs itself.

## Install

From a clone of this repository:

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[test]"
pytest -q
```

If you are co-developing EDSL, install `umriss` first and then overlay your local
EDSL checkout:

```bash
pip install -e .
pip install -e ../edsl
```

## Quick Start

Inspect or create battery metadata:

```bash
umriss battery inspect data/source/normalized/W157_SKILLIMP_metadata.json
```

Build support prompts from a generic design file:

```bash
umriss support build \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --design examples/latent_support_design.json \
  --tag w157_designed_support_n96 \
  --n-support 96 \
  --out data/computed_objects/support_prompts
```

Export an EP job bundle:

```bash
umriss support export \
  --prompts data/computed_objects/support_prompts/w157_designed_support_n96.jsonl \
  --path data/computed_objects/support_prompts/w157_designed_support_n96.jobs.ep
```

Run the resulting `.jobs.ep` file outside `umriss`, then register the `.results.ep`
file:

```bash
umriss support register-results \
  --results data/computed_objects/support_prompts/w157_designed_support_n96.results.ep \
  --prompts data/computed_objects/support_prompts/w157_designed_support_n96.jsonl \
  --tag w157_designed_support_n96 \
  --out data/computed_objects/support_raw_responses
```

Run leave-one-item-out evaluation:

```bash
umriss loo \
  --raw data/computed_objects/support_raw_responses/w157_designed_support_n96_raw.csv \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --respondents data/source/normalized/W157_SKILLIMP_respondents.csv \
  --tag w157_designed_support_n96 \
  --out data/derived
```

Build a report-writing data bundle:

```bash
umriss report-data build \
  --derived data/derived \
  --out output/report_data
```

The bundle contains normalized CSVs for support-bank summaries, method
comparisons, holdout detail, diagnostics flags, and support points, plus
`prose_facts.json`, `prose_facts.md`, and a reproducibility manifest.

## Guidance

`umriss guide` provides short, structured workflow guidance:

```bash
umriss guide
umriss guide designs
umriss guide ep-boundary
umriss guide paper-rewrite
umriss guide diagnostics
```

`umriss next` inspects artifacts for a support-bank tag and recommends the next
command:

```bash
umriss next \
  --tag w157_designed_support_n96 \
  --metadata data/source/normalized/W157_SKILLIMP_metadata.json \
  --design examples/latent_support_design.json
```

The recommendation is a `umriss` command except at the EP execution boundary,
where the correct next step is to run the `.jobs.ep` file externally.

## Design Files

Named prompt builders are intentionally not the main abstraction. Prefer generic
JSON or YAML design files that declare components such as latent axes, response
styles, option coverage rows, and sampling strategy.

Minimal example:

```json
{
  "components": [
    {
      "type": "axes",
      "name": "latent_axis_grid",
      "axes": {
        "technology_outlook": ["optimistic", "skeptical"],
        "response_style": ["moderate", "strong"]
      },
      "sampler": {"method": "maximin", "n": 96, "seed": 17},
      "guidance": "Answer each item using the response scale literally."
    }
  ]
}
```

## Project State

For incremental authoring, initialize a workspace:

```bash
umriss init
umriss battery create --battery-id demo --wave T1 --battery DEMO --topic "demo" --context "demo context"
umriss question add --battery demo --item q1 --question-stem "Stem?" --item-text "Item text" --option-code 1 --option Yes --option-code 2 --option No
umriss marginal add --battery demo --item q1 --proportion 0.25 --proportion 0.75
umriss battery compile --battery demo --path demo_metadata.json
umriss support build --battery demo --strategy pattern-coverage --tag demo_n12 --n-support 12 --out prompts
```

Workspace state lives under `.umriss/`. Paper-scale workflows can also skip local
state entirely and operate directly on metadata, prompt, raw-response, and
derived artifact paths.

## Specification

See [CLI_SPEC.md](CLI_SPEC.md) for the current command contract and paper rewrite
plan.
