# umriss

## Copy and paste into a coding agent

```text
Set up umriss and help me build an auditable digital-twin population from
reported survey marginals in this repository.

Install the current umriss main branch and its temporarily required EDSL
probabilistic-response branch:

uv tool install --upgrade --force \
  --with-executables-from "edsl @ git+https://github.com/expectedparrot/edsl.git@feature/probabilistic-response-contract" \
  "umriss @ git+https://github.com/expectedparrot/umriss.git@main"

Verify that `umriss` and `ep` resolve inside the directory reported by
`uv tool dir --bin`. Do not use an older executable. Then run:

umriss --help
ep --help
umriss guide

Treat the CLI as the workflow source of truth. After each stage, run:

umriss next --tag <tag> --metadata <metadata.json>

Follow the returned recommendation. Umriss prepares `.jobs.ep` objects but
does not execute model calls. Before external or paid inference, show me the
jobs, model, call count, and returned EDSL command and wait for my approval.
After execution, register the result with `umriss support register-results`.

Never display or commit API keys. Preserve generated prompts, jobs, result
registrations, support banks, fit diagnostics, validation outputs, and reports
as the run's audit trail. Continue until the workflow is complete or my input
or approval is required.
```

![Umriss artwork: a population of distinct parrots inside brackets](docs/assets/umriss-art.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![EDSL](https://img.shields.io/badge/built%20on-EDSL-brightgreen.svg)](https://github.com/expectedparrot/edsl)

`umriss` is an agent-facing Python CLI for constructing auditable synthetic
populations from reported survey marginals. It creates a broad bank of
synthetic response profiles, measures each profile's item-level response
probabilities, and estimates nonnegative mixture weights whose implied
marginals approximate the reported targets.

The result is a calibrated synthetic population, not a reconstruction of the
original respondents. Support design, model elicitation, and regularization are
consequential assumptions and remain visible in the run's audit trail.

The [worked browser tutorial](https://expectedparrot.github.io/umriss/) follows
a real five-item Pew Research Center example from aggregate targets through
support construction, uniformity diagnostics, leave-one-out validation,
weighting, EDSL `AgentList` export, and downstream probabilistic surveys.

## Install

Until EDSL's probabilistic-response contract is merged into its main branch,
install umriss with the required feature branch:

```bash
uv tool install --upgrade --force \
  --with-executables-from "edsl @ git+https://github.com/expectedparrot/edsl.git@feature/probabilistic-response-contract" \
  "umriss @ git+https://github.com/expectedparrot/umriss.git@main"
```

Verify the executables and inspect the authoritative workflow:

```bash
command -v umriss
command -v ep
umriss guide
```

For repository development:

```bash
uv sync --extra edsl --extra dev
uv run pytest -q
uv run ruff check umriss tests
```

## Start a run

Inspect the battery, then ask umriss for the next valid action:

```bash
umriss battery inspect examples/pew_w154/pew_w154_metadata.json
```

```bash
umriss next \
  --tag pew_w154_diff1 \
  --metadata examples/pew_w154/pew_w154_metadata.json
```

Run `umriss next` after every stage. The returned JSON envelope identifies the
recommended command and preserves the boundary between preparing EDSL jobs and
executing them externally.

## What the workflow produces

A complete run preserves:

- battery metadata and a resolved, versioned support design;
- prompt JSONL, readable prompt HTML, support plans, and coverage audits;
- git-backed EDSL Jobs and registered Results;
- strictly parsed persona and probability banks;
- pre-fit uniformity and diversity diagnostics;
- fitted weights, implied marginals, and leave-one-out predictions;
- comparison tables, plots, reports, and reusable weighted AgentLists.

Umriss does not silently normalize invalid probability vectors, truncate
declared coverage, execute model jobs, or describe generated profiles as
recovered respondents.

## Principal command groups

| Command group | Role |
|---|---|
| `umriss battery` and `umriss question` | Record and inspect survey wording, options, scale semantics, and targets |
| `umriss design` and `umriss support` | Declare support geometry, build prompts, register results, parse banks, and test coverage |
| `umriss fit` and `umriss validate` | Estimate weights and evaluate omitted marginals |
| `umriss baseline` and `umriss compare` | Construct auditable alternatives and compare their predictions |
| `umriss twins` | Export weighted EDSL agents and evaluate downstream surveys |
| `umriss plot` and `umriss report` | Produce figures and inspectable run reports |
| `umriss guide` and `umriss next` | Explain the lifecycle and return the next state-aware action |

Use `<command> --help` for exact arguments, choices, and defaults. Do not rely
on copied option inventories in prose.

## Identification warning

Reported marginals do not identify a unique joint distribution. Many synthetic
populations can reproduce the same one-way tables. Umriss makes one
construction explicit and tests whether weights fitted to some marginals
predict omitted marginals; it does not make the identification problem
disappear.

Target marginals must not enter support-generation prompts. Support uniformity
must be measured before fitting. Design searches, simulated moment conditions,
custom prompts, repair decisions, and excluded results must remain documented.

## Documentation

- [Worked tutorial](https://expectedparrot.github.io/umriss/)
- [Local tutorial source](docs/index.html)
- [Agent operating contract](AGENTS.md)
- [CLI specification](CLI_SPEC.md)
- [MIT license](LICENSE)
