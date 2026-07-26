# Agent Instructions

Use the CLI itself as the workflow source of truth:

```bash
umriss guide
umriss next --tag <tag> --metadata <metadata.json>
```

If `umriss` is not on `PATH` in this checkout, use
`./.venv/bin/umriss`. Do not substitute an older global installation.

Run `umriss next` after every completed stage and follow the returned
recommendation. Use `<command> --help` for current arguments and defaults
rather than inferring them from README or tutorial examples.

## Execution boundary

`umriss` prepares `.jobs.ep` objects but never executes model calls. At the
execution boundary:

1. Inspect the generated jobs and the EDSL command returned by umriss.
2. Report the model, expected call count, and likely cost when available.
3. Obtain any approval required for external or paid inference.
4. Run the returned `ep run` command externally.
5. Register the result with `umriss support register-results`.
6. Run `umriss next` again.

Do not replace a registered result with hand-authored or model-shaped test data.
Do not silently drop failed calls, renormalize incomplete weight mass, or
repair invalid probability vectors.

## Audit trail

Commands emit one JSON envelope for programmatic consumption. Preserve the
generated:

- battery metadata and resolved designs;
- prompts, prompt HTML, support plans, and coverage reports;
- `.jobs.ep` objects and result registrations;
- raw registered results and strictly parsed support banks;
- uniformity, diversity, fit, and validation diagnostics;
- fitted weights, comparisons, plots, reports, and export manifests.

Keep stable tags and job IDs intact so artifacts can be joined and audited.
Record retries explicitly; never disguise a merged retry as a complete first
run.

## Credentials and publication

Let EDSL manage authentication. Never print, copy, log, or commit API keys or
the contents of `.env` files.

Never publish `original_work/`, licensed respondent microdata, local credential
files, or private paper provenance. The public tutorial may contain derived
aggregate marginals and artifacts that are explicitly cleared for release.

## Repository checks

Before committing code or documentation, run:

```bash
uv run ruff check umriss tests
uv run pytest -q
git diff --check
```

Preserve unrelated user changes. Generated tutorial claims, tables, and plots
must come from the checked-in run artifacts; rerun the relevant analysis when a
claimed value changes.
