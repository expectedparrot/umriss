# Trust tutorial run

This directory contains the checked-in data and real EDSL run used by
`docs/index.html`.

- `trust_metadata.json` is a four-item teaching battery with known marginal
  targets.
- `run/prompts/trust_coverage_n12.jobs.ep` is the generated EDSL job package.
- `run/prompts/trust_coverage_n12.results.ep` contains twelve real model
  interviews run through Expected Parrot.
- `run/ep-run.json` is the captured `ep run` JSON envelope, with its saved
  path normalized to the checked-in repository path.
- `run/raw/`, `run/banks/`, `run/fits/`, `run/derived/`, and
  `run/report_data/` contain the registered and deterministic downstream
  artifacts.

The EDSL run completed with Results UUID
`a3e66e84-aced-4eba-b77b-05157cbedb80`. Its local `.results.ep` package is the
provenance source for every probability, fit, prediction, and score shown in
the documentation.
