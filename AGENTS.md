# Agent Instructions

Use the CLI itself as the workflow source of truth:

```bash
umriss guide
umriss next --tag <tag> --metadata <metadata.json>
```

Run `umriss next` after each stage and follow the returned recommendation.
`umriss` prepares `.jobs.ep` objects but never executes model calls. At the
execution boundary, run the returned EDSL command externally, then register the
result with `umriss support register-results`.

Commands emit one JSON envelope for programmatic consumption. Preserve the
generated prompts, result registrations, support banks, fit diagnostics, and
reports as the audit trail for a run.
