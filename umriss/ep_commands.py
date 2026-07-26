from __future__ import annotations

import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import UmrissError
from .jsonlio import read_jsonl, write_json
from .provenance import build_provenance, guard_manifest, path_sha256


def export_support_jobs(
    prompts_path: Path,
    path: Path,
    *,
    model: list[str] | None = None,
    service_name: str | None = None,
    temperature: float = 1.0,
    max_tokens: int = 2200,
    limit: int | None = None,
    workflow: str = "support",
    tag: str | None = None,
    registration_out: Path | None = None,
    job_ids_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved_tag = tag or prompts_path.name.removesuffix("_prompts.jsonl").removesuffix(".jsonl")
    raw_out = registration_out or path.parent / "raw"
    manifest_path = path.with_name(f"{resolved_tag}_manifest.json")
    model_specs = [
        {
            "model": spec.split(":", 1)[-1],
            "service_name": spec.split(":", 1)[0] if ":" in spec else service_name,
        }
        for spec in (model or ["gpt-5.5"])
    ]
    provenance = build_provenance(
        f"umriss {workflow} export",
        inputs={"prompts": prompts_path, "job_ids": job_ids_path},
        parameters={
            "tag": resolved_tag,
            "path": str(path),
            "models": model_specs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "limit": limit,
            "registration_out": str(raw_out),
            "job_ids_path": str(job_ids_path) if job_ids_path else None,
        },
    )
    existing = guard_manifest(manifest_path, provenance, outputs=[path], force=force)
    if existing is not None:
        return {**existing, "reused": True}
    try:
        os.environ.setdefault("EDSL_LOG_DIR", str(path.parent / "edsl_logs"))
        with redirect_stdout(StringIO()):
            from edsl import Jobs, Model, ModelList, QuestionFreeText, Scenario, ScenarioList, Survey
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to export .jobs.ep files.") from exc
    rows = read_jsonl(prompts_path)
    if not rows:
        raise UmrissError("invalid_input", f"No prompt rows found in {prompts_path}.")
    if job_ids_path:
        job_ids_frame = pd.read_csv(job_ids_path)
        if "job_id" not in job_ids_frame:
            raise UmrissError("invalid_input", "Retry job-ID file must contain a `job_id` column.")
        requested = set(job_ids_frame["job_id"].dropna().astype(str))
        rows = [row for row in rows if str(row.get("job_id")) in requested]
        found = {str(row["job_id"]) for row in rows}
        unknown = sorted(requested - found)
        if unknown:
            raise UmrissError(
                "invalid_input",
                f"Retry job-ID file contains {len(unknown)} IDs absent from the prompts.",
                context={"unknown_job_ids": unknown[:20]},
            )
    if limit is not None:
        rows = rows[:limit]
    scenarios = ScenarioList([Scenario({"job_id": row["job_id"], "prompt": row["prompt"]}) for row in rows])
    question = QuestionFreeText(question_name="resp", question_text="{{ scenario.prompt }}")
    survey = Survey([question], name=f"umriss_{workflow}_generation")
    models = []
    try:
        for spec in model or ["gpt-5.5"]:
            if ":" in spec:
                service, model_name = spec.split(":", 1)
            else:
                service, model_name = service_name, spec
            kwargs = {"temperature": temperature, "max_tokens": max_tokens}
            if service:
                kwargs["service_name"] = service
            models.append(Model(model_name, **kwargs))
    except (TypeError, ValueError) as exc:
        raise UmrissError(
            "model_unavailable",
            "Could not construct one or more requested EDSL models.",
            context={"models": model_specs, "error": str(exc)},
            hint="Check the model/service names and the active EDSL profile before exporting Jobs.",
        ) from exc
    jobs = Jobs(survey=survey, scenarios=scenarios, models=ModelList(models))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(jobs, "git") or not hasattr(Jobs, "git"):
        raise UmrissError(
            "edsl_outdated",
            "This EDSL installation does not support git-backed .ep packages.",
            hint="Upgrade EDSL to a build that provides jobs.git.save() and Jobs.git.load().",
        )
    with redirect_stdout(StringIO()):
        save = jobs.git.save(str(path), message=f"Create umriss {workflow} generation jobs")
        jobs_path = save["path"]
        Jobs.git.load(jobs_path)
    results_path = (
        path.with_name(path.name.replace(".jobs.ep", ".results.ep"))
        if path.name.endswith(".jobs.ep")
        else path.with_suffix(".results.ep")
    )
    run_command = f"ep run --jobs {jobs_path} --output {results_path}"
    register_command = (
        f"umriss {workflow} register-results --results {results_path} "
        f"--prompts {prompts_path} --tag {resolved_tag} --out {raw_out}"
    )
    manifest = {
        "schema_version": 1,
        "workflow": workflow,
        "tag": resolved_tag,
        "save_format": "edsl_ep",
        "jobs": jobs_path,
        "results": str(results_path),
        "prompts": str(prompts_path),
        "registration_out": str(raw_out),
        "scenarios": len(rows),
        "questions": 1,
        "iterations": 1,
        "model_count": len(model_specs),
        "model_calls": len(rows) * len(model_specs),
        "models": model_specs,
        "execution": {
            "owner": "external_ep",
            "mode": "unspecified",
            "cost_estimate": {
                "available": False,
                "reason": "Provider pricing is not available at job-export time.",
            },
        },
        "run_command": run_command,
        "register_command": register_command,
        "run_contract": {
            "owner": "external_ep",
            "jobs_path": jobs_path,
            "expected_results_path": str(results_path),
            "run_command": run_command,
            "register_command": register_command,
        },
        "next_steps": [run_command, register_command],
        "provenance": provenance,
        "outputs": {
            "jobs": {
                "path": str(jobs_path),
                "sha256": path_sha256(Path(jobs_path)),
            }
        },
        "reused": False,
    }
    manifest["manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest
