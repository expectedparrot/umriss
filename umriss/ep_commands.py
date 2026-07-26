from __future__ import annotations

import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

from .errors import UmrissError
from .jsonlio import read_jsonl, write_json


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
) -> dict[str, Any]:
    try:
        os.environ.setdefault("EDSL_LOG_DIR", str(path.parent / "edsl_logs"))
        with redirect_stdout(StringIO()):
            from edsl import Jobs, Model, ModelList, QuestionFreeText, Scenario, ScenarioList, Survey
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to export .jobs.ep files.") from exc
    rows = read_jsonl(prompts_path)
    if limit is not None:
        rows = rows[:limit]
    scenarios = ScenarioList([Scenario({"job_id": row["job_id"], "prompt": row["prompt"]}) for row in rows])
    question = QuestionFreeText(question_name="resp", question_text="{{ scenario.prompt }}")
    survey = Survey([question], name=f"umriss_{workflow}_generation")
    models = []
    for spec in model or ["gpt-5.5"]:
        if ":" in spec:
            service, model_name = spec.split(":", 1)
        else:
            service, model_name = service_name, spec
        kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        if service:
            kwargs["service_name"] = service
        models.append(Model(model_name, **kwargs))
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
    results_path = path.with_name(path.name.replace(".jobs.ep", ".results.ep")) if path.name.endswith(".jobs.ep") else path.with_suffix(".results.ep")
    manifest = {
        "save_format": "edsl_ep",
        "jobs": jobs_path,
        "results": str(results_path),
        "prompts": str(prompts_path),
        "scenarios": len(rows),
        "models": [str(m) for m in (model or ["gpt-5.5"])],
        "run_command": f"ep run --jobs {jobs_path} --output {results_path}",
        "run_contract": {
            "owner": "agent",
            "jobs_path": jobs_path,
            "expected_results_path": str(results_path),
            "run_command": f"ep run --jobs {jobs_path} --output {results_path}",
        },
        "next_steps": [
            f"Run `ep run --jobs {jobs_path} --output {results_path}`.",
            f"Run `umriss {workflow} register-results --results {results_path} --prompts {prompts_path}`.",
        ],
    }
    write_json(path.with_name("manifest.json"), manifest)
    return manifest
