from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import UmrissError
from .jsonlio import read_json, write_json

ROOT = Path(".umriss")
SCHEMA_VERSION = 1
DEFAULT_PROJECT_ID = "default"
PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_project_id(project_id: str) -> str:
    if not PROJECT_ID_RE.match(project_id):
        raise UmrissError(
            "invalid_project_id",
            f"Invalid project id: {project_id}.",
            hint="Use letters, numbers, dots, underscores, or hyphens; start with a letter or number.",
        )
    return project_id


def require_workspace() -> None:
    if not ROOT.exists():
        raise UmrissError("not_initialized", "No .umriss directory found.", hint="Run `umriss init`.")


def projects_dir() -> Path:
    return ROOT / "projects"


def head_path() -> Path:
    return ROOT / "HEAD"


def project_dir(project_id: str) -> Path:
    return projects_dir() / validate_project_id(project_id)


def active_project_id() -> str:
    require_workspace()
    env_project = os.environ.get("UMRISS_PROJECT")
    if env_project:
        project_id = validate_project_id(env_project.strip())
    elif head_path().exists():
        project_id = validate_project_id(head_path().read_text().strip())
    else:
        raise UmrissError("not_initialized", "No active umriss project is set.", hint="Run `umriss init`.")
    if not project_dir(project_id).exists():
        raise UmrissError("not_found", f"Active project does not exist: {project_id}.")
    return project_id


def active_project_dir() -> Path:
    return project_dir(active_project_id())


def init_workspace(output_dir: str = "umriss_work") -> dict[str, Any]:
    ROOT.mkdir(exist_ok=True)
    projects_dir().mkdir(parents=True, exist_ok=True)
    write_json(
        ROOT / "config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "output_dir": output_dir,
            "default_model": "gpt-5.5",
            "default_rho": [0.0003, 0.001, 0.003, 0.01, 0.03],
        },
    )
    create_project(DEFAULT_PROJECT_ID, title="Default umriss project", use=True)
    return {
        "path": str(ROOT),
        "schema_version": SCHEMA_VERSION,
        "active_project": DEFAULT_PROJECT_ID,
        "project_path": str(project_dir(DEFAULT_PROJECT_ID)),
    }


def create_project(project_id: str, title: str | None = None, use: bool = False) -> dict[str, Any]:
    project_id = validate_project_id(project_id)
    pdir = project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    for name in ["batteries", "designs", "runs", "support_prompts", "support_raw", "support_banks", "fits", "evaluations", "comparisons", "reports", "workflows"]:
        (pdir / name).mkdir(exist_ok=True)
    project = {
        "project_id": project_id,
        "title": title or project_id,
        "created_at": utc_now(),
        "schema_version": SCHEMA_VERSION,
    }
    write_json(pdir / "project.json", project)
    if not (pdir / "batteries.json").exists():
        write_json(pdir / "batteries.json", {"batteries": []})
    if use:
        head_path().write_text(project_id + "\n")
    return {"project": project, "project_path": str(pdir)}


def list_projects() -> list[dict[str, Any]]:
    require_workspace()
    rows = []
    for pdir in sorted(projects_dir().iterdir()):
        if pdir.is_dir() and (pdir / "project.json").exists():
            rows.append(read_json(pdir / "project.json"))
    return rows


def use_project(project_id: str) -> dict[str, Any]:
    pdir = project_dir(project_id)
    if not pdir.exists():
        raise UmrissError("not_found", f"Project does not exist: {project_id}.")
    head_path().write_text(validate_project_id(project_id) + "\n")
    return {"active_project": project_id, "project_path": str(pdir)}


def battery_dir(battery_id: str, project_id: str | None = None) -> Path:
    pdir = project_dir(project_id) if project_id else active_project_dir()
    return pdir / "batteries" / battery_id


def designs_dir(project_id: str | None = None) -> Path:
    pdir = project_dir(project_id) if project_id else active_project_dir()
    d = pdir / "designs"
    d.mkdir(exist_ok=True)  # projects created before designs existed
    return d


def design_path(design_id: str) -> Path:
    return designs_dir() / f"{validate_project_id(design_id)}.yaml"


def list_battery_ids() -> list[str]:
    batteries = active_project_dir() / "batteries"
    if not batteries.exists():
        return []
    return sorted(p.name for p in batteries.iterdir() if (p / "battery.json").exists())


def list_design_ids() -> list[str]:
    return sorted(p.stem for p in designs_dir().glob("*.yaml"))


def run_dir(tag: str, create: bool = False) -> Path:
    d = active_project_dir() / "runs" / validate_project_id(tag)
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def list_run_tags() -> list[str]:
    runs = active_project_dir() / "runs"
    if not runs.exists():
        return []
    return sorted(p.name for p in runs.iterdir() if p.is_dir())


def get_defaults() -> dict[str, str]:
    path = active_project_dir() / "defaults.json"
    return read_json(path) if path.exists() else {}


def set_default(kind: str, value: str) -> dict[str, str]:
    defaults = get_defaults()
    defaults[kind] = value
    write_json(active_project_dir() / "defaults.json", defaults)
    return defaults
