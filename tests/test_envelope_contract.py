"""Black-box output contract: one JSON envelope on stdout, end to end.

Runs the real console entry point in subprocesses (existing tests call main()
in-process), covering stdout purity, error envelopes with the documented exit
statuses, and smoke coverage for previously untested command groups.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from umriss.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
ENVELOPE_KEYS = {"schema_version", "command", "status", "argv", "data", "warnings", "errors", "next_steps"}


def run_umriss(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "umriss", *args],
        cwd=cwd, text=True, capture_output=True,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )


@pytest.mark.parametrize("argv", [
    ("version",),
    ("capabilities",),
    ("guide",),
    ("next",),
])
def test_stdout_is_exactly_one_envelope(argv: tuple[str, ...], tmp_path: Path) -> None:
    completed = run_umriss(*argv, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)  # fails if anything else is on stdout
    assert ENVELOPE_KEYS <= set(payload)
    assert payload["status"] == "ok"
    assert payload["argv"][0] == "umriss"


def test_expected_failures_exit_1_with_error_envelope(tmp_path: Path) -> None:
    completed = run_umriss("battery", "inspect", "missing.json", cwd=tmp_path)
    assert completed.returncode == 1, completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "error"
    assert payload["errors"][0]["code"] == "not_found"


def test_project_group_smoke(tmp_path: Path) -> None:
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    created = run_umriss("project", "create", "demo", "--title", "Demo", "--use", cwd=tmp_path)
    assert created.returncode == 0
    current = run_umriss("project", "current", cwd=tmp_path)
    assert json.loads(current.stdout)["data"]["active_project"] == "demo"
    listed = run_umriss("project", "list", cwd=tmp_path)
    assert any(p["project_id"] == "demo" for p in json.loads(listed.stdout)["data"]["projects"])


def test_marginal_group_smoke(tmp_path: Path) -> None:
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    steps = [
        ("battery", "create", "--battery-id", "demo", "--wave", "T1",
         "--battery", "DEMO", "--topic", "demo", "--context", "demo"),
        ("question", "add", "--battery", "demo", "--item", "q1",
         "--question-stem", "Pick one", "--item-text", "Item one",
         "--option", "Yes", "--option", "No",
         "--option-code", "1", "--option-code", "2",
         "--scale-type", "nominal"),
        ("marginal", "add", "--battery", "demo", "--item", "q1",
         "--proportion", "0.6", "--proportion", "0.4"),
        ("battery", "compile", "--battery", "demo", "--path", "demo_metadata.json"),
    ]
    for step in steps:
        completed = run_umriss(*step, cwd=tmp_path)
        assert completed.returncode == 0, (step, completed.stdout)
        payload = json.loads(completed.stdout)
        assert payload["status"] == "ok"
        assert payload["next_steps"], f"{step[0]} {step[1]} should suggest the next pipeline step"
    assert (tmp_path / "demo_metadata.json").exists()


def test_next_tag_discovers_run_directory(tmp_path: Path) -> None:
    # No manifest anywhere: next --tag falls back to cwd and recommends the
    # first pipeline step rather than pointing at a hard-coded paper layout.
    completed = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout
    data = json.loads(completed.stdout)["data"]
    assert data["stage"] == "build-prompts"
    assert "data/computed_objects" not in json.dumps(data)

    nested = tmp_path / "runs" / "demo_run"
    nested.mkdir(parents=True)
    (nested / "demo_manifest.json").write_text(json.dumps({
        "tag": "demo",
        "prompts": str(nested / "demo_prompts.jsonl"),
        "jobs": str(nested / "demo.jobs.ep"),
        "results": str(nested / "demo.results.ep"),
    }))
    (nested / "demo_prompts.jsonl").write_text("{}\n")
    completed = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout
    data = json.loads(completed.stdout)["data"]
    assert data["stage"] == "export-jobs"


def test_ambiguous_tag_manifests_fail_closed(tmp_path: Path) -> None:
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "demo_manifest.json").write_text(json.dumps({"tag": "demo"}))
    completed = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["errors"][0]["code"] == "ambiguous_tag"
    assert len(payload["errors"][0]["context"]["matches"]) == 2


def test_every_leaf_command_has_handler() -> None:
    problems: list[str] = []

    def walk(node: argparse.ArgumentParser, prefix: list[str]) -> None:
        for action in node._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    subs = [a for a in sub._actions if isinstance(a, argparse._SubParsersAction)]
                    if subs:
                        walk(sub, prefix + [name])
                    elif sub.get_default("func") is None:
                        problems.append(" ".join(prefix + [name]))

    walk(build_parser(), [])
    assert problems == [], "commands without handlers:\n" + "\n".join(problems)
