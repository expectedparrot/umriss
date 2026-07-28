"""Docs and CLI must not drift: documented commands exist, and commands are documented.

CLI_SPEC.md previously omitted ~25 real commands; its generated command
reference plus this test keep both directions honest from now on.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from umriss.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
DOCS = [REPO / "CLI_SPEC.md", REPO / "README.md", REPO / "docs" / "index.html"]


def registered_command_paths() -> set[str]:
    paths: set[str] = set()

    def walk(node: argparse.ArgumentParser, prefix: list[str]) -> None:
        for action in node._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    subs = [a for a in sub._actions if isinstance(a, argparse._SubParsersAction)]
                    if subs:
                        walk(sub, prefix + [name])
                    else:
                        paths.add(" ".join(prefix + [name]))

    walk(build_parser(), [])
    return paths


def documented_invocations(text: str) -> set[str]:
    """`umriss a [b]` paths from fenced blocks, inline code, and <pre><code> HTML."""
    invocations: list[str] = []
    for block in re.findall(r"```(?:bash|sh|console|text|yaml)?\n(.*?)```", text, re.DOTALL):
        for line in block.splitlines():
            stripped = line.strip().lstrip("$ ").strip()
            if stripped.startswith("umriss "):
                invocations.append(stripped)
    for span in re.findall(r"`([^`\n]+)`", text):
        if span.strip().startswith("umriss "):
            invocations.append(span.strip())
    for block in re.findall(r"<pre[^>]*>(?:<code[^>]*>)?(.*?)(?:</code>)?</pre>", text, re.DOTALL):
        for line in block.splitlines():
            stripped = line.strip().lstrip("$ ").strip()
            if stripped.startswith("umriss "):
                invocations.append(stripped)
    found: set[str] = set()
    for invocation in invocations:
        match = re.match(r"umriss\s+([a-z][a-z0-9-]*)(?:\s+([a-z][a-z0-9-]*))?", invocation)
        if match:
            first, second = match.group(1), match.group(2)
            found.add(first if second is None else f"{first} {second}")
    return found


def test_every_documented_command_exists() -> None:
    registered = registered_command_paths()
    groups = {path.split(" ")[0] for path in registered}
    documented: set[str] = set()
    for doc in DOCS:
        if doc.exists():
            documented |= documented_invocations(doc.read_text())
    problems = []
    for doc_path in sorted(documented):
        first = doc_path.split(" ")[0]
        if doc_path in registered or doc_path in groups:
            continue
        if first in groups:
            group_subs = {p.split(" ")[1] for p in registered if p.startswith(first + " ") and " " in p}
            if not group_subs or (" " in doc_path and doc_path.split(" ")[1] in group_subs):
                continue
            if " " not in doc_path:
                continue
            # second word may be an argument value to a flat command
            if first in registered:
                continue
            problems.append(doc_path)
        else:
            problems.append(doc_path)
    assert problems == [], "docs reference commands the CLI does not register:\n" + "\n".join(problems)


def test_every_registered_command_is_in_spec_reference() -> None:
    spec = (REPO / "CLI_SPEC.md").read_text()
    reference = spec[spec.index("## Command reference"):]
    missing = [
        path for path in sorted(registered_command_paths())
        if f"`umriss {path}`" not in reference
    ]
    assert missing == [], (
        "commands missing from CLI_SPEC.md's command reference:\n" + "\n".join(missing)
    )
