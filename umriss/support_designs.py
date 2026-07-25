from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .metadata import item_option_labels

SCHEMA_VERSION = 1
COHERENCE_MODES = {"global", "item_specific", "grouped", "explicit"}
DEFAULT_INTENSITIES = ["moderate", "strong"]


def load_design_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) if path.suffix.lower() in {".yaml", ".yml"} else json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a design object: {path}")
    return data


def pattern_coverage_preset(metadata: dict[str, Any], size: int | None, seed: int) -> dict[str, Any]:
    ordinal = all((meta.get("scale") or metadata.get("scale") or {}).get("type") == "ordinal" for meta in metadata["items"].values())
    patterns = (
        [
            {item: item_option_labels(metadata, item)[len(item_option_labels(metadata, item)) // 2] for item in metadata["items"]},
            {item: item_option_labels(metadata, item)[-1] for item in metadata["items"]},
        ]
        if ordinal
        else []
    )
    required = sum(len(item_option_labels(metadata, item)) for item in metadata["items"]) + len(patterns)
    return {
        "schema_version": SCHEMA_VERSION,
        "preset": "pattern-coverage",
        "size": size if size is not None else required,
        "seed": seed,
        "coverage": {"mode": "complete", "allocation": "balanced"},
        "components": [
            {
                "type": "option_coverage",
                "items": "all",
                "minimum_per_option": 1,
                "coherence": "item_specific",
                "intensity": {"values": DEFAULT_INTENSITIES, "allocation": "balanced"},
            },
            *([{"type": "pattern_anchors", "patterns": patterns, "coherence": "explicit"}] if patterns else []),
        ],
        "profile": {
            "framing": "synthetic_respondent",
            "forbid_demographic_invention": True,
        },
        "probabilities": {
            "interpretation": "subjective_response_probability",
            "require_sum": 1,
            "allow_zero": False,
            "minimum_probability": 0.01,
        },
        "guardrails": {
            "leak_target_marginals": False,
            "silent_coverage_truncation": False,
            "silent_invalid_response_repair": False,
            "describe_as_recovered_respondents": False,
        },
    }


def uniform_patterns_preset(metadata: dict[str, Any], size: int | None, seed: int) -> dict[str, Any]:
    pattern_count = 1
    for item in metadata["items"]:
        pattern_count *= len(item_option_labels(metadata, item))
    resolved_size = size if size is not None else max(96, pattern_count)
    return {
        "schema_version": SCHEMA_VERSION,
        "preset": "uniform-patterns",
        "size": resolved_size,
        "seed": seed,
        "coverage": {"mode": "complete", "allocation": "balanced"},
        "components": [{"type": "uniform_patterns", "coherence": "explicit"}],
        "profile": {
            "framing": "synthetic_respondent",
            "forbid_demographic_invention": True,
        },
        "probabilities": {
            "interpretation": "subjective_response_probability",
            "require_sum": 1,
            "allow_zero": False,
            "minimum_probability": 0.01,
        },
        "guardrails": {
            "leak_target_marginals": False,
            "silent_coverage_truncation": False,
            "silent_invalid_response_repair": False,
            "describe_as_recovered_respondents": False,
        },
    }


def preset_design(metadata: dict[str, Any], preset: str, size: int | None, seed: int) -> dict[str, Any]:
    if preset == "pattern-coverage":
        return pattern_coverage_preset(metadata, size, seed)
    if preset == "uniform-patterns":
        return uniform_patterns_preset(metadata, size, seed)
    raise ValueError(f"Unsupported preset: {preset}.")


def _component_type(component: dict[str, Any]) -> str:
    return str(component.get("type", "")).replace("-", "_")


def _coverage_items(component: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    raw = component.get("items", "all")
    items = list(metadata["items"]) if raw == "all" else list(raw) if isinstance(raw, list) else []
    unknown = [item for item in items if item not in metadata["items"]]
    if not items or unknown:
        raise ValueError(f"Invalid option-coverage items: {unknown or raw}.")
    return items


def _coverage_cells(component: dict[str, Any], metadata: dict[str, Any]) -> list[tuple[str, str]]:
    minimum = int(component.get("minimum_per_option", 1))
    if minimum < 1:
        raise ValueError("minimum_per_option must be at least 1.")
    return [
        (item, option)
        for item in _coverage_items(component, metadata)
        for option in item_option_labels(metadata, item)
        for _ in range(minimum)
    ]


def _required_rows(design: dict[str, Any], metadata: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    detail = []
    for component in design["components"]:
        ctype = _component_type(component)
        if ctype == "option_coverage":
            count = len(_coverage_cells(component, metadata))
        elif ctype == "pattern_anchors":
            count = len(component.get("patterns", []))
        elif ctype == "profiles":
            count = len(component.get("profiles", []))
        elif ctype == "uniform_patterns":
            count = int(design["size"])
        else:
            raise ValueError(f"Unsupported component type: {ctype}.")
        total += count
        detail.append({"component": ctype, "required": count})
    return total, detail


def validate_design(design: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    if design.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Design schema_version must be {SCHEMA_VERSION}.")
    if not isinstance(design.get("components"), list) or not design["components"]:
        raise ValueError("Design components must be a non-empty list.")
    size = int(design.get("size", 0))
    if size < 1:
        raise ValueError("Design size must be at least 1.")
    coverage = design.get("coverage", {})
    mode = coverage.get("mode", "complete")
    if mode not in {"complete", "partial"}:
        raise ValueError("coverage.mode must be complete or partial.")
    for component in design["components"]:
        coherence = component.get("coherence", "item_specific")
        if coherence not in COHERENCE_MODES:
            raise ValueError(f"Unsupported coherence mode: {coherence}.")
        ctype = _component_type(component)
        if ctype == "option_coverage":
            _coverage_cells(component, metadata)
            intensity = component.get("intensity", {})
            values = intensity.get("values", DEFAULT_INTENSITIES)
            if not isinstance(values, list) or not values:
                raise ValueError("intensity.values must be a non-empty list.")
            if intensity.get("allocation", "balanced") != "balanced":
                raise ValueError("Only balanced intensity allocation is supported in schema v1.")
        elif ctype == "pattern_anchors":
            patterns = component.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise ValueError("pattern_anchors requires a non-empty patterns list.")
            for pattern in patterns:
                if set(pattern) != set(metadata["items"]):
                    raise ValueError("Every anchor pattern must specify exactly every battery item.")
                for item, option in pattern.items():
                    if option not in item_option_labels(metadata, item):
                        raise ValueError(f"Unknown option {option!r} for item {item}.")
        elif ctype == "profiles":
            if not isinstance(component.get("profiles"), list) or not component["profiles"]:
                raise ValueError("profiles requires a non-empty profiles list.")
        elif ctype == "uniform_patterns":
            pass
        else:
            raise ValueError(f"Unsupported component type: {ctype}.")
    required, detail = _required_rows(design, metadata)
    if mode == "complete" and size < required:
        breakdown = ", ".join(f"{row['component']}: {row['required']}" for row in detail)
        raise ValueError(
            f"DESIGN_TOO_SMALL: requested {size} support points; declared coverage requires {required} "
            f"({breakdown}). Increase size or explicitly set coverage.mode to partial."
        )
    prompt = design.get("prompt", {})
    if prompt.get("template") and prompt.get("validation", "strict") != "strict":
        raise ValueError("Custom prompt templates require prompt.validation: strict.")
    if "summary_field" in design.get("profile", {}):
        raise ValueError("profile.summary_field is unsupported; support output uses the fixed second-person `persona` field.")
    return {"valid": True, "size": size, "required": required, "components": detail, "coverage_mode": mode}


def resolve_design(design: dict[str, Any], metadata: dict[str, Any], *, size: int | None = None, seed: int | None = None) -> dict[str, Any]:
    resolved = json.loads(json.dumps(design))
    if size is not None:
        resolved["size"] = size
    if seed is not None:
        resolved["seed"] = seed
    resolved.setdefault("coverage", {"mode": "complete", "allocation": "balanced"})
    resolved.setdefault("profile", {"framing": "synthetic_respondent", "forbid_demographic_invention": True})
    resolved.setdefault(
        "probabilities",
        {"interpretation": "subjective_response_probability", "require_sum": 1, "allow_zero": False, "minimum_probability": 0.01},
    )
    resolved.setdefault("guardrails", {})
    validate_design(resolved, metadata)
    return resolved


def _item_lines(metadata: dict[str, Any]) -> str:
    return "\n".join(
        f"- {item}: {meta['item_text']}\n  Options, in output order: "
        + "; ".join(f"{idx + 1}. {label}" for idx, label in enumerate(item_option_labels(metadata, item)))
        for item, meta in metadata["items"].items()
    )


def _schema(metadata: dict[str, Any]) -> str:
    probabilities = ",\n".join(f'    "{item}": [numbers in option order]' for item in metadata["items"])
    return '{\n  "persona": "Your views on ...",\n  "probabilities": {\n' + probabilities + "\n  }\n}"


def _coherence_instruction(row: dict[str, Any], design: dict[str, Any]) -> str:
    mode = row["coherence"]
    if mode == "item_specific":
        return "The targeted item may differ from the other items. Do not infer a uniform outlook across the battery."
    if mode == "global":
        return "Use one deliberately global outlook across the battery."
    if mode == "grouped":
        groups = row.get("groups") or design.get("item_groups", {})
        return f"Apply coherence within these declared item groups, not automatically across groups: {json.dumps(groups, sort_keys=True)}"
    return "Follow the complete response pattern explicitly supplied below."


def _default_prompt(metadata: dict[str, Any], row: dict[str, Any], design: dict[str, Any]) -> str:
    profile = design["profile"]
    probs = design["probabilities"]
    target = ""
    if row["design_type"] == "option_coverage":
        item = row["coverage_item"]
        target = (
            f'Construct a synthetic survey-response profile for whom "{row["coverage_option"]}" is a plausible '
            f'response to {item}: {metadata["items"][item]["item_text"]}.\n'
            f'Response intensity: {row["intensity"]}.'
        )
    elif row["design_type"] == "pattern_anchor":
        lines = "\n".join(f"- {item}: {option}" for item, option in row["pattern"].items())
        target = f"Construct a synthetic survey-response profile organized around this declared answer pattern:\n{lines}"
    else:
        target = f"Construct this declared synthetic survey-response profile:\n{row['profile']}"
    demographic = (
        "Do not invent demographic characteristics unless the design explicitly supplies them."
        if profile.get("forbid_demographic_invention", True)
        else ""
    )
    minimum = float(probs.get("minimum_probability", 0.0))
    return f"""Support point identifier: {row['job_id']}

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

{target}

{_coherence_instruction(row, design)}
{demographic}

Write a concise persona describing the attitudes that distinguish this response pattern. Address the persona in the
second person, beginning with "Your views..." or "You...". This text will become a visible EDSL agent trait, so include
only the persona description—not instructions about answering questions or probabilities. Then provide subjective response
probabilities for every item. Each vector must follow the displayed option order, contain nonnegative numbers, and sum to
1. Each probability must be at least {minimum:g}.

Items and response options:
{_item_lines(metadata)}

Return only valid JSON with exactly this schema:
{_schema(metadata)}"""


def _custom_prompt(template_path: Path, metadata: dict[str, Any], row: dict[str, Any], design: dict[str, Any]) -> str:
    try:
        from jinja2 import Environment, StrictUndefined
    except ImportError as exc:
        raise ValueError("Custom templates require Jinja2.") from exc
    template = Environment(undefined=StrictUndefined, autoescape=False).from_string(template_path.read_text())
    prompt_metadata = {key: value for key, value in metadata.items() if key != "truth"}
    try:
        prompt = template.render(metadata=prompt_metadata, support=row, design=design, items_text=_item_lines(metadata))
    except Exception as exc:
        raise ValueError(f"Custom prompt rendering failed: {exc}") from exc
    required = [*metadata["items"], "persona", "second person", "probabil", "sum to 1"]
    missing = [token for token in required if token.lower() not in prompt.lower()]
    if missing:
        raise ValueError(f"Custom prompt failed strict validation; missing: {', '.join(missing)}.")
    return prompt


def compile_support_plan(metadata: dict[str, Any], tag: str, design: dict[str, Any], design_path: Path | None = None) -> list[dict[str, Any]]:
    validate_design(design, metadata)
    size = int(design["size"])
    partial = design["coverage"].get("mode") == "partial"
    rows: list[dict[str, Any]] = []
    components = sorted(design["components"], key=lambda c: 0 if _component_type(c) == "option_coverage" else 1)
    for component in components:
        ctype = _component_type(component)
        if ctype == "option_coverage":
            cells = _coverage_cells(component, metadata)
            if partial:
                cells = cells[: max(0, size - len(rows))]
            intensities = [str(x) for x in component.get("intensity", {}).get("values", DEFAULT_INTENSITIES)]
            for idx, (item, option) in enumerate(cells):
                rows.append(
                    {
                        "design_type": "option_coverage",
                        "reason": f"cover {item} = {option}",
                        "coverage_item": item,
                        "coverage_option": option,
                        "intensity": intensities[idx % len(intensities)],
                        "coherence": component.get("coherence", "item_specific"),
                        "groups": component.get("groups"),
                    }
                )
        elif ctype == "pattern_anchors":
            for pattern in component["patterns"]:
                rows.append(
                    {
                        "design_type": "pattern_anchor",
                        "reason": "declared response-pattern anchor",
                        "pattern": pattern,
                        "coherence": component.get("coherence", "explicit"),
                    }
                )
        elif ctype == "profiles":
            for profile in component["profiles"]:
                rows.append(
                    {
                        "design_type": "profile",
                        "reason": "user-declared profile",
                        "profile": profile,
                        "coherence": component.get("coherence", "item_specific"),
                    }
                )
        elif ctype == "uniform_patterns":
            items = list(metadata["items"])
            patterns = list(itertools.product(*(item_option_labels(metadata, item) for item in items)))
            if size % len(patterns):
                raise ValueError(
                    f"uniform_patterns size must be a multiple of the full response-pattern count ({len(patterns)})."
                )
            for repeat in range(size // len(patterns)):
                for values in patterns:
                    rows.append(
                        {
                            "design_type": "pattern_anchor",
                            "reason": f"uniform full-pattern replicate {repeat + 1}",
                            "pattern": dict(zip(items, values, strict=True)),
                            "coherence": "explicit",
                        }
                    )
    rows = rows[:size]
    coverage_components = [component for component in components if _component_type(component) == "option_coverage"]
    fill_index = 0
    while len(rows) < size and coverage_components:
        component = coverage_components[fill_index % len(coverage_components)]
        cells = _coverage_cells(component, metadata)
        item, option = cells[(fill_index // len(coverage_components)) % len(cells)]
        intensities = [str(x) for x in component.get("intensity", {}).get("values", DEFAULT_INTENSITIES)]
        rows.append(
            {
                "design_type": "option_coverage",
                "reason": f"additional coverage for {item} = {option}",
                "coverage_item": item,
                "coverage_option": option,
                "intensity": intensities[fill_index % len(intensities)],
                "coherence": component.get("coherence", "item_specific"),
                "groups": component.get("groups"),
            }
        )
        fill_index += 1
    if len(rows) < size:
        raise ValueError(f"Design declares size {size}, but its components produce only {len(rows)} rows.")
    template = design.get("prompt", {}).get("template")
    if template:
        base = design_path.parent if design_path else Path.cwd()
        template_path = (base / template).resolve()
        design["prompt"]["template_sha256"] = hashlib.sha256(template_path.read_bytes()).hexdigest()
    for sid, row in enumerate(rows, 1):
        row.update(
            {
                "support_id": sid,
                "job_id": f"{tag}_{sid:03d}",
                "battery": f"{metadata['wave']}_{metadata['battery']}",
                "design_schema_version": SCHEMA_VERSION,
            }
        )
        row["prompt"] = _custom_prompt(template_path, metadata, row, design) if template else _default_prompt(metadata, row, design)
    return rows


def coverage_report(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    counts = Counter((row.get("coverage_item"), row.get("coverage_option")) for row in rows if row["design_type"] == "option_coverage")
    for row in rows:
        for item, option in (row.get("pattern") or {}).items():
            counts[(item, option)] += 1
    return [
        {
            "item": item,
            "option": option,
            "covered": counts[(item, option)] > 0,
            "support_rows": counts[(item, option)],
        }
        for item in metadata["items"]
        for option in item_option_labels(metadata, item)
    ]


def _write_prompt_html(rows: list[dict[str, Any]], path: Path, tag: str) -> None:
    cards = "\n".join(
        f"<details><summary>Support point {row['support_id']}: {html.escape(row['reason'])}</summary>"
        f"<pre>{html.escape(row['prompt'])}</pre></details>"
        for row in rows
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        f"<title>{html.escape(tag)} prompts</title><style>body{{font:16px/1.55 system-ui;max-width:960px;margin:3rem auto;"
        "padding:0 1rem;color:#18211d}}details{border:1px solid #ccd7d0;border-radius:8px;padding:1rem;margin:1rem 0}"
        "summary{font-weight:650;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f5;padding:1rem;"
        "border-radius:6px}</style>"
        f"<h1>{html.escape(tag)} prompts</h1><p>Review every consequential prompt before model execution.</p>{cards}"
    )


def write_support_outputs(
    rows: list[dict[str, Any]], metadata: dict[str, Any], tag: str, out_dir: Path, design: dict[str, Any]
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = out_dir / f"{tag}_resolved_design.yaml"
    plan_path = out_dir / f"{tag}_support_plan.csv"
    coverage_path = out_dir / f"{tag}_coverage.csv"
    prompts_path = out_dir / f"{tag}_prompts.jsonl"
    html_path = out_dir / f"{tag}_prompts.html"
    resolved_path.write_text(yaml.safe_dump(design, sort_keys=False))
    with prompts_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    fields = [
        "support_id",
        "job_id",
        "design_type",
        "reason",
        "coverage_item",
        "coverage_option",
        "intensity",
        "coherence",
        "pattern",
        "profile",
    ]
    with plan_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "pattern": json.dumps(row.get("pattern", {}), sort_keys=True)})
    coverage = coverage_report(rows, metadata)
    with coverage_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "option", "covered", "support_rows"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage)
    _write_prompt_html(rows, html_path, tag)
    duplicate_prompts = len(rows) - len({row["prompt"] for row in rows})
    return {
        "resolved_design_path": str(resolved_path),
        "support_plan_path": str(plan_path),
        "coverage_path": str(coverage_path),
        "prompts_path": str(prompts_path),
        "prompts_html_path": str(html_path),
        "coverage_complete": all(row["covered"] for row in coverage),
        "omitted_coverage_cells": sum(not row["covered"] for row in coverage),
        "anchors": sum(row["design_type"] == "pattern_anchor" for row in rows),
        "duplicate_prompts": duplicate_prompts,
    }
