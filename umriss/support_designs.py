from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metadata import item_option_labels

COVERAGE_STYLES = [
    "high-conviction but not deterministic",
    "moderate and somewhat ambivalent",
    "selective, applying the target response only where the item wording fits",
    "response-scale calibrated, avoiding extremes unless strongly justified",
    "cross-pressured, with mixed views across nearby items",
]

AGES = ["an 18-29 year old", "a 30-49 year old", "a 50-64 year old", "a 65+ year old"]
EDUCATION = ["with a high school education or less", "with some college or an associate degree", "with a bachelor's degree or more"]
OCCUPATION = [
    "working in a hands-on trade or manual occupation",
    "working in service, care, retail, or hospitality",
    "working in office, professional, education, or management work",
    "not currently employed or retired",
]
POLITICS = ["politically conservative", "politically moderate", "politically liberal"]
TECH = ["low exposure to advanced technology and AI tools", "moderate exposure to workplace technology", "high exposure to digital tools, analytics, or AI"]


def item_lines(metadata: dict[str, Any]) -> str:
    lines = []
    for item, meta in metadata["items"].items():
        labels = item_option_labels(metadata, item)
        lines.append(f"- {item}: {meta['item_text']}\n  Options: " + "; ".join(f"{idx + 1}. {label}" for idx, label in enumerate(labels)))
    return "\n".join(lines)


def schema_lines(metadata: dict[str, Any]) -> str:
    return ",\n".join(f'    "{item}": [numbers in option order]' for item in metadata["items"])


def first_question_stem(metadata: dict[str, Any]) -> str:
    return next(iter(metadata["items"].values()))["question_stem"]


def pattern_prompt(metadata: dict[str, Any], support_id: int, pattern: dict[str, str], style: str = "") -> str:
    pattern_lines = "\n".join(f"- {item}: {metadata['items'][item]['item_text']} -> {pattern[item]}" for item in metadata["items"])
    return f"""You are creating one synthetic support point for a marginal weighting exercise.

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

This support point is not a real person and not a demographic population
segment. It is a measured behavioral basis function. The target answer pattern
below is a scaffold: construct a coherent respondent type whose experiences,
beliefs, and response style make the pattern feel natural rather than random.

Target answer-pattern scaffold:
{pattern_lines}

Decision style:
{style or "Use ordinary survey-response uncertainty."}

Question stem:
{first_question_stem(metadata)}

Items and response options:
{item_lines(metadata)}

Task:
Return calibrated probabilities over the listed response options for this
support point on every item. The scaffold should guide the direction of each
item, but do not make probabilities exactly 0 or 1. Keep the profile coherent
across items and allow middle probabilities for mixed views.

Return only valid JSON with exactly this schema:
{{
  "persona": "coherent response-pattern support point {support_id}",
  "probabilities": {{
{schema_lines(metadata)}
  }}
}}"""


def coverage_prompt(metadata: dict[str, Any], support_id: int, item: str, option: str, style: str = "") -> str:
    item_text = metadata["items"][item]["item_text"]
    return f"""You are creating one synthetic support point for a marginal weighting exercise.

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

This support point is not a real person and not a demographic population
segment. It is a measured behavioral basis function.

Response-region target:
Create a coherent respondent type for whom the response option "{option}" is
natural on this item:
{item}: {item_text}

Decision style:
{style or "Use ordinary survey-response uncertainty."}

Apply the same underlying outlook consistently across the whole battery, but do
not force "{option}" on every item. The point of this support row is to make
sure the bank has measured coverage of that response region when it is
behaviorally plausible.

Question stem:
{first_question_stem(metadata)}

Items and response options:
{item_lines(metadata)}

Task:
Return calibrated probabilities over the listed response options for this
support point on every item. Avoid exact 0/1 probabilities.

Return only valid JSON with exactly this schema:
{{
  "persona": "response-option coverage support point {support_id}",
  "probabilities": {{
{schema_lines(metadata)}
  }}
}}"""


def moderate_pattern(metadata: dict[str, Any]) -> dict[str, str]:
    return {item: item_option_labels(metadata, item)[0 if len(item_option_labels(metadata, item)) == 2 else len(item_option_labels(metadata, item)) // 2] for item in metadata["items"]}


def anti_modal_pattern(metadata: dict[str, Any]) -> dict[str, str]:
    out = {}
    for item in metadata["items"]:
        labels = item_option_labels(metadata, item)
        out[item] = labels[min(len(labels) - 1, max(0, len(labels) // 2 + 1))]
    return out


def random_pattern(metadata: dict[str, Any], rng: np.random.Generator, center_weight: float = 0.0) -> dict[str, str]:
    pattern = {}
    for item in metadata["items"]:
        labels = item_option_labels(metadata, item)
        if center_weight > 0 and len(labels) > 2:
            weights = np.ones(len(labels), dtype=float)
            center = len(labels) // 2
            weights[center] += center_weight
            if center - 1 >= 0:
                weights[center - 1] += center_weight / 2
            if center + 1 < len(labels):
                weights[center + 1] += center_weight / 2
            weights /= weights.sum()
            pattern[item] = str(rng.choice(labels, p=weights))
        else:
            pattern[item] = str(rng.choice(labels))
    return pattern


def coverage_targets(metadata: dict[str, Any], max_rows: int) -> list[tuple[str, str]]:
    targets = []
    for item in metadata["items"]:
        labels = item_option_labels(metadata, item)
        indices = [0, len(labels) - 1]
        if len(labels) > 2:
            indices.extend([len(labels) // 2, max(0, len(labels) // 2 - 1)])
        for idx in dict.fromkeys(indices):
            targets.append((item, labels[idx]))
    if len(targets) <= max_rows:
        return targets
    stride = len(targets) / max_rows
    selected = []
    used = set()
    for pos in range(max_rows):
        idx = int(pos * stride)
        if idx not in used:
            selected.append(targets[idx])
            used.add(idx)
    return selected


def build_pattern_coverage(metadata: dict[str, Any], tag: str, seed: int, n_support: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    items = list(metadata["items"])
    d = sum(len(item_option_labels(metadata, item)) - 1 for item in items)
    n_coverage = min(max(12, len(items) * 2), max(12, n_support // 4))
    max_patterns = 1
    for item in items:
        max_patterns *= len(item_option_labels(metadata, item))
    n_patterns = min(n_support - n_coverage, max_patterns)
    rows = []
    seen: set[tuple[str, ...]] = set()
    for pattern in [moderate_pattern(metadata), anti_modal_pattern(metadata)]:
        key = tuple(pattern[item] for item in items)
        if key in seen:
            continue
        seen.add(key)
        sid = len(rows) + 1
        rows.append({"job_id": f"{tag}_{sid:03d}", "battery": f"{metadata['wave']}_{metadata['battery']}", "support_id": sid, "variant": "pattern_anchor", "moment_dimension": d, "pattern": pattern, "prompt": pattern_prompt(metadata, sid, pattern)})
    attempts = 0
    while len(rows) < n_patterns and attempts < max(5000, n_patterns * 500):
        attempts += 1
        center_weight = 3.0 if len(rows) % 3 == 0 else 0.0
        pattern = random_pattern(metadata, rng, center_weight)
        key = tuple(pattern[item] for item in items)
        if key in seen:
            continue
        seen.add(key)
        sid = len(rows) + 1
        rows.append({"job_id": f"{tag}_{sid:03d}", "battery": f"{metadata['wave']}_{metadata['battery']}", "support_id": sid, "variant": "uniform_pattern" if center_weight == 0 else "center_weighted_pattern", "moment_dimension": d, "pattern": pattern, "prompt": pattern_prompt(metadata, sid, pattern)})
    for item, option in coverage_targets(metadata, n_support - len(rows)):
        sid = len(rows) + 1
        style = COVERAGE_STYLES[(sid - 1) % len(COVERAGE_STYLES)]
        rows.append({"job_id": f"{tag}_{sid:03d}", "battery": f"{metadata['wave']}_{metadata['battery']}", "support_id": sid, "variant": "option_coverage", "moment_dimension": d, "coverage_item": item, "coverage_option": option, "coverage_style": style, "prompt": coverage_prompt(metadata, sid, item, option, style)})
        if len(rows) >= n_support:
            break
    all_targets = coverage_targets(metadata, 100000)
    fill_idx = 0
    while len(rows) < n_support:
        item, option = all_targets[fill_idx % len(all_targets)]
        style = COVERAGE_STYLES[(fill_idx // len(all_targets)) % len(COVERAGE_STYLES)]
        sid = len(rows) + 1
        rows.append({"job_id": f"{tag}_{sid:03d}", "battery": f"{metadata['wave']}_{metadata['battery']}", "support_id": sid, "variant": "option_coverage_repeat", "moment_dimension": d, "coverage_item": item, "coverage_option": option, "coverage_style": style, "prompt": coverage_prompt(metadata, sid, item, option, style)})
        fill_idx += 1
    return rows[:n_support]


def archetypes(limit: int) -> list[str]:
    combos = list(itertools.product(AGES, EDUCATION, OCCUPATION, POLITICS, TECH))
    step = max(1, len(combos) // limit)
    selected = combos[::step][:limit]
    return [f"{age} U.S. adult {edu}, {occ}, {pol}, with {tech}" for age, edu, occ, pol, tech in selected]


def archetype_prompt(metadata: dict[str, Any], persona: str) -> str:
    return f"""You are predicting how one synthetic support type would answer a survey battery.

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

Support type:
{persona}

Question stem:
{first_question_stem(metadata)}

Items:
{item_lines(metadata)}

Task:
For this support type, return calibrated probabilities over the response options
listed for each item. Some items have different response scales, so each vector
must have the same length as that item's option list. Keep the answers
internally coherent across items. These are not individual records; they are
support-point response distributions for a later marginal weighting exercise.

Return only valid JSON with exactly this schema:
{{
  "persona": "{persona}",
  "probabilities": {{
{schema_lines(metadata)}
  }}
}}"""


def build_archetype(metadata: dict[str, Any], tag: str, n_support: int) -> list[dict[str, Any]]:
    return [
        {"job_id": f"{tag}_{sid:03d}", "battery": f"{metadata['wave']}_{metadata['battery']}", "support_id": sid, "n_support": n_support, "persona": persona, "prompt": archetype_prompt(metadata, persona)}
        for sid, persona in enumerate(archetypes(n_support), start=1)
    ]


def load_design_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ValueError("YAML design files require PyYAML; use JSON or install PyYAML.") from exc
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected design object: {path}")
    return data


def validate_axes(axes: dict[str, Any]) -> dict[str, list[str]]:
    if not isinstance(axes, dict) or not axes:
        raise ValueError("Design axes must be a non-empty object.")
    out: dict[str, list[str]] = {}
    for name, levels in axes.items():
        if not isinstance(levels, list) or not levels:
            raise ValueError(f"Axis {name} must have at least one level.")
        out[str(name)] = [str(level) for level in levels]
    return out


def one_hot(row: tuple[int, ...], level_counts: list[int]) -> np.ndarray:
    parts = []
    for value, n_levels in zip(row, level_counts):
        vec = np.zeros(n_levels)
        vec[value] = 1
        parts.append(vec)
    return np.concatenate(parts)


def maximin_axis_rows(axes: dict[str, list[str]], n: int, seed: int) -> list[tuple[int, ...]]:
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    level_counts = [len(levels) for levels in axes.values()]
    candidates = list(itertools.product(*[range(k) for k in level_counts]))
    if not candidates:
        return []
    X = np.vstack([one_hot(row, level_counts) for row in candidates])
    first = int(rng.integers(len(candidates)))
    selected = [first]
    min_dist = np.linalg.norm(X - X[first], axis=1)
    min_dist[first] = -np.inf
    while len(selected) < min(n, len(candidates)):
        max_dist = min_dist.max()
        tied = np.flatnonzero(np.isclose(min_dist, max_dist))
        nxt = int(rng.choice(tied))
        selected.append(nxt)
        dist = np.linalg.norm(X - X[nxt], axis=1)
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf
    return [candidates[idx] for idx in selected]


def sample_axis_rows(axes: dict[str, list[str]], sampler: dict[str, Any], n_support: int, seed: int) -> list[tuple[int, ...]]:
    method = str(sampler.get("method", "maximin")).replace("_", "-")
    n = int(sampler.get("n", n_support))
    if n <= 0:
        return []
    local_seed = int(sampler.get("seed", seed))
    level_counts = [len(levels) for levels in axes.values()]
    candidates = list(itertools.product(*[range(k) for k in level_counts]))
    if method == "full-factorial":
        return candidates[:n]
    if method == "random":
        rng = np.random.default_rng(local_seed)
        replace = len(candidates) < n
        picks = rng.choice(len(candidates), size=n, replace=replace)
        return [candidates[int(idx)] for idx in picks]
    if method == "maximin":
        return maximin_axis_rows(axes, n, local_seed)
    raise ValueError(f"Unsupported axis sampler method: {method}.")


def persona_from_axes(row: tuple[int, ...], axes: dict[str, list[str]]) -> str:
    return "; ".join(f"{name.replace('_', ' ')}: {levels[idx]}" for idx, (name, levels) in zip(row, axes.items()))


def axis_prompt(metadata: dict[str, Any], support_id: int, persona: str, guidance: str = "") -> str:
    guidance_text = f"\nCalibration guidance:\n{guidance}\n" if guidance else ""
    return f"""You are predicting how one synthetic support type would answer a survey battery.

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

This support type is defined by a compact design over latent, demographic, or
response-style axes. Treat each axis value as a source of variation in how the
support type would evaluate these survey items.

Support type {support_id}:
{persona}
{guidance_text}
Question stem:
{first_question_stem(metadata)}

Items:
{item_lines(metadata)}

Task:
For this support type, return calibrated probabilities over the response
options listed for each item. Keep the answers internally coherent across
items, and use the support type to create real variation across the battery.
These are not individual records; they are support-point response distributions
for a later marginal weighting exercise.

Return only valid JSON with exactly this schema:
{{
  "persona": "{persona}",
  "probabilities": {{
{schema_lines(metadata)}
  }}
}}"""


def _component_rows(config: dict[str, Any], n_support: int, seed: int) -> list[dict[str, Any]]:
    if "components" in config:
        components = config["components"]
        if not isinstance(components, list) or not components:
            raise ValueError("Design components must be a non-empty list.")
        return [dict(component) for component in components]
    if "axes" in config:
        return [{"type": "axes", "name": config.get("name", "axis_design"), "axes": config["axes"], "sampler": config.get("sampler", {}), "guidance": config.get("guidance", "")}]
    return [{"type": "option-coverage", "n": n_support, "styles": config.get("styles", COVERAGE_STYLES)}]


def build_from_design_config(metadata: dict[str, Any], tag: str, config: dict[str, Any], n_support: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    moment_dimension = sum(len(item_option_labels(metadata, item)) - 1 for item in metadata["items"])
    for component in _component_rows(config, n_support, seed):
        if len(rows) >= n_support:
            return rows[:n_support]
        ctype = str(component.get("type", "axes")).replace("_", "-")
        name = str(component.get("name", ctype))
        if ctype == "axes":
            axes = validate_axes(component.get("axes", {}))
            sampler = component.get("sampler", {})
            if not isinstance(sampler, dict):
                raise ValueError("Axis component sampler must be an object.")
            for row in sample_axis_rows(axes, sampler, n_support - len(rows), seed):
                sid = len(rows) + 1
                persona = persona_from_axes(row, axes)
                axis_values = {axis: levels[idx] for idx, (axis, levels) in zip(row, axes.items())}
                rows.append(
                    {
                        "job_id": f"{tag}_{sid:03d}",
                        "battery": f"{metadata['wave']}_{metadata['battery']}",
                        "support_id": sid,
                        "variant": name,
                        "design_type": "axes",
                        "axis_values": axis_values,
                        "persona": persona,
                        "prompt": axis_prompt(metadata, sid, persona, str(component.get("guidance", config.get("guidance", "")))),
                    }
                )
                if len(rows) >= n_support:
                    return rows
        elif ctype in {"patterns", "pattern"}:
            patterns = component.get("patterns", [])
            if not isinstance(patterns, list) or not patterns:
                raise ValueError("Pattern component requires a non-empty patterns list.")
            for raw_pattern in patterns:
                if not isinstance(raw_pattern, dict):
                    raise ValueError("Each pattern must be an object keyed by item.")
                pattern = {item: str(raw_pattern[item]) for item in metadata["items"] if item in raw_pattern}
                missing = [item for item in metadata["items"] if item not in pattern]
                if missing:
                    raise ValueError(f"Pattern is missing items: {', '.join(missing)}.")
                sid = len(rows) + 1
                rows.append(
                    {
                        "job_id": f"{tag}_{sid:03d}",
                        "battery": f"{metadata['wave']}_{metadata['battery']}",
                        "support_id": sid,
                        "variant": name,
                        "design_type": "pattern",
                        "moment_dimension": moment_dimension,
                        "pattern": pattern,
                        "prompt": pattern_prompt(metadata, sid, pattern, str(component.get("style", ""))),
                    }
                )
                if len(rows) >= n_support:
                    return rows
        elif ctype in {"option-coverage", "coverage"}:
            count = int(component.get("n", n_support - len(rows)))
            styles = component.get("styles", COVERAGE_STYLES)
            if not isinstance(styles, list) or not styles:
                raise ValueError("Coverage component styles must be a non-empty list.")
            for idx, (item, option) in enumerate(coverage_targets(metadata, count)):
                sid = len(rows) + 1
                style = str(styles[idx % len(styles)])
                rows.append(
                    {
                        "job_id": f"{tag}_{sid:03d}",
                        "battery": f"{metadata['wave']}_{metadata['battery']}",
                        "support_id": sid,
                        "variant": name,
                        "design_type": "option_coverage",
                        "moment_dimension": moment_dimension,
                        "coverage_item": item,
                        "coverage_option": option,
                        "coverage_style": style,
                        "prompt": coverage_prompt(metadata, sid, item, option, style),
                    }
                )
                if len(rows) >= n_support:
                    return rows
        else:
            raise ValueError(f"Unsupported design component type: {ctype}.")
    return rows[:n_support]


def write_support_outputs(rows: list[dict[str, Any]], metadata: dict[str, Any], tag: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{tag}.jsonl"
    design = out_dir / f"{tag}_design.csv"
    with jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    items = list(metadata["items"])
    with design.open("w", newline="") as f:
        axis_names = sorted({axis for row in rows for axis in row.get("axis_values", {})})
        fieldnames = ["support_id", "job_id", "variant", "design_type", "coverage_item", "coverage_option", "coverage_style", "moment_dimension", "persona", *axis_names, *items]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            pattern = row.get("pattern", {})
            out = {key: row.get(key, "") for key in fieldnames}
            for axis, value in row.get("axis_values", {}).items():
                out[axis] = value
            for item in items:
                out[item] = pattern.get(item, "")
            writer.writerow(out)
    return {"prompts_path": str(jsonl), "design_path": str(design)}
