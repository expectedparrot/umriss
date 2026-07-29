from __future__ import annotations

import csv
import hashlib
import html
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog

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
    # Ordinal anchors sit at the scale extremes: one all-low and one all-high
    # whole-battery pattern. (Middle+last, used previously, missed the low
    # extreme entirely and degenerated to duplicate anchors on binary scales.)
    patterns = (
        [
            {item: item_option_labels(metadata, item)[0] for item in metadata["items"]},
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


def balanced_blueprints_preset(metadata: dict[str, Any], size: int | None, seed: int) -> dict[str, Any]:
    resolved_size = size if size is not None else max(128, 8 * len(metadata["items"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "preset": "balanced-blueprints",
        "size": resolved_size,
        "seed": seed,
        "coverage": {"mode": "complete", "allocation": "balanced"},
        "components": [{"type": "balanced_blueprints", "coherence": "explicit"}],
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
            "validate_blueprint_fidelity": True,
        },
    }


def target_informed_blueprint_design(
    metadata: dict[str, Any],
    targets: dict[str, Any],
    size: int,
    seed: int,
    target_source: str,
) -> dict[str, Any]:
    if size < 1:
        raise ValueError("Target-informed blueprint size must be at least 1.")
    accepted: dict[str, list[float]] = {}
    target_ids: list[str] = []
    for target in targets.get("targets", []):
        if target.get("status") != "accepted" or target.get("type") != "marginal":
            continue
        item = str(target["items"][0])
        if item not in metadata["items"]:
            raise ValueError(f"Accepted target references unknown item {item}.")
        values = [float(value) for value in target["values"]]
        if len(values) != len(item_option_labels(metadata, item)):
            raise ValueError(f"Accepted target {target['target_id']} has the wrong shape.")
        accepted[item] = values
        target_ids.append(str(target["target_id"]))
    if not accepted:
        raise ValueError("No accepted marginal targets are available for target-informed blueprints.")

    rng = random.Random(seed)
    items = list(metadata["items"])
    possible = 1
    for item in items:
        possible *= len(item_option_labels(metadata, item))
    if size > possible:
        raise ValueError(f"Requested {size} blueprints but only {possible} distinct patterns exist.")

    patterns: list[dict[str, str]] | None = None
    for _ in range(500):
        columns: dict[str, list[str]] = {}
        for item in items:
            labels = item_option_labels(metadata, item)
            if item in accepted:
                exact = np.asarray(accepted[item], dtype=float) * size
                counts = np.floor(exact).astype(int)
                remainder = size - int(counts.sum())
                order = np.argsort(-(exact - counts))
                for index in order[:remainder]:
                    counts[index] += 1
                values = [
                    label
                    for label, count in zip(labels, counts, strict=True)
                    for _ in range(int(count))
                ]
            else:
                values = [labels[index % len(labels)] for index in range(size)]
            rng.shuffle(values)
            columns[item] = values
        candidate = [{item: columns[item][row_index] for item in items} for row_index in range(size)]
        signatures = {tuple(pattern[item] for item in items) for pattern in candidate}
        if len(signatures) == size:
            patterns = candidate
            break
    if patterns is None:
        raise ValueError("Could not construct unique target-informed blueprints.")
    return {
        "schema_version": SCHEMA_VERSION,
        "preset": "target-informed-blueprints",
        "size": size,
        "seed": seed,
        "coverage": {"mode": "complete", "allocation": "target_informed_largest_remainder"},
        "components": [{"type": "pattern_anchors", "patterns": patterns, "coherence": "explicit"}],
        "target_informed": {
            "source": target_source,
            "accepted_target_ids": target_ids,
            "allocation": "largest_remainder",
            "population_marginals_in_individual_prompts": False,
        },
        "profile": {"framing": "synthetic_respondent", "forbid_demographic_invention": True},
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
            "validate_blueprint_fidelity": True,
        },
    }


def target_repair_blueprint_design(
    metadata: dict[str, Any],
    targets: dict[str, Any],
    support_path: Path,
    n_add: int,
    seed: int,
    target_source: str,
) -> dict[str, Any]:
    support = pd.read_csv(support_path)
    identities = support[["support_id", "job_id"]].drop_duplicates()
    base_n = len(identities)
    if base_n < 1 or n_add < 1:
        raise ValueError("Target repair requires positive base and addition sizes.")
    adjusted = json.loads(json.dumps(targets))
    adjustments: list[dict[str, Any]] = []
    for target in adjusted.get("targets", []):
        if target.get("status") != "accepted" or target.get("type") != "marginal":
            continue
        item = str(target["items"][0])
        group = support[support["item"].astype(str).eq(item)]
        current = (
            group.groupby("option_index")["probability"]
            .mean()
            .sort_index()
            .to_numpy(dtype=float)
        )
        desired = np.asarray(target["values"], dtype=float)
        if len(current) != len(desired):
            raise ValueError(f"Base support is incomplete for accepted target item {item}.")
        addition = ((base_n + n_add) * desired - base_n * current) / n_add
        if (addition < -1e-9).any() or (addition > 1 + 1e-9).any():
            raise ValueError(
                f"TARGET_REPAIR_TOO_SMALL: {n_add} additions cannot correct {item} without invalid allocation "
                f"(range {addition.min():.6f} to {addition.max():.6f}). Increase --n-add."
            )
        addition = np.clip(addition, 0.0, 1.0)
        if not np.isclose(addition.sum(), 1.0, atol=1e-8):
            raise ValueError(f"Derived repair allocation for {item} does not sum to one.")
        target["values"] = addition.tolist()
        adjustments.append(
            {
                "item": item,
                "base_equal_weight_prediction": current.tolist(),
                "population_target": desired.tolist(),
                "addition_blueprint_allocation": addition.tolist(),
            }
        )
    design = target_informed_blueprint_design(metadata, adjusted, n_add, seed, target_source)
    design["preset"] = "target-repair-blueprints"
    design["target_repair"] = {
        "base_support": str(support_path),
        "base_n": base_n,
        "n_add": n_add,
        "formula": "((base_n + n_add) * target - base_n * base_equal_weight_prediction) / n_add",
        "adjustments": adjustments,
    }
    return design


def geometry_repair_blueprint_design(
    metadata: dict[str, Any],
    targets: dict[str, Any],
    support_path: Path,
    n_add: int,
    seed: int,
    target_source: str,
) -> dict[str, Any]:
    """Build blueprints that move support across its minimax separating direction."""
    support = pd.read_csv(support_path)
    identities = support[["support_id", "job_id"]].drop_duplicates()
    if identities.empty or n_add < 1:
        raise ValueError("Geometry repair requires positive base and addition sizes.")

    matrices: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    accepted_items: list[str] = []
    for target in targets.get("targets", []):
        if target.get("status") != "accepted" or target.get("type") != "marginal":
            continue
        item = str(target["items"][0])
        group = support[support["item"].astype(str).eq(item)]
        pivot = group.pivot(index="support_id", columns="option_index", values="probability")
        matrix = pivot.reindex(identities["support_id"]).sort_index(axis=1).to_numpy(dtype=float)
        values = np.asarray(target["values"], dtype=float)
        if matrix.shape[1] != len(values) or np.isnan(matrix).any():
            raise ValueError(f"Base support is incomplete for accepted target item {item}.")
        matrices.append(matrix)
        vectors.append(values)
        accepted_items.append(item)
    if not matrices:
        raise ValueError("No accepted marginal targets are available for geometry repair.")

    feature_matrix = np.column_stack(matrices)
    target_vector = np.concatenate(vectors)
    n_support, n_cells = feature_matrix.shape
    objective = np.r_[np.zeros(n_support), 1.0]
    upper = np.c_[feature_matrix.T, -np.ones(n_cells)]
    lower = np.c_[-feature_matrix.T, -np.ones(n_cells)]
    result = linprog(
        objective,
        A_ub=np.r_[upper, lower],
        b_ub=np.r_[target_vector, -target_vector],
        A_eq=np.c_[np.ones((1, n_support)), np.zeros((1, 1))],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * n_support + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Geometry-repair feasibility solver failed: {result.message}")
    witness_prediction = feature_matrix.T @ result.x[:n_support]
    desired_direction = target_vector - witness_prediction
    # HiGHS reports nonpositive marginals for <= constraints. Their upper-minus-
    # lower difference is the normalized separating hyperplane q satisfying
    # q·target - max_i(q·support_i) == the minimax feasibility gap.
    inequality_marginals = np.asarray(result.ineqlin.marginals, dtype=float)
    separating_direction = (
        inequality_marginals[:n_cells] - inequality_marginals[n_cells:]
    )
    separation_gap = float(
        separating_direction @ target_vector
        - np.max(feature_matrix @ separating_direction)
    )

    directions: dict[str, np.ndarray] = {}
    offset = 0
    direction_rows: list[dict[str, Any]] = []
    for item, values in zip(accepted_items, vectors, strict=True):
        width = len(values)
        direction = separating_direction[offset : offset + width]
        directions[item] = direction
        labels = item_option_labels(metadata, item)
        for option_index, (label, target_value, prediction, delta) in enumerate(
            zip(
                labels,
                values,
                witness_prediction[offset : offset + width],
                direction,
                strict=True,
            )
        ):
            direction_rows.append(
                {
                    "item": item,
                    "option_index": option_index,
                    "option_label": label,
                    "target": float(target_value),
                    "witness_prediction": float(prediction),
                    "witness_residual": float(
                        desired_direction[offset + option_index]
                    ),
                    "separating_direction": float(delta),
                }
            )
        offset += width

    rng = np.random.default_rng(seed)
    items = list(metadata["items"])
    patterns: list[dict[str, str]] = []
    signatures: set[tuple[str, ...]] = set()
    attempts = 0
    while len(patterns) < n_add and attempts < max(10_000, n_add * 500):
        attempts += 1
        pattern: dict[str, str] = {}
        for item in items:
            labels = item_option_labels(metadata, item)
            if item in directions:
                direction = directions[item]
                if not patterns:
                    option_index = int(np.argmax(direction))
                else:
                    scale = max(float(np.max(np.abs(direction))) / 2.0, 0.005)
                    logits = (direction - direction.max()) / scale
                    shares = np.exp(logits)
                    shares = 0.9 * shares / shares.sum() + 0.1 / len(labels)
                    option_index = int(rng.choice(len(labels), p=shares))
            else:
                option_index = int(rng.integers(len(labels)))
            pattern[item] = labels[option_index]
        signature = tuple(pattern[item] for item in items)
        if signature in signatures:
            continue
        signatures.add(signature)
        patterns.append(pattern)
    if len(patterns) != n_add:
        raise ValueError(
            f"Could not construct {n_add} unique geometry-repair blueprints; built {len(patterns)}."
        )

    design = {
        "schema_version": SCHEMA_VERSION,
        "preset": "geometry-repair-blueprints",
        "size": n_add,
        "seed": seed,
        "coverage": {"mode": "complete", "allocation": "minimax_direction"},
        "components": [
            {"type": "pattern_anchors", "patterns": patterns, "coherence": "explicit"}
        ],
        "geometry_repair": {
            "base_support": str(support_path),
            "base_n": n_support,
            "target_source": target_source,
            "minimum_maximum_absolute_residual": float(result.x[-1]),
            "separation_gap": separation_gap,
            "direction": direction_rows,
            "allocation": (
                "First blueprint maximizes the LP dual separating direction; "
                "remaining unique blueprints use certificate-weighted exploration."
            ),
            "population_marginals_in_individual_prompts": False,
        },
        "profile": {
            "framing": "synthetic_respondent",
            "forbid_demographic_invention": True,
        },
        "probabilities": {
            "interpretation": "subjective_response_probability",
            "require_sum": 1,
            "allow_zero": False,
            "minimum_probability": 0.01,
            "minimum_intended_probability": 0.8,
        },
        "guardrails": {
            "leak_target_marginals": False,
            "silent_coverage_truncation": False,
            "silent_invalid_response_repair": False,
            "describe_as_recovered_respondents": False,
            "validate_blueprint_fidelity": True,
        },
    }
    validate_design(design, metadata)
    return design


def preset_design(metadata: dict[str, Any], preset: str, size: int | None, seed: int) -> dict[str, Any]:
    if preset == "pattern-coverage":
        return pattern_coverage_preset(metadata, size, seed)
    if preset == "uniform-patterns":
        return uniform_patterns_preset(metadata, size, seed)
    if preset == "balanced-blueprints":
        return balanced_blueprints_preset(metadata, size, seed)
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
        elif ctype == "balanced_blueprints":
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
        elif ctype == "balanced_blueprints":
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
    details = ",\n".join(
        f'    "{item}": "You explicitly believe or experience ..."'
        for item in metadata["items"]
    )
    probabilities = ",\n".join(f'    "{item}": [numbers in option order]' for item in metadata["items"])
    return (
        '{\n  "persona": "You are ...",\n  "persona_details": {\n'
        + details
        + '\n  },\n  "probabilities": {\n'
        + probabilities
        + "\n  }\n}"
    )


def _coherence_instruction(row: dict[str, Any], design: dict[str, Any], metadata: dict[str, Any]) -> str:
    mode = row["coherence"]
    if mode == "item_specific":
        if len(metadata.get("items", {})) <= 1:
            return "Ground the persona in this specific response tendency."
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
        target = (
            "Construct a synthetic survey-response profile that preserves every cell in this declared answer blueprint. "
            "Unusual combinations are intentional; do not replace them with a more stereotypically coherent profile.\n"
            f"{lines}"
        )
    else:
        target = f"Construct this declared synthetic survey-response profile:\n{row['profile']}"
    demographic = (
        "Do not invent demographic characteristics unless the design explicitly supplies them."
        if profile.get("forbid_demographic_invention", True)
        else ""
    )
    minimum = float(probs.get("minimum_probability", 0.0))
    intended_minimum = probs.get("minimum_intended_probability")
    blueprint_rule = (
        "For every declared blueprint cell, assign that option the largest probability in its item vector. "
        + (
            f"Assign every declared option probability at least {float(intended_minimum):g}. "
            if intended_minimum is not None
            else ""
        )
        + "The persona must make the complete combination understandable without changing any declared answer."
        if row["design_type"] == "pattern_anchor"
        else ""
    )
    return f"""Support point identifier: {row['job_id']}

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

{target}

{_coherence_instruction(row, design, metadata)}
{blueprint_rule}
{demographic}

Write a readable second-person synthesis in "persona", beginning with "Your " or "You ". Do not compress distinct
answers into labels such as "traditional", "egalitarian", or "mixed". In "persona_details", write one explicit,
second-person sentence for every item, using the exact item keys shown in the schema. Each sentence must state the
substantive position or circumstance represented by that item's declared response pattern. The complete persona will be
assembled from the synthesis and every detail, so its length should grow with the battery. Include only persona
description—not instructions about answering questions or probabilities.

Then provide subjective response probabilities for every item. Each vector must follow the displayed option order,
contain nonnegative numbers, and sum to 1. Each probability must be at least {minimum:g}.

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
    required = [
        *metadata["items"],
        "persona",
        "persona_details",
        "second person",
        "probabil",
        "sum to 1",
    ]
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
        elif ctype == "balanced_blueprints":
            items = list(metadata["items"])
            size = int(design["size"])
            possible = 1
            for item in items:
                possible *= len(item_option_labels(metadata, item))
            if size > possible:
                raise ValueError(
                    f"balanced_blueprints size {size} exceeds the number of distinct response patterns ({possible})."
                )
            rng = random.Random(int(design.get("seed", 0)))
            patterns: list[dict[str, str]] | None = None
            for _ in range(250):
                columns: dict[str, list[str]] = {}
                for item in items:
                    labels = item_option_labels(metadata, item)
                    values = [labels[index % len(labels)] for index in range(size)]
                    rng.shuffle(values)
                    columns[item] = values
                candidate = [
                    {item: columns[item][row_index] for item in items}
                    for row_index in range(size)
                ]
                signatures = {tuple(pattern[item] for item in items) for pattern in candidate}
                if len(signatures) == size:
                    patterns = candidate
                    break
            if patterns is None:
                raise ValueError("Could not construct unique balanced blueprints for the requested size and battery.")
            for pattern in patterns:
                rows.append(
                    {
                        "design_type": "pattern_anchor",
                        "reason": "balanced complete response blueprint",
                        "pattern": pattern,
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
