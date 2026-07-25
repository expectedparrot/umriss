from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import load_support_matrix
from .metadata import item_option_labels
from .support_designs import write_support_outputs


def diversity_metrics(support_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    support, mats = load_support_matrix(support_path)
    full = np.column_stack([mats[item] for item in metadata["items"]])
    independent = np.column_stack([mats[item][:, :-1] for item in metadata["items"]])
    unique_vectors = int(np.unique(np.round(full, 12), axis=0).shape[0])
    centered = independent - independent.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    nonzero = singular[singular > (singular.max() * 1e-10 if len(singular) else 0)]
    energy = nonzero**2
    shares = energy / energy.sum() if energy.sum() else np.array([])
    effective_rank = float(np.exp(-np.sum(shares * np.log(shares)))) if len(shares) else 0.0
    patterns = np.column_stack([np.argmax(mats[item], axis=1) for item in metadata["items"]])
    joint_patterns = int(np.unique(patterns, axis=0).shape[0])
    possible_patterns = int(np.prod([mats[item].shape[1] for item in metadata["items"]]))
    return {
        "n_support": len(support),
        "unique_probability_vectors": unique_vectors,
        "duplicate_probability_vectors": len(support) - unique_vectors,
        "duplicate_fraction": (len(support) - unique_vectors) / len(support),
        "joint_argmax_patterns": joint_patterns,
        "possible_joint_patterns": possible_patterns,
        "joint_pattern_fraction": joint_patterns / min(possible_patterns, len(support)),
        "matrix_rank": int(len(nonzero)),
        "effective_rank": effective_rank,
        "independent_moment_dimension": int(independent.shape[1]),
    }


def uniformity_rows(
    support_path: Path, metadata: dict[str, Any], tolerance: float
) -> tuple[pd.DataFrame, bool]:
    support, mats = load_support_matrix(support_path)
    rows = []
    for item in metadata["items"]:
        measured = mats[item].mean(axis=0)
        target = np.ones(len(measured)) / len(measured)
        gap = measured - target
        rows.append(
            {
                "item": item,
                "item_text": metadata["items"][item]["item_text"],
                "n_support": len(support),
                "equal_weight_marginal": json.dumps(measured.tolist()),
                "uniform_target": json.dumps(target.tolist()),
                "max_absolute_deviation": float(np.max(np.abs(gap))),
                "tolerance": tolerance,
                "passes": bool(np.max(np.abs(gap)) <= tolerance + 1e-12),
            }
        )
    frame = pd.DataFrame(rows)
    return frame, bool(frame["passes"].all())


def write_uniformity(
    support_path: Path,
    metadata: dict[str, Any],
    tolerance: float,
    out_path: Path,
    max_duplicate_fraction: float = 0.05,
    min_joint_pattern_fraction: float = 0.75,
) -> dict[str, Any]:
    frame, uniformity_passes = uniformity_rows(support_path, metadata, tolerance)
    diversity = diversity_metrics(support_path, metadata)
    diversity_passes = (
        diversity["duplicate_fraction"] <= max_duplicate_fraction + 1e-12
        and diversity["joint_pattern_fraction"] >= min_joint_pattern_fraction - 1e-12
    )
    passes = uniformity_passes and diversity_passes
    for key, value in diversity.items():
        frame[key] = value
    frame["max_duplicate_fraction"] = max_duplicate_fraction
    frame["min_joint_pattern_fraction"] = min_joint_pattern_fraction
    frame["diversity_passes"] = diversity_passes
    frame["preflight_passes"] = passes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return {
        "path": str(out_path),
        "n_support": int(frame["n_support"].max()),
        "tolerance": tolerance,
        "passes": passes,
        "uniformity_passes": uniformity_passes,
        "diversity_passes": diversity_passes,
        "max_absolute_deviation": float(frame["max_absolute_deviation"].max()),
        "failed_items": frame.loc[~frame["passes"], "item"].tolist(),
        **diversity,
    }


def _balanced_assignments(shares: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    raw = shares * n
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[idx] += 1
    values = np.concatenate([np.repeat(idx, count) for idx, count in enumerate(counts)])
    rng.shuffle(values)
    return values


def _adaptive_prompt(
    metadata: dict[str, Any],
    pattern: dict[str, str],
    support_id: int,
    minimum_probability: float,
) -> str:
    tendencies = "\n".join(
        f'- {item}: lean toward "{option}" for {metadata["items"][item]["item_text"]}'
        for item, option in pattern.items()
    )
    items = "\n".join(
        f"- {item}: {meta['item_text']}\n  Options, in output order: "
        + "; ".join(f"{idx + 1}. {label}" for idx, label in enumerate(item_option_labels(metadata, item)))
        for item, meta in metadata["items"].items()
    )
    schema = ",\n".join(f'    "{item}": [numbers in option order]' for item in metadata["items"])
    return f"""Support point identifier: {support_id}

Survey context: {metadata['context']}
Battery topic: {metadata['topic']}

Construct an additional synthetic survey-response profile for a support bank. Its purpose is to occupy a response pattern
that is underrepresented in the measured bank. Treat every item-specific tendency below as deliberate; do not replace the
pattern with one uniformly positive or negative outlook.

Target response tendencies:
{tendencies}

Provide a short substantive profile summary that could make this combination of responses coherent without inventing
demographic characteristics. Then provide subjective response probabilities for every item. Put more probability on each
targeted option than on the alternatives, while preserving genuine uncertainty. Each vector must follow the displayed
option order, contain nonnegative numbers, sum to 1, and assign at least {minimum_probability:g} to every option.

Items and response options:
{items}

Return only valid JSON with exactly this schema:
{{
  "profile_summary": "short substantive description",
  "probabilities": {{
{schema}
  }}
}}"""


def build_uniform_augmentation(
    support_path: Path,
    metadata: dict[str, Any],
    tag: str,
    n_add: int,
    tolerance: float,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    if n_add < 1:
        raise ValueError("n-add must be at least 1.")
    support, mats = load_support_matrix(support_path)
    diagnosis, passes = uniformity_rows(support_path, metadata, tolerance)
    diagnostic_path = out_dir / f"{tag}_pre_augmentation_uniformity.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnosis.to_csv(diagnostic_path, index=False)
    if passes:
        return {
            "already_balanced": True,
            "rows": 0,
            "uniformity_path": str(diagnostic_path),
            "n_support": len(support),
        }

    rng = np.random.default_rng(seed)
    assignments: dict[str, np.ndarray] = {}
    requested_shares: dict[str, list[float]] = {}
    n_existing = len(support)
    for item in metadata["items"]:
        current_sums = mats[item].sum(axis=0)
        desired_addition = (n_existing + n_add) / mats[item].shape[1] - current_sums
        desired_addition = np.maximum(desired_addition, 0)
        if desired_addition.sum() == 0:
            desired_addition = np.ones_like(desired_addition)
        shares = desired_addition / desired_addition.sum()
        assignments[item] = _balanced_assignments(shares, n_add, rng)
        requested_shares[item] = shares.tolist()

    minimum = 0.01
    rows = []
    start = int(pd.to_numeric(support["support_id"], errors="coerce").max()) + 1
    for offset in range(n_add):
        support_id = start + offset
        pattern = {
            item: item_option_labels(metadata, item)[int(assignments[item][offset])]
            for item in metadata["items"]
        }
        rows.append(
            {
                "support_id": support_id,
                "job_id": f"{tag}_{support_id:03d}",
                "battery": f"{metadata['wave']}_{metadata['battery']}",
                "design_schema_version": 1,
                "design_type": "adaptive_uniform",
                "reason": "repair measured equal-weight marginal deficits",
                "pattern": pattern,
                "coherence": "explicit",
                "prompt": _adaptive_prompt(metadata, pattern, support_id, minimum),
            }
        )
    design = {
        "schema_version": 1,
        "type": "adaptive_uniform_augmentation",
        "size": n_add,
        "seed": seed,
        "base_support": str(support_path),
        "base_n": n_existing,
        "uniform_tolerance": tolerance,
        "requested_target_shares": requested_shares,
        "probabilities": {"minimum_probability": minimum, "require_sum": 1},
    }
    paths = write_support_outputs(rows, metadata, tag, out_dir, design)
    return {
        **paths,
        "already_balanced": False,
        "rows": n_add,
        "base_n": n_existing,
        "uniformity_path": str(diagnostic_path),
    }


def merge_support_banks(base_path: Path, additions_path: Path, tag: str, out_dir: Path) -> dict[str, Any]:
    base = pd.read_csv(base_path)
    additions = pd.read_csv(additions_path)
    required = {"support_id", "job_id", "item", "option_index", "option_code", "option_label", "probability"}
    for label, frame in [("base", base), ("additions", additions)]:
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} support bank is missing columns: {', '.join(sorted(missing))}.")
    base_ids = base[["support_id"]].drop_duplicates()
    add_ids = additions[["support_id"]].drop_duplicates().reset_index(drop=True)
    next_id = int(pd.to_numeric(base_ids["support_id"], errors="coerce").max()) + 1
    id_map = {old: next_id + idx for idx, old in enumerate(add_ids["support_id"])}
    additions = additions.copy()
    additions["support_id"] = additions["support_id"].map(id_map)
    merged = pd.concat([base, additions], ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_probabilities.csv"
    merged.to_csv(path, index=False)
    return {
        "probabilities_path": str(path),
        "base_support_points": int(base["support_id"].nunique()),
        "added_support_points": int(additions["support_id"].nunique()),
        "support_points": int(merged["support_id"].nunique()),
    }
