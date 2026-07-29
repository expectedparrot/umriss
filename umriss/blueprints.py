from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import UmrissError
from .metadata import item_option_labels


def validate_blueprint_fidelity(
    support_path: Path,
    plan_path: Path,
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    *,
    minimum_match_fraction: float = 0.8,
    minimum_intended_probability: float = 0.35,
) -> dict[str, Any]:
    if not 0 <= minimum_match_fraction <= 1:
        raise UmrissError("invalid_input", "minimum_match_fraction must be between 0 and 1.")
    if not 0 <= minimum_intended_probability <= 1:
        raise UmrissError("invalid_input", "minimum_intended_probability must be between 0 and 1.")
    support = pd.read_csv(support_path)
    plan = pd.read_csv(plan_path)
    required_support = {"support_id", "job_id", "item", "option_index", "option_label", "probability"}
    if not required_support <= set(support):
        raise UmrissError(
            "support_invalid",
            "Support probabilities lack columns required for blueprint validation.",
            context={"missing": sorted(required_support - set(support))},
        )
    if not {"support_id", "job_id", "pattern"} <= set(plan):
        raise UmrissError("plan_invalid", "Support plan must contain support_id, job_id, and pattern.")
    patterns: dict[tuple[int, str], dict[str, str]] = {}
    for row in plan.itertuples(index=False):
        try:
            pattern = json.loads(row.pattern) if isinstance(row.pattern, str) else {}
        except json.JSONDecodeError as exc:
            raise UmrissError("plan_invalid", f"Invalid pattern JSON for {row.job_id}.") from exc
        if pattern:
            patterns[(int(row.support_id), str(row.job_id))] = pattern
    if not patterns:
        raise UmrissError(
            "blueprints_missing",
            "The support plan contains no complete response blueprints.",
            hint="Build support with --preset balanced-blueprints.",
        )

    cell_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    accepted_keys: set[tuple[int, str]] = set()
    grouped = {
        (int(support_id), str(job_id)): group
        for (support_id, job_id), group in support.groupby(["support_id", "job_id"], sort=False)
    }
    for key, pattern in patterns.items():
        group = grouped.get(key)
        if group is None:
            row_rows.append(
                {
                    "support_id": key[0],
                    "job_id": key[1],
                    "blueprint_cells": len(pattern),
                    "matched_cells": 0,
                    "match_fraction": 0.0,
                    "mean_intended_probability": 0.0,
                    "minimum_intended_probability": 0.0,
                    "accepted": False,
                    "problems": json.dumps(["missing_support_point"]),
                }
            )
            continue
        matched = 0
        intended_values: list[float] = []
        problems: list[str] = []
        for item, intended_label in pattern.items():
            if item not in metadata["items"]:
                problems.append(f"unknown_item:{item}")
                continue
            item_group = group[group["item"].astype(str).eq(str(item))].sort_values("option_index")
            labels = item_option_labels(metadata, item)
            if intended_label not in labels or len(item_group) != len(labels):
                problems.append(f"missing_or_invalid_item:{item}")
                continue
            intended_index = labels.index(intended_label)
            probabilities = item_group["probability"].astype(float).to_numpy()
            intended_probability = float(probabilities[intended_index])
            argmax_index = int(probabilities.argmax())
            argmax_match = argmax_index == intended_index
            matched += int(argmax_match)
            intended_values.append(intended_probability)
            cell_rows.append(
                {
                    "support_id": key[0],
                    "job_id": key[1],
                    "item": item,
                    "intended_option_index": intended_index,
                    "intended_option_label": intended_label,
                    "measured_argmax_index": argmax_index,
                    "measured_argmax_label": labels[argmax_index],
                    "intended_probability": intended_probability,
                    "argmax_match": argmax_match,
                }
            )
        denominator = len(pattern)
        match_fraction = matched / denominator if denominator else 0.0
        mean_probability = sum(intended_values) / len(intended_values) if intended_values else 0.0
        minimum_probability = min(intended_values) if intended_values else 0.0
        accepted = (
            not problems
            and match_fraction >= minimum_match_fraction
            and mean_probability >= minimum_intended_probability
        )
        if accepted:
            accepted_keys.add(key)
        row_rows.append(
            {
                "support_id": key[0],
                "job_id": key[1],
                "blueprint_cells": denominator,
                "matched_cells": matched,
                "match_fraction": match_fraction,
                "mean_intended_probability": mean_probability,
                "minimum_intended_probability": minimum_probability,
                "accepted": accepted,
                "problems": json.dumps(problems),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / f"{tag}_blueprint_cell_diagnostics.csv"
    rows_path = out_dir / f"{tag}_blueprint_fidelity.csv"
    accepted_path = out_dir / f"{tag}_validated_probabilities.csv"
    retry_path = out_dir / f"{tag}_retry_job_ids.csv"
    pd.DataFrame(cell_rows).to_csv(cells_path, index=False)
    row_frame = pd.DataFrame(row_rows)
    row_frame.to_csv(rows_path, index=False)
    accepted = support[
        support.apply(lambda row: (int(row["support_id"]), str(row["job_id"])) in accepted_keys, axis=1)
    ].copy()
    accepted.to_csv(accepted_path, index=False)
    rejected_jobs = row_frame.loc[~row_frame["accepted"], ["job_id"]].drop_duplicates()
    rejected_jobs.to_csv(retry_path, index=False)
    accepted_count = int(row_frame["accepted"].sum())
    total = len(row_frame)
    return {
        "support_points": total,
        "accepted_support_points": accepted_count,
        "rejected_support_points": total - accepted_count,
        "acceptance_fraction": accepted_count / total if total else 0.0,
        "mean_match_fraction": float(row_frame["match_fraction"].mean()) if total else 0.0,
        "minimum_match_fraction": minimum_match_fraction,
        "minimum_intended_probability": minimum_intended_probability,
        "cell_diagnostics_path": str(cells_path),
        "fidelity_path": str(rows_path),
        "validated_probabilities_path": str(accepted_path),
        "retry_job_ids_path": str(retry_path),
        "passes": accepted_count == total,
    }
