from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import UmrissError
from .jsonlio import write_jsonl
from .metadata import item_option_codes, item_option_labels
from .parsing import extract_json, normalized_vec, read_raw_rows


def build_item_extension_prompts(
    points_path: Path,
    metadata: dict[str, Any],
    items: list[str],
    joints: list[str],
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    points = pd.read_csv(points_path)
    required = {"support_id", "job_id", "persona"}
    if missing := required - set(points.columns):
        raise UmrissError("invalid_input", f"Points file missing columns: {sorted(missing)}.")
    if points["support_id"].duplicated().any():
        raise UmrissError("invalid_input", "Points file contains duplicate support_id values.")
    rows: list[dict[str, Any]] = []
    for item in items:
        if item not in metadata["items"]:
            raise UmrissError("item_not_found", f"Unknown extension item: {item}.")
    parsed_joints: list[tuple[str, str]] = []
    for pair in joints:
        parts = pair.split(":", 1)
        if len(parts) != 2 or any(item not in metadata["items"] for item in parts):
            raise UmrissError("invalid_input", f"Joint extension must be ITEM_A:ITEM_B: {pair}.")
        parsed_joints.append((parts[0], parts[1]))
    for _, point in points.iterrows():
        support_id = str(point["support_id"])
        persona = str(point["persona"])
        for item in items:
            labels = item_option_labels(metadata, item)
            spec = metadata["items"][item]
            is_checkbox = spec.get("question_type") == "checkbox"
            contract_text = (
                '"inclusion_probabilities": [one probability per option; values are independent inclusion rates and do not need to sum to 1]'
                if is_checkbox
                else '"probabilities": [one nonnegative probability per option, summing to 1]'
            )
            prompt = f"""Answer a new survey item from this existing persona's point of view.

Persona:
{persona}

Question: {spec['question_stem']} {spec['item_text']}
Options, in order: {json.dumps(labels)}

Return only valid JSON:
{{
  {contract_text}
}}
Do not add demographics or revise the persona."""
            rows.append(
                {
                    "job_id": f"{tag}_{support_id}_{item}",
                    "stage": "support_extension",
                    "support_id": support_id,
                    "source_job_id": str(point["job_id"]),
                    "target_id": f"{'checkbox' if is_checkbox else 'marginal'}:{item}",
                    "target_type": "checkbox_marginal" if is_checkbox else "marginal",
                    "items": [item],
                    "shape": [len(labels)],
                    "prompt": prompt,
                }
            )
        for left, right in parsed_joints:
            left_labels = item_option_labels(metadata, left)
            right_labels = item_option_labels(metadata, right)
            left_spec = metadata["items"][left]
            right_spec = metadata["items"][right]
            prompt = f"""Give this existing persona's joint response probabilities for two survey items.

Persona:
{persona}

First question: {left_spec['question_stem']} {left_spec['item_text']}
Rows: {json.dumps(left_labels)}

Second question: {right_spec['question_stem']} {right_spec['item_text']}
Columns: {json.dumps(right_labels)}

Return the joint distribution directly, including any residual dependence within this persona.
Return only valid JSON:
{{
  "joint_probabilities": [[one nonnegative cell per column] per row]
}}
The {len(left_labels)} by {len(right_labels)} matrix must sum to 1."""
            rows.append(
                {
                    "job_id": f"{tag}_{support_id}_{left}_x_{right}",
                    "stage": "support_extension",
                    "support_id": support_id,
                    "source_job_id": str(point["job_id"]),
                    "target_id": f"joint:{left}:{right}",
                    "target_type": "joint",
                    "items": [left, right],
                    "shape": [len(left_labels), len(right_labels)],
                    "prompt": prompt,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_extension_prompts.jsonl"
    write_jsonl(path, rows)
    return {
        "prompts_path": str(path),
        "personas": len(points),
        "marginal_items": len(items),
        "joint_pairs": len(parsed_joints),
        "jobs": len(rows),
    }


def parse_item_extension(
    raw_path: Path,
    prompts_path: Path,
    base_support_path: Path,
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    from .jsonlio import read_jsonl

    contracts = {str(row["job_id"]): row for row in read_jsonl(prompts_path)}
    seen: set[str] = set()
    marginal_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, row in enumerate(read_raw_rows(raw_path)):
        job_id = str(row.get("scenario.job_id") or row.get("job_id") or f"row_{index}")
        contract = contracts.get(job_id)
        if contract is None:
            raise UmrissError("invalid_input", f"Unknown extension job_id: {job_id}.")
        response = extract_json(str(row.get("answer.resp") or row.get("response") or ""))
        if contract["target_type"] in {"marginal", "checkbox_marginal"}:
            item = contract["items"][0]
            if contract["target_type"] == "checkbox_marginal":
                try:
                    vector = np.asarray(
                        response.get("inclusion_probabilities") if response else None,
                        dtype=float,
                    )
                except (TypeError, ValueError):
                    vector = np.asarray([])
                valid_checkbox = (
                    vector.shape == (int(contract["shape"][0]),)
                    and np.isfinite(vector).all()
                    and (vector >= 0).all()
                    and (vector <= 1).all()
                )
                diag = {"reason": None if valid_checkbox else "invalid_checkbox_probabilities"}
                if not valid_checkbox:
                    vector = None
            else:
                vector, diag = normalized_vec(
                    response.get("probabilities") if response else None,
                    int(contract["shape"][0]),
                )
            if vector is not None:
                for option_index, probability in enumerate(vector):
                    marginal_rows.append(
                        {
                            "support_id": contract["support_id"],
                            "job_id": contract["source_job_id"],
                            "item": item,
                            "option_index": option_index,
                            "option_code": item_option_codes(metadata, item)[option_index],
                            "option_label": item_option_labels(metadata, item)[option_index],
                            "probability": float(probability),
                        }
                    )
        else:
            shape = tuple(int(value) for value in contract["shape"])
            try:
                matrix = np.asarray(response.get("joint_probabilities") if response else None, dtype=float)
            except (TypeError, ValueError):
                matrix = np.asarray([])
            valid = (
                matrix.shape == shape
                and np.isfinite(matrix).all()
                and (matrix >= 0).all()
                and np.isclose(matrix.sum(), 1.0, atol=1e-5)
            )
            vector = matrix.reshape(-1) if valid else None
            diag = {"raw_sum": float(matrix.sum()) if matrix.size else None, "observed_shape": list(matrix.shape)}
            if vector is not None:
                for cell_index, probability in enumerate(vector):
                    joint_rows.append(
                        {
                            "support_id": contract["support_id"],
                            "job_id": contract["source_job_id"],
                            "target_id": contract["target_id"],
                            "cell_index": cell_index,
                            "probability": float(probability),
                            "feature_method": "direct_joint",
                        }
                    )
        is_valid = vector is not None
        diagnostics.append({"job_id": job_id, "target_id": contract["target_id"], "valid": is_valid, **diag})
        if is_valid:
            seen.add(job_id)
    missing = sorted(set(contracts) - seen)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / f"{tag}_extension_parse_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    if missing:
        raise UmrissError(
            "incomplete_results",
            f"Extension results are missing {len(missing)} of {len(contracts)} jobs.",
            context={"missing_job_ids": missing, "diagnostics_path": str(diagnostics_path)},
        )
    base = pd.read_csv(base_support_path)
    extended = pd.concat([base, pd.DataFrame(marginal_rows)], ignore_index=True)
    duplicate_cells = extended.duplicated(["support_id", "item", "option_index"], keep=False)
    if duplicate_cells.any():
        duplicates = extended.loc[duplicate_cells, ["support_id", "item", "option_index"]].drop_duplicates()
        raise UmrissError(
            "output_conflict",
            "Extension would overwrite existing support cells.",
            context={"duplicate_cells": duplicates.head(20).to_dict("records")},
        )
    probabilities_path = out_dir / f"{tag}_probabilities.csv"
    extended.to_csv(probabilities_path, index=False)
    joint_path = out_dir / f"{tag}_joint_features.csv"
    pd.DataFrame(joint_rows).to_csv(joint_path, index=False)
    return {
        "probabilities_path": str(probabilities_path),
        "joint_features_path": str(joint_path) if joint_rows else None,
        "marginal_rows_added": len(marginal_rows),
        "joint_rows_added": len(joint_rows),
        "diagnostics_path": str(diagnostics_path),
    }
