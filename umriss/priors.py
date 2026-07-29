from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import UmrissError
from .jsonlio import read_jsonl, write_json, write_jsonl
from .metadata import item_option_labels
from .parsing import extract_json, normalized_vec, read_raw_rows
from .provenance import build_provenance


def _model_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("model.model") or row.get("model") or "unknown"),
        str(row.get("model.inference_service") or row.get("service_name") or "unknown"),
    )


def build_joint_prior_prompts(
    metadata: dict[str, Any],
    pairs: list[str],
    population: str,
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        parts = pair.split(":", 1)
        if len(parts) != 2 or any(item not in metadata["items"] for item in parts):
            raise UmrissError("invalid_input", f"Joint pair must be ITEM_A:ITEM_B using known items: {pair}.")
        left, right = parts
        left_labels = item_option_labels(metadata, left)
        right_labels = item_option_labels(metadata, right)
        prompt = f"""Predict one joint population response distribution for two survey items.

Population: {population}
Survey context: {metadata.get('context', '')}
Battery topic: {metadata.get('topic', '')}

First item: {metadata['items'][left]['question_stem']} {metadata['items'][left]['item_text']}
First-item options, in row order: {json.dumps(left_labels)}

Second item: {metadata['items'][right]['question_stem']} {metadata['items'][right]['item_text']}
Second-item options, in column order: {json.dumps(right_labels)}

Return the joint distribution directly. Do not multiply separately estimated marginals.
Return only valid JSON:
{{
  "reasoning_summary": "brief explanation",
  "joint_probabilities": [[one nonnegative cell per column] per row]
}}
The matrix must have shape {len(left_labels)} by {len(right_labels)} and all cells together must sum to 1."""
        target_id = f"joint:{left}:{right}"
        rows.append(
            {
                "job_id": f"{tag}_{left}_x_{right}",
                "stage": "prior_joint",
                "target_id": target_id,
                "target_type": "joint",
                "items": [left, right],
                "shape": [len(left_labels), len(right_labels)],
                "population": population,
                "prompt": prompt,
            }
        )
    path = out_dir / f"{tag}_joint_prior_prompts.jsonl"
    write_jsonl(path, rows)
    return {"prompts_path": str(path), "pairs": len(rows), "jobs": len(rows), "population": population}


def _parse_contract_response(
    response: dict[str, Any] | None,
    contract: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if contract.get("target_type") == "joint":
        shape = tuple(int(x) for x in contract["shape"])
        raw = response.get("joint_probabilities") if response else None
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return None, {"reason": "not_numeric"}
        if arr.shape != shape:
            return None, {"reason": "shape_mismatch", "observed_shape": list(arr.shape), "expected_shape": list(shape)}
        if not np.isfinite(arr).all() or (arr < 0).any():
            return None, {"reason": "invalid_probability"}
        total = float(arr.sum())
        if not np.isclose(total, 1.0, atol=1e-5):
            return None, {"reason": "sum_not_one", "raw_sum": total}
        return arr.reshape(-1), {"raw_sum": total}
    if contract.get("target_type") == "checkbox_marginal":
        raw = response.get("inclusion_probabilities") if response else None
        try:
            arr = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            return None, {"reason": "not_numeric"}
        expected = int(contract["option_count"])
        if arr.shape != (expected,):
            return None, {"reason": "shape_mismatch", "observed_shape": list(arr.shape), "expected_shape": [expected]}
        if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
            return None, {"reason": "invalid_probability"}
        return arr, {}
    vec, diag = normalized_vec(
        response.get("probabilities") if response else None,
        int(contract["option_count"]),
    )
    return vec, diag


def parse_model_priors(
    raw_paths: list[Path],
    prompts_path: Path,
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    contracts = {str(row["job_id"]): row for row in read_jsonl(prompts_path)}
    for contract in contracts.values():
        if "target_type" not in contract:
            item = str(contract.get("holdout"))
            contract.update(
                {
                    "target_id": f"marginal:{item}",
                    "target_type": "marginal",
                    "items": [item],
                    "option_count": len(item_option_labels(metadata, item)),
                    "population": metadata.get("population", {}).get("id", "unspecified"),
                }
            )
    parsed: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    expected: set[tuple[str, str, str]] = set()
    valid: set[tuple[str, str, str]] = set()
    for raw_path in raw_paths:
        rows = read_raw_rows(raw_path)
        model_keys = {_model_key(row) for row in rows}
        expected.update((job_id, model, service) for job_id in contracts for model, service in model_keys)
        for index, row in enumerate(rows):
            job_id = str(row.get("scenario.job_id") or row.get("job_id") or f"row_{index}")
            contract = contracts.get(job_id)
            if contract is None:
                raise UmrissError("invalid_input", f"Raw result has unknown prior job_id: {job_id}.")
            model, service = _model_key(row)
            response = extract_json(str(row.get("answer.resp") or row.get("response") or ""))
            vector, diag = _parse_contract_response(response, contract)
            key = (job_id, model, service)
            is_valid = vector is not None
            diagnostics.append(
                {
                    "job_id": job_id,
                    "target_id": contract["target_id"],
                    "model": model,
                    "service_name": service,
                    "valid": is_valid,
                    **diag,
                }
            )
            if not is_valid:
                continue
            valid.add(key)
            parsed.append(
                {
                    "tag": tag,
                    "job_id": job_id,
                    "target_id": contract["target_id"],
                    "target_type": contract["target_type"],
                    "items": json.dumps(contract["items"]),
                    "shape": json.dumps(contract.get("shape") or [len(vector)]),
                    "population": contract.get("population", "unspecified"),
                    "model": model,
                    "service_name": service,
                    "prediction": json.dumps([round(float(x), 10) for x in vector]),
                    "reasoning_summary": str(response.get("reasoning_summary", "")) if response else "",
                }
            )
    missing = sorted(expected - valid)
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / f"{tag}_prior_parse_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    predictions_path = out_dir / f"{tag}_prior_predictions.csv"
    pd.DataFrame(parsed).to_csv(predictions_path, index=False)
    if missing and not allow_incomplete:
        raise UmrissError(
            "incomplete_results",
            f"Prior results are missing {len(missing)} of {len(expected)} required model-job responses.",
            context={
                "expected_model_jobs": len(expected),
                "valid_model_jobs": len(valid),
                "missing_model_jobs": [
                    {"job_id": job_id, "model": model, "service_name": service}
                    for job_id, model, service in missing[:50]
                ],
                "diagnostics_path": str(diagnostics_path),
                "partial_predictions_path": str(predictions_path),
            },
        )
    return {
        "predictions_path": str(predictions_path),
        "diagnostics_path": str(diagnostics_path),
        "predictions": len(parsed),
        "expected_model_jobs": len(expected),
        "complete": not missing,
        "missing_model_jobs": len(missing),
    }


def consensus_targets(
    prediction_paths: list[Path],
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    *,
    population: str,
    max_total_variation: float,
    max_option_difference: float,
    minimum_models: int,
    confidence_weight: float,
) -> dict[str, Any]:
    frames = [pd.read_csv(path) for path in prediction_paths]
    frame = pd.concat(frames, ignore_index=True)
    required = {"target_id", "target_type", "items", "shape", "model", "service_name", "prediction"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise UmrissError("invalid_input", f"Prior predictions missing columns: {sorted(missing_columns)}.")
    frame = frame.drop_duplicates(["target_id", "model", "service_name"], keep="last")
    audit_rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for target_id, group in frame.groupby("target_id", sort=False):
        vectors = [np.asarray(json.loads(value), dtype=float) for value in group["prediction"]]
        if not vectors or len({len(vector) for vector in vectors}) != 1:
            raise UmrissError("invalid_input", f"Inconsistent prediction lengths for {target_id}.")
        pairwise = [
            (
                float(np.abs(left - right).sum() / 2),
                float(np.abs(left - right).max()),
            )
            for left, right in itertools.combinations(vectors, 2)
        ]
        max_tv = max((value[0] for value in pairwise), default=0.0)
        max_diff = max((value[1] for value in pairwise), default=0.0)
        accepted = len(vectors) >= minimum_models and max_tv <= max_total_variation and max_diff <= max_option_difference
        consensus = np.mean(np.vstack(vectors), axis=0)
        model_specs = [
            {"model": str(row["model"]), "service_name": str(row["service_name"])}
            for _, row in group.iterrows()
        ]
        first = group.iloc[0]
        audit_rows.append(
            {
                "target_id": target_id,
                "target_type": first["target_type"],
                "models": len(vectors),
                "max_pairwise_total_variation": max_tv,
                "max_option_difference": max_diff,
                "accepted": accepted,
                "consensus_mean": json.dumps(consensus.tolist()),
            }
        )
        targets.append(
            {
                "target_id": target_id,
                "type": str(first["target_type"]),
                "items": json.loads(str(first["items"])),
                "shape": json.loads(str(first["shape"])),
                "values": consensus.tolist(),
                "status": "accepted" if accepted else "rejected",
                "population": population,
                "confidence_weight": confidence_weight,
                "source": {
                    "kind": "model_synthetic",
                    "aggregation": "arithmetic_mean",
                    "models": model_specs,
                    "agreement": {
                        "maximum_pairwise_total_variation": max_tv,
                        "maximum_option_difference": max_diff,
                        "rule": {
                            "minimum_models": minimum_models,
                            "maximum_pairwise_total_variation": max_total_variation,
                            "maximum_option_difference": max_option_difference,
                        },
                    },
                },
            }
        )
    provenance = build_provenance(
        "umriss prior consensus",
        inputs={f"predictions_{index + 1}": path for index, path in enumerate(prediction_paths)},
        parameters={
            "tag": tag,
            "population": population,
            "max_total_variation": max_total_variation,
            "max_option_difference": max_option_difference,
            "minimum_models": minimum_models,
            "confidence_weight": confidence_weight,
        },
    )
    artifact = {
        "schema_version": 1,
        "kind": "umriss_targets",
        "battery": {"wave": metadata["wave"], "battery": metadata["battery"]},
        "population": {"id": population},
        "targets": targets,
        "provenance": provenance,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    targets_path = out_dir / f"{tag}_targets.json"
    audit_path = out_dir / f"{tag}_consensus_audit.csv"
    write_json(targets_path, artifact)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    accepted_count = sum(target["status"] == "accepted" for target in targets)
    return {
        "targets_path": str(targets_path),
        "audit_path": str(audit_path),
        "targets": len(targets),
        "accepted": accepted_count,
        "rejected": len(targets) - accepted_count,
        "population": population,
    }
