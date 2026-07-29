from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .calibration import entropy_calibration_fit
from .errors import UmrissError
from .jsonlio import read_json, write_json
from .metadata import item_option_labels
from .provenance import build_provenance


def targets_from_metadata(
    metadata_path: Path,
    metadata: dict[str, Any],
    population: str,
    out_path: Path,
    confidence_weight: float,
) -> dict[str, Any]:
    truth = metadata.get("truth")
    if not isinstance(truth, dict) or not truth:
        raise UmrissError("marginal_missing", "Metadata has no observed truth marginals.")
    targets = []
    for item, values in truth.items():
        targets.append(
            {
                "target_id": f"marginal:{item}",
                "type": "marginal",
                "items": [item],
                "shape": [len(item_option_labels(metadata, item))],
                "values": list(values),
                "status": "accepted",
                "population": population,
                "confidence_weight": confidence_weight,
                "source": {"kind": "observed", "metadata_path": str(metadata_path)},
            }
        )
    artifact = {
        "schema_version": 1,
        "kind": "umriss_targets",
        "battery": {"wave": metadata["wave"], "battery": metadata["battery"]},
        "population": {"id": population},
        "targets": targets,
        "provenance": build_provenance(
            "umriss targets from-metadata",
            inputs={"metadata": metadata_path},
            parameters={"population": population, "confidence_weight": confidence_weight},
        ),
    }
    write_json(out_path, artifact)
    return {"targets_path": str(out_path), "targets": len(targets), "population": population}


def merge_target_artifacts(paths: list[Path], out_path: Path) -> dict[str, Any]:
    artifacts = [read_json(path) for path in paths]
    populations = {str((artifact.get("population") or {}).get("id")) for artifact in artifacts}
    if len(populations) != 1:
        raise UmrissError("population_mismatch", "All merged target artifacts must describe the same population.", context={"populations": sorted(populations)})
    targets_by_id: dict[str, dict[str, Any]] = {}
    for path, artifact in zip(paths, artifacts):
        for target in artifact.get("targets", []):
            target_id = str(target["target_id"])
            candidate = {**target, "artifact_source": str(path)}
            existing = targets_by_id.get(target_id)
            if existing is not None and existing.get("status") == "rejected" and candidate.get("status") == "accepted":
                targets_by_id[target_id] = candidate
                continue
            if existing is not None and existing.get("status") == "accepted" and candidate.get("status") == "rejected":
                continue
            if existing is not None:
                raise UmrissError(
                    "target_conflict",
                    f"Target {target_id} appears in more than one artifact.",
                    hint="Only an accepted target may automatically replace a rejected candidate with the same ID.",
                )
            targets_by_id[target_id] = candidate
    targets = list(targets_by_id.values())
    merged = {
        "schema_version": 1,
        "kind": "umriss_targets",
        "battery": artifacts[0].get("battery"),
        "population": artifacts[0].get("population"),
        "targets": targets,
        "provenance": build_provenance(
            "umriss targets merge",
            inputs={f"targets_{index + 1}": path for index, path in enumerate(paths)},
            parameters={},
        ),
    }
    write_json(out_path, merged)
    return {"targets_path": str(out_path), "targets": len(targets), "population": next(iter(populations))}


def audit_targets(
    path: Path,
    metadata: dict[str, Any],
    out_path: Path | None = None,
    consistency_tolerance: float = 0.05,
) -> dict[str, Any]:
    artifact = read_json(path)
    if artifact.get("schema_version") != 1 or artifact.get("kind") != "umriss_targets":
        raise UmrissError("targets_invalid", "Targets must declare schema_version 1 and kind `umriss_targets`.")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    accepted = 0
    artifact_population = str((artifact.get("population") or {}).get("id") or "")
    if not artifact_population:
        raise UmrissError("targets_invalid", "Targets artifact must declare population.id.")
    for target in artifact.get("targets", []):
        target_id = str(target.get("target_id", ""))
        problems: list[str] = []
        if not target_id or target_id in seen:
            problems.append("missing_or_duplicate_target_id")
        seen.add(target_id)
        target_type = target.get("type")
        items = target.get("items") or []
        if target_type not in {"marginal", "joint", "checkbox_marginal"}:
            problems.append("unsupported_type")
        if any(item not in metadata["items"] for item in items):
            problems.append("unknown_item")
        expected_shape = [len(item_option_labels(metadata, item)) for item in items if item in metadata["items"]]
        shape = list(target.get("shape") or [])
        if target_type in {"marginal", "checkbox_marginal"} and len(expected_shape) == 1:
            expected_shape = [expected_shape[0]]
        if shape != expected_shape:
            problems.append("shape_mismatch")
        values = np.asarray(target.get("values", []), dtype=float)
        if values.size != int(np.prod(shape)) if shape else True:
            problems.append("value_count_mismatch")
        elif not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            problems.append("invalid_probability")
        elif target_type != "checkbox_marginal" and not np.isclose(values.sum(), 1.0, atol=1e-5):
            problems.append("sum_not_one")
        if target_type == "joint" and target.get("feature_method") not in {None, "direct_joint", "conditional_independence"}:
            problems.append("invalid_feature_method")
        source_kind = (target.get("source") or {}).get("kind")
        if source_kind not in {"observed", "model_synthetic", "user_declared"}:
            problems.append("source_kind_missing")
        if str(target.get("population") or artifact_population) != artifact_population:
            problems.append("population_mismatch")
        try:
            confidence = float(target.get("confidence_weight", 1.0))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence <= 0:
            problems.append("invalid_confidence_weight")
        valid = not problems
        if valid and target.get("status") == "accepted":
            accepted += 1
        rows.append(
            {
                "target_id": target_id,
                "type": target_type,
                "status": target.get("status"),
                "valid": valid,
                "problems": json.dumps(problems),
                "source_kind": source_kind,
                "confidence_weight": confidence,
            }
        )
    accepted_marginals = {
        str(target["items"][0]): np.asarray(target["values"], dtype=float)
        for target in artifact.get("targets", [])
        if target.get("status") == "accepted" and target.get("type") == "marginal"
    }
    consistency_rows: list[dict[str, Any]] = []
    for target in artifact.get("targets", []):
        if target.get("status") != "accepted" or target.get("type") != "joint":
            continue
        left, right = target["items"]
        matrix = np.asarray(target["values"], dtype=float).reshape(target["shape"])
        for item, induced in ((left, matrix.sum(axis=1)), (right, matrix.sum(axis=0))):
            if item not in accepted_marginals:
                continue
            deviation = float(np.abs(induced - accepted_marginals[item]).max())
            consistency_rows.append(
                {"target_id": target["target_id"], "item": item, "maximum_absolute_deviation": deviation}
            )
            if deviation > consistency_tolerance:
                for row in rows:
                    if row["target_id"] == target["target_id"]:
                        row["valid"] = False
                        problems = json.loads(row["problems"])
                        problems.append("marginal_inconsistency")
                        row["problems"] = json.dumps(problems)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
    invalid = sum(not row["valid"] for row in rows)
    if invalid:
        raise UmrissError(
            "targets_invalid",
            f"{invalid} target definitions are invalid.",
            context={"audit_path": str(out_path) if out_path else None, "invalid_targets": [row for row in rows if not row["valid"]]},
        )
    return {
        "targets": len(rows),
        "accepted": accepted,
        "rejected": len(rows) - accepted,
        "population": artifact.get("population"),
        "audit_path": str(out_path) if out_path else None,
        "joint_marginal_consistency": consistency_rows,
        "consistency_tolerance": consistency_tolerance,
    }


def _support_identity(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["support_id", "job_id"]].drop_duplicates().reset_index(drop=True)


def diagnose_target_feasibility(
    support_path: Path,
    targets_path: Path,
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    *,
    tolerance: float = 0.01,
) -> dict[str, Any]:
    if tolerance < 0:
        raise UmrissError("invalid_input", "Feasibility tolerance must be nonnegative.")
    support_frame = pd.read_csv(support_path)
    support = _support_identity(support_frame)
    artifact = read_json(targets_path)
    audit_targets(targets_path, metadata)
    matrices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    cells: list[dict[str, Any]] = []
    for target in artifact.get("targets", []):
        if target.get("status") != "accepted":
            continue
        if target.get("type") not in {"marginal", "checkbox_marginal"}:
            raise UmrissError(
                "feasibility_features_missing",
                f"Feasibility currently requires marginal support features; {target['target_id']} is {target.get('type')}.",
                hint="Audit joint targets separately or provide a marginal-only target artifact.",
            )
        item = str(target["items"][0])
        group = support_frame[support_frame["item"].astype(str).eq(item)]
        pivot = group.pivot(index="support_id", columns="option_index", values="probability")
        matrix = pivot.reindex(support["support_id"]).sort_index(axis=1).to_numpy(dtype=float)
        target_values = np.asarray(target["values"], dtype=float)
        if matrix.shape[1] != len(target_values) or np.isnan(matrix).any():
            raise UmrissError("support_missing_item", f"Support bank is incomplete for target item {item}.")
        matrices.append(matrix)
        values.append(target_values)
        labels = item_option_labels(metadata, item)
        for option_index, target_value in enumerate(target_values):
            minimum = float(matrix[:, option_index].min())
            maximum = float(matrix[:, option_index].max())
            cells.append(
                {
                    "target_id": target["target_id"],
                    "item": item,
                    "option_index": option_index,
                    "option_label": labels[option_index],
                    "target": float(target_value),
                    "support_minimum": minimum,
                    "support_maximum": maximum,
                    "outside_cell_range": bool(target_value < minimum - tolerance or target_value > maximum + tolerance),
                }
            )
    if not matrices:
        raise UmrissError("targets_empty", "No accepted marginal targets are available for feasibility analysis.")
    X = np.column_stack(matrices)
    y = np.concatenate(values)
    n_support, n_cells = X.shape
    # Minimize the largest absolute constraint error over the support simplex.
    c = np.r_[np.zeros(n_support), 1.0]
    upper = np.c_[X.T, -np.ones(n_cells)]
    lower = np.c_[-X.T, -np.ones(n_cells)]
    result = linprog(
        c,
        A_ub=np.r_[upper, lower],
        b_ub=np.r_[y, -y],
        A_eq=np.c_[np.ones((1, n_support)), np.zeros((1, 1))],
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * n_support + [(0.0, None)],
        method="highs",
    )
    if not result.success:
        raise UmrissError("feasibility_solver_failed", result.message)
    weights = result.x[:n_support]
    prediction = X.T @ weights
    residual = prediction - y
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / f"{tag}_feasibility.csv"
    witness_path = out_dir / f"{tag}_feasibility_witness_weights.csv"
    summary_path = out_dir / f"{tag}_feasibility_summary.csv"
    pd.DataFrame(cells).to_csv(diagnostics_path, index=False)
    witness = support.copy()
    witness["weight"] = weights
    witness.to_csv(witness_path, index=False)
    maximum_residual = float(np.abs(residual).max())
    effective_support = float(1.0 / np.sum(weights**2))
    summary = {
        "tag": tag,
        "inside_convex_hull_at_tolerance": maximum_residual <= tolerance + 1e-9,
        "tolerance": tolerance,
        "minimum_maximum_absolute_residual": maximum_residual,
        "witness_rmse": float(np.sqrt(np.mean(residual**2))),
        "witness_effective_support": effective_support,
        "witness_max_weight": float(weights.max()),
        "support_points": n_support,
        "constraint_cells": n_cells,
        "outside_individual_cell_ranges": sum(row["outside_cell_range"] for row in cells),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    return {
        **summary,
        "diagnostics_path": str(diagnostics_path),
        "summary_path": str(summary_path),
        "witness_weights_path": str(witness_path),
    }


def fit_generalized_targets(
    support_path: Path,
    targets_path: Path,
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    rho_values: list[float],
    *,
    joint_features_path: Path | None = None,
    allow_conditional_independence: bool = False,
    minimum_effective_support: float | None = None,
    maximum_weight: float | None = None,
    require_convergence: bool = False,
) -> dict[str, Any]:
    support_frame = pd.read_csv(support_path)
    support = _support_identity(support_frame)
    target_artifact = read_json(targets_path)
    audit_targets(targets_path, metadata)
    marginal_mats: dict[str, np.ndarray] = {}
    for item, group in support_frame.groupby("item", sort=False):
        pivot = group.pivot(index="support_id", columns="option_index", values="probability")
        pivot = pivot.reindex(support["support_id"]).sort_index(axis=1)
        marginal_mats[str(item)] = pivot.to_numpy(dtype=float)
    direct_joint: dict[str, np.ndarray] = {}
    if joint_features_path:
        joint_frame = pd.read_csv(joint_features_path)
        for target_id, group in joint_frame.groupby("target_id", sort=False):
            pivot = group.pivot(index="support_id", columns="cell_index", values="probability")
            pivot = pivot.reindex(support["support_id"]).sort_index(axis=1)
            direct_joint[str(target_id)] = pivot.to_numpy(dtype=float)
    matrices: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    constraint_rows: list[dict[str, Any]] = []
    accepted_targets = [target for target in target_artifact.get("targets", []) if target.get("status") == "accepted"]
    if not accepted_targets:
        raise UmrissError("targets_empty", "No accepted targets are available for fitting.")
    column_start = 0
    for target in accepted_targets:
        target_id = str(target["target_id"])
        values = np.asarray(target["values"], dtype=float).reshape(-1)
        confidence = float(target.get("confidence_weight", 1.0))
        if confidence <= 0:
            raise UmrissError("targets_invalid", f"Target {target_id} has nonpositive confidence_weight.")
        if target["type"] in {"marginal", "checkbox_marginal"}:
            item = str(target["items"][0])
            if item not in marginal_mats:
                raise UmrissError("support_missing_item", f"Support bank lacks probabilities for target item {item}.")
            matrix = marginal_mats[item]
            method = "marginal"
        else:
            left, right = [str(item) for item in target["items"]]
            if target_id in direct_joint:
                matrix = direct_joint[target_id]
                method = "direct_joint"
            elif allow_conditional_independence or target.get("feature_method") == "conditional_independence":
                if left not in marginal_mats or right not in marginal_mats:
                    raise UmrissError("support_missing_item", f"Support bank lacks items needed for joint target {target_id}.")
                matrix = np.einsum("si,sj->sij", marginal_mats[left], marginal_mats[right]).reshape(len(support), -1)
                method = "conditional_independence"
            else:
                raise UmrissError(
                    "joint_features_missing",
                    f"Joint target {target_id} needs directly elicited features.",
                    hint="Pass --joint-features, or explicitly pass --allow-conditional-independence.",
                )
        if matrix.shape[1] != len(values):
            raise UmrissError("shape_mismatch", f"Target {target_id} has {len(values)} cells but support features have {matrix.shape[1]}.")
        scale = np.sqrt(confidence)
        matrices.append(matrix * scale)
        vectors.append(values * scale)
        constraint_rows.append(
            {
                "target_id": target_id,
                "type": target["type"],
                "items": json.dumps(target["items"]),
                "feature_method": method,
                "confidence_weight": confidence,
                "column_start": column_start,
                "column_count": len(values),
                "source_kind": (target.get("source") or {}).get("kind"),
            }
        )
        column_start += len(values)
    X = np.column_stack(matrices)
    y = np.concatenate(vectors)
    base = np.ones(len(support)) / len(support)
    fits = {rho: entropy_calibration_fit(X, y, base, rho) for rho in rho_values}

    def satisfies_declared_gates(candidate: Any) -> bool:
        return (
            (minimum_effective_support is None or candidate.effective_support >= minimum_effective_support)
            and (maximum_weight is None or float(candidate.weights.max()) <= maximum_weight)
            and (not require_convergence or candidate.converged)
        )

    eligible = [pair for pair in fits.items() if satisfies_declared_gates(pair[1])]
    selection_pool = eligible or list(fits.items())
    selected_rho, fit = min(
        selection_pool,
        key=lambda pair: pair[1].held_in_residual + 0.002 / max(pair[1].effective_support, 1.0),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{tag}_weights.csv"
    diagnostics_path = out_dir / f"{tag}_fit_diagnostics.csv"
    constraints_path = out_dir / f"{tag}_constraint_diagnostics.csv"
    weights = support.copy()
    weights["weight"] = fit.weights
    weights.to_csv(weights_path, index=False)
    residual_rows: list[dict[str, Any]] = []
    for row, matrix, target in zip(constraint_rows, matrices, accepted_targets):
        confidence = float(row["confidence_weight"])
        unscaled = matrix / np.sqrt(confidence)
        prediction = unscaled.T @ fit.weights
        truth = np.asarray(target["values"], dtype=float)
        residual_rows.append(
            {
                **row,
                "target": json.dumps(truth.tolist()),
                "prediction": json.dumps(prediction.tolist()),
                "rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
                "max_absolute_residual": float(np.abs(prediction - truth).max()),
            }
        )
    pd.DataFrame(residual_rows).to_csv(constraints_path, index=False)
    gate_violations: list[str] = []
    if minimum_effective_support is not None and fit.effective_support < minimum_effective_support:
        gate_violations.append("minimum_effective_support")
    if maximum_weight is not None and float(fit.weights.max()) > maximum_weight:
        gate_violations.append("maximum_weight")
    if require_convergence and not fit.converged:
        gate_violations.append("convergence")
    gates_pass = not gate_violations
    pd.DataFrame(
        [
            {
                "tag": tag,
                "selected_rho": selected_rho,
                "weighted_constraint_residual": fit.held_in_residual,
                "effective_support": fit.effective_support,
                "max_weight": float(fit.weights.max()),
                "n_support_valid": len(support),
                "targets": len(accepted_targets),
                "converged": fit.converged,
                "minimum_effective_support_gate": minimum_effective_support,
                "maximum_weight_gate": maximum_weight,
                "require_convergence": require_convergence,
                "gates_pass": gates_pass,
                "gate_violations": json.dumps(gate_violations),
            }
        ]
    ).to_csv(diagnostics_path, index=False)
    return {
        "weights_path": str(weights_path),
        "diagnostics_path": str(diagnostics_path),
        "constraint_diagnostics_path": str(constraints_path),
        "targets": len(accepted_targets),
        "selected_rho": selected_rho,
        "effective_support": fit.effective_support,
        "maximum_weight": float(fit.weights.max()),
        "converged": fit.converged,
        "gates_pass": gates_pass,
        "gate_violations": gate_violations,
    }
