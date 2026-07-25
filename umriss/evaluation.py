from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .balancing import diversity_metrics, uniformity_rows
from .calibration import fit_weights, load_support_matrix, rmse
from .metadata import marginals_from_metadata, weighted_truth_from_respondents
from .parsing import parse_support


def parse_vec(value: str | list[float]) -> np.ndarray:
    arr = np.array(value if isinstance(value, list) else ast.literal_eval(value), dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def cross_entropy(truth: np.ndarray, prediction: np.ndarray) -> float:
    pred = np.clip(np.asarray(prediction, dtype=float), 1e-12, 1.0)
    target = np.asarray(truth, dtype=float)
    return float(-np.sum(target * np.log(pred)))


def kl_divergence(truth: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(truth, dtype=float)
    target_safe = np.clip(target, 1e-12, 1.0)
    return float(np.sum(target * np.log(target_safe / np.clip(prediction, 1e-12, 1.0))))


def load_priors(
    one_shot_path: Path | None, two_step_path: Path | None, metadata: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_variable = {meta.get("variable"): item for item, meta in metadata["items"].items()}
    by_text = {meta.get("item_text"): item for item, meta in metadata["items"].items()}

    def load(path: Path | None, candidates: list[str]) -> dict[str, np.ndarray]:
        if path is None:
            return {}
        df = pd.read_csv(path)
        item_col = "item" if "item" in df.columns else "holdout"
        pred_col = next(col for col in candidates if col in df.columns)
        result = {}
        for _, row in df.iterrows():
            raw_item = str(row[item_col])
            item = raw_item if raw_item in metadata["items"] else by_variable.get(row.get("variable")) or by_text.get(row.get("item_text"))
            if item:
                result[item] = parse_vec(row[pred_col])
        return result

    return (
        load(one_shot_path, ["prediction", "one_shot_prior", "unconditioned_one_shot_pred"]),
        load(two_step_path, ["two_step_prior", "conditioned_one_shot_pred", "prediction"]),
    )


def run_loo(
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    *,
    raw_path: Path | None = None,
    support_path: Path | None = None,
    respondents_path: Path | None = None,
    one_shot_path: Path | None = None,
    two_step_path: Path | None = None,
    rho_values: list[float] | None = None,
    uniform_tolerance: float = 0.05,
    max_duplicate_fraction: float = 0.05,
    min_joint_pattern_fraction: float = 0.75,
    allow_nonuniform_support: bool = False,
) -> dict[str, str]:
    rho_values = rho_values or [0.0003, 0.001, 0.003, 0.01, 0.03]
    parsed_points_path: Path | None = None
    if support_path is None:
        if raw_path is None:
            raise ValueError("raw_path or support_path is required")
        parsed = parse_support(raw_path, metadata, tag, out_dir)
        support_path = Path(str(parsed["probabilities_path"]))
        parsed_points_path = Path(str(parsed["points_path"]))
    support, mats = load_support_matrix(support_path)
    uniformity, uniformity_passes = uniformity_rows(support_path, metadata, uniform_tolerance)
    if not uniformity_passes and not allow_nonuniform_support:
        failed = ", ".join(uniformity.loc[~uniformity["passes"], "item"])
        worst = float(uniformity["max_absolute_deviation"].max())
        raise ValueError(
            f"SUPPORT_NOT_UNIFORM: equal-weight support marginals fail tolerance {uniform_tolerance:g}; "
            f"worst deviation is {worst:.6f}; failed items: {failed}. "
            "Run `umriss support augment-uniform` and measure again."
        )
    diversity = diversity_metrics(support_path, metadata)
    if not allow_nonuniform_support and (
        diversity["duplicate_fraction"] > max_duplicate_fraction + 1e-12
        or diversity["joint_pattern_fraction"] < min_joint_pattern_fraction - 1e-12
    ):
        raise ValueError(
            f"SUPPORT_NOT_DIVERSE: duplicate fraction is {diversity['duplicate_fraction']:.6f} "
            f"(maximum {max_duplicate_fraction:g}); joint-pattern fraction is "
            f"{diversity['joint_pattern_fraction']:.6f} (minimum {min_joint_pattern_fraction:g}). "
            "Rebuild with `--preset uniform-patterns` or add genuinely distinct support points."
        )
    if respondents_path:
        truth = weighted_truth_from_respondents(metadata, respondents_path)
    elif "truth" in metadata:
        truth = marginals_from_metadata(metadata)
    else:
        raise ValueError("metadata truth or respondents_path is required")
    one_shot, conditioned = load_priors(one_shot_path, two_step_path, metadata)
    source = "Gallup" if str(metadata["wave"]).startswith("GALLUP") else "Pew"
    battery_label = f"{source} {metadata['wave']} {metadata['battery']}"
    rows = []
    diag_rows = []
    weight_rows = []
    items = list(metadata["items"])
    for holdout in items:
        held_in = [item for item in items if item != holdout]
        selected_rho, fit = fit_weights(mats, truth, held_in, rho_values)
        order = np.argsort(-fit.weights)
        ranks = np.empty(len(fit.weights), dtype=int)
        ranks[order] = np.arange(1, len(fit.weights) + 1)
        for idx, support_row in support.reset_index(drop=True).iterrows():
            weight_rows.append(
                {
                    "tag": tag,
                    "holdout": holdout,
                    "support_id": support_row["support_id"],
                    "job_id": support_row["job_id"],
                    "weight": float(fit.weights[idx]),
                    "rank": int(ranks[idx]),
                }
            )
        pred = mats[holdout].T @ fit.weights
        unweighted = mats[holdout].mean(axis=0)
        methods = [
            ("generated support mixture", pred),
            ("unweighted support bank", unweighted),
            ("uniform", np.ones(len(truth[holdout])) / len(truth[holdout])),
        ]
        if holdout in one_shot:
            methods.append(("unconditioned one-shot", one_shot[holdout]))
        if holdout in conditioned:
            methods.append(("conditioned direct prediction", conditioned[holdout]))
        for method, vec in methods:
            rows.append(
                {
                    "tag": tag,
                    "battery": battery_label,
                    "holdout": holdout,
                    "item_text": metadata["items"][holdout]["item_text"],
                    "method": method,
                    "rmse": rmse(vec, truth[holdout]),
                    "kl_divergence": kl_divergence(truth[holdout], vec),
                    "cross_entropy": cross_entropy(truth[holdout], vec),
                    "target_entropy": cross_entropy(truth[holdout], truth[holdout]),
                    "prediction": json.dumps(np.round(vec, 6).tolist()),
                    "truth": json.dumps(np.round(truth[holdout], 6).tolist()),
                    "selected_rho": selected_rho if method == "generated support mixture" else np.nan,
                    "held_in_residual": fit.held_in_residual if method == "generated support mixture" else np.nan,
                    "effective_support": fit.effective_support if method == "generated support mixture" else np.nan,
                    "n_support_valid": len(support),
                }
            )
        diag_rows.append(
            {
                "tag": tag,
                "holdout": holdout,
                "selected_rho": selected_rho,
                "held_in_residual": fit.held_in_residual,
                "effective_support": fit.effective_support,
                "max_weight": float(fit.weights.max()),
                "top10_weight_share": float(np.sort(fit.weights)[-10:].sum()),
                "n_support_valid": len(support),
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["tag", "method"], as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
            max_rmse=("rmse", "max"),
            mean_kl_divergence=("kl_divergence", "mean"),
            median_kl_divergence=("kl_divergence", "median"),
            max_kl_divergence=("kl_divergence", "max"),
            mean_cross_entropy=("cross_entropy", "mean"),
            mean_target_entropy=("target_entropy", "mean"),
            items=("holdout", "nunique"),
            n_support_valid=("n_support_valid", "max"),
        )
        .sort_values("mean_rmse")
    )
    detail_path = out_dir / f"{tag}_generated_support_detail.csv"
    summary_path = out_dir / f"{tag}_generated_support_summary.csv"
    diag_path = out_dir / f"{tag}_generated_support_diagnostics.csv"
    points_path = out_dir / f"{tag}_generated_support_points.csv"
    weights_path = out_dir / f"{tag}_generated_support_weights.csv"
    uniformity_path = out_dir / f"{tag}_support_uniformity.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)
    uniformity_output_rows = []
    for item in items:
        prediction = mats[item].mean(axis=0)
        target = np.ones(len(prediction)) / len(prediction)
        uniformity_output_rows.append(
            {
                "tag": tag,
                "item": item,
                "item_text": metadata["items"][item]["item_text"],
                "n_support_valid": len(support),
                "equal_weight_prediction": json.dumps(np.round(prediction, 6).tolist()),
                "uniform_target": json.dumps(np.round(target, 6).tolist()),
                "rmse_from_uniform": rmse(prediction, target),
                "max_absolute_deviation": float(np.max(np.abs(prediction - target))),
            }
        )
    pd.DataFrame(uniformity_output_rows).to_csv(uniformity_path, index=False)
    if parsed_points_path and parsed_points_path.exists():
        shutil.copyfile(parsed_points_path, points_path)
    else:
        support.to_csv(points_path, index=False)
    return {
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "diagnostics_path": str(diag_path),
        "points_path": str(points_path),
        "weights_path": str(weights_path),
        "uniformity_path": str(uniformity_path),
    }
