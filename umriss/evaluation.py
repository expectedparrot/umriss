from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import fit_weights, load_support_matrix, rmse
from .metadata import marginals_from_metadata, weighted_truth_from_respondents
from .parsing import parse_support


def parse_vec(value: str | list[float]) -> np.ndarray:
    arr = np.array(value if isinstance(value, list) else ast.literal_eval(value), dtype=float)
    total = arr.sum()
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def load_priors(one_shot_path: Path | None, two_step_path: Path | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    def load(path: Path | None, candidates: list[str]) -> dict[str, np.ndarray]:
        if path is None:
            return {}
        df = pd.read_csv(path)
        item_col = "item" if "item" in df.columns else "holdout"
        pred_col = next(col for col in candidates if col in df.columns)
        return {str(getattr(row, item_col)): parse_vec(getattr(row, pred_col)) for row in df.itertuples(index=False)}

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
    if respondents_path:
        truth = weighted_truth_from_respondents(metadata, respondents_path)
    elif "truth" in metadata:
        truth = marginals_from_metadata(metadata)
    else:
        raise ValueError("metadata truth or respondents_path is required")
    one_shot, conditioned = load_priors(one_shot_path, two_step_path)
    source = "Gallup" if str(metadata["wave"]).startswith("GALLUP") else "Pew"
    battery_label = f"{source} {metadata['wave']} {metadata['battery']}"
    rows = []
    diag_rows = []
    items = list(metadata["items"])
    for holdout in items:
        held_in = [item for item in items if item != holdout]
        selected_rho, fit = fit_weights(mats, truth, held_in, rho_values)
        pred = mats[holdout].T @ fit.weights
        unweighted = mats[holdout].mean(axis=0)
        methods = [
            ("generated support mixture", pred),
            ("unweighted archetype bank", unweighted),
            ("uniform", np.ones(len(truth[holdout])) / len(truth[holdout])),
        ]
        if holdout in one_shot:
            methods.append(("unconditioned one-shot", one_shot[holdout]))
        if holdout in conditioned:
            methods.append(("conditioned one-shot", conditioned[holdout]))
        for method, vec in methods:
            rows.append(
                {
                    "tag": tag,
                    "battery": battery_label,
                    "holdout": holdout,
                    "item_text": metadata["items"][holdout]["item_text"],
                    "method": method,
                    "rmse": rmse(vec, truth[holdout]),
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
        .agg(mean_rmse=("rmse", "mean"), median_rmse=("rmse", "median"), max_rmse=("rmse", "max"), items=("holdout", "nunique"), n_support_valid=("n_support_valid", "max"))
        .sort_values("mean_rmse")
    )
    detail_path = out_dir / f"{tag}_generated_support_detail.csv"
    summary_path = out_dir / f"{tag}_generated_support_summary.csv"
    diag_path = out_dir / f"{tag}_generated_support_diagnostics.csv"
    points_path = out_dir / f"{tag}_generated_support_points.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    if parsed_points_path and parsed_points_path.exists():
        shutil.copyfile(parsed_points_path, points_path)
    else:
        support.to_csv(points_path, index=False)
    return {"detail_path": str(detail_path), "summary_path": str(summary_path), "diagnostics_path": str(diag_path), "points_path": str(points_path)}
