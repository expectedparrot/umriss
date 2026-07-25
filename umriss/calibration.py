from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .errors import UmrissError


@dataclass
class FitResult:
    weights: np.ndarray
    held_in_residual: float
    effective_support: float
    converged: bool


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def entropy_calibration_fit(X: np.ndarray, y: np.ndarray, base_weights: np.ndarray, rho: float, maxiter: int = 700) -> FitResult:
    base = np.clip(base_weights.astype(float), 1e-12, None)
    base = base / base.sum()
    theta0 = np.log(base)

    def softmax(theta: np.ndarray) -> np.ndarray:
        z = theta - theta.max()
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        pi = softmax(theta)
        diff = X.T @ pi - y
        pi_safe = np.clip(pi, 1e-14, None)
        entropy = float(np.sum(pi_safe * np.log(pi_safe / base)))
        value = float(np.dot(diff, diff) + rho * entropy)
        grad_pi = 2.0 * (X @ diff) + rho * (np.log(pi_safe / base) + 1.0)
        grad_theta = pi * (grad_pi - np.dot(pi, grad_pi))
        return value, grad_theta

    opt = minimize(
        fun=lambda th: objective_and_grad(th)[0],
        x0=theta0,
        jac=lambda th: objective_and_grad(th)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )
    pi = softmax(opt.x)
    residual = float(np.sqrt(np.mean((X.T @ pi - y) ** 2)))
    effective_support = float(1.0 / np.sum(pi**2))
    return FitResult(pi, residual, effective_support, bool(opt.success))


def load_support_matrix(support_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = pd.read_csv(support_path)
    if df.empty:
        raise UmrissError("support_empty", f"Support probability file is empty: {support_path}.")
    support = df[["support_id", "job_id"]].drop_duplicates().reset_index(drop=True)
    mats: dict[str, np.ndarray] = {}
    for item, group in df.groupby("item", sort=False):
        pivot = group.pivot(index="support_id", columns="option_index", values="probability")
        pivot = pivot.reindex(support["support_id"]).sort_index(axis=1)
        mats[str(item)] = pivot.to_numpy(dtype=float)
    return support, mats


def fit_weights(mats: dict[str, np.ndarray], truth: dict[str, np.ndarray], held_in: list[str], rho_values: list[float], base: np.ndarray | None = None) -> tuple[float, FitResult]:
    if not held_in:
        raise UmrissError("invalid_input", "At least one held-in item is required.")
    n = next(iter(mats.values())).shape[0]
    base_weights = np.ones(n) / n if base is None else base
    y = np.concatenate([truth[item] for item in held_in])
    X = np.column_stack([mats[item] for item in held_in])
    fits = {rho: entropy_calibration_fit(X, y, base_weights, rho) for rho in rho_values}
    return min(fits.items(), key=lambda kv: kv[1].held_in_residual + 0.002 / max(kv[1].effective_support, 1.0))


def write_fit_outputs(
    support: pd.DataFrame,
    mats: dict[str, np.ndarray],
    metadata: dict[str, Any],
    tag: str,
    heldout_item: str,
    selected_rho: float,
    fit: FitResult,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{tag}_weights.csv"
    diag_path = out_dir / f"{tag}_fit_diagnostics.csv"
    pred_path = out_dir / f"{tag}_predictions.csv"
    weights = support.copy()
    weights["weight"] = fit.weights
    weights.to_csv(weights_path, index=False)
    pd.DataFrame(
        [
            {
                "tag": tag,
                "fit_id": tag,
                "heldout_item": heldout_item,
                "selected_rho": selected_rho,
                "held_in_residual": fit.held_in_residual,
                "effective_support": fit.effective_support,
                "max_weight": float(fit.weights.max()),
                "top10_weight_share": float(np.sort(fit.weights)[-10:].sum()),
                "n_support_valid": len(support),
                "converged": fit.converged,
            }
        ]
    ).to_csv(diag_path, index=False)
    rows = []
    for item, mat in mats.items():
        pred = mat.T @ fit.weights
        labels = metadata["items"][item].get("option_labels", metadata.get("option_labels", []))
        codes = metadata["items"][item].get("option_codes", metadata.get("option_codes", list(range(1, len(labels) + 1))))
        for idx, value in enumerate(pred):
            rows.append({"tag": tag, "fit_id": tag, "item": item, "option_index": idx, "option_code": codes[idx], "option_label": labels[idx], "prediction": float(value)})
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    return {"weights_path": str(weights_path), "diagnostics_path": str(diag_path), "predictions_path": str(pred_path)}
