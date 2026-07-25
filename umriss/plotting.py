from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import UmrissError
from .jsonlio import write_json

COLORS = {
    "generated support mixture": "#2f7d57",
    "unconditioned one-shot": "#d97706",
    "conditioned direct prediction": "#8b5cf6",
    "structured two-step": "#8b5cf6",
    "unweighted support bank": "#64748b",
    "uniform": "#cbd5e1",
}
LABELS = {
    "generated support mixture": "Marginally weighted twins",
    "unconditioned one-shot": "Direct one-shot",
    "conditioned direct prediction": "Conditioned direct",
    "structured two-step": "Conditioned direct",
    "unweighted support bank": "Unweighted support bank",
    "uniform": "Uniform",
}


def _pyplot() -> Any:
    cache_dir = tempfile.mkdtemp(prefix="umriss-plot-cache-")
    os.environ.setdefault("MPLCONFIGDIR", cache_dir)
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _path(derived: Path, tag: str, suffix: str) -> Path:
    path = derived / f"{tag}_{suffix}.csv"
    if not path.exists():
        raise UmrissError("not_found", f"Required plotting input does not exist: {path}")
    return path


def _finish(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.clear()


def plot_validation(
    derived: Path,
    tag: str,
    out_dir: Path,
    *,
    image_format: str = "svg",
    top_personas: int = 30,
) -> dict[str, Any]:
    if image_format not in {"svg", "png", "pdf"}:
        raise UmrissError("invalid_input", "--format must be svg, png, or pdf.")
    if top_personas < 1:
        raise UmrissError("invalid_input", "--top-personas must be positive.")
    detail = pd.read_csv(_path(derived, tag, "generated_support_detail"))
    summary = pd.read_csv(_path(derived, tag, "generated_support_summary"))
    diagnostics = pd.read_csv(_path(derived, tag, "generated_support_diagnostics"))
    weights = pd.read_csv(_path(derived, tag, "generated_support_weights"))
    uniformity = pd.read_csv(_path(derived, tag, "support_uniformity"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    paths: dict[str, str] = {}

    method_order = [
        method
        for method in [
            "unconditioned one-shot",
            "conditioned direct prediction",
            "structured two-step",
            "generated support mixture",
            "unweighted support bank",
            "uniform",
        ]
        if method in set(summary["method"])
    ]
    summary_plot = summary.set_index("method").loc[method_order]
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.55 * len(method_order))))
    for axis, column, title in [
        (axes[0], "mean_rmse", "Mean held-out RMSE"),
        (axes[1], "mean_kl_divergence", "Mean held-out KL divergence"),
    ]:
        values = summary_plot[column].astype(float)
        labels = [LABELS.get(method, method) for method in summary_plot.index]
        colors = [COLORS.get(method, "#64748b") for method in summary_plot.index]
        axis.barh(labels, values, color=colors)
        axis.invert_yaxis()
        axis.set_title(title)
        axis.set_xlabel("Lower is better")
        axis.bar_label(axis.containers[0], fmt="%.3f", padding=3)
        axis.spines[["top", "right"]].set_visible(False)
    path = out_dir / f"{tag}_method_comparison.{image_format}"
    _finish(fig, path)
    paths["method_comparison"] = str(path)

    item_detail = detail[detail["method"].isin(method_order)].copy()
    pivot = item_detail.pivot(index="holdout", columns="method", values="rmse")
    fig, axis = plt.subplots(figsize=(11, max(4, 0.65 * len(pivot))))
    pivot = pivot[[method for method in method_order if method in pivot.columns]]
    pivot.columns = [LABELS.get(method, method) for method in pivot.columns]
    pivot.plot.barh(ax=axis, color=[COLORS.get(method, "#64748b") for method in method_order if method in item_detail["method"].unique()])
    axis.invert_yaxis()
    axis.set_title("Error when each marginal is completely omitted")
    axis.set_xlabel("RMSE (lower is better)")
    axis.set_ylabel("")
    axis.legend(frameon=False, loc="best")
    axis.spines[["top", "right"]].set_visible(False)
    path = out_dir / f"{tag}_holdout_by_item.{image_format}"
    _finish(fig, path)
    paths["holdout_by_item"] = str(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    diag = diagnostics.sort_values("effective_support")
    axes[0].barh(diag["holdout"], diag["effective_support"], color="#2f7d57")
    axes[0].set_title("Effective number of weighted personas")
    axes[0].set_xlabel("1 / sum of squared weights")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].barh(diag["holdout"], diag["top10_weight_share"], color="#d97706")
    axes[1].set_title("Weight carried by the ten largest personas")
    axes[1].set_xlabel("Share of total weight")
    axes[1].set_xlim(0, 1)
    axes[1].spines[["top", "right"]].set_visible(False)
    path = out_dir / f"{tag}_weight_diagnostics.{image_format}"
    _finish(fig, path)
    paths["weight_diagnostics"] = str(path)

    weight_pivot = weights.pivot(index="support_id", columns="holdout", values="weight").fillna(0.0)
    selected = weight_pivot.max(axis=1).nlargest(min(top_personas, len(weight_pivot))).index
    shown = weight_pivot.loc[selected]
    fig, axis = plt.subplots(figsize=(max(7, 1.15 * len(shown.columns)), max(5, 0.28 * len(shown))))
    image = axis.imshow(shown.to_numpy(), aspect="auto", cmap="YlGn", interpolation="nearest")
    axis.set_xticks(np.arange(len(shown.columns)), shown.columns, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(shown.index)), [str(value) for value in shown.index])
    axis.set_xlabel("Marginal omitted during fitting")
    axis.set_ylabel("Persona support ID")
    axis.set_title(f"Persona weights: top {len(shown)} by maximum fold weight")
    fig.colorbar(image, ax=axis, label="Mixture weight", shrink=0.8)
    path = out_dir / f"{tag}_persona_weights.{image_format}"
    _finish(fig, path)
    paths["persona_weights"] = str(path)

    fig, axis = plt.subplots(figsize=(9, max(3.5, 0.55 * len(uniformity))))
    uniformity = uniformity.sort_values("max_absolute_deviation")
    axis.barh(uniformity["item"], uniformity["max_absolute_deviation"], color="#2f7d57")
    axis.set_title("Equal-weight support coverage before calibration")
    axis.set_xlabel("Maximum absolute deviation from a uniform marginal")
    axis.set_ylabel("")
    axis.spines[["top", "right"]].set_visible(False)
    path = out_dir / f"{tag}_support_uniformity.{image_format}"
    _finish(fig, path)
    paths["support_uniformity"] = str(path)

    manifest = {
        "kind": "umriss_loo_plots",
        "tag": tag,
        "source_dir": str(derived),
        "format": image_format,
        "plots": paths,
        "definitions": {
            "method_comparison": "Mean held-out RMSE and KL divergence across leave-one-out folds.",
            "holdout_by_item": "RMSE for every method and completely omitted survey item.",
            "weight_diagnostics": "Effective support and concentration of fitted persona weights.",
            "persona_weights": "Fold-specific mixture weights for the most influential generated personas.",
            "support_uniformity": "Distance of the unweighted support bank from uniform item marginals.",
        },
    }
    manifest_path = out_dir / f"{tag}_plots.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "plots": paths, "count": len(paths)}
