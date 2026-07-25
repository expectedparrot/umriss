from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .calibration import load_support_matrix
from .errors import UmrissError
from .jsonlio import read_jsonl
from .metadata import item_option_codes, item_option_labels


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise UmrissError("not_found", f"File does not exist: {path}.")
    if path.suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(path))
    return pd.read_csv(path)


def inspect_prompts(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    df = pd.DataFrame(rows)
    data: dict[str, Any] = {"path": str(path), "kind": "prompts", "rows": len(rows), "columns": list(df.columns)}
    if not df.empty:
        if "variant" in df.columns:
            data["variants"] = df["variant"].fillna("").value_counts().to_dict()
        if "support_id" in df.columns:
            data["support_points"] = int(df["support_id"].nunique())
        if "job_id" in df.columns:
            data["jobs"] = int(df["job_id"].nunique())
    return data


def inspect_raw(path: Path) -> dict[str, Any]:
    df = _read_table(path)
    job_col = "scenario.job_id" if "scenario.job_id" in df.columns else "job_id" if "job_id" in df.columns else None
    response_col = "answer.resp" if "answer.resp" in df.columns else "response" if "response" in df.columns else None
    return {
        "path": str(path),
        "kind": "raw",
        "rows": len(df),
        "columns": list(df.columns),
        "jobs": int(df[job_col].nunique()) if job_col else None,
        "has_response_column": bool(response_col),
        "empty_responses": int(df[response_col].isna().sum()) if response_col else None,
    }


def inspect_bank(path: Path) -> dict[str, Any]:
    df = _read_table(path)
    required = {"support_id", "job_id", "item", "option_index", "probability"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise UmrissError("invalid_input", f"Support bank is missing columns: {', '.join(missing)}.")
    probs = df["probability"].astype(float)
    by_item = df.groupby("item")["option_index"].nunique().to_dict()
    return {
        "path": str(path),
        "kind": "bank",
        "rows": len(df),
        "support_points": int(df["support_id"].nunique()),
        "jobs": int(df["job_id"].nunique()),
        "items": int(df["item"].nunique()),
        "option_counts": {str(k): int(v) for k, v in by_item.items()},
        "min_probability": float(probs.min()),
        "max_probability": float(probs.max()),
    }


def inspect_diagnostics(path: Path) -> dict[str, Any]:
    df = _read_table(path)
    status_counts = df["status"].fillna("").value_counts().to_dict() if "status" in df.columns else {}
    code_counts = df["code"].fillna("").value_counts().to_dict() if "code" in df.columns else {}
    return {"path": str(path), "kind": "diagnostics", "rows": len(df), "status": status_counts, "codes": code_counts}


def inspect_summary(path: Path) -> dict[str, Any]:
    df = _read_table(path)
    data: dict[str, Any] = {"path": str(path), "kind": "summary", "rows": len(df), "columns": list(df.columns)}
    if {"method", "mean_rmse"}.issubset(df.columns):
        ordered = df.sort_values("mean_rmse")
        data["best_method"] = str(ordered.iloc[0]["method"]) if len(ordered) else None
        data["methods"] = ordered[["method", "mean_rmse"]].to_dict(orient="records")
    return data


def inspect_artifact(*, prompts: Path | None, raw: Path | None, bank: Path | None, diagnostics: Path | None, summary: Path | None) -> dict[str, Any]:
    selected = [p for p in [prompts, raw, bank, diagnostics, summary] if p is not None]
    if len(selected) != 1:
        raise UmrissError("invalid_input", "Pass exactly one of --prompts, --raw, --bank, --diagnostics, or --summary.")
    if prompts:
        return inspect_prompts(prompts)
    if raw:
        return inspect_raw(raw)
    if bank:
        return inspect_bank(bank)
    if diagnostics:
        return inspect_diagnostics(diagnostics)
    if summary:
        return inspect_summary(summary)
    raise AssertionError("unreachable")


def parse_run_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec:
        return spec, "", spec
    tag, label = spec.split("=", 1)
    if ":" in label:
        battery, bank = label.split(":", 1)
    else:
        battery, bank = "", label
    return tag, battery, bank


BATTERY_DESIGNED_BANKS = [
    ("W154 DIFF1", "Generic demographic", "w154_archetype_support_n96"),
    ("W154 DIFF1", "Battery-designed", "w154_battery_anchor_support_n96"),
    ("W157 SKILLIMP", "Generic demographic", "w157_archetype_support_n96"),
    ("W157 SKILLIMP", "Battery-designed", "w157_latent_anchor_support_n96"),
    ("W158 CCPOLICY", "Generic demographic", "w158_archetype_support_n96"),
    ("W158 CCPOLICY", "Battery-designed", "w158_battery_anchor_support_n96"),
    ("W163 SM9", "Generic demographic", "w163_archetype_support_n96"),
    ("W163 SM9", "Battery-designed", "w163_battery_anchor_support_n96"),
    ("Gallup well-being", "Generic demographic", "gallup_wellbeing_archetype_support_n96"),
    ("Gallup well-being", "Battery-designed", "gallup_wellbeing_battery_anchor_support_n96"),
    ("Gallup remote/COVID", "Generic demographic", "gallup_remote_covid_archetype_support_n96"),
    ("Gallup remote/COVID", "Battery-designed", "gallup_remote_covid_battery_anchor_support_n96"),
]


PATTERN_COVERAGE_BANKS = [
    {
        "battery": "W154 DIFF1",
        "generic": "w154_archetype_support_n96",
        "pattern": "w154_pattern_coverage_support_n96",
        "overlay": "w154_pattern_coverage_overlay_shrinkage_summary.csv",
        "designed": "w154_battery_anchor_support_n96",
    },
    {
        "battery": "W157 SKILLIMP",
        "generic": "w157_archetype_support_n96",
        "pattern": "w157_pattern_coverage_support_n96",
        "overlay": "w157_pattern_coverage_overlay_shrinkage_summary.csv",
        "designed": "w157_designed_support_n96",
    },
    {
        "battery": "W158 CCPOLICY",
        "generic": "w158_archetype_support_n96",
        "pattern": "w158_pattern_coverage_support_n96",
        "overlay": "w158_pattern_coverage_overlay_shrinkage_summary.csv",
        "designed": "w158_battery_anchor_support_n96",
    },
    {
        "battery": "W163 SM9",
        "generic": "w163_archetype_support_n96",
        "pattern": "w163_pattern_coverage_support_n96",
        "overlay": "w163_pattern_coverage_overlay_shrinkage_summary.csv",
        "designed": "w163_battery_anchor_support_n96",
    },
    {
        "battery": "Gallup well-being",
        "generic": "gallup_wellbeing_archetype_support_n96",
        "pattern": "gallup_wellbeing_pattern_coverage_support_n96",
        "overlay": "gallup_wellbeing_pattern_coverage_overlay_shrinkage_summary.csv",
        "designed": "gallup_wellbeing_battery_anchor_support_n96",
    },
    {
        "battery": "Gallup remote/COVID",
        "generic": "gallup_remote_covid_archetype_support_n96",
        "pattern": "gallup_remote_covid_pattern_coverage_support_n96",
        "overlay": "gallup_remote_covid_pattern_coverage_overlay_shrinkage_summary.csv",
        "designed": "gallup_remote_covid_battery_anchor_support_n96",
    },
]


def _summary_path(derived_dir: Path, tag: str) -> Path:
    return derived_dir / f"{tag}_generated_support_summary.csv"


def _detail_path(derived_dir: Path, tag: str) -> Path:
    return derived_dir / f"{tag}_generated_support_detail.csv"


def _sibling_path(out_path: Path, suffix: str) -> Path:
    stem = out_path.stem
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    return out_path.with_name(f"{stem}_{suffix}{out_path.suffix}")


def compare_runs(run_specs: list[str], derived_dir: Path, out_path: Path, comparison_group: str = "default") -> dict[str, Any]:
    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for spec in run_specs:
        tag, battery, bank = parse_run_spec(spec)
        path = derived_dir / f"{tag}_generated_support_summary.csv"
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path)
        df.insert(0, "bank", bank)
        df.insert(0, "battery", battery)
        df.insert(0, "comparison_group", comparison_group)
        if "tag" not in df.columns:
            df.insert(3, "tag", tag)
        rows.append(df)
    if missing:
        raise UmrissError("not_found", "Some summary files were not found.", context={"missing": missing})
    if not rows:
        raise UmrissError("invalid_input", "At least one --run is required.")
    out = pd.concat(rows, ignore_index=True)
    preferred = ["comparison_group", "battery", "bank", "tag", "method", "mean_rmse", "median_rmse", "max_rmse", "items", "n_support_valid"]
    columns = [col for col in preferred if col in out.columns] + [col for col in out.columns if col not in preferred]
    out = out[columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    best = out.sort_values(["battery", "mean_rmse"]).groupby("battery", dropna=False).head(1)
    return {"path": str(out_path), "runs": len(run_specs), "rows": len(out), "best": best[["battery", "bank", "method", "mean_rmse"]].to_dict(orient="records")}


def battery_designed_recipe(derived_dir: Path, out_path: Path) -> dict[str, Any]:
    details = []
    for battery, bank, tag in BATTERY_DESIGNED_BANKS:
        df = pd.read_csv(_detail_path(derived_dir, tag))
        df["comparison_battery"] = battery
        df["bank"] = bank
        df["source_tag"] = tag
        details.append(df)
    all_details = pd.concat(details, ignore_index=True)
    all_details["display_method"] = all_details["method"].replace(
        {
            "generated support mixture": "Weighted support",
            "unweighted support bank": "Unweighted support",
            "unconditioned one-shot": "One-shot",
            "conditioned one-shot": "Conditioned one-shot",
            "uniform": "Uniform",
        }
    )
    support = all_details[all_details["display_method"] == "Weighted support"].copy()
    direct = (
        all_details[all_details["display_method"].isin(["One-shot", "Conditioned one-shot"])]
        .sort_values("rmse")
        .groupby(["comparison_battery", "holdout"], as_index=False)
        .first()
    )
    direct["bank"] = "Best direct prior (ex post)"
    direct["display_method"] = "Best direct prior (ex post)"
    direct["source_tag"] = "direct_prior"
    plot_detail = pd.concat([support, direct], ignore_index=True)
    summary = (
        plot_detail.groupby(["comparison_battery", "bank"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), se=("rmse", lambda x: x.std(ddof=1) / (len(x) ** 0.5) if len(x) > 1 else 0.0), items=("holdout", "nunique"))
    )
    wide = summary.pivot(index="comparison_battery", columns="bank", values="mean_rmse").reset_index()
    wide["designed_minus_generic"] = wide["Battery-designed"] - wide["Generic demographic"]
    wide["designed_minus_best_direct"] = wide["Battery-designed"] - wide["Best direct prior (ex post)"]

    detail_path = _sibling_path(out_path, "detail")
    wide_path = _sibling_path(out_path, "wide")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_detail.to_csv(detail_path, index=False)
    summary.to_csv(out_path, index=False)
    wide.to_csv(wide_path, index=False)
    return {"summary_path": str(out_path), "detail_path": str(detail_path), "wide_path": str(wide_path), "rows": len(summary)}


def _generated_row(derived_dir: Path, tag: str, label: str, battery: str) -> dict[str, Any]:
    df = pd.read_csv(_summary_path(derived_dir, tag))
    row = df[df["method"].eq("generated support mixture")].iloc[0]
    return {"battery": battery, "method": label, "rmse": float(row["mean_rmse"]), "overlay_base_mass": float("nan"), "final_overlay_weight": float("nan")}


def _direct_rows(derived_dir: Path, tag: str, battery: str) -> list[dict[str, Any]]:
    df = pd.read_csv(_summary_path(derived_dir, tag))
    rows = []
    for method, label in [("unconditioned one-shot", "Unconditioned prior"), ("conditioned one-shot", "Conditioned prior")]:
        hit = df[df["method"].eq(method)]
        if not hit.empty:
            rows.append({"battery": battery, "method": label, "rmse": float(hit.iloc[0]["mean_rmse"]), "overlay_base_mass": float("nan"), "final_overlay_weight": float("nan")})
    return rows


def _overlay_row(derived_dir: Path, filename: str, battery: str) -> dict[str, Any]:
    df = pd.read_csv(derived_dir / filename)
    overlay = df[df["method"].str.contains("overlay", case=False, na=False)].copy()
    row = overlay.sort_values(["mean_rmse", "overlay_base_mass"]).iloc[0]
    return {
        "battery": battery,
        "method": "Generic + pattern/coverage overlay",
        "rmse": float(row["mean_rmse"]),
        "overlay_base_mass": float(row["overlay_base_mass"]),
        "final_overlay_weight": float(row["final_overlay_weight"]),
    }


def pattern_coverage_recipe(derived_dir: Path, out_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in PATTERN_COVERAGE_BANKS:
        battery = str(spec["battery"])
        rows.extend(_direct_rows(derived_dir, str(spec["pattern"]), battery))
        rows.append(_generated_row(derived_dir, str(spec["generic"]), "Generic bank", battery))
        rows.append(_generated_row(derived_dir, str(spec["pattern"]), "Pattern/coverage bank", battery))
        rows.append(_overlay_row(derived_dir, str(spec["overlay"]), battery))
        rows.append(_generated_row(derived_dir, str(spec["designed"]), "Battery-designed bank", battery))

    plot = pd.DataFrame(rows)
    wide = plot.pivot_table(index="battery", columns="method", values="rmse", aggfunc="first").reset_index()
    method_cols = [c for c in wide.columns if c != "battery"]
    wide["best_method"] = wide[method_cols].idxmin(axis=1)
    wide["best_rmse"] = wide[method_cols].min(axis=1)
    wide["pattern_beats_generic"] = wide["Pattern/coverage bank"] < wide["Generic bank"]
    wide["overlay_beats_generic"] = wide["Generic + pattern/coverage overlay"] < wide["Generic bank"]
    wide["designed_beats_generic"] = wide["Battery-designed bank"] < wide["Generic bank"]

    plot_path = _sibling_path(out_path, "plot_data")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot.to_csv(plot_path, index=False)
    wide.to_csv(out_path, index=False)
    return {"summary_path": str(out_path), "plot_data_path": str(plot_path), "rows": len(wide)}


def _tag_from_generated_path(path: Path, suffix: str) -> str:
    return path.name[: -len(suffix)]


def _read_generated_tables(derived_dir: Path, suffix: str) -> list[tuple[str, Path, pd.DataFrame]]:
    tables: list[tuple[str, Path, pd.DataFrame]] = []
    for path in sorted(derived_dir.glob(f"*{suffix}")):
        if not path.is_file():
            continue
        tag = _tag_from_generated_path(path, suffix)
        df = pd.read_csv(path)
        if "tag" not in df.columns:
            df.insert(0, "tag", tag)
        tables.append((tag, path, df))
    return tables


def _method_label(method: str) -> str:
    return {
        "generated support mixture": "Weighted support",
        "unweighted support bank": "Unweighted support",
        "unconditioned one-shot": "One-shot",
        "conditioned one-shot": "Conditioned one-shot",
        "uniform": "Uniform",
    }.get(method, method)


def _write_empty_csv(path: Path, columns: list[str]) -> None:
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def build_report_data(derived_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Build a zwill-style report-data bundle from already-derived umriss artifacts."""
    if not derived_dir.exists():
        raise UmrissError("not_found", f"Derived directory does not exist: {derived_dir}.")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_tables = _read_generated_tables(derived_dir, "_generated_support_summary.csv")
    detail_tables = _read_generated_tables(derived_dir, "_generated_support_detail.csv")
    diagnostic_tables = _read_generated_tables(derived_dir, "_generated_support_diagnostics.csv")
    point_tables = _read_generated_tables(derived_dir, "_generated_support_points.csv")

    support_bank_summary_path = out_dir / "support_bank_summary.csv"
    method_comparison_path = out_dir / "method_comparison.csv"
    holdout_detail_path = out_dir / "holdout_detail.csv"
    diagnostics_flags_path = out_dir / "diagnostics_flags.csv"
    support_points_path = out_dir / "support_points.csv"
    prose_facts_json_path = out_dir / "prose_facts.json"
    prose_facts_md_path = out_dir / "prose_facts.md"
    manifest_path = out_dir / "manifest.json"

    if summary_tables:
        summary = pd.concat([df for _, _, df in summary_tables], ignore_index=True)
        if "method_label" not in summary.columns and "method" in summary.columns:
            summary["method_label"] = summary["method"].map(_method_label)
        summary.to_csv(method_comparison_path, index=False)

        generated = summary[summary["method"].eq("generated support mixture")].copy() if "method" in summary.columns else summary.copy()
        generated = generated.sort_values(["tag", "mean_rmse"]) if {"tag", "mean_rmse"}.issubset(generated.columns) else generated
        generated.to_csv(support_bank_summary_path, index=False)
    else:
        summary = pd.DataFrame()
        generated = pd.DataFrame()
        _write_empty_csv(method_comparison_path, ["tag", "method", "mean_rmse", "median_rmse", "max_rmse", "items"])
        _write_empty_csv(support_bank_summary_path, ["tag", "method", "mean_rmse", "median_rmse", "max_rmse", "items"])

    if detail_tables:
        detail = pd.concat([df for _, _, df in detail_tables], ignore_index=True)
        if "method_label" not in detail.columns and "method" in detail.columns:
            detail["method_label"] = detail["method"].map(_method_label)
        detail.to_csv(holdout_detail_path, index=False)
    else:
        _write_empty_csv(holdout_detail_path, ["tag", "holdout", "method", "rmse"])

    diagnostic_rows: list[dict[str, Any]] = []
    if diagnostic_tables:
        diagnostics = pd.concat([df for _, _, df in diagnostic_tables], ignore_index=True)
        for tag, group in diagnostics.groupby("tag", dropna=False):
            row: dict[str, Any] = {"tag": tag, "holdouts": len(group)}
            if "held_in_residual" in group.columns:
                residual = group["held_in_residual"].astype(float)
                row["mean_held_in_residual"] = float(residual.mean())
                row["max_held_in_residual"] = float(residual.max())
            if "effective_support" in group.columns:
                effective = group["effective_support"].astype(float)
                row["mean_effective_support"] = float(effective.mean())
                row["min_effective_support"] = float(effective.min())
            if "selected_rho" in group.columns:
                row["selected_rho_values"] = json.dumps(sorted(group["selected_rho"].dropna().unique().tolist()))
            diagnostic_rows.append(row)
    diagnostics_flags = pd.DataFrame(diagnostic_rows)
    if diagnostics_flags.empty:
        _write_empty_csv(diagnostics_flags_path, ["tag", "holdouts", "mean_effective_support", "max_held_in_residual"])
    else:
        diagnostics_flags.sort_values("tag").to_csv(diagnostics_flags_path, index=False)

    if point_tables:
        points = pd.concat([df for _, _, df in point_tables], ignore_index=True)
        points.to_csv(support_points_path, index=False)
    else:
        _write_empty_csv(support_points_path, ["tag", "support_id", "valid"])

    facts: dict[str, Any] = {
        "derived_dir": str(derived_dir),
        "out_dir": str(out_dir),
        "support_banks": len(summary_tables),
        "summary_rows": int(len(summary)),
        "detail_rows": int(sum(len(df) for _, _, df in detail_tables)),
        "diagnostic_rows": int(sum(len(df) for _, _, df in diagnostic_tables)),
        "best_generated_support": [],
        "best_method_by_tag": [],
    }
    if not generated.empty and {"tag", "mean_rmse"}.issubset(generated.columns):
        best_generated = generated.sort_values("mean_rmse").head(10)
        facts["best_generated_support"] = best_generated[["tag", "mean_rmse"]].to_dict(orient="records")
    if not summary.empty and {"tag", "method", "mean_rmse"}.issubset(summary.columns):
        best_method = summary.sort_values(["tag", "mean_rmse"]).groupby("tag", as_index=False).first()
        facts["best_method_by_tag"] = best_method[["tag", "method", "mean_rmse"]].to_dict(orient="records")

    prose_facts_json_path.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n")
    md_lines = ["# umriss report data", ""]
    md_lines.append(f"- support banks: {facts['support_banks']}")
    md_lines.append(f"- summary rows: {facts['summary_rows']}")
    md_lines.append(f"- holdout detail rows: {facts['detail_rows']}")
    md_lines.append(f"- diagnostic rows: {facts['diagnostic_rows']}")
    if facts["best_generated_support"]:
        md_lines.extend(["", "## Best Generated Support Banks", ""])
        md_lines.append("| tag | mean_rmse |")
        md_lines.append("| --- | ---: |")
        for row in facts["best_generated_support"]:
            md_lines.append(f"| {row['tag']} | {row['mean_rmse']:.6f} |")
    if facts["best_method_by_tag"]:
        md_lines.extend(["", "## Best Method By Tag", ""])
        md_lines.append("| tag | method | mean_rmse |")
        md_lines.append("| --- | --- | ---: |")
        for row in facts["best_method_by_tag"]:
            md_lines.append(f"| {row['tag']} | {row['method']} | {row['mean_rmse']:.6f} |")
    prose_facts_md_path.write_text("\n".join(md_lines) + "\n")

    manifest = {
        "kind": "umriss_report_data",
        "derived_dir": str(derived_dir),
        "out_dir": str(out_dir),
        "inputs": {
            "summary": [str(path) for _, path, _ in summary_tables],
            "detail": [str(path) for _, path, _ in detail_tables],
            "diagnostics": [str(path) for _, path, _ in diagnostic_tables],
            "points": [str(path) for _, path, _ in point_tables],
        },
        "outputs": {
            "support_bank_summary": str(support_bank_summary_path),
            "method_comparison": str(method_comparison_path),
            "holdout_detail": str(holdout_detail_path),
            "diagnostics_flags": str(diagnostics_flags_path),
            "support_points": str(support_points_path),
            "prose_facts_json": str(prose_facts_json_path),
            "prose_facts_md": str(prose_facts_md_path),
            "manifest": str(manifest_path),
        },
        "facts": facts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"manifest_path": str(manifest_path), "outputs": manifest["outputs"], "facts": facts}


def predict_from_weights(support_path: Path, weights_path: Path, metadata: dict[str, Any], items: list[str], out_path: Path) -> dict[str, Any]:
    support, mats = load_support_matrix(support_path)
    weights = pd.read_csv(weights_path)
    if "weight" not in weights.columns:
        raise UmrissError("invalid_input", f"Weights file is missing a weight column: {weights_path}.")
    merged = support.merge(weights[["support_id", "weight"]], on="support_id", how="left")
    if merged["weight"].isna().any():
        raise UmrissError("invalid_input", "Weights file does not cover every support point.")
    pi = merged["weight"].to_numpy(dtype=float)
    pi = pi / pi.sum()
    selected = items or list(mats)
    rows: list[dict[str, Any]] = []
    for item in selected:
        if item not in mats:
            raise UmrissError("item_not_found", f"Support bank does not contain item: {item}.")
        pred = mats[item].T @ pi
        labels = item_option_labels(metadata, item)
        codes = item_option_codes(metadata, item)
        for idx, value in enumerate(pred):
            rows.append({"item": item, "option_index": idx, "option_code": codes[idx], "option_label": labels[idx], "prediction": float(value)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return {"path": str(out_path), "items": len(selected), "rows": len(rows)}


def write_report(tag: str, derived_dir: Path, out_path: Path) -> dict[str, Any]:
    summary_path = derived_dir / f"{tag}_generated_support_summary.csv"
    diag_path = derived_dir / f"{tag}_generated_support_diagnostics.csv"
    points_path = derived_dir / f"{tag}_generated_support_points.csv"
    if not summary_path.exists():
        raise UmrissError("not_found", f"Summary file does not exist: {summary_path}.")
    summary = pd.read_csv(summary_path).sort_values("mean_rmse")
    lines = [f"# umriss report: {tag}", ""]
    lines.append("## Method Summary")
    lines.append("")
    lines.append("| method | mean_rmse | median_rmse | max_rmse | items |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.method} | {row.mean_rmse:.6f} | {row.median_rmse:.6f} | {row.max_rmse:.6f} | {int(row.items)} |")
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        lines.extend(["", "## Calibration Diagnostics", ""])
        lines.append(f"- holdouts: {len(diag)}")
        if "effective_support" in diag.columns:
            lines.append(f"- mean effective support: {diag['effective_support'].mean():.2f}")
        if "selected_rho" in diag.columns:
            lines.append(f"- selected rho values: {json.dumps(sorted(diag['selected_rho'].dropna().unique().tolist()))}")
    if points_path.exists():
        points = pd.read_csv(points_path)
        lines.extend(["", "## Support Points", ""])
        lines.append(f"- rows: {len(points)}")
        if "valid" in points.columns:
            lines.append(f"- valid rows: {int(points['valid'].astype(bool).sum())}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return {"path": str(out_path), "summary_path": str(summary_path), "diagnostics_path": str(diag_path) if diag_path.exists() else None}


GUIDES = {
    "workflow": {
        "summary": "End-to-end support-bank lifecycle.",
        "steps": [
            "Inspect or create battery metadata: `umriss battery inspect <metadata.json>`.",
            "Compile prompt rows: `umriss support build --metadata <metadata.json> --design <design.json> --tag <tag> --out <prompt_dir>`.",
            "Create EP jobs: `umriss support export --prompts <prompt_dir>/<tag>_prompts.jsonl --path <prompt_dir>/<tag>.jobs.ep`.",
            "Run the `.jobs.ep` outside umriss with EP/EDSL. Umriss does not run model jobs.",
            "Register results: `umriss support register-results --results <tag>.results.ep --prompts <tag>_prompts.jsonl --tag <tag> --out <raw_dir>`.",
            "Parse or evaluate: `umriss support parse ...` or `umriss validate marginals ...`.",
            "Compare/report: `umriss compare ...` and `umriss report ...`.",
        ],
    },
    "designs": {
        "summary": "How versioned declarative designs make support assumptions auditable.",
        "steps": [
            "Create an editable design with `umriss design create --preset pattern-coverage ...`.",
            "Validate feasibility with `umriss design validate` before generating prompts.",
            "Use `option_coverage`, `pattern_anchors`, and `profiles` components.",
            "Declare coherence, intensity allocation, probability semantics, and partial coverage explicitly.",
            "Review the resolved YAML, support plan, coverage table, and prompt HTML before model execution.",
        ],
    },
    "ep-boundary": {
        "summary": "What umriss does and does not run.",
        "steps": [
            "`umriss support export` creates `.jobs.ep` files and a run contract.",
            "The user or an external EP/EDSL runner executes `.jobs.ep` and writes `.results.ep`.",
            "`umriss support register-results` imports `.results.ep` into raw CSV artifacts.",
            "This boundary keeps prompt construction, model execution, and result parsing auditable.",
        ],
    },
    "paper-rewrite": {
        "summary": "Using umriss to shrink paper-specific scripts.",
        "steps": [
            "Use design JSON files plus `umriss support build` instead of bespoke prompt builders.",
            "Use `umriss validate marginals` for generated-support scoring.",
            "Use `umriss compare --recipe ...` for repeated comparison-table assembly.",
            "Keep raw model outputs and derived CSVs as explicit Makefile artifacts.",
        ],
    },
    "diagnostics": {
        "summary": "What to inspect after parsing and fitting.",
        "steps": [
            "Inspect prompts/raw/banks with `umriss support inspect`.",
            "Inspect parse health with `umriss support inspect --diagnostics <tag>_parse_diagnostics.csv`.",
            "Watch `effective_support`, `held_in_residual`, and selected rho in generated-support diagnostics.",
            "Use `umriss report --tag <tag> --derived <dir> --out <tag>.md` for a compact Markdown artifact.",
        ],
    },
}


def guide(topic: str) -> dict[str, Any]:
    if topic not in GUIDES:
        raise UmrissError("invalid_input", f"Unknown guide topic: {topic}.", context={"topics": sorted(GUIDES)})
    return {"topic": topic, "topics": sorted(GUIDES), **GUIDES[topic]}


def next_for_artifacts(
    tag: str,
    *,
    metadata: Path | None = None,
    design: Path | None = None,
    prompt_dir: Path = Path("data/computed_objects/support_prompts"),
    raw_dir: Path = Path("data/computed_objects/support_raw_responses"),
    derived_dir: Path = Path("data/derived"),
) -> dict[str, Any]:
    prompts = prompt_dir / f"{tag}_prompts.jsonl"
    jobs = prompt_dir / f"{tag}.jobs.ep"
    results = prompt_dir / f"{tag}.results.ep"
    raw = raw_dir / f"{tag}_raw.csv"
    probabilities = derived_dir / f"{tag}_probabilities.csv"
    summary = derived_dir / f"{tag}_generated_support_summary.csv"
    report = derived_dir / f"{tag}_report.md"

    artifacts = {
        "metadata": str(metadata) if metadata else None,
        "design": str(design) if design else None,
        "prompts": str(prompts),
        "jobs": str(jobs),
        "results": str(results),
        "raw": str(raw),
        "probabilities": str(probabilities),
        "summary": str(summary),
    }
    exists = {key: (Path(value).exists() if value else False) for key, value in artifacts.items()}

    if metadata and not metadata.exists():
        return {"stage": "metadata", "recommendation": f"Create or fix metadata file: {metadata}", "artifacts": artifacts, "exists": exists}
    if summary.exists():
        command = f"umriss report --tag {tag} --derived {derived_dir} --out {report}"
        return {"stage": "report-or-compare", "recommendation": command, "artifacts": artifacts, "exists": exists}
    if raw.exists():
        command = f"umriss validate marginals --raw {raw} --metadata {metadata or '<metadata.json>'} --tag {tag} --out {derived_dir}"
        return {"stage": "evaluate", "recommendation": command, "artifacts": artifacts, "exists": exists}
    if probabilities.exists():
        command = f"umriss validate marginals --support {probabilities} --metadata {metadata or '<metadata.json>'} --tag {tag} --out {derived_dir}"
        return {"stage": "evaluate", "recommendation": command, "artifacts": artifacts, "exists": exists}
    if not prompts.exists():
        if design:
            command = f"umriss support build --metadata {metadata or '<metadata.json>'} --design {design} --tag {tag} --out {prompt_dir}"
        else:
            command = f"umriss support build --metadata {metadata or '<metadata.json>'} --preset pattern-coverage --tag {tag} --out {prompt_dir}"
        return {"stage": "build-prompts", "recommendation": command, "artifacts": artifacts, "exists": exists}
    if not jobs.exists():
        command = f"umriss support export --prompts {prompts} --path {jobs}"
        return {"stage": "export-jobs", "recommendation": command, "artifacts": artifacts, "exists": exists}
    if not results.exists() and not raw.exists():
        command = f"ep run --jobs {jobs} --output {results}"
        return {"stage": "run-externally", "recommendation": command, "note": "Run this outside umriss; umriss only creates and registers EP files.", "artifacts": artifacts, "exists": exists}
    if results.exists() and not raw.exists():
        command = f"umriss support register-results --results {results} --prompts {prompts} --tag {tag} --out {raw_dir}"
        return {"stage": "register-results", "recommendation": command, "artifacts": artifacts, "exists": exists}
    return {"stage": "await-results", "recommendation": f"Place results at {results} or raw CSV at {raw}.", "artifacts": artifacts, "exists": exists}
