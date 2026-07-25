from __future__ import annotations

import ast
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import UmrissError
from .metadata import item_option_codes, item_option_labels


def extract_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            try:
                data = ast.literal_eval(candidate)
            except Exception:
                continue
        if isinstance(data, dict):
            return data
    return None


def normalized_vec(value: Any, k: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    diag: dict[str, Any] = {"raw_sum": None, "min_probability": None, "max_probability": None}
    if not isinstance(value, list) or len(value) != k:
        return None, diag
    try:
        arr = np.array(value, dtype=float)
    except Exception:
        return None, diag
    diag["min_probability"] = float(arr.min()) if len(arr) else None
    diag["max_probability"] = float(arr.max()) if len(arr) else None
    total = float(arr.sum())
    diag["raw_sum"] = total
    if (arr < 0).any() or not np.isclose(total, 1.0, atol=1e-5):
        return None, diag
    return arr, diag


def read_raw_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_support(raw_path: Path, metadata: dict[str, Any], tag: str, out_dir: Path) -> dict[str, str | int]:
    rows = read_raw_rows(raw_path)
    points: list[dict[str, Any]] = []
    probs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    items = list(metadata["items"])

    for idx, row in enumerate(rows):
        job_id = row.get("scenario.job_id") or row.get("job_id") or f"row_{idx}"
        raw_text = row.get("answer.resp") or row.get("response") or ""
        parsed = extract_json(str(raw_text))
        support_id = row.get("support_id") or row.get("scenario.support_id") or idx + 1
        point = {
            "support_id": support_id,
            "job_id": job_id,
            "variant": row.get("variant", ""),
            "profile_summary": "",
            "valid": False,
        }
        if not parsed or not isinstance(parsed.get("probabilities"), dict):
            diagnostics.append({"job_id": job_id, "support_id": support_id, "status": "invalid", "code": "probability_json_invalid", "message": "Could not parse probabilities object.", "item": "", "raw_sum": "", "min_probability": "", "max_probability": ""})
            points.append(point)
            continue
        point["profile_summary"] = parsed.get("profile_summary", "")
        item_vecs: dict[str, np.ndarray] = {}
        valid = True
        for item in items:
            k = len(item_option_labels(metadata, item))
            vec, diag = normalized_vec(parsed["probabilities"].get(item), k)
            if vec is None:
                valid = False
                diagnostics.append(
                    {
                        "job_id": job_id,
                        "support_id": support_id,
                        "status": "invalid",
                        "code": "probability_vector_invalid",
                        "message": f"Item {item} must contain exactly {k} nonnegative probabilities summing to 1.",
                        "item": item,
                        **diag,
                    }
                )
                continue
            item_vecs[item] = vec
            diagnostics.append({"job_id": job_id, "support_id": support_id, "status": "ok", "code": "", "message": "", "item": item, **diag})
        point["valid"] = valid
        points.append(point)
        if not valid:
            continue
        for item, vec in item_vecs.items():
            labels = item_option_labels(metadata, item)
            codes = item_option_codes(metadata, item)
            for option_index, probability in enumerate(vec):
                probs.append({"support_id": support_id, "job_id": job_id, "item": item, "option_index": option_index, "option_code": codes[option_index], "option_label": labels[option_index], "probability": float(probability)})

    if not any(point["valid"] for point in points):
        raise UmrissError("support_empty", f"No valid support points parsed from {raw_path}.")
    out_dir.mkdir(parents=True, exist_ok=True)
    points_path = out_dir / f"{tag}_points.csv"
    probs_path = out_dir / f"{tag}_probabilities.csv"
    diag_path = out_dir / f"{tag}_parse_diagnostics.csv"
    pd.DataFrame(points).to_csv(points_path, index=False)
    pd.DataFrame(probs).to_csv(probs_path, index=False)
    pd.DataFrame(diagnostics).to_csv(diag_path, index=False)
    return {"points_path": str(points_path), "probabilities_path": str(probs_path), "diagnostics_path": str(diag_path), "valid_support_points": int(sum(bool(point["valid"]) for point in points))}


def load_results_ep_to_pandas(results_path: Path) -> pd.DataFrame:
    try:
        from edsl import Results
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to register .results.ep files.") from exc
    rp = Path(results_path)
    try:
        results = None
        if hasattr(Results, "git"):
            try:
                results = Results.git.load(str(rp))
            except Exception:
                results = None
        # `ep run --save/--output X.results.ep` may write a plain-JSON file at the
        # literal path. Try that first so registration does not depend on EDSL's
        # extension-appending loader (which looks for X.results.ep.json[.gz] and
        # otherwise fails with a confusing "No such file" on the literal .ep name).
        if results is None and rp.is_file():
            try:
                results = Results.from_dict(json.loads(rp.read_text()))
            except Exception:
                results = None
        if results is None:
            results = Results.load(str(rp))
        return results.to_pandas(remove_prefix=False)
    except Exception as exc:
        raise UmrissError("invalid_input", f"Could not load results EP file: {results_path}.", context={"error": str(exc)}) from exc


def register_results(results_path: Path, prompts_path: Path | None, tag: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{tag}_raw.csv"
    jobs_path = out_dir / f"{tag}_jobs.csv"
    raw = load_results_ep_to_pandas(results_path)
    raw.to_csv(raw_path, index=False)
    if prompts_path:
        if prompts_path.suffix == ".jsonl":
            rows = []
            with prompts_path.open() as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            pd.DataFrame(rows).to_csv(jobs_path, index=False)
        else:
            shutil.copyfile(prompts_path, jobs_path)
    else:
        pd.DataFrame().to_csv(jobs_path, index=False)
    provenance = {"results": str(results_path), "prompts": str(prompts_path) if prompts_path else None, "raw": str(raw_path), "jobs": str(jobs_path)}
    (out_dir / f"{tag}_registration.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {"raw_path": str(raw_path), "jobs_path": str(jobs_path), "registration_path": str(out_dir / f"{tag}_registration.json")}
