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
from .provenance import build_provenance, guard_manifest, path_sha256


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
            "persona": "",
            "valid": False,
        }
        if not parsed or not isinstance(parsed.get("probabilities"), dict):
            diagnostics.append({"job_id": job_id, "support_id": support_id, "status": "invalid", "code": "probability_json_invalid", "message": "Could not parse probabilities object.", "item": "", "raw_sum": "", "min_probability": "", "max_probability": ""})
            points.append(point)
            continue
        persona = str(parsed.get("persona", "")).strip()
        if not (persona.startswith("Your ") or persona.startswith("You ")):
            diagnostics.append(
                {
                    "job_id": job_id,
                    "support_id": support_id,
                    "status": "invalid",
                    "code": "persona_invalid",
                    "message": "Persona must be written in the second person and begin with `Your ` or `You `.",
                    "item": "",
                    "raw_sum": "",
                    "min_probability": "",
                    "max_probability": "",
                }
            )
            points.append(point)
            continue
        point["persona"] = persona
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


def _result_job_column(frame: pd.DataFrame) -> str:
    column = next(
        (name for name in ("scenario.job_id", "job_id", "scenario_job_id") if name in frame.columns),
        None,
    )
    if column is None:
        raise UmrissError("invalid_input", "Results do not contain a stable scenario job ID.")
    return column


def _result_answer_column(frame: pd.DataFrame) -> str:
    column = next((name for name in frame.columns if name == "answer.resp"), None)
    if column is None:
        column = next((name for name in frame.columns if name.startswith("answer.")), None)
    if column is None:
        raise UmrissError("invalid_input", "Results do not contain an answer column.")
    return column


def _expected_job_ids(prompts_path: Path) -> list[str]:
    if prompts_path.suffix != ".jsonl":
        frame = pd.read_csv(prompts_path)
        if "job_id" not in frame:
            raise UmrissError("invalid_input", "Prompt table does not contain `job_id`.")
        return frame["job_id"].astype(str).tolist()
    rows = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    if any("job_id" not in row for row in rows):
        raise UmrissError("invalid_input", "Prompt JSONL contains a row without `job_id`.")
    return [str(row["job_id"]) for row in rows]


def _valid_result_rows(frame: pd.DataFrame) -> pd.DataFrame:
    job_column = _result_job_column(frame)
    answer_column = _result_answer_column(frame)
    valid = frame[job_column].notna() & frame[answer_column].notna()
    valid &= frame[answer_column].astype(str).str.strip().ne("")
    selected = frame.loc[valid].copy()
    selected["_umriss_job_id"] = selected[job_column].astype(str)
    return selected


def register_results(
    results_path: Path,
    prompts_path: Path | None,
    tag: str,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{tag}_raw.csv"
    jobs_path = out_dir / f"{tag}_jobs.csv"
    registration_path = out_dir / f"{tag}_registration.json"
    provenance = build_provenance(
        "umriss support register-results",
        inputs={"results": results_path, "prompts": prompts_path},
        parameters={"tag": tag, "out": str(out_dir)},
    )
    existing = guard_manifest(
        registration_path,
        provenance,
        outputs=[raw_path, jobs_path],
        force=force,
    )
    if existing is not None:
        return {
            "raw_path": str(raw_path),
            "jobs_path": str(jobs_path),
            "registration_path": str(registration_path),
            "rows": int(existing["rows"]),
            "reused": True,
        }
    raw = load_results_ep_to_pandas(results_path)
    if prompts_path:
        expected = set(_expected_job_ids(prompts_path))
        observed = set(_valid_result_rows(raw)["_umriss_job_id"])
        missing = sorted(expected - observed)
        if missing:
            raise UmrissError(
                "incomplete_results",
                f"Results are missing valid answers for {len(missing)} of {len(expected)} prompt jobs.",
                context={
                    "expected_jobs": len(expected),
                    "valid_jobs": len(expected) - len(missing),
                    "missing_jobs": missing[:20],
                },
                next_steps=[
                    f"umriss support audit-results --results {results_path} --prompts {prompts_path} "
                    f"--tag {tag} --out {out_dir / 'retry_audit'}"
                ],
            )
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
    registration = {
        "schema_version": 1,
        "kind": "umriss_result_registration",
        "tag": tag,
        "rows": len(raw),
        "provenance": provenance,
        "outputs": {
            "raw": {"path": str(raw_path), "sha256": path_sha256(raw_path)},
            "jobs": {"path": str(jobs_path), "sha256": path_sha256(jobs_path)},
        },
    }
    registration_path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    return {
        "raw_path": str(raw_path),
        "jobs_path": str(jobs_path),
        "registration_path": str(registration_path),
        "rows": len(raw),
        "reused": False,
    }


def audit_result_attempts(
    results_paths: list[Path],
    prompts_path: Path,
    tag: str,
    out_dir: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not results_paths:
        raise UmrissError("invalid_input", "At least one --results package is required.")
    expected_ids = _expected_job_ids(prompts_path)
    if len(expected_ids) != len(set(expected_ids)):
        raise UmrissError("invalid_input", "Prompt job IDs must be unique.")
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / f"{tag}_merged_raw.csv"
    coverage_path = out_dir / f"{tag}_retry_coverage.csv"
    missing_path = out_dir / f"{tag}_missing_job_ids.csv"
    manifest_path = out_dir / f"{tag}_retry_manifest.json"
    provenance = build_provenance(
        "umriss support audit-results",
        inputs={
            "prompts": prompts_path,
            **{f"results_{index}": path for index, path in enumerate(results_paths, start=1)},
        },
        parameters={"tag": tag, "results_order": [str(path) for path in results_paths]},
    )
    outputs = [merged_path, coverage_path, missing_path]
    existing = guard_manifest(manifest_path, provenance, outputs=outputs, force=force)
    if existing is not None:
        return {**existing["data"], "reused": True}

    selected: dict[str, pd.Series] = {}
    coverage_rows: list[dict[str, Any]] = []
    for run_index, results_path in enumerate(results_paths, start=1):
        frame = load_results_ep_to_pandas(results_path)
        valid = _valid_result_rows(frame)
        observed = set(valid["_umriss_job_id"])
        coverage_rows.append(
            {
                "run": run_index,
                "results_path": str(results_path),
                "valid_jobs": len(set(expected_ids) & observed),
                "expected_jobs": len(expected_ids),
                "missing_jobs": len(set(expected_ids) - observed),
            }
        )
        for _, record in valid.iterrows():
            job_id = str(record["_umriss_job_id"])
            if job_id in expected_ids and job_id not in selected:
                retained = record.copy()
                retained["_umriss_source_run"] = run_index
                retained["_umriss_source_results"] = str(results_path)
                selected[job_id] = retained

    missing = [job_id for job_id in expected_ids if job_id not in selected]
    merged = pd.DataFrame([selected[job_id] for job_id in expected_ids if job_id in selected])
    merged.to_csv(merged_path, index=False)
    coverage_rows.append(
        {
            "run": "merged",
            "results_path": "first valid response in supplied run order",
            "valid_jobs": len(selected),
            "expected_jobs": len(expected_ids),
            "missing_jobs": len(missing),
        }
    )
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    pd.DataFrame({"job_id": missing}).to_csv(missing_path, index=False)
    data = {
        "merged_raw_path": str(merged_path),
        "coverage_path": str(coverage_path),
        "missing_job_ids_path": str(missing_path),
        "attempts": len(results_paths),
        "expected_jobs": len(expected_ids),
        "valid_jobs": len(selected),
        "missing_jobs": len(missing),
        "complete": not missing,
        "manifest_path": str(manifest_path),
        "reused": False,
    }
    manifest = {
        "schema_version": 1,
        "kind": "umriss_retry_audit",
        "provenance": provenance,
        "outputs": {
            path.name: {"path": str(path), "sha256": path_sha256(path)}
            for path in outputs
        },
        "data": data,
    }
    registration_path = out_dir / f"{tag}_retry_manifest.json"
    registration_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return data
