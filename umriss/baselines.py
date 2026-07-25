from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import UmrissError
from .jsonlio import read_jsonl, write_jsonl
from .metadata import item_option_labels, marginals_from_metadata, weighted_truth_from_respondents
from .parsing import extract_json, normalized_vec, read_raw_rows


def _item_text(metadata: dict[str, Any], item: str) -> str:
    spec = metadata["items"][item]
    labels = item_option_labels(metadata, item)
    return (
        f"Question stem: {spec.get('question_stem', '')}\n"
        f"Survey item: {spec['item_text']}\n"
        f"Response options, in order: {json.dumps(labels)}"
    )


def build_baseline_prompts(
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
    *,
    mode: str = "both",
    respondents_path: Path | None = None,
) -> dict[str, Any]:
    modes = ["one_shot", "conditioned_direct"] if mode == "both" else [mode]
    if any(value not in {"one_shot", "conditioned_direct"} for value in modes):
        raise UmrissError("invalid_input", f"Unsupported baseline mode: {mode}")
    truth = None
    if "conditioned_direct" in modes:
        if respondents_path:
            truth = weighted_truth_from_respondents(metadata, respondents_path)
        elif "truth" in metadata:
            truth = marginals_from_metadata(metadata)
        else:
            raise UmrissError(
                "invalid_input",
                "Conditioned-direct jobs require metadata truth or --respondents to construct held-in marginals.",
            )
    context = metadata.get("context", "")
    rows: list[dict[str, Any]] = []
    for holdout in metadata["items"]:
        for baseline_mode in modes:
            evidence = ""
            held_in: list[str] = []
            if baseline_mode == "conditioned_direct":
                held_in = [item for item in metadata["items"] if item != holdout]
                evidence = (
                    "\nKnown population marginals for the other items:\n"
                    + "\n".join(
                        f"- {item} ({metadata['items'][item]['item_text']}): "
                        f"{json.dumps([round(float(x), 8) for x in truth[item]])}"
                        for item in held_in
                    )
                    + "\nUse these as evidence, but do not assume that response distributions are identical across items.\n"
                )
            prompt = f"""Predict the population response distribution for one survey item.

Survey context: {context}
Battery topic: {metadata.get('topic', '')}

{_item_text(metadata, holdout)}
{evidence}
Return only valid JSON:
{{
  "reasoning_summary": "brief explanation",
  "probabilities": [one nonnegative probability per option, summing to 1]
}}"""
            rows.append(
                {
                    "job_id": f"{tag}_{baseline_mode}_{holdout}",
                    "stage": "baseline",
                    "mode": baseline_mode,
                    "holdout": holdout,
                    "held_in": held_in,
                    "prompt": prompt,
                }
            )
    path = out_dir / f"{tag}_baseline_prompts.jsonl"
    write_jsonl(path, rows)
    return {
        "prompts_path": str(path),
        "jobs": len(rows),
        "items": len(metadata["items"]),
        "modes": modes,
        "leakage_rule": "The held-out marginal is never included in its conditioned-direct prompt.",
    }


def parse_baseline_results(
    raw_path: Path,
    prompts_path: Path,
    metadata: dict[str, Any],
    tag: str,
    out_dir: Path,
) -> dict[str, Any]:
    prompts = {str(row["job_id"]): row for row in read_jsonl(prompts_path)}
    parsed_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, row in enumerate(read_raw_rows(raw_path)):
        job_id = str(row.get("scenario.job_id") or row.get("job_id") or f"row_{idx}")
        contract = prompts.get(job_id)
        if contract is None:
            raise UmrissError("invalid_input", f"Raw result has unknown baseline job_id: {job_id}")
        response = extract_json(str(row.get("answer.resp") or row.get("response") or ""))
        holdout = contract["holdout"]
        vec, diag = normalized_vec(
            response.get("probabilities") if response else None,
            len(item_option_labels(metadata, holdout)),
        )
        valid = vec is not None
        diagnostics.append(
            {
                "job_id": job_id,
                "mode": contract["mode"],
                "holdout": holdout,
                "valid": valid,
                **diag,
            }
        )
        if valid:
            seen.add(job_id)
            parsed_rows.append(
                {
                    "tag": tag,
                    "item": holdout,
                    "holdout": holdout,
                    "mode": contract["mode"],
                    "prediction": json.dumps([round(float(x), 8) for x in vec]),
                    "reasoning_summary": str(response.get("reasoning_summary", "")),
                    "held_in": json.dumps(contract.get("held_in", [])),
                    "job_id": job_id,
                }
            )
    missing = sorted(set(prompts) - seen)
    if missing:
        raise UmrissError(
            "invalid_input",
            f"Baseline results are incomplete or invalid: {len(missing)} of {len(prompts)} jobs failed.",
            context={"missing_job_ids": missing[:20]},
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    frame = pd.DataFrame(parsed_rows)
    for mode, group in frame.groupby("mode", sort=False):
        suffix = "one_shot" if mode == "one_shot" else "conditioned_direct"
        path = out_dir / f"{tag}_{suffix}.csv"
        group.to_csv(path, index=False)
        outputs[f"{suffix}_path"] = str(path)
    diagnostics_path = out_dir / f"{tag}_baseline_parse_diagnostics.csv"
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    return {**outputs, "diagnostics_path": str(diagnostics_path), "predictions": len(parsed_rows)}
