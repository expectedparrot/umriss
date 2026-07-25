from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import UmrissError
from .jsonlio import write_json, write_jsonl
from .metadata import item_option_labels
from .parsing import extract_json, normalized_vec, read_raw_rows

SCHEMA_VERSION = 1


def _battery_description(metadata: dict[str, Any]) -> str:
    lines = [
        f"Survey context: {metadata.get('context', '')}",
        f"Battery topic: {metadata.get('topic', '')}",
        "",
        "Observed items:",
    ]
    for item, spec in metadata["items"].items():
        lines.append(f"- {item}: {spec['item_text']}")
    return "\n".join(lines)


def build_proposal_prompt(metadata: dict[str, Any], n_items: int, tag: str, out_dir: Path) -> dict[str, Any]:
    if n_items < 1:
        raise UmrissError("invalid_input", "--n-items must be positive.")
    labels = item_option_labels(metadata, next(iter(metadata["items"])))
    prompt = f"""Propose {n_items} new survey items that measure nearby aspects of the same topic.

{_battery_description(metadata)}

All proposed items must:
- be substantively distinct from the observed items and from one another;
- use exactly this response scale, in this order: {json.dumps(labels)};
- be understandable without referring to this task or to another item;
- measure the population, not describe a synthetic persona.

Return only valid JSON:
{{
  "items": [
    {{"item_text": "complete survey item text", "question_stem": "optional shared or item-specific stem"}}
  ]
}}"""
    path = out_dir / f"{tag}_proposal_prompts.jsonl"
    write_jsonl(path, [{"job_id": f"{tag}_proposal", "stage": "proposal", "prompt": prompt}])
    return {"prompts_path": str(path), "jobs": 1, "requested_items": n_items}


def parse_proposal(
    raw_path: Path, metadata: dict[str, Any], tag: str, out_dir: Path, expected_items: int | None = None
) -> dict[str, Any]:
    rows = read_raw_rows(raw_path)
    if len(rows) != 1:
        raise UmrissError("invalid_input", f"Expected exactly one proposal result; found {len(rows)}.")
    raw = rows[0].get("answer.resp") or rows[0].get("response") or ""
    parsed = extract_json(str(raw))
    candidates = parsed.get("items") if parsed else None
    if not isinstance(candidates, list) or not candidates:
        raise UmrissError("invalid_input", "Proposal result must contain a non-empty `items` array.")
    if expected_items is not None and len(candidates) != expected_items:
        raise UmrissError("invalid_input", f"Expected {expected_items} proposed items; found {len(candidates)}.")
    observed = {str(spec["item_text"]).strip().casefold() for spec in metadata["items"].values()}
    seen: set[str] = set()
    items: dict[str, Any] = {}
    labels = item_option_labels(metadata, next(iter(metadata["items"])))
    for idx, candidate in enumerate(candidates, 1):
        text = candidate.get("item_text") if isinstance(candidate, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise UmrissError("invalid_input", f"Proposed item {idx} has no item_text.")
        key = text.strip().casefold()
        if key in observed or key in seen:
            raise UmrissError("invalid_input", f"Proposed item is duplicated: {text}")
        seen.add(key)
        item_id = f"aux{idx:03d}"
        items[item_id] = {
            "item_text": text.strip(),
            "question_stem": str(candidate.get("question_stem", "")).strip(),
            "option_labels": labels,
        }
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "kind": "umriss_auxiliary_items",
        "tag": tag,
        "source": {"method": "frontier_model_proposal", "raw_path": str(raw_path)},
        "items": items,
    }
    path = out_dir / f"{tag}_auxiliary_items.json"
    write_json(path, artifact)
    return {"auxiliary_items_path": str(path), "items": len(items)}


def _aux_list(auxiliary: dict[str, Any]) -> str:
    return "\n".join(
        f"- {item}: {spec['item_text']}\n  Options: {json.dumps(spec['option_labels'])}"
        for item, spec in auxiliary["items"].items()
    )


def _response_contract(auxiliary: dict[str, Any]) -> str:
    shape = {item: ["one number per option, in listed order"] for item in auxiliary["items"]}
    return (
        "Return only valid JSON with exactly this outer structure:\n"
        + json.dumps({"probabilities": shape}, indent=2)
        + "\nEvery vector must contain nonnegative probabilities that sum to 1."
    )


def build_elicitation_prompts(
    metadata: dict[str, Any],
    auxiliary: dict[str, Any],
    support_path: Path,
    tag: str,
    out_dir: Path,
    *,
    conditioned_on: list[str] | None = None,
) -> dict[str, Any]:
    if auxiliary.get("schema_version") != SCHEMA_VERSION or auxiliary.get("kind") != "umriss_auxiliary_items":
        raise UmrissError("invalid_input", "Unsupported auxiliary-item artifact.")
    support = pd.read_csv(support_path)
    needed = {"support_id", "item", "option_index", "probability"}
    if not needed.issubset(support.columns):
        raise UmrissError("invalid_input", f"Support CSV is missing columns: {sorted(needed - set(support.columns))}")
    conditioned = conditioned_on or []
    truth = metadata.get("truth", {})
    unknown = [item for item in conditioned if item not in truth]
    if unknown:
        raise UmrissError("invalid_input", f"Conditioning marginals are unavailable for: {', '.join(unknown)}")
    condition_text = (
        "\nKnown population marginals (the only observed moments you may condition on):\n"
        + "\n".join(f"- {item}: {json.dumps(truth[item])}" for item in conditioned)
        if conditioned
        else "\nDo not assume access to any observed population marginals."
    )
    population_prompt = f"""Estimate population response distributions for proposed survey items.

{_battery_description(metadata)}
{condition_text}

Proposed auxiliary items:
{_aux_list(auxiliary)}

These are simulated moment conditions, not observed survey estimates. Express uncertainty through the distributions.
{_response_contract(auxiliary)}"""
    rows = [
        {
            "job_id": f"{tag}_population",
            "stage": "population",
            "support_id": "",
            "conditioned_on": conditioned,
            "prompt": population_prompt,
        }
    ]
    original_items = list(metadata["items"])
    for support_id, group in support.groupby("support_id", sort=False):
        signature = {}
        for item in original_items:
            item_rows = group[group["item"] == item].sort_values("option_index")
            if item_rows.empty:
                raise UmrissError("invalid_input", f"Support {support_id} is missing observed item {item}.")
            signature[item] = item_rows["probability"].astype(float).round(8).tolist()
        persona_prompt = f"""Predict how one synthetic survey-response profile would answer proposed items.

The profile is defined by its subjective response probabilities on the observed battery:
{json.dumps(signature, indent=2)}

Observed battery:
{_battery_description(metadata)}

Proposed auxiliary items:
{_aux_list(auxiliary)}

Preserve the profile's substantive response tendencies without inventing demographics or treating it as a real respondent.
{_response_contract(auxiliary)}"""
        rows.append(
            {
                "job_id": f"{tag}_support_{support_id}",
                "stage": "support",
                "support_id": support_id,
                "conditioned_on": [],
                "prompt": persona_prompt,
            }
        )
    path = out_dir / f"{tag}_moment_prompts.jsonl"
    write_jsonl(path, rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "umriss_moment_elicitation",
        "tag": tag,
        "auxiliary_items": auxiliary,
        "support_path": str(support_path),
        "population_conditioned_on": conditioned,
        "leakage_rule": "A fold-specific population job may contain held-in real marginals only.",
        "prompts_path": str(path),
    }
    manifest_path = out_dir / f"{tag}_moment_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "prompts_path": str(path),
        "manifest_path": str(manifest_path),
        "jobs": len(rows),
        "support_jobs": len(rows) - 1,
        "population_jobs": 1,
    }


def parse_elicitation(
    raw_path: Path, prompts_path: Path, auxiliary: dict[str, Any], tag: str, out_dir: Path
) -> dict[str, Any]:
    prompt_rows = {row["job_id"]: row for row in _read_jsonl(prompts_path)}
    population_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for idx, row in enumerate(read_raw_rows(raw_path)):
        job_id = str(row.get("scenario.job_id") or row.get("job_id") or f"row_{idx}")
        prompt = prompt_rows.get(job_id)
        if prompt is None:
            raise UmrissError("invalid_input", f"Raw result has unknown job_id: {job_id}")
        parsed = extract_json(str(row.get("answer.resp") or row.get("response") or ""))
        probs = parsed.get("probabilities") if parsed else None
        valid = isinstance(probs, dict)
        vectors: dict[str, Any] = {}
        for item, spec in auxiliary["items"].items():
            vec, diag = normalized_vec(probs.get(item) if isinstance(probs, dict) else None, len(spec["option_labels"]))
            diagnostics.append({"job_id": job_id, "stage": prompt["stage"], "item": item, "valid": vec is not None, **diag})
            if vec is None:
                valid = False
            else:
                vectors[item] = vec
        if not valid:
            continue
        target = population_rows if prompt["stage"] == "population" else support_rows
        for item, vec in vectors.items():
            for option_index, probability in enumerate(vec):
                target.append(
                    {
                        "job_id": job_id,
                        "support_id": prompt.get("support_id", ""),
                        "item": item,
                        "option_index": option_index,
                        "option_label": auxiliary["items"][item]["option_labels"][option_index],
                        "probability": float(probability),
                        "conditioned_on": json.dumps(prompt.get("conditioned_on", [])),
                    }
                )
    expected_support = sum(row["stage"] == "support" for row in prompt_rows.values())
    parsed_support = len({row["support_id"] for row in support_rows})
    if not population_rows or parsed_support != expected_support:
        raise UmrissError(
            "invalid_input",
            f"Incomplete auxiliary responses: population={bool(population_rows)}, support={parsed_support}/{expected_support}.",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    population_path = out_dir / f"{tag}_auxiliary_targets.csv"
    support_path = out_dir / f"{tag}_auxiliary_support.csv"
    diagnostics_path = out_dir / f"{tag}_auxiliary_parse_diagnostics.csv"
    pd.DataFrame(population_rows).to_csv(population_path, index=False)
    pd.DataFrame(support_rows).to_csv(support_path, index=False)
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    return {
        "targets_path": str(population_path),
        "support_path": str(support_path),
        "diagnostics_path": str(diagnostics_path),
        "support_points": parsed_support,
        "auxiliary_items": len(auxiliary["items"]),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
