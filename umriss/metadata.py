from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import UmrissError
from .jsonlio import append_jsonl, read_json, read_jsonl, write_json
from .state import battery_dir


def item_option_labels(metadata: dict[str, Any], item: str) -> list[str]:
    labels = metadata["items"][item].get("option_labels", metadata.get("option_labels", []))
    return [str(x) for x in labels]


def item_option_codes(metadata: dict[str, Any], item: str) -> list[int | str]:
    codes = metadata["items"][item].get("option_codes", metadata.get("option_codes"))
    if codes is None:
        return list(range(1, len(item_option_labels(metadata, item)) + 1))
    return list(codes)


def battery_id_from_metadata(metadata: dict[str, Any]) -> str:
    return f"{metadata.get('wave', 'battery')}_{metadata.get('battery', 'items')}".lower()


def validate_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for key in ["wave", "battery", "topic", "context", "items"]:
        if key not in metadata:
            raise UmrissError("metadata_invalid", f"Metadata missing required field: {key}.")
    if not isinstance(metadata["items"], dict) or not metadata["items"]:
        raise UmrissError("metadata_invalid", "Metadata must contain at least one item.")
    for item, item_meta in metadata["items"].items():
        if "item_text" not in item_meta or "question_stem" not in item_meta:
            raise UmrissError("metadata_invalid", f"Item {item} missing item_text or question_stem.")
        labels = item_option_labels(metadata, item)
        if not labels:
            raise UmrissError("metadata_invalid", f"Item {item} has no option labels.")
        codes = item_option_codes(metadata, item)
        if len(codes) != len(labels):
            raise UmrissError("metadata_invalid", f"Item {item} option_codes length does not match option_labels.")
    counts = {len(item_option_labels(metadata, item)) for item in metadata["items"]}
    if len(counts) > 1:
        warnings.append({"code": "mixed_option_counts", "message": "Items have different option counts."})
    return warnings


def inspect_metadata(path: Path) -> dict[str, Any]:
    metadata = read_json(path)
    warnings = validate_metadata(metadata)
    items = list(metadata["items"])
    option_counts = {item: len(item_option_labels(metadata, item)) for item in items}
    moment_dimension = sum(count - 1 for count in option_counts.values())
    return {
        "battery_id": battery_id_from_metadata(metadata),
        "wave": metadata["wave"],
        "battery": metadata["battery"],
        "items": len(items),
        "option_counts": option_counts,
        "moment_dimension": moment_dimension,
        "has_truth": bool(metadata.get("truth")),
        "warnings": warnings,
    }


def create_battery(args: Any) -> dict[str, Any]:
    bdir = battery_dir(args.battery_id)
    bdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "wave": args.wave,
        "battery": args.battery,
        "survey_key": args.battery_id,
        "topic": args.topic,
        "context": args.context,
        "items": {},
    }
    write_json(bdir / "battery.json", metadata)
    (bdir / "questions.jsonl").touch(exist_ok=True)
    (bdir / "marginals.jsonl").touch(exist_ok=True)
    return {"battery_id": args.battery_id, "battery_path": str(bdir / "battery.json")}


def import_battery(metadata_path: Path, battery_id: str | None = None, title: str | None = None) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    validate_metadata(metadata)
    bid = battery_id or metadata.get("survey_key") or battery_id_from_metadata(metadata)
    bdir = battery_dir(bid)
    bdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(metadata_path, bdir / "battery.json")
    rows = {"batteries": [{"battery_id": bid, "title": title or bid, "path": str(bdir / "battery.json")}]}
    write_json(bdir.parent.parent / "batteries.json", rows)
    return {"battery_id": bid, "battery_path": str(bdir / "battery.json")}


def add_question(args: Any) -> dict[str, Any]:
    bdir = battery_dir(args.battery)
    if not (bdir / "battery.json").exists():
        raise UmrissError("not_found", f"Battery does not exist: {args.battery}.")
    options = list(args.option or [])
    option_codes = list(args.option_code or range(1, len(options) + 1))
    if len(options) != len(option_codes):
        raise UmrissError("invalid_input", "--option and --option-code counts must match.")
    row = {
        "item": args.item,
        "variable": args.variable,
        "question_stem": args.question_stem,
        "item_text": args.item_text,
        "option_codes": option_codes,
        "option_labels": options,
    }
    append_jsonl(bdir / "questions.jsonl", row)
    compile_battery(args.battery, bdir / "battery.json")
    return {"battery_id": args.battery, "item": args.item}


def add_marginal(args: Any) -> dict[str, Any]:
    bdir = battery_dir(args.battery)
    metadata = compile_battery(args.battery, bdir / "battery.json")
    if args.item not in metadata["items"]:
        raise UmrissError("item_not_found", f"Unknown item: {args.item}.")
    vals = np.array(args.proportion, dtype=float)
    expected = len(item_option_labels(metadata, args.item))
    if len(vals) != expected:
        raise UmrissError("probability_length_mismatch", f"Item {args.item} expected {expected} proportions.")
    if (vals < 0).any():
        raise UmrissError("invalid_input", "Marginal proportions must be nonnegative.")
    total = float(vals.sum())
    if not np.isclose(total, 1.0, atol=1e-5):
        if not args.normalize:
            raise UmrissError("invalid_input", f"Marginal proportions sum to {total:.6f}.", hint="Pass --normalize to normalize.")
        vals = vals / total
    row = {"item": args.item, "proportions": vals.tolist(), "source": args.source}
    append_jsonl(bdir / "marginals.jsonl", row)
    compile_battery(args.battery, bdir / "battery.json")
    return {"battery_id": args.battery, "item": args.item, "proportions": vals.tolist()}


def compile_battery(battery_id: str, path: Path | None = None) -> dict[str, Any]:
    bdir = battery_dir(battery_id)
    metadata = read_json(bdir / "battery.json")
    items = {}
    if (bdir / "questions.jsonl").exists():
        for row in read_jsonl(bdir / "questions.jsonl"):
            item = row["item"]
            items[item] = {
                "variable": row.get("variable", item),
                "item_text": row["item_text"],
                "question_stem": row["question_stem"],
                "option_codes": row["option_codes"],
                "option_labels": row["option_labels"],
            }
    metadata["items"] = items
    truth = {}
    if (bdir / "marginals.jsonl").exists():
        for row in read_jsonl(bdir / "marginals.jsonl"):
            truth[row["item"]] = row["proportions"]
    if truth:
        metadata["truth"] = truth
    validate_metadata(metadata) if metadata["items"] else None
    output = path or (bdir / "battery.json")
    write_json(output, metadata)
    return metadata


def marginals_from_metadata(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    truth = metadata.get("truth")
    if not isinstance(truth, dict):
        raise UmrissError("marginal_missing", "Metadata does not contain truth marginals.")
    out = {}
    for item in metadata["items"]:
        if item not in truth:
            raise UmrissError("marginal_missing", f"Missing truth marginal for item {item}.")
        arr = np.array(truth[item], dtype=float)
        out[item] = arr / arr.sum()
    return out


def weighted_truth_from_respondents(metadata: dict[str, Any], respondents_path: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(respondents_path)
    weight_col = metadata.get("weight", "weight")
    weights = df[weight_col].to_numpy(dtype=float)
    truth = {}
    for item in metadata["items"]:
        col = f"item_{item}"
        codes = item_option_codes(metadata, item)
        vals = df[col].to_numpy()
        vec = np.array([weights[vals == code].sum() for code in codes], dtype=float)
        truth[item] = vec / vec.sum()
    return truth


def write_marginals_long(metadata: dict[str, Any], truth: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["battery", "item", "option_index", "option_code", "option_label", "proportion"])
        writer.writeheader()
        battery = f"{metadata['wave']}_{metadata['battery']}"
        for item, vec in truth.items():
            for idx, value in enumerate(vec):
                writer.writerow(
                    {
                        "battery": battery,
                        "item": item,
                        "option_index": idx,
                        "option_code": item_option_codes(metadata, item)[idx],
                        "option_label": item_option_labels(metadata, item)[idx],
                        "proportion": float(value),
                    }
                )
