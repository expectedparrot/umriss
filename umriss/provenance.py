from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .errors import UmrissError
from .jsonlio import read_json


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def path_sha256(path: Path) -> str:
    if not path.exists():
        raise UmrissError("not_found", f"Provenance input does not exist: {path}.")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_provenance(
    command: str,
    *,
    inputs: dict[str, Path | None],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    input_records = {
        name: {
            "path": str(path),
            "sha256": path_sha256(path),
        }
        for name, path in inputs.items()
        if path is not None
    }
    fingerprint = payload_sha256(
        {
            "command": command,
            "inputs": {name: record["sha256"] for name, record in input_records.items()},
            "parameters": parameters,
            "umriss_version": __version__,
        }
    )
    return {
        "schema_version": 1,
        "command": command,
        "created_at": utc_now(),
        "umriss_version": __version__,
        "inputs": input_records,
        "parameters": parameters,
        "fingerprint": fingerprint,
    }


def guard_manifest(
    manifest_path: Path,
    provenance: dict[str, Any],
    *,
    outputs: list[Path],
    force: bool,
) -> dict[str, Any] | None:
    if manifest_path.exists():
        existing = read_json(manifest_path)
        existing_fingerprint = existing.get("provenance", {}).get("fingerprint")
        if existing_fingerprint == provenance["fingerprint"] and all(path.exists() for path in outputs):
            return existing
        if not force:
            raise UmrissError(
                "output_conflict",
                f"Existing artifacts at `{manifest_path}` were created from different inputs or are incomplete.",
                context={
                    "manifest": str(manifest_path),
                    "existing_fingerprint": existing_fingerprint,
                    "requested_fingerprint": provenance["fingerprint"],
                },
                hint="Choose a new tag/output path or rerun with --force after reviewing the existing audit trail.",
            )
    elif any(path.exists() for path in outputs) and not force:
        raise UmrissError(
            "output_conflict",
            "Output artifacts already exist without a compatible provenance manifest.",
            context={"outputs": [str(path) for path in outputs if path.exists()]},
            hint="Choose a new tag/output path or rerun with --force after reviewing the existing artifacts.",
        )
    return None
