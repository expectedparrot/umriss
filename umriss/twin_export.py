from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import UmrissError
from .jsonlio import write_json


def export_edsl_agents(
    point_paths: list[Path],
    weights_path: Path,
    output_path: Path,
    *,
    holdout: str | None = None,
    minimum_weight: float = 0.0,
) -> dict[str, Any]:
    if not point_paths:
        raise UmrissError("invalid_input", "At least one --points file is required.")
    if minimum_weight < 0:
        raise UmrissError("invalid_input", "--minimum-weight cannot be negative.")
    points = pd.concat([pd.read_csv(path) for path in point_paths], ignore_index=True)
    required_points = {"job_id", "profile_summary"}
    if not required_points.issubset(points.columns):
        raise UmrissError(
            "invalid_input",
            f"Point files must contain: {', '.join(sorted(required_points))}.",
        )
    points = points.drop_duplicates("job_id", keep="last")
    points["profile_summary"] = points["profile_summary"].fillna("").astype(str).str.strip()
    weights = pd.read_csv(weights_path)
    required_weights = {"support_id", "weight"}
    if not required_weights.issubset(weights.columns):
        raise UmrissError("invalid_input", "Weights file must contain support_id and weight.")
    if "job_id" not in weights.columns:
        raise UmrissError(
            "invalid_input",
            "Weights must contain job_id so personas remain identifiable after support banks are merged.",
        )
    available_holdouts: list[str] = []
    if "holdout" in weights.columns:
        available_holdouts = [str(value) for value in weights["holdout"].dropna().unique()]
        if holdout is None and len(available_holdouts) > 1:
            raise UmrissError(
                "invalid_input",
                "This file contains multiple leave-one-out weight vectors; select one with --holdout.",
                context={"available_holdouts": available_holdouts},
            )
        selected_holdout = holdout or (available_holdouts[0] if available_holdouts else None)
        if selected_holdout is not None:
            weights = weights[weights["holdout"].astype(str).eq(selected_holdout)].copy()
            if weights.empty:
                raise UmrissError(
                    "invalid_input",
                    f"No weights found for holdout: {selected_holdout}.",
                    context={"available_holdouts": available_holdouts},
                )
    elif holdout is not None:
        raise UmrissError("invalid_input", "--holdout was supplied, but the weights file has no holdout column.")
    else:
        selected_holdout = None
    if weights["support_id"].duplicated().any():
        raise UmrissError("invalid_input", "Selected weights contain duplicate support IDs.")
    merged = weights.merge(points[["job_id", "profile_summary"]], on="job_id", how="left", validate="one_to_one")
    missing = merged["profile_summary"].isna() | merged["profile_summary"].eq("")
    if missing.any():
        raise UmrissError(
            "invalid_input",
            f"{int(missing.sum())} weighted personas have no profile summary.",
            hint="Pass every component *_points.csv file with repeated --points.",
            context={"support_ids": merged.loc[missing, "support_id"].astype(str).head(20).tolist()},
        )
    merged["weight"] = merged["weight"].astype(float)
    merged = merged[merged["weight"].ge(minimum_weight)].copy()
    if merged.empty or merged["weight"].sum() <= 0:
        raise UmrissError("invalid_input", "No positive-mass personas remain after filtering.")
    original_mass = float(merged["weight"].sum())
    merged["weight"] = merged["weight"] / original_mass
    merged = merged.sort_values("weight", ascending=False).reset_index(drop=True)

    try:
        with redirect_stdout(StringIO()):
            from edsl import Agent, AgentList
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to export an AgentList.") from exc
    agents = AgentList(
        [
            Agent(
                name=f"umriss_{row.support_id}",
                traits={
                    "_weight": float(row.weight),
                    "_umriss_support_id": str(row.support_id),
                    "_umriss_job_id": str(row.job_id),
                },
                instruction=row.profile_summary,
            )
            for row in merged.itertuples()
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(agents, "git") or not hasattr(AgentList, "git"):
        raise UmrissError(
            "edsl_outdated",
            "This EDSL installation does not support git-backed AgentList packages.",
        )
    with redirect_stdout(StringIO()):
        saved = agents.git.save(str(output_path), message="Export weighted umriss digital twins")
        agents_path = Path(saved["path"])
        loaded = AgentList.git.load(str(agents_path))
    if len(loaded) != len(agents):
        raise UmrissError("invalid_output", "Exported AgentList failed round-trip verification.")

    sidecar = merged[["support_id", "job_id", "weight", "profile_summary"]].copy()
    sidecar.insert(0, "agent_name", sidecar["support_id"].map(lambda value: f"umriss_{value}"))
    sidecar_path = output_path.with_name(f"{output_path.stem}_weights.csv")
    sidecar.to_csv(sidecar_path, index=False)
    manifest = {
        "kind": "umriss_edsl_agent_list",
        "agents_path": str(agents_path),
        "weights_path": str(sidecar_path),
        "source_weights": str(weights_path),
        "source_points": [str(path) for path in point_paths],
        "holdout": selected_holdout,
        "agents": len(agents),
        "minimum_weight": minimum_weight,
        "retained_mass_before_renormalization": original_mass,
        "instruction_contract": "Each Agent instruction is exactly its profile_summary; probability-elicitation prompts are excluded.",
        "weight_contract": (
            "Each Agent stores its normalized coefficient as the hidden `_weight` trait. "
            "EDSL excludes underscore-prefixed traits from prompts but does not automatically perform weighted sampling."
        ),
    }
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}
