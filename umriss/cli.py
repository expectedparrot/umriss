from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import (
    battery_designed_recipe,
    build_report_data,
    compare_runs,
    guide,
    inspect_artifact,
    next_for_artifacts,
    pattern_coverage_recipe,
    predict_from_weights,
    write_report,
)
from .balancing import build_uniform_augmentation, merge_support_banks, write_uniformity
from .baselines import build_baseline_prompts, parse_baseline_results
from .calibration import fit_weights, load_support_matrix, write_fit_outputs
from .ep_commands import export_support_jobs
from .errors import UmrissError
from .evaluation import run_marginal_validation
from .jsonlio import read_json, write_json
from .metadata import (
    add_marginal,
    add_question,
    compile_battery,
    create_battery,
    import_battery,
    inspect_metadata,
    marginals_from_metadata,
    weighted_truth_from_respondents,
    write_marginals_long,
)
from .parsing import audit_result_attempts, parse_support, register_results
from .plotting import plot_validation
from .provenance import build_provenance, guard_manifest, path_sha256
from .state import (
    ROOT,
    active_project_id,
    create_project,
    design_path,
    get_defaults,
    init_workspace,
    list_battery_ids,
    list_design_ids,
    list_projects,
    list_run_tags,
    project_dir,
    run_dir,
    set_default,
    use_project,
)
from .state import battery_dir as state_battery_dir
from .support_designs import (
    compile_support_plan,
    load_design_config,
    preset_design,
    resolve_design,
    validate_design,
    write_support_outputs,
)
from .twin_export import export_edsl_agents
from .twin_survey import (
    analyze_probabilistic_survey,
    analyze_resolution_experiment,
    analyze_token_probabilities,
    build_resolution_experiment,
    build_survey_jobs,
    compare_edsl_survey,
    embed_response_probabilities,
    export_edsl_survey,
    plot_survey_comparison,
)


ENVELOPE_SCHEMA_VERSION = "1.0"


def envelope(
    command: str,
    status: str,
    data: dict[str, Any] | None = None,
    *,
    warnings: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
    next_steps: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "command": command,
        "status": status,
        "argv": ["umriss", *sys.argv[1:]],
        "data": data or {},
        "warnings": warnings or [],
        "errors": errors or [],
        "next_steps": next_steps or [],
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


class EnvelopeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise UmrissError(
            "invalid_arguments",
            message,
            context={"usage": self.format_usage().strip()},
            hint="Run the command with --help to inspect its current arguments.",
        )


_GROUP_COMMANDS = {
    "project",
    "battery",
    "question",
    "marginal",
    "marginals",
    "design",
    "support",
    "baseline",
    "validate",
    "twins",
    "report-data",
    "plot",
}


def canonical_command(argv: list[str]) -> str:
    positional = [token for token in argv if not token.startswith("-")]
    if not positional:
        return "umriss"
    parts = ["umriss", positional[0]]
    if positional[0] in _GROUP_COMMANDS and len(positional) > 1:
        parts.append(positional[1])
    return " ".join(parts)


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    data = init_workspace(args.output_dir or "umriss_work")
    return envelope(
        "umriss init",
        "ok",
        data,
        next_steps=[
            "umriss battery import --metadata <metadata.json>",
            "umriss design create --metadata <metadata.json> --preset pattern-coverage --out design.yaml",
        ],
    )


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    project_id = active_project_id()
    pdir = project_dir(project_id)
    data = {
        "active_project": project_id,
        "project_path": str(pdir),
        "batteries": len(list((pdir / "batteries").glob("*"))) if (pdir / "batteries").exists() else 0,
        "support_prompts": len(list((pdir / "support_prompts").glob("*.jsonl")))
        if (pdir / "support_prompts").exists()
        else 0,
        "support_banks": len(list((pdir / "support_banks").glob("*_probabilities.csv")))
        if (pdir / "support_banks").exists()
        else 0,
        "evaluations": len(list((pdir / "evaluations").glob("*_summary.csv")))
        if (pdir / "evaluations").exists()
        else 0,
    }
    # Store runs first (the workspace is authoritative for tagged pipeline
    # state), then external runs found through their manifests.
    runs = []
    for tag in list_run_tags():
        directory = run_dir(tag)
        try:
            stage = next_for_artifacts(tag, prompt_dir=directory, raw_dir=directory,
                                       bank_dir=directory, derived_dir=directory)["stage"]
        except UmrissError:
            stage = "unknown"
        runs.append({"tag": tag, "location": "store", "run_dir": str(directory), "stage": stage})
    for manifest_path in discover_run_manifests():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tag = manifest.get("tag")
        if not tag or any(run["tag"] == tag for run in runs):
            continue
        try:
            stage = next_for_artifacts(tag, prompt_dir=manifest_path.parent,
                                       raw_dir=manifest_path.parent,
                                       bank_dir=manifest_path.parent,
                                       derived_dir=manifest_path.parent)["stage"]
        except UmrissError:
            stage = "unknown"
        runs.append({
            "tag": tag,
            "location": "external",
            "workflow": manifest.get("workflow"),
            "manifest": str(manifest_path),
            "stage": stage,
        })
    data["runs"] = runs
    data["run_count"] = len(runs)
    defaults = get_defaults()
    data["active_battery"] = defaults.get("battery")
    data["active_design"] = defaults.get("design")
    return envelope(
        "umriss status", "ok", data,
        next_steps=[f"umriss next --tag {runs[-1]['tag']}"] if runs else [],
    )


def cmd_export(args: argparse.Namespace) -> dict[str, Any]:
    """Publish a store run's artifacts to a plain directory (replication packages)."""
    import shutil

    directory = run_dir(args.tag)
    if not directory.exists():
        raise UmrissError(
            "not_found",
            f"No store run for tag: {args.tag}.",
            context={"known_tags": list_run_tags()},
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for entry in sorted(directory.iterdir()):
        if entry.is_file():
            shutil.copyfile(entry, out_dir / entry.name)
            copied.append(entry.name)
    return envelope(
        "umriss export", "ok",
        {"tag": args.tag, "out": str(out_dir), "files": copied, "file_count": len(copied)},
        next_steps=[f"Commit {out_dir} alongside your analysis for replication."],
    )


def cmd_project_create(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss project create", "ok", create_project(args.project_id, title=args.title, use=args.use))


def cmd_project_use(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss project use", "ok", use_project(args.project_id))


def cmd_project_current(args: argparse.Namespace) -> dict[str, Any]:
    project_id = active_project_id()
    return envelope(
        "umriss project current", "ok", {"active_project": project_id, "project_path": str(project_dir(project_id))}
    )


def cmd_project_list(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss project list", "ok", {"projects": list_projects()})


def cmd_project_show(args: argparse.Namespace) -> dict[str, Any]:
    project_id = args.project_id
    pdir = project_dir(project_id)
    if not pdir.exists():
        raise UmrissError("not_found", f"Project does not exist: {project_id}.")
    data = read_json(pdir / "project.json")
    data["project_path"] = str(pdir)
    return envelope("umriss project show", "ok", data)


def cmd_battery_inspect(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.metadata)
    data = inspect_metadata(path)
    if path.name == "battery.json" and path.parent.parent.name == "batteries":
        data["battery_id"] = path.parent.name
    warnings = data.pop("warnings")
    return envelope(
        "umriss battery inspect", "ok", data, warnings=warnings,
        next_steps=[f"umriss battery import --metadata {args.metadata}"],
    )


def cmd_battery_import(args: argparse.Namespace) -> dict[str, Any]:
    data = import_battery(Path(args.metadata), args.battery_id, args.title)
    return envelope(
        "umriss battery import", "ok",
        data,
        next_steps=[
            f"umriss design create --metadata {data['battery_id']} --preset pattern-coverage --out design.yaml",
            "umriss battery list",
        ],
    )


def cmd_battery_list(args: argparse.Namespace) -> dict[str, Any]:
    ids = list_battery_ids()
    return envelope(
        "umriss battery list", "ok",
        {"batteries": ids, "active_battery": get_defaults().get("battery"), "active_project": active_project_id()},
        next_steps=(["umriss battery use <id>"] if ids else ["umriss battery import --metadata <metadata.json>"]),
    )


def cmd_battery_use(args: argparse.Namespace) -> dict[str, Any]:
    stored = state_battery_dir(args.battery_id) / "battery.json"
    if not stored.exists():
        raise UmrissError(
            "not_found",
            f"Battery is not imported: {args.battery_id}.",
            context={"known_batteries": list_battery_ids()},
            hint="Import it first with `umriss battery import --metadata <metadata.json>`.",
        )
    set_default("battery", args.battery_id)
    return envelope(
        "umriss battery use", "ok",
        {"active_battery": args.battery_id},
        next_steps=["umriss design create --preset pattern-coverage --out design.yaml",
                    "umriss status"],
    )


def cmd_design_use(args: argparse.Namespace) -> dict[str, Any]:
    if not design_path(args.design_id).exists():
        raise UmrissError(
            "not_found",
            f"Design is not imported: {args.design_id}.",
            context={"known_designs": list_design_ids()},
            hint="Import it first with `umriss design import --design <design.yaml>`.",
        )
    set_default("design", args.design_id)
    return envelope(
        "umriss design use", "ok",
        {"active_design": args.design_id},
        next_steps=["umriss design validate", "umriss status"],
    )


def cmd_design_import(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.design)
    if not source.exists():
        raise UmrissError("not_found", f"Design file not found: {source}.")
    load_design_config(source)  # fail closed on unparseable designs
    design_id = args.design_id or source.stem
    target = design_path(design_id)
    if target.exists() and not args.force:
        raise UmrissError(
            "already_exists",
            f"Design already imported: {design_id}.",
            hint="Pass --force to replace it, or choose another --design-id.",
        )
    target.write_bytes(source.read_bytes())
    return envelope(
        "umriss design import", "ok",
        {"design_id": design_id, "design_path": str(target)},
        next_steps=[f"umriss design validate --metadata <battery_id> --design {design_id}"],
    )


def cmd_design_list(args: argparse.Namespace) -> dict[str, Any]:
    ids = list_design_ids()
    return envelope(
        "umriss design list", "ok",
        {"designs": ids, "active_design": get_defaults().get("design"), "active_project": active_project_id()},
        next_steps=([] if ids else ["umriss design import --design <design.yaml>"]),
    )


def cmd_battery_create(args: argparse.Namespace) -> dict[str, Any]:
    return envelope(
        "umriss battery create", "ok", create_battery(args),
        next_steps=[f"umriss question add --battery {args.battery_id} --item <item> --question-stem <stem> --item-text <text> --option <a> --option <b> --scale-type nominal"],
    )


def cmd_question_add(args: argparse.Namespace) -> dict[str, Any]:
    return envelope(
        "umriss question add", "ok", add_question(args),
        next_steps=[f"umriss marginal add --battery {args.battery} --item {args.item} --proportion <p1> --proportion <p2> ..."],
    )


def cmd_marginal_add(args: argparse.Namespace) -> dict[str, Any]:
    return envelope(
        "umriss marginal add", "ok", add_marginal(args),
        next_steps=[f"umriss battery compile --battery {args.battery} --path <metadata.json>"],
    )


def cmd_battery_compile(args: argparse.Namespace) -> dict[str, Any]:
    metadata = compile_battery(args.battery, Path(args.path) if args.path else None)
    return envelope(
        "umriss battery compile", "ok",
        {"battery": args.battery, "items": len(metadata["items"]), "path": args.path},
        next_steps=[
            f"umriss design create --metadata {args.path or '<metadata.json>'} --preset pattern-coverage --out design.yaml",
        ],
    )


def cmd_battery_export_edsl(args: argparse.Namespace) -> dict[str, Any]:
    return envelope(
        "umriss battery export-edsl",
        "ok",
        export_edsl_survey(
            Path(args.metadata),
            Path(args.path),
            use_code=args.use_code,
            probabilistic_resolution=args.probabilistic_resolution,
            resolution_seed=args.resolution_seed,
        ),
    )


def cmd_marginals_import(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    if args.respondents:
        truth = weighted_truth_from_respondents(metadata, Path(args.respondents))
    elif args.truth_from == "metadata":
        truth = marginals_from_metadata(metadata)
    else:
        raise UmrissError("invalid_input", "Pass --truth-from metadata or --respondents.")
    write_marginals_long(metadata, truth, Path(args.out))
    return envelope(
        "umriss marginals import", "ok", {"path": args.out, "items": len(truth)},
        next_steps=["umriss validate marginals --metadata <metadata.json> --support <bank_probabilities.csv> --tag <tag> --out <dir>"],
    )


def _metadata_source(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.metadata:
        metadata_path = Path(args.metadata)
        source = {"kind": "file", "path": str(metadata_path)}
    else:
        metadata_path = project_dir(active_project_id()) / "batteries" / args.battery / "battery.json"
        if not metadata_path.exists():
            raise UmrissError(
                "not_found",
                f"Battery does not exist in the active project: {args.battery}.",
                hint="Run `umriss battery create` or pass `--metadata <metadata.json>`.",
            )
        source = {"kind": "workspace", "battery_id": args.battery, "path": str(metadata_path)}
    return read_json(metadata_path), source


def cmd_design_create(args: argparse.Namespace) -> dict[str, Any]:
    metadata, metadata_source = _metadata_source(args)
    design = preset_design(metadata, args.preset, args.size, args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    path.write_text(yaml.safe_dump(design, sort_keys=False))
    report = validate_design(design, metadata)
    return envelope(
        "umriss design create",
        "ok",
        {"design_path": str(path), "metadata_source": metadata_source, **report},
        next_steps=[f"umriss design validate --metadata {metadata_source['path']} --design {path}"],
    )


def cmd_design_validate(args: argparse.Namespace) -> dict[str, Any]:
    metadata, metadata_source = _metadata_source(args)
    design = load_design_config(Path(args.design))
    try:
        report = validate_design(design, metadata)
    except ValueError as exc:
        raise UmrissError("design_invalid", str(exc)) from exc
    return envelope(
        "umriss design validate",
        "ok",
        {"design_path": args.design, "metadata_source": metadata_source, **report},
    )


def cmd_support_build(args: argparse.Namespace) -> dict[str, Any]:
    metadata, metadata_source = _metadata_source(args)
    tag = args.tag
    try:
        if args.design:
            design_path = Path(args.design)
            config = load_design_config(design_path)
            resolved = resolve_design(config, metadata, size=args.n_support, seed=args.seed)
        else:
            design_path = None
            resolved = preset_design(metadata, args.preset, args.n_support, args.seed or 20260625)
        rows = compile_support_plan(metadata, tag, resolved, design_path)
    except ValueError as exc:
        code = "design_too_small" if str(exc).startswith("DESIGN_TOO_SMALL") else "design_invalid"
        raise UmrissError(code, str(exc)) from exc
    out_dir = Path(args.out)
    output_paths = [
        out_dir / f"{tag}_resolved_design.yaml",
        out_dir / f"{tag}_support_plan.csv",
        out_dir / f"{tag}_coverage.csv",
        out_dir / f"{tag}_prompts.jsonl",
        out_dir / f"{tag}_prompts.html",
    ]
    manifest_path = out_dir / f"{tag}_build_manifest.json"
    provenance = build_provenance(
        "umriss support build",
        inputs={
            "metadata": Path(metadata_source["path"]),
            "design": Path(args.design) if args.design else None,
        },
        parameters={
            "tag": tag,
            "preset": args.preset,
            "n_support": args.n_support,
            "seed": args.seed,
            "resolved_design": resolved,
        },
    )
    existing = guard_manifest(manifest_path, provenance, outputs=output_paths, force=args.force)
    if existing is not None:
        return envelope(
            "umriss support build",
            "ok",
            {**existing["data"], "reused": True},
        )
    paths = write_support_outputs(rows, metadata, tag, out_dir, resolved)
    data = {
        **paths,
        "rows": len(rows),
        "tag": tag,
        "design": args.design,
        "preset": args.preset,
        "metadata_source": metadata_source,
        "manifest_path": str(manifest_path),
        "reused": False,
    }
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "umriss_support_build",
            "provenance": provenance,
            "outputs": {
                path.name: {"path": str(path), "sha256": path_sha256(path)}
                for path in output_paths
            },
            "data": data,
        },
    )
    return envelope(
        "umriss support build",
        "ok",
        data,
    )


def cmd_support_export(args: argparse.Namespace) -> dict[str, Any]:
    data = export_support_jobs(
        Path(args.prompts),
        Path(args.path),
        model=args.model,
        service_name=args.service_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        tag=args.tag,
        registration_out=Path(args.registration_out) if args.registration_out else None,
        job_ids_path=Path(args.job_ids) if args.job_ids else None,
        force=args.force,
    )
    return envelope("umriss support export", "ok", data, next_steps=data.get("next_steps", []))


def cmd_support_register_results(args: argparse.Namespace) -> dict[str, Any]:
    data = register_results(
        Path(args.results),
        Path(args.prompts) if args.prompts else None,
        args.tag,
        Path(args.out),
        force=args.force,
    )
    return envelope(
        "umriss support register-results",
        "ok",
        data,
        next_steps=[
            f"umriss support parse --raw {data['raw_path']} --metadata <metadata.json> "
            f"--tag {args.tag} --out {Path(args.out).parent / 'bank'}"
        ],
    )


def cmd_support_audit_results(args: argparse.Namespace) -> dict[str, Any]:
    data = audit_result_attempts(
        [Path(path) for path in args.results],
        Path(args.prompts),
        args.tag,
        Path(args.out),
        force=args.force,
    )
    if data["complete"]:
        next_steps = [
            f"umriss support parse --raw {data['merged_raw_path']} --metadata <metadata.json> "
            f"--tag {args.tag} --out {Path(args.out).parent / 'bank'}"
        ]
        warnings = []
    else:
        retry_jobs = Path(args.out) / f"{args.tag}_retry.jobs.ep"
        next_steps = [
            f"umriss support export --prompts {args.prompts} "
            f"--job-ids {data['missing_job_ids_path']} --path {retry_jobs} "
            f"--tag {args.tag}_retry --registration-out {args.out}"
        ]
        warnings = [
            {
                "code": "incomplete_results",
                "message": f"{data['missing_jobs']} prompt jobs still lack a valid response.",
            }
        ]
    return envelope(
        "umriss support audit-results",
        "ok",
        data,
        warnings=warnings,
        next_steps=next_steps,
    )


def cmd_support_parse(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = parse_support(Path(args.raw), metadata, args.tag, Path(args.out))
    return envelope(
        "umriss support parse", "ok", data,
        next_steps=[f"umriss support uniformity --support {data.get('probabilities', '<bank_probabilities.csv>')} --metadata {args.metadata} --out {args.out}"],
    )


def cmd_baseline_build(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = build_baseline_prompts(
        metadata,
        args.tag,
        Path(args.out),
        mode=args.mode,
        respondents_path=Path(args.respondents) if args.respondents else None,
    )
    return envelope(
        "umriss baseline build", "ok", data,
        next_steps=[f"umriss baseline export --prompts {data.get('prompts', '<prompts.jsonl>')} --path <tag>.jobs.ep --tag {args.tag} --registration-out {args.out}"],
    )


def cmd_baseline_export(args: argparse.Namespace) -> dict[str, Any]:
    data = export_support_jobs(
        Path(args.prompts),
        Path(args.path),
        model=args.model,
        service_name=args.service_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workflow="baseline",
        tag=args.tag,
        registration_out=Path(args.registration_out) if args.registration_out else None,
        job_ids_path=Path(args.job_ids) if args.job_ids else None,
        force=args.force,
    )
    return envelope("umriss baseline export", "ok", data, next_steps=data.get("next_steps", []))


def cmd_baseline_register_results(args: argparse.Namespace) -> dict[str, Any]:
    data = register_results(
        Path(args.results),
        Path(args.prompts),
        args.tag,
        Path(args.out),
        force=args.force,
    )
    return envelope(
        "umriss baseline register-results", "ok", data,
        next_steps=[f"umriss baseline parse --raw {data.get('raw', '<raw.csv>')} --prompts {args.prompts} --metadata <metadata.json> --tag {args.tag} --out {args.out}"],
    )


def cmd_baseline_parse(args: argparse.Namespace) -> dict[str, Any]:
    data = parse_baseline_results(
        Path(args.raw),
        Path(args.prompts),
        read_json(Path(args.metadata)),
        args.tag,
        Path(args.out),
    )
    return envelope(
        "umriss baseline parse", "ok", data,
        next_steps=["umriss compare --run <tag>=<battery>:<bank> --derived <dir> --out <comparison.csv>"],
    )


def cmd_support_inspect(args: argparse.Namespace) -> dict[str, Any]:
    data = inspect_artifact(
        prompts=Path(args.prompts) if args.prompts else None,
        raw=Path(args.raw) if args.raw else None,
        bank=Path(args.bank) if args.bank else None,
        diagnostics=Path(args.diagnostics) if args.diagnostics else None,
        summary=Path(args.summary) if args.summary else None,
    )
    return envelope("umriss support inspect", "ok", data)


def cmd_support_uniformity(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = write_uniformity(
        Path(args.support),
        metadata,
        args.tolerance,
        Path(args.out),
        args.max_duplicate_fraction,
        args.min_joint_pattern_fraction,
    )
    status = "ok" if data["passes"] else "needs_augmentation"
    if data["passes"]:
        next_steps = [
            f"umriss fit --support {args.support} --metadata {args.metadata} --tag <tag> --out <dir>",
        ]
    else:
        next_steps = [
            f"umriss support augment-uniform --support {args.support} --metadata {args.metadata} --tag <tag> --n-add <n> --out <dir>",
        ]
    return envelope("umriss support uniformity", status, data, next_steps=next_steps)


def cmd_support_augment_uniform(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    try:
        data = build_uniform_augmentation(
            Path(args.support),
            metadata,
            args.tag,
            args.n_add,
            args.tolerance,
            args.seed,
            Path(args.out),
        )
    except ValueError as exc:
        raise UmrissError("invalid_input", str(exc)) from exc
    return envelope(
        "umriss support augment-uniform", "ok", data,
        next_steps=[
            f"umriss support export --prompts {data.get('prompts', '<additions_prompts.jsonl>')} --path {args.tag}_additions.jobs.ep --tag {args.tag}_additions --registration-out <dir>",
        ],
    )


def cmd_support_merge(args: argparse.Namespace) -> dict[str, Any]:
    try:
        data = merge_support_banks(Path(args.base), Path(args.additions), args.tag, Path(args.out))
    except ValueError as exc:
        raise UmrissError("invalid_input", str(exc)) from exc
    return envelope(
        "umriss support merge", "ok", data,
        next_steps=[f"umriss support uniformity --support {data.get('probabilities', '<merged_probabilities.csv>')} --metadata <metadata.json> --out {args.out}"],
    )


def cmd_fit(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    support, mats = load_support_matrix(Path(args.support))
    if args.respondents:
        truth = weighted_truth_from_respondents(metadata, Path(args.respondents))
    else:
        truth = marginals_from_metadata(metadata)
    items = list(metadata["items"])
    held_in = [item for item in items if item not in set(args.exclude_item or [])]
    if args.include_item:
        held_in = list(args.include_item)
    heldout = ",".join(args.exclude_item or [])
    selected_rho, fit = fit_weights(mats, truth, held_in, args.rho)
    data = write_fit_outputs(support, mats, metadata, args.tag, heldout, selected_rho, fit, Path(args.out))
    return envelope(
        "umriss fit", "ok", data,
        next_steps=[
            f"umriss validate marginals --metadata {args.metadata} --support {args.support} --tag {args.tag} --out {args.out}",
            f"umriss predict --support {args.support} --weights {data.get('weights', '<weights.csv>')} --metadata {args.metadata} --out {args.out}",
        ],
    )


def cmd_validate_marginals(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    try:
        data = run_marginal_validation(
            metadata,
            args.tag,
            Path(args.out),
            raw_path=Path(args.raw) if args.raw else None,
            support_path=Path(args.support) if args.support else None,
            respondents_path=Path(args.respondents) if args.respondents else None,
            one_shot_path=Path(args.one_shot) if args.one_shot else None,
            two_step_path=Path(args.conditioned_direct) if args.conditioned_direct else None,
            rho_values=args.rho,
            uniform_tolerance=args.uniform_tolerance,
            max_duplicate_fraction=args.max_duplicate_fraction,
            min_joint_pattern_fraction=args.min_joint_pattern_fraction,
            allow_nonuniform_support=args.allow_nonuniform_support,
        )
    except ValueError as exc:
        if str(exc).startswith("SUPPORT_NOT_UNIFORM"):
            code = "support_not_uniform"
        elif str(exc).startswith("SUPPORT_NOT_DIVERSE"):
            code = "support_not_diverse"
        elif str(exc).startswith("BATTERY_TOO_SMALL"):
            code = "battery_too_small"
        else:
            code = "invalid_input"
        raise UmrissError(code, str(exc)) from exc
    return envelope(
        "umriss validate marginals", "ok", data,
        next_steps=["umriss compare --run <tag>=<battery>:<bank> --derived <dir> --out <comparison.csv>"],
    )


def cmd_predict(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = predict_from_weights(Path(args.support), Path(args.weights), metadata, args.item or [], Path(args.out))
    return envelope(
        "umriss predict", "ok", data,
        next_steps=["umriss twins export-edsl --points <points.csv> --weights <weights.csv> --path <agents.ep>"],
    )


def cmd_twins_export_edsl(args: argparse.Namespace) -> dict[str, Any]:
    data = export_edsl_agents(
        [Path(path) for path in args.points],
        Path(args.weights),
        Path(args.path),
        persona_trait=args.persona_trait,
        holdout=args.holdout,
        minimum_weight=args.minimum_weight,
    )
    return envelope(
        "umriss twins export-edsl", "ok", data,
        next_steps=[f"umriss twins build-survey-jobs --survey <survey.ep> --agents {args.path} --model <service:model> --path <survey.jobs.ep>"],
    )


def cmd_twins_compare_survey(args: argparse.Namespace) -> dict[str, Any]:
    data = compare_edsl_survey(
        Path(args.results),
        Path(args.metadata),
        Path(args.fit_predictions),
        Path(args.out),
        answers_use_code=args.answers_use_code,
    )
    return envelope(
        "umriss twins compare-survey", "ok", data,
        next_steps=[f"umriss twins plot-survey --comparison {data.get('comparison', '<comparison.csv>')} --out <dir>"],
    )


def cmd_twins_build_survey_jobs(args: argparse.Namespace) -> dict[str, Any]:
    data = build_survey_jobs(
        Path(args.survey),
        Path(args.agents),
        args.model,
        Path(args.path),
        args.service_name,
        args.temperature,
        args.logprobs,
        args.top_logprobs,
        args.limit_agents,
        args.limit_questions,
    )
    return envelope(
        "umriss twins build-survey-jobs", "ok", data,
        next_steps=[
            f"ep run {args.path} --output <survey.results.ep>",
            "umriss twins compare-survey --results <survey.results.ep> --metadata <metadata.json> --fit-predictions <predictions.csv> --out <dir>",
        ],
    )


def cmd_twins_embed_probabilities(args: argparse.Namespace) -> dict[str, Any]:
    data = embed_response_probabilities(
        Path(args.agents),
        Path(args.support),
        Path(args.metadata),
        Path(args.path),
        probability_trait=args.probability_trait,
    )
    return envelope(
        "umriss twins embed-probabilities", "ok", data,
        next_steps=[f"umriss twins analyze-probabilistic-survey --agents {args.path} --metadata {args.metadata} --out <dir>"],
    )


def cmd_twins_build_resolution_experiment(args: argparse.Namespace) -> dict[str, Any]:
    data = build_resolution_experiment(
        Path(args.agents),
        Path(args.support),
        Path(args.metadata),
        Path(args.agents_path),
        Path(args.survey_path),
        resolution_trait=args.resolution_trait,
        seed=args.seed,
    )
    return envelope(
        "umriss twins build-resolution-experiment", "ok", data,
        next_steps=[
            f"ep run {args.path} --output <resolution.results.ep>",
            "umriss twins analyze-resolution --results <resolution.results.ep> --out <dir>",
        ],
    )


def cmd_twins_analyze_resolution(args: argparse.Namespace) -> dict[str, Any]:
    data = analyze_resolution_experiment(
        Path(args.results),
        Path(args.metadata),
        Path(args.out),
        args.tag,
        resolution_trait=args.resolution_trait,
    )
    return envelope(
        "umriss twins analyze-resolution", "ok", data,
        next_steps=["umriss report --comparison <comparison.csv> --out <report.md>"],
    )


def cmd_twins_analyze_probabilistic_survey(args: argparse.Namespace) -> dict[str, Any]:
    data = analyze_probabilistic_survey(
        [Path(path) for path in args.results],
        Path(args.metadata),
        Path(args.fit_predictions),
        Path(args.out),
        args.tag,
        simulations=args.simulations,
        simulation_seed=args.simulation_seed,
    )
    return envelope(
        "umriss twins analyze-probabilistic-survey", "ok", data,
        next_steps=["umriss twins compare-survey --results <survey.results.ep> --metadata <metadata.json> --fit-predictions <predictions.csv> --out <dir>"],
    )


def cmd_twins_plot_survey(args: argparse.Namespace) -> dict[str, Any]:
    data = plot_survey_comparison(
        Path(args.comparison),
        Path(args.out),
        simulations_path=Path(args.simulations) if args.simulations else None,
    )
    return envelope("umriss twins plot-survey", "ok", data)


def cmd_twins_analyze_logprobs(args: argparse.Namespace) -> dict[str, Any]:
    data = analyze_token_probabilities(
        Path(args.results),
        Path(args.metadata),
        Path(args.support),
        Path(args.out),
        args.tag,
    )
    return envelope("umriss twins analyze-logprobs", "ok", data)


def cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.recipe == "battery-designed":
        data = battery_designed_recipe(Path(args.derived), Path(args.out))
    elif args.recipe == "pattern-coverage":
        data = pattern_coverage_recipe(Path(args.derived), Path(args.out))
    else:
        data = compare_runs(args.run or [], Path(args.derived), Path(args.out), args.comparison_group)
    return envelope(
        "umriss compare", "ok", data,
        next_steps=[f"umriss report --comparison {args.out} --out <report.md>"],
    )


def cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    data = write_report(args.tag, Path(args.derived), Path(args.out))
    return envelope(
        "umriss report", "ok", data,
        next_steps=["umriss report-data build --tag <tag> --out <report_data_dir>"],
    )


def cmd_report_data_build(args: argparse.Namespace) -> dict[str, Any]:
    data = build_report_data(Path(args.derived), Path(args.out))
    return envelope("umriss report-data build", "ok", data)


def cmd_plot_validation(args: argparse.Namespace) -> dict[str, Any]:
    data = plot_validation(
        Path(args.derived),
        args.tag,
        Path(args.out),
        image_format=args.format,
        top_personas=args.top_personas,
    )
    return envelope("umriss plot validation", "ok", data)


def cmd_guide(args: argparse.Namespace) -> dict[str, Any]:
    topic = args.topic_flag or args.topic or "workflow"
    return envelope("umriss guide", "ok", guide(topic))


def discover_run_manifests() -> list[Path]:
    """All `<tag>_manifest.json` run manifests beneath the working directory."""
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".umriss"}
    matches: list[Path] = []
    stack = [Path(".")]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir():
                if entry.name not in skip and not entry.name.startswith("."):
                    stack.append(entry)
            elif entry.name.endswith("_manifest.json"):
                matches.append(entry)
    return sorted(matches)


def discover_tag_manifest(tag: str) -> Path | None:
    """Locate `<tag>_manifest.json` beneath the working directory.

    The pipeline has no fixed layout (every command takes --out), so
    resumability comes from finding the run manifest wherever it was written.
    Ambiguity is an error, never a guess.
    """
    matches = [m for m in discover_run_manifests() if m.name == f"{tag}_manifest.json"]
    if len(matches) > 1:
        raise UmrissError(
            "ambiguous_tag",
            f"Found {len(matches)} manifests for tag `{tag}`.",
            context={"matches": [str(match) for match in sorted(matches)]},
            hint="Pass --prompt-dir to select the run explicitly.",
        )
    return matches[0] if matches else None


def cmd_next(args: argparse.Namespace) -> dict[str, Any]:
    if args.tag:
        prompt_dir = Path(args.prompt_dir) if args.prompt_dir else None
        if prompt_dir is None and ROOT.exists():
            stored = run_dir(args.tag)
            if stored.exists():
                prompt_dir = stored
        if prompt_dir is None:
            manifest = discover_tag_manifest(args.tag)
            prompt_dir = manifest.parent if manifest else Path(".")
        data = next_for_artifacts(
            args.tag,
            metadata=Path(args.metadata) if args.metadata else None,
            design=Path(args.design) if args.design else None,
            prompt_dir=prompt_dir,
            raw_dir=Path(args.raw_dir) if args.raw_dir else prompt_dir,
            bank_dir=Path(args.bank_dir) if args.bank_dir else prompt_dir,
            derived_dir=Path(args.derived_dir) if args.derived_dir else prompt_dir,
        )
        return envelope("umriss next", "ok", data)
    try:
        status = cmd_status(args)["data"]
    except UmrissError:
        return envelope("umriss next", "ok", {"recommendation": "Run `umriss init`.", "reason": "No active workspace."})
    runs = status.get("runs") or []
    incomplete = [run for run in runs if run["stage"] not in ("complete", "unknown")]
    if incomplete:
        run = incomplete[-1]
        return envelope("umriss next", "ok", {
            "recommendation": f"umriss next --tag {run['tag']}",
            "reason": f"Run `{run['tag']}` is at stage `{run['stage']}` ({run.get('manifest') or run.get('run_dir')}).",
            "runs": runs,
        })
    if status["batteries"] == 0:
        recommendation = "umriss battery import --metadata <metadata.json>"
    elif status["support_prompts"] == 0:
        recommendation = "umriss design create --metadata <metadata.json> --preset pattern-coverage --out design.yaml"
    elif status["support_banks"] == 0:
        recommendation = "umriss support export --prompts <tag>_prompts.jsonl --path <tag>.jobs.ep"
    elif status["evaluations"] == 0:
        recommendation = "umriss validate marginals --support <bank_probabilities.csv> --metadata <metadata.json> --tag <tag> --out <dir>"
    else:
        recommendation = "umriss compare --run <tag>=<battery>:<bank> --derived <dir> --out <comparison.csv>"
    return envelope("umriss next", "ok", {"recommendation": recommendation, "status": status})


def cmd_capabilities(args: argparse.Namespace) -> dict[str, Any]:
    return envelope(
        "umriss capabilities",
        "ok",
        {
            "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
            "output_contract": (
                "Every command prints one JSON envelope "
                "{schema_version, command, status, argv, data, warnings, errors, next_steps} "
                "to stdout. Expected failures exit 1; unexpected internal failures exit 2; "
                "both remain JSON-enveloped."
            ),
            "execution_boundary": (
                "umriss builds durable EDSL .jobs.ep packages and stops; model execution is "
                "external via `ep run` and requires user approval. umriss never executes "
                "packaged model calls."
            ),
            "workflow_commands": ["umriss guide", "umriss next", "umriss next --tag <tag>"],
            "provenance": (
                "Exports write hash-fingerprinted manifests with an output_conflict guard; "
                "audit-results preserves every attempt with per-row source attribution and "
                "emits missing_job_ids.csv for retry-only export."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = EnvelopeArgumentParser(
        prog="umriss",
        description="Build auditable digital twins from reported survey marginals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_init)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("export")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export)
    sub.add_parser("version").set_defaults(func=lambda args: envelope(
        "umriss version", "ok",
        {
            "version": __version__,
            "package_path": str(Path(__file__).resolve().parent),
            "envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        },
    ))
    sub.add_parser("capabilities").set_defaults(func=cmd_capabilities)

    project = sub.add_parser("project").add_subparsers(dest="project_command", required=True)
    p = project.add_parser("create")
    p.add_argument("project_id")
    p.add_argument("--title")
    p.add_argument("--use", action="store_true")
    p.set_defaults(func=cmd_project_create)
    p = project.add_parser("use")
    p.add_argument("project_id")
    p.set_defaults(func=cmd_project_use)
    project.add_parser("current").set_defaults(func=cmd_project_current)
    project.add_parser("list").set_defaults(func=cmd_project_list)
    p = project.add_parser("show")
    p.add_argument("project_id")
    p.set_defaults(func=cmd_project_show)

    battery = sub.add_parser("battery").add_subparsers(dest="battery_command", required=True)
    p = battery.add_parser("inspect")
    p.add_argument("metadata")
    p.set_defaults(func=cmd_battery_inspect)
    p = battery.add_parser("import")
    p.add_argument("--metadata", required=True)
    p.add_argument("--battery-id")
    p.add_argument("--title")
    p.set_defaults(func=cmd_battery_import)
    battery.add_parser("list").set_defaults(func=cmd_battery_list)
    p = battery.add_parser("use")
    p.add_argument("battery_id")
    p.set_defaults(func=cmd_battery_use)
    p = battery.add_parser("create")
    p.add_argument("--battery-id", required=True)
    p.add_argument("--wave", required=True)
    p.add_argument("--battery", required=True)
    p.add_argument("--topic", required=True)
    p.add_argument("--context", required=True)
    p.set_defaults(func=cmd_battery_create)
    p = battery.add_parser("compile")
    p.add_argument("--battery", required=True)
    p.add_argument("--path")
    p.set_defaults(func=cmd_battery_compile)
    p = battery.add_parser("export-edsl")
    p.add_argument("--metadata")
    p.add_argument("--path", required=True)
    p.add_argument("--use-code", action="store_true")
    p.add_argument("--probabilistic-resolution", choices=["none", "sample", "mode"])
    p.add_argument("--resolution-seed", type=int)
    p.set_defaults(func=cmd_battery_export_edsl)

    question = sub.add_parser("question").add_subparsers(dest="question_command", required=True)
    p = question.add_parser("add")
    p.add_argument("--battery", required=True)
    p.add_argument("--item", required=True)
    p.add_argument("--variable")
    p.add_argument("--question-stem", required=True)
    p.add_argument("--item-text", required=True)
    p.add_argument("--option", action="append", required=True)
    p.add_argument("--option-code", action="append")
    p.add_argument("--scale-type", choices=["ordinal", "nominal"], required=True)
    p.add_argument("--scale-direction", choices=["low_to_high", "high_to_low"])
    p.set_defaults(func=cmd_question_add)

    marginal = sub.add_parser("marginal").add_subparsers(dest="marginal_command", required=True)
    p = marginal.add_parser("add")
    p.add_argument("--battery", required=True)
    p.add_argument("--item", required=True)
    p.add_argument("--proportion", action="append", type=float, required=True)
    p.add_argument("--source")
    p.add_argument("--normalize", action="store_true")
    p.set_defaults(func=cmd_marginal_add)

    marginals = sub.add_parser("marginals").add_subparsers(dest="marginals_command", required=True)
    p = marginals.add_parser("import")
    p.add_argument("--metadata")
    truth_source = p.add_mutually_exclusive_group(required=True)
    truth_source.add_argument("--truth-from", choices=["metadata"])
    truth_source.add_argument("--respondents")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_marginals_import)

    design_cmd = sub.add_parser("design").add_subparsers(dest="design_command", required=True)
    p = design_cmd.add_parser("create")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--metadata")
    source.add_argument("--battery")
    p.add_argument("--preset", choices=["pattern-coverage", "uniform-patterns"], required=True)
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_design_create)
    p = design_cmd.add_parser("validate")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--metadata")
    source.add_argument("--battery")
    p.add_argument("--design")
    p.set_defaults(func=cmd_design_validate)
    p = design_cmd.add_parser("import")
    p.add_argument("--design", required=True, help="Path to a design YAML/JSON file to store in the active project.")
    p.add_argument("--design-id")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_design_import)
    design_cmd.add_parser("list").set_defaults(func=cmd_design_list)
    p = design_cmd.add_parser("use")
    p.add_argument("design_id")
    p.set_defaults(func=cmd_design_use)

    support = sub.add_parser("support").add_subparsers(dest="support_command", required=True)
    p = support.add_parser("build")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--metadata", help="Read battery metadata from an explicit JSON file.")
    source.add_argument("--battery", help="Read a battery authored in the active .umriss project.")
    design = p.add_mutually_exclusive_group()
    design.add_argument(
        "--preset", choices=["pattern-coverage", "uniform-patterns"], help="Compile a safe built-in preset."
    )
    design.add_argument("--design", help="Use a schema-v1 JSON or YAML support-design file.")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-support", type=int, help="Override design size; feasibility validation still applies.")
    p.add_argument("--seed", type=int, help="Override the design seed.")
    p.add_argument("--out")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_support_build)
    p = support.add_parser("export")
    p.add_argument("--prompts")
    p.add_argument("--path")
    p.add_argument("--model", action="append")
    p.add_argument("--service-name")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2200)
    p.add_argument("--limit", type=int)
    p.add_argument("--tag")
    p.add_argument("--registration-out")
    p.add_argument("--job-ids")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_support_export)
    p = support.add_parser("register-results")
    p.add_argument("--results", required=True)
    p.add_argument("--prompts")
    p.add_argument("--tag", required=True)
    p.add_argument("--out")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_support_register_results)
    p = support.add_parser("audit-results")
    p.add_argument("--results", action="append", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_support_audit_results)
    p = support.add_parser("parse")
    p.add_argument("--raw")
    p.add_argument("--metadata")
    p.add_argument("--tag", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_support_parse)
    p = support.add_parser("inspect")
    p.add_argument("--prompts")
    p.add_argument("--raw")
    p.add_argument("--bank")
    p.add_argument("--diagnostics")
    p.add_argument("--summary")
    p.set_defaults(func=cmd_support_inspect)
    p = support.add_parser("uniformity")
    p.add_argument("--support")
    p.add_argument("--tag")
    p.add_argument("--metadata")
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--max-duplicate-fraction", type=float, default=0.05)
    p.add_argument("--min-joint-pattern-fraction", type=float, default=0.75)
    p.add_argument("--out")
    p.set_defaults(func=cmd_support_uniformity)
    p = support.add_parser("augment-uniform")
    p.add_argument("--support", required=True)
    p.add_argument("--metadata")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-add", type=int, default=64)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support_augment_uniform)
    p = support.add_parser("merge")
    p.add_argument("--base", required=True)
    p.add_argument("--additions", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support_merge)

    baseline = sub.add_parser("baseline").add_subparsers(dest="baseline_command", required=True)
    p = baseline.add_parser("build")
    p.add_argument("--metadata")
    p.add_argument("--respondents")
    p.add_argument("--mode", choices=["one_shot", "conditioned_direct", "both"], default="both")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_baseline_build)
    p = baseline.add_parser("export")
    p.add_argument("--prompts", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--model", action="append")
    p.add_argument("--service-name")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--tag")
    p.add_argument("--registration-out")
    p.add_argument("--job-ids")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_baseline_export)
    p = baseline.add_parser("register-results")
    p.add_argument("--results", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_baseline_register_results)
    p = baseline.add_parser("parse")
    p.add_argument("--raw", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--metadata")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_baseline_parse)

    p = sub.add_parser("fit")
    p.add_argument("--support", required=True)
    p.add_argument("--metadata")
    p.add_argument("--respondents")
    p.add_argument("--exclude-item", action="append")
    p.add_argument("--include-item", action="append")
    p.add_argument("--rho", nargs="+", type=float, default=[0.0003, 0.001, 0.003, 0.01, 0.03])
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fit)

    validate = sub.add_parser("validate").add_subparsers(dest="validate_command", required=True)
    p = validate.add_parser("marginals")
    p.add_argument("--raw")
    p.add_argument("--support")
    p.add_argument("--metadata")
    p.add_argument("--respondents")
    p.add_argument("--one-shot")
    p.add_argument("--conditioned-direct")
    p.add_argument("--rho", nargs="+", type=float, default=[0.0003, 0.001, 0.003, 0.01, 0.03])
    p.add_argument("--uniform-tolerance", type=float, default=0.05)
    p.add_argument("--max-duplicate-fraction", type=float, default=0.05)
    p.add_argument("--min-joint-pattern-fraction", type=float, default=0.75)
    p.add_argument("--allow-nonuniform-support", action="store_true")
    p.add_argument("--tag", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_validate_marginals)

    p = sub.add_parser("predict")
    p.add_argument("--support", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--metadata")
    p.add_argument("--item", action="append")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_predict)

    twins = sub.add_parser("twins").add_subparsers(dest="twins_command", required=True)
    p = twins.add_parser("export-edsl")
    p.add_argument("--points", action="append", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--persona-trait", required=True)
    p.add_argument("--holdout")
    p.add_argument("--minimum-weight", type=float, default=0.0)
    p.add_argument("--path", required=True)
    p.set_defaults(func=cmd_twins_export_edsl)
    p = twins.add_parser("build-survey-jobs")
    p.add_argument("--survey", required=True)
    p.add_argument("--agents", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--service-name")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--logprobs", action="store_true")
    p.add_argument("--top-logprobs", type=int, default=5)
    p.add_argument("--limit-agents", type=int)
    p.add_argument("--limit-questions", type=int)
    p.add_argument("--path", required=True)
    p.set_defaults(func=cmd_twins_build_survey_jobs)
    p = twins.add_parser("embed-probabilities")
    p.add_argument("--agents", required=True)
    p.add_argument("--support", required=True)
    p.add_argument("--metadata")
    p.add_argument("--probability-trait", required=True)
    p.add_argument("--path", required=True)
    p.set_defaults(func=cmd_twins_embed_probabilities)
    p = twins.add_parser("build-resolution-experiment")
    p.add_argument("--agents", required=True)
    p.add_argument("--support", required=True)
    p.add_argument("--metadata")
    p.add_argument("--resolution-trait", required=True)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--agents-path", required=True)
    p.add_argument("--survey-path", required=True)
    p.set_defaults(func=cmd_twins_build_resolution_experiment)
    p = twins.add_parser("analyze-resolution")
    p.add_argument("--results", required=True)
    p.add_argument("--metadata")
    p.add_argument("--resolution-trait", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_twins_analyze_resolution)
    p = twins.add_parser("analyze-probabilistic-survey")
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--metadata")
    p.add_argument("--fit-predictions", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--simulations", type=int, default=0)
    p.add_argument("--simulation-seed", type=int, default=20260725)
    p.set_defaults(func=cmd_twins_analyze_probabilistic_survey)
    p = twins.add_parser("compare-survey")
    p.add_argument("--results", required=True)
    p.add_argument("--metadata")
    p.add_argument("--fit-predictions", required=True)
    p.add_argument("--answers-use-code", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_twins_compare_survey)
    p = twins.add_parser("plot-survey")
    p.add_argument("--comparison", required=True)
    p.add_argument("--simulations")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_twins_plot_survey)
    p = twins.add_parser("analyze-logprobs")
    p.add_argument("--results", required=True)
    p.add_argument("--metadata")
    p.add_argument("--support", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_twins_analyze_logprobs)

    p = sub.add_parser("compare")
    p.add_argument("--run", action="append")
    p.add_argument("--recipe", choices=["generic", "battery-designed", "pattern-coverage"], default="generic")
    p.add_argument("--derived", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--comparison-group", default="default")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("report")
    p.add_argument("--tag", required=True)
    p.add_argument("--derived", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report)

    report_data = sub.add_parser("report-data").add_subparsers(dest="report_data_command", required=True)
    p = report_data.add_parser("build")
    p.add_argument("--derived", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_report_data_build)

    plot = sub.add_parser("plot").add_subparsers(dest="plot_command", required=True)
    p = plot.add_parser("validation")
    p.add_argument("--derived")
    p.add_argument("--tag", required=True)
    p.add_argument("--out")
    p.add_argument("--format", choices=["svg", "png", "pdf"], default="svg")
    p.add_argument("--top-personas", type=int, default=30)
    p.set_defaults(func=cmd_plot_validation)

    p = sub.add_parser("guide")
    guide_topics = ["workflow", "designs", "ep-boundary", "migrating-scripts", "diagnostics"]
    p.add_argument("topic", nargs="?", choices=guide_topics)
    p.add_argument("--topic", dest="topic_flag", choices=guide_topics)
    p.set_defaults(func=cmd_guide)
    p = sub.add_parser("next")
    p.add_argument("--tag")
    p.add_argument("--metadata")
    p.add_argument("--design")
    p.add_argument("--prompt-dir", help="Directory holding <tag>_manifest.json and prompts/jobs. Default: discovered by searching for the run manifest under the working directory.")
    p.add_argument("--raw-dir", help="Registered raw-results directory (default: the discovered run directory, or as recorded in the manifest).")
    p.add_argument("--bank-dir", help="Parsed bank directory (default: the discovered run directory).")
    p.add_argument("--derived-dir", help="Derived outputs directory (default: the discovered run directory).")
    p.set_defaults(func=cmd_next)
    return parser


# Commands whose --metadata/--design are genuinely optional advisory inputs;
# active defaults are not injected and absence is not an error.
_ADVISORY_COMMANDS = {"umriss next"}

# Per-command store defaults for pipeline inputs/outputs, applied when the
# flag is omitted and --tag is set. "@dir" means the tag's store run
# directory; a filename template is a conventional artifact inside it.
# Input templates must already exist (fail closed with the producing command);
# templates suffixed "?" are filled only when present.
_RUN_DEFAULTS: dict[str, dict[str, str]] = {
    "umriss support build": {"out": "@dir"},
    "umriss support export": {"prompts": "{tag}_prompts.jsonl", "path": "{tag}.jobs.ep!"},
    "umriss support register-results": {"prompts": "{tag}_prompts.jsonl?", "out": "@dir"},
    "umriss support parse": {"raw": "{tag}_raw.csv", "out": "@dir"},
    "umriss support uniformity": {"support": "{tag}_probabilities.csv", "out": "@dir"},
    "umriss validate marginals": {"support": "{tag}_probabilities.csv", "out": "@dir"},
    "umriss plot validation": {"derived": "@dir", "out": "@dir"},
}

_PRODUCED_BY = {
    "{tag}_prompts.jsonl": "umriss support build --tag {tag}",
    "{tag}_raw.csv": "umriss support register-results --results <results.ep> --tag {tag}",
    "{tag}_probabilities.csv": "umriss support parse --tag {tag}",
}


def _apply_run_defaults(args: argparse.Namespace, command: str) -> dict[str, str]:
    """Fill omitted pipeline paths from the tag's store run directory."""
    spec = _RUN_DEFAULTS.get(command)
    resolved: dict[str, str] = {}
    if not spec or not ROOT.exists():
        return resolved
    tag = getattr(args, "tag", None)
    if not tag:
        missing = [name for name in spec if getattr(args, name, None) is None]
        if missing:
            raise UmrissError(
                "invalid_arguments",
                f"Pass --tag to use the workspace run store, or give explicit paths: {', '.join('--' + m.replace('_', '-') for m in missing)}.",
            )
        return resolved
    directory = run_dir(tag, create=True)
    for name, template in spec.items():
        if getattr(args, name, None) is not None:
            continue
        if template == "@dir":
            value = directory
        else:
            optional = template.endswith("?")
            output_file = template.endswith("!")
            template = template.rstrip("?!")
            value = directory / template.format(tag=tag)
            if not value.exists() and not output_file:
                if optional:
                    continue
                producer = _PRODUCED_BY.get(template, "the previous pipeline stage").format(tag=tag)
                raise UmrissError(
                    "not_found",
                    f"Expected {value} in the run store.",
                    hint=f"Run `{producer}` first, or pass --{name.replace('_', '-')} explicitly.",
                )
        setattr(args, name, str(value))
        resolved[name] = str(value)
    return resolved


def _resolve_store_ids(args: argparse.Namespace, command: str) -> dict[str, str]:
    """Let --metadata/--design accept workspace ids, and fill from `use` defaults.

    Resolution order per flag: an existing file path always wins (explicit
    paths behave exactly as before); a bare token matching a stored battery or
    design resolves to its file; an omitted flag falls back to the project's
    active default (`battery use` / `design use`); anything else fails closed
    with the known ids. Returns what was implicitly resolved, so envelopes can
    echo it and captured outputs stay self-describing.
    """
    resolved: dict[str, str] = {}
    if command in {"umriss battery import", "umriss design import"} or command in _ADVISORY_COMMANDS:
        return resolved
    if not ROOT.exists():
        return resolved

    def bare(value: str) -> bool:
        return "/" not in value and "\\" not in value and not value.startswith(".")

    metadata = getattr(args, "metadata", None)
    if isinstance(metadata, str) and metadata and not Path(metadata).is_file() and bare(metadata):
        stored = state_battery_dir(metadata) / "battery.json"
        if stored.exists():
            args.metadata = str(stored)
        elif not metadata.endswith((".json", ".yaml", ".csv")):
            raise UmrissError(
                "not_found",
                f"'{metadata}' is neither a file nor an imported battery.",
                context={"known_batteries": list_battery_ids()},
                hint="Import it first with `umriss battery import --metadata <metadata.json>`.",
            )
    elif metadata is None and "metadata" in vars(args) and getattr(args, "battery", None) is None:
        default_battery = get_defaults().get("battery")
        if default_battery:
            stored = state_battery_dir(default_battery) / "battery.json"
            if not stored.exists():
                raise UmrissError(
                    "not_found",
                    f"The active battery '{default_battery}' no longer exists in the project.",
                    context={"known_batteries": list_battery_ids()},
                    hint="Set a new one with `umriss battery use <id>`.",
                )
            args.metadata = str(stored)
            resolved["battery"] = default_battery
        else:
            raise UmrissError(
                "missing_battery",
                "No battery given and no active battery is set.",
                context={"known_batteries": list_battery_ids()},
                hint="Pass --metadata <file-or-id> (or --battery <id>), or set a default with `umriss battery use <id>`.",
            )

    design = getattr(args, "design", None)
    if isinstance(design, str) and design and not Path(design).is_file() and bare(design):
        stored = design_path(design) if not design.endswith((".yaml", ".yml", ".json")) else None
        if stored is not None and stored.exists():
            args.design = str(stored)
        elif not design.endswith((".yaml", ".yml", ".json")):
            raise UmrissError(
                "not_found",
                f"'{design}' is neither a file nor an imported design.",
                context={"known_designs": list_design_ids()},
                hint="Import it first with `umriss design import --design <design.yaml>`.",
            )
    elif design is None and "design" in vars(args) and getattr(args, "preset", None) is None:
        default_design = get_defaults().get("design")
        if default_design:
            stored = design_path(default_design)
            if not stored.exists():
                raise UmrissError(
                    "not_found",
                    f"The active design '{default_design}' no longer exists in the project.",
                    context={"known_designs": list_design_ids()},
                    hint="Set a new one with `umriss design use <id>`.",
                )
            args.design = str(stored)
            resolved["design"] = default_design
        else:
            raise UmrissError(
                "missing_design",
                "No design given and no active design is set.",
                context={"known_designs": list_design_ids()},
                hint="Pass --design <file-or-id>" + (" or --preset <name>" if "preset" in vars(args) else "") + ", or set a default with `umriss design use <id>`.",
            )
    return resolved


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    command = canonical_command(raw_argv)
    try:
        parser = build_parser()
        args = parser.parse_args(raw_argv)
        resolved = _resolve_store_ids(args, command)
        run_resolved = _apply_run_defaults(args, command)
        if run_resolved:
            resolved["run"] = run_resolved
        payload = args.func(args)
        if resolved and isinstance(payload.get("data"), dict):
            payload["data"].setdefault("resolved_defaults", resolved)
        print_json(payload)
        return 0
    except UmrissError as exc:
        print_json(
            envelope(
                command,
                "error",
                errors=[{"code": exc.code, "message": exc.message, "context": exc.context, "hint": exc.hint}],
                next_steps=exc.next_steps,
            )
        )
        return 1
    except Exception as exc:
        print_json(
            envelope(
                command,
                "error",
                errors=[
                    {
                        "code": "internal_error",
                        "message": "An unexpected internal error occurred.",
                        "context": {"exception_type": type(exc).__name__},
                        "hint": "Rerun with validated inputs and report this error if it persists.",
                    }
                ],
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
