from __future__ import annotations

import argparse
import json
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
from .ep_commands import export_support_jobs, run_ep_jobs
from .errors import UmrissError
from .evaluation import run_marginal_validation
from .jsonlio import read_json
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
from .parsing import parse_support, register_results
from .plotting import plot_validation
from .state import (
    active_project_id,
    create_project,
    init_workspace,
    list_projects,
    project_dir,
    use_project,
)
from .support_designs import (
    compile_support_plan,
    load_design_config,
    preset_design,
    resolve_design,
    validate_design,
    write_support_outputs,
)
from .twin_export import export_edsl_agents


def envelope(
    command: str,
    status: str,
    data: dict[str, Any] | None = None,
    *,
    warnings: list[dict[str, Any]] | None = None,
    errors: list[dict[str, Any]] | None = None,
    next_steps: list[str] | None = None,
) -> dict[str, Any]:
    return {"command": command, "status": status, "data": data or {}, "warnings": warnings or [], "errors": errors or [], "next_steps": next_steps or []}


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_init(args: argparse.Namespace) -> dict[str, Any]:
    data = init_workspace(args.output_dir or "umriss_work")
    return envelope("umriss init", "ok", data, next_steps=["umriss battery import --metadata <metadata.json>", "umriss design create --metadata <metadata.json> --preset pattern-coverage --out design.yaml"])


def cmd_status(args: argparse.Namespace) -> dict[str, Any]:
    project_id = active_project_id()
    pdir = project_dir(project_id)
    data = {
        "active_project": project_id,
        "project_path": str(pdir),
        "batteries": len(list((pdir / "batteries").glob("*"))) if (pdir / "batteries").exists() else 0,
        "support_prompts": len(list((pdir / "support_prompts").glob("*.jsonl"))) if (pdir / "support_prompts").exists() else 0,
        "support_banks": len(list((pdir / "support_banks").glob("*_probabilities.csv"))) if (pdir / "support_banks").exists() else 0,
        "evaluations": len(list((pdir / "evaluations").glob("*_summary.csv"))) if (pdir / "evaluations").exists() else 0,
    }
    return envelope("umriss status", "ok", data)


def cmd_project_create(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss project create", "ok", create_project(args.project_id, title=args.title, use=args.use))


def cmd_project_use(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss project use", "ok", use_project(args.project_id))


def cmd_project_current(args: argparse.Namespace) -> dict[str, Any]:
    project_id = active_project_id()
    return envelope("umriss project current", "ok", {"active_project": project_id, "project_path": str(project_dir(project_id))})


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
    data = inspect_metadata(Path(args.metadata))
    warnings = data.pop("warnings")
    return envelope("umriss battery inspect", "ok", data, warnings=warnings)


def cmd_battery_import(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss battery import", "ok", import_battery(Path(args.metadata), args.battery_id, args.title))


def cmd_battery_create(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss battery create", "ok", create_battery(args))


def cmd_question_add(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss question add", "ok", add_question(args))


def cmd_marginal_add(args: argparse.Namespace) -> dict[str, Any]:
    return envelope("umriss marginal add", "ok", add_marginal(args))


def cmd_battery_compile(args: argparse.Namespace) -> dict[str, Any]:
    metadata = compile_battery(args.battery, Path(args.path) if args.path else None)
    return envelope("umriss battery compile", "ok", {"battery": args.battery, "items": len(metadata["items"]), "path": args.path})


def cmd_marginals_import(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    if args.respondents:
        truth = weighted_truth_from_respondents(metadata, Path(args.respondents))
    elif args.truth_from == "metadata":
        truth = marginals_from_metadata(metadata)
    else:
        raise UmrissError("invalid_input", "Pass --truth-from metadata or --respondents.")
    write_marginals_long(metadata, truth, Path(args.out))
    return envelope("umriss marginals import", "ok", {"path": args.out, "items": len(truth)})


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
    paths = write_support_outputs(rows, metadata, tag, Path(args.out), resolved)
    return envelope(
        "umriss support build",
        "ok",
        {**paths, "rows": len(rows), "tag": tag, "design": args.design, "preset": args.preset, "metadata_source": metadata_source},
    )


def cmd_support_export(args: argparse.Namespace) -> dict[str, Any]:
    data = export_support_jobs(Path(args.prompts), Path(args.path), model=args.model, service_name=args.service_name, temperature=args.temperature, max_tokens=args.max_tokens, limit=args.limit)
    return envelope("umriss support export", "ok", data, next_steps=data.get("next_steps", []))


def cmd_support_register_results(args: argparse.Namespace) -> dict[str, Any]:
    data = register_results(Path(args.results), Path(args.prompts) if args.prompts else None, args.tag, Path(args.out))
    return envelope("umriss support register-results", "ok", data, next_steps=[f"umriss support parse --raw {data['raw_path']} --metadata <metadata.json> --tag {args.tag}"])


def cmd_support_parse(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = parse_support(Path(args.raw), metadata, args.tag, Path(args.out))
    return envelope("umriss support parse", "ok", data)


def cmd_baseline_build(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = build_baseline_prompts(
        metadata,
        args.tag,
        Path(args.out),
        mode=args.mode,
        respondents_path=Path(args.respondents) if args.respondents else None,
    )
    return envelope("umriss baseline build", "ok", data)


def cmd_baseline_export(args: argparse.Namespace) -> dict[str, Any]:
    data = export_support_jobs(
        Path(args.prompts),
        Path(args.path),
        model=args.model,
        service_name=args.service_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workflow="baseline",
    )
    return envelope("umriss baseline export", "ok", data, next_steps=data.get("next_steps", []))


def cmd_baseline_register_results(args: argparse.Namespace) -> dict[str, Any]:
    data = register_results(Path(args.results), Path(args.prompts), args.tag, Path(args.out))
    return envelope("umriss baseline register-results", "ok", data)


def cmd_baseline_run(args: argparse.Namespace) -> dict[str, Any]:
    data = run_ep_jobs(Path(args.jobs), Path(args.output))
    return envelope("umriss baseline run", "ok", data)


def cmd_baseline_parse(args: argparse.Namespace) -> dict[str, Any]:
    data = parse_baseline_results(
        Path(args.raw),
        Path(args.prompts),
        read_json(Path(args.metadata)),
        args.tag,
        Path(args.out),
    )
    return envelope("umriss baseline parse", "ok", data)


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
    return envelope("umriss support uniformity", status, data)


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
    return envelope("umriss support augment-uniform", "ok", data)


def cmd_support_merge(args: argparse.Namespace) -> dict[str, Any]:
    try:
        data = merge_support_banks(Path(args.base), Path(args.additions), args.tag, Path(args.out))
    except ValueError as exc:
        raise UmrissError("invalid_input", str(exc)) from exc
    return envelope("umriss support merge", "ok", data)


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
    return envelope("umriss fit", "ok", data)


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
        else:
            code = "invalid_input"
        raise UmrissError(code, str(exc)) from exc
    return envelope("umriss validate marginals", "ok", data)


def cmd_predict(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(Path(args.metadata))
    data = predict_from_weights(Path(args.support), Path(args.weights), metadata, args.item or [], Path(args.out))
    return envelope("umriss predict", "ok", data)


def cmd_twins_export_edsl(args: argparse.Namespace) -> dict[str, Any]:
    data = export_edsl_agents(
        [Path(path) for path in args.points],
        Path(args.weights),
        Path(args.path),
        holdout=args.holdout,
        minimum_weight=args.minimum_weight,
    )
    return envelope("umriss twins export-edsl", "ok", data)


def cmd_compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.recipe == "battery-designed":
        data = battery_designed_recipe(Path(args.derived), Path(args.out))
    elif args.recipe == "pattern-coverage":
        data = pattern_coverage_recipe(Path(args.derived), Path(args.out))
    else:
        data = compare_runs(args.run or [], Path(args.derived), Path(args.out), args.comparison_group)
    return envelope("umriss compare", "ok", data)


def cmd_report(args: argparse.Namespace) -> dict[str, Any]:
    data = write_report(args.tag, Path(args.derived), Path(args.out))
    return envelope("umriss report", "ok", data)


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


def cmd_next(args: argparse.Namespace) -> dict[str, Any]:
    if args.tag:
        data = next_for_artifacts(
            args.tag,
            metadata=Path(args.metadata) if args.metadata else None,
            design=Path(args.design) if args.design else None,
            prompt_dir=Path(args.prompt_dir),
            raw_dir=Path(args.raw_dir),
            derived_dir=Path(args.derived_dir),
        )
        return envelope("umriss next", "ok", data)
    try:
        status = cmd_status(args)["data"]
    except UmrissError:
        return envelope("umriss next", "ok", {"recommendation": "Run `umriss init`.", "reason": "No active workspace."})
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="umriss",
        description="Build auditable digital twins from reported survey marginals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_init)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("version").set_defaults(func=lambda args: envelope("umriss version", "ok", {"version": __version__}))

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
    p.add_argument("--metadata", required=True)
    truth_source = p.add_mutually_exclusive_group(required=True)
    truth_source.add_argument("--truth-from", choices=["metadata"])
    truth_source.add_argument("--respondents")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_marginals_import)

    design_cmd = sub.add_parser("design").add_subparsers(dest="design_command", required=True)
    p = design_cmd.add_parser("create")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata")
    source.add_argument("--battery")
    p.add_argument("--preset", choices=["pattern-coverage", "uniform-patterns"], required=True)
    p.add_argument("--size", type=int)
    p.add_argument("--seed", type=int, default=20260625)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_design_create)
    p = design_cmd.add_parser("validate")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata")
    source.add_argument("--battery")
    p.add_argument("--design", required=True)
    p.set_defaults(func=cmd_design_validate)

    support = sub.add_parser("support").add_subparsers(dest="support_command", required=True)
    p = support.add_parser("build")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata", help="Read battery metadata from an explicit JSON file.")
    source.add_argument("--battery", help="Read a battery authored in the active .umriss project.")
    design = p.add_mutually_exclusive_group(required=True)
    design.add_argument("--preset", choices=["pattern-coverage", "uniform-patterns"], help="Compile a safe built-in preset.")
    design.add_argument("--design", help="Use a schema-v1 JSON or YAML support-design file.")
    p.add_argument("--tag", required=True)
    p.add_argument("--n-support", type=int, help="Override design size; feasibility validation still applies.")
    p.add_argument("--seed", type=int, help="Override the design seed.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support_build)
    p = support.add_parser("export")
    p.add_argument("--prompts", required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--model", action="append")
    p.add_argument("--service-name")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2200)
    p.add_argument("--limit", type=int)
    p.set_defaults(func=cmd_support_export)
    p = support.add_parser("register-results")
    p.add_argument("--results", required=True)
    p.add_argument("--prompts")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support_register_results)
    p = support.add_parser("parse")
    p.add_argument("--raw", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support_parse)
    p = support.add_parser("inspect")
    p.add_argument("--prompts")
    p.add_argument("--raw")
    p.add_argument("--bank")
    p.add_argument("--diagnostics")
    p.add_argument("--summary")
    p.set_defaults(func=cmd_support_inspect)
    p = support.add_parser("uniformity")
    p.add_argument("--support", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--tolerance", type=float, default=0.05)
    p.add_argument("--max-duplicate-fraction", type=float, default=0.05)
    p.add_argument("--min-joint-pattern-fraction", type=float, default=0.75)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support_uniformity)
    p = support.add_parser("augment-uniform")
    p.add_argument("--support", required=True)
    p.add_argument("--metadata", required=True)
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
    p.add_argument("--metadata", required=True)
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
    p.set_defaults(func=cmd_baseline_export)
    p = baseline.add_parser("run")
    p.add_argument("--jobs", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_baseline_run)
    p = baseline.add_parser("register-results")
    p.add_argument("--results", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_baseline_register_results)
    p = baseline.add_parser("parse")
    p.add_argument("--raw", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_baseline_parse)

    p = sub.add_parser("fit")
    p.add_argument("--support", required=True)
    p.add_argument("--metadata", required=True)
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
    p.add_argument("--metadata", required=True)
    p.add_argument("--respondents")
    p.add_argument("--one-shot")
    p.add_argument("--conditioned-direct")
    p.add_argument("--rho", nargs="+", type=float, default=[0.0003, 0.001, 0.003, 0.01, 0.03])
    p.add_argument("--uniform-tolerance", type=float, default=0.05)
    p.add_argument("--max-duplicate-fraction", type=float, default=0.05)
    p.add_argument("--min-joint-pattern-fraction", type=float, default=0.75)
    p.add_argument("--allow-nonuniform-support", action="store_true")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_validate_marginals)

    p = sub.add_parser("predict")
    p.add_argument("--support", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--item", action="append")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_predict)

    twins = sub.add_parser("twins").add_subparsers(dest="twins_command", required=True)
    p = twins.add_parser("export-edsl")
    p.add_argument("--points", action="append", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--holdout")
    p.add_argument("--minimum-weight", type=float, default=0.0)
    p.add_argument("--path", required=True)
    p.set_defaults(func=cmd_twins_export_edsl)

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
    p.add_argument("--derived", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["svg", "png", "pdf"], default="svg")
    p.add_argument("--top-personas", type=int, default=30)
    p.set_defaults(func=cmd_plot_validation)

    p = sub.add_parser("guide")
    guide_topics = ["workflow", "designs", "ep-boundary", "paper-rewrite", "diagnostics"]
    p.add_argument("topic", nargs="?", choices=guide_topics)
    p.add_argument("--topic", dest="topic_flag", choices=guide_topics)
    p.set_defaults(func=cmd_guide)
    p = sub.add_parser("next")
    p.add_argument("--tag")
    p.add_argument("--metadata")
    p.add_argument("--design")
    p.add_argument("--prompt-dir", default="data/computed_objects/support_prompts")
    p.add_argument("--raw-dir", default="data/computed_objects/support_raw_responses")
    p.add_argument("--derived-dir", default="data/derived")
    p.set_defaults(func=cmd_next)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = args.func(args)
        print_json(payload)
        return 0
    except UmrissError as exc:
        print_json(
            envelope(
                "umriss",
                "error",
                errors=[{"code": exc.code, "message": exc.message, "context": exc.context, "hint": exc.hint}],
                next_steps=exc.next_steps,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
