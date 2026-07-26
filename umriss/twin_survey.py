from __future__ import annotations

import ast
import hashlib
import math
import random
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import UmrissError
from .jsonlio import read_json
from .metadata import item_option_codes, item_option_labels
from .parsing import load_results_ep_to_pandas


def export_edsl_survey(
    metadata_path: Path,
    output_path: Path,
    *,
    use_code: bool = False,
    probabilistic_resolution: str | None = None,
    resolution_seed: int | None = None,
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    try:
        with redirect_stdout(StringIO()):
            from edsl import ProbabilisticResponse, QuestionMultipleChoice, Survey
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to export a Survey.") from exc
    if probabilistic_resolution == "sample" and resolution_seed is None:
        raise UmrissError("invalid_input", "--resolution-seed is required when resolution is `sample`.")
    contract = (
        ProbabilisticResponse(
            resolution=probabilistic_resolution,
            seed=resolution_seed,
        )
        if probabilistic_resolution
        else None
    )

    questions = []
    for item, spec in metadata["items"].items():
        stem = str(spec.get("question_stem", "")).strip()
        item_text = str(spec.get("item_text", item)).strip()
        separator = "\n\n" if stem else ""
        questions.append(
            QuestionMultipleChoice(
                question_name=item,
                question_text=f"{stem}{separator}{item_text}",
                question_options=item_option_labels(metadata, item),
                use_code=use_code,
                probabilistic_response=contract,
            )
        )
    survey = Survey(questions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(survey, "git") or not hasattr(Survey, "git"):
        raise UmrissError("edsl_outdated", "This EDSL installation does not support git-backed Survey packages.")
    with redirect_stdout(StringIO()):
        saved = survey.git.save(str(output_path), message="Export umriss source battery")
        survey_path = Path(saved["path"])
        loaded = Survey.git.load(str(survey_path))
    if len(loaded.questions) != len(questions):
        raise UmrissError("invalid_output", "Exported Survey failed round-trip verification.")
    return {
        "survey_path": str(survey_path),
        "items": len(questions),
        "question_names": [question.question_name for question in questions],
        "use_code": use_code,
        "probabilistic_resolution": probabilistic_resolution,
        "resolution_seed": resolution_seed,
        "response_contract": "Each agent gives one ordinary multiple-choice answer per item.",
    }


def build_survey_jobs(
    survey_path: Path,
    agents_path: Path,
    model_name: str,
    output_path: Path,
    service_name: str | None = None,
    temperature: float = 0.5,
    logprobs: bool = False,
    top_logprobs: int = 5,
    limit_agents: int | None = None,
    limit_questions: int | None = None,
) -> dict[str, Any]:
    if not 0 <= temperature <= 2:
        raise UmrissError("invalid_input", "--temperature must be between 0 and 2.")
    if logprobs and not 1 <= top_logprobs <= 20:
        raise UmrissError("invalid_input", "--top-logprobs must be between 1 and 20.")
    if limit_agents is not None and limit_agents < 1:
        raise UmrissError("invalid_input", "--limit-agents must be positive.")
    if limit_questions is not None and limit_questions < 1:
        raise UmrissError("invalid_input", "--limit-questions must be positive.")
    try:
        with redirect_stdout(StringIO()):
            from edsl import AgentList, Jobs, Model, Survey
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to build survey jobs.") from exc
    try:
        with redirect_stdout(StringIO()):
            survey = Survey.git.load(str(survey_path))
            agents = AgentList.git.load(str(agents_path))
            if limit_agents is not None:
                agents = AgentList(list(agents)[:limit_agents])
            if limit_questions is not None:
                survey = Survey(list(survey.questions)[:limit_questions])
            model_parameters: dict[str, Any] = {"temperature": temperature}
            if logprobs:
                model_parameters.update(logprobs=True, top_logprobs=top_logprobs)
            model = Model(model_name, service_name=service_name, **model_parameters)
            jobs = survey.by(agents).by(model)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            saved = jobs.git.save(str(output_path), message="Build umriss ordinary-survey jobs")
            jobs_path = Path(saved["path"])
            Jobs.git.load(str(jobs_path))
    except Exception as exc:
        raise UmrissError(
            "invalid_input",
            "Could not build the ordinary-survey Jobs package.",
            context={"error": str(exc)},
        ) from exc
    return {
        "jobs_path": str(jobs_path),
        "agents": len(agents),
        "questions": len(survey.questions),
        "model": model_name,
        "service_name": service_name,
        "temperature": temperature,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs if logprobs else None,
        "limited": limit_agents is not None or limit_questions is not None,
        "interviews": len(agents),
        "model_answers": len(agents) * len(survey.questions),
    }


def embed_response_probabilities(
    agents_path: Path,
    support_path: Path,
    metadata_path: Path,
    output_path: Path,
    *,
    probability_trait: str,
) -> dict[str, Any]:
    if not probability_trait.isidentifier() or probability_trait.startswith("_"):
        raise UmrissError(
            "invalid_input",
            "--probability-trait must be a visible Python-style trait name.",
        )
    metadata = read_json(metadata_path)
    support = pd.read_csv(support_path)
    required = {"job_id", "item", "option_index", "probability"}
    if not required.issubset(support.columns):
        raise UmrissError("invalid_input", f"Support probabilities must contain: {', '.join(sorted(required))}.")
    try:
        with redirect_stdout(StringIO()):
            from edsl import Agent, AgentList
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to embed response probabilities.") from exc
    with redirect_stdout(StringIO()):
        source_agents = AgentList.git.load(str(agents_path))
    probability_text: dict[str, str] = {}
    for job_id, job_rows in support.groupby(support["job_id"].astype(str), sort=False):
        statements = []
        for item, spec in metadata["items"].items():
            item_rows = job_rows[job_rows["item"].astype(str).eq(str(item))].sort_values("option_index")
            labels = item_option_labels(metadata, item)
            if len(item_rows) != len(labels):
                raise UmrissError("invalid_input", f"Incomplete support probabilities for `{job_id}` / `{item}`.")
            pairs = "; ".join(
                f"“{label}” {float(probability):.1%}"
                for label, probability in zip(labels, item_rows["probability"], strict=True)
            )
            statements.append(f"{spec.get('item_text', item)}: {pairs}")
        probability_text[str(job_id)] = (
            "Your response propensities for this survey battery are listed below. "
            "They express uncertainty rather than a command to choose the most likely answer.\n" + "\n".join(statements)
        )
    agents = []
    for source in source_agents:
        traits = dict(source.traits)
        job_id = str(traits.get("_umriss_job_id", ""))
        if job_id not in probability_text:
            raise UmrissError("invalid_input", f"No support probabilities found for agent job ID `{job_id}`.")
        if probability_trait in traits:
            raise UmrissError("invalid_input", f"Agent already contains the trait `{probability_trait}`.")
        traits[probability_trait] = probability_text[job_id]
        agents.append(Agent(name=source.name, traits=traits))
    embedded = AgentList(agents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(StringIO()):
        saved = embedded.git.save(str(output_path), message="Embed umriss response probabilities")
        saved_path = Path(saved["path"])
        verified = AgentList.git.load(str(saved_path))
    if len(verified) != len(embedded):
        raise UmrissError("invalid_output", "Probability-bearing AgentList failed round-trip verification.")
    return {
        "agents_path": str(saved_path),
        "agents": len(embedded),
        "probability_trait": probability_trait,
        "source_agents": str(agents_path),
        "source_support": str(support_path),
        "contract": (
            "The added visible trait contains each support persona's previously elicited item-level probabilities. "
            "The original persona and hidden fitted weight are unchanged."
        ),
    }


def _stable_uniform(seed: int, job_id: str, item: str) -> float:
    digest = hashlib.sha256(f"{seed}|{job_id}|{item}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).random()


def build_resolution_experiment(
    agents_path: Path,
    support_path: Path,
    metadata_path: Path,
    output_agents_path: Path,
    output_survey_path: Path,
    *,
    resolution_trait: str,
    seed: int,
) -> dict[str, Any]:
    if not resolution_trait.isidentifier() or resolution_trait.startswith("_"):
        raise UmrissError("invalid_input", "--resolution-trait must be a visible Python-style trait name.")
    metadata = read_json(metadata_path)
    support = pd.read_csv(support_path)
    required = {"job_id", "item", "option_index", "probability"}
    if not required.issubset(support.columns):
        raise UmrissError("invalid_input", f"Support probabilities must contain: {', '.join(sorted(required))}.")
    try:
        with redirect_stdout(StringIO()):
            from edsl import Agent, AgentList, QuestionMultipleChoice, Survey
    except ImportError as exc:
        raise UmrissError("edsl_unavailable", "EDSL is required to build a resolution experiment.") from exc
    with redirect_stdout(StringIO()):
        source_agents = AgentList.git.load(str(agents_path))
    agents = []
    for source in source_agents:
        traits = dict(source.traits)
        job_id = str(traits.get("_umriss_job_id", ""))
        resolution: dict[str, dict[str, Any]] = {}
        for item in metadata["items"]:
            rows = support[
                support["job_id"].astype(str).eq(job_id) & support["item"].astype(str).eq(str(item))
            ].sort_values("option_index")
            if rows.empty:
                raise UmrissError("invalid_input", f"No support probabilities found for `{job_id}` / `{item}`.")
            resolution[item] = {
                "probabilities": [float(value) for value in rows["probability"]],
                "draw": _stable_uniform(seed, job_id, item),
            }
        traits[resolution_trait] = resolution
        agents.append(Agent(name=source.name, traits=traits))
    questions = []
    for item, spec in metadata["items"].items():
        probabilities = f"{{{{ agent.{resolution_trait}.{item}.probabilities }}}}"
        draw = f"{{{{ agent.{resolution_trait}.{item}.draw }}}}"
        question_text = (
            f"{str(spec.get('question_stem', '')).strip()}\n\n"
            f"{str(spec.get('item_text', item)).strip()}\n\n"
            "Resolve this response using the supplied probabilities and random draw.\n"
            f"Probabilities in option order: {probabilities}\n"
            f"Random draw: {draw}\n"
            "Starting at 0, form consecutive cumulative-probability intervals in option order. "
            "Choose the zero-based option code whose interval contains the random draw. "
            "This is deterministic: do not choose the most likely option unless the draw falls in its interval."
        )
        questions.append(
            QuestionMultipleChoice(
                question_name=item,
                question_text=question_text,
                question_options=item_option_labels(metadata, item),
                use_code=True,
            )
        )
    embedded_agents = AgentList(agents)
    survey = Survey(questions)
    output_agents_path.parent.mkdir(parents=True, exist_ok=True)
    output_survey_path.parent.mkdir(parents=True, exist_ok=True)
    with redirect_stdout(StringIO()):
        saved_agents = embedded_agents.git.save(
            str(output_agents_path),
            message="Build umriss externally resolved agents",
        )
        saved_survey = survey.git.save(
            str(output_survey_path),
            message="Build umriss externally resolved survey",
        )
        AgentList.git.load(str(saved_agents["path"]))
        Survey.git.load(str(saved_survey["path"]))
    return {
        "agents_path": str(saved_agents["path"]),
        "survey_path": str(saved_survey["path"]),
        "agents": len(embedded_agents),
        "questions": len(questions),
        "resolution_trait": resolution_trait,
        "seed": seed,
        "draw_contract": "Each agent-item draw is stable under SHA-256(seed, job_id, item).",
    }


def analyze_resolution_experiment(
    results_path: Path,
    metadata_path: Path,
    output_dir: Path,
    tag: str,
    *,
    resolution_trait: str,
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    raw = load_results_ep_to_pandas(results_path)
    trait_column = f"agent.{resolution_trait}"
    if trait_column not in raw:
        raise UmrissError("invalid_input", f"Results do not contain `{trait_column}`.")
    rows = []
    for _, record in raw.iterrows():
        resolution = record[trait_column]
        if isinstance(resolution, str):
            resolution = ast.literal_eval(resolution)
        for item in metadata["items"]:
            if f"answer.{item}" not in raw:
                continue
            probabilities = [float(value) for value in resolution[item]["probabilities"]]
            draw = float(resolution[item]["draw"])
            cumulative = 0.0
            expected = len(probabilities) - 1
            for option_index, probability in enumerate(probabilities):
                cumulative += probability
                if draw < cumulative:
                    expected = option_index
                    break
            observed = int(record[f"answer.{item}"])
            rows.append(
                {
                    "job_id": str(record["agent._umriss_job_id"]),
                    "support_id": str(record["agent._umriss_support_id"]),
                    "weight": float(record["agent._weight"]),
                    "item": item,
                    "probabilities": probabilities,
                    "draw": draw,
                    "expected_option_index": expected,
                    "observed_option_index": observed,
                    "correct": observed == expected,
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for item, group in detail.groupby("item", sort=False):
        weights = group["weight"] / group["weight"].sum()
        expected0 = group["expected_option_index"].eq(0).astype(float)
        observed0 = group["observed_option_index"].eq(0).astype(float)
        summary_rows.append(
            {
                "item": item,
                "pew_probability_0": float(metadata["truth"][item][0]),
                "direct_resolution_probability_0": float((weights * expected0).sum()),
                "model_resolution_probability_0": float((weights * observed0).sum()),
                "instruction_accuracy": float(group["correct"].mean()),
                "weighted_instruction_accuracy": float((weights * group["correct"].astype(float)).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"{tag}_resolution_details.csv"
    summary_path = output_dir / f"{tag}_resolution_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "answers": len(detail),
        "correct": int(detail["correct"].sum()),
        "accuracy": float(detail["correct"].mean()),
        "weighted_accuracy": float((detail["weight"] * detail["correct"]).sum() / detail["weight"].sum()),
    }


def analyze_probabilistic_survey(
    results_paths: list[Path],
    metadata_path: Path,
    fit_predictions_path: Path,
    output_dir: Path,
    tag: str,
    *,
    simulations: int = 0,
    simulation_seed: int = 20260725,
) -> dict[str, Any]:
    if simulations < 0:
        raise UmrissError("invalid_input", "--simulations cannot be negative.")
    if simulations == 1:
        raise UmrissError("invalid_input", "--simulations must be zero or at least two.")
    import numpy as np

    metadata = read_json(metadata_path)
    if not results_paths:
        raise UmrissError("invalid_input", "At least one Results package is required.")
    runs = [load_results_ep_to_pandas(path) for path in results_paths]
    fit = pd.read_csv(fit_predictions_path)
    job_column = "agent._umriss_job_id"
    weight_column = "agent._weight"
    for run_index, raw in enumerate(runs, start=1):
        missing = {job_column, weight_column} - set(raw.columns)
        if missing:
            raise UmrissError(
                "invalid_input",
                f"Results run {run_index} lacks required columns: {', '.join(sorted(missing))}.",
            )
    expected_jobs = set(runs[0][job_column].astype(str))
    if any(set(run[job_column].astype(str)) != expected_jobs for run in runs[1:]):
        raise UmrissError("invalid_input", "Results runs do not contain the same AgentList.")

    rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    contract_checks = 0
    contract_correct = 0
    nonmodal_draws = 0
    simulation_rows: list[dict[str, Any]] = []
    simulated_first_option: dict[str, Any] = {}
    rng = np.random.default_rng(simulation_seed)
    for item in metadata["items"]:
        distribution_column = f"distribution.{item}_distribution"
        answer_column = f"answer.{item}"
        draw_column = f"resolution_draw.{item}_resolution_draw"
        for run_index, raw in enumerate(runs, start=1):
            missing_fields = {
                distribution_column,
                answer_column,
                draw_column,
            } - set(raw.columns)
            if missing_fields:
                raise UmrissError(
                    "invalid_input",
                    f"Results run {run_index} lacks probabilistic response fields for `{item}`: "
                    f"{', '.join(sorted(missing_fields))}.",
                )
        merged: dict[str, tuple[Any, Any, float, int, float]] = {}
        for run_index, raw in enumerate(runs, start=1):
            valid = raw[distribution_column].notna() & raw[answer_column].notna()
            coverage_rows.append(
                {
                    "item": item,
                    "run": run_index,
                    "results_path": str(results_paths[run_index - 1]),
                    "valid_responses": int(valid.sum()),
                    "expected_responses": len(expected_jobs),
                    "valid_weight_mass": float(pd.to_numeric(raw.loc[valid, weight_column], errors="raise").sum()),
                }
            )
            for _, record in raw.loc[valid].iterrows():
                job_id = str(record[job_column])
                if job_id not in merged:
                    merged[job_id] = (
                        record[distribution_column],
                        record[answer_column],
                        float(record[weight_column]),
                        run_index,
                        float(record[draw_column]),
                    )
        missing_jobs = expected_jobs - set(merged)
        merged_weight = sum(record[2] for record in merged.values())
        coverage_rows.append(
            {
                "item": item,
                "run": "merged",
                "results_path": "first valid response in supplied run order",
                "valid_responses": len(merged),
                "expected_responses": len(expected_jobs),
                "valid_weight_mass": merged_weight,
            }
        )
        if missing_jobs:
            raise UmrissError(
                "incomplete_results",
                f"No valid probabilistic response for {len(missing_jobs)} persona(s) on `{item}`.",
                context={
                    "item": item,
                    "missing_responses": len(missing_jobs),
                    "valid_weight_mass": merged_weight,
                },
            )
        labels = item_option_labels(metadata, item)
        codes = item_option_codes(metadata, item)
        ordered = list(merged.values())
        parsed_distributions = []
        for distribution, _answer, _weight, _run_index, _draw in ordered:
            parsed_distributions.append(
                ast.literal_eval(distribution) if isinstance(distribution, str) else distribution
            )
        for distribution, answer, _weight, _run_index, draw in merged.values():
            if isinstance(distribution, str):
                distribution = ast.literal_eval(distribution)
            expected = next(
                index
                for index, upper in enumerate(pd.Series(distribution).cumsum())
                if draw <= float(upper) or index == len(distribution) - 1
            )
            observed = next(
                (
                    index
                    for index, (code, label) in enumerate(zip(codes, labels, strict=True))
                    if str(answer) in {str(label), str(code), str(index)}
                ),
                None,
            )
            contract_checks += 1
            contract_correct += observed == expected
            nonmodal_draws += observed != max(range(len(distribution)), key=lambda index: distribution[index])
        if simulations:
            probability_matrix = np.asarray(parsed_distributions, dtype=float)
            weights = np.asarray([record[2] for record in ordered], dtype=float)
            weights = weights / weights.sum()
            draws = rng.random((simulations, len(ordered)))
            cumulative = probability_matrix.cumsum(axis=1)
            cumulative[:, -1] = 1.0
            resolved = (draws[:, :, None] > cumulative[None, :, :]).sum(axis=2)
            for option_index, (code, label) in enumerate(zip(codes, labels, strict=True)):
                marginals = (resolved == option_index) @ weights
                if option_index == 0:
                    simulated_first_option[item] = marginals
                simulation_rows.append(
                    {
                        "item": item,
                        "option_index": option_index,
                        "option_code": code,
                        "option_label": label,
                        "simulations": simulations,
                        "simulation_seed": simulation_seed,
                        "mean": float(marginals.mean()),
                        "standard_deviation": float(marginals.std(ddof=1)),
                        "q025": float(np.quantile(marginals, 0.025)),
                        "median": float(np.quantile(marginals, 0.5)),
                        "q975": float(np.quantile(marginals, 0.975)),
                    }
                )
        for option_index, (code, label) in enumerate(zip(codes, labels, strict=True)):
            weighted_distribution = 0.0
            weighted_answer = 0.0
            total_weight = 0.0
            for distribution, answer, weight, _run_index, _draw in merged.values():
                if isinstance(distribution, str):
                    distribution = ast.literal_eval(distribution)
                if not isinstance(distribution, (list, tuple)) or len(distribution) != len(labels):
                    raise UmrissError("invalid_output", f"Invalid probability vector for `{item}`.")
                if any(
                    not math.isfinite(float(value)) or float(value) < 0 for value in distribution
                ) or not math.isclose(sum(map(float, distribution)), 1.0, abs_tol=1e-6):
                    raise UmrissError("invalid_output", f"Invalid probability vector for `{item}`.")
                weighted_distribution += weight * float(distribution[option_index])
                weighted_answer += weight * float(str(answer) in {str(label), str(code), str(option_index)})
                total_weight += weight
            fit_row = fit[fit["item"].astype(str).eq(str(item)) & fit["option_index"].astype(int).eq(option_index)]
            if len(fit_row) != 1:
                raise UmrissError("invalid_input", f"Fit predictions lack `{item}` option {option_index}.")
            rows.append(
                {
                    "item": item,
                    "option_index": option_index,
                    "option_code": code,
                    "option_label": label,
                    "pew_marginal": float(metadata["truth"][item][option_index]),
                    "umriss_fitted_mixture": float(fit_row.iloc[0]["prediction"]),
                    "meta_probability_mixture": weighted_distribution / total_weight,
                    "meta_resolved_answers": weighted_answer / total_weight,
                }
            )
    comparison = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"{tag}_comparison.csv"
    coverage_path = output_dir / f"{tag}_coverage.csv"
    simulation_path = output_dir / f"{tag}_simulations.csv"
    comparison.to_csv(comparison_path, index=False)
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    if simulations:
        pd.DataFrame(simulation_rows).to_csv(simulation_path, index=False)
    first = comparison[comparison["option_index"].eq(0)].copy()
    simulation_mae = None
    simulation_mae_q025 = None
    simulation_mae_q975 = None
    if simulations:
        truth = {str(row["item"]): float(row["pew_marginal"]) for _, row in first.iterrows()}
        mae_draws = np.mean(
            np.column_stack([np.abs(simulated_first_option[item] - truth[str(item)]) for item in metadata["items"]]),
            axis=1,
        )
        simulation_mae = float(mae_draws.mean())
        simulation_mae_q025 = float(np.quantile(mae_draws, 0.025))
        simulation_mae_q975 = float(np.quantile(mae_draws, 0.975))
    return {
        "comparison_path": str(comparison_path),
        "coverage_path": str(coverage_path),
        "simulation_path": str(simulation_path) if simulations else None,
        "runs": len(runs),
        "personas": len(expected_jobs),
        "persona_item_pairs": len(expected_jobs) * len(metadata["items"]),
        "scheduled_model_calls": len(runs) * len(expected_jobs) * len(metadata["items"]),
        "effective_personas": float(
            1
            / (
                (
                    pd.to_numeric(runs[0][weight_column], errors="raise")
                    / pd.to_numeric(runs[0][weight_column], errors="raise").sum()
                )
                ** 2
            ).sum()
        ),
        "contract_resolution_accuracy": contract_correct / contract_checks,
        "nonmodal_resolutions": nonmodal_draws,
        "simulations": simulations,
        "simulation_seed": simulation_seed if simulations else None,
        "simulation_mean_mae": simulation_mae,
        "simulation_mae_q025": simulation_mae_q025,
        "simulation_mae_q975": simulation_mae_q975,
        "items": len(metadata["items"]),
        "meta_probability_mae": float((first["meta_probability_mixture"] - first["pew_marginal"]).abs().mean()),
        "meta_resolved_mae": float((first["meta_resolved_answers"] - first["pew_marginal"]).abs().mean()),
        "umriss_fit_mae": float((first["umriss_fitted_mixture"] - first["pew_marginal"]).abs().mean()),
    }


def aggregate_survey_frame(
    raw: pd.DataFrame,
    metadata: dict[str, Any],
    fit_predictions: pd.DataFrame,
    *,
    answers_use_code: bool = False,
) -> pd.DataFrame:
    weight_column = next(
        (name for name in ("agent._weight", "_weight", "agent.weight", "weight") if name in raw.columns),
        None,
    )
    if weight_column is None:
        raise UmrissError("invalid_input", "Survey results do not contain the AgentList `_weight` trait.")
    weights = pd.to_numeric(raw[weight_column], errors="coerce")
    if weights.isna().any() or (weights < 0).any() or weights.sum() <= 0:
        raise UmrissError(
            "invalid_input", "Survey result weights must be nonnegative numbers with positive total mass."
        )

    rows: list[dict[str, Any]] = []
    for item in metadata["items"]:
        answer_column = next((name for name in (f"answer.{item}", item) if name in raw.columns), None)
        if answer_column is None:
            raise UmrissError("invalid_input", f"Survey results do not contain an answer for `{item}`.")
        labels = item_option_labels(metadata, item)
        codes = item_option_codes(metadata, item)
        answers = raw[answer_column]
        item_fit = fit_predictions[fit_predictions["item"].astype(str).eq(str(item))]
        for option_index, (code, label) in enumerate(zip(codes, labels, strict=True)):
            accepted = {str(option_index)} if answers_use_code else {str(label), str(code)}
            matches = answers.astype(str).isin(accepted)
            ordinary = float(weights[matches].sum() / weights.sum())
            fit_match = item_fit[item_fit["option_index"].astype(int).eq(option_index)]
            if len(fit_match) != 1:
                raise UmrissError(
                    "invalid_input", f"Fit predictions lack a unique row for `{item}` option {option_index}."
                )
            fitted = float(fit_match.iloc[0]["prediction"])
            truth = float(metadata["truth"][item][option_index])
            rows.append(
                {
                    "item": item,
                    "option_index": option_index,
                    "option_code": code,
                    "option_label": label,
                    "pew_marginal": truth,
                    "probability_mixture": fitted,
                    "ordinary_survey": ordinary,
                    "ordinary_minus_pew": ordinary - truth,
                    "mixture_minus_pew": fitted - truth,
                }
            )
    return pd.DataFrame(rows)


def compare_edsl_survey(
    results_path: Path,
    metadata_path: Path,
    fit_predictions_path: Path,
    output_path: Path,
    *,
    answers_use_code: bool = False,
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    raw = load_results_ep_to_pandas(results_path)
    comparison = aggregate_survey_frame(
        raw,
        metadata,
        pd.read_csv(fit_predictions_path),
        answers_use_code=answers_use_code,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    first_options = comparison[comparison["option_index"].eq(0)]
    ordinary_mae = float(first_options["ordinary_minus_pew"].abs().mean())
    mixture_mae = float(first_options["mixture_minus_pew"].abs().mean())
    return {
        "comparison_path": str(output_path),
        "results_path": str(results_path),
        "responses": len(raw),
        "items": len(metadata["items"]),
        "answers_use_code": answers_use_code,
        "ordinary_survey_mae": ordinary_mae,
        "probability_mixture_mae": mixture_mae,
        "interpretation": (
            "The probability mixture is the fitted expectation. The ordinary-survey series is the weighted "
            "aggregation of one categorical model answer per agent and need not equal that expectation."
        ),
    }


def plot_survey_comparison(
    comparison_path: Path,
    output_path: Path,
    *,
    simulations_path: Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    data = pd.read_csv(comparison_path)
    data = data[data["option_index"].eq(0)].copy()
    if data.empty:
        raise UmrissError("invalid_input", "The comparison file has no option_index=0 rows.")
    series = [
        ("pew_marginal", "Real Pew marginal", "#173f2b"),
        ("umriss_fitted_mixture", "Original fitted mixture", "#d17a22"),
        ("meta_probability_mixture", "New model distributions", "#6e8ea6"),
        ("meta_resolved_answers", "EDSL sampled answers", "#8b5c91"),
    ]
    error_bars: dict[str, tuple[Any, Any]] = {}
    if simulations_path is not None:
        simulations = pd.read_csv(simulations_path)
        simulations = simulations[simulations["option_index"].eq(0)][["item", "mean", "q025", "q975"]]
        data = data.merge(simulations, on="item", how="left", validate="one_to_one")
        if data[["mean", "q025", "q975"]].isna().any().any():
            raise UmrissError("invalid_input", "Simulation summary does not cover every plotted item.")
        series[-1] = ("mean", "Repeated EDSL draws", "#8b5c91")
        error_bars["mean"] = (data["mean"] - data["q025"], data["q975"] - data["mean"])
    if "probability_mixture" in data and "ordinary_survey" in data:
        series = [
            ("pew_marginal", "Real Pew marginal", "#173f2b"),
            ("probability_mixture", "Fitted probability mixture", "#d17a22"),
            ("ordinary_survey", "Weighted ordinary survey", "#6e8ea6"),
        ]
    missing = [column for column, _label, _color in series if column not in data]
    if missing:
        raise UmrissError("invalid_input", f"Comparison file lacks plot columns: {', '.join(missing)}.")
    x = np.arange(len(data))
    width = min(0.24, 0.8 / len(series))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * width
    for offset, (column, label, color) in zip(offsets, series, strict=True):
        yerr = error_bars.get(column)
        ax.bar(
            x + offset,
            data[column],
            width,
            label=label,
            color=color,
            yerr=np.vstack(yerr) if yerr else None,
            capsize=3 if yerr else 0,
        )
    ax.set_ylabel(f"Share answering “{data.iloc[0]['option_label']}”")
    ax.set_xticks(x, [str(item).replace("_", " ") for item in data["item"]])
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncols=min(len(series), 4), loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "plot_path": str(output_path),
        "items": len(data),
        "simulation_intervals": simulations_path is not None,
    }


def analyze_token_probabilities(
    results_path: Path,
    metadata_path: Path,
    support_path: Path,
    output_dir: Path,
    tag: str,
) -> dict[str, Any]:
    metadata = read_json(metadata_path)
    raw = load_results_ep_to_pandas(results_path)
    support = pd.read_csv(support_path)
    required = {"job_id", "item", "option_index", "probability"}
    if not required.issubset(support.columns):
        raise UmrissError("invalid_input", f"Support probabilities must contain: {', '.join(sorted(required))}.")
    rows: list[dict[str, Any]] = []
    for item in metadata["items"]:
        response_column = f"raw_model_response.{item}_raw_model_response"
        answer_column = f"answer.{item}"
        if response_column not in raw or answer_column not in raw:
            raise UmrissError("invalid_input", f"Results do not contain log-probability responses for `{item}`.")
        for _, record in raw.iterrows():
            response = record[response_column]
            if isinstance(response, str):
                response = ast.literal_eval(response)
            try:
                first_token = response["choices"][0]["logprobs"]["content"][0]
            except (KeyError, IndexError, TypeError) as exc:
                raise UmrissError("invalid_input", f"Missing token log probabilities for `{item}`.") from exc
            alternatives = {
                str(candidate["token"]).strip(): math.exp(float(candidate["logprob"]))
                for candidate in first_token["top_logprobs"]
            }
            p0_raw = alternatives.get("0")
            p1_raw = alternatives.get("1")
            if p0_raw is None and p1_raw is None:
                raise UmrissError("invalid_input", f"Neither valid answer code appears for `{item}`.")
            denominator = (p0_raw or 0.0) + (p1_raw or 0.0)
            p0 = (p0_raw or 0.0) / denominator
            sampled = int(record[answer_column])
            job_id = str(record["agent._umriss_job_id"])
            original = support[
                support["job_id"].astype(str).eq(job_id)
                & support["item"].astype(str).eq(str(item))
                & support["option_index"].astype(int).eq(0)
            ]
            if len(original) != 1:
                raise UmrissError(
                    "invalid_input", f"Could not find original support probability for `{job_id}` / `{item}`."
                )
            rows.append(
                {
                    "job_id": job_id,
                    "support_id": str(record["agent._umriss_support_id"]),
                    "weight": float(record["agent._weight"]),
                    "item": item,
                    "sampled_option_index": sampled,
                    "sampled_first_token": str(first_token["token"]).strip(),
                    "sampled_token_probability": math.exp(float(first_token["logprob"])),
                    "code_0_probability_raw": p0_raw,
                    "code_1_probability_raw": p1_raw,
                    "code_0_probability_conditional": p0,
                    "both_codes_observed": p0_raw is not None and p1_raw is not None,
                    "sampled_lower_probability_code": (sampled == 0 and p0 < 0.5) or (sampled == 1 and p0 > 0.5),
                    "original_elicited_probability_0": float(original.iloc[0]["probability"]),
                }
            )
    detail = pd.DataFrame(rows)
    summaries = []
    for item, group in detail.groupby("item", sort=False):
        weights = group["weight"] / group["weight"].sum()
        outcome0 = group["sampled_option_index"].eq(0).astype(float)
        token0 = group["code_0_probability_conditional"]
        elicited0 = group["original_elicited_probability_0"]
        summaries.append(
            {
                "item": item,
                "pew_probability_0": float(metadata["truth"][item][0]),
                "weighted_elicited_probability_0": float((weights * elicited0).sum()),
                "weighted_token_probability_0": float((weights * token0).sum()),
                "weighted_sampled_probability_0": float((weights * outcome0).sum()),
                "both_codes_observed_fraction": float(group["both_codes_observed"].mean()),
                "median_sampled_token_probability": float(group["sampled_token_probability"].median()),
                "weighted_token_brier": float((weights * (token0 - outcome0) ** 2).sum()),
                "lower_probability_draw_fraction": float(group["sampled_lower_probability_code"].mean()),
                "weighted_lower_probability_draw_fraction": float(
                    (weights * group["sampled_lower_probability_code"].astype(float)).sum()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / f"{tag}_token_probabilities.csv"
    summary_path = output_dir / f"{tag}_token_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "answers": len(detail),
        "both_codes_observed": int(detail["both_codes_observed"].sum()),
        "lower_probability_draws": int(detail["sampled_lower_probability_code"].sum()),
        "warning": (
            "When only one valid code appears among top_logprobs, its conditional probability is approximated as 1. "
            "The omitted code is bounded above by the least-probable returned alternative."
        ),
    }
