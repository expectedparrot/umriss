from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "examples" / "pew_w154" / "run"
DOCS = ROOT / "docs" / "index.html"
AUGMENTATION = ROOT / "examples" / "pew_w154" / "augmentation"
AUGMENTATION_DOC = ROOT / "docs" / "pew-augmentation.html"


def test_docs_example_uses_captured_ep_run() -> None:
    capture = json.loads((RUN / "ep-run.json").read_text())
    assert capture["status"] == "ok"
    assert capture["scenario_count"] == 12
    assert capture["result_count"] == 12
    assert capture["results_uuid"] == "c78751ef-75e2-4184-95ff-3d1df6c7ac18"
    assert zipfile.is_zipfile(RUN / "pew_w154_diff1_n12.results.ep")
    with (RUN / "raw" / "pew_w154_diff1_n12_raw.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(row["answer.resp"] for row in rows)


def test_docs_metrics_match_checked_in_run() -> None:
    summary = pd.read_csv(RUN / "derived" / "pew_w154_diff1_uniform_n208_generated_support_summary.csv")
    scores = dict(zip(summary["method"], summary["mean_rmse"], strict=True))
    uniformity = pd.read_csv(RUN / "banks" / "pew_w154_diff1_uniform_n208_preflight.csv")
    html = DOCS.read_text()
    assert f"{scores['generated support mixture']:.5f}" in html
    assert f"{scores['unweighted support bank']:.5f}" in html
    assert f"{scores['uniform']:.5f}" in html
    assert {"mean_kl_divergence", "mean_cross_entropy"} <= set(summary.columns)
    assert (ROOT / "docs" / "assets" / "pew_w154_diff1_uniform_n208_method_comparison.svg").exists()
    assert (ROOT / "docs" / "assets" / "pew_w154_diff1_uniform_n208_holdout_by_item.svg").exists()
    assert "ep run" in html
    assert "edsl run" not in html
    assert uniformity["passes"].all()
    assert uniformity["diversity_passes"].all()
    assert f"{uniformity['max_absolute_deviation'].max() * 100:.2f}" in html


def test_prompt_browser_covers_every_jsonl_record() -> None:
    prompts = [json.loads(line) for line in (RUN / "pew_w154_diff1_n12_prompts.jsonl").read_text().splitlines()]
    rendered = (RUN / "pew_w154_diff1_n12_prompts.html").read_text()
    assert len(prompts) == 12
    for prompt in prompts:
        assert prompt["reason"] in rendered
        assert prompt["prompt"].splitlines()[0] in rendered


def test_pew_augmentation_tutorial_matches_checked_in_artifacts() -> None:
    html = AUGMENTATION_DOC.read_text()
    support = AUGMENTATION / "support_v2"

    consensus = pd.read_csv(
        AUGMENTATION
        / "prior_probe"
        / "pew_work_family_consensus_consensus_audit.csv"
    )
    initial_fidelity = pd.read_csv(
        support
        / "initial"
        / "validation"
        / "pew_work_family_detailed_blueprint_fidelity.csv"
    )
    repair_fidelity = pd.read_csv(
        support
        / "repair_1"
        / "validation"
        / "pew_work_family_detailed_repair1_blueprint_fidelity.csv"
    )
    initial_feasibility = pd.read_csv(
        support
        / "initial"
        / "feasibility"
        / "pew_work_family_detailed_initial_feasibility_summary.csv"
    ).iloc[0]
    final_feasibility = pd.read_csv(
        support
        / "final_bank"
        / "feasibility"
        / "pew_work_family_detailed_288_feasibility_summary.csv"
    ).iloc[0]
    fit = pd.read_csv(
        support
        / "final_bank"
        / "fit"
        / "pew_work_family_detailed_288_fit_diagnostics.csv"
    ).iloc[0]
    constraints = pd.read_csv(
        support
        / "final_bank"
        / "fit"
        / "pew_work_family_detailed_288_constraint_diagnostics.csv"
    )

    assert int(consensus["accepted"].sum()) == 6
    assert initial_fidelity["accepted"].all()
    assert repair_fidelity["accepted"].all()
    assert (
        f"{initial_feasibility['minimum_maximum_absolute_residual'] * 100:.2f}"
        in html
    )
    assert (
        f"{final_feasibility['minimum_maximum_absolute_residual'] * 100:.2f}"
        in html
    )
    assert f"{fit['effective_support']:.2f}" in html
    assert f"{fit['max_weight'] * 100:.2f}%" in html
    assert f"{constraints['max_absolute_residual'].max() * 100:.2f}" in html

    weights = pd.read_csv(
        support
        / "final_bank"
        / "fit"
        / "pew_work_family_detailed_288_weights.csv"
    )
    points = pd.concat(
        [
            pd.read_csv(
                support
                / "initial"
                / "bank"
                / "pew_work_family_detailed_points.csv"
            ),
            pd.read_csv(
                support
                / "repair_1"
                / "bank"
                / "pew_work_family_detailed_repair1_points.csv"
            ),
        ],
        ignore_index=True,
    )
    top = (
        weights.merge(
            points[["job_id", "persona_summary", "persona_details"]],
            on="job_id",
        )
        .sort_values("weight", ascending=False)
        .iloc[0]
    )

    assert top["persona_summary"] in html
    details = json.loads(top["persona_details"])
    assert set(details) == {
        "hobbies",
        "physical_abilities",
        "parenting",
        "feelings",
        "workplace",
        "relationship_status",
        "children_at_home",
        "household_income",
        "marriage_importance",
        "parenthood_importance",
        "paid_work_priority",
        "childcare_division",
        "housework_division",
        "financial_decision_style",
        "career_sacrifice_rule",
    }
    assert all(detail in html for detail in details.values())
    assert "provisional" in html.lower()
