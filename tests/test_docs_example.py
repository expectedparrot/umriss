from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "examples" / "pew_w154" / "run"
DOCS = ROOT / "docs" / "index.html"


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
    assert (ROOT / "docs" / "assets" / "loo-summary.svg").exists()
    assert (ROOT / "docs" / "assets" / "loo-by-item.svg").exists()
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
