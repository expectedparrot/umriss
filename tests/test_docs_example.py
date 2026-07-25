from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "trust_tutorial"
RUN = EXAMPLE / "run"
DOCS = ROOT / "docs" / "index.html"


def test_docs_example_uses_captured_ep_run() -> None:
    capture = json.loads((RUN / "ep-run.json").read_text())
    meta = capture["data"]["meta"]
    assert capture["status"] == "ok"
    assert meta["scenario_count"] == 12
    assert meta["result_count"] == 12
    assert meta["local"] is False
    assert "a3e66e84-aced-4eba-b77b-05157cbedb80" in capture["warnings"][0]["output"]

    results = RUN / "prompts" / "trust_coverage_n12.results.ep"
    assert zipfile.is_zipfile(results)

    with (RUN / "raw" / "trust_coverage_n12_raw.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(row["answer.resp"] for row in rows)


def test_docs_metrics_match_checked_in_run() -> None:
    summary = pd.read_csv(RUN / "derived" / "trust_coverage_n12_generated_support_summary.csv")
    scores = dict(zip(summary["method"], summary["mean_rmse"], strict=True))
    html = DOCS.read_text()

    assert f"{scores['generated support mixture']:.4f}" in html
    assert f"{scores['unweighted archetype bank']:.4f}" in html
    assert f"{scores['uniform']:.4f}" in html
    assert "ep run --jobs" in html
    assert "edsl run" not in html


def test_rendered_prompt_browser_view_covers_every_jsonl_record() -> None:
    prompt_path = RUN / "prompts" / "trust_coverage_n12.jsonl"
    prompts = [json.loads(line) for line in prompt_path.read_text().splitlines()]
    rendered = (ROOT / "docs" / "trust_coverage_n12_prompts.html").read_text()

    assert len(prompts) == 12
    for prompt in prompts:
        assert prompt["job_id"] in rendered
        assert prompt["prompt"].splitlines()[0] in rendered
