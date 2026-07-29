"""Black-box output contract: one JSON envelope on stdout, end to end.

Runs the real console entry point in subprocesses (existing tests call main()
in-process), covering stdout purity, error envelopes with the documented exit
statuses, and smoke coverage for previously untested command groups.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from umriss.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
ENVELOPE_KEYS = {"schema_version", "command", "status", "argv", "data", "warnings", "errors", "next_steps"}


def run_umriss(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "umriss", *args],
        cwd=cwd, text=True, capture_output=True,
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
    )


@pytest.mark.parametrize("argv", [
    ("version",),
    ("capabilities",),
    ("guide",),
    ("next",),
])
def test_stdout_is_exactly_one_envelope(argv: tuple[str, ...], tmp_path: Path) -> None:
    completed = run_umriss(*argv, cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)  # fails if anything else is on stdout
    assert ENVELOPE_KEYS <= set(payload)
    assert payload["status"] == "ok"
    assert payload["argv"][0] == "umriss"


def test_expected_failures_exit_1_with_error_envelope(tmp_path: Path) -> None:
    completed = run_umriss("battery", "inspect", "missing.json", cwd=tmp_path)
    assert completed.returncode == 1, completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["status"] == "error"
    assert payload["errors"][0]["code"] == "not_found"


def test_project_group_smoke(tmp_path: Path) -> None:
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    created = run_umriss("project", "create", "demo", "--title", "Demo", "--use", cwd=tmp_path)
    assert created.returncode == 0
    current = run_umriss("project", "current", cwd=tmp_path)
    assert json.loads(current.stdout)["data"]["active_project"] == "demo"
    listed = run_umriss("project", "list", cwd=tmp_path)
    assert any(p["project_id"] == "demo" for p in json.loads(listed.stdout)["data"]["projects"])


def test_marginal_group_smoke(tmp_path: Path) -> None:
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    steps = [
        ("battery", "create", "--battery-id", "demo", "--wave", "T1",
         "--battery", "DEMO", "--topic", "demo", "--context", "demo"),
        ("question", "add", "--battery", "demo", "--item", "q1",
         "--question-stem", "Pick one", "--item-text", "Item one",
         "--option", "Yes", "--option", "No",
         "--option-code", "1", "--option-code", "2",
         "--scale-type", "nominal"),
        ("marginal", "add", "--battery", "demo", "--item", "q1",
         "--proportion", "0.6", "--proportion", "0.4"),
        ("battery", "compile", "--battery", "demo", "--path", "demo_metadata.json"),
    ]
    for step in steps:
        completed = run_umriss(*step, cwd=tmp_path)
        assert completed.returncode == 0, (step, completed.stdout)
        payload = json.loads(completed.stdout)
        assert payload["status"] == "ok"
        assert payload["next_steps"], f"{step[0]} {step[1]} should suggest the next pipeline step"
    assert (tmp_path / "demo_metadata.json").exists()


def test_next_tag_discovers_run_directory(tmp_path: Path) -> None:
    # No manifest anywhere: next --tag falls back to cwd and recommends the
    # first pipeline step rather than pointing at a hard-coded paper layout.
    completed = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout
    data = json.loads(completed.stdout)["data"]
    assert data["stage"] == "build-prompts"
    assert "data/computed_objects" not in json.dumps(data)

    nested = tmp_path / "runs" / "demo_run"
    nested.mkdir(parents=True)
    (nested / "demo_manifest.json").write_text(json.dumps({
        "tag": "demo",
        "prompts": str(nested / "demo_prompts.jsonl"),
        "jobs": str(nested / "demo.jobs.ep"),
        "results": str(nested / "demo.results.ep"),
    }))
    (nested / "demo_prompts.jsonl").write_text("{}\n")
    completed = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout
    data = json.loads(completed.stdout)["data"]
    assert data["stage"] == "export-jobs"


def test_ambiguous_tag_manifests_fail_closed(tmp_path: Path) -> None:
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "demo_manifest.json").write_text(json.dumps({"tag": "demo"}))
    completed = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["errors"][0]["code"] == "ambiguous_tag"
    assert len(payload["errors"][0]["context"]["matches"]) == 2


def test_store_ids_resolve_for_metadata_and_design(tmp_path: Path) -> None:
    import sys as _sys

    _sys.path.insert(0, str(REPO / "tests"))
    from test_umriss_cli import mini_metadata

    from umriss.jsonlio import write_json

    metadata_path = tmp_path / "mini_metadata.json"
    write_json(metadata_path, mini_metadata())

    assert run_umriss("init", cwd=tmp_path).returncode == 0
    imported = run_umriss("battery", "import", "--metadata", str(metadata_path),
                          "--battery-id", "mini", cwd=tmp_path)
    assert imported.returncode == 0, imported.stdout
    listed = json.loads(run_umriss("battery", "list", cwd=tmp_path).stdout)
    assert listed["data"]["batteries"] == ["mini"]

    # --metadata accepts the battery id everywhere, not just a path
    created = run_umriss("design", "create", "--metadata", "mini",
                         "--preset", "pattern-coverage", "--out", str(tmp_path / "d.yaml"), cwd=tmp_path)
    assert created.returncode == 0, created.stdout

    assert run_umriss("design", "import", "--design", str(tmp_path / "d.yaml"),
                      "--design-id", "v1", cwd=tmp_path).returncode == 0
    designs = json.loads(run_umriss("design", "list", cwd=tmp_path).stdout)
    assert designs["data"]["designs"] == ["v1"]

    # both ids resolve together
    validated = run_umriss("design", "validate", "--metadata", "mini", "--design", "v1", cwd=tmp_path)
    assert validated.returncode == 0, validated.stdout
    assert json.loads(validated.stdout)["data"]["valid"] is True

    # a second import must not clobber the registry
    write_json(tmp_path / "mini2.json", mini_metadata())
    assert run_umriss("battery", "import", "--metadata", str(tmp_path / "mini2.json"),
                      "--battery-id", "mini2", cwd=tmp_path).returncode == 0
    listed = json.loads(run_umriss("battery", "list", cwd=tmp_path).stdout)
    assert listed["data"]["batteries"] == ["mini", "mini2"]

    # unknown bare ids fail closed with the known ids, not file-not-found
    unknown = run_umriss("design", "validate", "--metadata", "nope", "--design", "v1", cwd=tmp_path)
    assert unknown.returncode == 1
    payload = json.loads(unknown.stdout)
    assert payload["errors"][0]["context"]["known_batteries"] == ["mini", "mini2"]

    # explicit paths still behave exactly as before
    validated = run_umriss("design", "validate", "--metadata", str(metadata_path),
                           "--design", str(tmp_path / "d.yaml"), cwd=tmp_path)
    assert validated.returncode == 0, validated.stdout

    # without defaults set, omitting the flags fails closed with guidance
    missing = run_umriss("design", "validate", cwd=tmp_path)
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["errors"][0]["code"] == "missing_battery"

    # `use` sets active defaults; flags become optional and the envelope
    # echoes what was implicitly resolved
    assert run_umriss("battery", "use", "mini", cwd=tmp_path).returncode == 0
    assert run_umriss("design", "use", "v1", cwd=tmp_path).returncode == 0
    bare = run_umriss("design", "validate", cwd=tmp_path)
    assert bare.returncode == 0, bare.stdout
    payload = json.loads(bare.stdout)
    assert payload["data"]["valid"] is True
    assert payload["data"]["resolved_defaults"] == {"battery": "mini", "design": "v1"}

    status = json.loads(run_umriss("status", cwd=tmp_path).stdout)
    assert status["data"]["active_battery"] == "mini"
    assert status["data"]["active_design"] == "v1"

    # explicit flags still override the defaults
    explicit = run_umriss("design", "validate", "--metadata", "mini2",
                          "--design", str(tmp_path / "d.yaml"), cwd=tmp_path)
    assert explicit.returncode == 0, explicit.stdout
    assert "resolved_defaults" not in json.loads(explicit.stdout)["data"]

    # `use` of something not imported fails closed
    bad = run_umriss("battery", "use", "ghost", cwd=tmp_path)
    assert bad.returncode == 1
    assert json.loads(bad.stdout)["errors"][0]["context"]["known_batteries"] == ["mini", "mini2"]


def test_run_store_pipeline_needs_no_path_flags(tmp_path: Path) -> None:
    import sys as _sys

    _sys.path.insert(0, str(REPO / "tests"))
    from test_umriss_cli import mini_metadata, write_raw

    from umriss.jsonlio import write_json

    metadata_path = tmp_path / "mini_metadata.json"
    write_json(metadata_path, mini_metadata())
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    assert run_umriss("battery", "import", "--metadata", str(metadata_path),
                      "--battery-id", "mini", cwd=tmp_path).returncode == 0
    assert run_umriss("battery", "use", "mini", cwd=tmp_path).returncode == 0

    # build: no --out; prompts land in the store run dir and the envelope says so
    built = run_umriss("support", "build", "--preset", "pattern-coverage",
                       "--tag", "demo", cwd=tmp_path)
    assert built.returncode == 0, built.stdout
    payload = json.loads(built.stdout)
    store_run = tmp_path / ".umriss" / "projects" / "default" / "runs" / "demo"
    assert (store_run / "demo_prompts.jsonl").exists()
    assert payload["data"]["resolved_defaults"]["run"]["out"] == ".umriss/projects/default/runs/demo"

    # parse: raw dropped into the store by convention; no --raw/--out needed
    write_raw(store_run / "demo_raw.csv")
    parsed = run_umriss("support", "parse", "--tag", "demo", cwd=tmp_path)
    assert parsed.returncode == 0, parsed.stdout
    assert (store_run / "demo_probabilities.csv").exists()

    # status knows the store run and its stage; next --tag needs no dirs
    status = json.loads(run_umriss("status", cwd=tmp_path).stdout)
    store_runs = [r for r in status["data"]["runs"] if r["location"] == "store"]
    assert store_runs and store_runs[0]["tag"] == "demo"
    nxt = run_umriss("next", "--tag", "demo", cwd=tmp_path)
    assert nxt.returncode == 0, nxt.stdout
    assert json.loads(nxt.stdout)["data"]["stage"]

    # a missing stage input names its producer instead of file-not-found
    missing = run_umriss("support", "parse", "--tag", "fresh", cwd=tmp_path)
    assert missing.returncode == 1
    assert "register-results" in json.loads(missing.stdout)["errors"][0]["hint"]

    # export publishes the run for replication packages
    exported = run_umriss("export", "--tag", "demo", "--out", str(tmp_path / "pkg"), cwd=tmp_path)
    assert exported.returncode == 0, exported.stdout
    names = json.loads(exported.stdout)["data"]["files"]
    assert "demo_prompts.jsonl" in names and "demo_probabilities.csv" in names
    assert (tmp_path / "pkg" / "demo_probabilities.csv").exists()

    # explicit --out still wins
    explicit = run_umriss("support", "build", "--preset", "pattern-coverage",
                          "--tag", "demo2", "--out", str(tmp_path / "elsewhere"), cwd=tmp_path)
    assert explicit.returncode == 0, explicit.stdout
    assert (tmp_path / "elsewhere" / "demo2_prompts.jsonl").exists()


def _teen_social_metadata(survey_key: str, truth: list[float]) -> dict:
    """One 3-option ordinal item, marginals per subpopulation — the shape of a
    typical published crosstab (unlike the multi-item binary Pew battery)."""
    return {
        "schema_version": 1,
        "wave": "ATS2026",
        "battery": "SOCIALTIME",
        "survey_key": survey_key,
        "topic": "daily time on social media among US teen girls",
        "context": "A survey of US teens; marginals reported by parental education.",
        "option_codes": [1, 2, 3],
        "option_labels": ["Less than one hour", "One to three hours", "Four or more hours"],
        "scale": {"type": "ordinal", "direction": "low_to_high"},
        "items": {"social_media_hours": {
            "variable": "SOCIALTIME_girls",
            "item_text": "About how much time do you spend on social media each day?",
            "question_stem": "Thinking about a typical day, about how much time do you spend on social media?",
        }},
        "truth": {"social_media_hours": truth},
    }


def test_single_item_ordinal_battery_two_populations(tmp_path: Path) -> None:
    from umriss.jsonlio import write_json

    write_json(tmp_path / "college.json", _teen_social_metadata("teen_college", [0.40, 0.37, 0.23]))
    write_json(tmp_path / "noncollege.json", _teen_social_metadata("teen_noncollege", [0.35, 0.29, 0.35]))
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    for battery_id, path in [("girls_college", "college.json"), ("girls_noncollege", "noncollege.json")]:
        assert run_umriss("battery", "import", "--metadata", str(tmp_path / path),
                          "--battery-id", battery_id, cwd=tmp_path).returncode == 0
    assert run_umriss("battery", "use", "girls_college", cwd=tmp_path).returncode == 0

    # inspect reports the workspace id it resolved, not a derived slug
    inspected = json.loads(run_umriss("battery", "inspect", "girls_college", cwd=tmp_path).stdout)
    assert inspected["data"]["battery_id"] == "girls_college"

    # ordinal preset anchors sit at the scale extremes
    created = run_umriss("design", "create", "--preset", "pattern-coverage",
                         "--out", str(tmp_path / "d.yaml"), cwd=tmp_path)
    assert created.returncode == 0, created.stdout
    design_text = (tmp_path / "d.yaml").read_text()
    assert "Less than one hour" in design_text and "Four or more hours" in design_text
    assert design_text.count("One to three hours") == 0  # no middle anchor

    assert run_umriss("design", "import", "--design", str(tmp_path / "d.yaml"),
                      "--design-id", "v1", cwd=tmp_path).returncode == 0
    assert run_umriss("design", "use", "v1", cwd=tmp_path).returncode == 0
    built = run_umriss("support", "build", "--tag", "teen", cwd=tmp_path)
    assert built.returncode == 0, built.stdout
    prompts_path = tmp_path / ".umriss" / "projects" / "default" / "runs" / "teen" / "teen_prompts.jsonl"
    prompts = prompts_path.read_text()
    assert "the other items" not in prompts  # single-item battery gets coherent prompt text
    assert "40" not in prompts.replace("N=782", "")  # no target leakage

    # one SYNTHETIC bank (mechanics only), two population fits over it
    bank = tmp_path / "SYNTHETIC_bank.csv"
    rows = ["support_id,job_id,item,option_index,option_code,option_label,probability"]
    vectors = {"s1": [0.70, 0.22, 0.08], "s2": [0.25, 0.55, 0.20], "s3": [0.08, 0.27, 0.65]}
    labels = ["Less than one hour", "One to three hours", "Four or more hours"]
    for sid, vec in vectors.items():
        for i, prob in enumerate(vec):
            rows.append(f"{sid},{sid},social_media_hours,{i},{i + 1},{labels[i]},{prob}")
    bank.write_text("\n".join(rows) + "\n")

    weights = {}
    for battery_id in ["girls_college", "girls_noncollege"]:
        fitted = run_umriss("fit", "--support", str(bank), "--metadata", battery_id,
                            "--tag", f"fit_{battery_id}", "--out", str(tmp_path / battery_id), cwd=tmp_path)
        assert fitted.returncode == 0, fitted.stdout
        import csv as _csv

        with open(json.loads(fitted.stdout)["data"]["weights_path"]) as f:
            weights[battery_id] = {r["support_id"]: float(r["weight"]) for r in _csv.DictReader(f)}
    # the heavy-use persona must gain weight under the non-college marginals
    assert weights["girls_noncollege"]["s3"] > weights["girls_college"]["s3"]

    # LOO on a one-item battery fails with a domain explanation, not a shrug
    loo = run_umriss("validate", "marginals", "--support", str(bank), "--metadata", "girls_college",
                     "--tag", "probe", "--out", str(tmp_path / "loo"),
                     "--allow-nonuniform-support", cwd=tmp_path)
    assert loo.returncode == 1, loo.stdout + loo.stderr
    error = json.loads(loo.stdout)["errors"][0]
    assert error["code"] == "battery_too_small"
    assert "umriss fit" in error["message"]


def test_prior_parse_is_model_complete_and_consensus_is_provenanced(tmp_path: Path) -> None:
    import csv

    from umriss.jsonlio import write_json

    metadata = _teen_social_metadata("probe", [0.4, 0.37, 0.23])
    metadata_path = tmp_path / "metadata.json"
    write_json(metadata_path, metadata)
    built = run_umriss(
        "prior", "build-marginals", "--metadata", str(metadata_path),
        "--tag", "probe", "--out", str(tmp_path), cwd=tmp_path,
    )
    assert built.returncode == 0, built.stdout
    prompts = tmp_path / "probe_baseline_prompts.jsonl"
    job_id = json.loads(prompts.read_text().strip())["job_id"]
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario.job_id", "model.model", "model.inference_service", "answer.resp"],
        )
        writer.writeheader()
        writer.writerow({
            "scenario.job_id": job_id, "model.model": "m1", "model.inference_service": "s1",
            "answer.resp": json.dumps({"probabilities": [0.2, 0.4, 0.4]}),
        })
        writer.writerow({
            "scenario.job_id": job_id, "model.model": "m2", "model.inference_service": "s2",
            "answer.resp": "",
        })
    incomplete = run_umriss(
        "prior", "parse", "--raw", str(raw), "--prompts", str(prompts),
        "--metadata", str(metadata_path), "--tag", "probe", "--out", str(tmp_path / "parsed"), cwd=tmp_path,
    )
    assert incomplete.returncode == 1, incomplete.stdout
    error = json.loads(incomplete.stdout)["errors"][0]
    assert error["code"] == "incomplete_results"
    assert error["context"]["missing_model_jobs"][0]["model"] == "m2"
    assert (tmp_path / "parsed" / "probe_prior_predictions.csv").exists()

    with raw.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario.job_id", "model.model", "model.inference_service", "answer.resp"],
        )
        writer.writerow({
            "scenario.job_id": job_id, "model.model": "m3", "model.inference_service": "s3",
            "answer.resp": json.dumps({"probabilities": [0.21, 0.39, 0.40]}),
        })
    # Replace the invalid m2 row rather than silently accepting it.
    rows = list(csv.DictReader(raw.open()))
    rows[1]["answer.resp"] = json.dumps({"probabilities": [0.19, 0.41, 0.40]})
    with raw.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    parsed = run_umriss(
        "prior", "parse", "--raw", str(raw), "--prompts", str(prompts),
        "--metadata", str(metadata_path), "--tag", "complete", "--out", str(tmp_path / "complete"), cwd=tmp_path,
    )
    assert parsed.returncode == 0, parsed.stdout
    predictions = tmp_path / "complete" / "complete_prior_predictions.csv"
    frame = __import__("pandas").read_csv(predictions)
    assert set(frame["model"]) == {"m1", "m2", "m3"}
    consensus = run_umriss(
        "prior", "consensus", "--predictions", str(predictions), "--metadata", str(metadata_path),
        "--population", "teen_girls", "--minimum-models", "3", "--tag", "consensus",
        "--out", str(tmp_path / "consensus"), cwd=tmp_path,
    )
    assert consensus.returncode == 0, consensus.stdout
    targets = json.loads((tmp_path / "consensus" / "consensus_targets.json").read_text())
    assert targets["targets"][0]["status"] == "accepted"
    assert len(targets["targets"][0]["source"]["models"]) == 3


def test_joint_targets_require_consistency_and_explicit_feature_method(tmp_path: Path) -> None:
    import csv

    from umriss.jsonlio import write_json

    metadata = {
        "schema_version": 1, "wave": "T1", "battery": "B", "topic": "t", "context": "c",
        "items": {
            "a": {"item_text": "A", "question_stem": "A?", "option_labels": ["No", "Yes"],
                  "option_codes": [0, 1], "scale": {"type": "nominal"}},
            "b": {"item_text": "B", "question_stem": "B?", "option_labels": ["No", "Yes"],
                  "option_codes": [0, 1], "scale": {"type": "nominal"}},
        },
    }
    metadata_path = tmp_path / "metadata.json"
    write_json(metadata_path, metadata)
    artifact = {
        "schema_version": 1, "kind": "umriss_targets", "population": {"id": "p"},
        "targets": [
            {"target_id": "marginal:a", "type": "marginal", "items": ["a"], "shape": [2],
             "values": [0.5, 0.5], "status": "accepted", "source": {"kind": "observed"}},
            {"target_id": "marginal:b", "type": "marginal", "items": ["b"], "shape": [2],
             "values": [0.5, 0.5], "status": "accepted", "source": {"kind": "model_synthetic"}},
            {"target_id": "joint:a:b", "type": "joint", "items": ["a", "b"], "shape": [2, 2],
             "values": [0.4, 0.1, 0.1, 0.4], "status": "accepted",
             "source": {"kind": "model_synthetic"}},
        ],
    }
    targets_path = tmp_path / "targets.json"
    write_json(targets_path, artifact)
    audited = run_umriss(
        "targets", "audit", "--targets", str(targets_path), "--metadata", str(metadata_path),
        "--out", str(tmp_path / "audit.csv"), cwd=tmp_path,
    )
    assert audited.returncode == 0, audited.stdout

    support = tmp_path / "support.csv"
    with support.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["support_id", "job_id", "item", "option_index", "option_code", "option_label", "probability"],
        )
        writer.writeheader()
        vectors = {"s1": {"a": [0.8, 0.2], "b": [0.8, 0.2]}, "s2": {"a": [0.2, 0.8], "b": [0.2, 0.8]}}
        for sid, items in vectors.items():
            for item, vector in items.items():
                for index, value in enumerate(vector):
                    writer.writerow({
                        "support_id": sid, "job_id": sid, "item": item, "option_index": index,
                        "option_code": index, "option_label": ["No", "Yes"][index], "probability": value,
                    })
    blocked = run_umriss(
        "targets", "fit", "--targets", str(targets_path), "--support", str(support),
        "--metadata", str(metadata_path), "--tag", "fit", "--out", str(tmp_path / "fit"), cwd=tmp_path,
    )
    assert blocked.returncode == 1
    assert json.loads(blocked.stdout)["errors"][0]["code"] == "joint_features_missing"
    fitted = run_umriss(
        "targets", "fit", "--targets", str(targets_path), "--support", str(support),
        "--metadata", str(metadata_path), "--allow-conditional-independence",
        "--tag", "fit", "--out", str(tmp_path / "fit"), cwd=tmp_path,
    )
    assert fitted.returncode == 0, fitted.stdout
    diagnostics = __import__("pandas").read_csv(tmp_path / "fit" / "fit_constraint_diagnostics.csv")
    assert "conditional_independence" in set(diagnostics["feature_method"])

    artifact["targets"][2]["values"] = [0.7, 0.1, 0.1, 0.1]
    write_json(targets_path, artifact)
    inconsistent = run_umriss(
        "targets", "audit", "--targets", str(targets_path), "--metadata", str(metadata_path),
        "--consistency-tolerance", "0.05", "--out", str(tmp_path / "bad_audit.csv"), cwd=tmp_path,
    )
    assert inconsistent.returncode == 1
    assert "marginal_inconsistency" in inconsistent.stdout


def test_support_extension_preserves_persona_ids_and_adds_direct_joint_features(tmp_path: Path) -> None:
    import csv

    from umriss.jsonlio import write_json

    metadata = {
        "schema_version": 1, "wave": "T", "battery": "B", "topic": "t", "context": "c",
        "items": {
            item: {"item_text": item, "question_stem": f"{item}?", "option_labels": ["No", "Yes"],
                   "option_codes": [0, 1], "scale": {"type": "nominal"}}
            for item in ("old", "new")
        },
    }
    metadata_path = tmp_path / "metadata.json"
    write_json(metadata_path, metadata)
    points = tmp_path / "points.csv"
    points.write_text("support_id,job_id,persona\ns1,j1,Your views are cautious.\n")
    built = run_umriss(
        "support", "extend-items", "--points", str(points), "--metadata", str(metadata_path),
        "--item", "new", "--joint", "old:new", "--tag", "ext", "--out", str(tmp_path), cwd=tmp_path,
    )
    assert built.returncode == 0, built.stdout
    prompts_path = tmp_path / "ext_extension_prompts.jsonl"
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines()]
    raw = tmp_path / "raw.csv"
    with raw.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario.job_id", "answer.resp"])
        writer.writeheader()
        for prompt in prompts:
            response = (
                {"probabilities": [0.3, 0.7]}
                if prompt["target_type"] == "marginal"
                else {"joint_probabilities": [[0.2, 0.3], [0.1, 0.4]]}
            )
            writer.writerow({"scenario.job_id": prompt["job_id"], "answer.resp": json.dumps(response)})
    base = tmp_path / "base.csv"
    base.write_text(
        "support_id,job_id,item,option_index,option_code,option_label,probability\n"
        "s1,j1,old,0,0,No,0.6\ns1,j1,old,1,1,Yes,0.4\n"
    )
    parsed = run_umriss(
        "support", "parse-extension", "--raw", str(raw), "--prompts", str(prompts_path),
        "--base-support", str(base), "--metadata", str(metadata_path),
        "--tag", "extended", "--out", str(tmp_path / "extended"), cwd=tmp_path,
    )
    assert parsed.returncode == 0, parsed.stdout
    probabilities = __import__("pandas").read_csv(tmp_path / "extended" / "extended_probabilities.csv")
    assert set(probabilities["item"]) == {"old", "new"}
    joints = __import__("pandas").read_csv(tmp_path / "extended" / "extended_joint_features.csv")
    assert set(joints["feature_method"]) == {"direct_joint"}
    assert set(joints["support_id"]) == {"s1"}


def test_every_leaf_command_has_handler() -> None:
    problems: list[str] = []

    def walk(node: argparse.ArgumentParser, prefix: list[str]) -> None:
        for action in node._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    subs = [a for a in sub._actions if isinstance(a, argparse._SubParsersAction)]
                    if subs:
                        walk(sub, prefix + [name])
                    elif sub.get_default("func") is None:
                        problems.append(" ".join(prefix + [name]))

    walk(build_parser(), [])
    assert problems == [], "commands without handlers:\n" + "\n".join(problems)


def test_status_and_next_report_discovered_runs(tmp_path: Path) -> None:
    assert run_umriss("init", cwd=tmp_path).returncode == 0
    nested = tmp_path / "out" / "run1"
    nested.mkdir(parents=True)
    (nested / "demo_manifest.json").write_text(json.dumps({
        "tag": "demo",
        "workflow": "support",
        "prompts": str(nested / "demo_prompts.jsonl"),
        "jobs": str(nested / "demo.jobs.ep"),
        "results": str(nested / "demo.results.ep"),
    }))
    (nested / "demo_prompts.jsonl").write_text("{}\n")

    status = json.loads(run_umriss("status", cwd=tmp_path).stdout)
    runs = status["data"]["runs"]
    assert len(runs) == 1
    assert runs[0]["tag"] == "demo"
    assert runs[0]["stage"] == "export-jobs"
    assert status["next_steps"] == ["umriss next --tag demo"]

    next_payload = json.loads(run_umriss("next", cwd=tmp_path).stdout)
    assert next_payload["data"]["recommendation"] == "umriss next --tag demo"
    assert "export-jobs" in next_payload["data"]["reason"]


def test_export_manifest_reports_cost_basis(tmp_path: Path) -> None:
    import os

    from umriss.cli import main as cli_main
    from umriss.jsonlio import write_json

    os.environ["EDSL_LOG_DIR"] = str(tmp_path / "edsl_logs")
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        import contextlib
        import io

        sys.path.insert(0, str(REPO / "tests"))
        from test_umriss_cli import mini_metadata

        metadata_path = tmp_path / "metadata.json"
        write_json(metadata_path, mini_metadata())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert cli_main(["support", "build", "--metadata", str(metadata_path),
                             "--preset", "pattern-coverage", "--tag", "mini",
                             "--out", str(tmp_path / "prompts")]) == 0
            assert cli_main(["support", "export",
                             "--prompts", str(tmp_path / "prompts" / "mini_prompts.jsonl"),
                             "--path", str(tmp_path / "prompts" / "mini.jobs.ep"),
                             "--model", "test", "--limit", "1"]) == 0
        manifest = json.loads((tmp_path / "prompts" / "mini_manifest.json").read_text())
        estimate = manifest["execution"]["cost_estimate"]
        assert estimate["available"] is True
        assert estimate["basis"] == "call_counts"
        assert estimate["expected_model_calls"] == estimate["scenarios"] * estimate["models"]
        assert "pricing_note" in estimate
    finally:
        os.chdir(cwd)
