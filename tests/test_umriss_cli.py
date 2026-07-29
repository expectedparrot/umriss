from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from umriss.artifacts import next_for_artifacts
from umriss.cli import build_parser, main
from umriss.errors import UmrissError
from umriss.parsing import audit_result_attempts, register_results
from umriss.support_designs import geometry_repair_blueprint_design
from umriss.twin_survey import aggregate_survey_frame, plot_survey_comparison


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def mini_metadata() -> dict:
    return {
        "schema_version": 1,
        "wave": "T1",
        "battery": "TEST",
        "topic": "test topic",
        "context": "A small test survey.",
        "items": {
            "a": {"variable": "A", "item_text": "Item A", "question_stem": "How important is this?", "scale": {"type": "nominal"}},
            "b": {"variable": "B", "item_text": "Item B", "question_stem": "How important is this?", "scale": {"type": "nominal"}},
        },
        "option_codes": [1, 2],
        "option_labels": ["Yes", "No"],
        "truth": {"a": [0.7, 0.3], "b": [0.4, 0.6]},
    }


def write_raw(path: Path) -> None:
    rows = [
        {
            "scenario.job_id": "s1",
            "answer.resp": json.dumps(
                {
                    "persona": "Your views generally favor yes.",
                    "persona_details": {
                        "a": "You tend to answer yes on Item A.",
                        "b": "You tend to answer no on Item B.",
                    },
                    "probabilities": {"a": [0.9, 0.1], "b": [0.2, 0.8]},
                }
            ),
        },
        {
            "scenario.job_id": "s2",
            "answer.resp": json.dumps(
                {
                    "persona": "Your views generally favor no.",
                    "persona_details": {
                        "a": "You tend to answer no on Item A.",
                        "b": "You tend to answer yes on Item B.",
                    },
                    "probabilities": {"a": [0.2, 0.8], "b": [0.8, 0.2]},
                }
            ),
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario.job_id", "answer.resp"])
        writer.writeheader()
        writer.writerows(rows)


class UmrissCliTests(unittest.TestCase):
    def test_geometry_repair_targets_minimax_separating_direction(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            support_path = root / "support.csv"
            rows = []
            vectors = {
                1: {"a": [0.9, 0.1], "b": [0.2, 0.8]},
                2: {"a": [0.2, 0.8], "b": [0.8, 0.2]},
            }
            for support_id, item_vectors in vectors.items():
                for item, vector in item_vectors.items():
                    for option_index, probability in enumerate(vector):
                        rows.append(
                            {
                                "support_id": support_id,
                                "job_id": f"j{support_id}",
                                "item": item,
                                "option_index": option_index,
                                "probability": probability,
                            }
                        )
            pd.DataFrame(rows).to_csv(support_path, index=False)
            targets = {
                "targets": [
                    {
                        "target_id": f"marginal:{item}",
                        "type": "marginal",
                        "items": [item],
                        "values": values,
                        "status": "accepted",
                    }
                    for item, values in {"a": [0.7, 0.3], "b": [0.7, 0.3]}.items()
                ]
            }

            design = geometry_repair_blueprint_design(
                mini_metadata(),
                targets,
                support_path,
                n_add=4,
                seed=42,
                target_source="targets.json",
            )

            geometry = design["geometry_repair"]
            self.assertGreater(geometry["minimum_maximum_absolute_residual"], 0)
            self.assertFalse(geometry["population_marginals_in_individual_prompts"])
            self.assertEqual(
                design["probabilities"]["minimum_intended_probability"],
                0.8,
            )
            patterns = design["components"][0]["patterns"]
            self.assertEqual(len(patterns), 4)
            self.assertEqual(len({tuple(pattern.items()) for pattern in patterns}), 4)
            direction = pd.DataFrame(geometry["direction"])
            for item in ["a", "b"]:
                best = direction[direction["item"].eq(item)].sort_values(
                    "separating_direction", ascending=False
                ).iloc[0]
                self.assertEqual(
                    patterns[0][item],
                    best["option_label"],
                )

    def test_commands_emit_one_json_envelope_and_canonical_errors(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["version"]), 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(
            set(payload),
            {"schema_version", "command", "status", "argv", "data", "warnings", "errors", "next_steps"},
        )
        self.assertEqual(payload["command"], "umriss version")

        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(main(["support", "build"]), 1)
        error = json.loads(stdout.getvalue())
        self.assertEqual(error["command"], "umriss support build")
        self.assertEqual(error["errors"][0]["code"], "invalid_arguments")
        self.assertEqual(stderr.getvalue(), "")

    def test_weighted_ordinary_survey_aggregation(self) -> None:
        metadata = mini_metadata()
        raw = pd.DataFrame(
            {
                "agent._weight": [0.75, 0.25],
                "answer.a": ["Yes", "No"],
                "answer.b": ["No", "No"],
            }
        )
        fit = pd.DataFrame(
            [
                {"item": item, "option_index": option, "prediction": value}
                for item, values in {"a": [0.7, 0.3], "b": [0.4, 0.6]}.items()
                for option, value in enumerate(values)
            ]
        )
        comparison = aggregate_survey_frame(raw, metadata, fit)
        yes_a = comparison[(comparison["item"] == "a") & (comparison["option_index"] == 0)].iloc[0]
        yes_b = comparison[(comparison["item"] == "b") & (comparison["option_index"] == 0)].iloc[0]
        self.assertEqual(yes_a["ordinary_survey"], 0.75)
        self.assertEqual(yes_b["ordinary_survey"], 0.0)
        coded = raw.assign(**{"answer.a": [0, 1], "answer.b": [1, 1]})
        coded_comparison = aggregate_survey_frame(coded, metadata, fit, answers_use_code=True)
        coded_yes_a = coded_comparison[
            (coded_comparison["item"] == "a") & (coded_comparison["option_index"] == 0)
        ].iloc[0]
        self.assertEqual(coded_yes_a["ordinary_survey"], 0.75)

    def test_marginals_import_uses_declared_microdata_columns(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata = mini_metadata()
            metadata["items"]["a"]["source_column"] = "pew_a"
            metadata["items"]["b"]["source_column"] = "pew_b"
            metadata_path = root / "metadata.json"
            respondents_path = root / "respondents.csv"
            output_path = root / "marginals.csv"
            write_json(metadata_path, metadata)
            pd.DataFrame(
                {
                    "weight": [1.0, 2.0, 1.0],
                    "pew_a": [1, 2, 2],
                    "pew_b": [2, 2, 1],
                }
            ).to_csv(respondents_path, index=False)
            self.assertEqual(
                main(
                    [
                        "marginals",
                        "import",
                        "--metadata",
                        str(metadata_path),
                        "--respondents",
                        str(respondents_path),
                        "--out",
                        str(output_path),
                    ]
                ),
                0,
            )
            rows = pd.read_csv(output_path)
            assert rows.loc[(rows["item"] == "a") & (rows["option_code"] == 1), "proportion"].iloc[0] == 0.25

    def test_design_create_validate_and_feasibility(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            design_path = root / "design.yaml"
            write_json(metadata_path, mini_metadata())
            self.assertEqual(
                main(
                    [
                        "design",
                        "create",
                        "--metadata",
                        str(metadata_path),
                        "--preset",
                        "pattern-coverage",
                        "--out",
                        str(design_path),
                    ]
                ),
                0,
            )
            self.assertEqual(main(["design", "validate", "--metadata", str(metadata_path), "--design", str(design_path)]), 0)
            design = design_path.read_text().replace("size: 4", "size: 3")
            design_path.write_text(design)
            self.assertEqual(main(["design", "validate", "--metadata", str(metadata_path), "--design", str(design_path)]), 1)

    def test_uniform_patterns_are_balanced_and_prompts_are_unique(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            write_json(metadata_path, mini_metadata())
            self.assertEqual(
                main(
                    [
                        "support",
                        "build",
                        "--metadata",
                        str(metadata_path),
                        "--preset",
                        "uniform-patterns",
                        "--n-support",
                        "8",
                        "--tag",
                        "balanced",
                        "--out",
                        str(root),
                    ]
                ),
                0,
            )
            rows = [json.loads(line) for line in (root / "balanced_prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertEqual(len({row["prompt"] for row in rows}), 8)
            self.assertTrue(all('"persona": "You are ..."' in row["prompt"] for row in rows))
            self.assertTrue(all('"persona_details": {' in row["prompt"] for row in rows))
            self.assertTrue(all('"a": "You explicitly believe or experience ..."' in row["prompt"] for row in rows))
            self.assertTrue(all("second-person" in row["prompt"] for row in rows))
            patterns = [tuple(row["pattern"][item] for item in ["a", "b"]) for row in rows]
            self.assertEqual(len(set(patterns)), 4)
            self.assertTrue(all(patterns.count(pattern) == 2 for pattern in set(patterns)))

    def test_balanced_blueprints_are_complete_and_fidelity_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata = mini_metadata()
            metadata_path = root / "metadata.json"
            write_json(metadata_path, metadata)
            self.assertEqual(
                main(
                    [
                        "support",
                        "build",
                        "--metadata",
                        str(metadata_path),
                        "--preset",
                        "balanced-blueprints",
                        "--n-support",
                        "4",
                        "--seed",
                        "42",
                        "--tag",
                        "blue",
                        "--out",
                        str(root / "prompts"),
                    ]
                ),
                0,
            )
            prompts = [
                json.loads(line)
                for line in (root / "prompts" / "blue_prompts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len({tuple(row["pattern"].items()) for row in prompts}), 4)
            self.assertTrue(all(set(row["pattern"]) == {"a", "b"} for row in prompts))
            for item in ["a", "b"]:
                self.assertEqual(
                    sorted(row["pattern"][item] for row in prompts),
                    ["No", "No", "Yes", "Yes"],
                )
            probability_rows = []
            for row in prompts:
                for item in ["a", "b"]:
                    intended = metadata["option_labels"].index(row["pattern"][item])
                    values = [0.9, 0.1] if intended == 0 else [0.1, 0.9]
                    if row["support_id"] == 4 and item == "b":
                        values = list(reversed(values))
                    for option_index, probability in enumerate(values):
                        probability_rows.append(
                            {
                                "support_id": row["support_id"],
                                "job_id": row["job_id"],
                                "item": item,
                                "option_index": option_index,
                                "option_code": option_index + 1,
                                "option_label": metadata["option_labels"][option_index],
                                "probability": probability,
                            }
                        )
            support_path = root / "support.csv"
            pd.DataFrame(probability_rows).to_csv(support_path, index=False)
            self.assertEqual(
                main(
                    [
                        "support",
                        "validate-blueprints",
                        "--support",
                        str(support_path),
                        "--plan",
                        str(root / "prompts" / "blue_support_plan.csv"),
                        "--metadata",
                        str(metadata_path),
                        "--tag",
                        "blue",
                        "--minimum-match-fraction",
                        "1",
                        "--out",
                        str(root / "validated"),
                    ]
                ),
                0,
            )
            fidelity = pd.read_csv(root / "validated" / "blue_blueprint_fidelity.csv")
            self.assertEqual(int(fidelity["accepted"].sum()), 3)
            validated = pd.read_csv(root / "validated" / "blue_validated_probabilities.csv")
            self.assertEqual(validated["support_id"].nunique(), 3)
            retries = pd.read_csv(root / "validated" / "blue_retry_job_ids.csv")
            self.assertEqual(retries["job_id"].tolist(), ["blue_004"])

    def test_target_feasibility_writes_convex_hull_witness(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata = mini_metadata()
            metadata_path = root / "metadata.json"
            targets_path = root / "targets.json"
            support_path = root / "support.csv"
            write_json(metadata_path, metadata)
            self.assertEqual(
                main(
                    [
                        "targets",
                        "from-metadata",
                        "--metadata",
                        str(metadata_path),
                        "--population",
                        "test",
                        "--out",
                        str(targets_path),
                    ]
                ),
                0,
            )
            rows = []
            patterns = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for support_id, pattern in enumerate(patterns, 1):
                for item_index, item in enumerate(["a", "b"]):
                    for option_index in range(2):
                        rows.append(
                            {
                                "support_id": support_id,
                                "job_id": f"s{support_id}",
                                "item": item,
                                "option_index": option_index,
                                "option_code": option_index + 1,
                                "option_label": metadata["option_labels"][option_index],
                                "probability": 1.0 if option_index == pattern[item_index] else 0.0,
                            }
                        )
            pd.DataFrame(rows).to_csv(support_path, index=False)
            self.assertEqual(
                main(
                    [
                        "targets",
                        "feasibility",
                        "--targets",
                        str(targets_path),
                        "--support",
                        str(support_path),
                        "--metadata",
                        str(metadata_path),
                        "--tolerance",
                        "0.000001",
                        "--tag",
                        "feasible",
                        "--out",
                        str(root / "fit"),
                    ]
                ),
                0,
            )
            summary = pd.read_csv(root / "fit" / "feasible_feasibility_summary.csv").iloc[0]
            self.assertTrue(bool(summary["inside_convex_hull_at_tolerance"]))
            self.assertAlmostEqual(float(summary["minimum_maximum_absolute_residual"]), 0.0)
            witness = pd.read_csv(root / "fit" / "feasible_feasibility_witness_weights.csv")
            self.assertAlmostEqual(float(witness["weight"].sum()), 1.0)

    def test_strict_parser_rejects_silent_probability_repair(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            raw_path = root / "raw.csv"
            write_json(metadata_path, mini_metadata())
            with raw_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["scenario.job_id", "answer.resp"])
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario.job_id": "bad",
                        "answer.resp": json.dumps(
                            {
                                "persona": "Your views are deliberately invalid here.",
                                "probabilities": {"a": [0.8, 0.8], "b": [-0.1, 1.1]},
                            }
                        ),
                    }
                )
            self.assertEqual(
                main(
                    [
                        "support",
                        "parse",
                        "--raw",
                        str(raw_path),
                        "--metadata",
                        str(metadata_path),
                        "--tag",
                        "bad",
                        "--out",
                        str(root / "banks"),
                    ]
                ),
                1,
            )

    def test_parser_preserves_item_complete_persona_details(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            raw_path = root / "raw.csv"
            write_json(metadata_path, mini_metadata())
            with raw_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["scenario.job_id", "answer.resp"])
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario.job_id": "detailed",
                        "answer.resp": json.dumps(
                            {
                                "persona": "You combine confidence on one issue with caution on another.",
                                "persona_details": {
                                    "a": "a: You would answer yes on Item A.",
                                    "b": "Your answer on Item B would be no.",
                                },
                                "probabilities": {
                                    "a": [0.9, 0.1],
                                    "b": [0.2, 0.8],
                                },
                            }
                        ),
                    }
                )
            self.assertEqual(
                main(
                    [
                        "support",
                        "parse",
                        "--raw",
                        str(raw_path),
                        "--metadata",
                        str(metadata_path),
                        "--tag",
                        "detailed",
                        "--out",
                        str(root / "banks"),
                    ]
                ),
                0,
            )
            point = pd.read_csv(root / "banks" / "detailed_points.csv").iloc[0]
            self.assertEqual(point["persona_detail_count"], 2)
            self.assertEqual(point["persona_detail_coverage"], 1.0)
            self.assertIn("a: You would answer yes on Item A.", point["persona"])
            self.assertIn("Your answer on Item B would be no.", point["persona"])

    def test_parser_rejects_incomplete_persona_details(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            raw_path = root / "raw.csv"
            write_json(metadata_path, mini_metadata())
            with raw_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["scenario.job_id", "answer.resp"])
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario.job_id": "incomplete",
                        "answer.resp": json.dumps(
                            {
                                "persona": "You have a partially described outlook.",
                                "persona_details": {
                                    "a": "You would answer yes on Item A."
                                },
                                "probabilities": {
                                    "a": [0.9, 0.1],
                                    "b": [0.2, 0.8],
                                },
                            }
                        ),
                    }
                )
            self.assertEqual(
                main(
                    [
                        "support",
                        "parse",
                        "--raw",
                        str(raw_path),
                        "--metadata",
                        str(metadata_path),
                        "--tag",
                        "incomplete",
                        "--out",
                        str(root / "banks"),
                    ]
                ),
                1,
            )

    def test_support_build_parse_and_loo(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            raw_path = root / "raw.csv"
            write_json(metadata_path, mini_metadata())
            write_raw(raw_path)

            self.assertEqual(
                main(["support", "build", "--metadata", str(metadata_path), "--preset", "pattern-coverage", "--tag", "mini", "--out", str(root / "prompts")]),
                0,
            )
            self.assertTrue((root / "prompts" / "mini_prompts.jsonl").exists())
            self.assertTrue((root / "prompts" / "mini_resolved_design.yaml").exists())
            self.assertTrue((root / "prompts" / "mini_support_plan.csv").exists())
            self.assertTrue((root / "prompts" / "mini_coverage.csv").exists())
            self.assertTrue((root / "prompts" / "mini_prompts.html").exists())

            self.assertEqual(main(["support", "parse", "--raw", str(raw_path), "--metadata", str(metadata_path), "--tag", "mini", "--out", str(root / "banks")]), 0)
            probs = pd.read_csv(root / "banks" / "mini_probabilities.csv")
            self.assertEqual(set(probs["item"]), {"a", "b"})
            self.assertEqual(probs["support_id"].nunique(), 2)
            self.assertEqual(
                main(
                    [
                        "support",
                        "uniformity",
                        "--support",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--metadata",
                        str(metadata_path),
                        "--tolerance",
                        "0.01",
                        "--out",
                        str(root / "banks" / "uniformity.csv"),
                    ]
                ),
                0,
            )
            uniformity_pre = pd.read_csv(root / "banks" / "uniformity.csv")
            self.assertFalse(bool(uniformity_pre["passes"].all()))
            self.assertEqual(
                main(
                    [
                        "support",
                        "augment-uniform",
                        "--support",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--metadata",
                        str(metadata_path),
                        "--tag",
                        "mini_balance",
                        "--n-add",
                        "6",
                        "--tolerance",
                        "0.01",
                        "--out",
                        str(root / "balance"),
                    ]
                ),
                0,
            )
            prompts = [json.loads(line) for line in (root / "balance" / "mini_balance_prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(prompts), 6)
            self.assertTrue(all(set(row["pattern"]) == {"a", "b"} for row in prompts))
            self.assertEqual(
                main(
                    [
                        "support",
                        "merge",
                        "--base",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--additions",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--tag",
                        "merged",
                        "--out",
                        str(root / "merged"),
                    ]
                ),
                0,
            )
            merged = pd.read_csv(root / "merged" / "merged_probabilities.csv")
            self.assertEqual(merged["support_id"].nunique(), 4)

            self.assertEqual(
                main(
                    [
                        "validate",
                        "marginals",
                        "--support",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--metadata",
                        str(metadata_path),
                        "--uniform-tolerance",
                        "0.01",
                        "--tag",
                        "rejected",
                        "--out",
                        str(root / "rejected"),
                    ]
                ),
                1,
            )
            self.assertEqual(main(["validate", "marginals", "--support", str(root / "banks" / "mini_probabilities.csv"), "--metadata", str(metadata_path), "--tag", "mini", "--out", str(root / "derived")]), 0)
            summary = pd.read_csv(root / "derived" / "mini_generated_support_summary.csv")
            self.assertIn("generated support mixture", set(summary["method"]))
            self.assertTrue({"mean_kl_divergence", "mean_cross_entropy", "mean_target_entropy"} <= set(summary.columns))
            self.assertTrue((summary["mean_kl_divergence"] >= 0).all())
            self.assertTrue((summary["mean_cross_entropy"] >= summary["mean_target_entropy"]).all())
            weights = pd.read_csv(root / "derived" / "mini_generated_support_weights.csv")
            self.assertEqual(set(weights["holdout"]), {"a", "b"})
            for _, fold in weights.groupby("holdout"):
                self.assertAlmostEqual(float(fold["weight"].sum()), 1.0)
            uniformity = pd.read_csv(root / "derived" / "mini_support_uniformity.csv")
            self.assertEqual(set(uniformity["item"]), {"a", "b"})
            self.assertTrue({"equal_weight_prediction", "uniform_target", "rmse_from_uniform", "max_absolute_deviation"} <= set(uniformity.columns))
            self.assertEqual(main(["support", "inspect", "--bank", str(root / "banks" / "mini_probabilities.csv")]), 0)
            self.assertEqual(main(["compare", "--run", "mini=Mini:Test bank", "--derived", str(root / "derived"), "--out", str(root / "derived" / "comparison.csv")]), 0)
            comparison = pd.read_csv(root / "derived" / "comparison.csv")
            self.assertEqual(set(comparison["bank"]), {"Test bank"})
            self.assertEqual(main(["report", "--tag", "mini", "--derived", str(root / "derived"), "--out", str(root / "derived" / "mini.md")]), 0)
            self.assertTrue((root / "derived" / "mini.md").read_text().startswith("# umriss report: mini"))
            self.assertEqual(main(["report-data", "build", "--derived", str(root / "derived"), "--out", str(root / "report_data")]), 0)
            manifest = json.loads((root / "report_data" / "manifest.json").read_text())
            self.assertEqual(manifest["facts"]["support_banks"], 1)
            self.assertTrue((root / "report_data" / "support_bank_summary.csv").exists())
            self.assertTrue((root / "report_data" / "method_comparison.csv").exists())
            self.assertTrue((root / "report_data" / "diagnostics_flags.csv").exists())
            self.assertIn("mini", (root / "report_data" / "prose_facts.md").read_text())

    def test_fit_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            raw_path = root / "raw.csv"
            write_json(metadata_path, mini_metadata())
            write_raw(raw_path)
            self.assertEqual(main(["support", "parse", "--raw", str(raw_path), "--metadata", str(metadata_path), "--tag", "mini", "--out", str(root / "banks")]), 0)
            self.assertEqual(
                main(
                    [
                        "fit",
                        "--support",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--metadata",
                        str(metadata_path),
                        "--exclude-item",
                        "a",
                        "--tag",
                        "mini_holdout_a",
                        "--out",
                        str(root / "derived"),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "predict",
                        "--support",
                        str(root / "banks" / "mini_probabilities.csv"),
                        "--weights",
                        str(root / "derived" / "mini_holdout_a_weights.csv"),
                        "--metadata",
                        str(metadata_path),
                        "--item",
                        "a",
                        "--out",
                        str(root / "derived" / "prediction.csv"),
                    ]
                ),
                0,
            )
            pred = pd.read_csv(root / "derived" / "prediction.csv")
            self.assertEqual(set(pred["item"]), {"a"})

    def test_project_incremental_authoring(self) -> None:
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            os.chdir(root)
            try:
                self.assertEqual(main(["init"]), 0)
                self.assertEqual(main(["battery", "create", "--battery-id", "demo", "--wave", "T1", "--battery", "DEMO", "--topic", "demo topic", "--context", "demo context"]), 0)
                self.assertEqual(
                    main(
                        [
                            "question",
                            "add",
                            "--battery",
                            "demo",
                            "--item",
                            "q1",
                            "--question-stem",
                            "Stem?",
                            "--item-text",
                            "Text",
                            "--option-code",
                            "1",
                            "--option",
                            "Yes",
                            "--option-code",
                            "2",
                            "--option",
                            "No",
                            "--scale-type",
                            "nominal",
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["marginal", "add", "--battery", "demo", "--item", "q1", "--proportion", "0.25", "--proportion", "0.75"]), 0)
                self.assertEqual(main(["battery", "compile", "--battery", "demo", "--path", str(root / "compiled.json")]), 0)
                compiled = json.loads((root / "compiled.json").read_text())
                self.assertEqual(compiled["truth"]["q1"], [0.25, 0.75])
                self.assertEqual(
                    main(
                        [
                            "support",
                            "build",
                            "--battery",
                            "demo",
                            "--preset",
                            "pattern-coverage",
                            "--tag",
                            "demo_support",
                            "--n-support",
                            "4",
                            "--out",
                            str(root / "prompts"),
                        ]
                    ),
                    0,
                )
                self.assertTrue((root / "prompts" / "demo_support_prompts.jsonl").exists())
            finally:
                os.chdir(cwd)

    def test_support_build_from_generic_design(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            design_path = root / "design.json"
            write_json(metadata_path, mini_metadata())
            write_json(
                design_path,
                {
                    "schema_version": 1,
                    "size": 5,
                    "seed": 17,
                    "coverage": {"mode": "partial", "allocation": "balanced"},
                    "components": [
                        {
                            "type": "option_coverage",
                            "items": "all",
                            "minimum_per_option": 1,
                            "coherence": "item_specific",
                            "intensity": {"values": ["moderate", "strong"], "allocation": "balanced"},
                        }
                    ],
                },
            )
            self.assertEqual(
                main(["support", "build", "--metadata", str(metadata_path), "--design", str(design_path), "--tag", "generic", "--n-support", "5", "--out", str(root / "prompts")]),
                0,
            )
            rows = [json.loads(line) for line in (root / "prompts" / "generic_prompts.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            plan = pd.read_csv(root / "prompts" / "generic_support_plan.csv")
            self.assertEqual(set(plan["design_type"]), {"option_coverage"})
            self.assertEqual(set(plan["coherence"]), {"item_specific"})

    def test_guides_and_artifact_next(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            design_path = root / "design.json"
            write_json(metadata_path, mini_metadata())
            write_json(
                design_path,
                {
                    "schema_version": 1,
                    "size": 4,
                    "coverage": {"mode": "complete", "allocation": "balanced"},
                    "components": [
                        {
                            "type": "option_coverage",
                            "items": "all",
                            "minimum_per_option": 1,
                            "coherence": "item_specific",
                            "intensity": {"values": ["moderate"], "allocation": "balanced"},
                        }
                    ],
                },
            )
            self.assertEqual(main(["guide", "workflow"]), 0)
            self.assertEqual(main(["guide", "--topic", "ep-boundary"]), 0)
            self.assertEqual(
                main(
                    [
                        "next",
                        "--tag",
                        "demo",
                        "--metadata",
                        str(metadata_path),
                        "--design",
                        str(design_path),
                        "--prompt-dir",
                        str(root / "prompts"),
                        "--raw-dir",
                        str(root / "raw"),
                        "--derived-dir",
                        str(root / "derived"),
                    ]
                ),
                0,
            )
            self.assertEqual(main(["support", "build", "--metadata", str(metadata_path), "--design", str(design_path), "--tag", "demo", "--out", str(root / "prompts")]), 0)
            self.assertEqual(main(["next", "--tag", "demo", "--metadata", str(metadata_path), "--prompt-dir", str(root / "prompts"), "--raw-dir", str(root / "raw"), "--derived-dir", str(root / "derived")]), 0)
            paths = {
                "prompt_dir": root / "prompts",
                "raw_dir": root / "raw",
                "bank_dir": root / "bank",
                "derived_dir": root / "derived",
            }
            state = next_for_artifacts("demo", metadata=metadata_path, **paths)
            self.assertEqual(state["stage"], "export-jobs")
            self.assertIn("--tag demo", state["recommendation"])

            jobs_path = root / "prompts" / "custom.jobs.ep"
            results_path = root / "results" / "custom.results.ep"
            jobs_path.mkdir()
            write_json(
                root / "prompts" / "demo_manifest.json",
                {
                    "tag": "demo",
                    "prompts": str(root / "prompts" / "demo_prompts.jsonl"),
                    "jobs": str(jobs_path),
                    "results": str(results_path),
                },
            )
            state = next_for_artifacts("demo", metadata=metadata_path, **paths)
            self.assertEqual(state["stage"], "run-externally")
            self.assertIn(str(jobs_path), state["recommendation"])

            results_path.parent.mkdir()
            results_path.mkdir()
            self.assertEqual(
                next_for_artifacts("demo", metadata=metadata_path, **paths)["stage"],
                "register-results",
            )
            (root / "raw").mkdir()
            (root / "raw" / "demo_raw.csv").write_text("scenario.job_id,answer.resp\n")
            self.assertEqual(
                next_for_artifacts("demo", metadata=metadata_path, **paths)["stage"],
                "parse-results",
            )
            (root / "bank").mkdir()
            (root / "bank" / "demo_probabilities.csv").write_text("job_id,item,option_index,probability\n")
            self.assertEqual(
                next_for_artifacts("demo", metadata=metadata_path, **paths)["stage"],
                "evaluate",
            )
            (root / "derived").mkdir()
            (root / "derived" / "demo_generated_support_summary.csv").write_text("holdout,rmse\n")
            self.assertEqual(
                next_for_artifacts("demo", metadata=metadata_path, **paths)["stage"],
                "report-or-compare",
            )

    def test_support_export_writes_jobs_ep(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            os.environ["EDSL_LOG_DIR"] = str(root / "edsl_logs")
            metadata_path = root / "metadata.json"
            write_json(metadata_path, mini_metadata())
            self.assertEqual(main(["support", "build", "--metadata", str(metadata_path), "--preset", "pattern-coverage", "--tag", "mini", "--out", str(root / "prompts")]), 0)
            self.assertEqual(main(["support", "build", "--metadata", str(metadata_path), "--preset", "pattern-coverage", "--tag", "mini", "--out", str(root / "prompts")]), 0)
            jobs_path = root / "prompts" / "mini.jobs.ep"
            self.assertEqual(main(["support", "export", "--prompts", str(root / "prompts" / "mini_prompts.jsonl"), "--path", str(jobs_path), "--model", "test", "--limit", "1"]), 0)
            self.assertEqual(main(["support", "export", "--prompts", str(root / "prompts" / "mini_prompts.jsonl"), "--path", str(jobs_path), "--model", "test", "--limit", "1"]), 0)
            self.assertTrue(jobs_path.exists())
            manifest = json.loads((root / "prompts" / "mini_manifest.json").read_text())
            self.assertEqual(manifest["run_contract"]["owner"], "external_ep")
            self.assertTrue(manifest["run_command"].startswith("ep run --jobs "))
            self.assertEqual(manifest["model_calls"], 1)
            self.assertIn("--tag mini", manifest["register_command"])
            self.assertIn("--out", manifest["register_command"])
            self.assertEqual(manifest["next_steps"], [manifest["run_command"], manifest["register_command"]])

            changed = mini_metadata()
            changed["context"] = "A changed input must not silently reuse the tag."
            write_json(metadata_path, changed)
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["support", "build", "--metadata", str(metadata_path), "--preset", "pattern-coverage", "--tag", "mini", "--out", str(root / "prompts")]), 1)
            self.assertEqual(json.loads(output.getvalue())["errors"][0]["code"], "output_conflict")

    def test_baseline_jobs_enforce_leave_one_out_boundary_and_parse(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            metadata_path = root / "metadata.json"
            prompts_path = root / "demo_baseline_prompts.jsonl"
            raw_path = root / "raw.csv"
            write_json(metadata_path, mini_metadata())
            self.assertEqual(
                main(
                    [
                        "baseline",
                        "build",
                        "--metadata",
                        str(metadata_path),
                        "--mode",
                        "both",
                        "--tag",
                        "demo",
                        "--out",
                        str(root),
                    ]
                ),
                0,
            )
            prompts = [json.loads(line) for line in prompts_path.read_text().splitlines()]
            self.assertEqual(len(prompts), 4)
            conditioned_a = next(
                row for row in prompts if row["mode"] == "conditioned_direct" and row["holdout"] == "a"
            )
            self.assertEqual(conditioned_a["held_in"], ["b"])
            self.assertIn("[0.4, 0.6]", conditioned_a["prompt"])
            self.assertNotIn("[0.7, 0.3]", conditioned_a["prompt"])
            with raw_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scenario.job_id", "answer.resp"])
                writer.writeheader()
                for row in prompts:
                    writer.writerow(
                        {
                            "scenario.job_id": row["job_id"],
                            "answer.resp": json.dumps(
                                {"reasoning_summary": "test", "probabilities": [0.6, 0.4]}
                            ),
                        }
                    )
            self.assertEqual(
                main(
                    [
                        "baseline",
                        "parse",
                        "--raw",
                        str(raw_path),
                        "--prompts",
                        str(prompts_path),
                        "--metadata",
                        str(metadata_path),
                        "--tag",
                        "demo",
                        "--out",
                        str(root / "parsed"),
                    ]
                ),
                0,
            )
            self.assertTrue((root / "parsed" / "demo_one_shot.csv").exists())
            self.assertTrue((root / "parsed" / "demo_conditioned_direct.csv").exists())

            parser = build_parser()
            top_level = parser._subparsers._group_actions[0].choices
            baseline_parser = top_level["baseline"]
            baseline_commands = baseline_parser._subparsers._group_actions[0].choices
            self.assertNotIn("run", baseline_commands)

    def test_retry_audit_preserves_attempt_attribution_and_missing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                "\n".join(
                    json.dumps({"job_id": job_id, "prompt": f"Prompt {job_id}"})
                    for job_id in ("j1", "j2", "j3")
                )
                + "\n"
            )
            attempts = [root / "attempt1.results.ep", root / "attempt2.results.ep"]
            for path in attempts:
                path.write_text(path.name)
            frames = [
                pd.DataFrame({"scenario.job_id": ["j1"], "answer.resp": ["one"]}),
                pd.DataFrame({"scenario.job_id": ["j2"], "answer.resp": ["two"]}),
            ]
            with patch("umriss.parsing.load_results_ep_to_pandas", side_effect=frames):
                data = audit_result_attempts(attempts, prompts, "demo", root / "audit")
            self.assertFalse(data["complete"])
            self.assertEqual(data["missing_jobs"], 1)
            self.assertEqual(
                pd.read_csv(data["missing_job_ids_path"])["job_id"].tolist(),
                ["j3"],
            )
            merged = pd.read_csv(data["merged_raw_path"])
            self.assertEqual(merged["_umriss_source_run"].tolist(), [1, 2])
            self.assertEqual(
                merged["_umriss_source_results"].tolist(),
                [str(attempts[0]), str(attempts[1])],
            )

    def test_registration_rejects_incomplete_results(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            prompts = root / "prompts.jsonl"
            prompts.write_text(
                '{"job_id":"j1","prompt":"One"}\n'
                '{"job_id":"j2","prompt":"Two"}\n'
            )
            results = root / "attempt.results.ep"
            results.write_text("fixture")
            frame = pd.DataFrame(
                {"scenario.job_id": ["j1"], "answer.resp": ["valid"]}
            )
            with patch("umriss.parsing.load_results_ep_to_pandas", return_value=frame):
                with self.assertRaises(UmrissError) as raised:
                    register_results(results, prompts, "demo", root / "raw")
            self.assertEqual(raised.exception.code, "incomplete_results")
            self.assertIn("audit-results", raised.exception.next_steps[0])

    def test_plot_validation_uses_generated_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repository = Path(__file__).resolve().parents[1]
            self.assertEqual(
                main(
                    [
                        "plot",
                        "validation",
                        "--derived",
                        str(repository / "examples" / "pew_w154" / "run" / "derived"),
                        "--tag",
                        "pew_w154_diff1_uniform_n208",
                        "--out",
                        str(root),
                    ]
                ),
                0,
            )
            manifest = json.loads((root / "pew_w154_diff1_uniform_n208_plots.json").read_text())
            self.assertEqual(len(manifest["plots"]), 5)
            self.assertTrue(all(Path(path).exists() for path in manifest["plots"].values()))

    def test_export_edsl_twins_uses_named_second_person_trait(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            from edsl import AgentList

            root = Path(d)
            points_path = root / "points.csv"
            weights_path = root / "weights.csv"
            agents_path = root / "twins.agents.ep"
            pd.DataFrame(
                [
                    {"support_id": 1, "job_id": "j1", "persona": "Your views favor cautious institutional reform."},
                    {"support_id": 2, "job_id": "j2", "persona": "Your views confidently defend existing institutions."},
                ]
            ).to_csv(points_path, index=False)
            pd.DataFrame(
                [
                    {"support_id": 1, "job_id": "j1", "holdout": "a", "weight": 0.75},
                    {"support_id": 2, "job_id": "j2", "holdout": "a", "weight": 0.25},
                ]
            ).to_csv(weights_path, index=False)
            self.assertEqual(
                main(
                    [
                        "twins",
                        "export-edsl",
                        "--points",
                        str(points_path),
                        "--weights",
                        str(weights_path),
                        "--persona-trait",
                        "institutional_attitudes",
                        "--holdout",
                        "a",
                        "--path",
                        str(agents_path),
                    ]
                ),
                0,
            )
            agents = AgentList.git.load(str(agents_path))
            self.assertEqual(
                [agent.traits["institutional_attitudes"] for agent in agents],
                [
                    "Your views favor cautious institutional reform.",
                    "Your views confidently defend existing institutions.",
                ],
            )
            self.assertTrue(all("instruction" not in agent.to_dict() for agent in agents))
            self.assertEqual([agent.traits["_weight"] for agent in agents], [0.75, 0.25])
            self.assertTrue(all("_weight" not in agent.prompt().text for agent in agents))
            sidecar = pd.read_csv(root / "twins.agents_weights.csv")
            self.assertEqual(sidecar["weight"].tolist(), [0.75, 0.25])
            support_path = root / "support.csv"
            metadata_path = root / "metadata.json"
            embedded_path = root / "embedded.agents.ep"
            write_json(metadata_path, mini_metadata())
            pd.DataFrame(
                [
                    {"job_id": job, "item": item, "option_index": option, "probability": probability}
                    for job, vectors in {
                        "j1": {"a": [0.8, 0.2], "b": [0.3, 0.7]},
                        "j2": {"a": [0.1, 0.9], "b": [0.6, 0.4]},
                    }.items()
                    for item, values in vectors.items()
                    for option, probability in enumerate(values)
                ]
            ).to_csv(support_path, index=False)
            self.assertEqual(
                main(
                    [
                        "twins",
                        "embed-probabilities",
                        "--agents",
                        str(agents_path),
                        "--support",
                        str(support_path),
                        "--metadata",
                        str(metadata_path),
                        "--probability-trait",
                        "response_propensities",
                        "--path",
                        str(embedded_path),
                    ]
                ),
                0,
            )
            embedded = AgentList.git.load(str(embedded_path))
            self.assertIn("“Yes” 80.0%", embedded[0].traits["response_propensities"])
            self.assertEqual(embedded[0].traits["_weight"], 0.75)

    def test_probabilistic_plot_accepts_simulation_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            comparison = root / "comparison.csv"
            simulations = root / "simulations.csv"
            output = root / "plot.svg"
            pd.DataFrame(
                [
                    {
                        "item": "a",
                        "option_index": 0,
                        "option_label": "Yes",
                        "pew_marginal": 0.7,
                        "umriss_fitted_mixture": 0.7,
                        "meta_probability_mixture": 0.68,
                        "meta_resolved_answers": 0.6,
                    }
                ]
            ).to_csv(comparison, index=False)
            pd.DataFrame(
                [{"item": "a", "option_index": 0, "mean": 0.68, "q025": 0.5, "q975": 0.82}]
            ).to_csv(simulations, index=False)
            result = plot_survey_comparison(comparison, output, simulations_path=simulations)
            self.assertTrue(output.exists())
            self.assertTrue(result["simulation_intervals"])


if __name__ == "__main__":
    unittest.main()
