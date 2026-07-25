from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from umriss.cli import main


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def mini_metadata() -> dict:
    return {
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
            "answer.resp": json.dumps({"persona": "yes type", "probabilities": {"a": [0.9, 0.1], "b": [0.2, 0.8]}}),
        },
        {
            "scenario.job_id": "s2",
            "answer.resp": json.dumps({"persona": "no type", "probabilities": {"a": [0.2, 0.8], "b": [0.8, 0.2]}}),
        },
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario.job_id", "answer.resp"])
        writer.writeheader()
        writer.writerows(rows)


class UmrissCliTests(unittest.TestCase):
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
            patterns = [tuple(row["pattern"][item] for item in ["a", "b"]) for row in rows]
            self.assertEqual(len(set(patterns)), 4)
            self.assertTrue(all(patterns.count(pattern) == 2 for pattern in set(patterns)))

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
                                "profile_summary": "invalid",
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
                        "loo",
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
            self.assertEqual(main(["loo", "--support", str(root / "banks" / "mini_probabilities.csv"), "--metadata", str(metadata_path), "--tag", "mini", "--out", str(root / "derived")]), 0)
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

    def test_support_export_writes_jobs_ep(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            os.environ["EDSL_LOG_DIR"] = str(root / "edsl_logs")
            metadata_path = root / "metadata.json"
            write_json(metadata_path, mini_metadata())
            self.assertEqual(main(["support", "build", "--metadata", str(metadata_path), "--preset", "pattern-coverage", "--tag", "mini", "--out", str(root / "prompts")]), 0)
            jobs_path = root / "prompts" / "mini.jobs.ep"
            self.assertEqual(main(["support", "export", "--prompts", str(root / "prompts" / "mini_prompts.jsonl"), "--path", str(jobs_path), "--limit", "1"]), 0)
            self.assertTrue(jobs_path.exists())
            manifest = json.loads((root / "prompts" / "manifest.json").read_text())
            self.assertEqual(manifest["run_contract"]["owner"], "agent")
            self.assertTrue(manifest["run_command"].startswith("ep run --jobs "))


if __name__ == "__main__":
    unittest.main()
