"""Lightweight contract tests; they never import or call a real search adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


EVALUATOR_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(EVALUATOR_DIR))
import evaluate_collection as evaluator  # noqa: E402


def _query_row(query_id: str, target_id: str, *, s001: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "query_id": query_id,
        "query": "测试一个公开开发检索问题",
        "case_role": "positive_any_hit",
        "primary_target_ids": [target_id],
        "target_grain": "knowledge_unit",
        "split": "public_development",
        "source_filter": "S001" if s001 else None,
    }
    if s001:
        row["subject_filter"] = "chemical-engineering-principles"
        row["expected_knowledge_unit_ids"] = [target_id]
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


class CollectionEvaluatorTests(unittest.TestCase):
    def test_combine_preserves_targets_and_forces_unfiltered_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            four = root / "four.jsonl"
            s001 = root / "s001.jsonl"
            _write_jsonl(
                four,
                [_query_row(f"four-{i:02d}", f"F-{i:02d}", s001=False) for i in range(16)],
            )
            _write_jsonl(
                s001,
                [_query_row(f"s001-{i:02d}", f"S-{i:02d}", s001=True) for i in range(16)],
            )
            rows = evaluator.combine_queries(four, s001)

        self.assertEqual(len(rows), 32)
        self.assertEqual(rows[0]["query_id"], "four-00")
        self.assertEqual(rows[-1]["primary_target_ids"], ["S-15"])
        self.assertTrue(all(row["source_filter"] is None for row in rows))
        self.assertTrue(all(row["subject_filter"] is None for row in rows))
        self.assertEqual(rows[-1]["input_source_filter"], "S001")
        self.assertEqual(
            rows[-1]["input_subject_filter"], "chemical-engineering-principles"
        )

    def test_query_input_must_be_public_development_and_exactly_sixteen(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "queries.jsonl"
            rows = [
                _query_row(f"q-{i:02d}", f"K-{i:02d}", s001=False)
                for i in range(15)
            ]
            _write_jsonl(path, rows)
            with self.assertRaises(evaluator.CollectionEvaluationError):
                evaluator.load_query_file(path, group="four")

            rows.append(_query_row("q-15", "K-15", s001=False))
            rows[-1]["split"] = "heldout"
            _write_jsonl(path, rows)
            with self.assertRaises(evaluator.CollectionEvaluationError):
                evaluator.load_query_file(path, group="four")

    def test_complete_ku_is_compared_exactly_but_not_written_to_projection(self) -> None:
        body = "保持完整正文的候选知识单元。"
        body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        expected = {
            "knowledge_unit_id": "K-1",
            "node_id": "K-1",
            "source_id": "S001",
            "source_chain_id": "curated-s001-upper-v1:S001",
            "text": body,
            "text_sha256": body_sha,
            "content_available": True,
            "retrieval_eligible": True,
            "embedding_eligible": True,
            "optional_metadata": None,
        }
        result = {
            "bundle_id": "curated-s001-upper-v1",
            "knowledge_unit_id": "K-1",
            "knowledge_unit": dict(expected),
            "score": 0.75,
        }
        projection, errors = evaluator.validate_returned_result(
            result,
            method="lexical",
            normative_by_id={"K-1": ("curated-s001-upper-v1", expected)},
        )
        self.assertEqual(errors, [])
        self.assertEqual(projection["text_sha256"], body_sha)
        self.assertNotIn("text", projection)
        self.assertEqual(projection["score"], 0.75)

        changed = dict(result)
        changed_unit = dict(expected)
        del changed_unit["optional_metadata"]
        changed_unit["text"] = "正文被改写。"
        changed["knowledge_unit"] = changed_unit
        _, changed_errors = evaluator.validate_returned_result(
            changed,
            method="lexical",
            normative_by_id={"K-1": ("curated-s001-upper-v1", expected)},
        )
        self.assertTrue(
            any(error.startswith("knowledge_unit_fields_mismatch:") for error in changed_errors)
        )
        self.assertIn(
            "knowledge_unit_fields_mismatch:optional_metadata,text",
            changed_errors,
        )

    def test_metrics_require_a_valid_complete_ku_and_expected_calls_are_ninety_six(self) -> None:
        query = {
            "primary_target_ids": ["K-1"],
        }
        diagnostics = [
            {
                "top5": [
                    {
                        "rank": 2,
                        "knowledge_unit_id": "K-1",
                        "validation_errors": [],
                    }
                ]
            }
        ]
        metrics = evaluator._metric_summary([query], diagnostics)
        self.assertEqual(metrics["hit_at_5"], 1.0)
        self.assertEqual(metrics["mrr_at_5"], 0.5)
        self.assertEqual(metrics["top1"], 0.0)
        self.assertEqual(len(evaluator.METHODS) * 32, 96)

        diagnostics[0]["top5"][0]["validation_errors"] = ["complete_knowledge_unit_missing"]
        self.assertEqual(evaluator._metric_summary([query], diagnostics)["hit_at_5"], 0.0)

        diagnostics[0]["top5"][0]["validation_errors"] = []
        diagnostics[0]["api_error"] = "search_returned_more_than_limit"
        self.assertEqual(evaluator._metric_summary([query], diagnostics)["hit_at_5"], 0.0)

    def test_over_limit_result_is_an_error_and_cannot_score(self) -> None:
        units = {}
        results = []
        for index in range(6):
            unit_id = f"K-{index}"
            body = f"完整正文 {index}"
            unit = {
                "knowledge_unit_id": unit_id,
                "node_id": unit_id,
                "source_id": "S001",
                "source_chain_id": "curated-s001-upper-v1:S001",
                "text": body,
                "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "content_available": True,
                "retrieval_eligible": True,
                "embedding_eligible": True,
            }
            units[unit_id] = ("curated-s001-upper-v1", unit)
            results.append(
                {
                    "bundle_id": "curated-s001-upper-v1",
                    "knowledge_unit_id": unit_id,
                    "knowledge_unit": unit,
                    "score": 1.0 - index / 10,
                }
            )

        class OverLimitCollection:
            def search(self, **kwargs: object) -> list[dict[str, object]]:
                return results

        query = {
            "input_group": "s001",
            "query_id": "over-limit",
            "query": "检索问题",
            "primary_target_ids": ["K-0"],
            "expected_knowledge_unit_ids": ["K-0"],
            "input_source_filter": "S001",
            "input_subject_filter": "chemical-engineering-principles",
        }
        report = evaluator.evaluate_method(
            OverLimitCollection(),
            [query],
            method="lexical",
            normative_by_id=units,
            model_dir=Path("model"),
            lock_path=Path("lock"),
            vendor_path=Path("vendor"),
        )
        self.assertEqual(report["api_call_count"], 1)
        self.assertEqual(report["validation_error_count"], 1)
        self.assertEqual(
            report["per_query"][0]["api_error"], "search_returned_more_than_limit"
        )
        self.assertEqual(report["aggregate"]["hit_at_5"], 0.0)

    def test_main_returns_nonzero_after_writing_error_report(self) -> None:
        fake_report = {
            "status": "review_only_machine_diagnostic",
            "scope": {"combined_query_count": 32},
            "execution": {
                "api_call_count": 96,
                "returned_validation_error_count": 1,
            },
        }
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "error-report.json"
            argv = [
                "--collection", "collection.json",
                "--knowledge-root", "knowledge",
                "--query-module", "query-module",
                "--four-queries", "four.jsonl",
                "--s001-queries", "s001.jsonl",
                "--model-dir", "model",
                "--lock", "lock.json",
                "--vendor", "vendor",
                "--output", str(output),
            ]
            with patch.object(evaluator, "run", return_value=fake_report):
                self.assertEqual(evaluator.main(argv), 1)
            self.assertTrue(output.is_file())

    def test_public_error_removes_absolute_path(self) -> None:
        test_drive = chr(67) + ":"
        test_path = test_drive + "\\private\\secret.json"
        error = evaluator._public_error(RuntimeError("failed at " + test_path))
        self.assertNotIn(test_drive, error)
        self.assertIn("<path>", error)


if __name__ == "__main__":
    unittest.main()
