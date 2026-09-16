"""Structural checks for the frozen S001 public retrieval inputs.

These tests deliberately do not import or call the retrieval API.  They only
check that the public examples, diagnostic cases, and frozen KU identities
remain a coherent review input before vectors are available.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest


RETRIEVAL_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_KUS = RETRIEVAL_ROOT / "knowledge" / "curated-s001-upper-v1" / "knowledge_units.jsonl"
POSITIVE = RETRIEVAL_ROOT / "examples" / "s001-public-development-queries.jsonl"
DIAGNOSTICS = RETRIEVAL_ROOT / "examples" / "s001-out-of-bundle-diagnostics.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class FrozenRetrievalInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.positive = _read_jsonl(POSITIVE)
        cls.diagnostics = _read_jsonl(DIAGNOSTICS)
        cls.units = {
            row["knowledge_unit_id"]: row
            for row in _read_jsonl(BUNDLE_KUS)
        }

    def test_positive_set_has_four_queries_per_chapter(self) -> None:
        self.assertEqual(len(self.positive), 16)
        ids = [row["query_id"] for row in self.positive]
        self.assertEqual(len(ids), len(set(ids)))
        counts: dict[str, int] = {}
        for row in self.positive:
            counts[row["chapter_id"]] = counts.get(row["chapter_id"], 0) + 1
        self.assertEqual(
            counts,
            {
                "S001-CURATED-CH02": 4,
                "S001-CURATED-CH03": 4,
                "S001-CURATED-CH04": 4,
                "S001-CURATED-CH05": 4,
            },
        )

    def test_positive_targets_are_frozen_bundle_units(self) -> None:
        for row in self.positive:
            self.assertEqual(row["expected_knowledge_unit_ids"], row["primary_target_ids"])
            self.assertEqual(len(row["expected_knowledge_unit_ids"]), 1)
            target = row["expected_knowledge_unit_ids"][0]
            self.assertIn(target, self.units)
            self.assertEqual(row["source_filter"], "S001")
            self.assertEqual(row["split"], "public_development")
            self.assertNotEqual(row["query"].strip(), self.units[target]["title"].strip())

    def test_diagnostics_are_not_false_rejection_cases(self) -> None:
        self.assertEqual(len(self.diagnostics), 4)
        ids = [row["query_id"] for row in self.diagnostics]
        self.assertEqual(len(ids), len(set(ids)))
        for row in self.diagnostics:
            self.assertEqual(row["expected_knowledge_unit_ids"], [])
            self.assertEqual(row["primary_target_ids"], [])
            self.assertFalse(row["auto_reject_expected"])
            self.assertEqual(row["case_role"], "diagnostic_only")
            self.assertEqual(row["split"], "public_out_of_bundle_diagnostic")


if __name__ == "__main__":
    unittest.main()
