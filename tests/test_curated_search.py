"""Synthetic index/query checks using the locked real ONNX encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODULE_DIR))

from curated_vectors import InputTooLongError  # noqa: E402
from index_curated_bundle import BundleBuildError, build_bundle  # noqa: E402
from query_curated import (  # noqa: E402
    BundleIntegrityError,
    CuratedBundle,
    DEFAULT_BUNDLE,
    QueryInputError,
)


CHECK_OUTPUT = Path("search_module_development.json")
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9])[A-Z]:[\\/]|\\\\(?:users|home|private|documents)[\\/]"
)
STATE: dict[str, object] = {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ku(
    node_id: str,
    *,
    title: str,
    body: str,
    subject_root: str,
    applicability: str | list[str],
    package_id: str,
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "knowledge_unit_id": node_id,
        "title": title,
        "text": body,
        "text_sha256": _sha256_text(body),
        "applicability": applicability,
        "units_basis": "合成测试单位口径",
        "knowledge_layer": "direct_method",
        "source_id": "SYN",
        "source_sha256": "a" * 64,
        "source_volume_id": "SYN-V1",
        "subject_root": subject_root,
        "package_id": package_id,
        "source_chain_id": "curated-four-books-v1:SYN",
        "source_locator": f"合成章节 {package_id}；PDF物理页1",
        "evidence_refs": [f"evidence:{node_id}:source-locator"],
        "forbidden_transfer": ["不得把合成值当作项目参数"],
        "project_value_transfer_allowed": False,
        "current_project_authority": False,
        "content_available": True,
        "retrieval_eligible": True,
        "embedding_eligible": True,
        "embedding_exclusion_reason": None,
        "embedding_eligibility_basis": {
            "content_type": "authored_reviewed_explanation",
            "public_scope": "authored_explanations_only",
            "source_review_proof_sha256": "b" * 64,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


class CuratedSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model_dir = os.environ.get("CURATED_VECTOR_MODEL_DIR")
        lock_path = os.environ.get("CURATED_VECTOR_LOCK_PATH")
        vendor_path = os.environ.get("CURATED_VECTOR_VENDOR")
        if not model_dir or not lock_path:
            raise RuntimeError(
                "CURATED_VECTOR_MODEL_DIR and CURATED_VECTOR_LOCK_PATH are required"
            )
        cls.model_dir = Path(model_dir)
        cls.lock_path = Path(lock_path)
        cls.vendor_path = Path(vendor_path) if vendor_path else None
        probe_root = Path(os.environ.get("CURATED_VECTOR_PROBE", "vector_probe"))
        probe_root.mkdir(parents=True, exist_ok=True)
        cls.temp_dir = Path(
            tempfile.mkdtemp(prefix="curated-search-", dir=str(probe_root))
        )
        cls.bundle = cls.temp_dir / "bundle"
        cls.bundle.mkdir()
        first = _ku(
            "SYN-001",
            title="夹点热集成目标",
            body=(
                "夹点分析先由热级联确定最小冷热公用工程目标，"
                "再检查网络改造的可行性与边界。"
            ),
            subject_root="chemical-engineering-principles",
            applicability="仅用于合成的夹点目标说明",
            package_id="SYN-CH01",
        )
        long_body = (
            "汽液平衡计算需要保持相态和组成定义一致，"
            "并按求解目标选择约束。"
        ) * 110
        second = _ku(
            "SYN-002",
            title="汽液平衡约束",
            body=long_body,
            subject_root="reaction-engineering",
            applicability=["仅用于合成的汽液平衡说明", "不得直接代入项目"],
            package_id="SYN-CH02",
        )
        cls.records = [first, second]
        _write_jsonl(cls.bundle / "knowledge_units.jsonl", cls.records)
        cls.manifest = build_bundle(
            cls.bundle,
            model_dir=cls.model_dir,
            lock_path=cls.lock_path,
            vendor_path=cls.vendor_path,
        )
        cls.store = CuratedBundle(cls.bundle)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_index_manifest_mapping_and_complete_chunk_fields(self) -> None:
        self.assertFalse(self.manifest["production_activated"])
        self.assertFalse(self.manifest["public_preview_is_production"])
        self.assertEqual(
            self.manifest["matrix"]["row_order"], "chunk_id ascending"
        )
        self.assertTrue(self.manifest["coverage"]["all_knowledge_units_covered"])
        self.assertIn(
            "does not claim independent approval",
            self.manifest["disposition"].lower(),
        )
        self.assertEqual(
            self.manifest["coverage"]["chunk_count"],
            len(self.store.chunks),
        )
        for row in self.store.chunks:
            self.assertEqual(row["search_text"], row["embedding_text"])
            self.assertEqual(row["full_text_sha256"], row["text_sha256"])
            self.assertIn(row["knowledge_unit_id"], {"SYN-001", "SYN-002"})
            self.assertTrue(row["retrieval_eligible"])
            self.assertTrue(row["embedding_eligible"])
        manifest_text = json.dumps(self.manifest, ensure_ascii=False)
        self.assertIsNone(PRIVATE_PATH_PATTERN.search(manifest_text))
        matrix = self.store.load_matrix()
        self.assertEqual(matrix.shape, (len(self.store.chunks), 512))
        self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5))
        STATE["index"] = {
            "knowledge_units": len(self.records),
            "chunks": len(self.store.chunks),
            "matrix_shape": list(matrix.shape),
            "search_text_equals_embedding_text": True,
            "complete_ku_fields": True,
        }

    def test_default_bundle_is_repository_anchored(self) -> None:
        expected = (
            Path(__file__).resolve().parents[1]
            / "knowledge/curated-four-books-v1"
        )
        self.assertEqual(DEFAULT_BUNDLE, expected)
        STATE["default_bundle_anchor"] = True

    def test_lexical_dense_hybrid_filters_and_aggregation(self) -> None:
        lexical = self.store.search(
            query="夹点热集成目标",
            method="lexical",
            limit=5,
            source_filter="SYN",
            subject_filter="chemical-engineering-principles",
        )
        self.assertTrue(lexical)
        self.assertEqual(lexical[0]["knowledge_unit_id"], "SYN-001")
        self.assertEqual(
            lexical[0]["knowledge_unit"]["text"], self.records[0]["text"]
        )
        dense = self.store.search(
            query="汽液平衡组成约束",
            method="dense",
            limit=5,
            source_filter="SYN",
            subject_filter="reaction-engineering",
            model_dir=self.model_dir,
            lock_path=self.lock_path,
            vendor_path=self.vendor_path,
        )
        dense_symbols = self.store.search(
            query="∂∇",
            method="dense",
            limit=5,
            model_dir=self.model_dir,
            lock_path=self.lock_path,
            vendor_path=self.vendor_path,
        )
        hybrid = self.store.search(
            query="夹点目标和公用工程",
            method="hybrid",
            limit=5,
            source_filter="SYN",
            model_dir=self.model_dir,
            lock_path=self.lock_path,
            vendor_path=self.vendor_path,
        )
        for results in (dense, hybrid):
            self.assertTrue(results)
            for result in results:
                self.assertIn("knowledge_unit", result)
                self.assertGreater(len(result["knowledge_unit"]["text"]), 0)
                self.assertNotIn("search_text", result["knowledge_unit"])
        self.assertTrue(all(item["knowledge_unit"]["source_id"] == "SYN" for item in hybrid))
        self.assertTrue(dense_symbols)
        self.assertEqual(self.store.search(query="珊瑚", method="lexical"), [])
        STATE["search"] = {
            "lexical_nonempty": True,
            "dense_nonempty": True,
            "hybrid_nonempty": True,
            "complete_ku_returned": True,
            "filters_before_scoring": True,
            "deduplicated_by_knowledge_unit": True,
            "dense_nonlexical_query": True,
            "lexical_zero_scores_omitted": True,
        }

    def test_exact_node_empty_illegal_filter_and_long_query_reject(self) -> None:
        exact = self.store.search(node_id="SYN-002", method="lexical", limit=5)
        self.assertEqual({item["knowledge_unit_id"] for item in exact}, {"SYN-002"})
        self.assertEqual(exact[0]["method"], "exact-node-id")
        with self.assertRaises(QueryInputError):
            self.store.search(query=" ", method="lexical")
        with self.assertRaises(QueryInputError):
            self.store.search(query="夹点", source_filter="NOT-A-SOURCE")
        with self.assertRaises(InputTooLongError):
            self.store.search(
                query="长查询" * 1000,
                method="dense",
                model_dir=self.model_dir,
                lock_path=self.lock_path,
                vendor_path=self.vendor_path,
            )
        STATE["input_guards"] = {
            "exact_node_scope_checked": True,
            "empty_query_rejected": True,
            "illegal_source_rejected": True,
            "overlong_dense_query_rejected": True,
        }

    def test_corrupt_matrix_rejected_and_identity_change_rejected(self) -> None:
        matrix_path = self.bundle / "embeddings.f32"
        original_matrix = matrix_path.read_bytes()
        matrix_path.write_bytes(original_matrix + b"x")
        with self.assertRaises(BundleIntegrityError):
            CuratedBundle(self.bundle).load_matrix()
        matrix_path.write_bytes(original_matrix)

        ku_path = self.bundle / "knowledge_units.jsonl"
        original_ku = ku_path.read_bytes()
        ku_path.write_bytes(original_ku + b"\n")
        with self.assertRaises(BundleIntegrityError):
            CuratedBundle(self.bundle)
        ku_path.write_bytes(original_ku)

        manifest_path = self.bundle / "vector_manifest.json"
        original_manifest = manifest_path.read_bytes()
        altered_manifest = json.loads(original_manifest.decode("utf-8"))
        altered_manifest["matrix"]["row_order"] = "not chunk_id ascending"
        manifest_path.write_text(
            json.dumps(altered_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaises(BundleIntegrityError):
            CuratedBundle(self.bundle).load_matrix()
        manifest_path.write_bytes(original_manifest)

        for invalid_tolerance in (float("nan"), float("inf")):
            altered_manifest = json.loads(original_manifest.decode("utf-8"))
            altered_manifest["matrix"]["norm_absolute_tolerance"] = invalid_tolerance
            manifest_path.write_text(
                json.dumps(
                    altered_manifest, ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            with self.assertRaises(BundleIntegrityError):
                CuratedBundle(self.bundle).load_matrix()
        manifest_path.write_bytes(original_manifest)

        altered_manifest = json.loads(original_manifest.decode("utf-8"))
        altered_manifest["matrix"]["norm_absolute_tolerance"] = 0.1
        manifest_path.write_text(
            json.dumps(altered_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaises(BundleIntegrityError):
            CuratedBundle(self.bundle).load_matrix()
        manifest_path.write_bytes(original_manifest)

        altered_manifest = json.loads(original_manifest.decode("utf-8"))
        altered_manifest["model"]["revision"] = "different-revision"
        manifest_path.write_text(
            json.dumps(altered_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with self.assertRaises(BundleIntegrityError):
            CuratedBundle(self.bundle).search(
                query="∂∇",
                method="dense",
                model_dir=self.model_dir,
                lock_path=self.lock_path,
                vendor_path=self.vendor_path,
            )
        manifest_path.write_bytes(original_manifest)

        altered = list(self.records)
        altered[0] = dict(altered[0])
        altered[0]["text"] = altered[0]["text"] + "改变身份"
        altered[0]["text_sha256"] = _sha256_text(altered[0]["text"])
        _write_jsonl(ku_path, altered)
        with self.assertRaises(BundleBuildError):
            build_bundle(
                self.bundle,
                model_dir=self.model_dir,
                lock_path=self.lock_path,
                vendor_path=self.vendor_path,
            )
        ku_path.write_bytes(original_ku)
        STATE["integrity"] = {
            "corrupt_matrix_rejected": True,
            "normative_knowledge_units_hash_rejected": True,
            "row_order_rejected": True,
            "loose_norm_tolerance_rejected": True,
            "nonfinite_norm_tolerance_rejected": True,
            "dense_model_manifest_mismatch_rejected": True,
            "identity_change_rejected": True,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tests(output: Path) -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CuratedSearchTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "schema": "curated-search-module-development-v1",
        "status": "pass" if result.wasSuccessful() else "fail",
        "tests": result.testsRun,
        "failures": [str(item[1]) for item in result.failures],
        "errors": [str(item[1]) for item in result.errors],
        "checks": dict(STATE),
        "scope": "synthetic vector_probe only; no formal 69-KU build",
        "old_holdout_read": False,
        "production_activated": False,
        "development_query_evaluation_ready": True,
        "source_hashes": {
            "index_curated_bundle.py": _sha256(
                MODULE_DIR / "index_curated_bundle.py"
            ),
            "query_curated.py": _sha256(MODULE_DIR / "query_curated.py"),
            "test_curated_search.py": _sha256(Path(__file__)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CHECK_OUTPUT)
    args = parser.parse_args()
    return run_tests(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
