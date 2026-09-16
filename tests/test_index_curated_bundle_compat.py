"""Regression tests for the explicit bundle-id compatibility candidate.

The candidate module is imported from the sibling ``src`` directory.  The
locked ``curated_vectors`` dependency is supplied by the test runner through
``PYTHONPATH``; this test file does not encode a machine-specific source path.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


CANDIDATE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(CANDIDATE_SRC))
import index_curated_bundle as candidate  # noqa: E402


def _record(
    *,
    bundle_id: str = candidate.DEFAULT_BUNDLE_ID,
    source_id: str = "SYN",
) -> dict[str, object]:
    body = "合成兼容测试正文。"
    return {
        "applicability": ["仅用于兼容性测试。"],
        "content_available": True,
        "current_project_authority": False,
        "embedding_eligibility_basis": {
            "content_type": "authored_reviewed_explanation",
            "public_scope": "candidate_test_only",
            "source_review_proof_sha256": "0" * 64,
        },
        "embedding_eligible": True,
        "embedding_exclusion_reason": None,
        "evidence_refs": ["evidence:synthetic-test"],
        "forbidden_transfer": ["合成测试记录不得作为项目输入。"],
        "knowledge_layer": "L1",
        "knowledge_unit_id": "SYN-001",
        "node_id": "SYN-001",
        "package_id": "SYN-PACK",
        "project_value_transfer_allowed": False,
        "retrieval_eligible": True,
        "source_chain_id": f"{bundle_id}:{source_id}",
        "source_id": source_id,
        "source_locator": "synthetic-test",
        "source_sha256": "0" * 64,
        "source_volume_id": "volume:syn:test",
        "subject_root": "chemical-engineering-principles",
        "text": body,
        "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "title": "合成兼容测试",
        "units_basis": "定性方法；无设计数值。",
    }


def _lock() -> dict[str, object]:
    return {
        "model_version": "BAAI/bge-small-zh-v1.5",
        "revision": candidate.MODEL_REVISION,
        "model_dir_sha256": {
            name: "0" * 64
            for name in (
                "config.json",
                "model_optimized.onnx",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            )
        },
    }


class _FakeEncoder:
    threads = 1
    batch_size = 1
    dependency_versions = {
        "tokenizers": "test",
        "onnxruntime": "test",
        "numpy": np.__version__,
    }

    def __init__(self, model_dir: object, *, lock_path: object, vendor_path: object = None):
        self.tokenizer = object()

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), candidate.DIMENSIONS), dtype="<f4")
        vectors[:, 0] = 1.0
        return vectors


def _fake_split(record: dict[str, object], tokenizer: object) -> list[dict[str, object]]:
    body = str(record["text"])
    return [
        {
            "body_start": 0,
            "body_end": len(body),
            "embedding_text": f"{record['title']}\n{body}\n合成兼容测试。",
            "segment_count": 1,
            "segment_index": 0,
            "token_count": 4,
        }
    ]


class CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        existing_path = os.environ.get("INDEX_COMPAT_EXISTING_KG")
        cls.existing_path = Path(existing_path) if existing_path else None

    def test_actual_default_69_records_validate(self) -> None:
        if self.existing_path is None:
            self.skipTest("pass --existing-kg to run the installed 69-record check")
        records = candidate._read_jsonl(self.existing_path)
        self.assertEqual(len(records), 69)
        self.assertEqual(
            Counter(record["source_id"] for record in records),
            Counter({"RE01": 14, "OC02": 13, "TH03": 25, "EN04": 17}),
        )
        for record in records:
            candidate._validate_ku(record)

    def test_explicit_batch_requires_exact_source_chain(self) -> None:
        batch = "curated-s001-upper-v1"
        record = _record(bundle_id=batch, source_id="S001")
        candidate._validate_ku(record, bundle_id=batch)
        with self.assertRaises(candidate.BundleBuildError):
            candidate._validate_ku(record, bundle_id="a/b")
        for invalid_chain in (
            "curated-four-books-v1:S001",
            f"{batch}:OTHER",
            f"{batch}:S001:extra",
            "S001",
        ):
            invalid = dict(record, source_chain_id=invalid_chain)
            with self.subTest(source_chain_id=invalid_chain):
                with self.assertRaises(candidate.BundleBuildError):
                    candidate._validate_ku(invalid, bundle_id=batch)

    def test_bundle_and_source_identifiers_reject_injection(self) -> None:
        for value in (
            "",
            " ",
            "a/b",
            r"a\b",
            "a:b",
            "a\nb",
            "../x",
            "/tmp",
            "a?b",
            None,
            "x" * 129,
        ):
            with self.subTest(bundle_id=value):
                with self.assertRaises(candidate.BundleBuildError):
                    candidate._validate_bundle_id(value)
        for value in ("", "a/b", "a:b", "a\nb", None):
            with self.subTest(source_id=value):
                with self.assertRaises(candidate.BundleBuildError):
                    candidate._validate_source_id(value)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-be-created"
            with self.assertRaises(candidate.BundleBuildError):
                candidate.build_bundle(
                    output,
                    model_dir=Path(temporary) / "model",
                    lock_path=Path(temporary) / "lock.json",
                    bundle_id="../invalid",
                )
            self.assertFalse(output.exists())

    def test_existing_eligibility_hash_and_path_gates_remain_strict(self) -> None:
        record = _record()
        private_path = chr(67) + ":" + "\\private\\source.pdf"
        invalid_cases = {
            "eligibility": dict(record, embedding_eligible=False),
            "text_hash": dict(record, text_sha256="0" * 64),
            "private_path": dict(record, source_locator=private_path),
            "project_authority": dict(record, current_project_authority=True),
        }
        for name, invalid in invalid_cases.items():
            with self.subTest(case=name):
                with self.assertRaises(candidate.BundleBuildError):
                    candidate._validate_ku(invalid)

    def test_build_accepts_explicit_batch_and_records_it_in_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _record(bundle_id="curated-s001-upper-v1", source_id="S001")
            output = root / "bundle"
            output.mkdir()
            (output / "knowledge_units.jsonl").write_text(
                json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(_lock()), encoding="utf-8")
            with patch.object(candidate, "CuratedOnnxEncoder", _FakeEncoder), patch.object(
                candidate, "split_record", _fake_split
            ):
                manifest = candidate.build_bundle(
                    output,
                    model_dir=root / "model-not-loaded",
                    lock_path=lock_path,
                    bundle_id="curated-s001-upper-v1",
                )
            self.assertEqual(manifest["input"]["bundle_id"], "curated-s001-upper-v1")
            self.assertEqual(manifest["coverage"]["knowledge_unit_count"], 1)
            self.assertTrue((output / "vector_manifest.json").is_file())
            self.assertFalse(manifest["production_activated"])

    def test_build_default_remains_usable_and_cli_forwards_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _record()
            output = root / "default-bundle"
            output.mkdir()
            (output / "knowledge_units.jsonl").write_text(
                json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(_lock()), encoding="utf-8")
            with patch.object(candidate, "CuratedOnnxEncoder", _FakeEncoder), patch.object(
                candidate, "split_record", _fake_split
            ):
                manifest = candidate.build_bundle(
                    output,
                    model_dir=root / "model-not-loaded",
                    lock_path=lock_path,
                )
            self.assertNotIn("bundle_id", manifest["input"])

        cli_manifest = {
            "scope": "test",
            "coverage": {"knowledge_unit_count": 0, "chunk_count": 0},
            "production_activated": False,
        }
        with patch.object(candidate, "build_bundle", return_value=cli_manifest) as build:
            with patch.object(
                sys,
                "argv",
                [
                    "index_curated_bundle.py",
                    "--bundle",
                    "bundle",
                    "--model-dir",
                    "model",
                    "--lock",
                    "lock",
                    "--bundle-id",
                    "curated-s001-upper-v1",
                ],
            ):
                self.assertEqual(candidate._cli(), 0)
        self.assertEqual(build.call_args.kwargs["bundle_id"], "curated-s001-upper-v1")

        with patch.object(candidate, "build_bundle", return_value=cli_manifest) as build:
            with patch.object(
                sys,
                "argv",
                [
                    "index_curated_bundle.py",
                    "--bundle",
                    "bundle",
                    "--model-dir",
                    "model",
                    "--lock",
                    "lock",
                ],
            ):
                self.assertEqual(candidate._cli(), 0)
        self.assertEqual(build.call_args.kwargs["bundle_id"], candidate.DEFAULT_BUNDLE_ID)

    def test_legacy_manifest_without_batch_defaults_only_to_old_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "bundle"
            output.mkdir()
            (output / "knowledge_units.jsonl").write_text(
                json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8"
            )
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(_lock()), encoding="utf-8")
            with patch.object(candidate, "CuratedOnnxEncoder", _FakeEncoder), patch.object(
                candidate, "split_record", _fake_split
            ):
                candidate.build_bundle(
                    output,
                    model_dir=root / "model-not-loaded",
                    lock_path=lock_path,
                )

            manifest_path = output / "vector_manifest.json"
            legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_path.write_text(
                json.dumps(legacy_manifest, ensure_ascii=False), encoding="utf-8"
            )
            with patch.object(candidate, "CuratedOnnxEncoder", _FakeEncoder), patch.object(
                candidate, "split_record", _fake_split
            ):
                candidate.build_bundle(
                    output,
                    model_dir=root / "model-not-loaded",
                    lock_path=lock_path,
                )
            explicit_source = _record(
                bundle_id="curated-s001-upper-v1", source_id="S001"
            )
            (output / "knowledge_units.jsonl").write_text(
                json.dumps(explicit_source, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                candidate.BundleBuildError, "existing bundle batch differs"
            ):
                candidate.build_bundle(
                    output,
                    model_dir=root / "model-not-loaded",
                    lock_path=lock_path,
                    bundle_id="curated-s001-upper-v1",
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-kg", type=Path)
    args, _ = parser.parse_known_args()
    if args.existing_kg is not None:
        os.environ["INDEX_COMPAT_EXISTING_KG"] = str(args.existing_kg)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(int(main()))
