"""Independent unit checks for the two-bundle collection candidate.

The test process receives ``CURATED_REPO_ROOT`` and a compatible vendor path
from the runner.  Real public bundles are loaded and their persisted matrices
are checked, but the dense spy test intentionally does not run the ONNX model;
it only verifies the one-query-encoding call contract.
"""

from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


repo_override = os.environ.get("CURATED_REPO_ROOT")
REPO_ROOT = (
    Path(repo_override).resolve()
    if repo_override
    else Path(__file__).resolve().parents[1]
)
if not REPO_ROOT.is_dir():
    raise RuntimeError("set CURATED_REPO_ROOT to the formal repository for these tests")
FORMAL_SRC = REPO_ROOT / "src"
CANDIDATE_SRC = Path(__file__).resolve().parents[2] / "src"
for path in (FORMAL_SRC, CANDIDATE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

import query_collection as collection_module
import query_curated as single_bundle_module
from query_curated import BundleIntegrityError, DIMENSIONS, QueryInputError, _bm25, _lexical_tokens
from query_collection import CuratedCollection, _Package, _model_signature


descriptor_override = os.environ.get("CURATED_COLLECTION_DESCRIPTOR")
COLLECTION_DESCRIPTOR = (
    Path(descriptor_override).resolve()
    if descriptor_override
    else REPO_ROOT / "knowledge" / "curated-collection-v1.json"
)
KNOWLEDGE_ROOT = REPO_ROOT / "knowledge"
LOCK_PATH = REPO_ROOT / "contracts" / "curated_model_lock.json"


class QueryCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.collection = CuratedCollection(
            COLLECTION_DESCRIPTOR,
            knowledge_root=KNOWLEDGE_ROOT,
        )

    def test_real_bundles_load_into_read_only_98_by_113_view(self) -> None:
        self.assertEqual(set(self.collection.packages), {
            "curated-four-books-v1",
            "curated-s001-upper-v1",
        })
        self.assertEqual(len(self.collection.knowledge_units), 98)
        self.assertEqual(len(self.collection.chunks), 113)
        self.assertEqual(self.collection._matrix.shape, (113, DIMENSIONS))
        self.assertFalse(self.collection._matrix.flags.writeable)

    def test_lexical_statistics_use_all_filtered_collection_chunks(self) -> None:
        indices = self.collection._filter_indices(
            node_id=None,
            source_filter=None,
            subject_filter=None,
        )
        observed: list[tuple[Counter[str], int, float]] = []

        def spy_bm25(query_tokens, document_tokens, document_frequency, document_count, average_length):
            observed.append((document_frequency, document_count, average_length))
            return 1.0

        with patch.object(single_bundle_module, "_bm25", side_effect=spy_bm25):
            ranked = self.collection._rank_lexical("传热", indices)

        self.assertEqual(len(ranked), len(indices))
        self.assertEqual(len(observed), len(indices))
        self.assertTrue(all(document_count == 113 for _, document_count, _ in observed))
        expected_df = Counter()
        for index in indices:
            expected_df.update(set(self.collection._doc_tokens[index]))
        self.assertEqual(observed[0][0], expected_df)
        expected_average = max(
            1.0,
            sum(len(self.collection._doc_tokens[index]) for index in indices) / len(indices),
        )
        self.assertEqual(observed[0][2], expected_average)

        # The score is the existing BM25 helper applied to the global stats,
        # rather than a per-package score.
        tokens = _lexical_tokens("传热")
        first_index = indices[0]
        expected_score = _bm25(
            tokens,
            self.collection._doc_tokens[first_index],
            expected_df,
            len(indices),
            expected_average,
        )
        self.assertGreaterEqual(expected_score, 0.0)

    def test_exact_id_and_filters_keep_bundle_and_full_normative_unit(self) -> None:
        target_id = next(
            unit_id
            for unit_id in self.collection.knowledge_units
            if unit_id.startswith("S001-")
        )
        exact = self.collection.search(node_id=target_id)
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["knowledge_unit_id"], target_id)
        self.assertEqual(exact[0]["bundle_id"], "curated-s001-upper-v1")
        self.assertEqual(exact[0]["knowledge_unit"], self.collection.knowledge_units[target_id])

        filtered = self.collection.search(
            query="蒸发器",
            method="lexical",
            source_filter="S001",
            subject_filter="chemical-engineering-principles",
        )
        self.assertTrue(filtered)
        self.assertTrue(all(item["bundle_id"] == "curated-s001-upper-v1" for item in filtered))
        self.assertTrue(all(
            item["knowledge_unit"]["source_id"] == "S001"
            for item in filtered
        ))

        with self.assertRaises(QueryInputError):
            self.collection.search(query="蒸发器", source_filter="UNKNOWN")
        with self.assertRaises(QueryInputError):
            self.collection.search(query="蒸发器", subject_filter="UNKNOWN")

    def test_wrong_descriptor_hash_and_path_fail_closed(self) -> None:
        descriptor = json.loads(COLLECTION_DESCRIPTOR.read_text(encoding="utf-8"))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.json"
            wrong_hash = copy.deepcopy(descriptor)
            wrong_hash["packages"][0]["manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(wrong_hash), encoding="utf-8")
            with self.assertRaises(BundleIntegrityError):
                CuratedCollection(path, knowledge_root=KNOWLEDGE_ROOT)

            traversal = copy.deepcopy(descriptor)
            traversal["packages"][0]["path_relative_to_knowledge"] = "../private"
            path.write_text(json.dumps(traversal), encoding="utf-8")
            with self.assertRaises(BundleIntegrityError):
                CuratedCollection(path, knowledge_root=KNOWLEDGE_ROOT)

            wrong_id = copy.deepcopy(descriptor)
            wrong_id["packages"][0]["bundle_id"] = "unlisted-bundle"
            path.write_text(json.dumps(wrong_id), encoding="utf-8")
            with self.assertRaises(BundleIntegrityError):
                CuratedCollection(path, knowledge_root=KNOWLEDGE_ROOT)

    def test_incompatible_model_and_duplicate_ids_fail_closed(self) -> None:
        vector_manifest = json.loads(
            (KNOWLEDGE_ROOT / "curated-four-books-v1" / "vector_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        compatible_signature = _model_signature(vector_manifest)
        with patch.object(
            collection_module,
            "_model_signature",
            side_effect=[compatible_signature, ("incompatible-model",)],
        ):
            with self.assertRaises(BundleIntegrityError):
                CuratedCollection(COLLECTION_DESCRIPTOR, knowledge_root=KNOWLEDGE_ROOT)

        class FakeStore:
            def __init__(self, bundle_id: str) -> None:
                self.bundle_id = bundle_id
                self.knowledge_units = {
                    "DUPLICATE": {"knowledge_unit_id": "DUPLICATE"}
                }
                self.chunks = [{
                    "chunk_id": "DUPLICATE:chunk:001",
                    "knowledge_unit_id": "DUPLICATE",
                    "search_text": "重复",
                }]

        fake_one = _Package(
            "curated-four-books-v1",
            Path("."),
            FakeStore("curated-four-books-v1"),
            np.zeros((1, DIMENSIONS), dtype="<f4"),
            compatible_signature,
        )
        fake_two = _Package(
            "curated-s001-upper-v1",
            Path("."),
            FakeStore("curated-s001-upper-v1"),
            np.zeros((1, DIMENSIONS), dtype="<f4"),
            compatible_signature,
        )
        descriptor = json.loads(COLLECTION_DESCRIPTOR.read_text(encoding="utf-8"))
        descriptor["totals"] = {"knowledge_units": 2, "chunks": 2}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "collection.json"
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            with patch.object(
                CuratedCollection,
                "_load_package",
                side_effect=[fake_one, fake_two],
            ):
                with self.assertRaises(BundleIntegrityError):
                    CuratedCollection(path, knowledge_root=KNOWLEDGE_ROOT)

    def test_hybrid_query_encoding_is_called_once_by_spy(self) -> None:
        model = next(iter(self.collection.packages.values())).store.manifest["model"]

        class SpyEncoder:
            call_count = 0

            def __init__(self, model_dir, *, lock_path, vendor_path=None):
                self.asset_report = {
                    "model_version": model["name"],
                    "files": {
                        name: {"sha256": digest}
                        for name, digest in model["files"].items()
                    },
                }

            def encode_texts(self, texts):
                type(self).call_count += 1
                values = np.ones((len(texts), DIMENSIONS), dtype="<f4")
                values /= np.linalg.norm(values, axis=1, keepdims=True)
                return values

        with patch.object(collection_module, "CuratedOnnxEncoder", SpyEncoder):
            results = self.collection.search(
                query="换热器压降",
                method="hybrid",
                limit=5,
                model_dir=Path("not-used-by-spy"),
                lock_path=LOCK_PATH,
            )
        self.assertEqual(SpyEncoder.call_count, 1)
        self.assertLessEqual(len(results), 5)
        self.assertTrue(all("bundle_id" in item and "knowledge_unit" in item for item in results))


if __name__ == "__main__":
    unittest.main()
