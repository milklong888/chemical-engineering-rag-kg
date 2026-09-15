"""Synthetic contract tests for validate_curated_bundle.py.

The fixture is intentionally small in content but complete in cardinality: it
has the frozen four sources and 69 KU identities without using any private
source data.  It exercises the structural gates before a real A/C bundle is
available.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_curated_bundle import validate_bundle  # noqa: E402


SOURCE_COUNTS = {"RE01": 14, "OC02": 13, "TH03": 25, "EN04": 17}
ROOTS = {
    "RE01": "reaction-engineering",
    "OC02": "chemical-engineering-principles",
    "TH03": "chemical-engineering-principles",
    "EN04": "chemical-engineering-principles",
}


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class SyntheticBundle:
    def __init__(self, root: Path, with_vectors: bool = False) -> None:
        self.root = root
        self.chapter_map_path = root.parent / "chapter_mapping.json"
        self.with_vectors = with_vectors
        self.kus: list[dict[str, object]] = []
        self.evidence: list[dict[str, object]] = []
        self.nodes: list[dict[str, object]] = []
        self.edges: list[dict[str, object]] = []
        self.external: list[dict[str, object]] = []
        self.map_records: list[dict[str, object]] = []
        self.map_volumes: list[dict[str, object]] = []
        self._build()

    def _build(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        source_sha = {source_id: sha_text(f"synthetic-source-{source_id}") for source_id in SOURCE_COUNTS}
        volume_ids = {
            source_id: f"volume:{source_id.lower()}:{ROOTS[source_id]}:synthetic"
            for source_id in SOURCE_COUNTS
        }
        for source_id, count in SOURCE_COUNTS.items():
            self.map_volumes.append(
                {
                    "source_id": source_id,
                    "source_volume_id": volume_ids[source_id],
                    "label": f"Synthetic {source_id}",
                    "kg_subject_root": ROOTS[source_id],
                    "source_sha256": source_sha[source_id],
                    "record_count": count,
                    "chapter_basis_file": f"synthetic/{source_id}/chapters.json",
                    "chapter_basis": "synthetic fixture",
                }
            )

        # The first source contains two ordinary chapters and one support group;
        # this verifies that the support-group context rule is exercised.
        for source_id, count in SOURCE_COUNTS.items():
            for number in range(1, count + 1):
                node_id = f"{source_id}-K{number:02d}"
                title = f"{source_id} synthetic topic {number}"
                body = f"Unique reviewed explanation for {source_id} knowledge unit {number}."
                if source_id == "RE01" and number == 1:
                    unit = {
                        "id": "RE01-SUPPORT-1",
                        "type": "cross_chapter_support_unit",
                        "label": "Synthetic support group",
                        "chapter_refs": [
                            {"id": "RE01-CH01", "number": 1, "label": "Synthetic chapter one"},
                            {"id": "RE01-CH02", "number": 2, "label": "Synthetic chapter two"},
                        ],
                    }
                elif source_id == "RE01" and number == 2:
                    unit = {"id": "RE01-CH02", "type": "chapter", "label": "Synthetic chapter two", "chapter_refs": [{"id": "RE01-CH02", "number": 2, "label": "Synthetic chapter two"}]}
                else:
                    unit = {"id": f"{source_id}-CH01", "type": "chapter", "label": "Synthetic chapter one", "chapter_refs": [{"id": f"{source_id}-CH01", "number": 1, "label": "Synthetic chapter one"}]}
                record = {
                    "source_id": source_id,
                    "kg_subject_root": ROOTS[source_id],
                    "source_volume_id": volume_ids[source_id],
                    "original_node_id": node_id,
                    "local_id": f"K{number:02d}",
                    "title": title,
                    "body": body,
                    "body_sha256": sha_text(body),
                    "source_sha256": source_sha[source_id],
                    "source_page_range": "PDF physical pages 1-2",
                    "source_physical_page_intervals": [{"start": 1, "end": 2}],
                    "chapter_or_support_unit": unit,
                    "legacy_related_node_ids": [],
                    "direct_cross_source_relations": [],
                    "evidence": {"source_record_file": f"synthetic/source_records/{source_id}/{node_id}.json"},
                }
                self.map_records.append(record)
                proof = sha_text(f"review-proof-{source_id}")
                evidence_id = f"evidence:{node_id}:source-locator"
                ku = {
                    "node_id": node_id,
                    "knowledge_unit_id": node_id,
                    "title": title,
                    "text": body,
                    "text_sha256": sha_text(body),
                    "applicability": ["Use only with the stated synthetic basis."],
                    "units_basis": "as stated in the source explanation",
                    "knowledge_layer": "L2",
                    "source_id": source_id,
                    "source_sha256": source_sha[source_id],
                    "source_volume_id": volume_ids[source_id],
                    "subject_root": ROOTS[source_id],
                    "package_id": f"context:{source_id}:{unit['id']}",
                    "source_chain_id": f"curated-four-books-v1:{source_id}",
                    "source_locator": "PDF physical pages 1-2",
                    "evidence_refs": [evidence_id],
                    "forbidden_transfer": ["Do not treat synthetic values as project inputs."],
                    "project_value_transfer_allowed": False,
                    "current_project_authority": False,
                    "content_available": True,
                    "retrieval_eligible": True,
                    "embedding_eligible": True,
                    "embedding_exclusion_reason": None,
                    "embedding_eligibility_basis": {
                        "content_type": "authored_reviewed_explanation",
                        "source_review_proof_sha256": proof,
                        "public_scope": "authored_explanations_only",
                    },
                }
                evidence = {
                    "evidence_id": evidence_id,
                    "type": "source_document_locator",
                    "source_id": source_id,
                    "source_sha256": source_sha[source_id],
                    "pdf_pages": [{"start": 1, "end": 2}],
                    "locator": "PDF physical pages 1-2",
                    "reviewed_text_sha256": sha_text(body),
                    "review_proof_sha256": proof,
                    "page_media_sha256": None,
                    "public_source_payload": False,
                }
                self.kus.append(ku)
                self.evidence.append(evidence)

        source_manifest = {
            "schema_version": "curated-selected-knowledge-1.0",
            "scope": "authored-selected-knowledge-preview",
            "sources": [],
        }
        for source_id, count in SOURCE_COUNTS.items():
            source_manifest["sources"].append(
                {
                    "source_id": source_id,
                    "bibliography": {"title": f"Synthetic {source_id}"},
                    "source_sha256": source_sha[source_id],
                    "bytes": 1000,
                    "total_pdf_pages": 100,
                    "is_encrypted": False,
                    "source_volume_id": volume_ids[source_id],
                    "subject_root": ROOTS[source_id],
                    "selected_knowledge_units_count": count,
                    "source_payload_public": False,
                    "full_source_digitization_complete": False,
                    "review_provenance": {
                        "producer_id": "synthetic-producer",
                        "reviewer_id": "synthetic-reviewer",
                        "content_review_file_sha256": sha_text(f"content-review-{source_id}"),
                        "candidate_manifest_sha256": sha_text(f"candidate-manifest-{source_id}"),
                        "locked_candidate_part_hashes": {"synthetic": sha_text(f"candidate-proof-{source_id}")},
                        "agent_review": True,
                        "human_review": False,
                    },
                }
            )
        write_json(self.root / "source_manifest.json", source_manifest)
        write_jsonl(self.root / "knowledge_units.jsonl", self.kus)
        write_jsonl(self.root / "evidence_registry.jsonl", self.evidence)

        root_labels = {
            "chemical-engineering-principles": "化学工程原理",
            "reaction-engineering": "反应工程",
        }
        self.nodes = [
            {"node_id": root, "node_type": "subject_root", "label": root_labels[root]}
            for root in sorted({"chemical-engineering-principles", "reaction-engineering"})
        ]
        for volume in self.map_volumes:
            self.nodes.append(
                {
                    "node_id": volume["source_volume_id"],
                    "node_type": "source_volume",
                    "label": volume["label"],
                    "subject_root": volume["kg_subject_root"],
                    "source_id": volume["source_id"],
                    "source_volume_id": volume["source_volume_id"],
                }
            )
        context_keys: set[tuple[str, str, str]] = set()
        for record in self.map_records:
            unit = record["chapter_or_support_unit"]
            context_keys.add((record["source_id"], unit["id"], unit["type"]))
        for source_id, unit_id, unit_type in sorted(context_keys):
            context_node_id = f"context:{source_id}:{unit_id}"
            context_node = {
                "node_id": context_node_id,
                "node_type": unit_type,
                "label": unit_id,
                "subject_root": ROOTS[source_id],
                "source_id": source_id,
                "source_volume_id": volume_ids[source_id],
            }
            if unit_type == "chapter":
                context_node["chapter_id"] = unit_id
                context_node["chapter_number"] = 1 if unit_id.endswith("CH01") else 2
            else:
                context_node["support_unit_id"] = unit_id
                unit = next(r["chapter_or_support_unit"] for r in self.map_records if r["chapter_or_support_unit"]["id"] == unit_id and r["source_id"] == source_id)
                context_node["chapter_refs"] = [ref["id"] for ref in unit["chapter_refs"]]
            context_node["group_id"] = context_node_id
            self.nodes.append(context_node)
        for ku in self.kus:
            self.nodes.append(
                {
                    "node_id": ku["node_id"],
                    "node_type": "knowledge_unit",
                    "label": ku["title"],
                    "subject_root": ku["subject_root"],
                    "source_id": ku["source_id"],
                    "source_volume_id": ku["source_volume_id"],
                    "knowledge_unit_id": ku["knowledge_unit_id"],
                }
            )
        for evidence in self.evidence:
            source_id = evidence["source_id"]
            self.nodes.append(
                {
                    "node_id": evidence["evidence_id"],
                    "node_type": "evidence",
                    "label": evidence["locator"],
                    "subject_root": ROOTS[source_id],
                    "source_id": source_id,
                    "source_volume_id": volume_ids[source_id],
                }
            )

        edge_number = 0

        def add_edge(source: str, target: str, relation: str) -> None:
            nonlocal edge_number
            edge_number += 1
            self.edges.append(
                {
                    "edge_id": f"edge:{edge_number:04d}",
                    "source_node_id": source,
                    "target_node_id": target,
                    "relation": relation,
                }
            )

        for volume in self.map_volumes:
            add_edge(volume["kg_subject_root"], volume["source_volume_id"], "contains")
        for source_id, unit_id, unit_type in sorted(context_keys):
            volume_id = volume_ids[source_id]
            context_node_id = f"context:{source_id}:{unit_id}"
            add_edge(volume_id, context_node_id, "contains")
        for record in self.map_records:
            unit = record["chapter_or_support_unit"]
            context_node_id = f"context:{record['source_id']}:{unit['id']}"
            add_edge(context_node_id, record["original_node_id"], "contains")
        for evidence in self.evidence:
            node_id = evidence["evidence_id"].split(":", 2)[1]
            add_edge(node_id, evidence["evidence_id"], "supported_by")
        support_context = {("RE01-SUPPORT-1", "RE01-CH01"), ("RE01-SUPPORT-1", "RE01-CH02")}
        for source_support, chapter_id in sorted(support_context):
            add_edge(f"context:RE01:{source_support}", f"context:RE01:{chapter_id}", "has_chapter_context")
        write_jsonl(self.root / "kg_nodes.jsonl", self.nodes)
        write_jsonl(self.root / "kg_edges.jsonl", self.edges)
        write_jsonl(self.root / "external_references.jsonl", self.external)

        map_data = {
            "mapping_version": "synthetic-test-1",
            "scope": "synthetic",
            "source_volume_count": 4,
            "record_count": 69,
            "source_volumes": self.map_volumes,
            "records": self.map_records,
            "direct_cross_source_relations": [],
        }
        write_json(self.chapter_map_path, map_data)

        if self.with_vectors:
            self._write_vectors()

    def _write_vectors(self) -> None:
        chunks: list[dict[str, object]] = []
        mappings: list[dict[str, object]] = []
        for ku in self.kus:
            body = ku["text"]
            title = ku["title"]
            applicability = ku["applicability"]
            segment = body
            search_text = f"{title}\n{segment}\n{'\n'.join(applicability)}"
            chunk_id = f"chunk:{ku['node_id']}"
            chunk = dict(ku)
            chunk.update(
                {
                    "chunk_id": chunk_id,
                    "search_text": search_text,
                    "embedding_text": search_text,
                    "body_start": 0,
                    "body_end": len(body),
                    "full_text_sha256": ku["text_sha256"],
                    "segment_index": 0,
                    "segment_count": 1,
                }
            )
            chunks.append(chunk)
            mappings.append({"chunk_id": chunk_id, "knowledge_unit_id": ku["node_id"]})
        write_jsonl(self.root / "rag_chunks.jsonl", chunks)
        write_jsonl(self.root / "chunk_to_kg.jsonl", mappings)
        row = struct.pack("<512f", 1.0, *([0.0] * 511))
        matrix = row * len(chunks)
        (self.root / "embeddings.f32").write_bytes(matrix)
        entries = []
        for index, chunk in enumerate(sorted(chunks, key=lambda item: item["chunk_id"])):
            entries.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "input_sha256": sha_text(chunk["search_text"]),
                    "vector_sha256": sha_bytes(row),
                }
            )
        write_json(
            self.root / "vector_manifest.json",
            {
                "matrix": {
                    "file": "embeddings.f32",
                    "rows": len(chunks),
                    "dimensions": 512,
                    "dtype": "<f4",
                    "row_order": "chunk_id ascending",
                    "sha256": sha_bytes(matrix),
                },
                "entries": entries,
            },
        )


class CuratedBundleValidatorTests(unittest.TestCase):
    def make(self, **kwargs: object) -> tuple[tempfile.TemporaryDirectory[str], SyntheticBundle]:
        temp = tempfile.TemporaryDirectory()
        bundle = SyntheticBundle(Path(temp.name) / "bundle", **kwargs)
        return temp, bundle

    def result(self, bundle: SyntheticBundle, **kwargs: object) -> dict[str, object]:
        return validate_bundle(bundle.root, chapter_map=bundle.chapter_map_path, **kwargs)

    def test_valid_synthetic_bundle(self) -> None:
        temp, bundle = self.make()
        try:
            report = self.result(bundle)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertEqual(report["counts"]["knowledge_units"], 69)
        finally:
            temp.cleanup()

    def test_public_structure_does_not_require_private_chapter_map(self) -> None:
        temp, bundle = self.make()
        try:
            report = validate_bundle(bundle.root)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertFalse(report["evidence_mode"]["chapter_map"])
        finally:
            temp.cleanup()

    def test_duplicate_id_rejected(self) -> None:
        temp, bundle = self.make()
        try:
            rows = json.loads("[" + ",".join((bundle.root / "knowledge_units.jsonl").read_text(encoding="utf-8").splitlines()) + "]")
            rows[1]["node_id"] = rows[0]["node_id"]
            write_jsonl(bundle.root / "knowledge_units.jsonl", rows)
            report = self.result(bundle)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("duplicate_ku_id", {item["code"] for item in report["errors"]})
        finally:
            temp.cleanup()

    def test_body_hash_rejected(self) -> None:
        temp, bundle = self.make()
        try:
            rows = [json.loads(line) for line in (bundle.root / "knowledge_units.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["text_sha256"] = "0" * 64
            write_jsonl(bundle.root / "knowledge_units.jsonl", rows)
            report = self.result(bundle)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("ku_text_hash", {item["code"] for item in report["errors"]})
        finally:
            temp.cleanup()

    def test_missing_qualification_rejected(self) -> None:
        temp, bundle = self.make()
        try:
            rows = [json.loads(line) for line in (bundle.root / "knowledge_units.jsonl").read_text(encoding="utf-8").splitlines()]
            rows[0]["retrieval_eligible"] = False
            write_jsonl(bundle.root / "knowledge_units.jsonl", rows)
            report = self.result(bundle)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("ku_eligibility", {item["code"] for item in report["errors"]})
        finally:
            temp.cleanup()

    def test_dangling_edge_and_wrong_source_rejected(self) -> None:
        temp, bundle = self.make()
        try:
            edges = [json.loads(line) for line in (bundle.root / "kg_edges.jsonl").read_text(encoding="utf-8").splitlines()]
            edges.append({"edge_id": "bad-edge", "source_node_id": "missing", "target_node_id": "missing", "relation": "contains"})
            write_jsonl(bundle.root / "kg_edges.jsonl", edges)
            nodes = [json.loads(line) for line in (bundle.root / "kg_nodes.jsonl").read_text(encoding="utf-8").splitlines()]
            ku_node = next(node for node in nodes if node.get("node_type") == "knowledge_unit")
            ku_node["source_id"] = "OC02" if ku_node["source_id"] != "OC02" else "RE01"
            write_jsonl(bundle.root / "kg_nodes.jsonl", nodes)
            report = self.result(bundle)
            self.assertEqual(report["status"], "FAIL")
            codes = {item["code"] for item in report["errors"]}
            self.assertIn("dangling_edge", codes)
        finally:
            temp.cleanup()

    def test_private_absolute_path_rejected(self) -> None:
        temp, bundle = self.make()
        try:
            manifest = json.loads((bundle.root / "source_manifest.json").read_text(encoding="utf-8"))
            manifest["sources"][0]["bibliography"]["note"] = "C:" + "\\private\\source.pdf"
            write_json(bundle.root / "source_manifest.json", manifest)
            report = self.result(bundle)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("private_absolute_path", {item["code"] for item in report["errors"]})
        finally:
            temp.cleanup()

    def test_public_https_url_is_not_a_private_path(self) -> None:
        temp, bundle = self.make()
        try:
            manifest = json.loads((bundle.root / "source_manifest.json").read_text(encoding="utf-8"))
            manifest["sources"][0]["bibliography"]["public_reference"] = "https://example.org/reference"
            write_json(bundle.root / "source_manifest.json", manifest)
            report = self.result(bundle)
            self.assertEqual(report["status"], "PASS", report["errors"])
        finally:
            temp.cleanup()

    def test_production_activation_claim_rejected(self) -> None:
        temp, bundle = self.make()
        try:
            write_json(
                bundle.root / "conversion_report.json",
                {
                    "status": "candidate_converted",
                    "production_activated": True,
                    "human_approved_anchor_count": 1,
                },
            )
            report = self.result(bundle)
            self.assertEqual(report["status"], "FAIL")
            codes = {item["code"] for item in report["errors"]}
            self.assertIn("conversion_report_production", codes)
            self.assertIn("conversion_report_human", codes)
        finally:
            temp.cleanup()

    def test_missing_body_coverage_rejected_in_vector_stage(self) -> None:
        temp, bundle = self.make(with_vectors=True)
        try:
            chunks = [json.loads(line) for line in (bundle.root / "rag_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
            chunks[0]["body_start"] = 1
            write_jsonl(bundle.root / "rag_chunks.jsonl", chunks)
            report = self.result(bundle, require_vectors=True)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("chunk_body_gap", {item["code"] for item in report["errors"]})
        finally:
            temp.cleanup()

    def test_valid_vectors_are_checked_without_model_execution(self) -> None:
        temp, bundle = self.make(with_vectors=True)
        try:
            report = self.result(bundle, require_vectors=True)
            self.assertEqual(report["status"], "PASS", report["errors"])
            self.assertTrue(report["vectors_required"])
        finally:
            temp.cleanup()


def run_and_write_probe() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CuratedBundleValidatorTests)
    def test_ids(value: unittest.TestSuite | unittest.TestCase) -> list[str]:
        if isinstance(value, unittest.TestSuite):
            result: list[str] = []
            for child in value:
                result.extend(test_ids(child))
            return result
        return [value.id()]

    names = test_ids(suite)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    probe_dir = Path(os.environ.get("VALIDATOR_PROBE_DIR", str(Path(__file__).resolve().parents[2] / "validator_probe")))
    probe_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        probe_dir / "synthetic_validator_tests.json",
        {
            "status": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "test_names": names,
        },
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(run_and_write_probe())
