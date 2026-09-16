"""Independent structural validator for the curated KG candidate bundle.

The validator is deliberately evidence-first.  It checks the public bundle and,
when the caller supplies private evidence roots, re-checks identity against the
frozen chapter map and source audit.  It does not load a tokenizer or an
embedding model.  ``--require-vectors`` only validates already-produced vector
bytes and their manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "curated-selected-knowledge-1.0"
ROOT_IDS = {"chemical-engineering-principles", "reaction-engineering"}
SOURCE_COUNTS = {"RE01": 14, "OC02": 13, "TH03": 25, "EN04": 17}
EXTERNAL_NAMESPACE = "chemical-engineering-skills"
OLD_BUNDLE_ID = "curated-four-books-v1"
S001_BUNDLE_ID = "curated-s001-upper-v1"
S001_PROFILE = {
    "source_counts": {"S001": 29},
    "root_ids": {"chemical-engineering-principles"},
    "record_count": 29,
    "volume_count": 1,
    "source_sha256": "c532a5f767fe40a0beef722048b22280bbcfca13d8bb160b3bdc0ffeb8ba7e12",
    "source_bytes": 83845722,
    "source_pages": 371,
    "source_volume_id": "volume:s001:chemical-engineering-principles:chemical-principles-upper",
    "source_chain": "curated-s001-upper-v1:S001",
    "chapter_ids": {"S001-CURATED-CH02", "S001-CURATED-CH03", "S001-CURATED-CH04", "S001-CURATED-CH05"},
    "chapter_ranges": {
        "S001-CURATED-CH02": (18, 120),
        "S001-CURATED-CH03": (121, 176),
        "S001-CURATED-CH04": (177, 300),
        "S001-CURATED-CH05": (301, 331),
    },
    "cross_bundle_count": 17,
    "external_reference_count": 95,
    "review_proofs": {
        "A": "385fd351fa70d1316909623fa977aca8934b22206fd16ff2ef1f40f71e2a138d",
        "B": "1c586c9082efd7ad43b0b63485f0ddf41486a50de5065aa9f3881fccf50dc26e",
        "C": "e6a3297b245946b56775bfb266017928bcd09f783fddb8cd77ecf9b80397b4f6",
    },
    "root_inputs_sha256": "07fc006f25283c629812146167f4a00988bd693820f5e524954ec1ff3df16c25",
    "old_knowledge_units_sha256": "0413d5ba942e6a66b9d8546f449a72e791028f9602db246e25ab709a01d51d6d",
}
PROFILES = {
    OLD_BUNDLE_ID: {
        "source_counts": SOURCE_COUNTS,
        "root_ids": ROOT_IDS,
        "source_chain_template": "curated-four-books-v1:{source_id}",
        "record_count": 69,
        "volume_count": 4,
    },
    S001_BUNDLE_ID: S001_PROFILE,
}
VECTOR_DIMENSIONS = 512
VECTOR_DTYPE = "<f4"
VECTOR_TOLERANCE = 1e-5
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PRIVATE_ABSOLUTE_PATH_RE = re.compile(
    r"(?:(?<![A-Z0-9])[A-Z]:[\\/]|\\\\|/(?:home|Users|mnt|opt|private|root|tmp|var|workspace)(?:/|$))"
)

REQUIRED_BUNDLE_FILES = (
    "source_manifest.json",
    "knowledge_units.jsonl",
    "evidence_registry.jsonl",
    "kg_nodes.jsonl",
    "kg_edges.jsonl",
    "external_references.jsonl",
)
OPTIONAL_BUNDLE_JSON_FILES = ("conversion_report.json", "manifest.json", "qa_summary.json")
OPTIONAL_VECTOR_FILES = ("rag_chunks.jsonl", "chunk_to_kg.jsonl", "vector_manifest.json")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(HASH_RE.fullmatch(value))


def _norm_hash(value: Any) -> str | None:
    return value.lower() if _is_hash(value) else None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _contains_cjk(value: Any) -> bool:
    return isinstance(value, str) and any("\u4e00" <= char <= "\u9fff" for char in value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_no} is not an object")
        rows.append(value)
    return rows


def _iter_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_strings(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _intervals(value: Any) -> list[tuple[int, int]] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, dict):
            start, end = item.get("start"), item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            return None
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            return None
        if start < 1 or end < start:
            return None
        result.append((start, end))
    return result


def _normalize_relation(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _unit_key(source_id: str, unit_id: str, unit_type: str) -> tuple[str, str, str]:
    return (source_id, unit_id, unit_type)


class BundleValidator:
    """Collecting validator; all checks are fail-closed and non-mutating."""

    def __init__(
        self,
        bundle: Path,
        chapter_map: Path | None = None,
        source_audit: Path | None = None,
        records_root: Path | None = None,
        require_vectors: bool = False,
        bundle_id: str = OLD_BUNDLE_ID,
        related_bundle: Path | None = None,
    ) -> None:
        self.bundle = bundle
        self.bundle_id = bundle_id
        self.profile = PROFILES.get(bundle_id)
        self.related_bundle = related_bundle
        self.chapter_map_path = chapter_map
        self.source_audit_path = source_audit
        self.records_root = records_root
        self.require_vectors = require_vectors
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.files: dict[str, Any] = {}
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.chapter: dict[str, Any] | None = None
        self.source_audit: dict[str, Any] | None = None
        self.source_manifest: dict[str, Any] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.kus: dict[str, dict[str, Any]] = {}
        self.evidence: dict[str, dict[str, Any]] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.external: list[dict[str, Any]] = []
        self.map_records: dict[str, dict[str, Any]] = {}
        self.map_sources: dict[str, dict[str, Any]] = {}
        self.map_units: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.map_kus_by_unit: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self.expected_internal: set[tuple[str, str]] = set()
        self.expected_external: set[tuple[str, str]] = set()
        self.cross_bundle: list[dict[str, Any]] = []
        if self.profile is None:
            self.error("unknown_bundle_profile", "bundle_id", "bundle_id is not a predefined profile")

    def error(self, code: str, location: str, message: str) -> None:
        self.errors.append({"code": code, "location": location, "message": message})

    def warn(self, code: str, location: str, message: str) -> None:
        self.warnings.append({"code": code, "location": location, "message": message})

    def load(self) -> None:
        if not self.bundle.exists() or not self.bundle.is_dir():
            self.error("bundle_missing", "bundle", "bundle directory does not exist")
            return
        required_files = list(REQUIRED_BUNDLE_FILES)
        if self.bundle_id == S001_BUNDLE_ID:
            required_files.append("cross_bundle_references.jsonl")
        for name in required_files:
            path = self.bundle / name
            if not path.is_file():
                self.error("required_file_missing", name, "required bundle file is missing")
                continue
            try:
                if name.endswith(".jsonl"):
                    self.rows[name] = _read_jsonl(path)
                else:
                    self.files[name] = _read_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("file_unreadable", name, str(exc))
        for name in OPTIONAL_BUNDLE_JSON_FILES:
            path = self.bundle / name
            if not path.is_file():
                continue
            try:
                self.files[name] = _read_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("file_unreadable", name, str(exc))
        for name in OPTIONAL_VECTOR_FILES:
            path = self.bundle / name
            if not path.is_file() or name in self.files or name in self.rows:
                continue
            try:
                if name.endswith(".jsonl"):
                    self.rows[name] = _read_jsonl(path)
                else:
                    self.files[name] = _read_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("file_unreadable", name, str(exc))
        if self.chapter_map_path is not None:
            try:
                self.chapter = _read_json(self.chapter_map_path)
                if not isinstance(self.chapter, dict):
                    raise ValueError("chapter map is not an object")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("chapter_map_unreadable", str(self.chapter_map_path), str(exc))
        if self.source_audit_path is not None:
            try:
                self.source_audit = _read_json(self.source_audit_path)
                if not isinstance(self.source_audit, dict):
                    raise ValueError("source audit is not an object")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("source_audit_unreadable", str(self.source_audit_path), str(exc))

    def check_public_strings(self) -> None:
        for filename, value in list(self.files.items()) + list(self.rows.items()):
            for location, text in _iter_strings(value, filename):
                if PRIVATE_ABSOLUTE_PATH_RE.search(text):
                    self.error(
                        "private_absolute_path",
                        location,
                        "public bundle contains a private absolute path",
                    )

    def prepare_mapping(self) -> None:
        if self.chapter is None:
            return
        records = self.chapter.get("records")
        volumes = self.chapter.get("source_volumes")
        if not isinstance(records, list) or not isinstance(volumes, list):
            self.error("chapter_map_shape", "chapter_map", "records/source_volumes must be arrays")
            return
        for record in records:
            if not isinstance(record, dict):
                self.error("chapter_map_record", "chapter_map.records", "record is not an object")
                continue
            node_id = record.get("original_node_id")
            if not _nonempty_string(node_id):
                self.error("chapter_map_record_id", "chapter_map.records", "original_node_id is required")
                continue
            if node_id in self.map_records:
                self.error("chapter_map_duplicate_id", f"chapter_map.records[{node_id}]", "duplicate original_node_id")
            self.map_records[node_id] = record
            unit = record.get("chapter_or_support_unit")
            if isinstance(unit, dict):
                unit_id = unit.get("id")
                unit_type = unit.get("type")
                source_id = record.get("source_id")
                if _nonempty_string(source_id) and _nonempty_string(unit_id) and _nonempty_string(unit_type):
                    key = _unit_key(source_id, unit_id, unit_type)
                    if key in self.map_units and self.map_units[key].get("id") != unit_id:
                        self.error("chapter_map_duplicate_unit", f"chapter_map.records[{node_id}]", "duplicate unit key")
                    self.map_units[key] = unit
                    self.map_kus_by_unit[key].add(node_id)
        for volume in volumes:
            if not isinstance(volume, dict):
                continue
            source_id = volume.get("source_id")
            if _nonempty_string(source_id):
                self.map_sources[source_id] = volume
        all_ids = set(self.map_records)
        for source_id, record in self.map_records.items():
            for target in _as_list(record.get("legacy_related_node_ids")):
                if not _nonempty_string(target):
                    self.error("legacy_relation_id", f"chapter_map.records[{source_id}]", "related ID is not text")
                    continue
                if target in all_ids:
                    self.expected_internal.add((source_id, target))
                else:
                    self.expected_external.add((source_id, target))
        direct = self.chapter.get("direct_cross_source_relations", [])
        if isinstance(direct, list):
            direct_pairs: set[tuple[str, str]] = set()
            for item in direct:
                if isinstance(item, dict) and _nonempty_string(item.get("source_node_id")) and _nonempty_string(item.get("target_node_id")):
                    direct_pairs.add((item["source_node_id"], item["target_node_id"]))
            if direct_pairs and not direct_pairs.issubset(self.expected_internal):
                self.error(
                    "chapter_map_relation_mismatch",
                    "chapter_map.direct_cross_source_relations",
                    "declared direct cross-source relation is absent from legacy internal IDs",
                )

    def check_source_manifest(self) -> None:
        self.source_manifest = self.files.get("source_manifest.json")
        if not isinstance(self.source_manifest, dict):
            self.error("source_manifest_shape", "source_manifest.json", "source manifest must be an object")
            return
        if self.profile is None:
            return
        profile_counts = self.profile["source_counts"]
        if self.source_manifest.get("schema_version") != SCHEMA_VERSION:
            self.error("schema_version", "source_manifest.schema_version", "unexpected schema version")
        if self.source_manifest.get("scope") != "authored-selected-knowledge-preview":
            self.error("scope", "source_manifest.scope", "unexpected public scope")
        if self.bundle_id == S001_BUNDLE_ID and self.source_manifest.get("canonical_project") != "chemical-engineering-rag-kg":
            self.error("s001_project", "source_manifest.canonical_project", "S001 canonical project is unexpected")
        sources = self.source_manifest.get("sources")
        if not isinstance(sources, list):
            self.error("source_list", "source_manifest.sources", "sources must be an array")
            return
        if len(sources) != len(profile_counts):
            self.error("source_count", "source_manifest.sources", "source count does not match the selected frozen profile")
        for index, source in enumerate(sources):
            location = f"source_manifest.sources[{index}]"
            if not isinstance(source, dict):
                self.error("source_shape", location, "source is not an object")
                continue
            source_id = source.get("source_id")
            if source_id in self.sources:
                self.error("duplicate_source_id", location, "duplicate source_id")
                continue
            if source_id not in profile_counts:
                self.error("source_id", location, "source_id is outside the selected frozen profile")
                continue
            self.sources[source_id] = source
            required = (
                "bibliography",
                "source_sha256",
                "bytes",
                "total_pdf_pages",
                "is_encrypted",
                "source_volume_id",
                "subject_root",
                "selected_knowledge_units_count",
                "source_payload_public",
                "full_source_digitization_complete",
                "review_provenance",
            )
            for key in required:
                if key not in source:
                    self.error("source_field_missing", f"{location}.{key}", "required source field is missing")
            if not isinstance(source.get("bibliography"), dict) or not source["bibliography"]:
                self.error("bibliography", f"{location}.bibliography", "bibliography must be a non-empty object")
            if not _is_hash(source.get("source_sha256")):
                self.error("source_hash", f"{location}.source_sha256", "source_sha256 must be a SHA-256")
            if not isinstance(source.get("bytes"), int) or source.get("bytes", 0) <= 0:
                self.error("source_bytes", f"{location}.bytes", "bytes must be a positive integer")
            if not isinstance(source.get("total_pdf_pages"), int) or source.get("total_pdf_pages", 0) <= 0:
                self.error("source_pages", f"{location}.total_pdf_pages", "total_pdf_pages must be positive")
            if not isinstance(source.get("is_encrypted"), bool):
                self.error("source_encryption", f"{location}.is_encrypted", "is_encrypted must be boolean")
            if self.bundle_id == S001_BUNDLE_ID:
                expected_root = "chemical-engineering-principles"
            else:
                expected_root = "reaction-engineering" if source_id == "RE01" else "chemical-engineering-principles"
            if source.get("subject_root") != expected_root:
                self.error("source_root", f"{location}.subject_root", "source is assigned to the wrong subject root")
            if source.get("selected_knowledge_units_count") != profile_counts[source_id]:
                self.error("source_record_count", f"{location}.selected_knowledge_units_count", "frozen source count mismatch")
            if source.get("source_payload_public") is not False:
                self.error("public_payload", f"{location}.source_payload_public", "source payload must not be public")
            if source.get("full_source_digitization_complete") is not False:
                self.error("full_digitization", f"{location}.full_source_digitization_complete", "full digitization cannot be claimed")
            self._check_review_provenance(source.get("review_provenance"), location)
        if set(self.sources) != set(profile_counts):
            self.error("source_set", "source_manifest.sources", "source IDs do not equal the selected frozen profile")
        if self.bundle_id == S001_BUNDLE_ID:
            source = self.sources.get("S001", {})
            expected = {
                "source_sha256": S001_PROFILE["source_sha256"],
                "bytes": S001_PROFILE["source_bytes"],
                "total_pdf_pages": S001_PROFILE["source_pages"],
                "is_encrypted": False,
                "source_volume_id": S001_PROFILE["source_volume_id"],
                "subject_root": "chemical-engineering-principles",
                "selected_knowledge_units_count": 29,
                "source_payload_public": False,
                "full_source_digitization_complete": False,
            }
            for key, value in expected.items():
                actual = source.get(key)
                if (key == "source_sha256" and _norm_hash(actual) != value) or (key != "source_sha256" and actual != value):
                    self.error("s001_source_identity", f"source_manifest.sources[S001].{key}", "S001 source identity differs from the fixed profile")
            proof = source.get("review_provenance", {})
            if isinstance(proof, dict):
                if _norm_hash(proof.get("content_review_file_sha256")) != S001_PROFILE["root_inputs_sha256"] or _norm_hash(proof.get("candidate_manifest_sha256")) != S001_PROFILE["root_inputs_sha256"]:
                    self.error("s001_review_identity", "source_manifest.sources[S001].review_provenance", "S001 review proof must bind ROOT_ACCEPTED_INPUTS")
                expected_parts = {
                    "part_a_revised_blocks.json": "eef695c15f7c84920b5867d8bca9a06dcc28665fb612a072361eb72cfb821f78",
                    "part_b_revised_blocks.json": "db0e80aafddfed3397e27a16f2f1d5b525f8a36b6541e0283ba0e16bc0730ea9",
                    "part_c_final_blocks.json": "05f0f8d223ae884dda7f445df9f9b1160da91d26b22d858e3db583070e6c4049",
                }
                expected_reviews = {
                    "A_REVISION_REVIEW.md": S001_PROFILE["review_proofs"]["A"],
                    "B_REVISION_REVIEW.md": S001_PROFILE["review_proofs"]["B"],
                    "ROOT_C_ACCEPTANCE.json": S001_PROFILE["review_proofs"]["C"],
                }
                if proof.get("locked_candidate_part_hashes") != expected_parts:
                    self.error("s001_part_identity", "source_manifest.sources[S001].review_provenance.locked_candidate_part_hashes", "S001 candidate part lock differs")
                if proof.get("locked_candidate_review_hashes") != expected_reviews:
                    self.error("s001_review_files", "source_manifest.sources[S001].review_provenance.locked_candidate_review_hashes", "S001 review-file lock differs")
        if self.chapter is not None:
            if self.chapter.get("source_volume_count") != self.profile["volume_count"] or self.chapter.get("record_count") != self.profile["record_count"]:
                self.error("chapter_map_counts", "chapter_map", "chapter map must describe four volumes and 69 records")
            for source_id, expected in profile_counts.items():
                volume = self.map_sources.get(source_id)
                source = self.sources.get(source_id)
                if volume is not None and source is not None:
                    if source.get("source_volume_id") != volume.get("source_volume_id"):
                        self.error("source_volume_identity", f"source_manifest.sources[{source_id}]", "source volume mismatch")
                    if _norm_hash(source.get("source_sha256")) != _norm_hash(volume.get("source_sha256")):
                        self.error("source_hash_identity", f"source_manifest.sources[{source_id}]", "source hash disagrees with chapter map")
                    if volume.get("record_count") != expected:
                        self.error("chapter_source_count", f"chapter_map.source_volumes[{source_id}]", "chapter map source count mismatch")
        self._check_source_audit_links()

    def check_optional_reports(self) -> None:
        report = self.files.get("conversion_report.json")
        if report is None:
            return
        if not isinstance(report, dict):
            self.error("conversion_report_shape", "conversion_report.json", "conversion report must be an object")
            return
        if report.get("status") != "candidate_converted":
            self.error("conversion_report_status", "conversion_report.status", "conversion report must remain candidate-only")
        if report.get("production_activated") is not False:
            self.error("conversion_report_production", "conversion_report.production_activated", "production activation must remain false")
        if report.get("human_approved_anchor_count") != 0:
            self.error("conversion_report_human", "conversion_report.human_approved_anchor_count", "human approval count must remain zero")

    def _check_review_provenance(self, proof: Any, location: str) -> None:
        if not isinstance(proof, dict):
            self.error("review_provenance", f"{location}.review_provenance", "review_provenance must be an object")
            return
        for key in ("producer_id", "reviewer_id", "content_review_file_sha256", "agent_review", "human_review"):
            if key not in proof:
                self.error("review_field_missing", f"{location}.review_provenance.{key}", "review field is missing")
        if not _nonempty_string(proof.get("producer_id")) or not _nonempty_string(proof.get("reviewer_id")):
            self.error("review_identity", f"{location}.review_provenance", "producer/reviewer IDs are required")
        if not _is_hash(proof.get("content_review_file_sha256")):
            self.error("review_hash", f"{location}.review_provenance.content_review_file_sha256", "review proof must be SHA-256")
        candidate_manifest_sha = proof.get("candidate_manifest_sha256")
        locked_parts = proof.get("locked_candidate_part_hashes")
        if not _is_hash(candidate_manifest_sha):
            self.error("candidate_manifest_proof", f"{location}.review_provenance.candidate_manifest_sha256", "candidate manifest proof must be SHA-256")
        if not isinstance(locked_parts, dict) or not locked_parts:
            self.error("candidate_part_proof", f"{location}.review_provenance.locked_candidate_part_hashes", "locked candidate part hashes are required")
        elif not all(_is_hash(value) for value in locked_parts.values()):
            self.error("candidate_part_proof_hash", f"{location}.review_provenance.locked_candidate_part_hashes", "candidate part proof contains a non-SHA value")
        if proof.get("agent_review") is not True:
            self.error("agent_review", f"{location}.review_provenance.agent_review", "agent_review must remain true")
        if proof.get("human_review") is not False:
            self.error("human_review", f"{location}.review_provenance.human_review", "human_review must remain false")
        for key in ("production_activated", "human_approved", "engineering_accepted"):
            if key in proof and proof.get(key) not in (False, 0, None):
                self.error("review_activation", f"{location}.review_provenance.{key}", "production or human approval cannot be claimed")

    def _check_source_audit_links(self) -> None:
        if self.source_audit is None:
            return
        audit_sources = self.source_audit.get("sources")
        if not isinstance(audit_sources, list):
            self.error("source_audit_shape", "source_audit.sources", "source audit sources must be an array")
            return
        audit_by_id = {item.get("source_id"): item for item in audit_sources if isinstance(item, dict)}
        for source_id, source in self.sources.items():
            audit = audit_by_id.get(source_id)
            if audit is None:
                self.error("source_audit_source", f"source_audit[{source_id}]", "source is absent from audit")
                continue
            asset = audit.get("asset", {})
            if _norm_hash(source.get("source_sha256")) != _norm_hash(asset.get("actual_sha256")):
                self.error("source_audit_hash", f"source_manifest.sources[{source_id}]", "source hash disagrees with source audit")
            if source.get("bytes") != asset.get("actual_bytes") or source.get("total_pdf_pages") != asset.get("expected_pages"):
                self.error("source_audit_asset", f"source_manifest.sources[{source_id}]", "source bytes/pages disagree with source audit")
            body_audit = audit.get("body_audit", {})
            if body_audit.get("formal_record_count") != SOURCE_COUNTS.get(source_id):
                self.error("source_audit_count", f"source_audit[{source_id}].body_audit", "formal body count disagrees")
            proof = source.get("review_provenance", {})
            receipts = audit.get("receipts", {})
            if isinstance(proof, dict) and isinstance(receipts, dict):
                if _norm_hash(proof.get("content_review_file_sha256")) != _norm_hash(receipts.get("content_review_sha256_actual")):
                    self.error("source_audit_review_file", f"source_manifest.sources[{source_id}].review_provenance", "content review proof disagrees with source audit")
                if _norm_hash(proof.get("candidate_manifest_sha256")) != _norm_hash(receipts.get("producer_proof_hashes", {}).get("candidate_manifest_sha256")):
                    self.error("source_audit_candidate_manifest", f"source_manifest.sources[{source_id}].review_provenance", "candidate manifest proof disagrees with source audit")
                actual_parts = receipts.get("producer_proof_hashes", {}).get("locked_candidate_part_hashes", {})
                expected_parts = proof.get("locked_candidate_part_hashes", {})
                if isinstance(actual_parts, dict) and isinstance(expected_parts, dict):
                    if {key: _norm_hash(value) for key, value in expected_parts.items()} != {key: _norm_hash(value) for key, value in actual_parts.items()}:
                        self.error("source_audit_candidate_parts", f"source_manifest.sources[{source_id}].review_provenance", "locked candidate part proofs disagree with source audit")

    def check_knowledge_units(self) -> None:
        raw = self.rows.get("knowledge_units.jsonl", [])
        expected_count = self.profile["record_count"] if self.profile is not None else 69
        if len(raw) != expected_count:
            self.error("ku_count", "knowledge_units.jsonl", f"exactly {expected_count} knowledge units are required")
        seen_text: dict[str, str] = {}
        self.kus = {}
        for index, ku in enumerate(raw):
            location = f"knowledge_units.jsonl:{index + 1}"
            if not isinstance(ku, dict):
                self.error("ku_shape", location, "knowledge unit is not an object")
                continue
            node_id = ku.get("node_id")
            if not _nonempty_string(node_id):
                self.error("ku_id", location, "node_id is required")
                continue
            if node_id in self.kus:
                self.error("duplicate_ku_id", location, "duplicate knowledge unit ID")
            else:
                self.kus[node_id] = ku
            if ku.get("knowledge_unit_id") != node_id:
                self.error("ku_identity", f"{location}.knowledge_unit_id", "knowledge_unit_id must equal node_id")
            for key in (
                "title", "text", "text_sha256", "applicability", "units_basis", "knowledge_layer",
                "source_id", "source_sha256", "source_volume_id", "subject_root", "package_id",
                "source_chain_id", "source_locator", "evidence_refs", "forbidden_transfer",
                "project_value_transfer_allowed", "current_project_authority", "content_available",
                "retrieval_eligible", "embedding_eligible", "embedding_exclusion_reason",
                "embedding_eligibility_basis",
            ):
                if key not in ku:
                    self.error("ku_field_missing", f"{location}.{key}", "required knowledge-unit field is missing")
            if not _nonempty_string(ku.get("title")) or not _nonempty_string(ku.get("text")):
                self.error("ku_text", location, "title and text must be non-empty strings")
            actual_hash = _hash_text(ku.get("text")) if isinstance(ku.get("text"), str) else None
            if actual_hash is not None and _norm_hash(ku.get("text_sha256")) != actual_hash:
                self.error("ku_text_hash", f"{location}.text_sha256", "text_sha256 does not hash the full text")
            elif actual_hash is not None and actual_hash in seen_text:
                self.error("duplicate_body", f"{location}.text_sha256", f"body duplicates {seen_text[actual_hash]}")
            if actual_hash is not None:
                seen_text[actual_hash] = node_id
            applicability = ku.get("applicability")
            if not isinstance(applicability, list) or not applicability or not all(_nonempty_string(x) for x in applicability):
                self.error("ku_applicability", f"{location}.applicability", "applicability must be a non-empty list of strings")
            if not isinstance(ku.get("units_basis"), (str, dict)) or not ku.get("units_basis"):
                self.error("ku_units_basis", f"{location}.units_basis", "units_basis is required")
            if not _nonempty_string(ku.get("knowledge_layer")):
                self.error("ku_layer", f"{location}.knowledge_layer", "knowledge_layer is required")
            source_id = ku.get("source_id")
            if source_id not in self.sources:
                self.error("ku_source", f"{location}.source_id", "KU source is not in source manifest")
            else:
                source = self.sources[source_id]
                for key in ("source_sha256", "source_volume_id", "subject_root"):
                    if ku.get(key) != source.get(key) and not (key == "source_sha256" and _norm_hash(ku.get(key)) == _norm_hash(source.get(key))):
                        self.error("ku_source_identity", f"{location}.{key}", "KU source metadata disagrees with manifest")
                expected_chain = (
                    S001_PROFILE["source_chain"] if self.bundle_id == S001_BUNDLE_ID
                    else f"curated-four-books-v1:{source_id}"
                )
                if ku.get("source_chain_id") != expected_chain:
                    self.error("ku_source_chain", f"{location}.source_chain_id", "unexpected source_chain_id")
            if not _nonempty_string(ku.get("package_id")):
                self.error("ku_package", f"{location}.package_id", "package_id is required")
            if not _nonempty_string(ku.get("source_locator")):
                self.error("ku_locator", f"{location}.source_locator", "source_locator is required")
            evidence_refs = ku.get("evidence_refs")
            if not isinstance(evidence_refs, list) or not evidence_refs:
                self.error("ku_evidence_refs", f"{location}.evidence_refs", "at least one evidence reference is required")
            if not isinstance(ku.get("forbidden_transfer"), list) or not ku.get("forbidden_transfer"):
                self.error("ku_transfer_boundary", f"{location}.forbidden_transfer", "forbidden_transfer must be explicit")
            if ku.get("project_value_transfer_allowed") is not False:
                self.error("ku_project_values", f"{location}.project_value_transfer_allowed", "current project values cannot be transferred")
            if ku.get("current_project_authority") is not False:
                self.error("ku_authority", f"{location}.current_project_authority", "KU cannot claim current-project authority")
            for key in ("content_available", "retrieval_eligible", "embedding_eligible"):
                if ku.get(key) is not True:
                    self.error("ku_eligibility", f"{location}.{key}", "explicit eligibility must be true")
            if ku.get("embedding_exclusion_reason") is not None:
                self.error("ku_embedding_exclusion", f"{location}.embedding_exclusion_reason", "eligible KU cannot carry exclusion reason")
            basis = ku.get("embedding_eligibility_basis")
            if not isinstance(basis, dict):
                self.error("ku_eligibility_basis", f"{location}.embedding_eligibility_basis", "eligibility basis is required")
            else:
                if basis.get("content_type") != "authored_reviewed_explanation":
                    self.error("ku_content_type", f"{location}.embedding_eligibility_basis.content_type", "unexpected content type")
                if not _is_hash(basis.get("source_review_proof_sha256")):
                    self.error("ku_review_proof", f"{location}.embedding_eligibility_basis.source_review_proof_sha256", "source review proof must be SHA-256")
                if self.bundle_id == S001_BUNDLE_ID and isinstance(node_id, str) and "-CURATED-" in node_id:
                    part = node_id.split("-CURATED-", 1)[1][:1]
                    expected_proof = S001_PROFILE["review_proofs"].get(part)
                    if _norm_hash(basis.get("source_review_proof_sha256")) != expected_proof:
                        self.error("s001_ku_review_proof", f"{location}.embedding_eligibility_basis.source_review_proof_sha256", "S001 KU review proof is not the locked part proof")
                if basis.get("public_scope") != "authored_explanations_only":
                    self.error("ku_public_scope", f"{location}.embedding_eligibility_basis.public_scope", "unexpected embedding public scope")
            self._compare_ku_to_mapping(ku, location)
        if self.bundle_id == S001_BUNDLE_ID:
            expected_ids = {
                *(f"S001-CURATED-A{i}" for i in range(1, 11)),
                *(f"S001-CURATED-B{i}" for i in range(1, 11)),
                *(f"S001-CURATED-C{i}" for i in range(1, 10)),
            }
            if set(self.kus) != expected_ids:
                self.error("s001_ku_id_set", "knowledge_units.jsonl", "KU IDs do not equal the fixed S001 29-ID set")
        elif self.chapter is not None and set(self.kus) != set(self.map_records):
            self.error("ku_id_set", "knowledge_units.jsonl", "KU IDs do not equal the frozen chapter-map records")
        self._check_records_root()
        self._check_source_audit_records()

    def _compare_ku_to_mapping(self, ku: dict[str, Any], location: str) -> None:
        if self.chapter is None:
            return
        node_id = ku.get("node_id")
        record = self.map_records.get(node_id)
        if record is None:
            self.error("ku_map_missing", location, "KU is absent from chapter map")
            return
        checks = (
            ("title", "title"),
            ("text", "body"),
            ("source_id", "source_id"),
            ("source_volume_id", "source_volume_id"),
            ("subject_root", "kg_subject_root"),
            ("source_locator", "source_page_range"),
        )
        for output_key, map_key in checks:
            if ku.get(output_key) != record.get(map_key):
                self.error("ku_map_identity", f"{location}.{output_key}", f"value disagrees with chapter map {map_key}")
        if _norm_hash(ku.get("text_sha256")) != _norm_hash(record.get("body_sha256")):
            self.error("ku_map_body_hash", f"{location}.text_sha256", "body hash disagrees with chapter map")
        if _norm_hash(ku.get("source_sha256")) != _norm_hash(record.get("source_sha256")):
            self.error("ku_map_source_hash", f"{location}.source_sha256", "source hash disagrees with chapter map")

    def _check_records_root(self) -> None:
        if self.records_root is None or self.chapter is None:
            return
        for node_id, record in self.map_records.items():
            evidence = record.get("evidence", {})
            relative = evidence.get("source_record_file") if isinstance(evidence, dict) else None
            path = self._locate_record_file(relative, node_id, record.get("source_id"))
            if path is None:
                self.error("source_record_missing", f"records_root[{node_id}]", "source record cannot be located under records-root")
                continue
            try:
                data = _read_json(path)
                body = data.get("text", data.get("body")) if isinstance(data, dict) else None
                body_hash = data.get("text_sha256", data.get("body_sha256")) if isinstance(data, dict) else None
                if body != record.get("body") or _norm_hash(body_hash) != _norm_hash(record.get("body_sha256")):
                    self.error("source_record_identity", f"records_root[{node_id}]", "source record body identity mismatch")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("source_record_unreadable", f"records_root[{node_id}]", str(exc))

    def _locate_record_file(self, relative: Any, node_id: str, source_id: Any) -> Path | None:
        if not self.records_root or not self.records_root.exists():
            return None
        root = self.records_root.resolve()
        candidates: list[Path] = []
        if isinstance(relative, str):
            rel_path = Path(relative)
            if not rel_path.is_absolute() and ".." not in rel_path.parts:
                candidates.append(root / rel_path)
                candidates.append(root / rel_path.name)
        if _nonempty_string(source_id):
            candidates.extend(
                [
                    root / "source_records" / source_id / f"{node_id}.json",
                    root / source_id / f"{node_id}.json",
                ]
            )
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved == root or root not in resolved.parents:
                continue
            if resolved.is_file():
                return resolved
        try:
            matches = [match for match in root.rglob(f"{node_id}.json") if root in match.resolve().parents]
        except OSError:
            matches = []
        return matches[0] if matches else None

    def _check_source_audit_records(self) -> None:
        if self.source_audit is None:
            return
        by_id: dict[str, dict[str, Any]] = {}
        for source in _as_list(self.source_audit.get("sources")):
            if isinstance(source, dict):
                for record in _as_list(source.get("records")):
                    if isinstance(record, dict) and _nonempty_string(record.get("node_id")):
                        by_id[record["node_id"]] = record
        for node_id, ku in self.kus.items():
            audit = by_id.get(node_id)
            if audit is None:
                self.error("source_audit_record", f"source_audit.records[{node_id}]", "KU is absent from source audit")
                continue
            for key in ("body_sha256", "body_sha256_recomputed", "candidate_body_sha256", "approved_body_sha256"):
                if key in audit and _norm_hash(audit.get(key)) != _norm_hash(ku.get("text_sha256")):
                    self.error("source_audit_body_hash", f"source_audit.records[{node_id}].{key}", "audit body hash disagrees with KU")
            if _norm_hash(audit.get("source_sha256")) != _norm_hash(ku.get("source_sha256")):
                self.error("source_audit_record_source", f"source_audit.records[{node_id}]", "audit source hash disagrees with KU")
            source_audit = next((item for item in _as_list(self.source_audit.get("sources")) if isinstance(item, dict) and item.get("source_id") == ku.get("source_id")), None)
            if isinstance(source_audit, dict):
                receipts = source_audit.get("receipts", {})
                proof = receipts.get("content_review_sha256_actual") if isinstance(receipts, dict) else None
                if proof is None and isinstance(receipts, dict):
                    reviewer = receipts.get("reviewer_proof", {})
                    proof = reviewer.get("review_proof_sha256") if isinstance(reviewer, dict) else None
                basis = ku.get("embedding_eligibility_basis", {})
                if _is_hash(proof) and _norm_hash(basis.get("source_review_proof_sha256")) != _norm_hash(proof):
                    self.error("source_audit_review_proof", f"knowledge_units[{node_id}].embedding_eligibility_basis", "KU review proof disagrees with source audit")

    def check_evidence(self) -> None:
        raw = self.rows.get("evidence_registry.jsonl", [])
        self.evidence = {}
        for index, item in enumerate(raw):
            location = f"evidence_registry.jsonl:{index + 1}"
            if not isinstance(item, dict):
                self.error("evidence_shape", location, "evidence row is not an object")
                continue
            evidence_id = item.get("evidence_id")
            if not _nonempty_string(evidence_id):
                self.error("evidence_id", location, "evidence_id is required")
                continue
            if evidence_id in self.evidence:
                self.error("duplicate_evidence_id", location, "duplicate evidence_id")
            self.evidence[evidence_id] = item
            for key in ("type", "source_id", "source_sha256", "pdf_pages", "locator", "reviewed_text_sha256", "review_proof_sha256", "page_media_sha256", "public_source_payload"):
                if key not in item:
                    self.error("evidence_field_missing", f"{location}.{key}", "required evidence field is missing")
            if item.get("type") != "source_document_locator":
                self.error("evidence_type", f"{location}.type", "unexpected evidence type")
            if item.get("source_id") not in self.sources:
                self.error("evidence_source", f"{location}.source_id", "evidence source is absent from manifest")
            if not _is_hash(item.get("source_sha256")):
                self.error("evidence_source_hash", f"{location}.source_sha256", "evidence source hash must be SHA-256")
            elif item.get("source_id") in self.sources and _norm_hash(item.get("source_sha256")) != _norm_hash(self.sources[item["source_id"]].get("source_sha256")):
                self.error("evidence_source_identity", f"{location}.source_sha256", "evidence source hash disagrees with manifest")
            parsed = _intervals(item.get("pdf_pages"))
            if parsed is None:
                self.error("evidence_pages", f"{location}.pdf_pages", "pdf_pages must be positive start/end intervals")
            else:
                source = self.sources.get(item.get("source_id"), {})
                total = source.get("total_pdf_pages")
                if isinstance(total, int) and any(end > total for _, end in parsed):
                    self.error("evidence_page_bound", f"{location}.pdf_pages", "evidence page exceeds source total")
            if not _nonempty_string(item.get("locator")):
                self.error("evidence_locator", f"{location}.locator", "locator is required")
            if not _is_hash(item.get("reviewed_text_sha256")) or not _is_hash(item.get("review_proof_sha256")):
                self.error("evidence_hash", location, "reviewed_text_sha256/review_proof_sha256 must be SHA-256")
            if self.bundle_id == S001_BUNDLE_ID and _is_hash(item.get("review_proof_sha256")):
                evidence_id = item.get("evidence_id")
                part = evidence_id.split("-CURATED-", 1)[1][:1] if isinstance(evidence_id, str) and "-CURATED-" in evidence_id else ""
                if _norm_hash(item.get("review_proof_sha256")) != self.profile["review_proofs"].get(part):
                    self.error("s001_evidence_review_proof", f"{location}.review_proof_sha256", "evidence review proof is not the locked A/B/C proof")
            if item.get("page_media_sha256") is not None:
                self.error("page_media_public", f"{location}.page_media_sha256", "page media must remain null")
            if item.get("public_source_payload") is not False:
                self.error("source_payload_public", f"{location}.public_source_payload", "source payload must remain private")
            for forbidden_key in ("page_sha256", "page_hash"):
                if forbidden_key in item:
                    self.error("page_hash_field", f"{location}.{forbidden_key}", "page hash is not a permitted public field")
            if item.get("page_media_sha256") is not None and _norm_hash(item.get("page_media_sha256")) == _norm_hash(item.get("source_sha256")):
                self.error("source_hash_as_page_hash", location, "source SHA cannot stand in for a page SHA")
        for node_id, ku in self.kus.items():
            refs = ku.get("evidence_refs", [])
            for ref in _as_list(refs):
                if ref not in self.evidence:
                    self.error("evidence_ref_dangling", f"knowledge_units[{node_id}].evidence_refs", f"unknown evidence {ref}")
                else:
                    evidence = self.evidence[ref]
                    if evidence.get("source_id") != ku.get("source_id"):
                        self.error("evidence_ku_source", f"evidence[{ref}]", "evidence source differs from KU")
                    if _norm_hash(evidence.get("reviewed_text_sha256")) != _norm_hash(ku.get("text_sha256")):
                        self.error("evidence_ku_body", f"evidence[{ref}]", "evidence text hash differs from KU")
        if set(self.evidence) and set(self.evidence) != {ref for ku in self.kus.values() for ref in _as_list(ku.get("evidence_refs"))}:
            self.error("evidence_set", "evidence_registry.jsonl", "evidence registry contains unreferenced or missing evidence")
        self._compare_evidence_to_mapping()

    def _compare_evidence_to_mapping(self) -> None:
        if self.chapter is None:
            return
        for node_id, record in self.map_records.items():
            ku = self.kus.get(node_id, {})
            refs = _as_list(ku.get("evidence_refs"))
            unit_intervals = _intervals(record.get("source_physical_page_intervals"))
            if unit_intervals is None:
                self.error("map_page_intervals", f"chapter_map.records[{node_id}]", "chapter map page intervals invalid")
                continue
            found = False
            for ref in refs:
                item = self.evidence.get(ref)
                if item is not None and _intervals(item.get("pdf_pages")) == unit_intervals:
                    found = True
            if not found:
                self.error("evidence_page_identity", f"knowledge_units[{node_id}].evidence_refs", "evidence pages do not preserve chapter-map intervals")

    def _node_context_key(self, node: dict[str, Any]) -> tuple[str, str, str] | None:
        source_id = node.get("source_id")
        if not _nonempty_string(source_id):
            return None
        node_type = node.get("node_type")
        if node_type not in ("chapter", "cross_chapter_support_unit"):
            return None
        unit_id = node.get("chapter_id") if node_type == "chapter" else node.get("support_unit_id")
        if not _nonempty_string(unit_id):
            unit_id = node.get("unit_id") or node.get("chapter_or_support_unit_id") or node.get("group_id")
        if not _nonempty_string(unit_id):
            node_id = str(node.get("node_id", ""))
            unit_id = node_id.rsplit(":", 1)[-1]
        direct_key = _unit_key(source_id, unit_id, node_type)
        if direct_key in self.map_units:
            return direct_key
        suffix = str(unit_id).rsplit(":", 1)[-1]
        suffix_key = _unit_key(source_id, suffix, node_type)
        return suffix_key if suffix_key in self.map_units else direct_key

    def check_graph(self) -> None:
        raw_nodes = self.rows.get("kg_nodes.jsonl", [])
        if self.bundle_id == S001_BUNDLE_ID:
            if len(raw_nodes) != 64:
                self.error("s001_node_count", "kg_nodes.jsonl", "S001 profile requires 64 graph nodes")
        self.nodes = {}
        for index, node in enumerate(raw_nodes):
            location = f"kg_nodes.jsonl:{index + 1}"
            if not isinstance(node, dict):
                self.error("node_shape", location, "KG node is not an object")
                continue
            node_id = node.get("node_id")
            if not _nonempty_string(node_id):
                self.error("node_id", location, "node_id is required")
                continue
            if node_id in self.nodes:
                self.error("duplicate_node_id", location, "duplicate node_id")
            self.nodes[node_id] = node
            if node.get("node_type") not in ("subject_root", "source_volume", "chapter", "cross_chapter_support_unit", "knowledge_unit", "evidence"):
                self.error("node_type", f"{location}.node_type", "unknown node_type")
            if not _nonempty_string(node.get("label")):
                self.error("node_label", f"{location}.label", "node label is required")
            if node.get("node_type") != "subject_root" and not _nonempty_string(node.get("subject_root")):
                self.error("node_root", f"{location}.subject_root", "non-root node must carry subject_root")
            if node.get("node_type") in ("source_volume", "chapter", "cross_chapter_support_unit", "knowledge_unit", "evidence"):
                if not _nonempty_string(node.get("source_id")) or not _nonempty_string(node.get("source_volume_id")):
                    self.error("node_source_scope", location, "source-scoped node must carry source_id/source_volume_id")
            if node.get("node_type") == "chapter":
                if isinstance(node.get("chapter_number"), bool) or not isinstance(node.get("chapter_number"), int) or node.get("chapter_number", 0) <= 0:
                    self.error("chapter_number", f"{location}.chapter_number", "chapter_number must be a positive integer")
            if node.get("node_type") == "cross_chapter_support_unit":
                refs = node.get("chapter_refs")
                if not isinstance(refs, list) or not refs or not all(
                    _nonempty_string(ref) or (isinstance(ref, dict) and _nonempty_string(ref.get("id")))
                    for ref in refs
                ):
                    self.error("support_chapter_refs", f"{location}.chapter_refs", "support group must retain chapter reference IDs")
        root_nodes = {node_id for node_id, node in self.nodes.items() if node.get("node_type") == "subject_root"}
        expected_roots = self.profile["root_ids"] if self.profile is not None else ROOT_IDS
        if root_nodes != expected_roots:
            self.error("root_set", "kg_nodes.jsonl", "subject-root set does not match the selected frozen profile")
        for root_id in root_nodes:
            if not _contains_cjk(self.nodes[root_id].get("label")):
                self.error("root_label", root_id, "subject-root label must be a Chinese display label")
        volume_nodes = {node_id: node for node_id, node in self.nodes.items() if node.get("node_type") == "source_volume"}
        expected_volumes = {source.get("source_volume_id") for source in self.sources.values()}
        if set(volume_nodes) != expected_volumes:
            self.error("volume_set", "kg_nodes.jsonl", "source volume nodes do not equal manifest volumes")
        if self.bundle_id == S001_BUNDLE_ID:
            chapter_nodes = {node_id for node_id, node in self.nodes.items() if node.get("node_type") == "chapter"}
            if chapter_nodes != S001_PROFILE["chapter_ids"]:
                self.error("s001_chapter_set", "kg_nodes.jsonl", "S001 chapter nodes do not equal the fixed four-chapter set")
            if len(volume_nodes) != 1:
                self.error("s001_volume_count", "kg_nodes.jsonl", "S001 requires one source volume node")
            for chapter_id in S001_PROFILE["chapter_ids"]:
                chapter = self.nodes.get(chapter_id)
                if chapter is None:
                    continue
                expected_number = int(chapter_id[-2:])
                if chapter.get("chapter_number") != expected_number or chapter.get("source_id") != "S001" or chapter.get("source_volume_id") != S001_PROFILE["source_volume_id"] or chapter.get("subject_root") != "chemical-engineering-principles":
                    self.error("s001_chapter_identity", chapter_id, "S001 chapter metadata differs from fixed profile")
        ku_nodes = {node_id: node for node_id, node in self.nodes.items() if node.get("node_type") == "knowledge_unit"}
        if set(ku_nodes) != set(self.kus):
            self.error("graph_ku_set", "kg_nodes.jsonl", "knowledge-unit nodes do not equal knowledge_units.jsonl")
        evidence_nodes = {node_id: node for node_id, node in self.nodes.items() if node.get("node_type") == "evidence"}
        if set(evidence_nodes) != set(self.evidence):
            self.error("graph_evidence_set", "kg_nodes.jsonl", "evidence nodes do not equal evidence registry")
        if self.chapter is not None:
            expected_units = set(self.map_units)
            actual_units: dict[tuple[str, str, str], str] = {}
            for node_id, node in self.nodes.items():
                key = self._node_context_key(node)
                if key is not None:
                    if key in actual_units:
                        self.error("duplicate_context_node", node_id, "duplicate chapter/support context node")
                    actual_units[key] = node_id
            if set(actual_units) != expected_units:
                self.error("context_unit_set", "kg_nodes.jsonl", "chapter/support nodes do not equal frozen map units")
            for key, record_unit in self.map_units.items():
                node_id = actual_units.get(key)
                if node_id is None:
                    continue
                node = self.nodes[node_id]
                if key[2] == "cross_chapter_support_unit":
                    refs = node.get("chapter_refs")
                    if not isinstance(refs, list):
                        self.error("support_chapter_refs", node_id, "support group must retain chapter_refs")
                    expected_refs = {str(ref.get("id")) for ref in _as_list(record_unit.get("chapter_refs")) if isinstance(ref, dict) and _nonempty_string(ref.get("id"))}
                    actual_refs = {str(ref.get("id")) if isinstance(ref, dict) else str(ref) for ref in _as_list(refs)}
                    if expected_refs != actual_refs:
                        self.error("support_chapter_ref_identity", node_id, "support group chapter_refs disagree with map")
        raw_edges = self.rows.get("kg_edges.jsonl", [])
        if self.bundle_id == S001_BUNDLE_ID and len(raw_edges) != 63:
            self.error("s001_edge_count", "kg_edges.jsonl", "S001 profile requires 63 graph edges")
        self.edges = []
        edge_ids: set[str] = set()
        for index, edge in enumerate(raw_edges):
            location = f"kg_edges.jsonl:{index + 1}"
            if not isinstance(edge, dict):
                self.error("edge_shape", location, "KG edge is not an object")
                continue
            edge_id = edge.get("edge_id")
            if not _nonempty_string(edge_id) or edge_id in edge_ids:
                self.error("edge_id", location, "edge_id is missing or duplicated")
            edge_ids.add(edge_id)
            source = edge.get("source_node_id")
            target = edge.get("target_node_id")
            relation = _normalize_relation(edge.get("relation"))
            if source not in self.nodes or target not in self.nodes:
                self.error("dangling_edge", location, "edge endpoint is not a KG node")
            self.edges.append(edge)
            if source in self.nodes and target in self.nodes:
                self._check_edge_layer(edge, self.nodes[source], self.nodes[target], location)
        self._check_hierarchy_edges()
        self._check_package_ids()
        self._check_related_edges()

    def check_s001_chapter_mapping(self) -> None:
        """Keep S001's accepted KU-to-chapter/page map explicit and fail closed."""
        if self.bundle_id != S001_BUNDLE_ID:
            return
        expected: dict[str, str] = {}
        for prefix, numbers, chapter in (
            ("A", range(1, 11), "S001-CURATED-CH02"),
            ("B", range(1, 8), "S001-CURATED-CH03"),
            ("C", range(1, 10), "S001-CURATED-CH04"),
            ("B", range(8, 11), "S001-CURATED-CH05"),
        ):
            expected.update({f"S001-CURATED-{prefix}{number}": chapter for number in numbers})
        contains_parents: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            if _normalize_relation(edge.get("relation")) == "contains" and self.nodes.get(edge.get("target_node_id"), {}).get("node_type") == "knowledge_unit":
                contains_parents[edge.get("target_node_id")].add(edge.get("source_node_id"))
        for ku_id, chapter_id in expected.items():
            ku = self.kus.get(ku_id)
            if ku is None:
                continue
            if ku.get("package_id") != chapter_id:
                self.error("s001_package_chapter", f"knowledge_units[{ku_id}].package_id", "package_id disagrees with the accepted S001 chapter map")
            if contains_parents.get(ku_id) != {chapter_id}:
                self.error("s001_parent_chapter", f"kg_edges.jsonl[{ku_id}]", "KU contains-parent disagrees with the accepted S001 chapter map")
            start, end = S001_PROFILE["chapter_ranges"][chapter_id]
            for ref in _as_list(ku.get("evidence_refs")):
                evidence = self.evidence.get(ref)
                parsed = _intervals(evidence.get("pdf_pages")) if evidence is not None else None
                if parsed is None or any(page_start < start or page_end > end for page_start, page_end in parsed):
                    self.error("s001_evidence_chapter", f"evidence[{ref}]", "evidence pages fall outside the accepted S001 chapter range")

    def _check_package_ids(self) -> None:
        context_nodes: dict[tuple[str, str, str], dict[str, Any]] = {}
        for node in self.nodes.values():
            key = self._node_context_key(node)
            if key is not None:
                context_nodes[key] = node
        context_by_group: dict[str, dict[str, Any]] = {}
        for node in self.nodes.values():
            if node.get("node_type") not in ("chapter", "cross_chapter_support_unit"):
                continue
            group_id = node.get("group_id") or node.get("node_id")
            if _nonempty_string(group_id):
                context_by_group[group_id] = node
        for node_id, ku in self.kus.items():
            package_id = ku.get("package_id")
            context = context_by_group.get(package_id) if _nonempty_string(package_id) else None
            if context is None and self.chapter is not None:
                record = self.map_records.get(node_id)
                unit = record.get("chapter_or_support_unit", {}) if isinstance(record, dict) else {}
                key = _unit_key(record.get("source_id"), unit.get("id"), unit.get("type")) if isinstance(record, dict) and isinstance(unit, dict) else None
                context = context_nodes.get(key) if key is not None else None
            if context is None:
                self.error("ku_package_identity", f"knowledge_units[{node_id}].package_id", "package_id must equal a graph chapter/support group_id")
                continue
            group_id = context.get("group_id") or context.get("node_id")
            if package_id != group_id:
                self.error("ku_package_identity", f"knowledge_units[{node_id}].package_id", "package_id must equal the graph chapter/support group_id")
            if context.get("source_id") != ku.get("source_id") or context.get("source_volume_id") != ku.get("source_volume_id"):
                self.error("ku_package_scope", f"knowledge_units[{node_id}].package_id", "package group source scope differs from KU")

    def _check_edge_layer(self, edge: dict[str, Any], source: dict[str, Any], target: dict[str, Any], location: str) -> None:
        relation = _normalize_relation(edge.get("relation"))
        pair = (source.get("node_type"), target.get("node_type"))
        allowed = {
            ("subject_root", "source_volume"): {"contains"},
            ("source_volume", "chapter"): {"contains"},
            ("source_volume", "cross_chapter_support_unit"): {"contains"},
            ("chapter", "knowledge_unit"): {"contains"},
            ("cross_chapter_support_unit", "knowledge_unit"): {"contains"},
            ("cross_chapter_support_unit", "chapter"): {"has_chapter_context"},
            ("knowledge_unit", "evidence"): {"supported_by"},
            ("knowledge_unit", "knowledge_unit"): {"related_to"},
        }
        if relation not in allowed.get(pair, set()):
            self.error("edge_layer", location, f"relation {relation!r} is invalid for node types {pair}")
        if source.get("node_type") == "subject_root" and target.get("node_type") != "source_volume":
            self.error("root_cross_layer_edge", location, "subject root may only contain source volumes")
        if source.get("node_type") != "subject_root" and source.get("subject_root") != target.get("subject_root") and relation in {"contains", "has_chapter_context", "supported_by"}:
            self.error("cross_root_edge", location, "hierarchy edge crosses subject roots")
        if relation in {"contains", "supported_by"} and source.get("source_id") and target.get("source_id") and source.get("source_id") != target.get("source_id"):
            self.error("cross_source_hierarchy_edge", location, "hierarchy edge crosses source volumes")

    def _edge_set(self, relation: str) -> set[tuple[str, str]]:
        return {(edge.get("source_node_id"), edge.get("target_node_id")) for edge in self.edges if _normalize_relation(edge.get("relation")) == relation}

    def _check_hierarchy_edges(self) -> None:
        contains = self._edge_set("contains")
        supported = self._edge_set("supported_by")
        context = self._edge_set("has_chapter_context")
        expected_root_volume: set[tuple[str, str]] = set()
        actual_context: dict[tuple[str, str, str], str] = {}
        for node_id, node in self.nodes.items():
            key = self._node_context_key(node)
            if key is not None:
                actual_context[key] = node_id
        for source_id, source in self.sources.items():
            volume_id = source.get("source_volume_id")
            if volume_id in self.nodes:
                expected_root_volume.add((source.get("subject_root"), volume_id))
        if expected_root_volume != {(a, b) for a, b in contains if self.nodes.get(a, {}).get("node_type") == "subject_root"}:
            self.error("root_volume_edges", "kg_edges.jsonl", "root-to-volume edges do not match source manifest")

        actual_volume_unit = {(a, b) for a, b in contains if self.nodes.get(a, {}).get("node_type") == "source_volume"}
        expected_volume_unit: set[tuple[str, str]] = set()
        for node_id, node in self.nodes.items():
            if node.get("node_type") not in ("chapter", "cross_chapter_support_unit"):
                continue
            volume_id = node.get("source_volume_id")
            if volume_id not in self.nodes or self.nodes[volume_id].get("node_type") != "source_volume":
                self.error("context_volume", node_id, "chapter/support group does not name a source volume node")
                continue
            expected_volume_unit.add((volume_id, node_id))
        if actual_volume_unit != expected_volume_unit:
            self.error("volume_context_edges", "kg_edges.jsonl", "volume-to-context edges do not close the public hierarchy")

        actual_unit_ku = {
            (a, b)
            for a, b in contains
            if self.nodes.get(a, {}).get("node_type") in ("chapter", "cross_chapter_support_unit")
        }
        expected_unit_ku: set[tuple[str, str]] = set()
        for node_id, node in self.nodes.items():
            if node.get("node_type") != "knowledge_unit":
                continue
            parents = [source for source, target in actual_unit_ku if target == node_id]
            if len(parents) != 1:
                self.error("ku_parent_closure", node_id, "each KU must have exactly one chapter/support parent")
                continue
            parent = self.nodes.get(parents[0], {})
            if parent.get("source_id") != node.get("source_id") or parent.get("source_volume_id") != node.get("source_volume_id"):
                self.error("ku_parent_scope", node_id, "KU parent source scope differs from KU")
            expected_unit_ku.add((parents[0], node_id))
        if actual_unit_ku != expected_unit_ku:
            self.error("context_ku_edges", "kg_edges.jsonl", "context-to-KU edges do not close the public hierarchy")
        expected_supported = {(ku_id, ref) for ku_id, ku in self.kus.items() for ref in _as_list(ku.get("evidence_refs"))}
        if supported != expected_supported:
            self.error("ku_evidence_edges", "kg_edges.jsonl", "KU-to-evidence edges do not match evidence_refs")
        expected_context: set[tuple[str, str]] = set()
        for support_id, support in self.nodes.items():
            if support.get("node_type") != "cross_chapter_support_unit":
                continue
            refs = _as_list(support.get("chapter_refs"))
            for chapter_ref in refs:
                ref_id = chapter_ref.get("id") if isinstance(chapter_ref, dict) else chapter_ref
                if not _nonempty_string(ref_id):
                    continue
                candidates = []
                for chapter_id, chapter in self.nodes.items():
                    if chapter.get("node_type") != "chapter" or chapter.get("source_id") != support.get("source_id"):
                        continue
                    identifiers = {
                        str(chapter_id),
                        str(chapter.get("chapter_id", "")),
                        str(chapter.get("group_id", "")),
                    }
                    identifiers.update(identifier.rsplit(":", 1)[-1] for identifier in list(identifiers) if identifier)
                    if str(ref_id) in identifiers or str(ref_id).rsplit(":", 1)[-1] in identifiers:
                        candidates.append(chapter_id)
                if len(candidates) == 1:
                    expected_context.add((support_id, candidates[0]))
                elif not candidates:
                    self.error("support_context_target", support_id, f"chapter_ref {ref_id} has no chapter node")
                else:
                    self.error("support_context_target", support_id, f"chapter_ref {ref_id} maps to multiple chapter nodes")
        if context != expected_context:
            self.error("support_context_edges", "kg_edges.jsonl", "support-group chapter context edges do not match public chapter_refs")
        if self.chapter is not None:
            expected_map_volume_unit: set[tuple[str, str]] = set()
            expected_map_unit_ku: set[tuple[str, str]] = set()
            for key in self.map_units:
                context_id = actual_context.get(key)
                volume_id = self.sources.get(key[0], {}).get("source_volume_id")
                if context_id and volume_id:
                    expected_map_volume_unit.add((volume_id, context_id))
                    expected_map_unit_ku.update((context_id, ku_id) for ku_id in self.map_kus_by_unit.get(key, set()))
            if actual_volume_unit != expected_map_volume_unit:
                self.error("map_volume_context_edges", "kg_edges.jsonl", "volume-to-context edges disagree with chapter map")
            if actual_unit_ku != expected_map_unit_ku:
                self.error("map_context_ku_edges", "kg_edges.jsonl", "context-to-KU edges disagree with chapter map")

    def _check_related_edges(self) -> None:
        actual = self._edge_set("related_to")
        internal_actual: set[tuple[str, str]] = set()
        for source_id, target_id in actual:
            if source_id in self.kus and target_id in self.kus:
                internal_actual.add((source_id, target_id))
            else:
                self.error("related_target_namespace", f"kg_edges[{source_id}->{target_id}]", "internal related_to must target a KU")
        if self.chapter is not None and internal_actual != self.expected_internal:
            self.error("internal_relation_set", "kg_edges.jsonl", "internal related_to edges do not match frozen legacy IDs")
        if self.chapter is None and actual:
            if self.bundle_id == S001_BUNDLE_ID:
                self.error("s001_local_related_edge", "kg_edges.jsonl", "S001 cross-bundle relations must remain in cross_bundle_references.jsonl")
            else:
                self.warn("relation_evidence_unavailable", "kg_edges.jsonl", "related_to edges cannot be identity-checked without chapter map")

    def check_cross_bundle_references(self) -> None:
        """Validate S001's explicit links against a supplied frozen old KU file."""
        if self.bundle_id != S001_BUNDLE_ID:
            return
        raw = self.rows.get("cross_bundle_references.jsonl", [])
        if len(raw) != S001_PROFILE["cross_bundle_count"]:
            self.error("s001_cross_count", "cross_bundle_references.jsonl", "S001 profile requires exactly 17 cross-bundle references")
        if self.related_bundle is None:
            self.error("related_bundle_required", "cross_bundle_references.jsonl", "S001 cross-bundle checks require --related-bundle")
            return
        related_path = self.related_bundle / "knowledge_units.jsonl" if self.related_bundle.is_dir() else self.related_bundle
        related_manifest = self.related_bundle / "manifest.json" if self.related_bundle.is_dir() else None
        if related_manifest is not None and related_manifest.is_file():
            try:
                manifest = _read_json(related_manifest)
                if manifest.get("bundle_id") != OLD_BUNDLE_ID:
                    self.error("related_bundle_id", str(related_manifest), "related bundle is not the frozen old four-book bundle")
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                self.error("related_bundle_unreadable", str(related_manifest), str(exc))
        if not related_path.is_file():
            self.error("related_knowledge_units_missing", str(related_path), "related frozen knowledge_units.jsonl is required")
            return
        try:
            payload = related_path.read_bytes()
        except OSError as exc:
            self.error("related_knowledge_units_unreadable", str(related_path), str(exc))
            return
        if _hash_bytes(payload) != S001_PROFILE["old_knowledge_units_sha256"]:
            self.error("related_knowledge_units_hash", str(related_path), "related KU file is not the frozen old four-book file")
        try:
            related_rows = _read_jsonl(related_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.error("related_knowledge_units_unreadable", str(related_path), str(exc))
            return
        related: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(related_rows):
            node_id = item.get("node_id")
            if not _nonempty_string(node_id) or node_id in related:
                self.error("related_knowledge_unit_id", f"{related_path.name}:{index + 1}", "related KU ID is missing or duplicated")
                continue
            related[node_id] = item
            if not isinstance(item.get("text"), str) or _norm_hash(item.get("text_sha256")) != _hash_text(item.get("text", "")):
                self.error("related_body_hash", f"{related_path.name}:{index + 1}", "related KU body hash is invalid")
        self.cross_bundle = []
        seen: set[tuple[str, str, str]] = set()
        for index, item in enumerate(raw):
            location = f"cross_bundle_references.jsonl:{index + 1}"
            source = item.get("from_knowledge_unit_id")
            target_bundle = item.get("target_bundle_id")
            target = item.get("target_node_id")
            target_hash = item.get("target_text_sha256")
            if source not in self.kus:
                self.error("cross_source", location, "cross-bundle source must be a current S001 KU")
            if item.get("relation") != "related_to":
                self.error("cross_relation", location, "cross-bundle relation must be related_to")
            if target_bundle != OLD_BUNDLE_ID:
                self.error("cross_target_bundle", location, "cross-bundle target bundle must be curated-four-books-v1")
            if not _nonempty_string(target) or not _is_hash(target_hash):
                self.error("cross_target", location, "cross-bundle target and target body hash are required")
            key = (source, target_bundle, target)
            if key in seen:
                self.error("duplicate_cross_reference", location, "duplicate cross-bundle reference")
            seen.add(key)
            old = related.get(target)
            if old is None:
                self.error("cross_target_dangling", location, "target node is absent from --related-bundle knowledge_units.jsonl")
            elif _norm_hash(target_hash) != _norm_hash(old.get("text_sha256")) or (isinstance(old.get("text"), str) and _hash_text(old["text"]) != _norm_hash(target_hash)):
                self.error("cross_target_body_hash", location, "target body hash does not match the supplied old KU")
            self.cross_bundle.append(item)

    def check_external_references(self) -> None:
        raw = self.rows.get("external_references.jsonl", [])
        if self.bundle_id == S001_BUNDLE_ID and len(raw) != S001_PROFILE["external_reference_count"]:
            self.error("s001_external_count", "external_references.jsonl", "S001 profile requires 95 external references")
        self.external = []
        seen: set[tuple[str, str]] = set()
        for index, item in enumerate(raw):
            location = f"external_references.jsonl:{index + 1}"
            if not isinstance(item, dict):
                self.error("external_shape", location, "external reference is not an object")
                continue
            source = item.get("from_knowledge_unit_id")
            target = item.get("target_node_id")
            if source not in self.kus:
                self.error("external_source", location, "external source is not a KU")
            if not _nonempty_string(target):
                self.error("external_target", location, "target_node_id is required")
            if item.get("target_project") != EXTERNAL_NAMESPACE:
                self.error("external_namespace", location, "external target project is not the contract namespace")
            if item.get("relation") != "related_to":
                self.error("external_relation", location, "external relation must be related_to")
            pair = (source, target)
            if pair in seen:
                self.error("duplicate_external", location, "duplicate external reference")
            seen.add(pair)
            if target in self.kus:
                self.error("external_internal_target", location, "internal target must not be emitted as external reference")
            self.external.append(item)
        actual = {(item.get("from_knowledge_unit_id"), item.get("target_node_id")) for item in self.external}
        if self.chapter is not None and actual != self.expected_external:
            self.error("external_relation_set", "external_references.jsonl", "external references do not match frozen legacy IDs")

    def check_vectors(self) -> None:
        if not self.require_vectors:
            return
        required = ("rag_chunks.jsonl", "chunk_to_kg.jsonl", "vector_manifest.json", "embeddings.f32")
        for name in required:
            if not (self.bundle / name).is_file():
                self.error("vector_file_missing", name, "required vector-stage file is missing")
        if any(error["code"] == "vector_file_missing" for error in self.errors):
            return
        try:
            chunks = _read_jsonl(self.bundle / "rag_chunks.jsonl")
            mappings = _read_jsonl(self.bundle / "chunk_to_kg.jsonl")
            manifest = _read_json(self.bundle / "vector_manifest.json")
            matrix = (self.bundle / "embeddings.f32").read_bytes()
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            self.error("vector_unreadable", "vector_stage", str(exc))
            return
        chunks_by_id: dict[str, dict[str, Any]] = {}
        intervals_by_ku: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            location = f"rag_chunks.jsonl:{index + 1}"
            chunk_id = chunk.get("chunk_id")
            ku_id = chunk.get("knowledge_unit_id", chunk.get("node_id"))
            if not _nonempty_string(chunk_id) or chunk_id in chunks_by_id:
                self.error("chunk_id", location, "chunk_id is missing or duplicated")
                continue
            chunks_by_id[chunk_id] = chunk
            if ku_id not in self.kus:
                self.error("chunk_ku", location, "chunk does not map to a known KU")
                continue
            ku = self.kus[ku_id]
            for key, value in ku.items():
                if chunk.get(key) != value:
                    self.error("chunk_ku_inheritance", f"{location}.{key}", "chunk does not preserve the complete KU field")
            if chunk.get("text") != ku.get("text"):
                self.error("chunk_full_body", location, "chunk text must remain the complete original KU body")
            if _norm_hash(chunk.get("full_text_sha256")) != _norm_hash(ku.get("text_sha256")):
                self.error("chunk_full_body_hash", location, "full_text_sha256 must equal the complete KU text hash")
            for optional_full_body_key in ("original_body", "full_text", "source_text"):
                if optional_full_body_key in chunk and chunk.get(optional_full_body_key) != ku.get("text"):
                    self.error("chunk_full_body", f"{location}.{optional_full_body_key}", "optional full-body field differs from KU text")
            start, end = chunk.get("body_start"), chunk.get("body_end")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(ku.get("text", "")):
                self.error("chunk_span", location, "chunk body span is out of bounds")
                continue
            segment = ku["text"][start:end]
            applicability = ku.get("applicability", [])
            expected_search = f"{ku.get('title', '')}\n{segment}\n{'\n'.join(applicability)}"
            if chunk.get("search_text") != expected_search:
                self.error("chunk_search_text", location, "search_text does not equal the contract input template")
            if chunk.get("embedding_text") != chunk.get("search_text"):
                self.error("chunk_embedding_text", location, "embedding_text must equal search_text")
            for key in ("retrieval_eligible", "embedding_eligible"):
                if chunk.get(key) is not True:
                    self.error("chunk_eligibility", f"{location}.{key}", "chunk eligibility must be explicit")
            intervals_by_ku[ku_id].append((start, end))
        for ku_id, intervals in intervals_by_ku.items():
            intervals.sort()
            cursor = 0
            length = len(self.kus[ku_id].get("text", ""))
            for start, end in intervals:
                if start > cursor:
                    self.error("chunk_body_gap", ku_id, "chunk spans leave a gap in the complete KU body")
                cursor = max(cursor, end)
            if cursor != length:
                self.error("chunk_body_coverage", ku_id, "chunk spans do not cover the complete KU body")
        if set(intervals_by_ku) != set(self.kus):
            self.error("chunk_ku_coverage", "rag_chunks.jsonl", "every KU must have at least one chunk")
        mapped: dict[str, str] = {}
        for index, item in enumerate(mappings):
            location = f"chunk_to_kg.jsonl:{index + 1}"
            chunk_id = item.get("chunk_id")
            ku_id = item.get("knowledge_unit_id", item.get("node_id", item.get("kg_node_id")))
            if chunk_id in mapped:
                self.error("chunk_mapping_duplicate", location, "chunk mapping is duplicated")
            mapped[chunk_id] = ku_id
            if chunk_id not in chunks_by_id or ku_id not in self.kus:
                self.error("chunk_mapping_dangling", location, "chunk mapping endpoint is unknown")
            elif (chunks_by_id[chunk_id].get("knowledge_unit_id", chunks_by_id[chunk_id].get("node_id")) != ku_id):
                self.error("chunk_mapping_identity", location, "chunk mapping disagrees with rag chunk")
        if set(mapped) != set(chunks_by_id):
            self.error("chunk_mapping_set", "chunk_to_kg.jsonl", "chunk mapping is not one-to-one with rag chunks")
        self._check_vector_manifest(chunks_by_id, manifest, matrix)

    def _check_vector_manifest(self, chunks: dict[str, dict[str, Any]], manifest: Any, matrix: bytes) -> None:
        if not isinstance(manifest, dict):
            self.error("vector_manifest_shape", "vector_manifest.json", "vector manifest must be an object")
            return
        info = manifest.get("matrix")
        entries = manifest.get("entries")
        if not isinstance(info, dict) or not isinstance(entries, list):
            self.error("vector_manifest_fields", "vector_manifest.json", "matrix and entries are required")
            return
        if info.get("rows") != len(entries) or info.get("rows") != len(chunks):
            self.error("vector_row_count", "vector_manifest.matrix.rows", "vector rows must equal chunk count")
        if info.get("dimensions") != VECTOR_DIMENSIONS or info.get("dtype") != VECTOR_DTYPE:
            self.error("vector_shape", "vector_manifest.matrix", "vectors must be 512-dimensional little-endian float32")
        if info.get("row_order") != "chunk_id ascending":
            self.error("vector_order", "vector_manifest.matrix.row_order", "vector order must be chunk_id ascending")
        if _norm_hash(info.get("sha256")) != _hash_bytes(matrix):
            self.error("vector_matrix_hash", "vector_manifest.matrix.sha256", "matrix SHA-256 mismatch")
        expected_bytes = len(entries) * VECTOR_DIMENSIONS * 4
        if len(matrix) != expected_bytes:
            self.error("vector_matrix_bytes", "embeddings.f32", "matrix byte length does not match rows/dimensions")
            return
        sorted_ids = sorted(chunks)
        entry_ids: list[str] = []
        for index, entry in enumerate(entries):
            location = f"vector_manifest.entries[{index}]"
            if not isinstance(entry, dict):
                self.error("vector_entry_shape", location, "vector entry is not an object")
                continue
            chunk_id = entry.get("chunk_id")
            entry_ids.append(chunk_id)
            if chunk_id != (sorted_ids[index] if index < len(sorted_ids) else None):
                self.error("vector_entry_order", location, "vector entry order is not chunk_id ascending")
            chunk = chunks.get(chunk_id)
            if chunk is None:
                self.error("vector_entry_chunk", location, "vector entry references unknown chunk")
                continue
            if _norm_hash(entry.get("input_sha256")) != _hash_text(chunk.get("search_text", "")):
                self.error("vector_input_hash", location, "vector input SHA does not hash search_text")
            row_start = index * VECTOR_DIMENSIONS * 4
            row = matrix[row_start : row_start + VECTOR_DIMENSIONS * 4]
            if _norm_hash(entry.get("vector_sha256")) != _hash_bytes(row):
                self.error("vector_row_hash", location, "vector row SHA mismatch")
            values = struct.unpack("<512f", row)
            if not all(math.isfinite(value) for value in values):
                self.error("vector_finite", location, "vector contains non-finite value")
            norm = math.sqrt(sum(value * value for value in values))
            if abs(norm - 1.0) > VECTOR_TOLERANCE:
                self.error("vector_norm", location, "vector is not L2-normalized")
        if entry_ids != sorted_ids:
            self.error("vector_entry_set", "vector_manifest.entries", "vector entries are not exactly the chunk set")

    def run(self) -> dict[str, Any]:
        if self.profile is None:
            return {
                "schema_version": "curated-bundle-validation-1.0",
                "status": "FAIL",
                "scope": "independent_candidate_bundle_validation",
                "profile": self.bundle_id,
                "errors": self.errors,
                "warnings": self.warnings,
            }
        self.load()
        self.check_public_strings()
        self.prepare_mapping()
        self.check_source_manifest()
        self.check_optional_reports()
        self.check_knowledge_units()
        self.check_evidence()
        self.check_graph()
        self.check_s001_chapter_mapping()
        self.check_external_references()
        self.check_cross_bundle_references()
        self.check_vectors()
        status = "PASS" if not self.errors else "FAIL"
        result = {
            "schema_version": "curated-bundle-validation-1.0",
            "status": status,
            "scope": "independent_candidate_bundle_validation",
            "evidence_mode": {
                "chapter_map": self.chapter is not None,
                "source_audit": self.source_audit is not None,
                "records_root": self.records_root is not None,
            },
            "vectors_required": self.require_vectors,
            "counts": {
                "sources": len(self.sources),
                "knowledge_units": len(self.kus),
                "evidence": len(self.evidence),
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "external_references": len(self.external),
            },
            "errors": self.errors,
            "warnings": self.warnings,
        }
        if self.bundle_id == S001_BUNDLE_ID:
            result["profile"] = self.bundle_id
            result["content_validation"] = "structure_only; source-content acceptance remains separate"
            result["counts"]["cross_bundle_references"] = len(self.cross_bundle)
        return result


def validate_bundle(
    bundle: str | Path,
    chapter_map: str | Path | None = None,
    source_audit: str | Path | None = None,
    records_root: str | Path | None = None,
    require_vectors: bool = False,
    bundle_id: str = OLD_BUNDLE_ID,
    related_bundle: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a bundle and return a JSON-serializable report."""

    if bundle_id not in PROFILES:
        return {
            "schema_version": "curated-bundle-validation-1.0",
            "status": "FAIL",
            "scope": "independent_candidate_bundle_validation",
            "profile": bundle_id,
            "errors": [{"code": "unknown_bundle_profile", "location": "bundle_id", "message": "bundle_id is not a predefined profile"}],
            "warnings": [],
        }
    if bundle_id == S001_BUNDLE_ID and any(value is not None for value in (chapter_map, source_audit, records_root)):
        return {
            "schema_version": "curated-bundle-validation-1.0",
            "status": "FAIL",
            "scope": "independent_candidate_bundle_validation",
            "profile": bundle_id,
            "errors": [{"code": "s001_private_evidence_unsupported", "location": "arguments", "message": "S001 does not use the legacy chapter_map/source_audit/records_root routes"}],
            "warnings": [],
        }
    return BundleValidator(
        Path(bundle),
        Path(chapter_map) if chapter_map is not None else None,
        Path(source_audit) if source_audit is not None else None,
        Path(records_root) if records_root is not None else None,
        require_vectors,
        bundle_id,
        Path(related_bundle) if related_bundle is not None else None,
    ).run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--chapter-map", type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--records-root", type=Path)
    parser.add_argument("--require-vectors", action="store_true")
    parser.add_argument("--bundle-id", choices=tuple(PROFILES), default=OLD_BUNDLE_ID)
    parser.add_argument("--related-bundle", type=Path, help="frozen old four-book bundle or knowledge_units.jsonl for S001")
    parser.add_argument("--json", action="store_true", help="emit the full JSON report (default)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_bundle(
        args.bundle,
        chapter_map=args.chapter_map,
        source_audit=args.source_audit,
        records_root=args.records_root,
        require_vectors=args.require_vectors,
        bundle_id=args.bundle_id,
        related_bundle=args.related_bundle,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
