#!/usr/bin/env python3
"""Build the deterministic first-stage four-book candidate bundle.

The converter deliberately accepts all input locations at the command line.  It
only emits the seven files defined by CONVERTER_INTERFACE.md and refuses the
accepted parent inputs when either locked input hash has drifted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_INPUT_SHA256 = {
    "chapter_mapping.json": "539f491e2eb54b3a70bbd70f5f8e7d95772aad1733af159eceda4ad87b764120",
    "source_reuse_audit.json": "6be764f1d2c5a865de3a0ca5283f19e40cc4487795aa12be19afea9dc034619d",
}
SOURCE_ORDER = ("RE01", "OC02", "TH03", "EN04")
EXPECTED_SOURCE_COUNTS = {"RE01": 14, "OC02": 13, "TH03": 25, "EN04": 17}
EXPECTED_ROOTS = {"chemical-engineering-principles", "reaction-engineering"}
ROOT_LABELS = {
    "chemical-engineering-principles": "化工原理",
    "reaction-engineering": "反应工程",
}
EXPECTED_INTERNAL_EDGE_COUNT = 8
EXPECTED_EXTERNAL_REFERENCE_COUNT = 252


class ConversionError(RuntimeError):
    """Raised for a locked-input or structural validation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read JSON input: {path}: {exc}") from exc


def stable_json(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            separators=(",", ": "),
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return (text + "\n").encode("utf-8")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, stable_json(value, pretty=True))


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    data = b"".join(stable_json(value) for value in values)
    write_bytes(path, data)


def as_dict(value: Any, context: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{context} must be an object")
    return value


def as_list(value: Any, context: str) -> list[Any]:
    require(isinstance(value, list), f"{context} must be an array")
    return value


def source_record_path(records_root: Path, source_id: str, node_id: str) -> Path:
    # The source record file name is derived from the locked node ID.  The
    # mapping's legacy relative path is checked separately, but never used as
    # an output value or as an unrestricted filesystem traversal.
    return records_root / source_id / f"{node_id}.json"


def validate_input_hash(path: Path, role: str) -> str:
    actual = sha256_file(path)
    expected = EXPECTED_INPUT_SHA256[role]
    require(actual == expected, f"locked input hash mismatch for {role}")
    return actual


def validate_audit_sources(
    audit: dict[str, Any],
    mapping_volumes: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    summary = as_dict(audit.get("summary"), "source audit summary")
    require(summary.get("source_count") == 4, "source audit source count is not 4")
    require(summary.get("formal_record_count") == 69, "source audit record count is not 69")
    require(summary.get("expected_formal_record_count") == 69, "source audit expected record count is not 69")
    require(summary.get("all_formal_record_counts_match") is True, "source audit count gate is false")
    require(summary.get("all_pdf_bytes_sha_pages_ok") is True, "source audit PDF asset gate is false")
    require(summary.get("all_body_audits_ok") is True, "source audit body gate is false")

    entries = as_list(audit.get("sources"), "source audit sources")
    by_source: dict[str, dict[str, Any]] = {}
    audit_records: dict[str, dict[str, Any]] = {}
    for entry_value in entries:
        entry = as_dict(entry_value, "source audit source")
        source_id = entry.get("source_id")
        require(source_id in SOURCE_ORDER, "source audit contains an unexpected source ID")
        require(source_id not in by_source, "duplicate source audit source ID")
        by_source[source_id] = entry

        volume = mapping_volumes[source_id]
        asset = as_dict(entry.get("asset"), f"{source_id} asset")
        readability = as_dict(asset.get("pdf_readability"), f"{source_id} PDF readability")
        body_audit = as_dict(entry.get("body_audit"), f"{source_id} body audit")
        require(asset.get("actual_bytes") == asset.get("expected_bytes"), f"{source_id} asset bytes mismatch")
        require(asset.get("actual_sha256") == asset.get("expected_sha256"), f"{source_id} asset hash mismatch")
        require(asset.get("actual_sha256") == volume.get("source_sha256"), f"{source_id} asset/map hash mismatch")
        require(readability.get("page_count") == asset.get("expected_pages"), f"{source_id} PDF page count mismatch")
        require(readability.get("is_encrypted") is False, f"{source_id} PDF is encrypted")
        require(body_audit.get("formal_record_count") == EXPECTED_SOURCE_COUNTS[source_id], f"{source_id} record count mismatch")
        require(body_audit.get("candidate_record_count") == EXPECTED_SOURCE_COUNTS[source_id], f"{source_id} candidate count mismatch")
        require(body_audit.get("approved_node_count") == EXPECTED_SOURCE_COUNTS[source_id], f"{source_id} approved count mismatch")
        for key in (
            "count_matches_expected",
            "all_body_hashes_match_recomputed",
            "all_body_hashes_match_candidate",
            "all_body_hashes_match_approved",
            "all_source_ids_match",
            "all_source_hashes_match_asset",
            "all_candidate_sources_match_formal",
        ):
            require(body_audit.get(key) is True, f"{source_id} body audit flag is false: {key}")

        source_records = as_list(entry.get("records"), f"{source_id} audit records")
        require(len(source_records) == EXPECTED_SOURCE_COUNTS[source_id], f"{source_id} audit record array count mismatch")
        for record_value in source_records:
            record = as_dict(record_value, f"{source_id} audit record")
            node_id = record.get("node_id")
            require(isinstance(node_id, str) and node_id, f"{source_id} audit record has no node ID")
            require(node_id not in audit_records, "duplicate audit record node ID")
            audit_records[node_id] = record

    require(set(by_source) == set(SOURCE_ORDER), "source audit IDs do not match the locked four sources")
    require(len(audit_records) == 69, "source audit record IDs are not unique 69 records")
    return by_source, audit_records


def validate_related_audit(
    audit: dict[str, Any],
    records: list[dict[str, Any]],
    node_ids: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    internal: list[tuple[str, str]] = []
    external: list[tuple[str, str]] = []
    by_source: dict[str, dict[str, int]] = {
        source_id: {"internal": 0, "external": 0} for source_id in SOURCE_ORDER
    }
    internal_targets: set[str] = set()
    external_targets: set[str] = set()

    for item in records:
        record = item["record"]
        node_id = record["node_id"]
        related = as_list(record.get("related_node_ids"), f"{node_id}.related_node_ids")
        require(len(related) == len(set(related)), f"duplicate related_node_ids in {node_id}")
        for target_value in related:
            require(isinstance(target_value, str) and target_value, f"invalid related target in {node_id}")
            if target_value in node_ids:
                internal.append((node_id, target_value))
                internal_targets.add(target_value)
                by_source[item["source_id"]]["internal"] += 1
            else:
                external.append((node_id, target_value))
                external_targets.add(target_value)
                by_source[item["source_id"]]["external"] += 1

    relation_audit = as_dict(audit.get("related_node_audit"), "related node audit")
    require(relation_audit.get("record_node_count") == 69, "related audit record count mismatch")
    require(relation_audit.get("internal_unique_count") == len(internal_targets) == 6, "internal unique target count mismatch")
    require(relation_audit.get("external_unique_count") == len(external_targets) == 115, "external unique target count mismatch")
    require(relation_audit.get("internal_edge_count") == len(internal) == EXPECTED_INTERNAL_EDGE_COUNT, "internal edge count mismatch")
    require(relation_audit.get("external_edge_count") == len(external) == EXPECTED_EXTERNAL_REFERENCE_COUNT, "external edge count mismatch")
    require(set(relation_audit.get("internal_node_ids", [])) == internal_targets, "internal target inventory mismatch")
    require(set(relation_audit.get("external_node_ids", [])) == external_targets, "external target inventory mismatch")
    audit_by_source = as_dict(relation_audit.get("by_source"), "related audit by source")
    for source_id in SOURCE_ORDER:
        source_audit = as_dict(audit_by_source.get(source_id), f"related audit {source_id}")
        require(source_audit.get("record_count") == EXPECTED_SOURCE_COUNTS[source_id], f"related audit {source_id} record count")
        require(source_audit.get("internal_edge_count") == by_source[source_id]["internal"], f"related audit {source_id} internal count")
        require(source_audit.get("external_edge_count") == by_source[source_id]["external"], f"related audit {source_id} external count")
        require(set(source_audit.get("internal_node_ids", [])) == {target for source, target in internal if source.startswith(source_id + "-")}, f"related audit {source_id} internal inventory")
        require(set(source_audit.get("external_node_ids", [])) == {target for source, target in external if source.startswith(source_id + "-")}, f"related audit {source_id} external inventory")
    return internal, external


def build_bundle(
    mapping_path: Path,
    audit_path: Path,
    records_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    mapping_hash = validate_input_hash(mapping_path, "chapter_mapping.json")
    audit_hash = validate_input_hash(audit_path, "source_reuse_audit.json")
    mapping = as_dict(load_json(mapping_path), "chapter mapping")
    audit = as_dict(load_json(audit_path), "source reuse audit")

    require(mapping.get("record_count") == 69, "chapter mapping record count is not 69")
    mapping_records = as_list(mapping.get("records"), "chapter mapping records")
    require(len(mapping_records) == 69, "chapter mapping record array is not 69")
    mapping_volumes_list = as_list(mapping.get("source_volumes"), "chapter mapping source volumes")
    require(len(mapping_volumes_list) == 4, "chapter mapping does not have four source volumes")
    mapping_volumes = {as_dict(item, "source volume")["source_id"]: as_dict(item, "source volume") for item in mapping_volumes_list}
    require(set(mapping_volumes) == set(SOURCE_ORDER), "chapter mapping source IDs do not match the locked four sources")
    require(len(mapping_volumes) == len(mapping_volumes_list), "duplicate source volume IDs")
    for source_id in SOURCE_ORDER:
        require(mapping_volumes[source_id].get("record_count") == EXPECTED_SOURCE_COUNTS[source_id], f"mapping {source_id} count mismatch")
        require(mapping_volumes[source_id].get("kg_subject_root") in EXPECTED_ROOTS, f"mapping {source_id} root invalid")

    audit_sources, audit_records = validate_audit_sources(audit, mapping_volumes)
    mapping_by_node: dict[str, dict[str, Any]] = {}
    for value in mapping_records:
        item = as_dict(value, "chapter mapping record")
        node_id = item.get("original_node_id")
        require(isinstance(node_id, str) and node_id, "chapter mapping record has no original node ID")
        require(node_id not in mapping_by_node, "duplicate chapter mapping node ID")
        mapping_by_node[node_id] = item
    require(len(mapping_by_node) == 69, "chapter mapping node IDs are not unique 69 records")

    ordered_mapping = sorted(
        mapping_by_node.values(),
        key=lambda item: (SOURCE_ORDER.index(item["source_id"]), item["original_node_id"]),
    )
    actual_records: list[dict[str, Any]] = []
    for map_item in ordered_mapping:
        source_id = map_item["source_id"]
        node_id = map_item["original_node_id"]
        require(map_item.get("kg_subject_root") == mapping_volumes[source_id].get("kg_subject_root"), f"{node_id} root mismatch")
        require(map_item.get("source_volume_id") == mapping_volumes[source_id].get("source_volume_id"), f"{node_id} volume mismatch")
        require(map_item.get("source_sha256") == mapping_volumes[source_id].get("source_sha256"), f"{node_id} mapping source hash mismatch")
        record_path = source_record_path(records_root, source_id, node_id)
        require(record_path.is_file(), f"missing source record for {node_id}")
        record_file_hash = sha256_file(record_path)
        evidence = as_dict(map_item.get("evidence"), f"{node_id} mapping evidence")
        require(record_file_hash == evidence.get("source_record_file_sha256"), f"source record file hash mismatch for {node_id}")
        record = as_dict(load_json(record_path), f"source record {node_id}")
        audit_record = audit_records.get(node_id)
        require(audit_record is not None, f"missing source audit record for {node_id}")
        source = as_dict(record.get("source"), f"{node_id}.source")
        require(record.get("node_id") == node_id, f"source record node ID mismatch for {node_id}")
        require(record.get("source_id", source.get("source_id")) == source_id, f"source record source ID mismatch for {node_id}")
        require(source.get("source_id") == source_id, f"nested source ID mismatch for {node_id}")
        require(source.get("sha256") == map_item.get("source_sha256"), f"source record source hash mismatch for {node_id}")
        require(source.get("locator") == map_item.get("source_page_range"), f"source record locator mismatch for {node_id}")
        require(record.get("title") == map_item.get("title"), f"title mismatch for {node_id}")
        require(record.get("text") == map_item.get("body"), f"body mismatch for {node_id}")
        body_hash = sha256_bytes(record["text"].encode("utf-8"))
        require(body_hash == record.get("text_sha256") == map_item.get("body_sha256"), f"body hash mismatch for {node_id}")
        require(audit_record.get("body_sha256") == body_hash, f"audit body hash mismatch for {node_id}")
        require(audit_record.get("source_sha256") == source.get("sha256"), f"audit source hash mismatch for {node_id}")
        require(audit_record.get("source_locator") == source.get("locator"), f"audit locator mismatch for {node_id}")
        for field, expected in (
            ("content_available", True),
            ("retrieval_eligible", True),
            ("source_payload_available", False),
            ("current_project_authority", False),
            ("project_value_transfer_allowed", False),
        ):
            require(record.get(field) is expected, f"{node_id}.{field} is not the locked value")
        intervals = as_list(map_item.get("source_physical_page_intervals"), f"{node_id} physical page intervals")
        require(intervals, f"{node_id} has no physical page intervals")
        total_pages = audit_sources[source_id]["asset"]["pdf_readability"]["page_count"]
        for interval_value in intervals:
            interval = as_dict(interval_value, f"{node_id} physical page interval")
            require(isinstance(interval.get("start"), int) and isinstance(interval.get("end"), int), f"{node_id} page interval is not integral")
            require(1 <= interval["start"] <= interval["end"] <= total_pages, f"{node_id} page interval is outside source PDF")
        group = as_dict(map_item.get("chapter_or_support_unit"), f"{node_id} chapter/support unit")
        require(group.get("type") in ("chapter", "cross_chapter_support_unit"), f"{node_id} group type invalid")
        chapter_refs = as_list(group.get("chapter_refs"), f"{node_id} chapter refs")
        require(chapter_refs, f"{node_id} has no chapter refs")
        actual_records.append(
            {
                "source_id": source_id,
                "map": map_item,
                "record": record,
                "record_file_hash": record_file_hash,
                "audit_record": audit_record,
                "audit_source": audit_sources[source_id],
            }
        )

    counts = Counter(item["source_id"] for item in actual_records)
    require(dict(counts) == EXPECTED_SOURCE_COUNTS, "actual source record counts do not match the locked counts")
    node_ids = {item["record"]["node_id"] for item in actual_records}
    internal_relations, external_relations = validate_related_audit(audit, actual_records, node_ids)

    group_by_id: dict[str, dict[str, Any]] = {}
    group_source: dict[str, str] = {}
    chapter_nodes: dict[str, dict[str, Any]] = {}
    support_ids: set[str] = set()
    for item in actual_records:
        map_item = item["map"]
        source_id = item["source_id"]
        group = as_dict(map_item["chapter_or_support_unit"], f"{item['record']['node_id']} group")
        group_id = group["id"]
        if group_id in group_by_id:
            require(group_by_id[group_id] == group, f"inconsistent group definition for {group_id}")
        else:
            group_by_id[group_id] = copy.deepcopy(group)
            group_source[group_id] = source_id
        if group["type"] == "cross_chapter_support_unit":
            support_ids.add(group_id)
        for chapter_value in group["chapter_refs"]:
            chapter = as_dict(chapter_value, f"{item['record']['node_id']} chapter ref")
            chapter_id = chapter.get("id")
            require(isinstance(chapter_id, str) and chapter_id, f"{item['record']['node_id']} chapter ref ID missing")
            if chapter_id in chapter_nodes:
                require(chapter_nodes[chapter_id]["chapter"] == chapter and chapter_nodes[chapter_id]["source_id"] == source_id, f"inconsistent chapter definition for {chapter_id}")
            else:
                chapter_nodes[chapter_id] = {"chapter": copy.deepcopy(chapter), "source_id": source_id}
    require(len(support_ids) == 4, "expected exactly four cross-chapter support groups")

    roots = sorted({mapping_volumes[source_id]["kg_subject_root"] for source_id in SOURCE_ORDER})
    require(set(roots) == EXPECTED_ROOTS and len(roots) == 2, "expected exactly two subject roots")

    source_manifest_sources: list[dict[str, Any]] = []
    knowledge_units: list[dict[str, Any]] = []
    evidence_registry: list[dict[str, Any]] = []
    for source_id in SOURCE_ORDER:
        volume = mapping_volumes[source_id]
        audit_source = audit_sources[source_id]
        receipts = as_dict(audit_source.get("receipts"), f"{source_id} receipts")
        reviewer = as_dict(receipts.get("reviewer_proof"), f"{source_id} reviewer proof")
        producer_proof = as_dict(receipts.get("producer_proof_hashes"), f"{source_id} producer proof")
        bibliography = copy.deepcopy(as_dict(audit_source.get("bibliography"), f"{source_id} bibliography"))
        source_manifest_sources.append(
            {
                "source_id": source_id,
                "bibliography": bibliography,
                "source_sha256": volume["source_sha256"],
                "bytes": audit_source["asset"]["actual_bytes"],
                "total_pdf_pages": audit_source["asset"]["pdf_readability"]["page_count"],
                "is_encrypted": False,
                "source_volume_id": volume["source_volume_id"],
                "subject_root": volume["kg_subject_root"],
                "selected_knowledge_units_count": counts[source_id],
                "source_payload_public": False,
                "full_source_digitization_complete": False,
                "review_provenance": {
                    "producer_id": receipts["producer_id"],
                    "reviewer_id": reviewer["reviewer_id"],
                    "content_review_file_sha256": receipts["content_review_sha256_actual"],
                    "candidate_manifest_sha256": producer_proof["candidate_manifest_sha256"],
                    "locked_candidate_part_hashes": copy.deepcopy(producer_proof["locked_candidate_part_hashes"]),
                    "agent_review": True,
                    "human_review": False,
                },
            }
        )

    for item in actual_records:
        source_id = item["source_id"]
        map_item = item["map"]
        record = item["record"]
        group_id = map_item["chapter_or_support_unit"]["id"]
        review_hash = item["audit_source"]["receipts"]["content_review_sha256_actual"]
        evidence_id = f"evidence:{record['node_id']}:source-locator"
        evidence_registry.append(
            {
                "evidence_id": evidence_id,
                "type": "source_document_locator",
                "source_id": source_id,
                "source_sha256": record["source"]["sha256"],
                "pdf_pages": copy.deepcopy(map_item["source_physical_page_intervals"]),
                "locator": record["source"]["locator"],
                "reviewed_text_sha256": record["text_sha256"],
                "review_proof_sha256": review_hash,
                "page_media_sha256": None,
                "public_source_payload": False,
            }
        )
        knowledge_units.append(
            {
                "node_id": record["node_id"],
                "knowledge_unit_id": record["node_id"],
                "title": record["title"],
                "text": record["text"],
                "text_sha256": record["text_sha256"],
                "applicability": copy.deepcopy(record["applicability"]),
                "units_basis": record["units_basis"],
                "knowledge_layer": record["knowledge_layer"],
                "source_id": source_id,
                "source_sha256": record["source"]["sha256"],
                "source_volume_id": map_item["source_volume_id"],
                "subject_root": map_item["kg_subject_root"],
                "package_id": group_id,
                "source_chain_id": f"curated-four-books-v1:{source_id}",
                "source_locator": record["source"]["locator"],
                "evidence_refs": [evidence_id],
                "forbidden_transfer": copy.deepcopy(record["forbidden_transfer"]),
                "project_value_transfer_allowed": False,
                "current_project_authority": False,
                "content_available": True,
                "retrieval_eligible": True,
                "embedding_eligible": True,
                "embedding_exclusion_reason": None,
                "embedding_eligibility_basis": {
                    "content_type": "authored_reviewed_explanation",
                    "source_review_proof_sha256": review_hash,
                    "public_scope": "authored_explanations_only",
                },
            }
        )

    kg_nodes: list[dict[str, Any]] = []
    for root in roots:
        kg_nodes.append({"node_id": root, "node_type": "subject_root", "label": ROOT_LABELS[root]})
    for source_id in SOURCE_ORDER:
        volume = mapping_volumes[source_id]
        kg_nodes.append(
            {
                "node_id": volume["source_volume_id"],
                "node_type": "source_volume",
                "label": volume["label"],
                "subject_root": volume["kg_subject_root"],
                "source_id": source_id,
                "source_volume_id": volume["source_volume_id"],
            }
        )
    for chapter_id in sorted(chapter_nodes):
        data = chapter_nodes[chapter_id]
        source_id = data["source_id"]
        volume = mapping_volumes[source_id]
        kg_nodes.append(
            {
                "node_id": chapter_id,
                "node_type": "chapter",
                "label": data["chapter"]["label"],
                "chapter_number": data["chapter"]["number"],
                "subject_root": volume["kg_subject_root"],
                "source_id": source_id,
                "source_volume_id": volume["source_volume_id"],
            }
        )
    for group_id in sorted(support_ids):
        group = group_by_id[group_id]
        source_id = group_source[group_id]
        volume = mapping_volumes[source_id]
        kg_nodes.append(
            {
                "node_id": group_id,
                "node_type": "cross_chapter_support_unit",
                "label": group["label"],
                "chapter_refs": [chapter["id"] for chapter in group["chapter_refs"]],
                "subject_root": volume["kg_subject_root"],
                "source_id": source_id,
                "source_volume_id": volume["source_volume_id"],
            }
        )
    for item in actual_records:
        record = item["record"]
        map_item = item["map"]
        kg_nodes.append(
            {
                "node_id": record["node_id"],
                "node_type": "knowledge_unit",
                "label": record["title"],
                "subject_root": map_item["kg_subject_root"],
                "source_id": item["source_id"],
                "source_volume_id": map_item["source_volume_id"],
                "knowledge_unit_id": record["node_id"],
            }
        )
    for evidence in evidence_registry:
        item = next(record for record in actual_records if record["record"]["node_id"] == evidence["evidence_id"].split(":", 2)[1])
        map_item = item["map"]
        kg_nodes.append(
            {
                "node_id": evidence["evidence_id"],
                "node_type": "evidence",
                "label": evidence["locator"],
                "subject_root": map_item["kg_subject_root"],
                "source_id": item["source_id"],
                "source_volume_id": map_item["source_volume_id"],
            }
        )

    kg_edges: list[dict[str, Any]] = []

    def add_edge(source_node_id: str, target_node_id: str, relation: str, suffix: str | None = None) -> None:
        edge_id = f"edge:{source_node_id}:{relation}:{target_node_id}"
        if suffix is not None:
            edge_id = f"{edge_id}:{suffix}"
        kg_edges.append(
            {
                "edge_id": edge_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "relation": relation,
            }
        )

    for source_id in SOURCE_ORDER:
        volume = mapping_volumes[source_id]
        add_edge(volume["kg_subject_root"], volume["source_volume_id"], "contains")
    for group_id in sorted(group_by_id):
        source_id = group_source[group_id]
        add_edge(mapping_volumes[source_id]["source_volume_id"], group_id, "contains")
    for group_id in sorted(support_ids):
        group = group_by_id[group_id]
        for chapter in sorted(group["chapter_refs"], key=lambda value: value["id"]):
            add_edge(group_id, chapter["id"], "has_chapter_context")
    for item in actual_records:
        group_id = item["map"]["chapter_or_support_unit"]["id"]
        add_edge(group_id, item["record"]["node_id"], "contains")
    for item in actual_records:
        node_id = item["record"]["node_id"]
        add_edge(node_id, f"evidence:{node_id}:source-locator", "supported_by")
    for source_node_id, target_node_id in internal_relations:
        add_edge(source_node_id, target_node_id, "related_to")

    external_references = [
        {
            "from_knowledge_unit_id": source_node_id,
            "target_project": "chemical-engineering-skills",
            "target_node_id": target_node_id,
            "relation": "related_to",
        }
        for source_node_id, target_node_id in external_relations
    ]

    node_id_set = {node["node_id"] for node in kg_nodes}
    require(len(node_id_set) == len(kg_nodes), "KG node IDs are not unique")
    edge_id_set = {edge["edge_id"] for edge in kg_edges}
    require(len(edge_id_set) == len(kg_edges), "KG edge IDs are not unique")
    for edge in kg_edges:
        require(edge["source_node_id"] in node_id_set, f"KG edge source is not closed: {edge['edge_id']}")
        require(edge["target_node_id"] in node_id_set, f"KG edge target is not closed: {edge['edge_id']}")
    require(len(external_references) == EXPECTED_EXTERNAL_REFERENCE_COUNT, "external reference count is not 252")
    require(len(knowledge_units) == 69 and len(evidence_registry) == 69, "KU/evidence output count is not 69")

    source_manifest = {
        "schema_version": "curated-selected-knowledge-1.0",
        "scope": "authored-selected-knowledge-preview",
        "canonical_project": "chemical-engineering-rag-kg",
        "sources": source_manifest_sources,
    }
    output_values: dict[str, bytes] = {
        "source_manifest.json": stable_json(source_manifest, pretty=True),
        "knowledge_units.jsonl": b"".join(stable_json(value) for value in knowledge_units),
        "evidence_registry.jsonl": b"".join(stable_json(value) for value in evidence_registry),
        "kg_nodes.jsonl": b"".join(stable_json(value) for value in kg_nodes),
        "kg_edges.jsonl": b"".join(stable_json(value) for value in kg_edges),
        "external_references.jsonl": b"".join(stable_json(value) for value in external_references),
    }

    body_identity = [
        {
            "node_id": item["record"]["node_id"],
            "source_id": item["source_id"],
            "source_record_file_sha256": item["record_file_hash"],
            "body_sha256": item["record"]["text_sha256"],
            "body_match": item["record"]["text"] == item["map"]["body"] and item["record"]["text_sha256"] == item["map"]["body_sha256"],
            "source_match": item["record"]["source"]["sha256"] == item["map"]["source_sha256"],
        }
        for item in actual_records
    ]
    require(all(item["body_match"] and item["source_match"] for item in body_identity), "body identity verification failed")
    output_hashes = {name: sha256_bytes(data) for name, data in output_values.items()}
    source_counts = {source_id: counts[source_id] for source_id in SOURCE_ORDER}
    report = {
        "status": "candidate_converted",
        "production_activated": False,
        "independent_audit_claimed": False,
        "human_approved_anchor_count": 0,
        "vector_outputs_written": False,
        "parent_input_hashes": {
            "chapter_mapping.json": {"accepted_sha256": EXPECTED_INPUT_SHA256["chapter_mapping.json"], "actual_sha256": mapping_hash},
            "source_reuse_audit.json": {"accepted_sha256": EXPECTED_INPUT_SHA256["source_reuse_audit.json"], "actual_sha256": audit_hash},
        },
        "counts": {
            "sources": 4,
            "subject_roots": len(roots),
            "source_volumes": 4,
            "chapters": len(chapter_nodes),
            "cross_chapter_support_units": len(support_ids),
            "knowledge_units": len(knowledge_units),
            "evidence_records": len(evidence_registry),
            "kg_nodes": len(kg_nodes),
            "kg_edges": len(kg_edges),
            "internal_related_edges": len(internal_relations),
            "external_references": len(external_references),
            "source_record_counts": source_counts,
        },
        "body_identity_verification": body_identity,
        "output_file_hashes": output_hashes,
        "output_file_count": 7,
        "report_file_hash_embedded": False,
        "command_result": {
            "status": "ok",
            "exit_code": 0,
            "arguments_supplied": ["--mapping", "--source-audit", "--records-root", "--output-dir"],
        },
    }
    output_values["conversion_report.json"] = stable_json(report, pretty=True)

    # This is the sole write phase.  Every input and graph closure check above
    # runs before any candidate output is changed.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in output_values.items():
        write_bytes(output_dir / name, data)
    return report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the locked four-book candidate bundle.")
    parser.add_argument("--mapping", required=True, help="Accepted chapter_mapping.json")
    parser.add_argument("--source-audit", required=True, help="Accepted source_reuse_audit.json")
    parser.add_argument("--records-root", required=True, help="Read-only source_records directory")
    parser.add_argument("--output-dir", required=True, help="Candidate output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = build_bundle(
            Path(args.mapping),
            Path(args.source_audit),
            Path(args.records_root),
            Path(args.output_dir),
        )
    except ConversionError as exc:
        print(f"conversion refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "knowledge_units": report["counts"]["knowledge_units"],
                "internal_related_edges": report["counts"]["internal_related_edges"],
                "external_references": report["counts"]["external_references"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
