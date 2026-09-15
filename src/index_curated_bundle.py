"""Build a private candidate retrieval/vector bundle for curated knowledge units.

The builder accepts only a converted, explicit-eligibility KU JSONL file and
writes the four bundle payload files.  It never mutates the source KU file,
uses no hash-vector fallback, and never writes a production activation flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from curated_vectors import CuratedOnnxEncoder, split_record


MODEL_REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
DIMENSIONS = 512
NORM_TOLERANCE = 0.00001
REQUIRED_KU_FIELDS = {
    "node_id",
    "knowledge_unit_id",
    "title",
    "text",
    "text_sha256",
    "applicability",
    "units_basis",
    "knowledge_layer",
    "source_id",
    "source_sha256",
    "source_volume_id",
    "subject_root",
    "package_id",
    "source_chain_id",
    "source_locator",
    "evidence_refs",
    "forbidden_transfer",
    "project_value_transfer_allowed",
    "current_project_authority",
    "content_available",
    "retrieval_eligible",
    "embedding_eligible",
    "embedding_exclusion_reason",
    "embedding_eligibility_basis",
}
OUTPUT_NAMES = (
    "rag_chunks.jsonl",
    "chunk_to_kg.jsonl",
    "embeddings.f32",
    "vector_manifest.json",
)
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9])[A-Z]:[\\/]|\\\\(?:users|home|private|documents)[\\/]"
)


class BundleBuildError(RuntimeError):
    """Raised for any input, identity, or output-contract violation."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def _ensure_no_private_paths(value: Any, location: str) -> None:
    if isinstance(value, str):
        if PRIVATE_PATH_PATTERN.search(value):
            raise BundleBuildError(f"private absolute path in {location}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _ensure_no_private_paths(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_no_private_paths(item, f"{location}[{index}]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BundleBuildError(f"missing knowledge unit file: {path.name}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleBuildError(
                f"invalid JSON at {path.name}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise BundleBuildError(f"non-object JSON at {path.name}:{line_number}")
        records.append(value)
    return records


def _require_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise BundleBuildError(f"{field} must be a non-empty string")
    return value


def _validate_ku(record: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KU_FIELDS - record.keys())
    if missing:
        raise BundleBuildError(
            f"KU {record.get('node_id', '<unknown>')} missing fields: {missing}"
        )
    node_id = _require_string(record, "node_id")
    if record.get("knowledge_unit_id") != node_id:
        raise BundleBuildError(f"KU {node_id} has mismatched knowledge_unit_id")
    _require_string(record, "title")
    _require_string(record, "text")
    _require_string(record, "units_basis")
    _require_string(record, "knowledge_layer")
    _require_string(record, "source_id")
    source_sha = _require_string(record, "source_sha256")
    if len(source_sha) != 64 or any(
        char not in "0123456789abcdef" for char in source_sha
    ):
        raise BundleBuildError(f"KU {node_id} source_sha256 is not lowercase hex")
    _require_string(record, "source_volume_id")
    subject_root = _require_string(record, "subject_root")
    if subject_root not in {
        "chemical-engineering-principles",
        "reaction-engineering",
    }:
        raise BundleBuildError(f"KU {node_id} has unknown subject_root")
    _require_string(record, "package_id")
    source_chain_id = _require_string(record, "source_chain_id")
    if not source_chain_id.startswith("curated-four-books-v1:"):
        raise BundleBuildError(f"KU {node_id} has invalid source_chain_id")
    _require_string(record, "source_locator")
    if not isinstance(record.get("evidence_refs"), list) or not record["evidence_refs"]:
        raise BundleBuildError(f"KU {node_id} evidence_refs must be a non-empty array")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in record["evidence_refs"]
    ):
        raise BundleBuildError(f"KU {node_id} evidence_refs contain a non-string")
    if not isinstance(record.get("embedding_eligibility_basis"), dict) or not record[
        "embedding_eligibility_basis"
    ]:
        raise BundleBuildError(
            f"KU {node_id} embedding_eligibility_basis must be non-empty"
        )
    for field in ("content_type", "public_scope", "source_review_proof_sha256"):
        _require_string(record["embedding_eligibility_basis"], field)
    for field in (
        "content_available",
        "retrieval_eligible",
        "embedding_eligible",
    ):
        if record.get(field) is not True:
            raise BundleBuildError(f"KU {node_id} has {field} != true")
    if record.get("embedding_exclusion_reason") is not None:
        raise BundleBuildError(
            f"KU {node_id} has an exclusion reason while embedding-eligible"
        )
    if record.get("project_value_transfer_allowed") is not False:
        raise BundleBuildError(f"KU {node_id} project value transfer is not false")
    if record.get("current_project_authority") is not False:
        raise BundleBuildError(f"KU {node_id} current project authority is not false")
    if not isinstance(record["text_sha256"], str) or len(record["text_sha256"]) != 64:
        raise BundleBuildError(f"KU {node_id} has invalid text_sha256")
    if any(char not in "0123456789abcdef" for char in record["text_sha256"]):
        raise BundleBuildError(f"KU {node_id} text_sha256 is not lowercase hex")
    actual_text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
    if actual_text_hash != record["text_sha256"]:
        raise BundleBuildError(f"KU {node_id} text_sha256 does not match text")
    if not isinstance(record["applicability"], (str, list, tuple)):
        raise BundleBuildError(f"KU {node_id} has invalid applicability")
    _ensure_no_private_paths(record, f"KU {node_id}")


def _canonical_ku_identity(records: Iterable[dict[str, Any]]) -> str:
    canonical = b"\n".join(
        _json_bytes(record)
        for record in sorted(records, key=lambda item: item["knowledge_unit_id"])
    )
    return _sha256_bytes(canonical)


def _load_lock(lock_path: Path) -> dict[str, Any]:
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleBuildError(f"cannot read model lock: {lock_path}") from exc
    if lock.get("model_version") != "BAAI/bge-small-zh-v1.5":
        raise BundleBuildError("model lock is not BAAI/bge-small-zh-v1.5")
    if lock.get("revision") != MODEL_REVISION:
        raise BundleBuildError("model lock revision does not match curated contract")
    hashes = lock.get("model_dir_sha256")
    if not isinstance(hashes, dict) or set(hashes) != {
        "config.json",
        "model_optimized.onnx",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }:
        raise BundleBuildError("model lock does not contain exactly six model hashes")
    return lock


def _relative_sha(path: Path) -> str:
    return _sha256_file(path)


def _check_existing_identity(
    bundle: Path,
    *,
    input_identity: str,
    input_file_sha: str,
    lock: dict[str, Any],
) -> None:
    existing = bundle / "vector_manifest.json"
    existing_outputs = [bundle / name for name in OUTPUT_NAMES if (bundle / name).exists()]
    if not existing.exists():
        if existing_outputs:
            raise BundleBuildError(
                "partial output exists without vector_manifest; refusing reuse"
            )
        return
    try:
        manifest = json.loads(existing.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleBuildError("existing vector_manifest is not valid JSON") from exc
    old_input = manifest.get("input", {})
    old_model = manifest.get("model", {})
    if old_input.get("knowledge_units_identity_sha256") != input_identity:
        raise BundleBuildError("existing bundle identity differs; refusing silent reuse")
    if old_input.get("knowledge_units_sha256") != input_file_sha:
        raise BundleBuildError(
            "knowledge_units file identity differs; refusing silent reuse"
        )
    if old_model.get("revision") != lock["revision"]:
        raise BundleBuildError("existing model revision differs; refusing reuse")
    if old_model.get("files") != lock["model_dir_sha256"]:
        raise BundleBuildError("existing model hashes differ; refusing reuse")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rendered = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(rendered, encoding="utf-8", newline="\n")


def _sanitized_model_report(
    encoder: CuratedOnnxEncoder, lock: dict[str, Any]
) -> dict[str, Any]:
    return {
        "name": lock["model_version"],
        "revision": lock["revision"],
        "files": dict(lock["model_dir_sha256"]),
        "dimensions": DIMENSIONS,
        "dtype": "<f4",
        "normalization": "L2",
        "pooling": "CLS for rank-3 output; passthrough for rank-2 output",
        "provider": "CPUExecutionProvider",
        "threads": encoder.threads,
        "batch_size": encoder.batch_size,
    }


def build_bundle(
    bundle: str | Path,
    *,
    model_dir: str | Path,
    lock_path: str | Path,
    vendor_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the four candidate vector-bundle files under bundle."""

    bundle_path = Path(bundle).resolve()
    bundle_path.mkdir(parents=True, exist_ok=True)
    lock_file = Path(lock_path).resolve()
    lock = _load_lock(lock_file)
    ku_path = bundle_path / "knowledge_units.jsonl"
    ku_file_sha = _sha256_file(ku_path) if ku_path.is_file() else ""
    records = _read_jsonl(ku_path)
    if not records:
        raise BundleBuildError("knowledge_units.jsonl is empty")
    seen: set[str] = set()
    for record in records:
        _validate_ku(record)
        node_id = record["knowledge_unit_id"]
        if node_id in seen:
            raise BundleBuildError(f"duplicate knowledge_unit_id: {node_id}")
        seen.add(node_id)
    records = sorted(records, key=lambda item: item["knowledge_unit_id"])
    identity = _canonical_ku_identity(records)
    _check_existing_identity(
        bundle_path,
        input_identity=identity,
        input_file_sha=ku_file_sha,
        lock=lock,
    )

    encoder = CuratedOnnxEncoder(
        model_dir,
        lock_path=lock_file,
        vendor_path=vendor_path,
    )
    chunk_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for record in records:
        segments = split_record(record, encoder.tokenizer)
        for segment in segments:
            index = int(segment["segment_index"]) + 1
            if index > 999:
                raise BundleBuildError(
                    f"KU {record['knowledge_unit_id']} has more than 999 chunks"
                )
            chunk_id = f"{record['knowledge_unit_id']}:chunk:{index:03d}"
            chunk = dict(record)
            chunk.update(
                {
                    "chunk_id": chunk_id,
                    "search_text": segment["embedding_text"],
                    "embedding_text": segment["embedding_text"],
                    "body_start": segment["body_start"],
                    "body_end": segment["body_end"],
                    "full_text_sha256": record["text_sha256"],
                    "token_count": segment["token_count"],
                    "segment_index": segment["segment_index"],
                    "segment_count": segment["segment_count"],
                }
            )
            if chunk["search_text"] != segment["embedding_text"]:
                raise BundleBuildError(f"search_text mismatch for {chunk_id}")
            _ensure_no_private_paths(chunk, f"chunk {chunk_id}")
            chunk_rows.append(chunk)
            mapping_rows.append(
                {
                    "chunk_id": chunk_id,
                    "knowledge_unit_id": record["knowledge_unit_id"],
                }
            )

    chunk_rows.sort(key=lambda item: item["chunk_id"])
    mapping_rows.sort(key=lambda item: item["chunk_id"])
    chunk_ids = [row["chunk_id"] for row in chunk_rows]
    if chunk_ids != sorted(chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
        raise BundleBuildError("chunk IDs are not unique sorted identifiers")
    vectors = encoder.encode_texts([row["search_text"] for row in chunk_rows])
    vectors = np.asarray(vectors, dtype="<f4", order="C")
    if vectors.shape != (len(chunk_rows), DIMENSIONS):
        raise BundleBuildError(f"unexpected matrix shape: {vectors.shape}")
    if not np.all(np.isfinite(vectors)):
        raise BundleBuildError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0), initial=0.0))
    if max_norm_error > NORM_TOLERANCE:
        raise BundleBuildError(f"embedding norm error {max_norm_error} exceeds tolerance")

    matrix_bytes = vectors.astype("<f4", copy=False).tobytes(order="C")
    entries = []
    for row, vector in zip(chunk_rows, vectors, strict=True):
        vector_bytes = np.asarray(vector, dtype="<f4").tobytes(order="C")
        input_hash = _sha256_bytes(row["search_text"].encode("utf-8"))
        entries.append(
            {
                "row": len(entries),
                "chunk_id": row["chunk_id"],
                "knowledge_unit_id": row["knowledge_unit_id"],
                "input_sha256": input_hash,
                "vector_sha256": _sha256_bytes(vector_bytes),
            }
        )

    rag_path = bundle_path / "rag_chunks.jsonl"
    mapping_path = bundle_path / "chunk_to_kg.jsonl"
    matrix_path = bundle_path / "embeddings.f32"
    _write_jsonl(rag_path, chunk_rows)
    _write_jsonl(mapping_path, mapping_rows)
    matrix_path.write_bytes(matrix_bytes)

    manifest = {
        "schema_version": "curated-vector-bundle-manifest-1.0",
        "status": "candidate_vector_preview",
        "scope": "authored-selected-knowledge-preview",
        "production_activated": False,
        "human_approved_anchor_count": 0,
        "public_preview_is_production": False,
        "engineering_certification": False,
        "model": _sanitized_model_report(encoder, lock),
        "runtime": {
            "implementation": "direct-onnxruntime",
            "fastembed_used": False,
            "tokenizers": encoder.dependency_versions["tokenizers"],
            "onnxruntime": encoder.dependency_versions["onnxruntime"],
            "numpy": encoder.dependency_versions["numpy"],
            "provider": "CPUExecutionProvider",
            "threads": encoder.threads,
            "batch_size": encoder.batch_size,
            "tokenizer_truncation": False,
            "pooling": "CLS for rank-3 output; passthrough for rank-2 output",
        },
        "input": {
            "knowledge_units_file": "knowledge_units.jsonl",
            "knowledge_units_sha256": ku_file_sha,
            "knowledge_units_identity_sha256": identity,
            "knowledge_unit_count": len(records),
            "chunk_order": "chunk_id ascending",
            "input_template": (
                "title + LF + original_body[body_start:body_end] + LF + "
                "applicability_joined_by_LF"
            ),
            "max_tokens_including_special_tokens": 512,
            "truncation_allowed": False,
            "body_character_coverage": "complete; overlapping spans allowed",
        },
        "matrix": {
            "file": "embeddings.f32",
            "sha256": _sha256_bytes(matrix_bytes),
            "rows": len(chunk_rows),
            "dimensions": DIMENSIONS,
            "dtype": "<f4",
            "row_order": "chunk_id ascending",
            "finite": True,
            "norm_absolute_tolerance": NORM_TOLERANCE,
            "max_norm_absolute_error": max_norm_error,
        },
        "chunks": {
            "file": "rag_chunks.jsonl",
            "sha256": _relative_sha(rag_path),
            "rows": len(chunk_rows),
        },
        "chunk_to_kg": {
            "file": "chunk_to_kg.jsonl",
            "sha256": _relative_sha(mapping_path),
            "rows": len(mapping_rows),
        },
        "coverage": {
            "knowledge_unit_count": len(records),
            "chunk_count": len(chunk_rows),
            "all_knowledge_units_covered": True,
            "all_chunks_mapped": True,
        },
        "entries": entries,
        "disposition": (
            "This build does not claim independent approval; independent review "
            "is recorded separately from this artifact."
        ),
    }
    _ensure_no_private_paths(manifest, "vector_manifest")
    manifest_path = bundle_path / "vector_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor", type=Path)
    args = parser.parse_args()
    manifest = build_bundle(
        args.bundle,
        model_dir=args.model_dir,
        lock_path=args.lock,
        vendor_path=args.vendor,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "bundle_scope": manifest["scope"],
                "knowledge_units": manifest["coverage"]["knowledge_unit_count"],
                "chunks": manifest["coverage"]["chunk_count"],
                "production_activated": manifest["production_activated"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
