#!/usr/bin/env python3
"""Fail-closed checks for this sanitized public repository."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024
CURATED_BUNDLE = "knowledge/curated-four-books-v1"
ALLOWED_VECTOR_FILE = CURATED_BUNDLE + "/embeddings.f32"
S001_BUNDLE = "knowledge/curated-s001-upper-v1"
S001_ALLOWED_VECTOR_FILE = S001_BUNDLE + "/embeddings.f32"
S001_REPORTS = {
    "reports/s001-independent-validation.json",
    "reports/s001-vector-audit.json",
    "reports/s001-dev-evaluation.json",
}

DENIED_SUFFIXES = {
    ".pdf",
    ".zip",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".webp",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".onnx",
    ".safetensors",
    ".npy",
    ".npz",
    ".pem",
    ".key",
}

TEXT_SUFFIXES = {".md", ".json", ".jsonl", ".py", ".yml", ".yaml", ".txt"}

PRIVATE_PATTERNS = {
    "windows_absolute_path": re.compile(r"(?i)(?<![A-Z0-9])[A-Z]:[\\/]"),
    "windows_user_path": re.compile(r"(?i)[A-Z]:\\Users\\[^\\\s]+"),
    "unix_home_path": re.compile(r"/(?:home|Users)/[^/\s]+"),
    "credential_assignment": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}"
    ),
    "github_token": re.compile(r"(?:ghp|gho|github_pat)_[A-Za-z0-9_]{20,}"),
}

REQUIRED_FILES = {
    "README.md",
    "CONTRIBUTING.md",
    "LICENSE-NOTICE.md",
    "THIRD_PARTY_NOTICES.md",
    "docs/INGESTION_SPEC.md",
    "docs/SOURCE_STATUS.md",
    "examples/submission-manifest.example.json",
    "reports/stage_snapshot.json",
    "src/retrieval_eval.py",
    "docs/READING_GUIDE.md",
    "contracts/curated_knowledge_contract.json",
    "contracts/curated_model_lock.json",
    CURATED_BUNDLE + "/manifest.json",
    CURATED_BUNDLE + "/qa_summary.json",
}

CURATED_PAYLOAD_FILES = {
    "source_manifest.json", "knowledge_units.jsonl", "evidence_registry.jsonl",
    "kg_nodes.jsonl", "kg_edges.jsonl", "external_references.jsonl",
    "conversion_report.json", "rag_chunks.jsonl", "chunk_to_kg.jsonl",
    "embeddings.f32", "vector_manifest.json", "qa_summary.json",
}
S001_EXTRA_PAYLOAD_FILES = {"cross_bundle_references.jsonl"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_curated_vectors(root: Path, bundle_rel: str = CURATED_BUNDLE) -> list[str]:
    """The sole binary exception is a hash-bound authored-knowledge preview."""
    failures: list[str] = []
    matrix_path = root / bundle_rel / "embeddings.f32"
    manifest_path = root / bundle_rel / "vector_manifest.json"
    if not matrix_path.exists() and not manifest_path.exists():
        return failures
    if not matrix_path.exists() or not manifest_path.exists():
        return ["curated vector matrix and manifest must appear together"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        matrix = manifest["matrix"]
        entries = manifest["entries"]
        payload = matrix_path.read_bytes()
        rows = matrix["rows"]
        if type(rows) is not int or rows <= 0:
            raise ValueError("invalid vector row count")
        if (matrix["file"] != "embeddings.f32" or matrix["dimensions"] != 512
                or matrix["dtype"] != "<f4" or matrix["row_order"] != "chunk_id ascending"):
            raise ValueError("unexpected vector format")
        if len(payload) != rows * 512 * 4 or len(entries) != rows:
            raise ValueError("matrix size or entry count mismatch")
        if hashlib.sha256(payload).hexdigest() != matrix["sha256"]:
            raise ValueError("matrix SHA mismatch")
        chunk_path = root / bundle_rel / "rag_chunks.jsonl"
        chunks = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        eligible = {c["chunk_id"]: c for c in chunks if c.get("embedding_eligible") is True}
        row_ids = [entry["chunk_id"] for entry in entries]
        if row_ids != sorted(eligible) or len(set(row_ids)) != rows:
            raise ValueError("matrix rows do not match all eligible chunks in order")
        for i, entry in enumerate(entries):
            chunk = eligible[entry["chunk_id"]]
            if chunk.get("retrieval_eligible") is not True:
                raise ValueError("embedding row is not retrieval eligible")
            if hashlib.sha256(chunk["search_text"].encode("utf-8")).hexdigest() != entry["input_sha256"]:
                raise ValueError("vector input hash mismatch")
            row_bytes = payload[i * 2048:(i + 1) * 2048]
            if hashlib.sha256(row_bytes).hexdigest() != entry["vector_sha256"]:
                raise ValueError("vector row hash mismatch")
            values = struct.unpack("<512f", row_bytes)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite vector value")
            if abs(math.sqrt(sum(value * value for value in values)) - 1.0) > 1e-5:
                raise ValueError("vector is not L2-normalized")
    except (KeyError, ValueError, TypeError, OSError) as exc:
        failures.append(f"curated vector validation: {exc}")
    return failures


def check_curated_release(
    root: Path,
    bundle_rel: str = CURATED_BUNDLE,
    bundle_id: str = "curated-four-books-v1",
    extra_payload: set[str] | None = None,
    related_bundle: Path | None = None,
) -> list[str]:
    """Bind the public snapshot to exact payload/report bytes and graph checks."""
    failures: list[str] = []
    bundle = root / bundle_rel
    expected_payload = CURATED_PAYLOAD_FILES | (extra_payload or set())
    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("status") != "validated_public_preview"
                or manifest.get("production_activated") is not False):
            raise ValueError("invalid curated release status")
        if manifest.get("bundle_id") != bundle_id:
            raise ValueError("release manifest has the wrong bundle_id")
        if set(manifest["files"]) != expected_payload:
            raise ValueError("release manifest does not enumerate the exact payload")
        for name, info in manifest["files"].items():
            payload = (bundle / name).read_bytes()
            if len(payload) != info["bytes"] or hashlib.sha256(payload).hexdigest() != info["sha256"]:
                raise ValueError(f"curated payload identity mismatch: {name}")
        for name, info in manifest["reports"].items():
            path = (root / name).resolve()
            if not path.is_relative_to((root / "reports").resolve()):
                raise ValueError("report path escapes reports directory")
            payload = path.read_bytes()
            if len(payload) != info["bytes"] or hashlib.sha256(payload).hexdigest() != info["sha256"]:
                raise ValueError(f"curated report identity mismatch: {name}")
        if bundle_id == "curated-s001-upper-v1" and set(manifest["reports"]) != S001_REPORTS:
            raise ValueError("S001 release manifest does not bind the three registered reports")
        from validate_curated_bundle import validate_bundle
        report = validate_bundle(
            bundle,
            require_vectors=True,
            bundle_id=bundle_id,
            related_bundle=related_bundle,
        )
        failures.extend("curated structure: " + error["code"] + " at " + error["location"]
                        for error in report["errors"])
    except (KeyError, ValueError, TypeError, OSError) as exc:
        failures.append(f"curated release validation: {exc}")
    return failures


def main() -> int:
    failures: list[str] = []
    files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts]
    paths = {rel(p) for p in files}

    missing = sorted(REQUIRED_FILES - paths)
    if missing:
        failures.append(f"missing required files: {missing}")

    for path in files:
        relative = rel(path)
        if path.suffix.lower() in DENIED_SUFFIXES:
            failures.append(f"denied payload extension: {relative}")
        if path.suffix.lower() == ".f32" and relative not in {ALLOWED_VECTOR_FILE, S001_ALLOWED_VECTOR_FILE}:
            failures.append(f"vector outside the approved curated previews: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB: {relative}")

        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append(f"invalid JSON {relative}: {exc}")
        if path.suffix.lower() == ".jsonl":
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(f"line {line_number} is not an object")
            except Exception as exc:
                failures.append(f"invalid JSONL {relative}: {exc}")

        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"non-UTF-8 text file: {relative}")
                continue
            for name, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{name} matched in {relative}")

    failures.extend(check_curated_vectors(ROOT, CURATED_BUNDLE))
    failures.extend(check_curated_release(ROOT, CURATED_BUNDLE))
    s001_path = ROOT / S001_BUNDLE
    if s001_path.exists():
        failures.extend(check_curated_vectors(ROOT, S001_BUNDLE))
        failures.extend(
            check_curated_release(
                ROOT,
                S001_BUNDLE,
                bundle_id="curated-s001-upper-v1",
                extra_payload=S001_EXTRA_PAYLOAD_FILES,
                related_bundle=ROOT / CURATED_BUNDLE,
            )
        )

    snapshot_path = ROOT / "reports" / "stage_snapshot.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        identity = snapshot["source_identity"]
        preview = snapshot["current_global_preview"]
        if identity["known_pdf_sources"] != 8:
            failures.append("known_pdf_sources must equal 8 for this snapshot")
        if identity["known_pdf_pages"] != 3977:
            failures.append("known_pdf_pages must equal 3977 for this snapshot")
        if preview["rag_chunks"] != preview["chunk_to_kg_maps"]:
            failures.append("every preview chunk must have one chunk-to-KG map")
        if preview["embedding_eligible"] > preview["retrieval_eligible"]:
            failures.append("embedding_eligible cannot exceed retrieval_eligible")
        if snapshot["production_gates"]["production_activated"] is not False:
            failures.append("public stage snapshot must remain non-production")

    print(
        json.dumps(
            {
                "status": "PASS" if not failures else "FAIL",
                "file_count": len(files),
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
