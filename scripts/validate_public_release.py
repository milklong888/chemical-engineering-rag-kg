#!/usr/bin/env python3
"""Fail-closed checks for this sanitized public repository."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 10 * 1024 * 1024

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
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


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
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append(f"file exceeds 10 MiB: {relative}")

        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append(f"invalid JSON {relative}: {exc}")

        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                failures.append(f"non-UTF-8 text file: {relative}")
                continue
            for name, pattern in PRIVATE_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{name} matched in {relative}")

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
