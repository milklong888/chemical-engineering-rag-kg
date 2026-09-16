"""Review-only evaluator for the frozen S001 public development queries.

When explicitly run after the vector stage is accepted, this driver calls the
repository's ``CuratedBundle.search`` API for lexical, dense and hybrid
methods.  It verifies that each API result carries the complete KU object and
the matching full-body SHA-256, while the report retains only its identity,
hash, rank and score fields.  It does not tune queries, scores, filters or
fusion and does not treat an out-of-bundle diagnostic as an
automatic-rejection test.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


LIMIT = 5
METHODS = ("lexical", "dense", "hybrid")
PUBLIC_SPLIT = "public_development"
DIAGNOSTIC_SPLIT = "public_out_of_bundle_diagnostic"


class RetrievalEvaluationError(RuntimeError):
    """Raised when a query contract or returned KU fails closed validation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RetrievalEvaluationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RetrievalEvaluationError(f"cannot hash {path.name}: {exc}") from exc
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RetrievalEvaluationError(f"cannot read {path.name}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalEvaluationError(f"invalid JSON at {path.name}:{line_number}") from exc
        _require(isinstance(value, dict), f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def _string_list(row: dict[str, Any], field: str) -> list[str]:
    value = row.get(field, [])
    _require(isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value), f"{field} must be a list of non-empty strings")
    _require(len(value) == len(set(value)), f"{field} contains duplicate IDs")
    return list(value)


def _load_positive_queries(path: Path) -> list[dict[str, Any]]:
    _require("holdout" not in str(path).lower(), "holdout input is forbidden")
    rows = _read_jsonl(path)
    _require(len(rows) == 16, f"expected 16 public development queries, got {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        query_id = row.get("query_id")
        _require(isinstance(query_id, str) and query_id, f"positive row {index} has no query_id")
        _require(query_id not in seen, f"duplicate positive query_id {query_id}")
        seen.add(query_id)
        _require(row.get("split") == PUBLIC_SPLIT, f"{query_id} is not in {PUBLIC_SPLIT}")
        _require(isinstance(row.get("query"), str) and row["query"].strip(), f"{query_id} has no query")
        expected = _string_list(row, "expected_knowledge_unit_ids")
        primary = _string_list(row, "primary_target_ids")
        _require(expected == primary, f"{query_id}: expected IDs and primary target IDs differ")
        _require(expected, f"{query_id}: positive query has no expected target")
        _require(isinstance(row.get("chapter_id"), str) and row["chapter_id"].startswith("S001-CURATED-CH"), f"{query_id}: chapter_id is outside S001")
        _require(isinstance(row.get("reason"), str) and row["reason"].strip(), f"{query_id}: reason is missing")
        _require(row.get("case_role") == "positive_any_hit", f"{query_id}: unsupported positive role")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["chapter_id"]] = counts.get(row["chapter_id"], 0) + 1
    _require(set(counts) == {f"S001-CURATED-CH{number:02d}" for number in (2, 3, 4, 5)}, "positive queries do not cover exactly four S001 chapters")
    _require(all(count == 4 for count in counts.values()), "positive queries must contain four rows per chapter")
    return rows


def _load_diagnostics(path: Path) -> list[dict[str, Any]]:
    _require("holdout" not in str(path).lower(), "holdout input is forbidden")
    rows = _read_jsonl(path)
    _require(len(rows) == 4, f"expected 4 out-of-bundle diagnostics, got {len(rows)}")
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        query_id = row.get("query_id")
        _require(isinstance(query_id, str) and query_id, f"diagnostic row {index} has no query_id")
        _require(query_id not in seen, f"duplicate diagnostic query_id {query_id}")
        seen.add(query_id)
        _require(row.get("split") == DIAGNOSTIC_SPLIT, f"{query_id} is not in {DIAGNOSTIC_SPLIT}")
        _require(row.get("case_role") == "diagnostic_only", f"{query_id}: unsupported diagnostic role")
        _require(isinstance(row.get("query"), str) and row["query"].strip(), f"{query_id} has no query")
        _require(_string_list(row, "expected_knowledge_unit_ids") == [], f"{query_id} must have no expected IDs")
        _require(_string_list(row, "primary_target_ids") == [], f"{query_id} must have no primary targets")
        _require(row.get("auto_reject_expected") is False, f"{query_id} cannot claim automatic rejection")
        _require(isinstance(row.get("reason"), str) and row["reason"].strip(), f"{query_id}: reason is missing")
    return rows


def _load_query_module(path: Path) -> Any:
    path = path.resolve()
    _require(path.is_file(), f"query module does not exist: {path.name}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("s001_query_curated", path)
    _require(spec is not None and spec.loader is not None, "cannot load query module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(hasattr(module, "CuratedBundle"), "query module has no CuratedBundle API")
    return module


def _verify_result(store: Any, result: Any, rank: int) -> dict[str, Any]:
    _require(isinstance(result, dict), f"rank {rank} is not a result object")
    unit_id = result.get("knowledge_unit_id")
    _require(isinstance(unit_id, str) and unit_id, f"rank {rank} has no knowledge_unit_id")
    unit = result.get("knowledge_unit")
    _require(isinstance(unit, dict), f"rank {rank} does not return a complete knowledge_unit object")
    _require(unit.get("knowledge_unit_id") == unit_id and unit.get("node_id") == unit_id, f"rank {rank} KU identity mismatch")
    text = unit.get("text")
    expected_hash = unit.get("text_sha256")
    _require(isinstance(text, str) and text, f"rank {rank} KU body is empty")
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _require(actual_hash == expected_hash, f"rank {rank} body SHA mismatch for {unit_id}")
    normative = getattr(store, "knowledge_units", {}).get(unit_id)
    _require(normative is not None and unit == normative, f"rank {rank} KU is not the complete normative object for {unit_id}")
    return {
        "rank": rank,
        "knowledge_unit_id": unit_id,
        "body_sha256": actual_hash,
        "body_hash_match": True,
        "complete_knowledge_unit_verified": True,
        "best_chunk_id": result.get("best_chunk_id"),
        "body_start": result.get("body_start"),
        "body_end": result.get("body_end"),
        "score": result.get("score"),
        "channel_score": result.get("channel_score"),
        "method": result.get("method"),
        "candidate_status": result.get("candidate_status"),
        "engineering_certification": result.get("engineering_certification"),
    }


def _search_one(store: Any, row: dict[str, Any], method: str, *, model_dir: Path, lock_path: Path, vendor_path: Path | None) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "query": row["query"],
        "method": method,
        "limit": LIMIT,
        "source_filter": row.get("source_filter"),
        "subject_filter": row.get("subject_filter"),
    }
    if method in {"dense", "hybrid"}:
        kwargs.update({"model_dir": model_dir, "lock_path": lock_path, "vendor_path": vendor_path})
    raw_results = store.search(**kwargs)
    _require(isinstance(raw_results, list) and len(raw_results) <= LIMIT, f"{row['query_id']} {method} returned an invalid result count")
    verified = [_verify_result(store, result, rank) for rank, result in enumerate(raw_results, 1)]
    ids = [item["knowledge_unit_id"] for item in verified]
    _require(len(ids) == len(set(ids)), f"{row['query_id']} {method} returned duplicate KU IDs")
    return verified


def _positive_case(store: Any, row: dict[str, Any], method: str, *, model_dir: Path, lock_path: Path, vendor_path: Path | None) -> dict[str, Any]:
    results = _search_one(store, row, method, model_dir=model_dir, lock_path=lock_path, vendor_path=vendor_path)
    ranked_ids = [item["knowledge_unit_id"] for item in results]
    expected = row["expected_knowledge_unit_ids"]
    ranks = {target: (ranked_ids.index(target) + 1 if target in ranked_ids else None) for target in expected}
    found = [rank for rank in ranks.values() if rank is not None and rank <= LIMIT]
    first_rank = min(found) if found else None
    return {
        "query_id": row["query_id"],
        "chapter_id": row["chapter_id"],
        "query": row["query"],
        "reason": row["reason"],
        "expected_knowledge_unit_ids": expected,
        "method": method,
        "ranked_ids": ranked_ids,
        "actual_ranks": ranks,
        "first_expected_rank": first_rank,
        "hit_at_5": first_rank is not None,
        "mrr_at_5": 0.0 if first_rank is None else 1.0 / first_rank,
        "top1": bool(ranked_ids and ranked_ids[0] in expected),
        "results": results,
    }


def _diagnostic_case(store: Any, row: dict[str, Any], method: str, *, model_dir: Path, lock_path: Path, vendor_path: Path | None) -> dict[str, Any]:
    results = _search_one(store, row, method, model_dir=model_dir, lock_path=lock_path, vendor_path=vendor_path)
    return {
        "query_id": row["query_id"],
        "query": row["query"],
        "reason": row["reason"],
        "expected_knowledge_unit_ids": [],
        "method": method,
        "returned_ids": [item["knowledge_unit_id"] for item in results],
        "results": results,
        "diagnostic_only": True,
        "auto_reject_expected": False,
        "automatic_rejection_assessed": False,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "query_count": count,
        "hit_at_5": sum(float(row["hit_at_5"]) for row in rows) / count if count else None,
        "mrr_at_5": sum(row["mrr_at_5"] for row in rows) / count if count else None,
        "top1": sum(float(row["top1"]) for row in rows) / count if count else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    positives = _load_positive_queries(args.public_queries)
    diagnostics = _load_diagnostics(args.diagnostics)
    query_module = _load_query_module(args.query_module)
    store = query_module.CuratedBundle(args.bundle)
    normative = getattr(store, "knowledge_units", {})
    target_ids = {target for row in positives for target in row["expected_knowledge_unit_ids"]}
    _require(target_ids.issubset(normative), "a positive target is absent from the normative bundle")

    positive_results: dict[str, list[dict[str, Any]]] = {}
    diagnostic_results: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        positive_results[method] = [
            _positive_case(store, row, method, model_dir=args.model_dir, lock_path=args.lock_path, vendor_path=args.vendor)
            for row in positives
        ]
        diagnostic_results[method] = [
            _diagnostic_case(store, row, method, model_dir=args.model_dir, lock_path=args.lock_path, vendor_path=args.vendor)
            for row in diagnostics
        ]

    knowledge_name = store.manifest.get("input", {}).get("knowledge_units_file", "knowledge_units.jsonl")
    knowledge_path = args.bundle / knowledge_name
    _require(knowledge_path.is_file(), "bundle knowledge_units.jsonl is missing")
    vector_file_names = (
        "vector_manifest.json",
        "embeddings.f32",
        "rag_chunks.jsonl",
        "chunk_to_kg.jsonl",
    )
    vector_paths: dict[str, Path] = {}
    for name in vector_file_names:
        path = args.bundle / name
        _require(path.is_file(), f"bundle {name} is missing")
        vector_paths[name] = path
    return {
        "schema": "s001-public-development-retrieval-evaluation-v1",
        "status": "review_only_machine_diagnostic",
        "scope": "public_development_examples_plus_out_of_bundle_diagnostics",
        "query_count": len(positives),
        "diagnostic_count": len(diagnostics),
        "methods": {
            method: {
                "aggregate": _aggregate(positive_results[method]),
                "per_query": positive_results[method],
            }
            for method in METHODS
        },
        "out_of_bundle_diagnostics": diagnostic_results,
        "ranking_contract": {
            "limit": LIMIT,
            "methods": list(METHODS),
            "ranking_source": "CuratedBundle.search",
            "target_grain": "knowledge_unit",
            "complete_knowledge_unit_returned": True,
            "full_body_sha256_verified": True,
            "query_or_score_tuning": False,
            "fusion_parameter_tuning": False,
            "failed_cases_retained": True,
        },
        "input_hashes": {
            "public_queries": _sha256_file(args.public_queries),
            "out_of_bundle_diagnostics": _sha256_file(args.diagnostics),
            "knowledge_units.jsonl": _sha256_file(knowledge_path),
            **{name: _sha256_file(path) for name, path in vector_paths.items()},
            "query_module": _sha256_file(args.query_module),
            "model_lock": _sha256_file(args.lock_path),
        },
        "holdout_read": False,
        "production_activated": False,
        "engineering_certification": False,
        "automatic_rejection_claimed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--public-queries", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--query-module", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", dest="lock_path", type=Path, required=True)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "query_count": report["query_count"], "diagnostic_count": report["diagnostic_count"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
