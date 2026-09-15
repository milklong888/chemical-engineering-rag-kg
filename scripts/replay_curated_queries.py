"""Replay the frozen 16-query report through CuratedBundle.search.

This is an evidence script, not a ranking implementation.  It calls the
formal repository query API once for every query/method pair and compares the
returned top-five KU IDs with the formal report.  Target IDs are used only for
the post-hoc comparison and are never passed to search.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


METHODS = ("lexical", "dense", "hybrid")
LIMIT = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != 16 or any(row.get("split") != "public_development" for row in rows):
        raise ValueError("query input is not exactly the frozen 16-row public development set")
    return rows


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("query_count") != 16:
        raise ValueError("formal report does not contain 16 queries")
    return report


def audit_evaluator_code(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function_names = {"_indices", "_unit_ranking", "_dense_rank", "_hybrid_units", "_rank_public_queries"}
    forbidden_target_names = {"target_ids", "primary_target_ids"}
    ranking_hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in function_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id in forbidden_target_names:
                    ranking_hits.append(f"{node.name}:{child.id}")
                if isinstance(child, ast.Constant) and child.value == "primary_target_ids":
                    ranking_hits.append(f"{node.name}:literal")
    run_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    run_target_checks = [
        node.lineno
        for node in ast.walk(run_function)
        if isinstance(node, ast.Name) and node.id == "target_ids"
    ]
    return {
        "ranking_functions_contain_target_id_references": ranking_hits,
        "run_target_set_validation_lines": run_target_checks,
        "ranking_target_dependency_absent": not ranking_hits,
    }


def replay(
    bundle_path: Path,
    queries_path: Path,
    report_path: Path,
    query_module_path: Path,
    eval_module_path: Path,
    model_dir: Path,
    lock_path: Path,
    vendor_path: Path | None,
) -> dict[str, Any]:
    query_rows = load_queries(queries_path)
    formal_report = load_report(report_path)
    code_audit = audit_evaluator_code(eval_module_path)

    query_parent = query_module_path.parent
    if str(query_parent) not in sys.path:
        sys.path.insert(0, str(query_parent))
    if vendor_path is not None and str(vendor_path) not in sys.path:
        # Put the locked wheelhouse ahead of the workspace Python packages
        # before query_curated imports numpy/onnxruntime/tokenizers.
        sys.path.insert(0, str(vendor_path))
    from query_curated import CuratedBundle  # type: ignore[import-not-found]

    store = CuratedBundle(bundle_path)
    comparisons: list[dict[str, Any]] = []
    calls = 0
    mismatches: list[dict[str, Any]] = []
    for method in METHODS:
        report_rows = {row["query_id"]: row for row in formal_report["methods"][method]["per_query"]}
        for row in query_rows:
            kwargs: dict[str, Any] = {
                "query": row["query"],
                "method": method,
                "limit": LIMIT,
            }
            if row.get("source_filter") is not None:
                kwargs["source_filter"] = row["source_filter"]
            if row.get("subject_filter") is not None:
                kwargs["subject_filter"] = row["subject_filter"]
            if method in {"dense", "hybrid"}:
                kwargs.update(
                    {
                        "model_dir": model_dir,
                        "lock_path": lock_path,
                        "vendor_path": vendor_path,
                    }
                )
            returned = store.search(**kwargs)
            calls += 1
            actual_ids = [item["knowledge_unit_id"] for item in returned]
            expected_ids = report_rows[row["query_id"]]["ranked_ids"]
            match = actual_ids == expected_ids
            item = {
                "query_id": row["query_id"],
                "method": method,
                "actual_top5": actual_ids,
                "reported_top5": expected_ids,
                "match": match,
                "search_kwargs_excluded_target_ids": "primary_target_ids",
            }
            comparisons.append(item)
            if not match:
                mismatches.append(item)

    return {
        "schema": "curated-search-independent-replay-v1",
        "status": "PASS" if calls == 48 and not mismatches and code_audit["ranking_target_dependency_absent"] else "FAIL",
        "calls": calls,
        "methods": list(METHODS),
        "queries": 16,
        "mismatches": mismatches,
        "all_48_top5_match": calls == 48 and not mismatches,
        "evaluator_code_audit": code_audit,
        "comparisons": comparisons,
        "binding": {
            "evaluate_curated.py": sha256(eval_module_path),
            "query_curated.py": sha256(query_module_path),
            "formal_report": sha256(report_path),
            "queries.jsonl": sha256(queries_path),
            "knowledge_units.jsonl": sha256(bundle_path / "knowledge_units.jsonl"),
            "rag_chunks.jsonl": sha256(bundle_path / "rag_chunks.jsonl"),
            "chunk_to_kg.jsonl": sha256(bundle_path / "chunk_to_kg.jsonl"),
            "embeddings.f32": sha256(bundle_path / "embeddings.f32"),
            "vector_manifest.json": sha256(bundle_path / "vector_manifest.json"),
            "model_lock.json": sha256(lock_path),
        },
        "model_boundary": {
            "search_api": "formal query_curated.CuratedBundle.search",
            "ranking_reimplemented": False,
            "target_ids_used_for_ranking": False,
            "model_execution_claim": "not claimed by this audit; this compares the formal API with the frozen report",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--query-module", type=Path, required=True)
    parser.add_argument("--evaluate-module", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(
        args.bundle,
        args.queries,
        args.report,
        args.query_module,
        args.evaluate_module,
        args.model_dir,
        args.lock,
        args.vendor,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": result["status"], "calls": result["calls"], "mismatches": len(result["mismatches"])}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
