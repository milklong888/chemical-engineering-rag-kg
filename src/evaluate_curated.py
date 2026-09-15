"""Evaluate the frozen curated bundle on the public 16-query development set.

This driver supplies unchanged KU rankings to the repository's review-only
``retrieval_eval.evaluate_rows`` implementation.  It reuses one locked ONNX
encoder and one batched query encoding for dense and hybrid rankings; it does
not read holdout data, tune weights, or activate production retrieval.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Any

from curated_vectors import CuratedOnnxEncoder
from query_curated import CuratedBundle, RRF_K


LIMIT = 5
METHODS = ("lexical", "dense", "hybrid")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load retrieval evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_public_queries(path: Path) -> list[dict[str, Any]]:
    if "holdout" in str(path).lower():
        raise ValueError("holdout inputs are forbidden for this public evaluation")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"query row {line_number} is not an object")
        if value.get("split") != "public_development":
            raise ValueError(
                f"query row {line_number} is not in public_development split"
            )
        if not isinstance(value.get("query"), str) or not value["query"].strip():
            raise ValueError(f"query row {line_number} has no query text")
        rows.append(value)
    if len(rows) != 16:
        raise ValueError(f"expected exactly 16 public development queries, got {len(rows)}")
    return rows


def _indices(store: CuratedBundle, row: dict[str, Any]) -> list[int]:
    source_filter = row.get("source_filter")
    subject_filter = row.get("subject_filter")
    return store._filter_indices(  # noqa: SLF001 - fixed query module path
        node_id=None,
        source_filter=source_filter,
        subject_filter=subject_filter,
    )


def _unit_ranking(
    store: CuratedBundle,
    ranked: list[tuple[int, float]],
) -> list[str]:
    grouped = store._group_best(ranked)  # noqa: SLF001 - same frozen ranking path
    ordered = sorted(grouped.items(), key=lambda item: (-item[1][1], item[0]))
    return [unit_id for unit_id, _ in ordered[:LIMIT]]


def _dense_rank(
    store: CuratedBundle,
    indices: list[int],
    query_vector: Any,
    matrix: Any,
) -> list[tuple[int, float]]:
    scored = [(index, float(matrix[index] @ query_vector)) for index in indices]
    return sorted(
        scored,
        key=lambda item: (-item[1], store.chunks[item[0]]["chunk_id"]),
    )


def _hybrid_units(
    store: CuratedBundle,
    lexical_rank: list[tuple[int, float]],
    dense_rank: list[tuple[int, float]],
) -> list[str]:
    lexical_units = store._group_best(lexical_rank)  # noqa: SLF001
    dense_units = store._group_best(dense_rank)  # noqa: SLF001
    rrf_scores: dict[str, float] = {}
    for rank, (unit_id, _) in enumerate(
        sorted(lexical_units.items(), key=lambda item: (-item[1][1], item[0])),
        start=1,
    ):
        rrf_scores[unit_id] = rrf_scores.get(unit_id, 0.0) + 1.0 / (RRF_K + rank)
    for rank, (unit_id, _) in enumerate(
        sorted(dense_units.items(), key=lambda item: (-item[1][1], item[0])),
        start=1,
    ):
        rrf_scores[unit_id] = rrf_scores.get(unit_id, 0.0) + 1.0 / (RRF_K + rank)
    return [
        unit_id
        for unit_id, _ in sorted(
            rrf_scores.items(), key=lambda item: (-item[1], item[0])
        )[:LIMIT]
    ]


def _rank_public_queries(
    store: CuratedBundle,
    rows: list[dict[str, Any]],
    *,
    model_dir: Path,
    lock_path: Path,
    vendor_path: Path | None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, float]]:
    lexical_by_query: dict[str, list[str]] = {}
    lexical_ranks: dict[str, list[tuple[int, float]]] = {}
    lexical_started = time.perf_counter()
    for row in rows:
        query_id = row["query_id"]
        indices = _indices(store, row)
        lexical_rank = store._rank_lexical(row["query"], indices)  # noqa: SLF001
        lexical_ranks[query_id] = lexical_rank
        lexical_by_query[query_id] = _unit_ranking(store, lexical_rank)
    lexical_seconds = time.perf_counter() - lexical_started

    encoder_started = time.perf_counter()
    encoder = CuratedOnnxEncoder(
        model_dir,
        lock_path=lock_path,
        vendor_path=vendor_path,
    )
    store._validate_dense_model(encoder, lock_path)  # noqa: SLF001
    matrix = store.load_matrix()
    encoder_init_seconds = time.perf_counter() - encoder_started

    encode_started = time.perf_counter()
    query_vectors = encoder.encode_texts([row["query"] for row in rows])
    encode_seconds = time.perf_counter() - encode_started

    dense_by_query: dict[str, list[str]] = {}
    dense_ranks: dict[str, list[tuple[int, float]]] = {}
    dense_started = time.perf_counter()
    for row, query_vector in zip(rows, query_vectors, strict=True):
        query_id = row["query_id"]
        dense_rank = _dense_rank(store, _indices(store, row), query_vector, matrix)
        dense_ranks[query_id] = dense_rank
        dense_by_query[query_id] = _unit_ranking(store, dense_rank)
    dense_seconds = time.perf_counter() - dense_started

    hybrid_started = time.perf_counter()
    hybrid_by_query = {
        row["query_id"]: _hybrid_units(
            store,
            lexical_ranks[row["query_id"]],
            dense_ranks[row["query_id"]],
        )
        for row in rows
    }
    hybrid_seconds = time.perf_counter() - hybrid_started

    return (
        {
            "lexical": lexical_by_query,
            "dense": dense_by_query,
            "hybrid": hybrid_by_query,
        },
        {
            "lexical_ranking": lexical_seconds,
            "encoder_init_and_contract_check": encoder_init_seconds,
            "dense_query_encode_batch": encode_seconds,
            "dense_ranking": dense_seconds,
            "hybrid_rrf_fusion": hybrid_seconds,
        },
    )


def _method_report(
    retrieval_eval: Any,
    rows: list[dict[str, Any]],
    ranked: dict[str, list[str]],
    method: str,
) -> dict[str, Any]:
    evaluation = retrieval_eval.evaluate_rows(
        rows,
        ranked,
        route_name=f"curated_bundle_{method}",
        k=LIMIT,
        mrr_k=LIMIT,
        ndcg_k=LIMIT,
    )
    query_by_id = {row["query_id"]: row for row in rows}
    diagnostics = []
    for diagnostic in evaluation["diagnostics"]:
        query_id = diagnostic["query_id"]
        diagnostics.append(
            {
                "query_id": query_id,
                "query": query_by_id[query_id]["query"],
                "primary_target_ids": query_by_id[query_id]["primary_target_ids"],
                "ranked_ids": ranked[query_id],
                "first_primary_rank": diagnostic["first_primary_rank"],
                "hit_at_5": diagnostic["hit_at_k"],
                "mrr_at_5": diagnostic["mrr_at_k"],
                "rank1": diagnostic["rank1"],
            }
        )
    aggregate = evaluation["aggregate"]
    return {
        "aggregate": {
            "query_count": len(rows),
            "hit_at_5": aggregate["hit_at_k"],
            "mrr_at_5": aggregate["mrr_at_k"],
            "rank1_rate": aggregate["rank1_rate"],
        },
        "per_query": diagnostics,
        "review_only_evaluation": evaluation,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_public_queries(args.queries)
    retrieval_eval = _load_module(args.retrieval_eval, "curated_retrieval_eval")
    store = CuratedBundle(args.bundle)
    target_ids = {
        target
        for row in rows
        for target in row.get("primary_target_ids", [])
    }
    if not target_ids.issubset(store.knowledge_units):
        raise ValueError("public target ID is absent from normative bundle KU set")

    rankings, timings = _rank_public_queries(
        store,
        rows,
        model_dir=args.model_dir,
        lock_path=args.lock,
        vendor_path=args.vendor,
    )
    methods = {
        method: _method_report(retrieval_eval, rows, rankings[method], method)
        for method in METHODS
    }
    bundle_files = {
        name: _sha256(args.bundle / name)
        for name in (
            "knowledge_units.jsonl",
            "rag_chunks.jsonl",
            "chunk_to_kg.jsonl",
            "embeddings.f32",
            "vector_manifest.json",
        )
    }
    code_files = {
        "evaluate_curated.py": _sha256(Path(__file__).resolve()),
        "index_curated_bundle.py": _sha256(Path(__file__).with_name("index_curated_bundle.py")),
        "query_curated.py": _sha256(Path(__file__).with_name("query_curated.py")),
        "retrieval_eval.py": _sha256(args.retrieval_eval),
    }
    return {
        "schema": "curated-public-development-evaluation-v1",
        "status": "review_only_machine_diagnostic",
        "scope": "public_development_examples_only",
        "query_count": len(rows),
        "methods": methods,
        "timing_seconds": timings,
        "timing_note": (
            "One locked encoder and one batched query encoding were reused for "
            "dense and hybrid; hybrid timing is the fixed RRF fusion step."
        ),
        "ranking_contract": {
            "limit": LIMIT,
            "lexical": "Chinese characters plus adjacent bigrams and lowercased English/alphanumeric tokens scored by BM25(k1=1.2,b=0.75)",
            "dense": "locked direct ONNX BGE embeddings with exact matrix cosine",
            "hybrid": "fixed reciprocal rank fusion with RRF_K=60",
            "target_grain": "knowledge_unit",
            "no_parameter_tuning": True,
        },
        "input_hashes": {
            "queries.jsonl": _sha256(args.queries),
            **bundle_files,
            "model_lock.json": _sha256(args.lock),
        },
        "code_hashes": code_files,
        "bundle_manifest": {
            "knowledge_unit_count": store.manifest["coverage"]["knowledge_unit_count"],
            "chunk_count": store.manifest["coverage"]["chunk_count"],
            "dimensions": store.manifest["matrix"]["dimensions"],
            "production_activated": store.manifest["production_activated"],
            "human_approved_anchor_count": store.manifest[
                "human_approved_anchor_count"
            ],
        },
        "holdout_read": False,
        "production_activated": False,
        "engineering_certification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--retrieval-eval", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": report["status"], "query_count": report["query_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
