"""Review-only evaluator for caller-supplied ranked identifiers.

The module implements four mutually exclusive evaluation roles. It never opens a
retrieval index, creates embeddings, changes fusion weights, or treats machine
metrics as human approval.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


POSITIVE_ANY_HIT = "positive_any_hit"
POSITIVE_ALL_RELEVANT = "positive_all_relevant_ordered_parts"
NEGATIVE_SOURCE_ROUTING = "negative_source_routing"
RELATION_ONLY = "relation_only_not_primary_scored"
CASE_ROLES = {
    POSITIVE_ANY_HIT,
    POSITIVE_ALL_RELEVANT,
    NEGATIVE_SOURCE_ROUTING,
    RELATION_ONLY,
}


class InputContractError(ValueError):
    """Raised when a supplied query or ranked-result row violates the contract."""


def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputContractError("{}:{}: invalid JSON: {}".format(path, line_number, exc))
        if not isinstance(value, dict):
            raise InputContractError("{}:{}: each JSONL row must be an object".format(path, line_number))
        rows.append(value)
    return rows


def _string_list(row, field):
    value = row.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise InputContractError("{} must be a list of non-empty strings".format(field))
    if len(value) != len(set(value)):
        raise InputContractError("{} must not contain duplicate identifiers".format(field))
    return list(value)


def normalize_query_row(row):
    query_id = row.get("query_id")
    case_role = row.get("case_role")
    if not isinstance(query_id, str) or not query_id:
        raise InputContractError("query_id must be a non-empty string")
    if case_role not in CASE_ROLES:
        raise InputContractError("{}: unsupported case_role {!r}".format(query_id, case_role))

    primary = _string_list(row, "primary_target_ids")
    context = _string_list(row, "relation_context_chunk_ids")
    must_not = _string_list(row, "must_not_chunk_ids")
    overlap = (set(primary) & set(context)) | (set(primary) & set(must_not)) | (set(context) & set(must_not))
    if overlap:
        raise InputContractError("{}: primary/context/must-not identifiers must be disjoint".format(query_id))

    if case_role in {POSITIVE_ANY_HIT, POSITIVE_ALL_RELEVANT} and not primary:
        raise InputContractError("{}: positive roles require primary_target_ids".format(query_id))
    if case_role in {NEGATIVE_SOURCE_ROUTING, RELATION_ONLY} and primary:
        raise InputContractError("{}: non-primary roles must not define primary_target_ids".format(query_id))
    if case_role == NEGATIVE_SOURCE_ROUTING and not must_not:
        raise InputContractError("{}: negative_source_routing requires must_not_chunk_ids".format(query_id))
    if case_role != NEGATIVE_SOURCE_ROUTING and must_not:
        raise InputContractError("{}: must_not_chunk_ids are exclusive to negative_source_routing".format(query_id))

    return {
        "query_id": query_id,
        "case_role": case_role,
        "expected_behavior": row.get("expected_behavior"),
        "primary_target_ids": primary,
        "relation_context_chunk_ids": context,
        "must_not_chunk_ids": must_not,
    }


def normalize_ranked_rows(rows):
    result = {}
    for row in rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise InputContractError("ranked row query_id must be a non-empty string")
        if query_id in result:
            raise InputContractError("duplicate ranked row for query_id {}".format(query_id))
        result[query_id] = _string_list(row, "ranked_ids")
    return result


def _ranks(ranked_ids, targets, cutoff):
    target_set = set(targets)
    return [rank for rank, item_id in enumerate(ranked_ids[:cutoff], 1) if item_id in target_set]


def _ndcg_at_k(ranked_ids, relevant_ids, cutoff):
    relevant = set(relevant_ids)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item_id in enumerate(ranked_ids[:cutoff], 1) if item_id in relevant)
    ideal_hits = min(len(relevant), cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0.0 else dcg / idcg


def evaluate_case(query, ranked_ids, k, mrr_k, ndcg_k):
    """Evaluate one normalized query without rebuilding or changing its ranking."""
    if k <= 0 or mrr_k <= 0 or ndcg_k <= 0:
        raise InputContractError("metric cutoffs must be positive integers")
    row = normalize_query_row(query)
    if len(ranked_ids) != len(set(ranked_ids)):
        raise InputContractError("{}: ranked_ids must not contain duplicates".format(row["query_id"]))
    if any(not isinstance(item, str) or not item for item in ranked_ids):
        raise InputContractError("{}: ranked_ids must be non-empty strings".format(row["query_id"]))

    primary = row["primary_target_ids"]
    context = row["relation_context_chunk_ids"]
    must_not = row["must_not_chunk_ids"]
    primary_ranks = _ranks(ranked_ids, primary, k)
    context_ranks = _ranks(ranked_ids, context, k)
    must_not_ranks = _ranks(ranked_ids, must_not, k)
    role = row["case_role"]

    diagnostic = {
        "query_id": row["query_id"],
        "case_role": role,
        "expected_behavior": row["expected_behavior"],
        "primary_target_count": len(primary),
        "relation_context_count": len(context),
        "must_not_count": len(must_not),
        "observed_relation_context_ranks_at_k": context_ranks,
        "hit_at_k": None,
        "first_primary_rank": None,
        "mrr_at_k": None,
        "rank1": None,
        "recall_at_k": None,
        "ndcg_at_k": None,
        "must_not_clean_at_k": None,
        "must_not_violations_at_k": [],
        "primary_metric_population": "excluded",
    }

    if role in {POSITIVE_ANY_HIT, POSITIVE_ALL_RELEVANT}:
        all_primary_ranks = _ranks(ranked_ids, primary, max(k, mrr_k))
        first_rank = min(all_primary_ranks) if all_primary_ranks else None
        diagnostic.update({
            "hit_at_k": bool(primary_ranks),
            "first_primary_rank": first_rank,
            "mrr_at_k": 0.0 if first_rank is None or first_rank > mrr_k else 1.0 / first_rank,
            "rank1": first_rank == 1,
            "primary_metric_population": "included",
        })
        if role == POSITIVE_ALL_RELEVANT:
            unique_hits = len(set(ranked_ids[:k]) & set(primary))
            diagnostic["recall_at_k"] = unique_hits / len(primary)
            diagnostic["ndcg_at_k"] = _ndcg_at_k(ranked_ids, primary, ndcg_k)
    elif role == NEGATIVE_SOURCE_ROUTING:
        diagnostic["must_not_clean_at_k"] = not must_not_ranks
        diagnostic["must_not_violations_at_k"] = [ranked_ids[rank - 1] for rank in must_not_ranks]
    elif role == RELATION_ONLY:
        pass

    return diagnostic


def _mean(values):
    return None if not values else sum(values) / len(values)


def aggregate_diagnostics(diagnostics):
    role_counts = Counter(row["case_role"] for row in diagnostics)
    positive = [row for row in diagnostics if row["case_role"] in {POSITIVE_ANY_HIT, POSITIVE_ALL_RELEVANT}]
    all_relevant = [row for row in diagnostics if row["case_role"] == POSITIVE_ALL_RELEVANT]
    negatives = [row for row in diagnostics if row["case_role"] == NEGATIVE_SOURCE_ROUTING]
    return {
        "case_count": len(diagnostics),
        "case_role_counts": {role: role_counts.get(role, 0) for role in sorted(CASE_ROLES)},
        "positive_case_count": len(positive),
        "positive_any_hit_case_count": role_counts.get(POSITIVE_ANY_HIT, 0),
        "positive_all_relevant_case_count": role_counts.get(POSITIVE_ALL_RELEVANT, 0),
        "negative_source_routing_case_count": len(negatives),
        "relation_only_not_primary_scored_count": role_counts.get(RELATION_ONLY, 0),
        "hit_at_k": _mean([float(row["hit_at_k"]) for row in positive]),
        "mrr_at_k": _mean([float(row["mrr_at_k"]) for row in positive]),
        "rank1_rate": _mean([float(row["rank1"]) for row in positive]),
        "all_relevant_recall_at_k": _mean([float(row["recall_at_k"]) for row in all_relevant]),
        "all_relevant_ndcg_at_k": _mean([float(row["ndcg_at_k"]) for row in all_relevant]),
        "negative_must_not_clean_rate_at_k": _mean([float(row["must_not_clean_at_k"]) for row in negatives]),
    }


def evaluate_rows(query_rows, ranked_by_query, route_name, k=20, mrr_k=10, ndcg_k=10):
    normalized = [normalize_query_row(row) for row in query_rows]
    query_ids = [row["query_id"] for row in normalized]
    if len(query_ids) != len(set(query_ids)):
        raise InputContractError("query_id values must be unique")
    unknown = sorted(set(ranked_by_query) - set(query_ids))
    missing = sorted(set(query_ids) - set(ranked_by_query))
    if unknown or missing:
        raise InputContractError("ranked/query key mismatch: missing={!r}, unknown={!r}".format(missing, unknown))
    diagnostics = [evaluate_case(row, list(ranked_by_query[row["query_id"]]), k, mrr_k, ndcg_k) for row in normalized]
    return {
        "schema_version": "case-role-ranked-id-evaluation-2.0",
        "status": "REVIEW_ONLY_MACHINE_DIAGNOSTIC",
        "route": route_name,
        "ranking_source": "caller_supplied_ranked_ids_unchanged",
        "k": k,
        "mrr_k": mrr_k,
        "ndcg_k": ndcg_k,
        "aggregate": aggregate_diagnostics(diagnostics),
        "diagnostics": diagnostics,
        "human_review": {"status": "UNFILLED", "decision": None, "reviewer": None},
        "human_pass_substitute": False,
        "embedding_created": False,
        "index_built": False,
        "fusion_recomputed": False,
        "kg_activated": False,
        "production": False,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate supplied ranked IDs with explicit four-role semantics")
    parser.add_argument("--queries", type=Path, required=True, help="JSONL query contract rows")
    parser.add_argument("--ranked-results", type=Path, required=True, help="JSONL query_id/ranked_ids rows")
    parser.add_argument("--route-name", required=True, help="Label for the supplied ranking, for example lexical or route_safe")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--mrr-k", type=int, default=10)
    parser.add_argument("--ndcg-k", type=int, default=10)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        query_rows = load_jsonl(args.queries)
        ranked_by_query = normalize_ranked_rows(load_jsonl(args.ranked_results))
        report = evaluate_rows(query_rows, ranked_by_query, args.route_name, args.k, args.mrr_k, args.ndcg_k)
    except InputContractError as exc:
        raise SystemExit("input contract error: {}".format(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
