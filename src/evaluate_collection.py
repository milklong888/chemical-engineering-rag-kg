"""Evaluate the fixed two-bundle collection on public development queries.

This is a read-only evaluator for the ``query_collection.CuratedCollection``
contract.  It does not build vectors, invoke an encoder directly, discover
bundles, read held-out data, or invent an answer-rejection threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


METHODS: tuple[str, ...] = ("lexical", "dense", "hybrid")
EXPECTED_BUNDLES: tuple[str, ...] = (
    "curated-four-books-v1",
    "curated-s001-upper-v1",
)
EXPECTED_QUERY_COUNTS: Mapping[str, int] = {"four": 16, "s001": 16}
LIMIT = 5
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ABSOLUTE_PATH_RE = re.compile(
    r"(?:(?:[A-Za-z]:[\\/])|(?:\\\\))[^\s\"'<>;,)]*"
)


class CollectionEvaluationError(RuntimeError):
    """Raised for a fail-closed evaluator setup or input contract error."""


class _BundleSnapshot:
    def __init__(
        self,
        *,
        bundle_id: str,
        units: dict[str, dict[str, Any]],
        file_hashes: dict[str, str],
        manifest: dict[str, Any],
    ) -> None:
        self.bundle_id = bundle_id
        self.units = units
        self.file_hashes = file_hashes
        self.manifest = manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CollectionEvaluationError(f"cannot read {label}") from exc
    if not isinstance(value, dict):
        raise CollectionEvaluationError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CollectionEvaluationError(f"cannot read {label}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CollectionEvaluationError(
                f"invalid JSON in {label}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise CollectionEvaluationError(
                f"non-object JSON in {label}:{line_number}"
            )
        rows.append(value)
    return rows


def _public_error(exc: BaseException) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return f"{exc.__class__.__name__}: {ABSOLUTE_PATH_RE.sub('<path>', text)}"


def _validate_string_list(value: Any, *, field: str, context: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CollectionEvaluationError(f"{context} has invalid {field}")
    return list(value)


def _validate_query_row(
    row: Mapping[str, Any], *, label: str, line_number: int
) -> dict[str, Any]:
    context = f"{label}:{line_number}"
    query_id = row.get("query_id")
    query = row.get("query")
    if not isinstance(query_id, str) or not query_id.strip():
        raise CollectionEvaluationError(f"{context} has invalid query_id")
    if not isinstance(query, str) or not query.strip() or "\x00" in query:
        raise CollectionEvaluationError(f"{context} has invalid query")
    if row.get("case_role") != "positive_any_hit":
        raise CollectionEvaluationError(f"{context} is not positive_any_hit")
    if row.get("split") != "public_development":
        raise CollectionEvaluationError(f"{context} is not public_development")
    # The four-book fixture has target_grain; the existing S001 fixture does
    # not.  Missing target_grain is the established knowledge-unit default.
    target_grain = row.get("target_grain", "knowledge_unit")
    if target_grain != "knowledge_unit":
        raise CollectionEvaluationError(f"{context} has an unsupported target grain")
    primary_ids = _validate_string_list(
        row.get("primary_target_ids"),
        field="primary_target_ids",
        context=context,
    )
    expected = row.get("expected_knowledge_unit_ids")
    expected_ids = list(primary_ids) if expected is None else _validate_string_list(
        expected,
        field="expected_knowledge_unit_ids",
        context=context,
    )
    if expected_ids != primary_ids:
        raise CollectionEvaluationError(
            f"{context} expected IDs differ from primary_target_ids"
        )
    for field in ("source_filter", "subject_filter"):
        value = row.get(field)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise CollectionEvaluationError(f"{context} has invalid {field}")
    return {
        "query_id": query_id,
        "query": query,
        "primary_target_ids": primary_ids,
        "expected_knowledge_unit_ids": expected_ids,
        "case_role": row["case_role"],
        "target_grain": target_grain,
        "split": row["split"],
        "input_source_filter": row.get("source_filter"),
        "input_subject_filter": row.get("subject_filter"),
    }


def load_query_file(path: Path, *, group: str) -> list[dict[str, Any]]:
    """Load one exact 16-question public input file."""

    if group not in EXPECTED_QUERY_COUNTS:
        raise CollectionEvaluationError(f"unknown input group: {group}")
    rows = _read_jsonl(path, label=f"{group} query input")
    expected_count = EXPECTED_QUERY_COUNTS[group]
    if len(rows) != expected_count:
        raise CollectionEvaluationError(
            f"{group} query input must contain exactly {expected_count} rows"
        )
    normalized = [
        _validate_query_row(row, label=group, line_number=index)
        for index, row in enumerate(rows, start=1)
    ]
    ids = [row["query_id"] for row in normalized]
    if len(ids) != len(set(ids)):
        raise CollectionEvaluationError(f"{group} query input has duplicate query_id")
    return normalized


def combine_queries(four_path: Path, s001_path: Path) -> list[dict[str, Any]]:
    """Combine both fixed inputs, forcing both filters to null at call time."""

    combined: list[dict[str, Any]] = []
    for group, path in (("four", four_path), ("s001", s001_path)):
        for row in load_query_file(path, group=group):
            combined.append(
                {
                    "input_group": group,
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "primary_target_ids": list(row["primary_target_ids"]),
                    "expected_knowledge_unit_ids": list(
                        row["expected_knowledge_unit_ids"]
                    ),
                    "case_role": row["case_role"],
                    "target_grain": row["target_grain"],
                    "split": row["split"],
                    "input_source_filter": row["input_source_filter"],
                    "input_subject_filter": row["input_subject_filter"],
                    "source_filter": None,
                    "subject_filter": None,
                }
            )
    ids = [row["query_id"] for row in combined]
    if len(combined) != 32 or len(ids) != len(set(ids)):
        raise CollectionEvaluationError(
            "combined public development input must contain 32 unique queries"
        )
    return combined


def _safe_relative(root: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CollectionEvaluationError(f"{label} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise CollectionEvaluationError(f"{label} must stay under its root")
    resolved = (root / relative).resolve()
    root = root.resolve()
    if root not in resolved.parents and resolved != root:
        raise CollectionEvaluationError(f"{label} escapes its root")
    return resolved


def _bundle_paths(
    collection_path: Path, knowledge_root: Path
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve the final descriptor's two paths, relative to its parent only."""

    descriptor = _read_json(collection_path, label="collection descriptor")
    descriptor_parent = collection_path.parent.resolve()
    knowledge_root = knowledge_root.resolve()
    if descriptor_parent != knowledge_root:
        raise CollectionEvaluationError(
            "knowledge-root must equal the curated collection descriptor parent"
        )
    packages = descriptor.get("packages")
    if not isinstance(packages, list):
        raise CollectionEvaluationError("collection descriptor packages are missing")
    entries: dict[str, Any] = {}
    for entry in packages:
        if not isinstance(entry, dict) or not isinstance(entry.get("bundle_id"), str):
            raise CollectionEvaluationError("collection descriptor has an invalid package")
        bundle_id = entry["bundle_id"]
        if bundle_id in entries:
            raise CollectionEvaluationError("collection descriptor has duplicate bundle IDs")
        entries[bundle_id] = entry
    if set(entries) != set(EXPECTED_BUNDLES):
        raise CollectionEvaluationError(
            "collection descriptor must list exactly the two accepted bundles"
        )
    paths: dict[str, Path] = {}
    for bundle_id in EXPECTED_BUNDLES:
        entry = entries[bundle_id]
        path = _safe_relative(
            descriptor_parent,
            entry.get("path_relative_to_knowledge"),
            label=f"path for {bundle_id}",
        )
        if not path.is_dir() or not (path / "knowledge_units.jsonl").is_file():
            raise CollectionEvaluationError(f"accepted bundle is missing: {bundle_id}")
        paths[bundle_id] = path
    return paths, descriptor


def _manifest_file_names(bundle_path: Path, manifest: Mapping[str, Any]) -> list[str]:
    names = {"manifest.json", "vector_manifest.json", "knowledge_units.jsonl"}
    files = manifest.get("files")
    if isinstance(files, dict):
        names.update(name for name in files if isinstance(name, str))
    return sorted(
        name
        for name in names
        if _safe_relative(bundle_path, name, label="manifest file").is_file()
    )


def load_normative_snapshots(
    collection_path: Path, knowledge_root: Path
) -> tuple[
    dict[str, _BundleSnapshot],
    dict[str, tuple[str, dict[str, Any]]],
    dict[str, Any],
]:
    """Read the two accepted KU files and bind their persisted file hashes."""

    bundle_paths, descriptor = _bundle_paths(collection_path, knowledge_root)
    snapshots: dict[str, _BundleSnapshot] = {}
    global_units: dict[str, tuple[str, dict[str, Any]]] = {}
    for bundle_id in EXPECTED_BUNDLES:
        bundle_path = bundle_paths[bundle_id]
        manifest = _read_json(bundle_path / "manifest.json", label=f"{bundle_id} manifest")
        if manifest.get("bundle_id") != bundle_id:
            raise CollectionEvaluationError(f"{bundle_id} manifest bundle identity mismatch")
        descriptor_entry = next(
            entry for entry in descriptor["packages"] if entry["bundle_id"] == bundle_id
        )
        for field, filename in (
            ("manifest_sha256", "manifest.json"),
            ("vector_manifest_sha256", "vector_manifest.json"),
        ):
            expected = descriptor_entry.get(field)
            if expected is not None and expected != sha256_file(bundle_path / filename):
                raise CollectionEvaluationError(f"{bundle_id} {filename} hash mismatch")
        rows = _read_jsonl(
            bundle_path / "knowledge_units.jsonl",
            label=f"{bundle_id} knowledge units",
        )
        units: dict[str, dict[str, Any]] = {}
        for row in rows:
            unit_id = row.get("knowledge_unit_id")
            if not isinstance(unit_id, str) or not unit_id.strip():
                raise CollectionEvaluationError(f"{bundle_id} has an invalid KU ID")
            if unit_id in units:
                raise CollectionEvaluationError(f"{bundle_id} has duplicate KU ID")
            text = row.get("text")
            if not isinstance(text, str) or not text:
                raise CollectionEvaluationError(f"{bundle_id} has an empty KU body")
            if row.get("node_id") != unit_id:
                raise CollectionEvaluationError(f"{bundle_id} has a node/KU identity mismatch")
            if row.get("text_sha256") != _sha256_bytes(text.encode("utf-8")):
                raise CollectionEvaluationError(f"{bundle_id} has a body hash mismatch")
            units[unit_id] = row
            if unit_id in global_units:
                old_bundle, _ = global_units[unit_id]
                raise CollectionEvaluationError(
                    f"KU ID conflict across bundles: {old_bundle} and {bundle_id}"
                )
            global_units[unit_id] = (bundle_id, row)
        file_hashes = {
            name: sha256_file(bundle_path / name)
            for name in _manifest_file_names(bundle_path, manifest)
        }
        snapshots[bundle_id] = _BundleSnapshot(
            bundle_id=bundle_id,
            units=units,
            file_hashes=file_hashes,
            manifest=manifest,
        )
    return snapshots, global_units, descriptor


def _module_code_hashes(query_module_dir: Path, module: ModuleType) -> dict[str, str]:
    paths: dict[str, Path] = {}
    query_module_dir = query_module_dir.resolve()
    for path in query_module_dir.glob("*.py"):
        if path.is_file():
            paths[f"query_module/{path.name}"] = path.resolve()
    # Bind the two formal Python dependencies used by the current adapter when
    # they are exposed by the imported module; labels disclose no local path.
    for attribute in ("_single_bundle", "CuratedOnnxEncoder"):
        value = getattr(module, attribute, None)
        dependency = value if isinstance(value, ModuleType) else None
        if dependency is None and value is not None:
            dependency = sys.modules.get(getattr(value, "__module__", ""))
        source = getattr(dependency, "__file__", None)
        if isinstance(source, str) and source.lower().endswith(".py"):
            paths.setdefault(
                f"dependency/{dependency.__name__.replace('.', '/')}.py",
                Path(source).resolve(),
            )
    result = {
        label: sha256_file(path)
        for label, path in sorted(paths.items())
        if path.is_file()
    }
    if "query_module/query_collection.py" not in result:
        raise CollectionEvaluationError("query module source hash is missing")
    return result


def load_query_module(query_module_dir: Path) -> tuple[ModuleType, dict[str, str]]:
    """Load the supplied adapter only; construction/search happen in ``run``."""

    query_module_dir = query_module_dir.resolve()
    if not query_module_dir.is_dir():
        raise CollectionEvaluationError("query-module must be a directory")
    source_path = query_module_dir / "query_collection.py"
    if not source_path.is_file():
        raise CollectionEvaluationError("query-module is missing query_collection.py")
    module_name = f"collection_eval_query_{sha256_file(source_path)[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise CollectionEvaluationError("cannot load query_collection module")
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(query_module_dir))
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise CollectionEvaluationError(
            f"query_collection import failed: {_public_error(exc)}"
        ) from exc
    finally:
        sys.path[:] = old_path
    collection_type = getattr(module, "CuratedCollection", None)
    if not callable(collection_type) or not callable(getattr(collection_type, "search", None)):
        raise CollectionEvaluationError("CuratedCollection.search is missing")
    return module, _module_code_hashes(query_module_dir, module)


def _field_mismatch_names(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> list[str]:
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing_or_extra = (expected_keys - actual_keys) | (actual_keys - expected_keys)
    differing_shared = {
        name
        for name in expected_keys & actual_keys
        if expected[name] != actual[name]
    }
    return sorted(missing_or_extra | differing_shared)


def validate_returned_result(
    result: Any,
    *,
    method: str,
    normative_by_id: Mapping[str, tuple[str, dict[str, Any]]],
) -> tuple[dict[str, Any], list[str]]:
    """Validate a complete KU and return a body-free top-five projection."""

    del method  # retained in the helper signature for method-specific callers
    errors: list[str] = []
    if not isinstance(result, dict):
        return (
            {"bundle_id": None, "knowledge_unit_id": None, "text_sha256": None, "score": None},
            ["result_is_not_object"],
        )
    bundle_id = result.get("bundle_id")
    ku_id = result.get("knowledge_unit_id")
    if not isinstance(bundle_id, str) or not bundle_id:
        errors.append("missing_bundle_id")
    if not isinstance(ku_id, str) or not ku_id:
        errors.append("missing_knowledge_unit_id")
    pair = normative_by_id.get(ku_id) if isinstance(ku_id, str) else None
    expected_bundle: str | None = None
    expected_unit: dict[str, Any] | None = None
    if pair is None:
        errors.append("unknown_knowledge_unit_id")
    else:
        expected_bundle, expected_unit = pair
        if bundle_id != expected_bundle:
            errors.append("bundle_id_for_knowledge_unit_mismatch")

    unit = result.get("knowledge_unit")
    if not isinstance(unit, dict):
        errors.append("complete_knowledge_unit_missing")
    elif expected_unit is not None:
        mismatch_names = _field_mismatch_names(expected_unit, unit)
        if mismatch_names:
            errors.append("knowledge_unit_fields_mismatch:" + ",".join(mismatch_names))
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        errors.append("score_is_not_numeric")
        safe_score: float | None = None
    elif not math.isfinite(float(score)):
        errors.append("score_is_not_finite")
        safe_score = None
    else:
        safe_score = float(score)
    text_sha = unit.get("text_sha256") if isinstance(unit, dict) else None
    if not isinstance(text_sha, str) or not SHA256_RE.fullmatch(text_sha):
        text_sha = None
        if "complete_knowledge_unit_missing" not in errors:
            errors.append("body_hash_missing_or_invalid")
    return (
        {
            "bundle_id": bundle_id if isinstance(bundle_id, str) else None,
            "knowledge_unit_id": ku_id if isinstance(ku_id, str) else None,
            "text_sha256": text_sha,
            "score": safe_score,
        },
        errors,
    )


def _metric_summary(
    queries: Sequence[Mapping[str, Any]], per_query: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    hits = 0
    rank1 = 0
    reciprocal_sum = 0.0
    for query, diagnostic in zip(queries, per_query, strict=True):
        if diagnostic.get("api_error") is not None:
            continue
        targets = set(query["primary_target_ids"])
        hit_rank: int | None = None
        for item in diagnostic.get("top5", []):
            if item.get("validation_errors"):
                continue
            if item.get("knowledge_unit_id") in targets:
                hit_rank = int(item["rank"])
                break
        if hit_rank is not None:
            hits += 1
            reciprocal_sum += 1.0 / hit_rank
            rank1 += hit_rank == 1
    total = len(queries)
    return {
        "query_count": total,
        "hit_at_5": hits / total if total else 0.0,
        "mrr_at_5": reciprocal_sum / total if total else 0.0,
        "top1": rank1 / total if total else 0.0,
        "valid_complete_ku_hit_count": hits,
    }


def evaluate_method(
    collection: Any,
    queries: Sequence[Mapping[str, Any]],
    *,
    method: str,
    normative_by_id: Mapping[str, tuple[str, dict[str, Any]]],
    model_dir: Path,
    lock_path: Path,
    vendor_path: Path,
) -> dict[str, Any]:
    if method not in METHODS:
        raise CollectionEvaluationError(f"unknown evaluation method: {method}")
    diagnostics: list[dict[str, Any]] = []
    validation_error_count = 0
    for query in queries:
        diagnostic: dict[str, Any] = {
            "input_group": query["input_group"],
            "query_id": query["query_id"],
            "query": query["query"],
            "primary_target_ids": list(query["primary_target_ids"]),
            "expected_knowledge_unit_ids": list(query["expected_knowledge_unit_ids"]),
            "input_source_filter": query["input_source_filter"],
            "input_subject_filter": query["input_subject_filter"],
            "source_filter": None,
            "subject_filter": None,
            "top5": [],
            "api_error": None,
        }
        try:
            raw = collection.search(
                query=query["query"],
                method=method,
                limit=LIMIT,
                source_filter=None,
                subject_filter=None,
                model_dir=model_dir,
                lock_path=lock_path,
                vendor_path=vendor_path,
            )
            if not isinstance(raw, list):
                raise CollectionEvaluationError("search did not return a list")
            if len(raw) > LIMIT:
                diagnostic["api_error"] = "search_returned_more_than_limit"
                validation_error_count += 1
                raw = raw[:LIMIT]
            seen: set[tuple[Any, Any]] = set()
            for rank, result in enumerate(raw, start=1):
                projection, errors = validate_returned_result(
                    result,
                    method=method,
                    normative_by_id=normative_by_id,
                )
                identity = (projection["bundle_id"], projection["knowledge_unit_id"])
                if identity in seen:
                    errors = list(errors) + ["duplicate_result_identity"]
                seen.add(identity)
                projection["rank"] = rank
                projection["validation_errors"] = errors
                diagnostic["top5"].append(projection)
                validation_error_count += len(errors)
        except Exception as exc:
            diagnostic["api_error"] = _public_error(exc)
            validation_error_count += 1
        diagnostics.append(diagnostic)
    return {
        "api_call_count": len(queries),
        "aggregate": _metric_summary(queries, diagnostics),
        "validation_error_count": validation_error_count,
        "per_query": diagnostics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    four_path = Path(args.four_queries).resolve()
    s001_path = Path(args.s001_queries).resolve()
    collection_path = Path(args.collection).resolve()
    knowledge_root = Path(args.knowledge_root).resolve()
    model_dir = Path(args.model_dir).resolve()
    lock_path = Path(args.lock).resolve()
    vendor_path = Path(args.vendor).resolve()
    for path, label, kind in (
        (four_path, "four-book query input", "file"),
        (s001_path, "S001 query input", "file"),
        (collection_path, "collection descriptor", "file"),
        (knowledge_root, "knowledge root", "directory"),
        (model_dir, "model directory", "directory"),
        (lock_path, "model lock", "file"),
        (vendor_path, "vendor directory", "directory"),
    ):
        if kind == "file" and not path.is_file():
            raise CollectionEvaluationError(f"missing {label}")
        if kind == "directory" and not path.is_dir():
            raise CollectionEvaluationError(f"missing {label}")
    queries = combine_queries(four_path, s001_path)
    query_module, code_hashes = load_query_module(Path(args.query_module))
    snapshots, normative_by_id, descriptor = load_normative_snapshots(
        collection_path, knowledge_root
    )
    target_ids = {
        target_id
        for query in queries
        for target_id in query["primary_target_ids"]
    }
    if not target_ids.issubset(normative_by_id):
        raise CollectionEvaluationError("public target ID is absent from accepted KU files")
    try:
        collection = query_module.CuratedCollection(
            collection_path,
            knowledge_root=knowledge_root,
        )
    except Exception as exc:
        raise CollectionEvaluationError(
            f"CuratedCollection construction failed: {_public_error(exc)}"
        ) from exc
    methods = {
        method: evaluate_method(
            collection,
            queries,
            method=method,
            normative_by_id=normative_by_id,
            model_dir=model_dir,
            lock_path=lock_path,
            vendor_path=vendor_path,
        )
        for method in METHODS
    }
    validation_error_count = sum(
        report["validation_error_count"] for report in methods.values()
    )
    return {
        "schema": "curated-collection-public-development-evaluation-v1",
        "status": "review_only_machine_diagnostic",
        "scope": {
            "input_sets": {
                "four": {"query_count": EXPECTED_QUERY_COUNTS["four"]},
                "s001": {"query_count": EXPECTED_QUERY_COUNTS["s001"]},
            },
            "combined_query_count": len(queries),
            "methods": list(METHODS),
            "calls_expected": len(queries) * len(METHODS),
            "target_grain": "knowledge_unit",
            "source_filter": None,
            "subject_filter": None,
            "input_filters_overridden": True,
            "split": "public_development",
            "heldout_read": False,
            "no_answer_threshold": None,
            "limit": LIMIT,
        },
        "execution": {
            "api_call_count": len(queries) * len(METHODS),
            "methods": methods,
            "returned_validation_error_count": validation_error_count,
        },
        "accepted_normative_coverage": {
            "bundle_ids": list(EXPECTED_BUNDLES),
            "knowledge_unit_counts": {
                bundle_id: len(snapshot.units)
                for bundle_id, snapshot in snapshots.items()
            },
            "combined_knowledge_unit_count": len(normative_by_id),
            "global_knowledge_unit_ids_unique": True,
        },
        "validation": {
            "complete_knowledge_units_compared_field_by_field": True,
            "id_and_batch_conflicts": [],
            "returned_results_with_any_error": validation_error_count,
            "metric_basis": "target IDs count only when the returned complete KU passes exact field comparison",
        },
        "provenance_hashes": {
            "input_files": {
                "curated-dev-queries.jsonl": sha256_file(four_path),
                "s001-public-development-queries.jsonl": sha256_file(s001_path),
            },
            "curated-collection-v1.json": sha256_file(collection_path),
            "accepted_bundle_files": {
                bundle_id: snapshot.file_hashes
                for bundle_id, snapshot in snapshots.items()
            },
            "model_lock.json": sha256_file(lock_path),
            "evaluator.py": sha256_file(Path(__file__).resolve()),
            "query_module_files": code_hashes,
        },
        "model_policy": {
            "model_dir_supplied_by_cli": True,
            "vendor_supplied_by_cli": True,
            "automatic_download": False,
            "model_switching": False,
            "fake_vectors": False,
        },
        "descriptor_no_answer_threshold": descriptor.get("no_answer_threshold"),
        "production_activated": False,
        "engineering_certification": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--knowledge-root", type=Path, required=True)
    parser.add_argument("--query-module", type=Path, required=True)
    parser.add_argument("--four-queries", type=Path, required=True)
    parser.add_argument("--s001-queries", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    try:
        report = run(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except CollectionEvaluationError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "status": report["status"],
                "combined_query_count": report["scope"]["combined_query_count"],
                "api_call_count": report["execution"]["api_call_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 1 if report["execution"]["returned_validation_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
