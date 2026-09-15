"""Candidate lexical/dense/hybrid query over a curated vector bundle.

Lexical search is a transparent character/bigram plus English-token BM25
implementation.  Dense search uses only the verified BGE matrix and the
direct ONNX encoder; it has no silent lexical or hash-vector fallback.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np

from curated_vectors import CuratedOnnxEncoder


DIMENSIONS = 512
NORM_TOLERANCE = 0.00001
RRF_K = 60
DEFAULT_LIMIT = 5
DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "knowledge/curated-four-books-v1"
PRIVATE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Z0-9])[A-Z]:[\\/]|\\\\(?:users|home|private|documents)[\\/]"
)
ENGLISH_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*")
CHINESE_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)
CHUNK_ONLY_FIELDS = {
    "chunk_id",
    "search_text",
    "embedding_text",
    "body_start",
    "body_end",
    "full_text_sha256",
    "token_count",
    "segment_index",
    "segment_count",
}


class QueryError(RuntimeError):
    """Base error for fail-closed bundle query operations."""


class QueryInputError(QueryError):
    """Raised for an invalid query or filter."""


class BundleIntegrityError(QueryError):
    """Raised when persisted bundle files or metadata drift."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BundleIntegrityError(f"missing bundle file: {path.name}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleIntegrityError(
                f"invalid JSON at {path.name}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise BundleIntegrityError(f"non-object JSON at {path.name}:{line_number}")
        rows.append(value)
    return rows


def _safe_relative(bundle: Path, relative_name: str) -> Path:
    path = Path(relative_name)
    if path.is_absolute() or ".." in path.parts:
        raise BundleIntegrityError("manifest contains a non-relative payload path")
    resolved = (bundle / path).resolve()
    if bundle not in resolved.parents and resolved != bundle:
        raise BundleIntegrityError("manifest payload escapes bundle")
    return resolved


def _is_chinese(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in CHINESE_RANGES)


def _lexical_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        if _is_chinese(text[index]):
            start = index
            while index < len(text) and _is_chinese(text[index]):
                index += 1
            run = text[start:index]
            tokens.extend(run)
            tokens.extend(run[pos : pos + 2] for pos in range(len(run) - 1))
            continue
        match = ENGLISH_TOKEN.match(text, index)
        if match:
            tokens.append(match.group(0).lower())
            index = match.end()
            continue
        index += 1
    return tokens


def _bm25(
    query_tokens: list[str],
    document_tokens: list[str],
    document_frequency: Counter[str],
    document_count: int,
    average_length: float,
) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    frequencies = Counter(document_tokens)
    query_frequency = Counter(query_tokens)
    k1 = 1.2
    b = 0.75
    score = 0.0
    length = len(document_tokens)
    for token, qtf in query_frequency.items():
        df = document_frequency.get(token, 0)
        if not df:
            continue
        idf = np.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
        term_frequency = frequencies[token]
        saturation = (
            term_frequency * (k1 + 1.0)
            / (term_frequency + k1 * (1.0 - b + b * length / average_length))
        )
        score += float(idf) * saturation * (1.0 + np.log1p(qtf))
    return float(score)


class CuratedBundle:
    """Read-only integrity-checked bundle and bounded candidate search."""

    def __init__(self, bundle: str | Path = DEFAULT_BUNDLE) -> None:
        self.bundle = Path(bundle).resolve()
        manifest_path = self.bundle / "vector_manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError("cannot read vector_manifest.json") from exc
        if self.manifest.get("production_activated") is not False:
            raise BundleIntegrityError("candidate bundle has invalid production flag")
        if self.manifest.get("public_preview_is_production") is not False:
            raise BundleIntegrityError("candidate bundle has invalid preview flag")
        self.chunks = _jsonl(
            _safe_relative(self.bundle, self.manifest["chunks"]["file"])
        )
        self.mapping = _jsonl(
            _safe_relative(self.bundle, self.manifest["chunk_to_kg"]["file"])
        )
        self._load_normative_knowledge_units()
        self._validate_chunk_payload()
        self._validate_mapping()
        self._load_knowledge_units()
        self._doc_tokens = [_lexical_tokens(row["search_text"]) for row in self.chunks]
        self._matrix: np.ndarray | None = None

    def _load_normative_knowledge_units(self) -> None:
        input_info = self.manifest.get("input")
        if not isinstance(input_info, dict):
            raise BundleIntegrityError("manifest input contract is missing")
        ku_name = input_info.get("knowledge_units_file")
        if not isinstance(ku_name, str):
            raise BundleIntegrityError("manifest knowledge_units_file is missing")
        ku_path = _safe_relative(self.bundle, ku_name)
        raw = ku_path.read_bytes()
        if _sha256_bytes(raw) != input_info.get("knowledge_units_sha256"):
            raise BundleIntegrityError("knowledge_units.jsonl hash mismatch")
        rows = _jsonl(ku_path)
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            unit_id = row.get("knowledge_unit_id")
            if not isinstance(unit_id, str) or not unit_id:
                raise BundleIntegrityError("normative KU has invalid knowledge_unit_id")
            if unit_id in by_id:
                raise BundleIntegrityError(f"duplicate normative KU: {unit_id}")
            if row.get("node_id") != unit_id:
                raise BundleIntegrityError(f"normative KU {unit_id} has mismatched node_id")
            if not isinstance(row.get("text"), str) or not row["text"]:
                raise BundleIntegrityError(f"normative KU {unit_id} has empty text")
            if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row.get(
                "text_sha256"
            ):
                raise BundleIntegrityError(f"normative KU {unit_id} text hash mismatch")
            for field in (
                "content_available",
                "retrieval_eligible",
                "embedding_eligible",
            ):
                if row.get(field) is not True:
                    raise BundleIntegrityError(
                        f"normative KU {unit_id} has {field} != true"
                    )
            if row.get("embedding_exclusion_reason") is not None:
                raise BundleIntegrityError(
                    f"normative KU {unit_id} has an exclusion reason"
                )
            if PRIVATE_PATH_PATTERN.search(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
            ):
                raise BundleIntegrityError(
                    f"private absolute path in normative KU {unit_id}"
                )
            by_id[unit_id] = row
        if input_info.get("knowledge_unit_count") != len(by_id):
            raise BundleIntegrityError("normative KU count mismatch")
        self.normative_knowledge_units = by_id

    def _validate_chunk_payload(self) -> None:
        expected_sha = self.manifest.get("chunks", {}).get("sha256")
        actual_sha = _sha256_file(
            _safe_relative(self.bundle, self.manifest["chunks"]["file"])
        )
        if expected_sha != actual_sha:
            raise BundleIntegrityError("rag_chunks.jsonl hash mismatch")
        expected_rows = self.manifest.get("chunks", {}).get("rows")
        if expected_rows != len(self.chunks):
            raise BundleIntegrityError("rag_chunks row count mismatch")
        ids = [row.get("chunk_id") for row in self.chunks]
        if any(not isinstance(value, str) for value in ids):
            raise BundleIntegrityError("chunk_id must be a string")
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise BundleIntegrityError("chunk IDs are not sorted unique values")
        for row in self.chunks:
            for field in (
                "knowledge_unit_id",
                "title",
                "text",
                "text_sha256",
                "applicability",
                "units_basis",
                "source_id",
                "subject_root",
                "search_text",
                "embedding_text",
                "full_text_sha256",
            ):
                if field not in row:
                    raise BundleIntegrityError(
                        f"chunk {row.get('chunk_id')} missing {field}"
                    )
            if row["search_text"] != row["embedding_text"]:
                raise BundleIntegrityError(
                    f"search_text/embedding_text mismatch for {row['chunk_id']}"
                )
            if row["full_text_sha256"] != row["text_sha256"]:
                raise BundleIntegrityError(
                    f"full text hash mismatch for {row['chunk_id']}"
                )
            if _sha256_bytes(row["search_text"].encode("utf-8")) != (
                self._entry_by_chunk_id().get(row["chunk_id"], {}).get("input_sha256")
            ):
                raise BundleIntegrityError(
                    f"search input hash mismatch for {row['chunk_id']}"
                )
            if row.get("retrieval_eligible") is not True:
                raise BundleIntegrityError(
                    f"chunk {row['chunk_id']} is not retrieval eligible"
                )
            if row.get("embedding_eligible") is not True:
                raise BundleIntegrityError(
                    f"chunk {row['chunk_id']} is not embedding eligible"
                )
            if row.get("content_available") is not True:
                raise BundleIntegrityError(
                    f"chunk {row['chunk_id']} is not content available"
                )
            if row.get("embedding_exclusion_reason") is not None:
                raise BundleIntegrityError(
                    f"chunk {row['chunk_id']} has an exclusion reason"
                )
            if int(row["token_count"]) > 512:
                raise BundleIntegrityError(
                    f"chunk {row['chunk_id']} exceeds token limit"
                )
            if PRIVATE_PATH_PATTERN.search(
                json.dumps(row, ensure_ascii=False, sort_keys=True)
            ):
                raise BundleIntegrityError(
                    f"private absolute path in chunk {row['chunk_id']}"
                )

    def _entry_by_chunk_id(self) -> dict[str, dict[str, Any]]:
        entries = self.manifest.get("entries")
        if not isinstance(entries, list):
            raise BundleIntegrityError("manifest entries are missing")
        return {
            entry["chunk_id"]: entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("chunk_id"), str)
        }

    def _validate_mapping(self) -> None:
        expected_sha = self.manifest.get("chunk_to_kg", {}).get("sha256")
        actual_sha = _sha256_file(
            _safe_relative(self.bundle, self.manifest["chunk_to_kg"]["file"])
        )
        if expected_sha != actual_sha:
            raise BundleIntegrityError("chunk_to_kg.jsonl hash mismatch")
        if self.manifest.get("chunk_to_kg", {}).get("rows") != len(self.mapping):
            raise BundleIntegrityError("chunk_to_kg row count mismatch")
        chunk_ids = [row.get("chunk_id") for row in self.mapping]
        if chunk_ids != sorted(chunk_ids) or chunk_ids != [
            row["chunk_id"] for row in self.chunks
        ]:
            raise BundleIntegrityError("chunk mapping order does not match chunks")
        for chunk, mapping in zip(self.chunks, self.mapping, strict=True):
            if mapping.get("knowledge_unit_id") != chunk.get("knowledge_unit_id"):
                raise BundleIntegrityError(
                    f"chunk mapping mismatch for {chunk['chunk_id']}"
                )

    def _load_knowledge_units(self) -> None:
        by_id: dict[str, dict[str, Any]] = {}
        for row in self.chunks:
            unit_id = row["knowledge_unit_id"]
            chunk_unit = {
                key: value
                for key, value in row.items()
                if key not in CHUNK_ONLY_FIELDS
            }
            unit = self.normative_knowledge_units.get(unit_id)
            if unit is None:
                raise BundleIntegrityError(
                    f"chunk {row['chunk_id']} references unknown normative KU"
                )
            if chunk_unit != unit:
                raise BundleIntegrityError(
                    f"chunk copy of KU {unit_id} differs from normative KU"
                )
            previous = by_id.get(unit_id)
            if previous is not None and previous != unit:
                raise BundleIntegrityError(
                    f"chunk copies of KU {unit_id} are inconsistent"
                )
            by_id[unit_id] = unit
        expected_count = self.manifest.get("coverage", {}).get(
            "knowledge_unit_count"
        )
        if expected_count != len(by_id):
            raise BundleIntegrityError("knowledge unit coverage count mismatch")
        if set(by_id) != set(self.normative_knowledge_units):
            raise BundleIntegrityError("chunk/KU coverage identity mismatch")
        if not self.manifest.get("coverage", {}).get(
            "all_knowledge_units_covered"
        ):
            raise BundleIntegrityError("manifest does not assert KU coverage")
        self.knowledge_units = by_id

    def load_matrix(self) -> np.ndarray:
        if self._matrix is not None:
            return self._matrix
        info = self.manifest.get("matrix", {})
        matrix_path = _safe_relative(self.bundle, info["file"])
        raw = matrix_path.read_bytes()
        if _sha256_bytes(raw) != info.get("sha256"):
            raise BundleIntegrityError("embedding matrix hash mismatch")
        rows = int(info.get("rows", -1))
        dimensions = int(info.get("dimensions", -1))
        if dimensions != DIMENSIONS or info.get("dtype") != "<f4":
            raise BundleIntegrityError("embedding matrix dtype/dimension contract mismatch")
        if rows != len(self.chunks) or len(raw) != rows * dimensions * 4:
            raise BundleIntegrityError("embedding matrix shape/byte-size mismatch")
        matrix = np.frombuffer(raw, dtype="<f4").reshape(rows, dimensions)
        if not np.all(np.isfinite(matrix)):
            raise BundleIntegrityError("embedding matrix has non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        max_error = float(np.max(np.abs(norms - 1.0), initial=0.0))
        if info.get("row_order") != "chunk_id ascending":
            raise BundleIntegrityError("embedding matrix row_order contract mismatch")
        tolerance = float(info.get("norm_absolute_tolerance", NORM_TOLERANCE))
        if not np.isfinite(tolerance) or tolerance <= 0 or tolerance > NORM_TOLERANCE:
            raise BundleIntegrityError("embedding norm tolerance is too loose")
        if max_error > tolerance:
            raise BundleIntegrityError("embedding matrix norm check failed")
        entries = self.manifest.get("entries")
        if not isinstance(entries, list) or len(entries) != rows:
            raise BundleIntegrityError("embedding entries row count mismatch")
        for row_number, (entry, chunk, vector) in enumerate(
            zip(entries, self.chunks, matrix, strict=True)
        ):
            if entry.get("row") != row_number or entry.get("chunk_id") != chunk["chunk_id"]:
                raise BundleIntegrityError("embedding entry order mismatch")
            vector_bytes = np.asarray(vector, dtype="<f4").tobytes(order="C")
            if entry.get("vector_sha256") != _sha256_bytes(vector_bytes):
                raise BundleIntegrityError(
                    f"vector hash mismatch for {chunk['chunk_id']}"
                )
            if entry.get("input_sha256") != _sha256_bytes(
                chunk["search_text"].encode("utf-8")
            ):
                raise BundleIntegrityError(
                    f"input hash mismatch for {chunk['chunk_id']}"
                )
        self._matrix = matrix
        return matrix

    def _filter_indices(
        self,
        *,
        node_id: str | None,
        source_filter: str | None,
        subject_filter: str | None,
    ) -> list[int]:
        if source_filter is not None:
            if not source_filter.strip():
                raise QueryInputError("source filter must not be empty")
            if source_filter not in {row["source_id"] for row in self.chunks}:
                raise QueryInputError(f"unknown source filter: {source_filter}")
        if subject_filter is not None:
            if not subject_filter.strip():
                raise QueryInputError("subject filter must not be empty")
            if subject_filter not in {row["subject_root"] for row in self.chunks}:
                raise QueryInputError(f"unknown subject filter: {subject_filter}")
        indices = []
        for index, row in enumerate(self.chunks):
            if source_filter is not None and row["source_id"] != source_filter:
                continue
            if subject_filter is not None and row["subject_root"] != subject_filter:
                continue
            if node_id is not None and row["knowledge_unit_id"] != node_id:
                continue
            indices.append(index)
        return indices

    def _rank_lexical(self, query: str, indices: list[int]) -> list[tuple[int, float]]:
        tokens = _lexical_tokens(query)
        if not tokens:
            raise QueryInputError("query has no searchable Chinese/English tokens")
        document_frequency: Counter[str] = Counter()
        for index in indices:
            document_frequency.update(set(self._doc_tokens[index]))
        average_length = max(
            1.0, sum(len(self._doc_tokens[index]) for index in indices) / len(indices)
        )
        scored = [
            (
                index,
                _bm25(
                    tokens,
                    self._doc_tokens[index],
                    document_frequency,
                    len(indices),
                    average_length,
                ),
            )
            for index in indices
        ]
        return sorted(
            [item for item in scored if item[1] > 0.0],
            key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]),
        )

    def _validate_dense_model(
        self,
        encoder: CuratedOnnxEncoder,
        lock_path: str | Path,
    ) -> None:
        try:
            lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BundleIntegrityError("cannot read dense model lock") from exc
        model = self.manifest.get("model")
        runtime = self.manifest.get("runtime")
        if not isinstance(model, dict) or not isinstance(runtime, dict):
            raise BundleIntegrityError("dense model/runtime manifest is missing")
        if model.get("name") != encoder.asset_report["model_version"]:
            raise BundleIntegrityError("dense model name differs from manifest")
        if model.get("revision") != lock.get("revision"):
            raise BundleIntegrityError("dense model revision differs from lock")
        if model.get("files") != lock.get("model_dir_sha256"):
            raise BundleIntegrityError("dense model hashes differ from lock")
        actual_files = {
            name: info["sha256"]
            for name, info in encoder.asset_report["files"].items()
        }
        if actual_files != model["files"]:
            raise BundleIntegrityError("dense model hashes differ from loaded assets")
        if runtime.get("implementation") != "direct-onnxruntime":
            raise BundleIntegrityError("dense runtime implementation differs")
        if runtime.get("provider") != "CPUExecutionProvider":
            raise BundleIntegrityError("dense runtime provider differs")
        if runtime.get("threads") != 4 or runtime.get("batch_size") != 32:
            raise BundleIntegrityError("dense runtime thread/batch contract differs")
        if runtime.get("tokenizer_truncation") is not False:
            raise BundleIntegrityError("dense tokenizer truncation contract differs")

    def _rank_dense(
        self,
        query: str,
        indices: list[int],
        *,
        model_dir: str | Path | None,
        lock_path: str | Path | None,
        vendor_path: str | Path | None,
    ) -> list[tuple[int, float]]:
        if model_dir is None or lock_path is None:
            raise QueryInputError(
                "dense/hybrid query requires --model-dir and --lock"
            )
        matrix = self.load_matrix()
        encoder = CuratedOnnxEncoder(
            model_dir,
            lock_path=lock_path,
            vendor_path=vendor_path,
        )
        self._validate_dense_model(encoder, lock_path)
        query_vector = encoder.encode_texts([query])[0]
        scored = [(index, float(matrix[index] @ query_vector)) for index in indices]
        return sorted(
            scored,
            key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]),
        )

    def _group_best(
        self, ranked: Iterable[tuple[int, float]]
    ) -> dict[str, tuple[int, float]]:
        best: dict[str, tuple[int, float]] = {}
        for index, score in ranked:
            unit_id = self.chunks[index]["knowledge_unit_id"]
            old = best.get(unit_id)
            if old is None or score > old[1] or (
                score == old[1]
                and self.chunks[index]["chunk_id"] < self.chunks[old[0]]["chunk_id"]
            ):
                best[unit_id] = (index, score)
        return best

    def _result(
        self,
        index: int,
        score: float,
        *,
        method: str,
        channel_score: float | None = None,
    ) -> dict[str, Any]:
        chunk = self.chunks[index]
        unit = self.knowledge_units[chunk["knowledge_unit_id"]]
        return {
            "knowledge_unit_id": chunk["knowledge_unit_id"],
            "knowledge_unit": unit,
            "best_chunk_id": chunk["chunk_id"],
            "body_start": chunk["body_start"],
            "body_end": chunk["body_end"],
            "token_count": chunk["token_count"],
            "score": float(score),
            "channel_score": (
                float(channel_score) if channel_score is not None else None
            ),
            "method": method,
            "candidate_status": "candidate_vector_preview",
            "engineering_certification": False,
        }

    def search(
        self,
        *,
        query: str | None = None,
        node_id: str | None = None,
        method: str = "lexical",
        limit: int = DEFAULT_LIMIT,
        source_filter: str | None = None,
        subject_filter: str | None = None,
        model_dir: str | Path | None = None,
        lock_path: str | Path | None = None,
        vendor_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        if method not in {"lexical", "dense", "hybrid"}:
            raise QueryInputError(f"unknown method: {method}")
        if limit <= 0:
            raise QueryInputError("limit must be positive")
        if query is not None and not isinstance(query, str):
            raise QueryInputError("query must be a string")
        if node_id is not None and (not isinstance(node_id, str) or not node_id.strip()):
            raise QueryInputError("node_id must be a non-empty string")
        if query is not None and "\x00" in query:
            raise QueryInputError("query contains NUL")
        if query is None or not query.strip():
            if node_id is None:
                raise QueryInputError("query or node_id is required")
            query = None
        indices = self._filter_indices(
            node_id=node_id,
            source_filter=source_filter,
            subject_filter=subject_filter,
        )
        if not indices:
            return []
        if node_id is not None and query is None:
            exact = self._group_best(
                (index, 1.0) for index in indices
            )
            ordered = sorted(exact.items(), key=lambda item: item[0])[:limit]
            return [
                self._result(index, score, method="exact-node-id")
                for _, (index, score) in ordered
            ]

        if query is None:
            raise QueryInputError("query is required for scored search")
        if method == "lexical":
            lexical_rank = self._rank_lexical(query, indices)
            grouped = self._group_best(lexical_rank)
            ranked_units = sorted(
                grouped.items(),
                key=lambda item: (-item[1][1], item[0]),
            )[:limit]
            return [
                self._result(index, score, method="lexical", channel_score=score)
                for _, (index, score) in ranked_units
            ]

        dense_rank = self._rank_dense(
            query,
            indices,
            model_dir=model_dir,
            lock_path=lock_path,
            vendor_path=vendor_path,
        )
        if method == "dense":
            grouped = self._group_best(dense_rank)
            ranked_units = sorted(
                grouped.items(),
                key=lambda item: (-item[1][1], item[0]),
            )[:limit]
            return [
                self._result(index, score, method="dense", channel_score=score)
                for _, (index, score) in ranked_units
            ]

        lexical_rank = self._rank_lexical(query, indices)
        lexical_units = self._group_best(lexical_rank)
        dense_units = self._group_best(dense_rank)
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        for rank, (unit_id, _) in enumerate(
            sorted(
                lexical_units.items(),
                key=lambda item: (-item[1][1], item[0]),
            ),
            start=1,
        ):
            rrf_scores[unit_id] += 1.0 / (RRF_K + rank)
        for rank, (unit_id, _) in enumerate(
            sorted(
                dense_units.items(),
                key=lambda item: (-item[1][1], item[0]),
            ),
            start=1,
        ):
            rrf_scores[unit_id] += 1.0 / (RRF_K + rank)
        ranked_units = sorted(
            rrf_scores.items(), key=lambda item: (-item[1], item[0])
        )[:limit]
        results = []
        for unit_id, score in ranked_units:
            lexical_item = lexical_units.get(unit_id)
            dense_item = dense_units.get(unit_id)
            if lexical_item is None:
                chosen_index, channel_score = dense_item
            elif dense_item is None:
                chosen_index, channel_score = lexical_item
            else:
                chosen_index, channel_score = dense_item
            results.append(
                self._result(
                    chosen_index,
                    score,
                    method="hybrid",
                    channel_score=channel_score,
                )
            )
        return results


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--query")
    parser.add_argument("--node-id")
    parser.add_argument(
        "--method", choices=("lexical", "dense", "hybrid"), default="lexical"
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--source", dest="source_filter")
    parser.add_argument("--subject", dest="subject_filter")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--vendor", type=Path)
    args = parser.parse_args()
    store = CuratedBundle(args.bundle)
    results = store.search(
        query=args.query,
        node_id=args.node_id,
        method=args.method,
        limit=args.limit,
        source_filter=args.source_filter,
        subject_filter=args.subject_filter,
        model_dir=args.model_dir,
        lock_path=args.lock,
        vendor_path=args.vendor,
    )
    print(
        json.dumps(
            {
                "status": "candidate_result",
                "method": args.method,
                "query": args.query,
                "node_id": args.node_id,
                "limit": args.limit,
                "source": args.source_filter,
                "subject": args.subject_filter,
                "lexical_definition": (
                    "Chinese characters and adjacent bigrams plus lowercased "
                    "English/alphanumeric tokens scored with BM25"
                ),
                "rrf_k": RRF_K if args.method == "hybrid" else None,
                "no_answer_threshold": None,
                "candidate_scope": (
                    "Public preview only; not production retrieval or "
                    "engineering certification."
                ),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
