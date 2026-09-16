"""Read-only lexical/dense/hybrid search over two explicit curated bundles.

This candidate adapter composes the existing ``query_curated.CuratedBundle``
integrity and result contracts.  It does not discover bundles, rewrite source
records, build a third persisted index, or add an answer-rejection threshold.
The collection descriptor is intentionally explicit: only the bundles listed
there are loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

try:
    import query_curated as _single_bundle
    from curated_vectors import CuratedOnnxEncoder
    from query_curated import (
        BundleIntegrityError,
        DIMENSIONS,
        NORM_TOLERANCE,
        PRIVATE_PATH_PATTERN,
        QueryInputError,
        RRF_K,
        CuratedBundle,
        _lexical_tokens,
    )
except ImportError as exc:  # pragma: no cover - exercised by environment setup
    raise ImportError(
        "query_collection requires the formal src directory on PYTHONPATH"
    ) from exc


DEFAULT_COLLECTION = (
    Path(_single_bundle.__file__).resolve().parents[1]
    / "knowledge"
    / "curated-collection-v1.json"
)
_EXPECTED_PACKAGE_IDS = {
    "curated-four-books-v1",
    "curated-s001-upper-v1",
}
_MODEL_COMPATIBILITY_FIELDS = (
    "name",
    "revision",
    "files",
    "dimensions",
    "dtype",
    "normalization",
    "pooling",
)
_RUNTIME_COMPATIBILITY_FIELDS = (
    "implementation",
    "provider",
    "threads",
    "batch_size",
    "tokenizer_truncation",
    "pooling",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleIntegrityError(f"cannot read {description}") from exc
    if not isinstance(value, dict):
        raise BundleIntegrityError(f"{description} must be a JSON object")
    return value


def _safe_collection_path(root: Path, relative_name: str, *, description: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name.strip():
        raise BundleIntegrityError(f"{description} must be a relative path")
    path = Path(relative_name)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise BundleIntegrityError(f"{description} must stay under its root")
    resolved = (root / path).resolve()
    if root not in resolved.parents and resolved != root:
        raise BundleIntegrityError(f"{description} escapes its root")
    return resolved


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise BundleIntegrityError(f"cannot read {description}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BundleIntegrityError(
                f"invalid JSON in {description}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise BundleIntegrityError(
                f"non-object JSON in {description}:{line_number}"
            )
        rows.append(value)
    return rows


def _validate_manifest_files(package_dir: Path, manifest: dict[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleIntegrityError("package manifest has no payload file map")

    listed_names: set[str] = set()
    for name, metadata in files.items():
        if not isinstance(name, str) or not name:
            raise BundleIntegrityError("package manifest has an invalid file name")
        payload = _safe_collection_path(
            package_dir, name, description=f"package payload {name}"
        )
        relative = payload.relative_to(package_dir).as_posix()
        if relative != name:
            raise BundleIntegrityError("package payload paths must be normalized")
        if not payload.is_file():
            raise BundleIntegrityError(f"missing package payload: {name}")
        if not isinstance(metadata, dict):
            raise BundleIntegrityError(f"invalid metadata for package payload: {name}")
        expected_bytes = metadata.get("bytes")
        expected_sha = metadata.get("sha256")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise BundleIntegrityError(f"invalid byte count for package payload: {name}")
        if (
            not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or expected_sha.lower() != expected_sha
            or any(char not in "0123456789abcdef" for char in expected_sha)
        ):
            raise BundleIntegrityError(f"invalid SHA-256 for package payload: {name}")
        if payload.stat().st_size != expected_bytes:
            raise BundleIntegrityError(f"byte count mismatch for package payload: {name}")
        if _sha256_file(payload) != expected_sha:
            raise BundleIntegrityError(f"SHA-256 mismatch for package payload: {name}")
        listed_names.add(name)

        if payload.suffix.lower() in {".json", ".jsonl"}:
            try:
                payload_bytes = payload.read_bytes()
            except OSError as exc:
                raise BundleIntegrityError(f"cannot scan package payload: {name}") from exc
            if PRIVATE_PATH_PATTERN.search(payload_bytes.decode("utf-8")):
                raise BundleIntegrityError(f"private absolute path in package payload: {name}")

    actual_names = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
    }
    if actual_names != listed_names | {"manifest.json"}:
        extras = sorted(actual_names - listed_names - {"manifest.json"})
        missing = sorted(listed_names - actual_names)
        raise BundleIntegrityError(
            "package payload file set differs from manifest "
            f"(extras={extras!r}, missing={missing!r})"
        )


def _model_signature(vector_manifest: dict[str, Any]) -> tuple[Any, ...]:
    model = vector_manifest.get("model")
    runtime = vector_manifest.get("runtime")
    matrix = vector_manifest.get("matrix")
    if not isinstance(model, dict) or not isinstance(runtime, dict) or not isinstance(matrix, dict):
        raise BundleIntegrityError("vector manifest model/runtime/matrix contract is missing")

    if model.get("dimensions") != DIMENSIONS or model.get("dtype") != "<f4":
        raise BundleIntegrityError("vector model dimension/dtype contract is invalid")
    if matrix.get("dimensions") != DIMENSIONS or matrix.get("dtype") != "<f4":
        raise BundleIntegrityError("vector matrix dimension/dtype contract is invalid")
    if model.get("normalization") != "L2":
        raise BundleIntegrityError("vector model normalization must be L2")
    if not isinstance(model.get("pooling"), str) or not model["pooling"].strip():
        raise BundleIntegrityError("vector model pooling contract is missing")
    if runtime.get("implementation") != "direct-onnxruntime":
        raise BundleIntegrityError("vector runtime implementation is incompatible")
    if runtime.get("provider") != "CPUExecutionProvider":
        raise BundleIntegrityError("vector runtime provider is incompatible")
    if runtime.get("threads") != 4 or runtime.get("batch_size") != 32:
        raise BundleIntegrityError("vector runtime thread/batch contract is incompatible")
    if runtime.get("tokenizer_truncation") is not False:
        raise BundleIntegrityError("vector runtime permits tokenizer truncation")
    if runtime.get("pooling") != model.get("pooling"):
        raise BundleIntegrityError("vector model/runtime pooling contracts differ")
    if runtime.get("fastembed_used") is not False:
        raise BundleIntegrityError("vector runtime fastembed contract is incompatible")

    model_part = tuple((field, model.get(field)) for field in _MODEL_COMPATIBILITY_FIELDS)
    runtime_part = tuple((field, runtime.get(field)) for field in _RUNTIME_COMPATIBILITY_FIELDS)
    matrix_part = (matrix.get("dimensions"), matrix.get("dtype"))
    return model_part, runtime_part, matrix_part


def _validate_lock_metadata(vector_manifest: dict[str, Any], lock_path: Path) -> None:
    lock = _read_json(lock_path, description="model lock")
    model = vector_manifest.get("model")
    if not isinstance(model, dict):
        raise BundleIntegrityError("vector manifest model contract is missing")
    if model.get("name") != lock.get("model_version"):
        raise BundleIntegrityError("collection model name differs from lock")
    if model.get("revision") != lock.get("revision"):
        raise BundleIntegrityError("collection model revision differs from lock")
    if model.get("files") != lock.get("model_dir_sha256"):
        raise BundleIntegrityError("collection model hashes differ from lock")


@dataclass(frozen=True)
class _Package:
    bundle_id: str
    path: Path
    store: CuratedBundle
    matrix: np.ndarray
    model_signature: tuple[Any, ...]


@dataclass(frozen=True)
class _ChunkRef:
    bundle_id: str
    local_index: int


class CuratedCollection(CuratedBundle):
    """Read-only global view over the explicitly listed public bundles."""

    def __init__(
        self,
        collection: str | Path = DEFAULT_COLLECTION,
        *,
        knowledge_root: str | Path | None = None,
        lock_path: str | Path | None = None,
    ) -> None:
        self.collection_path = Path(collection).resolve()
        descriptor = _read_json(self.collection_path, description="collection descriptor")
        if descriptor.get("schema_version") != "curated-collection-candidate-1.0":
            raise BundleIntegrityError("collection descriptor schema is unsupported")
        if descriptor.get("canonical_project") != "chemical-engineering-rag-kg":
            raise BundleIntegrityError("collection descriptor project identity is invalid")
        if descriptor.get("collection_id") != "curated-two-bundles-v1":
            raise BundleIntegrityError("collection descriptor identity is invalid")
        if descriptor.get("knowledge_root") != ".":
            raise BundleIntegrityError("collection descriptor knowledge_root must be .")
        if descriptor.get("status") != "candidate_read_only":
            raise BundleIntegrityError("collection descriptor is not read-only candidate status")
        if descriptor.get("production_activated") is not False:
            raise BundleIntegrityError("collection descriptor has invalid production flag")
        if descriptor.get("no_answer_threshold") is not None:
            raise BundleIntegrityError("collection descriptor must not define an answer threshold")
        if descriptor.get("full_corpus_auto_merge") is not False:
            raise BundleIntegrityError("collection descriptor enables automatic corpus merge")
        packages = descriptor.get("packages")
        if not isinstance(packages, list) or not packages:
            raise BundleIntegrityError("collection descriptor has no package list")
        package_ids = [item.get("bundle_id") for item in packages if isinstance(item, dict)]
        if len(package_ids) != len(packages) or len(package_ids) != len(set(package_ids)):
            raise BundleIntegrityError("collection descriptor has duplicate or invalid bundle IDs")
        if set(package_ids) != _EXPECTED_PACKAGE_IDS:
            raise BundleIntegrityError("collection descriptor must list exactly the two public bundles")

        if knowledge_root is None:
            knowledge_root = self.collection_path.parent
        self.knowledge_root = Path(knowledge_root).resolve()
        if not self.knowledge_root.is_dir():
            raise BundleIntegrityError("knowledge root does not exist")

        lock = Path(lock_path).resolve() if lock_path is not None else None
        self.descriptor = descriptor
        self._packages: list[_Package] = []
        self.packages: dict[str, _Package] = {}
        self._model_signature: tuple[Any, ...] | None = None
        for entry in packages:
            package = self._load_package(entry, lock_path=lock)
            if package.bundle_id in self.packages:
                raise BundleIntegrityError(f"duplicate package loaded: {package.bundle_id}")
            if self._model_signature is None:
                self._model_signature = package.model_signature
            elif package.model_signature != self._model_signature:
                raise BundleIntegrityError("bundles have incompatible model/pooling/matrix contracts")
            self._packages.append(package)
            self.packages[package.bundle_id] = package

        expected_totals = descriptor.get("totals")
        if not isinstance(expected_totals, dict):
            raise BundleIntegrityError("collection descriptor totals are missing")
        actual_unit_count = sum(len(package.store.knowledge_units) for package in self._packages)
        actual_chunk_count = sum(len(package.store.chunks) for package in self._packages)
        if expected_totals.get("knowledge_units") != actual_unit_count:
            raise BundleIntegrityError("collection KU total mismatch")
        if expected_totals.get("chunks") != actual_chunk_count:
            raise BundleIntegrityError("collection chunk total mismatch")

        # Validate cross-package identity before initializing the inherited
        # single-bundle view.  A KU may legitimately have several chunks in
        # one package; only package-level KU overlap is a collection conflict.
        preflight_units: set[str] = set()
        preflight_chunks: set[str] = set()
        for package in self._packages:
            package_units = package.store.knowledge_units
            overlap = preflight_units.intersection(package_units)
            if overlap:
                raise BundleIntegrityError(
                    f"duplicate KU ID across packages: {sorted(overlap)[0]}"
                )
            preflight_units.update(package_units)
            for chunk in package.store.chunks:
                chunk_id = chunk.get("chunk_id")
                unit_id = chunk.get("knowledge_unit_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise BundleIntegrityError("merged chunk has invalid chunk_id")
                if not isinstance(unit_id, str) or not unit_id:
                    raise BundleIntegrityError("merged chunk has invalid knowledge_unit_id")
                if chunk_id in preflight_chunks:
                    raise BundleIntegrityError(
                        f"duplicate chunk ID across packages: {chunk_id}"
                    )
                if unit_id not in package_units:
                    raise BundleIntegrityError(
                        f"merged chunk references unknown KU: {unit_id}"
                    )
                preflight_chunks.add(chunk_id)

        # Initialize the inherited single-bundle contract once, then replace
        # its document view with the explicitly listed packages below.  This
        # keeps the established search, grouping, filtering, and result logic
        # as the collection's compatibility surface.
        super().__init__(self._packages[0].path)

        self.chunks: list[dict[str, Any]] = []
        self.knowledge_units: dict[str, dict[str, Any]] = {}
        self._chunk_refs: list[_ChunkRef] = []
        seen_chunks: set[str] = set()
        for package in self._packages:
            package_units = package.store.knowledge_units
            package_unit_ids = set(package_units)
            self.knowledge_units.update(package_units)
            for local_index, chunk in enumerate(package.store.chunks):
                chunk_id = chunk.get("chunk_id")
                unit_id = chunk.get("knowledge_unit_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise BundleIntegrityError("merged chunk has invalid chunk_id")
                if not isinstance(unit_id, str) or not unit_id:
                    raise BundleIntegrityError("merged chunk has invalid knowledge_unit_id")
                if chunk_id in seen_chunks:
                    raise BundleIntegrityError(f"duplicate chunk ID across packages: {chunk_id}")
                seen_chunks.add(chunk_id)
                if unit_id not in package_units:
                    raise BundleIntegrityError(
                        f"merged chunk references unknown KU: {unit_id}"
                    )
                self.chunks.append(chunk)
                self._chunk_refs.append(_ChunkRef(package.bundle_id, local_index))

        self.mapping = [
            mapping
            for package in self._packages
            for mapping in package.store.mapping
        ]
        self.normative_knowledge_units = dict(self.knowledge_units)

        matrices = [package.matrix for package in self._packages]
        self._matrix = np.concatenate(matrices, axis=0)
        self._matrix.setflags(write=False)
        if self._matrix.shape != (len(self.chunks), DIMENSIONS):
            raise BundleIntegrityError("merged matrix shape mismatch")
        self._doc_tokens = [
            _lexical_tokens(chunk["search_text"])
            for chunk in self.chunks
        ]
        self._sources = {chunk["source_id"] for chunk in self.chunks}
        self._subjects = {chunk["subject_root"] for chunk in self.chunks}
        if len(self.knowledge_units) != actual_unit_count or len(self.chunks) != actual_chunk_count:
            raise BundleIntegrityError("merged identity counts mismatch")

    def _load_package(
        self,
        entry: Any,
        *,
        lock_path: Path | None,
    ) -> _Package:
        if not isinstance(entry, dict):
            raise BundleIntegrityError("collection package entry must be an object")
        bundle_id = entry.get("bundle_id")
        if not isinstance(bundle_id, str) or not bundle_id:
            raise BundleIntegrityError("collection package entry has no bundle_id")
        package_dir = _safe_collection_path(
            self.knowledge_root,
            entry.get("path_relative_to_knowledge"),
            description=f"path for {bundle_id}",
        )
        if not package_dir.is_dir():
            raise BundleIntegrityError(f"bundle directory does not exist: {bundle_id}")
        manifest_path = package_dir / "manifest.json"
        vector_manifest_path = package_dir / "vector_manifest.json"
        if not manifest_path.is_file() or not vector_manifest_path.is_file():
            raise BundleIntegrityError(f"bundle manifests are missing: {bundle_id}")
        expected_manifest_sha = entry.get("manifest_sha256")
        expected_vector_sha = entry.get("vector_manifest_sha256")
        if _sha256_file(manifest_path) != expected_manifest_sha:
            raise BundleIntegrityError(f"bundle manifest hash mismatch: {bundle_id}")
        if _sha256_file(vector_manifest_path) != expected_vector_sha:
            raise BundleIntegrityError(f"bundle vector manifest hash mismatch: {bundle_id}")

        manifest = _read_json(manifest_path, description=f"{bundle_id} manifest")
        if manifest.get("bundle_id") != bundle_id:
            raise BundleIntegrityError(f"bundle identity mismatch in manifest: {bundle_id}")
        if manifest.get("canonical_project") != "chemical-engineering-rag-kg":
            raise BundleIntegrityError(f"bundle project identity mismatch: {bundle_id}")
        if manifest.get("production_activated") is not False:
            raise BundleIntegrityError(f"bundle production flag is invalid: {bundle_id}")
        _validate_manifest_files(package_dir, manifest)

        vector_manifest = _read_json(
            vector_manifest_path, description=f"{bundle_id} vector manifest"
        )
        if vector_manifest.get("production_activated") is not False:
            raise BundleIntegrityError(f"vector production flag is invalid: {bundle_id}")
        if vector_manifest.get("public_preview_is_production") is not False:
            raise BundleIntegrityError(f"vector preview flag is invalid: {bundle_id}")
        if vector_manifest.get("engineering_certification") is not False:
            raise BundleIntegrityError(f"vector certification flag is invalid: {bundle_id}")
        if vector_manifest.get("human_approved_anchor_count") != 0:
            raise BundleIntegrityError(f"vector human approval count is invalid: {bundle_id}")
        input_contract = vector_manifest.get("input")
        if isinstance(input_contract, dict) and input_contract.get("bundle_id") not in (None, bundle_id):
            raise BundleIntegrityError(f"vector input bundle identity mismatch: {bundle_id}")
        model_signature = _model_signature(vector_manifest)
        if lock_path is not None:
            _validate_lock_metadata(vector_manifest, lock_path)

        store = CuratedBundle(package_dir)
        matrix = store.load_matrix()
        if matrix.shape != (len(store.chunks), DIMENSIONS):
            raise BundleIntegrityError(f"bundle matrix shape mismatch: {bundle_id}")
        if not np.all(np.isfinite(matrix)):
            raise BundleIntegrityError(f"bundle matrix has non-finite values: {bundle_id}")
        if int(entry.get("knowledge_unit_count", -1)) != len(store.knowledge_units):
            raise BundleIntegrityError(f"bundle KU count mismatch: {bundle_id}")
        if int(entry.get("chunk_count", -1)) != len(store.chunks):
            raise BundleIntegrityError(f"bundle chunk count mismatch: {bundle_id}")
        return _Package(bundle_id, package_dir, store, matrix, model_signature)

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
            raise QueryInputError("dense/hybrid query requires --model-dir and --lock")
        encoder = CuratedOnnxEncoder(
            model_dir,
            lock_path=lock_path,
            vendor_path=vendor_path,
        )
        for package in self._packages:
            package.store._validate_dense_model(encoder, lock_path)
        query_values = np.asarray(encoder.encode_texts([query]), dtype="<f4")
        if query_values.shape != (1, DIMENSIONS):
            raise BundleIntegrityError(
                f"query encoder returned incompatible shape: {query_values.shape!r}"
            )
        if not np.all(np.isfinite(query_values)):
            raise BundleIntegrityError("query encoder returned non-finite values")
        norm = float(np.linalg.norm(query_values[0]))
        if not np.isfinite(norm) or norm <= 0.0:
            raise BundleIntegrityError("query encoder returned a zero-norm vector")
        query_vector = query_values[0]
        scored = [(index, float(self._matrix[index] @ query_vector)) for index in indices]
        return sorted(
            scored,
            key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]),
        )

    def _result(
        self,
        index: int,
        score: float,
        *,
        method: str,
        channel_score: float | None = None,
    ) -> dict[str, Any]:
        result = super()._result(
            index,
            score,
            method=method,
            channel_score=channel_score,
        )
        result["bundle_id"] = self._chunk_refs[index].bundle_id
        return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--knowledge-root", type=Path)
    parser.add_argument("--query")
    parser.add_argument("--node-id")
    parser.add_argument("--method", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--source", dest="source_filter")
    parser.add_argument("--subject", dest="subject_filter")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--vendor", type=Path)
    args = parser.parse_args()
    collection = CuratedCollection(
        args.collection,
        knowledge_root=args.knowledge_root,
        lock_path=args.lock,
    )
    results = collection.search(
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
                "collection_id": collection.descriptor["collection_id"],
                "bundle_ids": sorted(collection.packages),
                "method": args.method,
                "query": args.query,
                "node_id": args.node_id,
                "limit": args.limit,
                "source": args.source_filter,
                "subject": args.subject_filter,
                "rrf_k": RRF_K if args.method == "hybrid" else None,
                "no_answer_threshold": None,
                "candidate_scope": (
                    "Explicit public preview bundles only; not production retrieval "
                    "or engineering certification, with no automatic answer rejection."
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
