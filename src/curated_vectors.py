"""Private learned-vector candidate for the curated KG payload.

This module deliberately uses the locked BGE ONNX graph directly.  It does
not import FastEmbed and it has no download, model substitution, truncation,
hash-vector, or pseudo-vector fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


MODEL_VERSION = "BAAI/bge-small-zh-v1.5"
MAX_TOKENS = 512
EMBEDDING_DIMENSION = 512
DEFAULT_THREADS = 4
DEFAULT_BATCH_SIZE = 32

LOCKED_MODEL_HASHES: dict[str, str] = {
    "config.json": "9088751d39abbf86ec3d19ffca92ad62ad19075f7e59712e6c71217fa125d1d3",
    "model_optimized.onnx": "1294ea4b6331115a353d81f96b85e8c8d7fdcc284453d5b2fab5b016230aad38",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
    "tokenizer_config.json": "e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a",
    "vocab.txt": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c",
}

EXPECTED_DEPENDENCIES = {
    "tokenizers": "0.23.1",
    "onnxruntime": "1.27.0",
    "numpy": "2.5.1",
}


class CuratedVectorError(RuntimeError):
    """Base error for a fail-closed candidate operation."""


class ModelContractError(CuratedVectorError):
    """Raised when the locked model/runtime contract is not met."""


class RecordValidationError(CuratedVectorError):
    """Raised when a KU cannot safely become a retrieval segment."""


class InputTooLongError(CuratedVectorError):
    """Raised instead of silently truncating a text above the model limit."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_unsafe_text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise RecordValidationError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise RecordValidationError(f"{field} must not be empty or whitespace")
    if "\x00" in value:
        raise RecordValidationError(f"{field} contains NUL")
    if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
        raise RecordValidationError(f"{field} contains an unpaired surrogate")
    return value


def verify_model_assets(
    model_dir: str | Path,
    lock_path: str | Path,
) -> dict[str, Any]:
    """Verify the six locked model files and the independent lock record."""

    directory = Path(model_dir).resolve()
    lock_file = Path(lock_path).resolve()
    if not directory.is_dir():
        raise ModelContractError(f"model directory does not exist: {directory}")
    if not lock_file.is_file():
        raise ModelContractError(f"model lock does not exist: {lock_file}")

    try:
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"cannot read model lock: {lock_file}") from exc
    if lock.get("model_version") != MODEL_VERSION:
        raise ModelContractError(
            f"locked model version mismatch: {lock.get('model_version')!r}"
        )
    locked_hashes = lock.get("model_dir_sha256")
    if not isinstance(locked_hashes, dict):
        raise ModelContractError("model lock has no model_dir_sha256 object")

    files: dict[str, dict[str, Any]] = {}
    for name, expected in LOCKED_MODEL_HASHES.items():
        path = directory / name
        if not path.is_file():
            raise ModelContractError(f"locked model file is missing: {path}")
        actual = _sha256(path)
        lock_hash = locked_hashes.get(name)
        if actual != expected or lock_hash != expected:
            raise ModelContractError(
                f"model hash mismatch for {name}: actual={actual}, "
                f"expected={expected}, lock={lock_hash}"
            )
        files[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
            "expected_sha256": expected,
            "lock_sha256": lock_hash,
            "match": True,
        }
    return {
        "model_version": MODEL_VERSION,
        "model_dir": str(directory),
        "lock_path": str(lock_file),
        "files": files,
        "all_six_match": True,
    }


def _load_dependencies(vendor_path: str | Path | None) -> tuple[Any, Any, Any, dict[str, str]]:
    if vendor_path is not None:
        vendor = Path(vendor_path).resolve()
        if not vendor.is_dir():
            raise ModelContractError(f"vendor directory does not exist: {vendor}")
        vendor_text = str(vendor)
        if vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)

    try:
        import numpy as np
        import onnxruntime as ort
        import tokenizers
    except ImportError as exc:
        raise ModelContractError(
            "locked runtime dependencies are unavailable; no fallback is allowed"
        ) from exc

    versions = {
        "tokenizers": str(tokenizers.__version__),
        "onnxruntime": str(ort.__version__),
        "numpy": str(np.__version__),
    }
    mismatches = {
        name: {"actual": versions[name], "expected": expected}
        for name, expected in EXPECTED_DEPENDENCIES.items()
        if versions[name] != expected
    }
    if mismatches:
        raise ModelContractError(
            "dependency version mismatch for locked runtime: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    return np, ort, tokenizers.Tokenizer, versions


def _disable_tokenizer_truncation(tokenizer: Any) -> None:
    no_truncation = getattr(tokenizer, "no_truncation", None)
    if no_truncation is None:
        raise ModelContractError("tokenizer does not expose no_truncation()")
    no_truncation()


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoding = tokenizer.encode(text, add_special_tokens=True)
    ids = getattr(encoding, "ids", None)
    if ids is None:
        raise ModelContractError("tokenizer encoding has no ids")
    return list(ids)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(_token_ids(tokenizer, text))


def _embedding_text(title: str, body: str, applicability: Sequence[str]) -> str:
    return title + "\n" + body + "\n" + "\n".join(applicability)


def _validate_record(record: dict[str, Any]) -> tuple[str, str, list[str], str]:
    if not isinstance(record, dict):
        raise RecordValidationError("record must be an object")
    for flag in (
        "content_available",
        "retrieval_eligible",
        "embedding_eligible",
    ):
        if record.get(flag) is not True:
            raise RecordValidationError(f"{flag} is not true")

    node_id = _reject_unsafe_text(record.get("node_id"), "node_id")
    title = _reject_unsafe_text(record.get("title"), "title")
    body = _reject_unsafe_text(record.get("text"), "text")
    applicability_value = record.get("applicability")
    if isinstance(applicability_value, str):
        applicability_value = [applicability_value]
    elif isinstance(applicability_value, (list, tuple)):
        applicability_value = list(applicability_value)
    else:
        raise RecordValidationError(
            "applicability must be a string or a non-empty string array"
        )
    if not applicability_value:
        raise RecordValidationError("applicability must be a non-empty array")
    applicability = [
        _reject_unsafe_text(item, f"applicability[{index}]")
        for index, item in enumerate(applicability_value)
    ]

    supplied_hash = record.get("text_sha256")
    if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
        raise RecordValidationError("text_sha256 must be a 64-character hex string")
    if any(char not in "0123456789abcdef" for char in supplied_hash):
        raise RecordValidationError("text_sha256 must be lowercase hexadecimal")
    actual_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if supplied_hash != actual_hash:
        raise RecordValidationError("text_sha256 does not match the full text")
    return node_id, title, applicability, body


def _max_fitting_end(
    *,
    body: str,
    start: int,
    title: str,
    applicability: Sequence[str],
    tokenizer: Any,
    max_tokens: int,
) -> int:
    """Find a fitting prefix; returning a shorter valid prefix is intentional.

    BPE token counts are normally monotone enough for binary search, but the
    first-character check and final validation make the operation fail closed
    even if a tokenizer has an unusual merge boundary.
    """

    length = len(body)
    first_end = start + 1
    first_count = _token_count(
        tokenizer, _embedding_text(title, body[start:first_end], applicability)
    )
    if first_count > max_tokens:
        fitting_end = None
        for candidate in range(first_end + 1, length + 1):
            count = _token_count(
                tokenizer,
                _embedding_text(title, body[start:candidate], applicability),
            )
            if count <= max_tokens:
                fitting_end = candidate
                break
        if fitting_end is None:
            raise InputTooLongError(
                "title/applicability leave no encodable body character under "
                f"{max_tokens} tokens"
            )
        low = fitting_end
    else:
        low = first_end

    best = low
    high = length + 1
    while low < high:
        middle = (low + high) // 2
        count = _token_count(
            tokenizer, _embedding_text(title, body[start:middle], applicability)
        )
        if count <= max_tokens:
            best = middle
            low = middle + 1
        else:
            high = middle

    final_count = _token_count(
        tokenizer, _embedding_text(title, body[start:best], applicability)
    )
    if final_count > max_tokens or best <= start:
        raise InputTooLongError(
            f"no valid segment begins at character {start} under {max_tokens} tokens"
        )
    return best


def _sentence_preferred_end(
    *,
    body: str,
    start: int,
    best_end: int,
    title: str,
    applicability: Sequence[str],
    tokenizer: Any,
    max_tokens: int,
) -> int:
    if best_end >= len(body):
        return best_end
    sentence_endings = set("。！？!?；;.!?\n")
    minimum_boundary_length = max(
        1, ((best_end - start) * 3 + 4) // 5
    )
    for index in range(best_end - 1, start, -1):
        if body[index] not in sentence_endings:
            continue
        candidate = index + 1
        if candidate <= start:
            continue
        if candidate - start < minimum_boundary_length:
            continue
        candidate_text = _embedding_text(
            title, body[start:candidate], applicability
        )
        if _token_count(tokenizer, candidate_text) <= max_tokens:
            return candidate
    return best_end


def _validate_coverage(segments: Sequence[dict[str, Any]], length: int) -> None:
    if not segments:
        raise RecordValidationError("segmentation produced no segments")
    if segments[0]["start"] != 0 or segments[-1]["end"] != length:
        raise RecordValidationError("segments do not cover the full body")
    previous_end = 0
    for segment in segments:
        start, end = int(segment["start"]), int(segment["end"])
        if start < 0 or end <= start or end > length:
            raise RecordValidationError("invalid segment interval")
        if start > previous_end:
            raise RecordValidationError("segment intervals have a gap")
        overlap = previous_end - start if previous_end else 0
        if overlap > 60:
            raise RecordValidationError("segment overlap exceeds 60 characters")
        previous_end = max(previous_end, end)


def split_record(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_tokens: int = MAX_TOKENS,
    max_overlap_chars: int = 60,
) -> list[dict[str, Any]]:
    """Split one qualified KU into lossless, token-bounded retrieval segments."""

    if max_tokens != MAX_TOKENS:
        raise ValueError(f"the locked model contract requires max_tokens={MAX_TOKENS}")
    if not 0 <= max_overlap_chars <= 60:
        raise ValueError("max_overlap_chars must be between 0 and 60")
    node_id, title, applicability, body = _validate_record(record)
    _disable_tokenizer_truncation(tokenizer)

    base = _embedding_text(title, "", applicability)
    if _token_count(tokenizer, base) > max_tokens:
        raise InputTooLongError(
            "title and applicability alone exceed the locked 512-token budget"
        )

    segments: list[dict[str, Any]] = []
    start = 0
    while start < len(body):
        fitting_end = _max_fitting_end(
            body=body,
            start=start,
            title=title,
            applicability=applicability,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        end = _sentence_preferred_end(
            body=body,
            start=start,
            best_end=fitting_end,
            title=title,
            applicability=applicability,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        )
        segment_text = body[start:end]
        embedding_text = _embedding_text(title, segment_text, applicability)
        token_count = _token_count(tokenizer, embedding_text)
        if token_count > max_tokens:
            raise InputTooLongError(
                f"segment {start}:{end} unexpectedly has {token_count} tokens"
            )
        segments.append(
            {
                "node_id": node_id,
                "title": title,
                "text": segment_text,
                "full_text": body,
                "text_sha256": record["text_sha256"],
                "applicability": list(applicability),
                "embedding_text": embedding_text,
                "start": start,
                "end": end,
                "body_start": start,
                "body_end": end,
                "source_text_length": len(body),
                "segment_index": len(segments),
                "token_count": token_count,
            }
        )
        if end == len(body):
            break
        if max_overlap_chars == 0 or end - start <= 1:
            next_start = end
        else:
            overlap = min(max_overlap_chars, end - start - 1)
            next_start = end - overlap
        start = next_start

    for index, segment in enumerate(segments):
        segment["segment_index"] = index
        segment["segment_count"] = len(segments)
    _validate_coverage(segments, len(body))
    return segments


class CuratedOnnxEncoder:
    """Direct CPU ONNX encoder for the locked 512-dimensional BGE model."""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        lock_path: str | Path,
        vendor_path: str | Path | None = None,
        threads: int = DEFAULT_THREADS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if threads != DEFAULT_THREADS:
            raise ValueError(f"the contract requires threads={DEFAULT_THREADS}")
        if batch_size != DEFAULT_BATCH_SIZE:
            raise ValueError(f"the contract requires batch_size={DEFAULT_BATCH_SIZE}")
        self.model_dir = Path(model_dir).resolve()
        self.lock_path = Path(lock_path).resolve()
        self.threads = threads
        self.batch_size = batch_size
        self.asset_report = verify_model_assets(self.model_dir, self.lock_path)
        self.np, self.ort, tokenizer_type, self.dependency_versions = (
            _load_dependencies(vendor_path)
        )
        self.tokenizer = tokenizer_type.from_file(
            str(self.model_dir / "tokenizer.json")
        )
        _disable_tokenizer_truncation(self.tokenizer)

        session_options = self.ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = self.ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = (
            self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.session = self.ort.InferenceSession(
            str(self.model_dir / "model_optimized.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        providers = list(self.session.get_providers())
        if providers != ["CPUExecutionProvider"]:
            raise ModelContractError(
                f"unexpected ONNX providers: {providers!r}; CPU only is required"
            )
        self._input_names = {item.name for item in self.session.get_inputs()}
        required_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        if self._input_names != required_inputs:
            raise ModelContractError(
                f"unexpected ONNX inputs: {sorted(self._input_names)!r}"
            )
        outputs = self.session.get_outputs()
        if len(outputs) != 1:
            raise ModelContractError(
                f"expected one last_hidden_state output, got {len(outputs)}"
            )
        self._output_name = outputs[0].name
        self.last_pooling_mode: str | None = None

    def contract_report(self) -> dict[str, Any]:
        return {
            "implementation": "direct-onnxruntime",
            "fastembed_used": False,
            "model_version": MODEL_VERSION,
            "model_dimension": EMBEDDING_DIMENSION,
            "model_assets": self.asset_report,
            "dependency_versions": dict(self.dependency_versions),
            "onnx_providers": list(self.session.get_providers()),
            "threads": self.threads,
            "batch_size": self.batch_size,
            "token_limit": MAX_TOKENS,
            "tokenizer_truncation": False,
            "pooling": "CLS: rank3 output uses first token; rank2 output passes through",
            "normalization": "L2 float32",
            "last_pooling_mode": self.last_pooling_mode,
        }

    def _pool_and_normalize(
        self, output: Any, *, expected_rows: int | None = None
    ) -> Any:
        values = self.np.asarray(output, dtype=self.np.float32)
        if not self.np.all(self.np.isfinite(values)):
            raise ModelContractError("ONNX output contains non-finite values")
        if values.ndim == 3:
            pooled = values[:, 0, :]
            self.last_pooling_mode = "cls_rank3_first_token"
        elif values.ndim == 2:
            pooled = values
            self.last_pooling_mode = "rank2_passthrough"
        else:
            raise ModelContractError(
                f"unexpected ONNX output rank {values.ndim}; expected rank 3 or 2"
            )
        if pooled.ndim != 2 or pooled.shape[1] != EMBEDDING_DIMENSION:
            raise ModelContractError(
                f"unexpected pooled shape {tuple(pooled.shape)}"
            )
        if expected_rows is not None and pooled.shape[0] != expected_rows:
            raise ModelContractError(
                f"ONNX returned {pooled.shape[0]} rows for {expected_rows} inputs"
            )
        norms = self.np.linalg.norm(pooled, axis=1, keepdims=True).astype(
            self.np.float32, copy=False
        )
        if self.np.any(norms <= self.np.float32(0.0)):
            raise ModelContractError("zero embedding norm")
        return self.np.asarray(pooled / norms, dtype=self.np.float32)

    def _encode_batch(self, texts: Sequence[str]) -> Any:
        encodings = [self.tokenizer.encode(text, add_special_tokens=True) for text in texts]
        lengths = [len(encoding.ids) for encoding in encodings]
        over_limit = [
            (index, count) for index, count in enumerate(lengths) if count > MAX_TOKENS
        ]
        if over_limit:
            raise InputTooLongError(
                "input exceeds 512 tokens and was not truncated: "
                + repr(over_limit)
            )
        max_length = max(lengths, default=0)
        if max_length <= 0:
            raise InputTooLongError("empty token batch")
        input_ids = self.np.zeros((len(encodings), max_length), dtype=self.np.int64)
        attention_mask = self.np.zeros_like(input_ids)
        token_type_ids = self.np.zeros_like(input_ids)
        for row, encoding in enumerate(encodings):
            length = len(encoding.ids)
            input_ids[row, :length] = encoding.ids
            attention_mask[row, :length] = encoding.attention_mask
            token_type_ids[row, :length] = encoding.type_ids
        output = self.session.run(
            [self._output_name],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )[0]
        return self._pool_and_normalize(
            output, expected_rows=len(encodings)
        )

    def encode_texts(self, texts: Sequence[str]) -> Any:
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        values = list(texts)
        if not values:
            return self.np.empty((0, EMBEDDING_DIMENSION), dtype=self.np.float32)
        for index, text in enumerate(values):
            _reject_unsafe_text(text, f"texts[{index}]")
        batches = []
        for start in range(0, len(values), self.batch_size):
            batches.append(self._encode_batch(values[start : start + self.batch_size]))
        return self.np.ascontiguousarray(self.np.concatenate(batches, axis=0))


def encode_texts(
    texts: Sequence[str],
    model_dir: str | Path,
    *,
    lock_path: str | Path,
    vendor_path: str | Path | None = None,
    threads: int = DEFAULT_THREADS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Any:
    """Encode complete, already-segmented texts with the locked ONNX model."""

    encoder = CuratedOnnxEncoder(
        model_dir,
        lock_path=lock_path,
        vendor_path=vendor_path,
        threads=threads,
        batch_size=batch_size,
    )
    return encoder.encode_texts(texts)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoder = CuratedOnnxEncoder(
        args.model_dir,
        lock_path=args.lock,
        vendor_path=args.vendor,
    )
    vectors = encoder.encode_texts(args.text)
    report = encoder.contract_report()
    report.update(
        {
            "input_count": len(args.text),
            "vector_shape": list(vectors.shape),
            "vector_dtype": str(vectors.dtype),
            "norms": [
                float(encoder.np.linalg.norm(vector)) for vector in vectors
            ],
        }
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
