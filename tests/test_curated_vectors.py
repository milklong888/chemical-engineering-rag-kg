"""Bounded tests and synthetic learned-vector smoke check for curated_vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(MODULE_DIR))

from curated_vectors import (  # noqa: E402
    CuratedOnnxEncoder,
    InputTooLongError,
    ModelContractError,
    RecordValidationError,
    split_record,
)


CHECK_OUTPUT = Path("vector_module_check.json")

STATE: dict[str, object] = {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record(
    body: str,
    *,
    title: str = "合成传热知识",
    applicability: list[str] | None = None,
    content_available: bool = True,
    retrieval_eligible: bool = True,
    embedding_eligible: bool = True,
) -> dict[str, object]:
    return {
        "node_id": "SYNTH-001",
        "title": title,
        "text": body,
        "text_sha256": _sha256_text(body),
        "applicability": applicability or ["仅用于合成测试，不是项目参数"],
        "content_available": content_available,
        "retrieval_eligible": retrieval_eligible,
        "embedding_eligible": embedding_eligible,
    }


class CuratedVectorTests(unittest.TestCase):
    model_dir: Path
    lock_path: Path
    vendor_path: Path | None

    @classmethod
    def setUpClass(cls) -> None:
        cls.encoder = CuratedOnnxEncoder(
            cls.model_dir,
            lock_path=cls.lock_path,
            vendor_path=cls.vendor_path,
        )
        cls.tokenizer = cls.encoder.tokenizer

    def test_split_coverage_and_lossless_segments(self) -> None:
        long_sentence = (
            "在固定物性和边界条件下，换热量取决于有效面积与实际温差推动；"
            "流型改变时应重新核对修正因子，并检查热阻控制侧是否发生变化。"
        )
        body = ("短句。" + long_sentence + "短句！") * 100
        record = _record(body)
        segments = split_record(record, self.tokenizer)

        self.assertGreater(len(segments), 1)
        self.assertEqual(segments[0]["body_start"], 0)
        self.assertEqual(segments[-1]["body_end"], len(body))
        for previous, current in zip(segments, segments[1:]):
            self.assertLess(previous["body_start"], current["body_start"])
            self.assertLessEqual(current["body_start"], previous["body_end"])
            self.assertLessEqual(
                previous["body_end"] - current["body_start"], 60
            )
        for segment in segments:
            start = int(segment["body_start"])
            end = int(segment["body_end"])
            self.assertEqual(segment["text"], body[start:end])
            self.assertEqual(segment["full_text"], body)
            self.assertEqual(segment["text_sha256"], record["text_sha256"])
            expected = (
                record["title"]
                + "\n"
                + body[start:end]
                + "\n"
                + "\n".join(record["applicability"])
            )
            self.assertEqual(segment["embedding_text"], expected)
            self.assertLessEqual(int(segment["token_count"]), 512)
        no_overlap = split_record(record, self.tokenizer, max_overlap_chars=0)
        for previous, current in zip(no_overlap, no_overlap[1:]):
            self.assertEqual(previous["body_end"], current["body_start"])
        STATE["split"] = {
            "segments": len(segments),
            "body_length": len(body),
            "coverage": True,
            "max_token_count": max(
                int(segment["token_count"]) for segment in segments
            ),
            "max_overlap_chars": max(
                (
                    previous["body_end"] - current["body_start"]
                    for previous, current in zip(segments, segments[1:])
                ),
                default=0,
            ),
        }

    def test_overlong_input_is_rejected_without_truncation(self) -> None:
        body = "长文本字符不能被静默截断。" * 500
        text = _record(body)
        embedding_text = (
            text["title"] + "\n" + text["text"] + "\n" + text["applicability"][0]
        )
        token_count = len(
            self.tokenizer.encode(embedding_text, add_special_tokens=True).ids
        )
        self.assertGreater(token_count, 512)
        with self.assertRaises(InputTooLongError):
            self.encoder.encode_texts([embedding_text])
        STATE["overlong_rejection"] = {
            "token_count": token_count,
            "rejected": True,
            "truncated": False,
        }

    def test_invalid_records_fail_closed(self) -> None:
        with self.assertRaises(RecordValidationError):
            split_record(_record(""), self.tokenizer)
        with self.assertRaises(RecordValidationError):
            split_record(
                _record("正文", retrieval_eligible=False), self.tokenizer
            )
        with self.assertRaises(RecordValidationError):
            split_record(
                _record("正文", embedding_eligible=False), self.tokenizer
            )
        string_condition = _record("正文")
        string_condition["applicability"] = "单字符串适用条件"
        string_segments = split_record(string_condition, self.tokenizer)
        self.assertEqual(
            string_segments[0]["applicability"], ["单字符串适用条件"]
        )
        bad_hash = _record("正文")
        bad_hash["text_sha256"] = "0" * 64
        with self.assertRaises(RecordValidationError):
            split_record(bad_hash, self.tokenizer)
        huge_condition = _record(
            "正文", applicability=["条件" * 4000]
        )
        with self.assertRaises(InputTooLongError):
            split_record(huge_condition, self.tokenizer)
        STATE["invalid_rejection"] = {
            "empty_body": True,
            "ineligible_flag": True,
            "hash_mismatch": True,
            "embedding_ineligible": True,
            "overlong_condition": True,
        }

    def test_pooling_contract_rank3_and_rank2(self) -> None:
        rank3 = np.zeros((1, 2, 512), dtype=np.float32)
        rank3[0, 0, 0] = 3.0
        rank3[0, 1, 1] = 7.0
        pooled3 = self.encoder._pool_and_normalize(rank3)
        self.assertEqual(self.encoder.last_pooling_mode, "cls_rank3_first_token")
        self.assertEqual(pooled3.shape, (1, 512))
        self.assertAlmostEqual(float(pooled3[0, 0]), 1.0, places=6)

        rank2 = np.zeros((1, 512), dtype=np.float32)
        rank2[0, 1] = 4.0
        pooled2 = self.encoder._pool_and_normalize(rank2)
        self.assertEqual(self.encoder.last_pooling_mode, "rank2_passthrough")
        self.assertEqual(pooled2.shape, (1, 512))
        self.assertAlmostEqual(float(pooled2[0, 1]), 1.0, places=6)
        STATE["pooling"] = {
            "rank3": "first-token CLS",
            "rank2": "passthrough",
        }
        with self.assertRaises(ModelContractError):
            self.encoder._pool_and_normalize(rank2, expected_rows=2)
        nonfinite = rank2.copy()
        nonfinite[0, 1] = np.nan
        with self.assertRaises(ModelContractError):
            self.encoder._pool_and_normalize(nonfinite, expected_rows=1)

    def test_actual_embedding_shape_norm_repeat_and_reopen(self) -> None:
        texts = [
            "传热系数必须与所采用的面积基准和温差定义保持一致。",
            "阀门节流与透平膨胀的能量约束不能混用。",
            "相变换热的适用边界需要结合局部工况核查。",
        ]
        first = self.encoder.encode_texts(texts)
        repeated = self.encoder.encode_texts(texts)
        reopened = CuratedOnnxEncoder(
            self.model_dir,
            lock_path=self.lock_path,
            vendor_path=self.vendor_path,
        ).encode_texts(texts)
        self.assertEqual(first.shape, (3, 512))
        self.assertEqual(first.dtype, np.float32)
        norms = np.linalg.norm(first, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))
        self.assertTrue(np.array_equal(first, repeated))
        self.assertTrue(np.array_equal(first, reopened))
        STATE["smoke"] = {
            "input_count": len(texts),
            "shape": list(first.shape),
            "dtype": str(first.dtype),
            "norms": [float(value) for value in norms],
            "repeat_exact": True,
            "reopen_exact": True,
            "max_repeat_abs_diff": float(np.max(np.abs(first - repeated))),
            "max_reopen_abs_diff": float(np.max(np.abs(first - reopened))),
        }
        STATE["actual_pooling_mode"] = self.encoder.last_pooling_mode


def _module_sha256() -> str:
    digest = hashlib.sha256()
    with (MODULE_DIR / "curated_vectors.py").open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_tests(
    output: Path,
    model_dir: Path,
    lock_path: Path,
    vendor_path: Path | None,
) -> int:
    CuratedVectorTests.model_dir = model_dir
    CuratedVectorTests.lock_path = lock_path
    CuratedVectorTests.vendor_path = vendor_path
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CuratedVectorTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    encoder = getattr(CuratedVectorTests, "encoder", None)
    report: dict[str, object] = {
        "schema": "curated-vector-module-check-v1",
        "status": "pass" if result.wasSuccessful() else "fail",
        "module_sha256": _module_sha256(),
        "test_count": result.testsRun,
        "failures": [str(item[1]) for item in result.failures],
        "errors": [str(item[1]) for item in result.errors],
        "checks": dict(STATE),
    }
    if encoder is not None:
        contract = encoder.contract_report()
        contract["last_pooling_mode"] = STATE.get("actual_pooling_mode")
        contract["observed_actual_pooling_mode"] = STATE.get(
            "actual_pooling_mode"
        )
        contract["synthetic_pooling_modes"] = STATE.get("pooling")
        report["runtime_contract"] = contract
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--vendor", type=Path)
    parser.add_argument("--output", type=Path, default=CHECK_OUTPUT)
    args = parser.parse_args()
    model_dir = args.model_dir or os.environ.get("CURATED_VECTOR_MODEL_DIR")
    lock_path = args.lock or os.environ.get("CURATED_VECTOR_LOCK_PATH")
    vendor_path = args.vendor or os.environ.get("CURATED_VECTOR_VENDOR")
    if not model_dir or not lock_path:
        parser.error(
            "--model-dir/--lock or CURATED_VECTOR_MODEL_DIR/"
            "CURATED_VECTOR_LOCK_PATH are required"
        )
    return run_tests(
        args.output,
        Path(model_dir),
        Path(lock_path),
        Path(vendor_path) if vendor_path else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
