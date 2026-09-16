"""Focused descriptor checks for the publication candidate validator."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


repo_override = os.environ.get("CURATED_REPO_ROOT")
REPO_ROOT = (
    Path(repo_override).resolve()
    if repo_override
    else Path(__file__).resolve().parents[1]
)
SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from validate_public_release import check_collection_descriptor


class CollectionDescriptorValidatorTests(unittest.TestCase):
    def test_real_descriptor_passes(self) -> None:
        self.assertEqual(check_collection_descriptor(REPO_ROOT), [])

    def test_descriptor_absence_keeps_legacy_noop(self) -> None:
        with TemporaryDirectory() as temporary:
            self.assertEqual(check_collection_descriptor(Path(temporary)), [])

    def test_descriptor_directory_is_not_treated_as_absence(self) -> None:
        with TemporaryDirectory() as temporary:
            descriptor_path = Path(temporary) / "knowledge" / "curated-collection-v1.json"
            descriptor_path.mkdir(parents=True)
            failures = check_collection_descriptor(Path(temporary))
            self.assertIn(
                "collection descriptor path exists but is not a file",
                failures,
            )

    def _fixture_root(self, descriptor: dict) -> tuple[TemporaryDirectory, Path]:
        temporary = TemporaryDirectory()
        root = Path(temporary.name)
        knowledge = root / "knowledge"
        knowledge.mkdir()
        source_knowledge = REPO_ROOT / "knowledge"
        (root / "src").mkdir()
        shutil.copy2(REPO_ROOT / "src" / "query_collection.py", root / "src" / "query_collection.py")
        (knowledge / "curated-collection-v1.json").write_text(
            json.dumps(descriptor), encoding="utf-8"
        )
        for bundle_id in ("curated-four-books-v1", "curated-s001-upper-v1"):
            package = knowledge / bundle_id
            package.mkdir()
            source_package = source_knowledge / bundle_id
            for name in ("manifest.json", "vector_manifest.json"):
                shutil.copy2(source_package / name, package / name)
        return temporary, root

    def test_wrong_manifest_sha_is_rejected(self) -> None:
        descriptor = json.loads(
            (REPO_ROOT / "knowledge" / "curated-collection-v1.json").read_text(
                encoding="utf-8"
            )
        )
        descriptor["packages"][0]["manifest_sha256"] = "0" * 64
        temporary, root = self._fixture_root(descriptor)
        try:
            failures = check_collection_descriptor(root)
            self.assertTrue(any("manifest_sha256 mismatch" in item for item in failures))
        finally:
            temporary.cleanup()

    def test_duplicate_package_is_rejected(self) -> None:
        descriptor = json.loads(
            (REPO_ROOT / "knowledge" / "curated-collection-v1.json").read_text(
                encoding="utf-8"
            )
        )
        descriptor["packages"][1] = copy.deepcopy(descriptor["packages"][0])
        temporary, root = self._fixture_root(descriptor)
        try:
            failures = check_collection_descriptor(root)
            self.assertTrue(any("duplicate package ids" in item for item in failures))
        finally:
            temporary.cleanup()

    def test_wrong_package_path_is_rejected(self) -> None:
        descriptor = json.loads(
            (REPO_ROOT / "knowledge" / "curated-collection-v1.json").read_text(
                encoding="utf-8"
            )
        )
        descriptor["packages"][0]["path_relative_to_knowledge"] = "other-bundle"
        temporary, root = self._fixture_root(descriptor)
        try:
            failures = check_collection_descriptor(root)
            self.assertTrue(any("package path is not its bundle id" in item for item in failures))
        finally:
            temporary.cleanup()

    def test_wrong_total_is_rejected(self) -> None:
        descriptor = json.loads(
            (REPO_ROOT / "knowledge" / "curated-collection-v1.json").read_text(
                encoding="utf-8"
            )
        )
        descriptor["totals"]["chunks"] = 112
        temporary, root = self._fixture_root(descriptor)
        try:
            failures = check_collection_descriptor(root)
            self.assertTrue(any("total chunks must be 113" in item for item in failures))
        finally:
            temporary.cleanup()

    def test_unknown_id_is_rejected_without_reading_external_path(self) -> None:
        descriptor = json.loads(
            (REPO_ROOT / "knowledge" / "curated-collection-v1.json").read_text(
                encoding="utf-8"
            )
        )
        descriptor["packages"][0]["bundle_id"] = "../external"
        descriptor["packages"][0]["path_relative_to_knowledge"] = "../external"
        temporary, root = self._fixture_root(descriptor)
        try:
            external = root / "external"
            external.mkdir()
            (external / "manifest.json").write_text("external", encoding="utf-8")
            (external / "vector_manifest.json").write_text("external", encoding="utf-8")
            original_read_bytes = Path.read_bytes

            def guarded_read_bytes(path: Path) -> bytes:
                if path.resolve().is_relative_to(external.resolve()):
                    raise AssertionError("unknown package path was read")
                return original_read_bytes(path)

            with patch.object(Path, "read_bytes", guarded_read_bytes):
                failures = check_collection_descriptor(root)
            self.assertTrue(any("package ids" in item for item in failures))
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
