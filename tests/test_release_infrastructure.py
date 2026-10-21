from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare_runs  # noqa: E402
import fetch_public_data  # noqa: E402
import release_lib  # noqa: E402
import release_manifest  # noqa: E402
import validate_repository_policy  # noqa: E402
import validate_requirements_lock  # noqa: E402


class ReleaseInfrastructureTests(unittest.TestCase):
    def test_strict_json_rejects_nonstandard_constants_and_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text('{"missing": null, "value": 1.5}\n', encoding="utf-8")
            self.assertEqual(release_lib.strict_json_load(valid)["value"], 1.5)

            invalid_nan = root / "nan.json"
            invalid_nan.write_text('{"value": NaN}\n', encoding="utf-8")
            with self.assertRaises(release_lib.IntegrityError):
                release_lib.strict_json_load(invalid_nan)

            duplicate = root / "duplicate.json"
            duplicate.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
            with self.assertRaises(release_lib.IntegrityError):
                release_lib.strict_json_load(duplicate)

            with self.assertRaises(release_lib.IntegrityError):
                release_lib.strict_json_dumps({"value": float("inf")})

    def test_manifest_detects_tampering_and_unlisted_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tree"
            root.mkdir()
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            manifest_path = Path(directory) / "manifest.json"
            payload = release_manifest.build_manifest(root)
            release_lib.strict_json_write(manifest_path, payload)
            self.assertTrue(
                release_manifest.verify_manifest(root, manifest_path)["valid"]
            )

            (root / "a.txt").write_text("changed\n", encoding="utf-8")
            self.assertFalse(
                release_manifest.verify_manifest(root, manifest_path)["valid"]
            )

            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            (root / "new.txt").write_text("unlisted\n", encoding="utf-8")
            report = release_manifest.verify_manifest(root, manifest_path)
            self.assertFalse(report["valid"])
            self.assertEqual(report["unlisted"], ["new.txt"])

    def test_run_comparison_requires_exact_bytes_unless_exactly_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left"
            right = Path(directory) / "right"
            left.mkdir()
            right.mkdir()
            for root in (left, right):
                (root / "metrics.json").write_text('{"value": 1}\n', encoding="utf-8")
                (root / "environment.json").write_text('{"host": "x"}\n', encoding="utf-8")
            self.assertTrue(compare_runs.compare_runs(left, right, set())["valid"])

            (right / "environment.json").write_text('{"host": "y"}\n', encoding="utf-8")
            self.assertFalse(compare_runs.compare_runs(left, right, set())["valid"])
            self.assertTrue(
                compare_runs.compare_runs(left, right, {"environment.json"})["valid"]
            )

    def test_local_source_verification_uses_size_sha256_and_upstream_md5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            relative = Path("example/source.bin")
            content = b"public test bytes"
            target = destination / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(content)
            record = {
                "bytes": len(content),
                "md5": hashlib.md5(content, usedforsecurity=False).hexdigest(),
                "relative_path": relative.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": "https://example.invalid/source.bin",
            }
            registry = {
                "datasets": {
                    "example": {
                        "files": [record],
                        "status": "download_enabled",
                    }
                }
            }
            results = fetch_public_data.fetch_dataset(
                registry,
                "example",
                destination,
                verify_only=True,
                force=False,
            )
            self.assertEqual(results[0]["sha256"], record["sha256"])
            self.assertTrue(results[0]["verified"])

    def test_requirements_lock_requires_pin_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements-lock.txt"
            path.write_text(
                "example-package==1.2.3 \\\n    --hash=sha256:" + "a" * 64 + "\n",
                encoding="utf-8",
            )
            self.assertEqual(validate_requirements_lock.validate_lock(path), 1)
            path.write_text("example-package==1.2.3\n", encoding="utf-8")
            with self.assertRaises(release_lib.IntegrityError):
                validate_requirements_lock.validate_lock(path)

    def test_current_repository_is_policy_clean(self) -> None:
        report = validate_repository_policy.validate_repository(REPOSITORY_ROOT)
        self.assertTrue(report["valid"], report["problems"])
        self.assertEqual(report["public_author"], "Zhiyuan Zhang")


if __name__ == "__main__":
    unittest.main()
