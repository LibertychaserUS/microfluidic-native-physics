#!/usr/bin/env python3
"""Reject raw data, sealed labels, superseded runs, model binaries, secrets, private paths and identity drift."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from release_lib import IntegrityError, repository_files, strict_json_dumps


FORBIDDEN_NAMES = {
    "blind_features.csv",
    "development_labeled.csv",
    "sealed_blind_labels.csv",
    "sealed_primary_features.csv",
    "sealed_primary_sealed_labels.csv",
    "sealed_secondary_features.csv",
    "sealed_secondary_sealed_labels.csv",
    "sealed_features.csv",
    "sealed_labels.csv",
    "sealed_prediction_bundle.csv",
    "sealed_scored_rows.csv",
    "table_comprehensive.csv",
    "table_external.csv",
    "table_upstream932.csv",
}
FORBIDDEN_NAME_PATTERNS = (
    re.compile(r"^loops.*\.csv$"),   # per-loop tables stay in the workspace (manuscript Section 8)
    re.compile(r"^budget\.csv$"),
    re.compile(r"^worker_tasks_phase[0-9]+\.json$"),
)
FORBIDDEN_SUFFIXES = {".ckpt", ".joblib", ".mat", ".pkl", ".pt", ".pth", ".ubj", ".xls", ".xlsx"}
FORBIDDEN_DIRECTORY_PATTERNS = (
    re.compile(r"^expanded_artifacts_(?:fast|finalfast|run[0-9]+|auditfast)$"),
    re.compile(r"^v2_artifacts_smoke[0-9]*$"),
    re.compile(r"^w[345]_artifacts_fast_smoke$"),
    re.compile(r"^posthoc_analysis$"),
    re.compile(r"^blind_raw$"),
    re.compile(r"^fidelity_run[0-9]+_attempt[0-9]+$"),
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile("/" + r"Users/[^/\s]+/"),
    re.compile("/" + r"home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\" + r"Users\\[^\\\s]+\\"),
)


def validate_repository(root: Path) -> dict[str, object]:
    root = root.resolve()
    problems: list[dict[str, str]] = []
    files = repository_files(root)
    for path in files:
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES or any(p.match(path.name) for p in FORBIDDEN_NAME_PATTERNS):
            problems.append({"path": relative.as_posix(), "problem": "forbidden_source_table_or_workspace_only_file"})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append({"path": relative.as_posix(), "problem": "forbidden_raw_or_model_format"})
        for part in relative.parts[:-1]:
            if any(pattern.match(part) for pattern in FORBIDDEN_DIRECTORY_PATTERNS):
                problems.append({"path": relative.as_posix(), "problem": "superseded_or_smoke_directory"})
                break
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            problems.append({"path": relative.as_posix(), "problem": "possible_secret"})
        if any(pattern.search(text) for pattern in PRIVATE_PATH_PATTERNS):
            problems.append({"path": relative.as_posix(), "problem": "private_absolute_path"})

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    # The identity check applies to the top-level ``authors`` block; the
    # ``preferred-citation`` reference repeats the same single author.
    top_level, _, preferred = citation.partition("preferred-citation:")
    for block in (top_level, preferred):
        if not block.strip():
            continue
        if block.count("family-names:") != 1 or "family-names: Zhang" not in block:
            problems.append({"path": "CITATION.cff", "problem": "author_family_name_not_exactly_one_Zhang"})
        if block.count("given-names:") != 1 or "given-names: Zhiyuan" not in block:
            problems.append({"path": "CITATION.cff", "problem": "author_given_name_not_exactly_one_Zhiyuan"})
    notice = (root / "NOTICE").read_text(encoding="utf-8")
    if "Copyright 2026 Zhiyuan Zhang" not in notice:
        problems.append({"path": "NOTICE", "problem": "copyright_identity_mismatch"})
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "## Release status" not in readme or "documentation and release metadata only" in readme:
        problems.append({"path": "README.md", "problem": "release_status_section_missing_or_stale"})
    return {
        "files_checked": len(files),
        "problems": problems,
        "public_author": "Zhiyuan Zhang",
        "schema_version": "repository-publication-policy-v1",
        "valid": not problems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = validate_repository(args.root)
    except (IntegrityError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    print(strict_json_dumps(result), end="")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
