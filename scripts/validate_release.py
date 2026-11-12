#!/usr/bin/env python3
"""Validate the Python-only release without rerunning scientific analyses."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(path: str, errors: list[str]) -> Path | None:
    candidate = ROOT / path
    if not candidate.exists():
        errors.append(f"missing required path: {path}")
        return None
    return candidate


def check_split(path: str, expected: tuple[int, int, int], errors: list[str]) -> None:
    candidate = require(path, errors)
    if candidate is None:
        return
    data = load_json(candidate)
    actual = tuple(len(data.get(name, [])) for name in ("train", "validation", "test"))
    if actual != expected:
        errors.append(f"{path}: expected train/validation/test={expected}, found {actual}")


def check_manifest_paths(path: str, errors: list[str]) -> None:
    candidate = require(path, errors)
    if candidate is None:
        return
    data = load_json(candidate)
    paths: list[str] = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"path", "logical_path", "same_bytes_as"} and isinstance(item, str):
                    paths.append(item)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(data)
    for item in sorted(set(paths)):
        # Logical paths may point to compressed/uncompressed alternatives.
        if item.startswith("results/") or item.startswith("scripts/") or item.startswith("manifests/"):
            if not (ROOT / item).exists() and not (ROOT / f"{item}.gz").exists():
                errors.append(f"{path}: referenced file is absent: {item}")


def check_sha256_manifest(errors: list[str]) -> None:
    manifest = require("SHA256SUMS.txt", errors)
    if manifest is None:
        return
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"SHA256SUMS.txt: malformed line: {line}")
            continue
        expected, relative = parts
        target = ROOT / relative
        if not target.exists():
            errors.append(f"SHA256SUMS.txt: missing file: {relative}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        checked += 1
        if actual != expected:
            errors.append(f"SHA256SUMS.txt: hash mismatch: {relative}")
    print(f"checked {checked} SHA-256 entries")


def main() -> int:
    errors: list[str] = []
    required = [
        "README.md",
        "CITATION.cff",
        "RELEASE_NOTES.md",
        "docs/REPRODUCIBILITY.md",
        "docs/REVISED_MANUSCRIPT_MAP.md",
        "docs/RELEASE_SCOPE.md",
        "docs/VALIDATION_EXPERIMENTS.md",
        "manifests/SPLITS.json",
        "manifests/PAPER_TABLE_SOURCES.json",
        "manifests/RESULT_PROVENANCE.json",
        "manifests/REPRODUCIBILITY_INDEX.json",
        "results/validation/E6_fp32_regime_comparison_v1/analysis/E6_main_table.csv",
        "results/validation/E6_fp32_regime_comparison_v1/analysis/E6_primary_support.json",
    ]
    for path in required:
        require(path, errors)

    check_split("manifests/splits/E1_315_67_68.json", (315, 67, 68), errors)
    check_split("manifests/splits/E3_meaning_split.json", (315, 67, 68), errors)
    check_split("manifests/splits/E4_meaning_split.json", (315, 67, 68), errors)
    check_split("manifests/splits/E5_head_meaning_split.json", (315, 67, 68), errors)
    check_split("manifests/splits/E6_E7_309_67_68_support.json", (309, 67, 68), errors)

    support_path = ROOT / "results/validation/E6_fp32_regime_comparison_v1/analysis/E6_primary_support.json"
    if support_path.exists():
        support = load_json(support_path)
        actual = tuple(len(support.get(name, [])) for name in ("train", "validation", "test"))
        if actual != (309, 67, 68):
            errors.append(f"E6_primary_support.json: expected (309, 67, 68), found {actual}")
        if len(support.get("common_complete_meanings", [])) != 444:
            errors.append("E6_primary_support.json: expected 444 common complete meanings")

    for manifest in (
        "manifests/PAPER_TABLE_SOURCES.json",
        "manifests/RESULT_PROVENANCE.json",
        "manifests/REPRODUCIBILITY_INDEX.json",
    ):
        check_manifest_paths(manifest, errors)
    check_sha256_manifest(errors)

    if errors:
        print("RELEASE VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("RELEASE VALIDATION PASSED")
    print("No notebooks or license are required by this Python-only release scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
