from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/mvp/train.jsonl"
HOLDOUT_PATH = ROOT / "data/mvp/holdout.jsonl"
SUMMARY_PATH = ROOT / "data/mvp/manifest_summary.json"
EXPECTED_TAXONOMIES = {"document_field_bind", "chart_table_cell"}
MINIMUM_PER_SPLIT = 64


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing manifest: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail(f"{path.relative_to(ROOT)}:{line_number} invalid JSONL: {exc}")
    if not rows:
        fail(f"empty manifest: {path.relative_to(ROOT)}")
    return rows


def check_row(row: dict[str, Any], expected_split: str) -> None:
    required_keys = [
        "curriculum_id",
        "sample_id",
        "split",
        "taxonomy_id",
        "taxonomy",
        "prompt",
        "expected_answers",
        "expected_answer_key",
        "hard_negatives",
        "full_image_path",
        "teacher_label_is_candidate",
        "source",
    ]
    for key in required_keys:
        if key not in row:
            fail(f"{row.get('sample_id', '<unknown>')} missing key: {key}")
    if row["split"] != expected_split:
        fail(f"{row['sample_id']} split mismatch: {row['split']} != {expected_split}")
    if row["taxonomy_id"] not in EXPECTED_TAXONOMIES:
        fail(f"{row['sample_id']} unexpected taxonomy_id: {row['taxonomy_id']}")
    if not row["expected_answers"]:
        fail(f"{row['sample_id']} has no expected_answers")
    if len(row["hard_negatives"]) < 3:
        fail(f"{row['sample_id']} needs at least 3 hard_negatives")
    if row["expected_answer_key"] in {str(item).strip().lower() for item in row["hard_negatives"]}:
        fail(f"{row['sample_id']} has expected answer in hard_negatives")
    if row["teacher_label_is_candidate"] is not True:
        fail(f"{row['sample_id']} teacher_label_is_candidate must be true")
    if not str(row["full_image_path"]).startswith("hf://datasets/"):
        fail(f"{row['sample_id']} full_image_path must be an hf://datasets reference")
    source = row["source"]
    for key in ["dataset", "config", "split", "row_idx", "image_field"]:
        if key not in source:
            fail(f"{row['sample_id']} source missing key: {key}")


def main() -> None:
    train_rows = load_jsonl(TRAIN_PATH)
    holdout_rows = load_jsonl(HOLDOUT_PATH)
    for row in train_rows:
        check_row(row, "train")
    for row in holdout_rows:
        check_row(row, "holdout")

    train_counts = Counter(row["taxonomy_id"] for row in train_rows)
    holdout_counts = Counter(row["taxonomy_id"] for row in holdout_rows)
    for taxonomy_id in EXPECTED_TAXONOMIES:
        if train_counts[taxonomy_id] != MINIMUM_PER_SPLIT:
            fail(f"{taxonomy_id} train count is {train_counts[taxonomy_id]}, expected 64")
        if holdout_counts[taxonomy_id] != MINIMUM_PER_SPLIT:
            fail(f"{taxonomy_id} holdout count is {holdout_counts[taxonomy_id]}, expected 64")

    train_answer_keys: dict[str, set[str]] = defaultdict(set)
    holdout_answer_keys: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        train_answer_keys[row["taxonomy_id"]].add(row["expected_answer_key"])
    for row in holdout_rows:
        holdout_answer_keys[row["taxonomy_id"]].add(row["expected_answer_key"])
    for taxonomy_id in EXPECTED_TAXONOMIES:
        overlap = train_answer_keys[taxonomy_id] & holdout_answer_keys[taxonomy_id]
        if overlap:
            fail(f"{taxonomy_id} train/holdout answer overlap: {sorted(overlap)[:5]}")

    if not SUMMARY_PATH.is_file():
        fail("missing data/mvp/manifest_summary.json")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    for taxonomy_id in EXPECTED_TAXONOMIES:
        info = summary["taxonomies"].get(taxonomy_id)
        if not info:
            fail(f"summary missing taxonomy: {taxonomy_id}")
        if info["train_count"] != 64 or info["holdout_count"] != 64:
            fail(f"summary count mismatch for {taxonomy_id}")
        if info["train_holdout_answer_overlap"]:
            fail(f"summary reports overlap for {taxonomy_id}")

    print("M2 manifest checks passed.")


if __name__ == "__main__":
    main()
