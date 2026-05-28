from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

BUCKET_NAMES = [
    "base_correct_lora_wrong",
    "base_wrong_lora_correct",
    "both_wrong",
    "both_correct",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON file: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_sample_id(result: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = result.get("rows")
    if not isinstance(rows, list):
        fail(f"{label} result has no rows list")

    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        sample_id = row.get("sample_id")
        if not sample_id:
            fail(f"{label} result contains row without sample_id")
        if sample_id in indexed:
            duplicates.append(sample_id)
        indexed[sample_id] = row
    if duplicates:
        fail(f"{label} result has duplicate sample_id values: {duplicates[:5]}")
    return indexed


def binary_score(row: dict[str, Any], label: str, allow_errors: bool) -> int | None:
    if row.get("error"):
        if allow_errors:
            return None
        fail(f"{label} row {row.get('sample_id')} has runtime error: {row.get('error')}")
    score = row.get("score")
    if score in {0, 0.0}:
        return 0
    if score in {1, 1.0}:
        return 1
    if score is None and allow_errors:
        return None
    fail(f"{label} row {row.get('sample_id')} has non-binary score: {score!r}")
    return None


def bucket_name(base_score: int, lora_score: int) -> str:
    if base_score == 1 and lora_score == 0:
        return "base_correct_lora_wrong"
    if base_score == 0 and lora_score == 1:
        return "base_wrong_lora_correct"
    if base_score == 0 and lora_score == 0:
        return "both_wrong"
    return "both_correct"


def compact_row(base_row: dict[str, Any], lora_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": base_row["sample_id"],
        "split": base_row.get("split"),
        "taxonomy_id": base_row.get("taxonomy_id"),
        "prompt": base_row.get("prompt"),
        "expected_answers": base_row.get("expected_answers"),
        "full_image_path": base_row.get("full_image_path"),
        "source": base_row.get("source"),
        "base": {
            "answer_text": base_row.get("answer_text"),
            "score": base_row.get("score"),
            "score_method": base_row.get("score_method"),
            "latency_ms": base_row.get("latency_ms"),
        },
        "lora": {
            "answer_text": lora_row.get("answer_text"),
            "score": lora_row.get("score"),
            "score_method": lora_row.get("score_method"),
            "latency_ms": lora_row.get("latency_ms"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-result", default="results/m3/qwen3_vl_4b_base_holdout.json")
    parser.add_argument("--lora-result", default="results/m5/chart_table_cell_r4_v1_correct_holdout.json")
    parser.add_argument("--output", default="results/m5/chart_table_cell_r4_v1_holdout_delta.json")
    parser.add_argument("--taxonomy-id", default=None)
    parser.add_argument("--allow-errors", action="store_true")
    args = parser.parse_args()

    base_path = ROOT / args.base_result
    lora_path = ROOT / args.lora_result
    output_path = ROOT / args.output

    base_result = read_json(base_path)
    lora_result = read_json(lora_path)
    base_rows = rows_by_sample_id(base_result, "base")
    lora_rows = rows_by_sample_id(lora_result, "lora")
    common_ids = sorted(set(base_rows) & set(lora_rows))
    if not common_ids:
        fail("base and LoRA results have no common sample_id values")

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in BUCKET_NAMES}
    skipped_ids: list[str] = []
    for sample_id in common_ids:
        base_row = base_rows[sample_id]
        lora_row = lora_rows[sample_id]
        taxonomy_id = base_row.get("taxonomy_id")
        if args.taxonomy_id and taxonomy_id != args.taxonomy_id:
            continue
        if lora_row.get("taxonomy_id") != taxonomy_id:
            fail(f"taxonomy mismatch for {sample_id}: {taxonomy_id} != {lora_row.get('taxonomy_id')}")

        base_score = binary_score(base_row, "base", args.allow_errors)
        lora_score = binary_score(lora_row, "lora", args.allow_errors)
        if base_score is None or lora_score is None:
            skipped_ids.append(sample_id)
            continue
        buckets[bucket_name(base_score, lora_score)].append(compact_row(base_row, lora_row))

    total_compared = sum(len(rows) for rows in buckets.values())
    if total_compared == 0:
        fail("no comparable rows remained after filtering")

    base_correct = len(buckets["base_correct_lora_wrong"]) + len(buckets["both_correct"])
    lora_correct = len(buckets["base_wrong_lora_correct"]) + len(buckets["both_correct"])
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "base_result": args.base_result,
            "lora_result": args.lora_result,
            "taxonomy_id": args.taxonomy_id,
        },
        "summary": {
            "common_rows": len(common_ids),
            "compared_rows": total_compared,
            "skipped_rows": len(skipped_ids),
            "base_correct": base_correct,
            "lora_correct": lora_correct,
            "lora_minus_base_correct": lora_correct - base_correct,
            "counts": {name: len(rows) for name, rows in buckets.items()},
        },
        "skipped_sample_ids": skipped_ids,
        "buckets": buckets,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
