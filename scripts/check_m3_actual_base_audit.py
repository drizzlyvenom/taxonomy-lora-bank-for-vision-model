from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = {
    "train": {
        "manifest": ROOT / "data/mvp/train.jsonl",
        "result": ROOT / "results/m3/qwen3_vl_4b_base_train.json",
    },
    "holdout": {
        "manifest": ROOT / "data/mvp/holdout.jsonl",
        "result": ROOT / "results/m3/qwen3_vl_4b_base_holdout.json",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fail(message: str) -> None:
    print(f"M3 check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    passed: dict[str, Any] = {}

    for split, paths in RESULTS.items():
        manifest_path = paths["manifest"]
        result_path = paths["result"]
        if not result_path.exists():
            fail(f"missing result file: {result_path.relative_to(ROOT)}")

        manifest_rows = read_jsonl(manifest_path)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = result.get("rows") or []

        manifest_ids = [row["sample_id"] for row in manifest_rows]
        result_ids = [row.get("sample_id") for row in rows]

        if len(rows) != len(manifest_rows):
            fail(f"{split} row count mismatch: result={len(rows)} manifest={len(manifest_rows)}")
        if len(set(result_ids)) != len(result_ids):
            fail(f"{split} has duplicate result sample_id")
        if set(result_ids) != set(manifest_ids):
            missing = sorted(set(manifest_ids) - set(result_ids))
            extra = sorted(set(result_ids) - set(manifest_ids))
            fail(f"{split} sample ids mismatch: missing={missing[:5]} extra={extra[:5]}")

        if result.get("summary", {}).get("error_count") != 0:
            fail(f"{split} summary error_count is not zero")

        for row in rows:
            if row.get("split") != split:
                fail(f"{split} row has wrong split: {row.get('sample_id')}")
            if row.get("error"):
                fail(f"{split} row has runtime error: {row.get('sample_id')}")
            if row.get("score") not in (0.0, 1.0):
                fail(f"{split} row score is not binary actual score: {row.get('sample_id')}")
            if "http://" in json.dumps(row) or "https://" in json.dumps(row):
                fail(f"{split} row stores signed/raw URL: {row.get('sample_id')}")

        taxonomy_counts = Counter(row["taxonomy_id"] for row in rows)
        expected_counts = Counter(row["taxonomy_id"] for row in manifest_rows)
        if taxonomy_counts != expected_counts:
            fail(f"{split} taxonomy counts mismatch: result={taxonomy_counts} manifest={expected_counts}")

        run = result.get("run", {})
        if run.get("milestone") != "M3":
            fail(f"{split} run milestone is not M3")
        if run.get("adapter") is not None:
            fail(f"{split} base audit result must not use adapter")
        if run.get("visual_policy") != "qwen3_vl_fixed_pixel_budget":
            fail(f"{split} unexpected visual policy")

        passed[split] = result["summary"]

    print(f"M3 actual base audit checks passed: {json.dumps(passed, ensure_ascii=False, sort_keys=True)}")


if __name__ == "__main__":
    main()
