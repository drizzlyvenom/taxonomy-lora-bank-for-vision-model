from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/mvp/holdout.jsonl"
RESULT = ROOT / "results/m3/qwen3_vl_4b_base_holdout.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fail(message: str) -> None:
    print(f"M3 check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not RESULT.exists():
        fail(f"missing result file: {RESULT.relative_to(ROOT)}")

    manifest_rows = read_jsonl(MANIFEST)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = result.get("rows") or []

    manifest_ids = [row["sample_id"] for row in manifest_rows]
    result_ids = [row.get("sample_id") for row in rows]

    if len(rows) != len(manifest_rows):
        fail(f"row count mismatch: result={len(rows)} manifest={len(manifest_rows)}")
    if len(set(result_ids)) != len(result_ids):
        fail("duplicate result sample_id")
    if set(result_ids) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(manifest_ids))
        fail(f"sample ids mismatch: missing={missing[:5]} extra={extra[:5]}")

    if result.get("summary", {}).get("error_count") != 0:
        fail("summary error_count is not zero")

    for row in rows:
        if row.get("error"):
            fail(f"row has runtime error: {row.get('sample_id')}")
        if row.get("score") not in (0.0, 1.0):
            fail(f"row score is not binary actual score: {row.get('sample_id')}")
        if "http://" in json.dumps(row) or "https://" in json.dumps(row):
            fail(f"row stores signed/raw URL: {row.get('sample_id')}")

    taxonomy_counts = Counter(row["taxonomy_id"] for row in rows)
    expected_counts = Counter(row["taxonomy_id"] for row in manifest_rows)
    if taxonomy_counts != expected_counts:
        fail(f"taxonomy counts mismatch: result={taxonomy_counts} manifest={expected_counts}")

    run = result.get("run", {})
    if run.get("milestone") != "M3":
        fail("run milestone is not M3")
    if run.get("adapter") is not None:
        fail("base audit result must not use adapter")
    if run.get("visual_policy") != "qwen3_vl_fixed_pixel_budget":
        fail("unexpected visual policy")

    summary = result["summary"]
    print(
        "M3 actual base audit checks passed: "
        f"rows={len(rows)} score={summary['score']} per_taxonomy={summary['per_taxonomy']}"
    )


if __name__ == "__main__":
    main()
