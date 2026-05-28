from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail(f"missing JSON file: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing JSONL file: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail(f"{path.relative_to(ROOT)}:{line_number} invalid JSONL: {exc}")
    if not rows:
        fail(f"empty JSONL file: {path.relative_to(ROOT)}")
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_eval_result(
    result: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    expected_split: str,
    adapter_id: str,
    taxonomy_id: str,
) -> float:
    run = result.get("run", {})
    if run.get("milestone") != "M5":
        fail(f"{expected_split} result milestone mismatch")
    if run.get("adapter") != adapter_id:
        fail(f"{expected_split} result adapter mismatch: {run.get('adapter')}")
    if run.get("visual_policy") != "qwen3_vl_fixed_pixel_budget":
        fail(f"{expected_split} visual_policy mismatch")
    if run.get("roi_source") is not None:
        fail(f"{expected_split} roi_source must be null")

    rows = result.get("rows", [])
    if len(rows) != len(manifest_rows):
        fail(f"{expected_split} row count mismatch: {len(rows)} != {len(manifest_rows)}")
    if len(rows) != 64:
        fail(f"{expected_split} must contain 64 actual rows")

    manifest_ids = [row["sample_id"] for row in manifest_rows]
    result_ids = [row["sample_id"] for row in rows]
    if result_ids != manifest_ids:
        fail(f"{expected_split} sample order or ids changed")

    for row in rows:
        if row.get("split") != expected_split:
            fail(f"{row['sample_id']} split mismatch")
        if row.get("taxonomy_id") != taxonomy_id:
            fail(f"{row['sample_id']} taxonomy_id mismatch")
        if row.get("error"):
            fail(f"{row['sample_id']} has runtime error: {row['error']}")
        if row.get("score") not in {0.0, 1.0}:
            fail(f"{row['sample_id']} score must be binary")
        if not str(row.get("full_image_path", "")).startswith("hf://datasets/"):
            fail(f"{row['sample_id']} full_image_path must avoid raw signed URLs")

    summary = result.get("summary", {})
    if summary.get("count") != 64 or summary.get("error_count") != 0:
        fail(f"{expected_split} summary count/error mismatch")
    per_taxonomy = summary.get("per_taxonomy", {}).get(taxonomy_id)
    if not per_taxonomy or per_taxonomy.get("count") != 64:
        fail(f"{expected_split} per-taxonomy summary missing")
    return float(per_taxonomy["score"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-id", default="chart_table_cell_r4_v1")
    parser.add_argument("--taxonomy-id", default="chart_table_cell")
    parser.add_argument("--train-manifest", default="data/curricula/chart_table_cell_train.jsonl")
    parser.add_argument("--holdout-manifest", default="data/curricula/chart_table_cell_holdout.jsonl")
    parser.add_argument("--train-result", default="results/m5/chart_table_cell_r4_v1_correct_train.json")
    parser.add_argument("--holdout-result", default="results/m5/chart_table_cell_r4_v1_correct_holdout.json")
    parser.add_argument("--train-log", default="results/m5/chart_table_cell_r4_v1_train_log.json")
    parser.add_argument("--m3-train-result", default="results/m3/qwen3_vl_4b_base_train.json")
    parser.add_argument("--m3-holdout-result", default="results/m3/qwen3_vl_4b_base_holdout.json")
    parser.add_argument("--summary-output", default="results/m5/chart_table_cell_r4_v1_m5_summary.json")
    args = parser.parse_args()

    train_manifest = read_jsonl(ROOT / args.train_manifest)
    holdout_manifest = read_jsonl(ROOT / args.holdout_manifest)
    train_result = read_json(ROOT / args.train_result)
    holdout_result = read_json(ROOT / args.holdout_result)
    train_log = read_json(ROOT / args.train_log)
    m3_train = read_json(ROOT / args.m3_train_result)
    m3_holdout = read_json(ROOT / args.m3_holdout_result)

    correct_train_score = check_eval_result(
        train_result, train_manifest, "train", args.adapter_id, args.taxonomy_id
    )
    correct_holdout_score = check_eval_result(
        holdout_result, holdout_manifest, "holdout", args.adapter_id, args.taxonomy_id
    )
    base_train_score = float(m3_train["summary"]["per_taxonomy"][args.taxonomy_id]["score"])
    base_holdout_score = float(m3_holdout["summary"]["per_taxonomy"][args.taxonomy_id]["score"])
    train_gain = correct_train_score - base_train_score
    holdout_gain = correct_holdout_score - base_holdout_score

    train_gate = train_gain > 0
    holdout_gate = holdout_gain >= 0
    if train_gate and holdout_gate:
        gate_status = "pass"
        failure_reason = None
    elif train_gate:
        gate_status = "soft_pass"
        failure_reason = "holdout_no_gain_after_train_gain"
    else:
        gate_status = "fail"
        failure_reason = "no_learning_signal"

    run = train_log["run"]
    summary = {
        "milestone": "M5",
        "adapter_id": args.adapter_id,
        "taxonomy_id": args.taxonomy_id,
        "base_backbone": "qwen3_vl_4b_reference",
        "adapter_slot": "language_decoder",
        "label_mask_mode": run.get("label_mask_mode"),
        "target_modules": run.get("target_modules"),
        "rank": run.get("rank"),
        "alpha": run.get("alpha"),
        "steps": run.get("steps"),
        "scores": {
            "base_train": base_train_score,
            "correct_lora_train": correct_train_score,
            "base_holdout": base_holdout_score,
            "correct_lora_holdout": correct_holdout_score,
        },
        "gains": {
            "train_gain_vs_base": train_gain,
            "holdout_gain_vs_base": holdout_gain,
        },
        "gates": {
            "correct_lora_train_score_gt_actual_base_train_score": train_gate,
            "correct_lora_holdout_score_gte_actual_base_holdout_score": holdout_gate,
            "train_overfit_allowed_for_first_diagnosis": True,
            "holdout_no_gain_failure_reason_recorded": failure_reason is not None,
        },
        "status": gate_status,
        "failure_reason": failure_reason,
        "result_paths": {
            "train_log": args.train_log,
            "train_eval": args.train_result,
            "holdout_eval": args.holdout_result,
        },
    }
    write_json(ROOT / args.summary_output, summary)
    print(f"M5 single LoRA checks passed with status={gate_status}: {json.dumps(summary['scores'], sort_keys=True)}")


if __name__ == "__main__":
    main()
