from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST = ROOT / "data/mvp/train.jsonl"
HOLDOUT_MANIFEST = ROOT / "data/mvp/holdout.jsonl"
CURRICULUM_MANIFEST = ROOT / "data/curricula/curriculum_manifest.jsonl"
TEACHER_ANNOTATIONS = ROOT / "data/curricula/teacher_annotations.jsonl"
ADAPTER_PLAN = ROOT / "configs/track_a/adapter_candidate_plan.yaml"
CERTIFICATION_PLAN = ROOT / "configs/track_a/certification_eval_plan.yaml"

EXPECTED_TAXONOMIES = {"document_field_bind", "chart_table_cell"}
EXPECTED_SPLITS = {"train", "holdout"}
EXPECTED_PER_TAXONOMY_SPLIT = 64
REQUIRED_COMPARISONS = {
    "base_no_adapter",
    "correct_lora",
    "wrong_adapter",
    "random_untrained_lora",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def read_text(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path.relative_to(ROOT)}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        fail(f"empty file: {path.relative_to(ROOT)}")
    return text


def check_curriculum_against_sources() -> list[dict[str, Any]]:
    source_rows = load_jsonl(TRAIN_MANIFEST) + load_jsonl(HOLDOUT_MANIFEST)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    curriculum_rows = load_jsonl(CURRICULUM_MANIFEST)

    if len(curriculum_rows) != len(source_rows):
        fail(f"curriculum row count mismatch: {len(curriculum_rows)} != {len(source_rows)}")

    curriculum_ids = {row["sample_id"] for row in curriculum_rows}
    if curriculum_ids != set(source_by_id):
        missing = sorted(set(source_by_id) - curriculum_ids)[:5]
        extra = sorted(curriculum_ids - set(source_by_id))[:5]
        fail(f"curriculum sample_id mismatch, missing={missing}, extra={extra}")

    counts = Counter((row["taxonomy_id"], row["split"]) for row in curriculum_rows)
    for taxonomy_id in EXPECTED_TAXONOMIES:
        for split in EXPECTED_SPLITS:
            if counts[(taxonomy_id, split)] != EXPECTED_PER_TAXONOMY_SPLIT:
                fail(f"{taxonomy_id}/{split} count is {counts[(taxonomy_id, split)]}, expected 64")

    for row in curriculum_rows:
        source = source_by_id[row["sample_id"]]
        if row["split"] != source["split"]:
            fail(f"{row['sample_id']} split changed")
        if row["taxonomy_id"] != source["taxonomy_id"]:
            fail(f"{row['sample_id']} taxonomy_id changed")
        if row["taxonomy"] != source["taxonomy"]:
            fail(f"{row['sample_id']} taxonomy changed")
        if row["expected_answers"] != source["expected_answers"]:
            fail(f"{row['sample_id']} expected_answers changed")
        if row["hard_negatives"] != source["hard_negatives"]:
            fail(f"{row['sample_id']} hard_negatives changed")
        if row.get("teacher_label_is_candidate") is not True:
            fail(f"{row['sample_id']} teacher_label_is_candidate must be true")
        if not row.get("teacher_annotation_id"):
            fail(f"{row['sample_id']} missing teacher_annotation_id")
        if row.get("annotation_source") not in {"rule", "manual", "gemma"}:
            fail(f"{row['sample_id']} unexpected annotation_source: {row.get('annotation_source')}")
        if row.get("taxonomy_id") not in EXPECTED_TAXONOMIES:
            fail(f"{row['sample_id']} unexpected taxonomy_id: {row.get('taxonomy_id')}")
        if not isinstance(row.get("taxonomy"), dict):
            fail(f"{row['sample_id']} taxonomy must be an object")
        for taxonomy_key in ["domain", "evidence_type", "operation", "failure_mode"]:
            if taxonomy_key not in row["taxonomy"]:
                fail(f"{row['sample_id']} taxonomy missing {taxonomy_key}")
        diagnostic = row.get("base_actual_diagnostic")
        if not isinstance(diagnostic, dict):
            fail(f"{row['sample_id']} missing base_actual_diagnostic")
        if diagnostic.get("model_id") != "qwen3_vl_4b_base":
            fail(f"{row['sample_id']} base diagnostic model_id mismatch")
        if diagnostic.get("visual_policy") != "qwen3_vl_fixed_pixel_budget":
            fail(f"{row['sample_id']} visual_policy mismatch")

    return curriculum_rows


def check_teacher_annotations(curriculum_rows: list[dict[str, Any]]) -> None:
    annotations = load_jsonl(TEACHER_ANNOTATIONS)
    annotation_by_id = {row["teacher_annotation_id"]: row for row in annotations}
    expected_ids = {row["teacher_annotation_id"] for row in curriculum_rows}
    if set(annotation_by_id) != expected_ids:
        missing = sorted(expected_ids - set(annotation_by_id))[:5]
        extra = sorted(set(annotation_by_id) - expected_ids)[:5]
        fail(f"teacher annotation id mismatch, missing={missing}, extra={extra}")
    for annotation in annotations:
        if annotation.get("teacher_label_is_candidate") is not True:
            fail(f"{annotation['teacher_annotation_id']} teacher_label_is_candidate must be true")
        if annotation.get("annotation_source") != "rule":
            fail(f"{annotation['teacher_annotation_id']} annotation_source must be rule for M4 MVP")
        if annotation.get("teacher_model") != "rule_mvp_manifest_v1":
            fail(f"{annotation['teacher_annotation_id']} teacher_model mismatch")


def check_split_files(curriculum_rows: list[dict[str, Any]]) -> None:
    ids_by_taxonomy_split: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in curriculum_rows:
        ids_by_taxonomy_split[(row["taxonomy_id"], row["split"])].add(row["sample_id"])

    for taxonomy_id in EXPECTED_TAXONOMIES:
        for split in EXPECTED_SPLITS:
            path = ROOT / f"data/curricula/{taxonomy_id}_{split}.jsonl"
            rows = load_jsonl(path)
            ids = {row["sample_id"] for row in rows}
            if ids != ids_by_taxonomy_split[(taxonomy_id, split)]:
                fail(f"{path.relative_to(ROOT)} does not match combined curriculum manifest")


def check_plan_files() -> None:
    adapter_plan = read_text(ADAPTER_PLAN)
    certification_plan = read_text(CERTIFICATION_PLAN)

    for token in [
        "plan_name: \"m4_adapter_candidate_plan\"",
        "teacher_label_is_candidate_required: true",
        "rule_annotations_are_candidate_supervision_only: true",
        "adapter_id: \"chart_table_cell_r4_v1\"",
        "m4_status: \"training_ready\"",
        "adapter_id: \"document_field_bind_r4_v1\"",
        "m4_status: \"compiled_but_deferred_for_strong_claim\"",
        "train_holdout_split_preserved: true",
    ]:
        if token not in adapter_plan:
            fail(f"adapter plan missing token: {token}")

    for token in [
        "plan_name: \"m4_certification_eval_plan\"",
        "actual_eval_required: true",
        "no_smoke_validation: true",
        "same_holdout_samples: true",
        "same_visual_policy: true",
        "same_roi_source: true",
    ]:
        if token not in certification_plan:
            fail(f"certification plan missing token: {token}")

    for comparison in REQUIRED_COMPARISONS:
        if f"\"{comparison}\"" not in certification_plan:
            fail(f"certification plan missing comparison: {comparison}")


def main() -> None:
    curriculum_rows = check_curriculum_against_sources()
    check_teacher_annotations(curriculum_rows)
    check_split_files(curriculum_rows)
    check_plan_files()
    print("M4 curriculum checks passed.")


if __name__ == "__main__":
    main()
