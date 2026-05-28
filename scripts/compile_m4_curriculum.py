from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

TRAIN_MANIFEST = ROOT / "data/mvp/train.jsonl"
HOLDOUT_MANIFEST = ROOT / "data/mvp/holdout.jsonl"
M3_TRAIN_RESULT = ROOT / "results/m3/qwen3_vl_4b_base_train.json"
M3_HOLDOUT_RESULT = ROOT / "results/m3/qwen3_vl_4b_base_holdout.json"

CURRICULUM_DIR = ROOT / "data/curricula"
CURRICULUM_MANIFEST = CURRICULUM_DIR / "curriculum_manifest.jsonl"
TEACHER_ANNOTATIONS = CURRICULUM_DIR / "teacher_annotations.jsonl"
ADAPTER_PLAN = ROOT / "configs/track_a/adapter_candidate_plan.yaml"
CERTIFICATION_PLAN = ROOT / "configs/track_a/certification_eval_plan.yaml"

BASE_BACKBONE = "qwen3_vl_4b_reference"
BASE_MODEL_ID = "qwen3_vl_4b_base"
M4_COMPILE_VERSION = 1

TAXONOMY_PLAN = {
    "chart_table_cell": {
        "adapter_id": "chart_table_cell_r4_v1",
        "priority": 1,
        "m4_status": "training_ready",
        "difficulty_status": "usable",
        "claim_status": "eligible_for_m5_first_pass",
        "notes": [
            "M3 base score leaves room for adapter-sensitive learning.",
            "Use this taxonomy as the first M5 candidate.",
        ],
    },
    "document_field_bind": {
        "adapter_id": "document_field_bind_r4_v1",
        "priority": 2,
        "m4_status": "compiled_but_deferred_for_strong_claim",
        "difficulty_status": "too_easy",
        "claim_status": "needs_harder_split_before_strong_claim",
        "notes": [
            "M3 base score is too high for a strong LoRA gain claim.",
            "Keep rows compiled for reproducibility, but prefer harder document data before certification claims.",
        ],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.relative_to(ROOT)}:{line_number} invalid JSONL: {exc}") from exc
    return rows


def load_m3_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows_by_id = {row["sample_id"]: row for row in data["rows"]}
    return {
        "run": data["run"],
        "summary": data["summary"],
        "rows_by_id": rows_by_id,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_dump(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{pad}-")
                lines.append(yaml_dump(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{pad}-")
                lines.append(yaml_dump(item, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{yaml_scalar(value)}"


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_dump(data) + "\n", encoding="utf-8", newline="\n")


def base_summary_for(summary: dict[str, Any], taxonomy_id: str) -> dict[str, Any]:
    per_taxonomy = summary["per_taxonomy"][taxonomy_id]
    return {
        "count": per_taxonomy["count"],
        "score": per_taxonomy["score"],
        "failure_types": per_taxonomy["failure_types"],
    }


def make_teacher_annotation(row: dict[str, Any], m3_row: dict[str, Any]) -> dict[str, Any]:
    taxonomy = row["taxonomy"]
    return {
        "teacher_annotation_id": f"rule_{row['sample_id']}",
        "sample_id": row["sample_id"],
        "teacher_model": "rule_mvp_manifest_v1",
        "annotation_source": "rule",
        "taxonomy": taxonomy,
        "failure_mode_hypothesis": taxonomy["failure_mode"],
        "rationale": (
            "Rule annotation preserves the M2 expected answer and hard negatives; "
            "M3 base output is attached as diagnostic evidence only."
        ),
        "hard_negatives": row["hard_negatives"],
        "teacher_label_is_candidate": True,
        "base_actual_diagnostic": {
            "model_id": BASE_MODEL_ID,
            "answer_text": m3_row.get("answer_text"),
            "score": m3_row.get("score"),
            "score_method": m3_row.get("score_method"),
        },
    }


def make_curriculum_row(row: dict[str, Any], m3_row: dict[str, Any]) -> dict[str, Any]:
    taxonomy_id = row["taxonomy_id"]
    plan = TAXONOMY_PLAN[taxonomy_id]
    compiled = dict(row)
    compiled["teacher_annotation_id"] = f"rule_{row['sample_id']}"
    compiled["teacher_model"] = "rule_mvp_manifest_v1"
    compiled["annotation_source"] = "rule"
    compiled["teacher_label_is_candidate"] = True
    compiled["m4_compile_version"] = M4_COMPILE_VERSION
    compiled["m4_candidate_status"] = plan["m4_status"]
    compiled["m4_priority"] = plan["priority"]
    compiled["m4_claim_status"] = plan["claim_status"]
    compiled["base_difficulty_status"] = plan["difficulty_status"]
    compiled["curriculum_role"] = "train_supervision" if row["split"] == "train" else "holdout_certification"
    compiled["base_actual_diagnostic"] = {
        "model_id": BASE_MODEL_ID,
        "adapter": None,
        "visual_policy": "qwen3_vl_fixed_pixel_budget",
        "roi_source": None,
        "score": m3_row.get("score"),
        "score_method": m3_row.get("score_method"),
        "answer_text": m3_row.get("answer_text"),
        "failure_observed": m3_row.get("score") == 0.0,
    }
    return compiled


def collect_rows() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    train_rows = load_jsonl(TRAIN_MANIFEST)
    holdout_rows = load_jsonl(HOLDOUT_MANIFEST)
    m3_train = load_m3_result(M3_TRAIN_RESULT)
    m3_holdout = load_m3_result(M3_HOLDOUT_RESULT)

    compiled_rows: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for source_rows, m3_result in [(train_rows, m3_train), (holdout_rows, m3_holdout)]:
        for row in source_rows:
            m3_row = m3_result["rows_by_id"].get(row["sample_id"])
            if not m3_row:
                raise ValueError(f"missing M3 row for sample_id: {row['sample_id']}")
            compiled_rows.append(make_curriculum_row(row, m3_row))
            annotations.append(make_teacher_annotation(row, m3_row))
    return compiled_rows, m3_train, m3_holdout, annotations


def make_adapter_plan(compiled_rows: list[dict[str, Any]], m3_train: dict[str, Any], m3_holdout: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "holdout": 0})
    for row in compiled_rows:
        counts[row["taxonomy_id"]][row["split"]] += 1

    candidates = []
    for taxonomy_id, plan in sorted(TAXONOMY_PLAN.items(), key=lambda item: item[1]["priority"]):
        candidates.append(
            {
                "adapter_id": plan["adapter_id"],
                "taxonomy_id": taxonomy_id,
                "priority": plan["priority"],
                "m4_status": plan["m4_status"],
                "claim_status": plan["claim_status"],
                "base_backbone": BASE_BACKBONE,
                "backbone_type": "vlm_llm",
                "adapter_slot": "language_decoder",
                "teacher_mode": "rule",
                "teacher_label_is_candidate_required": True,
                "taxonomy": next(row["taxonomy"] for row in compiled_rows if row["taxonomy_id"] == taxonomy_id),
                "curriculum": {
                    "train_manifest": f"data/curricula/{taxonomy_id}_train.jsonl",
                    "holdout_manifest": f"data/curricula/{taxonomy_id}_holdout.jsonl",
                    "train_rows": counts[taxonomy_id]["train"],
                    "holdout_rows": counts[taxonomy_id]["holdout"],
                    "train_holdout_split_preserved": True,
                },
                "base_actual": {
                    "train": base_summary_for(m3_train["summary"], taxonomy_id),
                    "holdout": base_summary_for(m3_holdout["summary"], taxonomy_id),
                    "difficulty_status": plan["difficulty_status"],
                },
                "training_defaults": {
                    "rank": 4,
                    "alpha": 8,
                    "target_modules": ["q_proj", "v_proj"],
                    "label_mask_mode": "answer_only",
                    "weights_path": f"models/loras/{plan['adapter_id']}",
                },
                "notes": plan["notes"],
            }
        )

    return {
        "plan_name": "m4_adapter_candidate_plan",
        "milestone": "M4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_curriculum_manifest": "data/curricula/curriculum_manifest.jsonl",
        "source_teacher_annotations": "data/curricula/teacher_annotations.jsonl",
        "base_backbone": BASE_BACKBONE,
        "teacher_handling": {
            "mode": "rule",
            "teacher_label_is_candidate_required": True,
            "rule_annotations_are_candidate_supervision_only": True,
        },
        "candidates": candidates,
    }


def make_certification_plan(adapter_plan: dict[str, Any], m3_train: dict[str, Any], m3_holdout: dict[str, Any]) -> dict[str, Any]:
    eval_candidates = []
    for candidate in adapter_plan["candidates"]:
        adapter_id = candidate["adapter_id"]
        taxonomy_id = candidate["taxonomy_id"]
        eval_candidates.append(
            {
                "adapter_id": adapter_id,
                "taxonomy_id": taxonomy_id,
                "m4_status": candidate["m4_status"],
                "train_manifest": candidate["curriculum"]["train_manifest"],
                "holdout_manifest": candidate["curriculum"]["holdout_manifest"],
                "holdout_samples": 64,
                "base_reference_result": "results/m3/qwen3_vl_4b_base_holdout.json",
                "required_actual_runs": [
                    {
                        "name": "base_no_adapter",
                        "adapter": None,
                        "result_path": "results/m3/qwen3_vl_4b_base_holdout.json",
                    },
                    {
                        "name": "correct_lora",
                        "adapter": adapter_id,
                        "result_path": f"results/m5/{adapter_id}_correct_holdout.json",
                    },
                    {
                        "name": "wrong_adapter",
                        "adapter": "other_taxonomy_adapter",
                        "result_path": f"results/m6/{adapter_id}_wrong_holdout.json",
                    },
                    {
                        "name": "random_untrained_lora",
                        "adapter": "same_rank_random_adapter",
                        "result_path": f"results/m6/{adapter_id}_random_holdout.json",
                    },
                ],
                "gates_required_for_certification": {
                    "actual_fields_complete": True,
                    "same_holdout_samples": True,
                    "same_visual_policy": True,
                    "same_roi_source": True,
                    "gain_vs_base_positive": True,
                    "correct_beats_wrong": True,
                    "correct_beats_random": True,
                },
            }
        )

    return {
        "plan_name": "m4_certification_eval_plan",
        "milestone": "M4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_eval_required": True,
        "no_smoke_validation": True,
        "base_backbone": BASE_BACKBONE,
        "base_model_id": BASE_MODEL_ID,
        "m3_base_results": {
            "train": {
                "path": "results/m3/qwen3_vl_4b_base_train.json",
                "score": m3_train["summary"]["score"],
                "error_count": m3_train["summary"]["error_count"],
            },
            "holdout": {
                "path": "results/m3/qwen3_vl_4b_base_holdout.json",
                "score": m3_holdout["summary"]["score"],
                "error_count": m3_holdout["summary"]["error_count"],
            },
        },
        "eval_controls": {
            "visual_policy": "qwen3_vl_fixed_pixel_budget",
            "min_pixels": 200704,
            "max_pixels": 1003520,
            "roi_source": None,
            "scoring_policy": "normalized_exact_or_contains_numeric_only_when_expected_has_no_letters",
            "same_prompt_required": True,
            "same_expected_answers_required": True,
        },
        "required_comparisons": [
            "base_no_adapter",
            "correct_lora",
            "wrong_adapter",
            "random_untrained_lora",
        ],
        "candidates": eval_candidates,
    }


def main() -> None:
    compiled_rows, m3_train, m3_holdout, annotations = collect_rows()

    write_jsonl(CURRICULUM_MANIFEST, compiled_rows)
    write_jsonl(TEACHER_ANNOTATIONS, annotations)

    rows_by_taxonomy_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compiled_rows:
        rows_by_taxonomy_split[(row["taxonomy_id"], row["split"])].append(row)
    for (taxonomy_id, split), rows in sorted(rows_by_taxonomy_split.items()):
        write_jsonl(CURRICULUM_DIR / f"{taxonomy_id}_{split}.jsonl", rows)

    adapter_plan = make_adapter_plan(compiled_rows, m3_train, m3_holdout)
    certification_plan = make_certification_plan(adapter_plan, m3_train, m3_holdout)
    write_yaml(ADAPTER_PLAN, adapter_plan)
    write_yaml(CERTIFICATION_PLAN, certification_plan)

    print(f"Wrote {CURRICULUM_MANIFEST.relative_to(ROOT)} ({len(compiled_rows)} rows)")
    print(f"Wrote {TEACHER_ANNOTATIONS.relative_to(ROOT)} ({len(annotations)} rows)")
    print(f"Wrote {ADAPTER_PLAN.relative_to(ROOT)}")
    print(f"Wrote {CERTIFICATION_PLAN.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
