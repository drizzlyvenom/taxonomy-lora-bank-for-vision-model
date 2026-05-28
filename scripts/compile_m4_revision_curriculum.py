from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from compile_m4_curriculum import (
    BASE_BACKBONE,
    BASE_MODEL_ID,
    M4_COMPILE_VERSION,
    load_jsonl,
    load_m3_result,
    write_jsonl,
    write_yaml,
)


ROOT = Path(__file__).resolve().parents[1]


def root_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT).as_posix())


def base_summary_for(summary: dict[str, Any], taxonomy_id: str) -> dict[str, Any]:
    per_taxonomy = summary["per_taxonomy"][taxonomy_id]
    return {
        "count": per_taxonomy["count"],
        "score": per_taxonomy["score"],
        "failure_types": per_taxonomy["failure_types"],
    }


def make_teacher_annotation(row: dict[str, Any], m3_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "teacher_annotation_id": f"rule_{row['sample_id']}",
        "sample_id": row["sample_id"],
        "teacher_model": "rule_mvp_manifest_v1",
        "annotation_source": "rule",
        "taxonomy": row["taxonomy"],
        "failure_mode_hypothesis": row["taxonomy"]["failure_mode"],
        "rationale": (
            "Rule annotation preserves the revised expected answer and hard negatives; "
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


def make_curriculum_row(row: dict[str, Any], m3_row: dict[str, Any], adapter_id: str) -> dict[str, Any]:
    compiled = dict(row)
    compiled["teacher_annotation_id"] = f"rule_{row['sample_id']}"
    compiled["teacher_model"] = "rule_mvp_manifest_v1"
    compiled["annotation_source"] = "rule"
    compiled["teacher_label_is_candidate"] = True
    compiled["m4_compile_version"] = M4_COMPILE_VERSION
    compiled["m4_candidate_status"] = "training_ready"
    compiled["m4_priority"] = 1
    compiled["m4_claim_status"] = "eligible_for_m5_rerun"
    compiled["base_difficulty_status"] = "measured_by_revision_m3"
    compiled["curriculum_role"] = "train_supervision" if row["split"] == "train" else "holdout_certification"
    compiled["adapter_candidate_id"] = adapter_id
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


def collect_rows(
    train_manifest: Path,
    holdout_manifest: Path,
    m3_train_result: Path,
    m3_holdout_result: Path,
    adapter_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    train_rows = load_jsonl(train_manifest)
    holdout_rows = load_jsonl(holdout_manifest)
    m3_train = load_m3_result(m3_train_result)
    m3_holdout = load_m3_result(m3_holdout_result)

    compiled_rows: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for source_rows, m3_result in [(train_rows, m3_train), (holdout_rows, m3_holdout)]:
        for row in source_rows:
            m3_row = m3_result["rows_by_id"].get(row["sample_id"])
            if not m3_row:
                raise ValueError(f"missing M3 row for sample_id: {row['sample_id']}")
            compiled_rows.append(make_curriculum_row(row, m3_row, adapter_id))
            annotations.append(make_teacher_annotation(row, m3_row))
    return compiled_rows, m3_train, m3_holdout, annotations


def make_adapter_plan(
    compiled_rows: list[dict[str, Any]],
    m3_train: dict[str, Any],
    m3_holdout: dict[str, Any],
    taxonomy_id: str,
    adapter_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "holdout": 0})
    for row in compiled_rows:
        counts[row["taxonomy_id"]][row["split"]] += 1

    return {
        "plan_name": "m4_adapter_candidate_plan_revision",
        "milestone": "M4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_curriculum_manifest": root_relative(output_dir / "curriculum_manifest.jsonl"),
        "source_teacher_annotations": root_relative(output_dir / "teacher_annotations.jsonl"),
        "base_backbone": BASE_BACKBONE,
        "teacher_handling": {
            "mode": "rule",
            "teacher_label_is_candidate_required": True,
            "rule_annotations_are_candidate_supervision_only": True,
        },
        "candidates": [
            {
                "adapter_id": adapter_id,
                "taxonomy_id": taxonomy_id,
                "priority": 1,
                "m4_status": "training_ready",
                "claim_status": "eligible_for_m5_rerun",
                "base_backbone": BASE_BACKBONE,
                "backbone_type": "vlm_llm",
                "adapter_slot": "language_decoder",
                "teacher_mode": "rule",
                "teacher_label_is_candidate_required": True,
                "taxonomy": next(row["taxonomy"] for row in compiled_rows if row["taxonomy_id"] == taxonomy_id),
                "curriculum": {
                    "train_manifest": root_relative(output_dir / f"{taxonomy_id}_train.jsonl"),
                    "holdout_manifest": root_relative(output_dir / f"{taxonomy_id}_holdout.jsonl"),
                    "train_rows": counts[taxonomy_id]["train"],
                    "holdout_rows": counts[taxonomy_id]["holdout"],
                    "train_holdout_split_preserved": True,
                },
                "base_actual": {
                    "train": base_summary_for(m3_train["summary"], taxonomy_id),
                    "holdout": base_summary_for(m3_holdout["summary"], taxonomy_id),
                    "difficulty_status": "measured_by_revision_m3",
                },
                "training_defaults": {
                    "rank": 4,
                    "alpha": 8,
                    "target_modules": ["q_proj", "v_proj"],
                    "label_mask_mode": "answer_only",
                    "weights_path": f"models/loras/{adapter_id}",
                },
                "notes": [
                    "ChartQAPro replaces the earlier ChartQA source for this revision run.",
                    "This plan preserves actual 64/64 evaluation and does not open certification by itself.",
                ],
            }
        ],
    }


def make_certification_plan(
    adapter_id: str,
    taxonomy_id: str,
    output_dir: Path,
    m3_train: dict[str, Any],
    m3_holdout: dict[str, Any],
    m3_train_path: Path,
    m3_holdout_path: Path,
    m5_result_dir: Path,
) -> dict[str, Any]:
    return {
        "plan_name": "m4_certification_eval_plan_revision",
        "milestone": "M4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "actual_eval_required": True,
        "no_smoke_validation": True,
        "base_backbone": BASE_BACKBONE,
        "base_model_id": BASE_MODEL_ID,
        "m3_base_results": {
            "train": {
                "path": root_relative(m3_train_path),
                "score": m3_train["summary"]["score"],
                "error_count": m3_train["summary"]["error_count"],
            },
            "holdout": {
                "path": root_relative(m3_holdout_path),
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
        "candidates": [
            {
                "adapter_id": adapter_id,
                "taxonomy_id": taxonomy_id,
                "m4_status": "training_ready",
                "train_manifest": root_relative(output_dir / f"{taxonomy_id}_train.jsonl"),
                "holdout_manifest": root_relative(output_dir / f"{taxonomy_id}_holdout.jsonl"),
                "holdout_samples": 64,
                "base_reference_result": root_relative(m3_holdout_path),
                "required_actual_runs": [
                    {
                        "name": "base_no_adapter",
                        "adapter": None,
                        "result_path": root_relative(m3_holdout_path),
                    },
                    {
                        "name": "correct_lora",
                        "adapter": adapter_id,
                        "result_path": root_relative(m5_result_dir / f"{adapter_id}_correct_holdout.json"),
                    },
                    {
                        "name": "wrong_adapter",
                        "adapter": "other_taxonomy_adapter",
                        "result_path": f"results/m6_revisions/{adapter_id}_wrong_holdout.json",
                    },
                    {
                        "name": "random_untrained_lora",
                        "adapter": "same_rank_random_adapter",
                        "result_path": f"results/m6_revisions/{adapter_id}_random_holdout.json",
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
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/mvp_revisions/chartqapro_v1/train.jsonl")
    parser.add_argument("--holdout-manifest", default="data/mvp_revisions/chartqapro_v1/holdout.jsonl")
    parser.add_argument("--m3-train-result", default="results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_train.json")
    parser.add_argument("--m3-holdout-result", default="results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_holdout.json")
    parser.add_argument("--output-dir", default="data/curricula_revisions/chartqapro_v1")
    parser.add_argument("--plan-output", default="configs/track_a/revisions/chartqapro_v1/adapter_candidate_plan.yaml")
    parser.add_argument(
        "--certification-output",
        default="configs/track_a/revisions/chartqapro_v1/certification_eval_plan.yaml",
    )
    parser.add_argument("--m5-result-dir", default="results/m5_revisions/chartqapro_v1")
    parser.add_argument("--adapter-id", default="chartqapro_table_cell_r4_v1")
    parser.add_argument("--taxonomy-id", default="chart_table_cell")
    args = parser.parse_args()

    train_manifest = ROOT / args.train_manifest
    holdout_manifest = ROOT / args.holdout_manifest
    m3_train_path = ROOT / args.m3_train_result
    m3_holdout_path = ROOT / args.m3_holdout_result
    output_dir = ROOT / args.output_dir
    plan_output = ROOT / args.plan_output
    certification_output = ROOT / args.certification_output
    m5_result_dir = ROOT / args.m5_result_dir

    compiled_rows, m3_train, m3_holdout, annotations = collect_rows(
        train_manifest,
        holdout_manifest,
        m3_train_path,
        m3_holdout_path,
        args.adapter_id,
    )

    write_jsonl(output_dir / "curriculum_manifest.jsonl", compiled_rows)
    write_jsonl(output_dir / "teacher_annotations.jsonl", annotations)

    rows_by_taxonomy_split: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compiled_rows:
        rows_by_taxonomy_split[(row["taxonomy_id"], row["split"])].append(row)
    for (taxonomy_id, split), rows in sorted(rows_by_taxonomy_split.items()):
        write_jsonl(output_dir / f"{taxonomy_id}_{split}.jsonl", rows)

    adapter_plan = make_adapter_plan(compiled_rows, m3_train, m3_holdout, args.taxonomy_id, args.adapter_id, output_dir)
    certification_plan = make_certification_plan(
        args.adapter_id,
        args.taxonomy_id,
        output_dir,
        m3_train,
        m3_holdout,
        m3_train_path,
        m3_holdout_path,
        m5_result_dir,
    )
    write_yaml(plan_output, adapter_plan)
    write_yaml(certification_output, certification_plan)

    summary = {
        "revision": "chartqapro_v1",
        "adapter_id": args.adapter_id,
        "taxonomy_id": args.taxonomy_id,
        "compiled_rows": len(compiled_rows),
        "teacher_annotations": len(annotations),
        "outputs": {
            "curriculum_manifest": root_relative(output_dir / "curriculum_manifest.jsonl"),
            "teacher_annotations": root_relative(output_dir / "teacher_annotations.jsonl"),
            "adapter_plan": root_relative(plan_output),
            "certification_plan": root_relative(certification_output),
        },
    }
    (output_dir / "compile_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {root_relative(output_dir / 'curriculum_manifest.jsonl')} ({len(compiled_rows)} rows)")
    print(f"Wrote {root_relative(output_dir / 'teacher_annotations.jsonl')} ({len(annotations)} rows)")
    print(f"Wrote {root_relative(plan_output)}")
    print(f"Wrote {root_relative(certification_output)}")


if __name__ == "__main__":
    main()
