from __future__ import annotations

import argparse
import ast
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

DATASET_SPEC = {
    "taxonomy_id": "chart_table_cell",
    "dataset": "ahmed-masry/ChartQAPro",
    "config": "default",
    "source_split": "test",
    "question_field": "Question",
    "answer_field": "Answer",
    "question_type_field": "Question Type",
    "image_field": "image",
    "taxonomy": {
        "domain": "chart",
        "evidence_type": "table_cell",
        "operation": "locate",
        "failure_mode": "label_value_mismatch",
    },
}


def fetch_rows(offset: int, length: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_SPEC["dataset"],
            "config": DATASET_SPEC["config"],
            "split": DATASET_SPEC["source_split"],
            "offset": offset,
            "length": length,
        }
    )
    url = f"{DATASET_ROWS_ENDPOINT}?{query}"
    last_error: Exception | None = None
    for wait_seconds in [5, 15, 45, 90, 120]:
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                time.sleep(wait_seconds)
            else:
                time.sleep(min(wait_seconds, 10))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(wait_seconds)
    raise RuntimeError(f"failed to fetch ChartQAPro rows at offset {offset}") from last_error


def normalize_answer(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def flatten_values(value: Any) -> list[Any]:
    if isinstance(value, list):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(flatten_values(item))
        return flattened
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return [value]
            if isinstance(parsed, list):
                return flatten_values(parsed)
        return [value]
    if value is None:
        return []
    return [value]


def as_list(value: Any) -> list[str]:
    return [str(item).strip() for item in flatten_values(value) if str(item).strip()]


def iter_qa_pairs(item: dict[str, Any]) -> list[dict[str, Any]]:
    row = item["row"]
    questions = as_list(row.get(DATASET_SPEC["question_field"]))
    answers = as_list(row.get(DATASET_SPEC["answer_field"]))
    if not questions or not answers:
        return []

    pairs: list[dict[str, Any]] = []
    if len(questions) == len(answers):
        iterable = enumerate(zip(questions, answers, strict=False))
    elif len(questions) == 1:
        iterable = enumerate((questions[0], answer) for answer in answers)
    elif len(answers) == 1:
        iterable = enumerate((question, answers[0]) for question in questions)
    else:
        iterable = enumerate(zip(questions, answers, strict=False))

    for pair_index, (question, answer) in iterable:
        answer_key = normalize_answer(answer)
        if not question.strip() or not answer_key:
            continue
        pairs.append(
            {
                "source_row_idx": item["row_idx"],
                "question_index": pair_index,
                "question": question.strip(),
                "expected_answers": [answer.strip()],
                "expected_answer_key": answer_key,
                "question_type": row.get(DATASET_SPEC["question_type_field"]),
            }
        )
    return pairs


def collect_candidates(needed: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_answer_keys: set[str] = set()
    seen_prompts: set[tuple[str, str]] = set()
    offset = 0
    page_size = 100

    while len(selected) < needed:
        payload = fetch_rows(offset, page_size)
        rows = payload.get("rows") or []
        if not rows:
            raise RuntimeError(f"not enough ChartQAPro rows: got {len(selected)}, need {needed}")

        for item in rows:
            for pair in iter_qa_pairs(item):
                answer_key = pair["expected_answer_key"]
                prompt_key = (pair["question"].lower(), answer_key)
                if answer_key in seen_answer_keys or prompt_key in seen_prompts:
                    continue
                seen_answer_keys.add(answer_key)
                seen_prompts.add(prompt_key)
                selected.append(pair)
                if len(selected) >= needed:
                    break
            if len(selected) >= needed:
                break

        offset += page_size
        if offset >= payload.get("num_rows_total", offset) and len(selected) < needed:
            raise RuntimeError(f"exhausted ChartQAPro: got {len(selected)}, need {needed}")

    return selected


def build_hard_negatives(
    selected: list[dict[str, Any]], current_index: int, split_start: int, split_end: int, count: int = 3
) -> list[str]:
    negatives: list[str] = []
    current_keys = {normalize_answer(answer) for answer in selected[current_index]["expected_answers"]}
    for idx in range(split_start, split_end):
        if idx == current_index:
            continue
        for answer in selected[idx]["expected_answers"]:
            key = normalize_answer(answer)
            if key and key not in current_keys and answer not in negatives:
                negatives.append(answer)
            if len(negatives) >= count:
                return negatives
    raise RuntimeError("could not build enough hard negatives")


def manifest_row(selected: list[dict[str, Any]], index: int, split: str, curriculum_id: str) -> dict[str, Any]:
    item = selected[index]
    split_start = 0 if split == "train" else 64
    split_end = 64 if split == "train" else 128
    taxonomy_id = DATASET_SPEC["taxonomy_id"]
    source_ref = (
        f"hf://datasets/{DATASET_SPEC['dataset']}/{DATASET_SPEC['config']}/"
        f"{DATASET_SPEC['source_split']}/{item['source_row_idx']}/{DATASET_SPEC['image_field']}"
    )

    return {
        "curriculum_id": curriculum_id,
        "sample_id": f"{taxonomy_id}_{split}_{index - split_start:06d}",
        "split": split,
        "taxonomy_id": taxonomy_id,
        "taxonomy": DATASET_SPEC["taxonomy"],
        "prompt": item["question"],
        "expected_answers": item["expected_answers"],
        "expected_answer_key": item["expected_answer_key"],
        "hard_negatives": build_hard_negatives(selected, index, split_start, split_end),
        "full_image_path": source_ref,
        "roi_box": None,
        "teacher_annotation_id": None,
        "teacher_label_is_candidate": True,
        "source": {
            "dataset": DATASET_SPEC["dataset"],
            "config": DATASET_SPEC["config"],
            "split": DATASET_SPEC["source_split"],
            "row_idx": item["source_row_idx"],
            "image_field": DATASET_SPEC["image_field"],
            "question_field": DATASET_SPEC["question_field"],
            "answer_field": DATASET_SPEC["answer_field"],
            "question_index": item["question_index"],
            "question_type": item["question_type"],
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/mvp_revisions/chartqapro_v1")
    parser.add_argument("--curriculum-id", default="mvp_chartqapro_v1")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--holdout-count", type=int, default=64)
    args = parser.parse_args()

    if args.train_count != 64 or args.holdout_count != 64:
        raise RuntimeError("this revision script is intentionally fixed to actual 64/64 manifests")

    selected = collect_candidates(args.train_count + args.holdout_count)
    train_rows = [manifest_row(selected, idx, "train", args.curriculum_id) for idx in range(args.train_count)]
    holdout_rows = [
        manifest_row(selected, idx, "holdout", args.curriculum_id)
        for idx in range(args.train_count, args.train_count + args.holdout_count)
    ]

    train_keys = {row["expected_answer_key"] for row in train_rows}
    holdout_keys = {row["expected_answer_key"] for row in holdout_rows}
    summary = {
        "revision": "chartqapro_v1",
        "purpose": "replace the earlier ChartQA chart_table_cell source with ChartQAPro for an actual M4/M5 rerun",
        "minimum_per_taxonomy": {"train": args.train_count, "holdout": args.holdout_count},
        "taxonomies": {
            DATASET_SPEC["taxonomy_id"]: {
                "source_dataset": DATASET_SPEC["dataset"],
                "source_config": DATASET_SPEC["config"],
                "source_split": DATASET_SPEC["source_split"],
                "image_storage": "Dataset Viewer binary image field cached locally at runtime",
                "train_count": len(train_rows),
                "holdout_count": len(holdout_rows),
                "train_holdout_answer_overlap": sorted(train_keys & holdout_keys),
            }
        },
    }

    output_dir = ROOT / args.output_dir
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "holdout.jsonl", holdout_rows)
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train_rows)} train rows and {len(holdout_rows)} holdout rows")


if __name__ == "__main__":
    main()
