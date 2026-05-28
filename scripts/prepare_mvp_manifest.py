from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

TAXONOMIES = {
    "document_field_bind": {
        "dataset": "lmms-lab/DocVQA",
        "config": "DocVQA",
        "source_split": "validation",
        "hf_config": "DocVQA",
        "taxonomy": {
            "domain": "document",
            "evidence_type": "field_value",
            "operation": "bind_label_to_value",
            "failure_mode": "distractor_confusion",
        },
        "allowed_question_types": {"form", "layout", "table/list"},
        "answer_field": "answers",
        "question_field": "question",
    },
    "chart_table_cell": {
        "dataset": "lmms-lab/ChartQA",
        "config": "default",
        "source_split": "test",
        "hf_config": "default",
        "taxonomy": {
            "domain": "chart",
            "evidence_type": "table_cell",
            "operation": "locate",
            "failure_mode": "label_value_mismatch",
        },
        "allowed_question_types": None,
        "answer_field": "answer",
        "question_field": "question",
    },
}


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{DATASET_ROWS_ENDPOINT}?{query}", timeout=90) as response:
        return json.load(response)


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def answer_list(row: dict[str, Any], answer_field: str) -> list[str]:
    value = row.get(answer_field)
    if isinstance(value, list):
        answers = [str(item).strip() for item in value if str(item).strip()]
    elif value is None:
        answers = []
    else:
        answers = [str(value).strip()] if str(value).strip() else []
    return answers


def row_matches_taxonomy(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    allowed_types = spec["allowed_question_types"]
    if not allowed_types:
        return True
    row_types = set(row.get("question_types") or [])
    return bool(row_types & allowed_types)


def collect_candidates(taxonomy_id: str, needed: int) -> list[dict[str, Any]]:
    spec = TAXONOMIES[taxonomy_id]
    selected: list[dict[str, Any]] = []
    seen_answer_keys: set[str] = set()
    offset = 0
    page_size = 100

    while len(selected) < needed:
        payload = fetch_rows(spec["dataset"], spec["config"], spec["source_split"], offset, page_size)
        rows = payload.get("rows", [])
        if not rows:
            raise RuntimeError(f"not enough rows for {taxonomy_id}: got {len(selected)}, need {needed}")

        for item in rows:
            row = item["row"]
            if not row_matches_taxonomy(row, spec):
                continue

            answers = answer_list(row, spec["answer_field"])
            if not answers:
                continue

            answer_key = normalize_answer(answers[0])
            if not answer_key or answer_key in seen_answer_keys:
                continue

            seen_answer_keys.add(answer_key)
            selected.append(
                {
                    "source_row_idx": item["row_idx"],
                    "source_row": row,
                    "expected_answers": answers,
                    "expected_answer_key": answer_key,
                }
            )
            if len(selected) >= needed:
                break

        offset += page_size
        if offset >= payload.get("num_rows_total", offset):
            if len(selected) < needed:
                raise RuntimeError(f"exhausted {taxonomy_id}: got {len(selected)}, need {needed}")

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


def manifest_row(taxonomy_id: str, selected: list[dict[str, Any]], index: int, split: str) -> dict[str, Any]:
    spec = TAXONOMIES[taxonomy_id]
    item = selected[index]
    row = item["source_row"]
    split_start = 0 if split == "train" else 64
    split_end = 64 if split == "train" else 128
    source_ref = (
        f"hf://datasets/{spec['dataset']}/{spec['config']}/"
        f"{spec['source_split']}/{item['source_row_idx']}/image"
    )

    return {
        "curriculum_id": f"mvp_{taxonomy_id}_v1",
        "sample_id": f"{taxonomy_id}_{split}_{index - split_start:06d}",
        "split": split,
        "taxonomy_id": taxonomy_id,
        "taxonomy": spec["taxonomy"],
        "prompt": row[spec["question_field"]],
        "expected_answers": item["expected_answers"],
        "expected_answer_key": item["expected_answer_key"],
        "hard_negatives": build_hard_negatives(selected, index, split_start, split_end),
        "full_image_path": source_ref,
        "roi_box": None,
        "teacher_annotation_id": None,
        "teacher_label_is_candidate": True,
        "source": {
            "dataset": spec["dataset"],
            "config": spec["config"],
            "split": spec["source_split"],
            "row_idx": item["source_row_idx"],
            "image_field": "image",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/mvp")
    args = parser.parse_args()

    train_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "minimum_per_taxonomy": {"train": 64, "holdout": 64},
        "taxonomies": {},
    }

    for taxonomy_id in TAXONOMIES:
        selected = collect_candidates(taxonomy_id, needed=128)
        taxonomy_train = [manifest_row(taxonomy_id, selected, idx, "train") for idx in range(64)]
        taxonomy_holdout = [manifest_row(taxonomy_id, selected, idx, "holdout") for idx in range(64, 128)]
        train_rows.extend(taxonomy_train)
        holdout_rows.extend(taxonomy_holdout)

        train_keys = {row["expected_answer_key"] for row in taxonomy_train}
        holdout_keys = {row["expected_answer_key"] for row in taxonomy_holdout}
        summary["taxonomies"][taxonomy_id] = {
            "source_dataset": TAXONOMIES[taxonomy_id]["dataset"],
            "source_config": TAXONOMIES[taxonomy_id]["config"],
            "source_split": TAXONOMIES[taxonomy_id]["source_split"],
            "train_count": len(taxonomy_train),
            "holdout_count": len(taxonomy_holdout),
            "train_holdout_answer_overlap": sorted(train_keys & holdout_keys),
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
