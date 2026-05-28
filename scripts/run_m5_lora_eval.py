from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from run_m3_actual_base_audit import (
    build_messages,
    fetch_dataset_row,
    resolve_image_reference,
    score_prediction,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_answer(
    model: PeftModel,
    processor: AutoProcessor,
    image_url: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = build_messages(image_url, prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/curricula/chart_table_cell_holdout.jsonl")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--adapter-path", default="models/loras/chart_table_cell_r4_v1")
    parser.add_argument("--adapter-id", default="chart_table_cell_r4_v1")
    parser.add_argument("--model-id", default="qwen3_vl_4b_chart_table_cell_r4_v1")
    parser.add_argument("--output", default="results/m5/chart_table_cell_r4_v1_correct_holdout.json")
    parser.add_argument("--cache-dir", default="data/cache/m3_dataset_rows")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    adapter_path = ROOT / args.adapter_path
    output_path = ROOT / args.output
    cache_dir = ROOT / args.cache_dir
    manifest_rows = read_jsonl(manifest_path)

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        completed = {row["sample_id"]: row for row in previous.get("rows", []) if not row.get("error")}

    pending_rows = [row for row in manifest_rows if row["sample_id"] not in completed]
    print(f"prefetching dataset rows: {len(pending_rows)} pending", flush=True)
    for index, manifest_row in enumerate(pending_rows, start=1):
        fetch_dataset_row(manifest_row["source"], cache_dir)
        print(f"[prefetch {index}/{len(pending_rows)}] {manifest_row['sample_id']}", flush=True)

    print(f"loading processor: {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        ROOT / args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    print(f"loading base model: {args.model_path}", flush=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        ROOT / args.model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"loading adapter: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    run_metadata = {
        "milestone": "M5",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "adapter": args.adapter_id,
        "adapter_path": args.adapter_path,
        "manifest": args.manifest,
        "visual_policy": "qwen3_vl_fixed_pixel_budget",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "roi_source": None,
        "max_new_tokens": args.max_new_tokens,
        "scoring_policy": "normalized_exact_or_contains_numeric_only_when_expected_has_no_letters",
        "started_at_utc": started_at.isoformat(),
        "resumed_from_existing_output": bool(args.resume and output_path.exists()),
    }

    for index, manifest_row in enumerate(manifest_rows, start=1):
        sample_id = manifest_row["sample_id"]
        if sample_id in completed:
            rows.append(completed[sample_id])
            print(f"[{index}/{len(manifest_rows)}] resume {sample_id}", flush=True)
            continue

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "split": manifest_row["split"],
            "taxonomy_id": manifest_row["taxonomy_id"],
            "prompt": manifest_row["prompt"],
            "expected_answers": manifest_row["expected_answers"],
            "full_image_path": manifest_row["full_image_path"],
            "source": manifest_row["source"],
        }

        try:
            dataset_row = fetch_dataset_row(manifest_row["source"], cache_dir)
            image_url, image_size = resolve_image_reference(dataset_row, manifest_row, cache_dir)
            record["image_size"] = image_size

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start_time = time.perf_counter()
            answer = generate_answer(model, processor, image_url, manifest_row["prompt"], args.max_new_tokens)
            latency_ms = (time.perf_counter() - start_time) * 1000
            score, score_method = score_prediction(answer, manifest_row["expected_answers"])
            record.update(
                {
                    "answer_text": answer,
                    "score": score,
                    "score_method": score_method,
                    "latency_ms": round(latency_ms, 2),
                    "peak_cuda_allocated_mb": (
                        round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else None
                    ),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "answer_text": "",
                    "score": None,
                    "score_method": "runtime_error",
                    "latency_ms": None,
                    "peak_cuda_allocated_mb": None,
                    "error": repr(exc),
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        rows.append(record)
        payload = {
            "run": {**run_metadata, "updated_at_utc": datetime.now(timezone.utc).isoformat()},
            "summary": summarize(rows),
            "rows": rows,
        }
        write_json(output_path, payload)
        print(
            f"[{index}/{len(manifest_rows)}] {sample_id} score={record['score']} "
            f"method={record['score_method']} answer={record['answer_text'][:80]!r}",
            flush=True,
        )

    final_payload = {
        "run": {
            **run_metadata,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    write_json(output_path, final_payload)
    print(json.dumps(final_payload["summary"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
