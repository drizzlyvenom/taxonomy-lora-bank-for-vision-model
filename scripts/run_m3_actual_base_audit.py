from __future__ import annotations

import argparse
import json
import math
import re
import time
import unicodedata
import urllib.parse
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


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


def fetch_dataset_row(source: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    cache_key = (
        f"{source['dataset'].replace('/', '__')}__{source['config']}__"
        f"{source['split']}__{source['row_idx']}.json"
    )
    cache_path = cache_dir / cache_key
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    query = urllib.parse.urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
            "offset": source["row_idx"],
            "length": 1,
        }
    )
    url = f"{DATASET_ROWS_ENDPOINT}?{query}"
    last_error: Exception | None = None
    backoff_seconds = [5, 15, 45, 90, 120]
    for attempt, wait_seconds in enumerate(backoff_seconds, start=1):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                payload = json.load(response)
            rows = payload.get("rows") or []
            if not rows:
                raise RuntimeError(f"Dataset Viewer returned no rows for {source}")
            row = rows[0]
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            return row
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                time.sleep(wait_seconds)
            else:
                time.sleep(min(wait_seconds, 10))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(wait_seconds)

    raise RuntimeError(f"failed to fetch dataset row after retries: {source}") from last_error


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^0-9a-z가-힣]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def numbers(value: str) -> list[float]:
    found: list[float] = []
    for match in re.findall(r"-?\d+(?:\.\d+)?", value.replace(",", "")):
        try:
            found.append(float(match))
        except ValueError:
            pass
    return found


def numeric_match(prediction: str, expected: str) -> bool:
    normalized_expected = normalize_text(expected)
    if re.search(r"[a-z가-힣]", normalized_expected):
        return False

    predicted_numbers = numbers(prediction)
    expected_numbers = numbers(expected)
    if not predicted_numbers or not expected_numbers:
        return False

    for exp in expected_numbers:
        for pred in predicted_numbers:
            tolerance = max(0.01, abs(exp) * 0.01)
            if math.isclose(pred, exp, abs_tol=tolerance, rel_tol=0.01):
                return True
    return False


def score_prediction(prediction: str, expected_answers: list[str]) -> tuple[float, str]:
    if not prediction.strip():
        return 0.0, "empty_output"

    normalized_prediction = normalize_text(prediction)
    for expected in expected_answers:
        normalized_expected = normalize_text(expected)
        if normalized_expected and normalized_prediction == normalized_expected:
            return 1.0, "normalized_exact"

    for expected in expected_answers:
        normalized_expected = normalize_text(expected)
        if normalized_expected and normalized_expected in normalized_prediction:
            return 1.0, "normalized_contains"

    for expected in expected_answers:
        if numeric_match(prediction, expected):
            return 1.0, "numeric_tolerance"

    return 0.0, "no_match"


def build_messages(image_url: str, prompt: str) -> list[dict[str, Any]]:
    text = (
        f"{prompt}\n\n"
        "Answer with only the final answer text. Do not explain. "
        "If the answer is a number or code, return only that number or code."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": text},
            ],
        }
    ]


def generate_answer(
    model: Qwen3VLForConditionalGeneration,
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("score") is not None]
    by_taxonomy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_taxonomy[row["taxonomy_id"]].append(row)

    per_taxonomy: dict[str, Any] = {}
    for taxonomy_id, items in sorted(by_taxonomy.items()):
        score_sum = sum(float(item["score"]) for item in items)
        per_taxonomy[taxonomy_id] = {
            "count": len(items),
            "score": score_sum / len(items) if items else None,
            "failure_types": dict(Counter(item["score_method"] for item in items)),
        }

    score_sum = sum(float(item["score"]) for item in scored)
    return {
        "count": len(scored),
        "score": score_sum / len(scored) if scored else None,
        "per_taxonomy": per_taxonomy,
        "error_count": sum(1 for row in rows if row.get("error")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/mvp/holdout.jsonl")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model-id", default="qwen3_vl_4b_base")
    parser.add_argument("--output", default="results/m3/qwen3_vl_4b_base_holdout.json")
    parser.add_argument("--cache-dir", default="data/cache/m3_dataset_rows")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    output_path = ROOT / args.output
    cache_dir = ROOT / args.cache_dir
    manifest_rows = read_jsonl(manifest_path)
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        completed = {row["sample_id"]: row for row in previous.get("rows", []) if not row.get("error")}

    pending_rows = [row for row in manifest_rows if row["sample_id"] not in completed]
    print(f"prefetching dataset rows: {len(pending_rows)} pending", flush=True)
    for index, manifest_row in enumerate(pending_rows, start=1):
        fetch_dataset_row(manifest_row["source"], cache_dir)
        print(f"[prefetch {index}/{len(pending_rows)}] {manifest_row['sample_id']}", flush=True)

    print(f"loading model: {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        ROOT / args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        ROOT / args.model_path,
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    run_metadata = {
        "milestone": "M3",
        "model_id": args.model_id,
        "model_path": args.model_path,
        "adapter": None,
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
            image_field = manifest_row["source"].get("image_field", "image")
            image_info = dataset_row["row"][image_field]
            image_url = image_info["src"]
            record["image_size"] = {
                "width": image_info.get("width"),
                "height": image_info.get("height"),
            }

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
        summary = summarize(rows)
        payload = {
            "run": {**run_metadata, "updated_at_utc": datetime.now(timezone.utc).isoformat()},
            "summary": summary,
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
