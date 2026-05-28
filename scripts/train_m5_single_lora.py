from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from run_m3_actual_base_audit import build_messages, fetch_dataset_row


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


def build_training_messages(image_url: str, prompt: str, answer: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    user_messages = build_messages(image_url, prompt)
    full_messages = [
        user_messages[0],
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]
    return user_messages, full_messages


def encode_answer_only(
    processor: AutoProcessor,
    image_url: str,
    prompt: str,
    answer: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    user_messages, full_messages = build_training_messages(image_url, prompt, answer)
    prompt_inputs = processor.apply_chat_template(
        user_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_inputs = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_length = prompt_inputs["input_ids"].shape[-1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    if torch.all(labels == -100):
        raise RuntimeError("answer-only label mask produced no trainable labels")

    full_inputs["labels"] = labels
    return {key: value.to(device) for key, value in full_inputs.items()}


def prefetch_image_urls(rows: list[dict[str, Any]], cache_dir: Path) -> dict[str, str]:
    urls: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        dataset_row = fetch_dataset_row(row["source"], cache_dir)
        image_field = row["source"].get("image_field", "image")
        urls[row["sample_id"]] = dataset_row["row"][image_field]["src"]
        print(f"[prefetch {index}/{len(rows)}] {row['sample_id']}", flush=True)
    return urls


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    trainable = 0
    total = 0
    trainable_names: list[str] = []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            if len(trainable_names) < 20:
                trainable_names.append(name)
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total if total else 0.0,
        "trainable_name_examples": trainable_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-id", default="chart_table_cell_r4_v1")
    parser.add_argument("--taxonomy-id", default="chart_table_cell")
    parser.add_argument("--train-manifest", default="data/curricula/chart_table_cell_train.jsonl")
    parser.add_argument("--model-path", default="models/qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output-dir", default="models/loras/chart_table_cell_r4_v1")
    parser.add_argument("--metadata-output", default="results/m5/chart_table_cell_r4_v1_train_log.json")
    parser.add_argument("--cache-dir", default="data/cache/m3_dataset_rows")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    manifest_path = ROOT / args.train_manifest
    output_dir = ROOT / args.output_dir
    metadata_output = ROOT / args.metadata_output
    cache_dir = ROOT / args.cache_dir
    train_rows = [row for row in read_jsonl(manifest_path) if row["taxonomy_id"] == args.taxonomy_id]
    if len(train_rows) != 64:
        raise RuntimeError(f"expected 64 train rows for {args.taxonomy_id}, got {len(train_rows)}")

    started_at = datetime.now(timezone.utc)
    run_metadata: dict[str, Any] = {
        "milestone": "M5",
        "adapter_id": args.adapter_id,
        "taxonomy_id": args.taxonomy_id,
        "base_model_path": args.model_path,
        "train_manifest": args.train_manifest,
        "output_dir": args.output_dir,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "label_mask_mode": "answer_only",
        "visual_policy": "qwen3_vl_fixed_pixel_budget",
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "roi_source": None,
        "seed": args.seed,
        "started_at_utc": started_at.isoformat(),
    }
    write_json(metadata_output, {"run": run_metadata, "steps": []})

    print(f"prefetching {len(train_rows)} training rows", flush=True)
    image_urls = prefetch_image_urls(train_rows, cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading processor: {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(
        ROOT / args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    print(f"loading model on {device}: {args.model_path}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        ROOT / args.model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.train()
    parameter_summary = trainable_parameter_summary(model)
    print(json.dumps(parameter_summary, ensure_ascii=False, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    step_logs: list[dict[str, Any]] = []
    losses: list[float] = []
    start_time = time.perf_counter()

    for step in range(1, args.steps + 1):
        row = train_rows[(step - 1) % len(train_rows)]
        answer = str(row["expected_answers"][0])
        inputs = encode_answer_only(processor, image_urls[row["sample_id"]], row["prompt"], answer, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**inputs)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - start_time
            log_row = {
                "step": step,
                "sample_id": row["sample_id"],
                "loss": loss_value,
                "elapsed_seconds": round(elapsed, 2),
                "peak_cuda_allocated_mb": (
                    round(torch.cuda.max_memory_allocated() / (1024**2), 2) if torch.cuda.is_available() else None
                ),
            }
            step_logs.append(log_row)
            write_json(
                metadata_output,
                {
                    "run": {
                        **run_metadata,
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "parameter_summary": parameter_summary,
                        "loss_first": losses[0],
                        "loss_latest": losses[-1],
                        "loss_min": min(losses),
                    },
                    "steps": step_logs,
                },
            )
            print(json.dumps(log_row, ensure_ascii=False, sort_keys=True), flush=True)

    print(f"saving adapter: {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    finished_at = datetime.now(timezone.utc)
    write_json(
        metadata_output,
        {
            "run": {
                **run_metadata,
                "finished_at_utc": finished_at.isoformat(),
                "parameter_summary": parameter_summary,
                "loss_first": losses[0],
                "loss_latest": losses[-1],
                "loss_min": min(losses),
                "loss_mean": sum(losses) / len(losses),
                "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
            },
            "steps": step_logs,
        },
    )
    print("training complete", flush=True)


if __name__ == "__main__":
    main()
