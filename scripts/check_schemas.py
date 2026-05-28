from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEXT_REQUIREMENTS = {
    "schemas/adapter_card_v3.example.yaml": [
        "schema_name: AdapterCardV3",
        "adapter_id:",
        "base_backbone:",
        "backbone_type:",
        "taxonomy:",
        "weights:",
        "adapter_slot:",
        "actual_certification:",
        "actual_fields_complete:",
        "same_holdout_samples:",
        "same_visual_policy:",
        "same_roi_source:",
    ],
    "schemas/route_trace.example.yaml": [
        "schema_name: RouteTrace",
        "trace_id:",
        "sample_id:",
        "perception:",
        "perception_artifact_id:",
        "router:",
        "selected_adapter_id:",
        "execution:",
        "output:",
        "memory:",
    ],
    "schemas/backbone_contract.example.yaml": [
        "schema_name: BackboneContract",
        "backbone_id:",
        "backbone_type:",
        "inputs:",
        "outputs:",
        "adapter_slots:",
        "certification_supported:",
        "minimum_train_per_taxonomy: 64",
        "minimum_holdout_per_taxonomy: 64",
    ],
    "schemas/perception_contract.example.yaml": [
        "schema_name: PerceptionContract",
        "perception_model_id:",
        "inputs:",
        "outputs:",
        "visual_evidence_summary",
        "latent_state",
        "evidence_confidence",
        "minimum_train_per_taxonomy: 64",
        "minimum_holdout_per_taxonomy: 64",
        "no_smoke_validation: true",
    ],
    "schemas/router_contract.example.yaml": [
        "schema_name: RouterContract",
        "router_model_id:",
        "inputs:",
        "outputs:",
        "certified_adapter_registry",
        "selected_adapter_id",
        "route_confidence",
        "requires_actual_certified_adapters: 2",
        "no_smoke_validation: true",
    ],
    "schemas/actual_certification_result.example.yaml": [
        "schema_name: ActualCertificationResult",
        "adapter_id:",
        "base_backbone:",
        "adapter_slot:",
        "holdout_samples: 64",
        "scores:",
        "margins:",
        "gates:",
        "actual_fields_complete: true",
        "same_holdout_samples: true",
        "same_visual_policy: true",
        "same_roi_source: true",
        "status: actual_certified",
    ],
}

JSONL_REQUIREMENTS = {
    "schemas/curriculum_manifest.example.jsonl": [
        "curriculum_id",
        "sample_id",
        "split",
        "taxonomy",
        "prompt",
        "expected_answers",
        "hard_negatives",
        "full_image_path",
        "teacher_label_is_candidate",
    ],
    "schemas/teacher_annotation.example.jsonl": [
        "teacher_annotation_id",
        "sample_id",
        "teacher_model",
        "annotation_source",
        "taxonomy",
        "failure_mode_hypothesis",
        "hard_negatives",
        "teacher_label_is_candidate",
    ],
}

FORBIDDEN_TOKENS = [
    "proxy_certified",
    "mixed_actual_proxy_score",
    "base_score_from_heuristic",
    "wrong_score_from_base_minus_epsilon",
    "difficulty_proxy_as_base_score",
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def read_required_file(relative_path: str) -> str:
    path = ROOT / relative_path
    if not path.is_file():
        fail(f"missing required file: {relative_path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        fail(f"empty required file: {relative_path}")
    return text


def check_text_examples() -> None:
    for relative_path, required_tokens in TEXT_REQUIREMENTS.items():
        text = read_required_file(relative_path)
        for token in required_tokens:
            if token not in text:
                fail(f"{relative_path} missing token: {token}")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                fail(f"{relative_path} contains forbidden token: {token}")


def check_jsonl_examples() -> None:
    for relative_path, required_keys in JSONL_REQUIREMENTS.items():
        text = read_required_file(relative_path)
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{relative_path}:{line_number} invalid JSONL: {exc}")
            rows.append(row)
            for key in required_keys:
                if key not in row:
                    fail(f"{relative_path}:{line_number} missing key: {key}")
            if row.get("teacher_label_is_candidate") is not True:
                fail(f"{relative_path}:{line_number} teacher_label_is_candidate must be true")
        if not rows:
            fail(f"{relative_path} has no JSONL rows")


def check_independent_contracts() -> None:
    perception = read_required_file("schemas/perception_contract.example.yaml")
    router = read_required_file("schemas/router_contract.example.yaml")

    if "selected_adapter_id" in perception:
        fail("PerceptionContract must not select an adapter")
    if "image\n" in router or "- image" in router:
        fail("RouterContract must consume perception output, not raw image input")
    if "visual_evidence_summary_or_latent_state" not in router:
        fail("RouterContract must accept perception output")


def main() -> None:
    check_text_examples()
    check_jsonl_examples()
    check_independent_contracts()
    print("M1 schema checks passed.")


if __name__ == "__main__":
    main()
