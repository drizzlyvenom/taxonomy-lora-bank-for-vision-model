# Offline-Certified Taxonomy LoRA Bank

Status: M3 actual base audit
Scope: Track A-first research repo

이 레포는 현재 Qwen 계열 VLM을 reference backbone으로 사용해 여러 vision specialist를
통합하기 위한 **offline-certified taxonomy LoRA bank** 프레임워크를 검증한다.
최종 목표는 같은 protocol을 JEPA/LeWM 계열 world-model backbone으로 옮기는 것이다.

핵심 질문은 단순하다.

```text
실패 trace를 taxonomy별 adapter-sensitive curriculum으로 컴파일하고,
학습된 LoRA 후보를 actual-only certification으로 검증한 뒤,
certified adapter만 runtime online routing에 사용할 수 있는가?
```

## Track A 원칙

Track A의 중심은 LoRA routing 자체가 아니라, routing 전에 adapter가 실제로 유효한지
검증하는 offline loop다.

```text
RouteTrace / failures
  -> taxonomy assignment
  -> adapter-sensitive curriculum
  -> candidate LoRA training
  -> actual base/correct/wrong/random certification
  -> certified AdapterCard registry
  -> online top-1 routing
```

Certification에는 proxy를 넣지 않는다.

```text
No proxy in certification.
No actual eval, no certification.
Failure is a valid result.
```

검증 기본 단위는 taxonomy별 `train 64 / holdout 64` actual run이다. Smoke test는
기본 검증 경로로 사용하지 않는다.

Foveation, OCR ROI, low-resolution survey path는 새 novelty claim이 아니다. 이 레포에서는
visual evidence cost control 또는 입력 증거 생성 모듈로만 사용한다.

## Backbone migration 원칙

Qwen2/Qwen3 기반 실험은 최종 구조가 아니라, 단일 RTX 3090에서 PEFT와 image-to-text
평가를 안정적으로 돌리기 위한 reference implementation이다.

나중에 JEPA/LeWM world-model backbone으로 바꿀 때 이식되는 것은 LoRA weight가 아니라
아래 protocol이다.

- taxonomy schema
- curriculum manifest
- AdapterCard 구조
- base/correct/wrong/random actual certification
- online RouteTrace
- offline LoRA learning loop

반대로 아래 항목은 backbone-specific으로 다시 만든다.

- LoRA weight
- target modules
- adapter slot
- latent representation
- scoring head 또는 answer head
- memory/latency profile

즉, Qwen용 adapter와 LeWM용 adapter는 서로 다른 bank로 관리한다. 같은 것은 검증 규약이고,
weight 자체가 아니다.

M12에서는 이 migration을 더 쪼개서 본다. 현재 Qwen backbone이 함께 맡는 perception과
LoRA routing을 분리하고, `perception model`은 Qwen/VLM 또는 JEPA/LeWM으로, `LoRA router`
는 oracle/taxonomy/learned/random router로 바꿔가며 같은 `64/64 actual` protocol을 반복한다.

## M0 산출물

- [프레임워크 문서](docs/00_overview/framework_ko.md)
- [검증 마일스톤](docs/10_protocols/validation_milestones_ko.md)
- [수학적 검증 프로토콜](docs/10_protocols/track_a_v2_mathematical_validation_protocol_ko.md)
- [로컬 artifact 정책](docs/10_protocols/local_artifact_policy_ko.md)
- [결과 브리프 정책](docs/20_results/README.md)

M0 통과 조건:

- README가 Track A를 메인으로 둔다.
- Foveation은 support/input-cost control 역할로만 둔다.
- Certification은 actual-only로 명시한다.

## 현재 M3 실행 구도

M3에서는 먼저 `Qwen3-VL-4B-Instruct`를 adapter 없이 실행해 base difficulty를 actual로
측정한다. 같은 holdout split에서 `Qwen3-VL-8B-Instruct`는 작은 Qwen3 + LoRA와 비교할
동세대 큰 backbone baseline으로 사용한다.

```text
Qwen3-VL-4B base
Qwen3-VL-4B + taxonomy LoRA
Qwen3-VL-8B base
```

첫 actual audit 단위는 `data/mvp/holdout.jsonl` 128개 샘플이다. 이 결과는 certification이
아니라 M5/M6 전에 task 난이도가 적절한지 확인하는 base difficulty evidence다.

현재 Qwen3-VL-4B holdout actual base audit 결과는 다음과 같다.

```text
train overall: 104/128 = 0.8125
train document_field_bind: 62/64 = 0.96875
train chart_table_cell: 42/64 = 0.65625

holdout overall: 108/128 = 0.84375
holdout document_field_bind: 61/64 = 0.953125
holdout chart_table_cell: 47/64 = 0.734375
```

따라서 `chart_table_cell`은 M5/M6 후보로 유지하고, `document_field_bind`는 현재 split이
너무 쉬우므로 harder document split 또는 더 엄격한 field-binding prompt/scoring 보강이 필요하다.

## 현재 공개 범위

이 public repo에는 문서, 스키마, 설정, 코드, 작은 예시 파일만 커밋한다. GGUF, safetensors,
checkpoint, 학습 산출물, 원본 데이터셋 같은 대용량 또는 라이선스 확인이 필요한 artifact는
로컬에만 둔다.
