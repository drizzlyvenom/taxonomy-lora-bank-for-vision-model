# Offline-Certified Taxonomy LoRA Bank

Status: M0 clean repo scaffold  
Scope: Track A-first research repo

이 레포는 shared VLM backbone 위에서 여러 vision specialist를 통합하기 위한
**offline-certified taxonomy LoRA bank** 프레임워크를 검증한다.

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

Foveation, OCR ROI, low-resolution survey path는 새 novelty claim이 아니다. 이 레포에서는
visual evidence cost control 또는 입력 증거 생성 모듈로만 사용한다.

## M0 산출물

- [프레임워크 문서](docs/00_overview/framework_ko.md)
- [검증 마일스톤](docs/10_protocols/validation_milestones_ko.md)
- [로컬 artifact 정책](docs/10_protocols/local_artifact_policy_ko.md)
- [결과 브리프 정책](docs/20_results/README.md)

M0 통과 조건:

- README가 Track A를 메인으로 둔다.
- Foveation은 support/input-cost control 역할로만 둔다.
- Certification은 actual-only로 명시한다.

## 현재 공개 범위

이 public repo에는 문서, 스키마, 설정, 코드, 작은 예시 파일만 커밋한다. GGUF, safetensors,
checkpoint, 학습 산출물, 원본 데이터셋 같은 대용량 또는 라이선스 확인이 필요한 artifact는
로컬에만 둔다.

