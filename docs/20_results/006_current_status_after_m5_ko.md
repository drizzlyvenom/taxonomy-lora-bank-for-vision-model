# 현재 상태 요약: M5 first pass 이후

Status: paused_after_m5_soft_pass

## 1. 목적

이 문서는 M5 first pass 이후 잠시 멈추기 전에, 현재 검증 상태와 다음 선택지를 짧게 고정한다.
새 protocol을 추가하거나 milestone 본문을 바꾸지는 않는다.

## 2. 완료된 범위

milestone | 상태 | 요약
--- | --- | ---
M0 | closed | Track A-first repo 구조와 actual-only 원칙 정리
M1 | closed | schema / contract 예시와 checker 통과
M2 | closed | MVP `train 64 / holdout 64` manifest 생성
M3 | closed | Qwen3-VL-4B no-adapter base actual audit 완료
M4 | closed | curriculum manifest, teacher annotations, adapter/eval plan 생성
M5 | soft_pass | `chart_table_cell_r4_v1` train gain 확인, holdout gain은 없음

## 3. 핵심 결과

taxonomy | split | base | correct LoRA | gain | 판단
--- | --- | ---: | ---: | ---: | ---
`chart_table_cell` | train | 0.656250 | 0.765625 | +0.109375 | 학습 신호 있음
`chart_table_cell` | holdout | 0.734375 | 0.718750 | -0.015625 | 일반화 claim 보류
`document_field_bind` | train | 0.968750 | not run | n/a | base too high
`document_field_bind` | holdout | 0.953125 | not run | n/a | base too high

M5 결과는 LoRA가 output을 바꿀 수 있다는 진단 근거로는 유효하지만, M6의
correct-vs-wrong/random certification으로 바로 넘어가기에는 약하다.

## 4. 현재 판단

`chart_table_cell`은 현재 taxonomy가 너무 넓거나 train/holdout 분포가 어긋났을 수 있다.
따라서 다음 단계에서는 단순히 M6으로 넘기기보다 taxonomy 설계 또는 학습 조건을 먼저 다시 본다.

`taxonomy fitness optimization`은 논의만 했고 아직 문서 protocol로 채택하지 않았다. 채택한다면
runtime loop가 아니라 offline-only registry compiler 단계로 제한한다.

```yaml
runtime_allowed:
  - certified AdapterCard registry lookup
  - top-1 adapter selection
  - fallback to base_no_adapter
  - failure trace enqueue

runtime_forbidden:
  - taxonomy split or merge
  - convex taxonomy optimization
  - LoRA training
  - uncertified adapter routing
  - multi-adapter search
```

## 5. 다음 선택지

1. `chart_table_cell_r4_v1`을 rank 8 또는 낮은 learning rate로 다시 actual 64/64 M5 실행
2. `chart_table_cell`을 하위 taxonomy 후보로 쪼갠 뒤 M4b 설계 문서 작성
3. OCR-VQA 또는 ChartQAPro 후보를 추가해 MVP taxonomy 일부 교체 가능성 검토

현재는 여기서 멈춘다. 다음 작업을 시작하기 전에는 이 문서와 `005_m5_single_lora_learns_ko.md`를
먼저 확인한다.

## 6. 참고 산출물

```text
docs/20_results/003_m3_actual_base_audit_ko.md
docs/20_results/004_m4_offline_curriculum_compiler_ko.md
docs/20_results/005_m5_single_lora_learns_ko.md
results/m5/chart_table_cell_r4_v1_m5_summary.json
```
