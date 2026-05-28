# MVP 검토: M3 이후 상태

Status: after_m3_review

## 1. 검토 기준

MVP는 milestone 문서의 기본 단위인 taxonomy별 `train 64 / holdout 64` actual run을 기준으로
검토한다. 이번 검토는 Qwen3-VL-4B base audit 결과만 사용하며, LoRA 성능이나 router utility를
아직 주장하지 않는다.

## 2. 현재 MVP 상태

```text
taxonomy | train base | holdout base | status
document_field_bind | 0.968750 | 0.953125 | too_easy
chart_table_cell | 0.656250 | 0.734375 | usable
overall | 0.812500 | 0.843750 | pass_but_high
```

M3 기준으로 전체 holdout score는 허용 band `0.30 <= score <= 0.85` 안에 있다. 다만
`document_field_bind`가 train/holdout 모두 0.95 이상이라 LoRA gain을 보기에는 현재 split이
너무 쉽다.

## 3. Taxonomy별 판단

`chart_table_cell`은 MVP의 1순위 adapter-sensitive taxonomy로 유지한다. train 0.65625,
holdout 0.734375라 base가 충분히 틀리고, 실패 유형도 table/chart value lookup, ratio/value
혼동, row/column binding 혼동으로 LoRA curriculum을 만들기 좋다.

`document_field_bind`는 현재 그대로 M5/M6의 핵심 gain claim에 쓰기 어렵다. base가 이미 너무
강하므로 correct LoRA가 개선할 여지가 작고, Gate 1/2에서 `base_too_high` 또는
`no_gain_vs_base` 위험이 크다.

## 4. M4로 넘길 작업

```yaml
keep_for_m4:
  - chart_table_cell train/holdout 64/64
  - Qwen3-VL-4B base train/holdout actual score
  - failure rows from chart_table_cell as curriculum candidates

revise_before_strong_claim:
  - document_field_bind harder split
  - stricter document field-binding prompts
  - exact field label/value binding checks
  - distractor-heavy document samples

do_not_claim_yet:
  - LoRA improves base
  - correct LoRA beats wrong/random
  - router utility
  - 4B+LoRA beats 8B base
```

## 5. 결론

MVP는 계속 진행 가능하다. 다만 M4/M5의 우선순위는 `chart_table_cell`을 먼저 닫고,
`document_field_bind`는 harder subset을 보강한 뒤 LoRA 학습 후보로 다시 올리는 것이다.
