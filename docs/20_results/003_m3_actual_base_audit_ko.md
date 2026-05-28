# M3 결과: Qwen3-VL-4B Actual Base Audit

Status: m3_base_audit_complete

## 1. 목적

Qwen3-VL-4B base model이 MVP train/holdout task를 너무 쉽게 풀거나 너무 못 풀지 않는지
actual evaluation으로 확인한다. 이 결과는 LoRA certification이 아니라 M5/M6 전에 task
난이도를 판정하기 위한 base difficulty evidence다.

## 2. 입력과 실행 조건

```text
model: Qwen3-VL-4B-Instruct
adapter: none
manifests:
  - data/mvp/train.jsonl
  - data/mvp/holdout.jsonl
samples: 256 total, train 128 + holdout 128
visual_policy: qwen3_vl_fixed_pixel_budget
min_pixels: 200704
max_pixels: 1003520
roi_source: none
max_new_tokens: 64
scoring: normalized exact/contains, numeric tolerance only when expected answer has no letters
result_json:
  - results/m3/qwen3_vl_4b_base_train.json
  - results/m3/qwen3_vl_4b_base_holdout.json
```

초기 `qwen3_vl_default_dynamic_resolution` 실행은 고해상도 DocVQA page에서 CUDA OOM이 발생해
validation evidence로 사용하지 않았다. 최종 결과는 fixed pixel budget으로 다시 실행한 값이다.

Dataset Viewer HTTP 429는 resume으로 재시도했고 최종 `error_count`는 0이다.

## 3. Actual score 표

```text
split | taxonomy | count | correct | score | M3 difficulty status
train | document_field_bind | 64 | 62 | 0.968750 | too_easy
train | chart_table_cell | 64 | 42 | 0.656250 | usable
train | overall | 128 | 104 | 0.812500 | pass
holdout | document_field_bind | 64 | 61 | 0.953125 | too_easy
holdout | chart_table_cell | 64 | 47 | 0.734375 | usable
holdout | overall | 128 | 108 | 0.843750 | pass_but_high
```

M3 권장 base holdout band는 `0.30 <= score <= 0.85`다. 전체 score는 0.84375로 band 안에
있지만 상단에 매우 가깝다.

## 4. 실패 샘플 요약

`document_field_bind`는 train/holdout 모두 대부분 맞혔고, 실패는 부분 답변, 과잉 답변,
비슷한 entity 혼동이었다.

```text
document_field_bind_train_000022 | expected: y. c. deveshwar | answer: K. B. Vaidyanath
document_field_bind_train_000039 | expected: GENERAL FOOD FUND, INC | answer: GENERAL FOODS FUND, INC.
document_field_bind_holdout_000021 | expected: series c no4 | answer: C No4 1971
document_field_bind_holdout_000062 | expected: 1R4F REFERENCE RESULTS / 1R4F reference | answer: 1R4F
document_field_bind_holdout_000063 | expected: AVERAGE 1R4F RESPONSES PER S9 LOT STRAIN TA100 | answer: AVERAGE 1R4F RESPONSES PER S9 LOT
```

`chart_table_cell`은 train에서 22개, holdout에서 17개를 틀렸다. 대표 실패는 인접 숫자,
비율 계산, row/column binding 혼동, series/color binding 혼동이다.

```text
chart_table_cell_train_000000 | expected: 14 | answer: 10
chart_table_cell_train_000007 | expected: Yes | answer: 95
chart_table_cell_train_000017 | expected: green line | answer: Child Labor (Boys, World, 2000-2012) (ILO)
chart_table_cell_holdout_000003 | expected: 1.051388889 | answer: 1.97
chart_table_cell_holdout_000008 | expected: 141 | answer: 126
chart_table_cell_holdout_000010 | expected: 1.684722222 | answer: 1.5384615384615385
chart_table_cell_holdout_000022 | expected: 37 | answer: 35
chart_table_cell_holdout_000024 | expected: 8 | answer: 49
chart_table_cell_holdout_000039 | expected: Agree | answer: 0
```

## 5. Pass/fail 판정

M3 base audit는 train/holdout actual run으로 완료되었다.

```yaml
actual_rows_complete: true
error_count: 0
overall_base_train_score: 0.8125
overall_base_holdout_score: 0.84375
overall_status: pass_but_high
document_field_bind_status: too_easy
chart_table_cell_status: usable
```

해석:

- `document_field_bind`는 Qwen3-VL-4B base가 이미 너무 잘 풀어서, 현재 holdout만으로는 LoRA gain을 보기 어렵다.
- `chart_table_cell`은 train 0.65625, holdout 0.734375라 M5/M6의 adapter-sensitive task로 먼저 쓰기 좋다.
- 전체 score는 M3 band 안이지만 상단에 가까워, M4 이후에는 harder document split 또는 더 엄격한 document prompt/scoring을 준비해야 한다.

## 6. 재현 명령

```text
.\.venv\Scripts\python.exe scripts\run_m3_actual_base_audit.py --manifest data/mvp/train.jsonl --model-path models/qwen/Qwen3-VL-4B-Instruct --model-id qwen3_vl_4b_base --output results/m3/qwen3_vl_4b_base_train.json --resume
.\.venv\Scripts\python.exe scripts\run_m3_actual_base_audit.py --manifest data/mvp/holdout.jsonl --model-path models/qwen/Qwen3-VL-4B-Instruct --model-id qwen3_vl_4b_base --output results/m3/qwen3_vl_4b_base_holdout.json --resume
```

검증:

```text
.\.venv\Scripts\python.exe scripts\check_m3_actual_base_audit.py
```
