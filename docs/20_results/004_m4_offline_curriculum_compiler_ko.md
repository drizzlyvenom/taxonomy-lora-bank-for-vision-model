# M4 Offline Curriculum Compiler 결과

## 1. 목적

M4의 목적은 M2 MVP manifest와 M3 base actual 결과를 받아, LoRA 학습 후보가 사용할
curriculum manifest와 이후 certification evaluation 계획으로 컴파일하는 것이다.

이번 M4에서는 새 claim을 열지 않는다. `chart_table_cell`은 M5 1순위 학습 후보로 두고,
`document_field_bind`는 현재 split이 너무 쉬우므로 재현 가능한 curriculum으로만 보존하되
강한 LoRA gain claim 후보에서는 뒤로 미룬다.

## 2. 입력과 실행 조건

```yaml
inputs:
  - data/mvp/train.jsonl
  - data/mvp/holdout.jsonl
  - results/m3/qwen3_vl_4b_base_train.json
  - results/m3/qwen3_vl_4b_base_holdout.json

teacher_mode: rule
teacher_label_is_candidate: true
base_backbone: qwen3_vl_4b_reference
base_model_id: qwen3_vl_4b_base
visual_policy: qwen3_vl_fixed_pixel_budget
roi_source: none
```

Rule annotation은 M2의 expected answer와 hard negative를 보존하는 후보 supervision이다.
M3 base output은 curriculum row에 diagnostic으로만 붙이며, certification score로 새로
사용하지 않는다.

## 3. 산출물

path | rows / content | status
--- | ---: | ---
`data/curricula/curriculum_manifest.jsonl` | 256 rows | pass
`data/curricula/teacher_annotations.jsonl` | 256 rows | pass
`data/curricula/chart_table_cell_train.jsonl` | 64 rows | pass
`data/curricula/chart_table_cell_holdout.jsonl` | 64 rows | pass
`data/curricula/document_field_bind_train.jsonl` | 64 rows | pass
`data/curricula/document_field_bind_holdout.jsonl` | 64 rows | pass
`configs/track_a/adapter_candidate_plan.yaml` | 2 candidates | pass
`configs/track_a/certification_eval_plan.yaml` | actual eval plan | pass

## 4. Adapter 후보 판정

taxonomy | adapter_id | base train | base holdout | M4 status
--- | --- | ---: | ---: | ---
`chart_table_cell` | `chart_table_cell_r4_v1` | 0.656250 | 0.734375 | `training_ready`
`document_field_bind` | `document_field_bind_r4_v1` | 0.968750 | 0.953125 | `compiled_but_deferred_for_strong_claim`

`chart_table_cell`은 base가 완전히 풀지 못한 실패 row가 충분해서 M5의 첫 correct LoRA 학습
대상으로 사용한다. `document_field_bind`는 산출물에는 남기지만, 현재 split만으로는 LoRA
gain을 주장하지 않는다.

## 5. Pass / Fail

M4는 pass다.

```yaml
pass_if:
  curriculum_rows_point_to_taxonomy: true
  train_holdout_split_is_preserved: true
  teacher_label_is_candidate_is_true: true
  certification_plan_requires_actual_eval: true
```

Certification plan은 `base_no_adapter`, `correct_lora`, `wrong_adapter`,
`random_untrained_lora` 비교를 요구한다. 모든 비교는 같은 holdout sample, 같은 visual
policy, 같은 ROI source를 사용해야 한다.

## 6. 다음 작업

M5에서는 `chart_table_cell_r4_v1`을 먼저 학습한다. 목표는 train overfit을 허용한 상태에서
correct LoRA가 base보다 실제 output을 바꿀 수 있는지 확인하는 것이다. 이후 holdout에서
gain이 보이면 M6에서 wrong/random adapter 비교로 넘어간다.

## 7. 재현 명령

```powershell
.\.venv\Scripts\python.exe scripts\compile_m4_curriculum.py
.\.venv\Scripts\python.exe scripts\check_m4_curriculum.py
.\.venv\Scripts\python.exe -m py_compile scripts\compile_m4_curriculum.py scripts\check_m4_curriculum.py
```
