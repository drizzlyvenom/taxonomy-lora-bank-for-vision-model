# M5 Gate 1 — Single LoRA Learns 결과

## 1. 목적

M5의 목적은 taxonomy별 correct LoRA가 실제로 output을 바꾸고 학습 신호를 먹는지 확인하는
것이다. 이번 실행은 M4에서 1순위로 둔 `chart_table_cell`에 대해 먼저 진행했다.

`document_field_bind`는 M3에서 base score가 너무 높았기 때문에 이번 M5 first pass에서는
학습하지 않았다. 해당 taxonomy는 harder split 또는 replacement 후보 검증 후 다시 M5에
올리는 것이 안전하다.

## 2. 입력과 실행 조건

```yaml
taxonomy_id: chart_table_cell
adapter_id: chart_table_cell_r4_v1
base_backbone: qwen3_vl_4b_reference
base_model_path: models/qwen/Qwen3-VL-4B-Instruct
train_manifest: data/curricula/chart_table_cell_train.jsonl
holdout_manifest: data/curricula/chart_table_cell_holdout.jsonl
train_rows: 64
holdout_rows: 64
label_mask_mode: answer_only
target_modules:
  - q_proj
  - v_proj
rank: 4
alpha: 8
steps: 120
learning_rate: 0.0002
visual_policy: qwen3_vl_fixed_pixel_budget
roi_source: none
```

LoRA는 `language_model.layers.*.self_attn.q_proj/v_proj`에 붙었다. Trainable parameter는
1,474,560개이며 전체 parameter 대비 약 0.0332%다.

## 3. Actual score

taxonomy | split | base_no_adapter | correct_lora | gain | status
--- | --- | ---: | ---: | ---: | ---
`chart_table_cell` | train | 0.656250 | 0.765625 | +0.109375 | pass
`chart_table_cell` | holdout | 0.734375 | 0.718750 | -0.015625 | no_gain

Train score는 base보다 올랐으므로 학습 신호는 확인된다. Holdout score는 base보다
1개 sample 차이만큼 낮아서 strong pass는 아니다.

## 4. Pass / Fail

M5 first pass 판정은 `soft_pass`다.

```yaml
gates:
  correct_lora_train_score_gt_actual_base_train_score: true
  correct_lora_holdout_score_gte_actual_base_holdout_score: false
  train_overfit_allowed_for_first_diagnosis: true
  holdout_no_gain_failure_reason_recorded: true

failure_reason: holdout_no_gain_after_train_gain
```

이 결과는 LoRA가 task를 배울 수 있다는 진단 근거로는 충분하지만, M6 certification으로 바로
넘기기에는 약하다. Correct LoRA가 wrong/random보다 낫다는 claim도 아직 열지 않는다.

## 5. 실행 중 이슈

Holdout 평가 첫 실행에서 Dataset Viewer cache에 남은 오래된 image URL 때문에 33개 row가
image decode error를 냈다. 실패 row의 ignored cache file을 삭제하고 같은 manifest/output에
`--resume`으로 재실행했으며, 최종 결과는 64 rows, error_count 0이다.

## 6. 다음 수정 방향

다음 선택지는 두 가지다.

1. `chart_table_cell`에 대해 rank 8 또는 더 낮은 learning rate로 M5를 한 번 더 실행한다.
2. `OCR-VQA`나 `ChartQAPro` 후보를 추가해 base 난이도와 holdout 일반화가 더 좋은 taxonomy를 찾는다.

현재 결과만으로는 데이터셋을 즉시 교체할 필요까지는 없지만, holdout gain이 음수라서 M6
certification 전에 한 번 더 학습 조건 또는 dataset 후보를 점검하는 편이 안전하다.

## 7. 산출물

```text
models/loras/chart_table_cell_r4_v1
results/m5/chart_table_cell_r4_v1_train_log.json
results/m5/chart_table_cell_r4_v1_correct_train.json
results/m5/chart_table_cell_r4_v1_correct_holdout.json
results/m5/chart_table_cell_r4_v1_m5_summary.json
```

`models/loras/chart_table_cell_r4_v1`은 로컬 artifact이며 git에 커밋하지 않는다.

## 8. 재현 명령

```powershell
.\.venv\Scripts\python.exe scripts\train_m5_single_lora.py --adapter-id chart_table_cell_r4_v1 --taxonomy-id chart_table_cell --train-manifest data/curricula/chart_table_cell_train.jsonl --output-dir models/loras/chart_table_cell_r4_v1 --metadata-output results/m5/chart_table_cell_r4_v1_train_log.json --steps 120 --learning-rate 0.0002 --rank 4 --alpha 8 --log-every 5
.\.venv\Scripts\python.exe scripts\run_m5_lora_eval.py --manifest data/curricula/chart_table_cell_train.jsonl --adapter-path models/loras/chart_table_cell_r4_v1 --adapter-id chart_table_cell_r4_v1 --model-id qwen3_vl_4b_chart_table_cell_r4_v1 --output results/m5/chart_table_cell_r4_v1_correct_train.json --resume
.\.venv\Scripts\python.exe scripts\run_m5_lora_eval.py --manifest data/curricula/chart_table_cell_holdout.jsonl --adapter-path models/loras/chart_table_cell_r4_v1 --adapter-id chart_table_cell_r4_v1 --model-id qwen3_vl_4b_chart_table_cell_r4_v1 --output results/m5/chart_table_cell_r4_v1_correct_holdout.json --resume
.\.venv\Scripts\python.exe scripts\check_m5_single_lora.py
```
