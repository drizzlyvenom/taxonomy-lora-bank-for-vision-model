# ChartQAPro M4/M5 재실행 결과

Status: M5 revision completed, M6 not opened

Update note: 이후 정확도와 VRAM claim을 함께 재검토한 결과, 현재 Qwen3-VL-4B
same-backbone LoRA 구조는 `LoRA routing reduces VRAM` claim을 지지하지 않는 것으로
기록했다. 부정 결과 정리는 `docs/20_results/008_negative_accuracy_vram_claim_review_ko.md`를
본다.

## 1. 목적

이번 실행의 목적은 큰 구조 변경 없이 `chart_table_cell` 데이터셋 후보를 기존 ChartQA에서
ChartQAPro로 바꿔, 같은 `64/64 actual` 조건에서 M4/M5를 다시 확인하는 것이다.

기존 M5 first pass는 train gain은 있었지만 holdout에서 base보다 낮았다. 이번 revision은
데이터셋 분포를 바꿨을 때 학습 신호와 holdout delta가 어떻게 달라지는지 확인한다.

## 2. 입력과 실행 조건

```yaml
revision: chartqapro_v1
taxonomy_id: chart_table_cell
source_dataset: ahmed-masry/ChartQAPro
source_config: default
source_split: test
train_rows: 64
holdout_rows: 64
adapter_id: chartqapro_table_cell_r4_v1
base_model_path: models/qwen/Qwen3-VL-4B-Instruct
label_mask_mode: answer_only
target_modules:
  - q_proj
  - v_proj
rank: 4
alpha: 8
steps: 120
visual_policy: qwen3_vl_fixed_pixel_budget
roi_source: none
```

ChartQAPro의 Dataset Viewer `image` field는 URL이 아니라 base64 binary image로 내려온다.
실행 스크립트는 이를 `data/cache/chartqapro_v1_dataset_rows/images/` 아래 로컬 이미지로
풀어 Qwen processor에 전달한다. `data/cache/`는 git에 커밋하지 않는다.

## 3. M3 base actual

taxonomy | split | base_no_adapter | correct / total | error_count
--- | --- | ---: | ---: | ---:
`chart_table_cell` | train | 0.437500 | 28 / 64 | 0
`chart_table_cell` | holdout | 0.359375 | 23 / 64 | 0

ChartQAPro 후보는 기존 ChartQA split보다 base 난이도가 높다. 따라서 학습 여지는 더 크지만,
그만큼 정답 분포와 scoring caveat를 더 조심해서 봐야 한다.

## 4. M4 compile

path | rows / content | status
--- | ---: | ---
`data/curricula_revisions/chartqapro_v1/curriculum_manifest.jsonl` | 128 rows | pass
`data/curricula_revisions/chartqapro_v1/teacher_annotations.jsonl` | 128 rows | pass
`data/curricula_revisions/chartqapro_v1/chart_table_cell_train.jsonl` | 64 rows | pass
`data/curricula_revisions/chartqapro_v1/chart_table_cell_holdout.jsonl` | 64 rows | pass
`configs/track_a/revisions/chartqapro_v1/adapter_candidate_plan.yaml` | 1 candidate | pass
`configs/track_a/revisions/chartqapro_v1/certification_eval_plan.yaml` | actual eval plan | pass

M4 revision은 기존 milestone 본문을 바꾸지 않고 별도 revision 경로에 산출물을 둔다.

## 5. M5 actual score

taxonomy | split | base_no_adapter | correct_lora | gain | 판단
--- | --- | ---: | ---: | ---: | ---
`chart_table_cell` | train | 0.437500 | 0.812500 | +0.375000 | 학습 신호 강함
`chart_table_cell` | holdout | 0.359375 | 0.359375 | +0.000000 | 동률

M5 checker 기준으로는 train gain이 있고 holdout이 base보다 낮지 않으므로 `pass`다.
다만 holdout gain은 0이라서, 이 결과만으로 M6 certification이나 router utility claim을
열지는 않는다.

## 6. Holdout delta

`scripts/analyze_m5_delta.py`로 같은 holdout sample에서 base와 LoRA의 정오답 전환을 나눴다.

bucket | count | 의미
--- | ---: | ---
`base_correct_lora_wrong` | 3 | base는 맞고 LoRA가 틀린 sample
`base_wrong_lora_correct` | 3 | base는 틀리고 LoRA가 맞은 sample
`both_wrong` | 38 | 둘 다 틀린 sample
`both_correct` | 20 | 둘 다 맞은 sample

해석은 보수적으로 둔다. LoRA는 train split을 크게 학습했지만, holdout에서는 맞고 틀리는
sample이 3개씩 교환되어 net gain이 없다. 따라서 다음 판단은 `both_wrong` 38개와
두 delta bucket을 비교해 어떤 chart/question 유형에서 전환이 생겼는지 보는 쪽이 맞다.

## 7. Scoring caveat

이번 재실행은 기존 M3/M5 scoring policy를 그대로 사용했다.

```text
normalized_exact
normalized_contains
numeric_tolerance when expected answer has no letters
```

이 정책은 재현성을 위해 그대로 유지했지만, year-like answer에서는 1% numeric tolerance가
넉넉하게 작동할 수 있다. 따라서 certification 전에 numeric scoring을 더 엄격히 나눌지
검토해야 한다. 이번 문서에서는 이 점을 caveat로만 기록하고, 점수 재계산은 하지 않았다.

## 8. 산출물

```text
data/mvp_revisions/chartqapro_v1/train.jsonl
data/mvp_revisions/chartqapro_v1/holdout.jsonl
results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_train.json
results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_holdout.json
data/curricula_revisions/chartqapro_v1/curriculum_manifest.jsonl
configs/track_a/revisions/chartqapro_v1/adapter_candidate_plan.yaml
results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_train_log.json
results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_train.json
results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_holdout.json
results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_m5_summary.json
results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_holdout_delta.json
```

`models/loras/chartqapro_table_cell_r4_v1`은 로컬 artifact이며 git에 커밋하지 않는다.

## 9. 재현 명령

```powershell
.\.venv\Scripts\python.exe scripts\prepare_chartqapro_manifest.py
.\.venv\Scripts\python.exe scripts\run_m3_actual_base_audit.py --manifest data/mvp_revisions/chartqapro_v1/train.jsonl --output results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_train.json --cache-dir data/cache/chartqapro_v1_dataset_rows --resume
.\.venv\Scripts\python.exe scripts\run_m3_actual_base_audit.py --manifest data/mvp_revisions/chartqapro_v1/holdout.jsonl --output results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_holdout.json --cache-dir data/cache/chartqapro_v1_dataset_rows --resume
.\.venv\Scripts\python.exe scripts\compile_m4_revision_curriculum.py
.\.venv\Scripts\python.exe scripts\train_m5_single_lora.py --adapter-id chartqapro_table_cell_r4_v1 --taxonomy-id chart_table_cell --train-manifest data/curricula_revisions/chartqapro_v1/chart_table_cell_train.jsonl --output-dir models/loras/chartqapro_table_cell_r4_v1 --metadata-output results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_train_log.json --cache-dir data/cache/chartqapro_v1_dataset_rows --steps 120 --rank 4 --alpha 8 --target-modules q_proj v_proj
.\.venv\Scripts\python.exe scripts\run_m5_lora_eval.py --manifest data/curricula_revisions/chartqapro_v1/chart_table_cell_train.jsonl --adapter-path models/loras/chartqapro_table_cell_r4_v1 --adapter-id chartqapro_table_cell_r4_v1 --model-id qwen3_vl_4b_chartqapro_table_cell_r4_v1 --output results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_train.json --cache-dir data/cache/chartqapro_v1_dataset_rows --resume
.\.venv\Scripts\python.exe scripts\run_m5_lora_eval.py --manifest data/curricula_revisions/chartqapro_v1/chart_table_cell_holdout.jsonl --adapter-path models/loras/chartqapro_table_cell_r4_v1 --adapter-id chartqapro_table_cell_r4_v1 --model-id qwen3_vl_4b_chartqapro_table_cell_r4_v1 --output results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_holdout.json --cache-dir data/cache/chartqapro_v1_dataset_rows --resume
.\.venv\Scripts\python.exe scripts\check_m5_single_lora.py --adapter-id chartqapro_table_cell_r4_v1 --taxonomy-id chart_table_cell --train-manifest data/curricula_revisions/chartqapro_v1/chart_table_cell_train.jsonl --holdout-manifest data/curricula_revisions/chartqapro_v1/chart_table_cell_holdout.jsonl --train-result results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_train.json --holdout-result results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_holdout.json --train-log results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_train_log.json --m3-train-result results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_train.json --m3-holdout-result results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_holdout.json --summary-output results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_m5_summary.json
.\.venv\Scripts\python.exe scripts\analyze_m5_delta.py --base-result results/m3_revisions/chartqapro_v1/qwen3_vl_4b_base_holdout.json --lora-result results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_correct_holdout.json --output results/m5_revisions/chartqapro_v1/chartqapro_table_cell_r4_v1_holdout_delta.json --taxonomy-id chart_table_cell
```
