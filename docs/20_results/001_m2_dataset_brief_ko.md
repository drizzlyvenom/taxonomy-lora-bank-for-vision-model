# M2 MVP Adapter-Sensitive Dataset 결과 브리프

Status: pass

## 1. 목적

M2의 목적은 MVP taxonomy별로 `train 64 / holdout 64` actual manifest를 만들고, expected answer와 hard negative가 있는지 확인하는 것이다.

## 2. 입력과 실행 조건

사용한 Hugging Face source는 두 개다.

```text
document_field_bind: lmms-lab/DocVQA, config=DocVQA, source_split=validation, sha=539088ef8a8ada01ac8e2e6d4e372586748a265e
chart_table_cell: lmms-lab/ChartQA, config=default, source_split=test, sha=9e63b7df1592a1c2158e735cc1725454aef0d6d9
```

DocVQA는 `form`, `layout`, `table/list` question type만 골라 document field/value binding에 맞췄다.
ChartQA는 chart answer를 table-cell-like value lookup taxonomy의 MVP proxy가 아니라 actual source row로 사용한다.

이미지 파일은 커밋하지 않았고, manifest에는 `hf://datasets/...` row reference만 기록했다.
자세한 출처, Hub URL, source split row 수, selection filter는 `data/mvp/dataset_sources.json`에 고정했다.

## 3. Actual manifest 표

```text
taxonomy | train | holdout | source | train/holdout answer overlap
document_field_bind | 64 | 64 | lmms-lab/DocVQA validation | 0
chart_table_cell | 64 | 64 | lmms-lab/ChartQA test | 0
```

생성된 파일:

```text
data/mvp/train.jsonl
data/mvp/holdout.jsonl
data/mvp/manifest_summary.json
data/mvp/dataset_sources.json
```

## 4. Pass/fail 판정

M2는 통과다.

```text
python scripts/check_mvp_manifest.py
M2 manifest checks passed.
```

## 5. 실패 사유 또는 다음 수정

실패 사유는 없다.

다음 M3에서는 이 manifest를 기준으로 adapter 없이 base model actual audit를 실행한다. DocVQA와 ChartQA의 원본 split이 각각 validation/test 중심이라, 이 repo의 `train`/`holdout`은 project-level split이라는 점을 결과표에 계속 명시한다.

## 6. 재현 명령

```text
python scripts/prepare_mvp_manifest.py
python scripts/check_mvp_manifest.py
```
