# 결과 브리프 정책

Status: M0 policy

`docs/20_results/`에는 사람이 읽을 수 있는 짧은 결과 브리프를 둔다. 원본 로그나 대용량
artifact를 그대로 넣는 곳이 아니다.

## 브리프 형식

각 브리프는 가능하면 아래 순서를 따른다.

```text
1. 목적
2. 입력과 실행 조건
3. actual score 표
4. pass/fail 판정
5. 실패 사유 또는 다음 수정
6. 재현 명령
```

## Certification 표 규칙

Certification 표에는 actual evaluation 값만 들어간다.

```text
taxonomy | base | correct_lora | wrong_lora | random_lora | status | failure_reason
```

아래 값은 certification 표에 넣지 않는다.

- deterministic difficulty proxy
- teacher confidence
- heuristic score
- mixed actual/proxy score

Smoke test 결과는 validation evidence로 쓰지 않는다. 기본 결과 브리프는 taxonomy별
`train 64 / holdout 64` actual run을 기준으로 작성한다.

## 파일 이름

권장 이름은 milestone 번호를 앞에 붙인다.

```text
000_m1_schema_contracts_ko.md
001_m2_dataset_brief_ko.md
002_m3_qwen3_model_prefetch_ko.md
003_actual_base_audit_ko.md
004_single_lora_learns_ko.md
005_actual_certification_ko.md
```

## 현재 결과 브리프

```text
000_m1_schema_contracts_ko.md
001_m2_dataset_brief_ko.md
002_m3_qwen3_model_prefetch_ko.md
003_m3_actual_base_audit_ko.md
mvp_review_after_m3_ko.md
```
