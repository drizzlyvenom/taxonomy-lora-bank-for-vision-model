# M1 Schema Contracts 결과 브리프

Status: pass

## 1. 목적

M1의 목적은 Track A v2 actual validation을 시작하기 전에 데이터 계약을 먼저 고정하는 것이다.

## 2. 입력과 실행 조건

검증 대상은 `schemas/`의 예시 계약 파일과 `scripts/check_schemas.py` checker다.

```text
python scripts/check_schemas.py
```

## 3. 계약 검사 표

```text
contract | check | status
AdapterCardV3 | certification 필수 필드와 금지 proxy 필드 검사 | pass
RouteTrace | perception/router/execution/output/memory 구획 검사 | pass
BackboneContract | actual eval과 adapter slot 지원 필드 검사 | pass
PerceptionContract | adapter 선택 없이 perception output만 생성하는지 검사 | pass
RouterContract | perception output을 입력으로 받아 adapter를 선택하는지 검사 | pass
CurriculumManifest | JSONL 파싱과 train row 필수 키 검사 | pass
TeacherAnnotation | JSONL 파싱과 candidate label 표시 검사 | pass
ActualCertificationResult | 64 holdout, actual score, gate 필드 검사 | pass
```

## 4. Pass/fail 판정

M1은 통과다.

```text
M1 schema checks passed.
```

## 5. 다음 수정

M2에서는 이 계약을 기준으로 taxonomy별 `train 64 / holdout 64` actual dataset manifest를 만든다.

## 6. 재현 명령

```text
python scripts/check_schemas.py
```
