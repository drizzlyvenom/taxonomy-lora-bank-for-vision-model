# 로컬 Artifact 정책

Status: M0 policy

이 레포는 public repo로 운영한다. 따라서 모델 가중치, 원본 데이터, 학습 산출물처럼 크거나
라이선스 확인이 필요한 파일은 기본적으로 커밋하지 않는다.

## 커밋 가능한 것

- Markdown 문서
- YAML/JSON/JSONL 스키마 예시
- 작은 설정 파일
- 재현용 스크립트
- 작은 synthetic manifest
- 요약 결과표와 result brief

## 커밋하지 않는 것

- GGUF 모델 파일
- LoRA weight 파일
- safetensors/checkpoint 파일
- 원본 이미지 데이터셋
- 학습 중간 산출물
- API key, token, 개인 경로가 들어간 로그

## 로컬 모델 배치

로컬 teacher/backbone 모델과 학습된 LoRA bank는 아래처럼 둘 수 있다.

```text
models/
  gemma/
    *.gguf
  qwen/
    *.gguf
  loras/
    *.safetensors
```

단, `models/` 폴더 전체는 `.gitignore`로 제외한다. public repo에는 모델 이름, 해시,
출처, 라이선스 확인 상태, 사용한 backbone role만 별도 문서나 manifest로 기록한다.

## 결과 기록 원칙

actual certification 결과를 공개할 때는 모델 weight가 아니라 다음을 기록한다.

- 사용한 backbone과 adapter 식별자
- holdout sample 수
- visual policy와 ROI source
- base/correct/wrong/random actual score
- gate 통과 여부
- 실패 사유

결과가 실패여도 숨기지 않는다. 실패 사유가 명확하면 그것도 유효한 연구 결과다.
