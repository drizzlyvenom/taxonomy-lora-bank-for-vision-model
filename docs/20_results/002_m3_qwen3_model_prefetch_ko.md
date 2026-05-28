# M3 준비: Qwen3-VL-4B Reference Model Prefetch

Status: ready_for_m3_base_audit

## 1. 목적

M3 actual base audit를 시작하기 전에 Qwen3 reference VLM backbone을 로컬에 준비한다.

## 2. 입력과 실행 조건

사용한 모델:

```text
repo: Qwen/Qwen3-VL-4B-Instruct
url: https://hf.co/Qwen/Qwen3-VL-4B-Instruct
revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
license: apache-2.0
pipeline: image-text-to-text
local path: models/qwen/Qwen3-VL-4B-Instruct
```

`models/`는 `.gitignore`로 제외되어 모델 가중치는 커밋하지 않는다.

## 3. 로컬 파일 표

```text
file | bytes
config.json | 1505
preprocessor_config.json | 390
tokenizer.json | 7032403
model.safetensors.index.json | 64742
model-00001-of-00002.safetensors | 4967229296
model-00002-of-00002.safetensors | 3908490048
```

총 로컬 파일 크기:

```text
8887292732 bytes
```

## 4. Pass/fail 판정

M3 준비 상태는 pass다. 필수 config/tokenizer/index/weight shard 파일이 모두 존재한다.

## 5. 다음 수정

M3에서는 이 모델을 adapter 없이 실행해 `data/mvp/holdout.jsonl` 기준 base actual score를 측정한다.

## 6. 재현 명령

```text
@'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-4B-Instruct",
    revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
    local_dir="models/qwen/Qwen3-VL-4B-Instruct",
)
'@ | python -
```
