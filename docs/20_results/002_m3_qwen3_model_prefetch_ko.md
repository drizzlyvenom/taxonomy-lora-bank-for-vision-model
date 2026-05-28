# M3 준비: Qwen3-VL Reference and 8B Comparison Prefetch

Status: ready_for_m3_base_audit

## 1. 목적

M3 actual base audit를 시작하기 전에 Qwen3 reference VLM backbone과 같은 세대의 큰
비교군 backbone을 로컬에 준비한다.

핵심 비교 구도는 아래와 같다.

```text
Qwen3-VL-4B base
Qwen3-VL-4B + taxonomy LoRA
Qwen3-VL-8B base
```

## 2. 입력과 실행 조건

주 reference 모델:

```text
repo: Qwen/Qwen3-VL-4B-Instruct
url: https://hf.co/Qwen/Qwen3-VL-4B-Instruct
revision: ebb281ec70b05090aa6165b016eac8ec08e71b17
license: apache-2.0
pipeline: image-text-to-text
local path: models/qwen/Qwen3-VL-4B-Instruct
```

동세대 큰 비교군 모델:

```text
repo: Qwen/Qwen3-VL-8B-Instruct
url: https://hf.co/Qwen/Qwen3-VL-8B-Instruct
revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
license: apache-2.0
pipeline: image-text-to-text
local path: models/qwen/Qwen3-VL-8B-Instruct
```

`models/`는 `.gitignore`로 제외되어 모델 가중치는 커밋하지 않는다.

## 3. 로컬 파일 표: Qwen3-VL-4B

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

Qwen3-VL-4B 준비 상태는 pass다. 필수 config/tokenizer/index/weight shard 파일이 모두
존재한다.

## 5. 로컬 파일 표: Qwen3-VL-8B

```text
file | bytes
config.json | 1474
preprocessor_config.json | 390
tokenizer.json | 7032403
model.safetensors.index.json | 67759
model-00001-of-00004.safetensors | 4902275944
model-00002-of-00004.safetensors | 4915962496
model-00003-of-00004.safetensors | 4999831048
model-00004-of-00004.safetensors | 2716270024
```

총 로컬 파일 크기:

```text
17545915883 bytes
```

Qwen3-VL-8B 준비 상태는 pass다. 필수 config/tokenizer/index/weight shard 파일이 모두
존재한다.

## 6. Runtime preflight

M3용 repo-local `.venv`에서 아래 import와 CUDA preflight를 확인했다.

```text
torch: 2.12.0+cu126
transformers: 5.9.0
accelerate: 1.13.0
qwen_vl_utils: installed
cuda device: NVIDIA GeForce RTX 3090
model_type: qwen3_vl
processor: Qwen3VLProcessor
model class: Qwen3VLForConditionalGeneration
```

## 7. 다음 수정

M3에서는 먼저 Qwen3-VL-4B를 adapter 없이 실행해 `data/mvp/holdout.jsonl` 기준 base actual
score를 측정한다. 이후 같은 holdout에서 Qwen3-VL-8B no-LoRA를 scale baseline으로 측정하고,
Qwen3-VL-4B + taxonomy LoRA 결과와 비교한다.

## 8. 재현 명령

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

```text
@'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
    local_dir="models/qwen/Qwen3-VL-8B-Instruct",
)
'@ | python -
```
