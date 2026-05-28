# 정확도와 VRAM claim 재검토

Status: negative result recorded, M6 deferred

## 1. 목적

이 문서는 기존 ChartQA M5 first pass와 ChartQAPro M4/M5 revision을 합쳐, 현재 Qwen
single-backbone + LoRA 구조가 원래 의도한 resource claim을 지지하는지 재검토한다.

결론부터 적으면, 현재 결과는 `LoRA routing reduces VRAM` claim을 지지하지 않는다. LoRA는
같은 backbone 위에서 resident VRAM을 크게 줄이지 못했고, holdout 정확도도 base 대비 유의미한
증가를 보이지 않았다.

## 2. 정확도 결과

taxonomy / source | split | base_no_adapter | correct_lora | gain | 판단
--- | --- | ---: | ---: | ---: | ---
`chart_table_cell` / ChartQA | holdout | 0.734375 | 0.718750 | -0.015625 | base보다 1개 낮음
`chart_table_cell` / ChartQAPro | holdout | 0.359375 | 0.359375 | +0.000000 | 동률

ChartQAPro는 train에서는 큰 gain을 보였지만, holdout에서는 base와 LoRA가 23/64로 같다.
따라서 데이터셋을 바꿔도 `correct_lora > base`라는 M6 첫 gate를 열 정도의 evidence는 아직 없다.

Holdout delta도 같은 결론을 지지한다.

source | base_correct_lora_wrong | base_wrong_lora_correct | both_wrong | both_correct | net
--- | ---: | ---: | ---: | ---: | ---:
ChartQA | 2 | 1 | 16 | 45 | -1
ChartQAPro | 3 | 3 | 38 | 20 | 0

## 3. Peak VRAM

저장된 result JSON의 `peak_cuda_allocated_mb`를 기준으로 비교했다. ChartQA base는 전체 holdout
128개 중 `chart_table_cell` 64개만 따로 집계했다.

case | max allocated MiB | mean allocated MiB | p95 allocated MiB
--- | ---: | ---: | ---:
ChartQA base holdout | 8816.19 | 8611.81 | 8757.12
ChartQA LoRA holdout | 8824.22 | 8617.42 | 8761.93
ChartQAPro base holdout | 9079.81 | 8900.23 | 9077.40
ChartQAPro LoRA holdout | 9084.30 | 8905.39 | 9083.95

LoRA eval은 base eval보다 peak allocated가 조금 더 높지만, 차이는 매우 작다. 반대로 ChartQA와
ChartQAPro 사이의 차이는 sample image / visual token 조건 쪽에서 더 크게 나타난다.

Training log 기준 peak allocated는 다음과 같다.

case | logged training peak MiB
--- | ---:
ChartQA LoRA train | 10433.60
ChartQAPro LoRA train | 10845.87

## 4. Resident VRAM

대표 holdout 첫 sample을 별도 Python process로 띄워 `nvidia-smi`의 GPU memory used delta를
비교했다. 이 값은 현재 Windows desktop background usage의 영향을 받으므로 certification
metric이 아니라 resource sanity check다.

case | resident load delta MiB | resident generate delta MiB | torch peak during generate MiB
--- | ---: | ---: | ---:
ChartQA base sample | 8689 | 8897 | 8566.22
ChartQA LoRA sample | 8745 | 8903 | 8571.84
ChartQAPro base sample | 8689 | 8897 | 8576.78
ChartQAPro LoRA sample | 8812 | 8960 | 8582.40

같은 Qwen3-VL-4B backbone을 올려둔 상태에서는 LoRA가 resident VRAM을 줄이지 않는다.
LoRA 추가분은 작지만, backbone 자체와 visual processing 비용은 그대로 남는다.

주의할 점은, 이 문서가 `Qwen2.5 + LoRA`를 직접 측정한 결과는 아니라는 것이다. 다만 현재
결과는 Qwen2.5 계열로 가더라도 VRAM 절감이 생긴다면 그 원인은 LoRA가 아니라 더 작은 backbone,
양자화, 또는 낮은 visual token budget이어야 함을 시사한다.

## 5. Claim boundary

현재 evidence로는 아래 claim은 쓰지 않는다.

```text
LoRA routing reduces VRAM.
Same-backbone LoRA routing gives meaningful resident VRAM reduction.
Qwen single-backbone LoRA bank is enough to support the resource-saving claim.
```

현재 evidence로 쓸 수 있는 보수적 claim은 아래 정도다.

```text
Same-backbone LoRA adapters add little resident VRAM over an already loaded VLM backbone.
Same-backbone LoRA does not reduce the backbone VRAM itself.
The current Qwen3-VL-4B + LoRA setup is useful as an offline certification prototype, not as a VRAM-saving proof.
```

가능하지만 아직 검증하지 않은 future claim은 다음처럼 재정의해야 한다.

```text
A smaller or quantized backbone plus certified LoRA may be cheaper than a larger monolithic backbone if holdout accuracy is recovered.
A split perception/world-model backbone plus lightweight routing model may reduce resident cost by avoiding full VLM reasoning on every step.
```

## 6. M6 판단

M6의 통과 조건은 `correct_score > base_score`, `correct_score > wrong_score + 0.05`,
`correct_score > random_score + 0.05`다. 현재 ChartQAPro revision은 첫 조건부터 만족하지
못한다.

```yaml
decision: do_not_open_m6_now
primary_reason: no_positive_holdout_gain_vs_base
secondary_reason: same_backbone_lora_does_not_support_vram_reduction_claim
expected_m6_outcome_if_run_now: actual_failed_no_gain_vs_base
```

따라서 지금은 M6 certification을 진행하지 않는다. 이 결과는 폐기하지 않고, Qwen
single-backbone 구조의 한계를 보여주는 negative evidence로 보존한다.

## 7. 다음 방향 후보

다음 방향은 아직 결정하지 않는다. 다만 현재 결과를 기준으로 보면 선택지는 두 갈래다.

1. Offline loop를 certification / registry compiler protocol로 더 깎고, VRAM 절감 claim은
   중심에서 내린다.
2. 원래 resource claim을 살리려면 M12의 split architecture를 앞당겨
   `perception world model + LoRA routing LLM/JEPA` 구조로 다시 설계한다.

이 문서는 방향 결정을 강제하지 않는다. 현재 실험이 지지하지 않는 claim을 분명히 닫아두는
기록으로 사용한다.
