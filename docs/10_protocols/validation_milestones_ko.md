# 새 레포 검증 마일스톤: Offline-Certified Taxonomy LoRA Bank

Status: validation plan draft  
Scope: Track A-only restart / actual-only certification  
Working title: **Offline-Certified Taxonomy LoRA Banks for Shared-Backbone Vision Specialists**

---

## 0. 검증 원칙

새 레포의 검증 원칙은 단순하다.

```text
No proxy in certification.
No actual eval, no certification.
Failure is a result.
```

Certification에는 proxy를 넣지 않는다.  
Teacher, difficulty heuristic, rule annotation은 curriculum/diagnostic에만 사용한다.  
최종 AdapterCard certification은 actual model evaluation으로만 결정한다.

---

## 1. 전체 마일스톤

```text
M0. Clean repo scaffold
M1. Schema contracts
M2. MVP adapter-sensitive dataset
M3. Actual base audit
M4. Offline curriculum compiler
M5. Single LoRA learns
M6. Actual certification: correct beats wrong
M7. AdapterCard registry update
M8. Online router evaluation
M9. Offline loop closure
M10. Optional bank serving cost
M11. Paper-ready MVP table
M12. Backbone migration
```

핵심 gate:

```yaml
Gate_1:
  name: single_lora_learns
  asks: "LoRA가 실제로 output을 바꾸고 task를 배울 수 있는가?"

Gate_2:
  name: correct_beats_wrong
  asks: "correct LoRA가 wrong/random LoRA보다 나은가?"

Gate_3:
  name: router_selects_adapter
  asks: "router가 oracle adapter에 가까운 선택을 하는가?"
```

Gate 2 전에는 routing utility claim을 열지 않는다.

---

## M0. Clean Repo Scaffold

### 목표

기존 실험 꼬임 없이 Track A-only repo를 시작한다.

### 작업

```yaml
tasks:
  - initialize new repo
  - write README with Track A-only thesis
  - add framework doc
  - add validation milestone doc
  - add local artifact policy
  - add result brief policy
```

### 산출물

```text
README.md
docs/00_overview/framework_ko.md
docs/10_protocols/validation_milestones_ko.md
docs/20_results/README.md
```

### 통과 조건

```yaml
pass_if:
  - README says Track A is main
  - Foveation is support/input-cost control only
  - certification says actual-only
```

---

## M1. Schema Contracts

### 목표

모든 데이터 구조를 먼저 고정한다.

### 필요한 스키마

```text
schemas/adapter_card_v3.example.yaml
schemas/route_trace.example.yaml
schemas/curriculum_manifest.example.jsonl
schemas/teacher_annotation.example.jsonl
schemas/actual_certification_result.example.yaml
```

### 통과 조건

```yaml
pass_if:
  - schema examples exist
  - lightweight checker passes
  - AdapterCard certification has no proxy fields
```

### 금지

```yaml
forbidden:
  - proxy_certified
  - mixed_actual_proxy_score
  - base_score_from_heuristic in certification
```

---

## M2. MVP Adapter-Sensitive Dataset

### 목표

correct/wrong adapter 차이가 날 가능성이 높은 dataset을 만든다.

### 초기 taxonomy

```yaml
taxonomies:
  document_field_bind:
    domain: document
    evidence_type: field_value
    operation: bind_label_to_value
    failure_mode: distractor_confusion

  chart_table_cell:
    domain: chart
    evidence_type: table_cell
    operation: locate
    failure_mode: label_value_mismatch
```

### 규모

```yaml
minimum:
  train_per_taxonomy: 64
  holdout_per_taxonomy: 64

better:
  train_per_taxonomy: 128
  holdout_per_taxonomy: 128
```

### dataset hardening

```yaml
hardening:
  - multiple distractor codes
  - similar labels
  - target outside center
  - small fonts
  - low contrast variants
  - train/holdout code values disjoint
  - wrong-domain distractors
```

### 산출물

```text
data/mvp/train.jsonl
data/mvp/holdout.jsonl
docs/20_results/000_dataset_brief_ko.md
```

### 통과 조건

```yaml
pass_if:
  - each taxonomy has train and holdout split
  - expected answers are present
  - hard negatives are present
  - train/holdout code values are disjoint
```

---

## M3. Actual Base Audit

### 목표

base model이 task를 너무 잘하거나 너무 못하지 않는지 actual evaluation으로 확인한다.

### 실행

```yaml
model: Qwen3-VL-4B reference
adapter: none
visual_policy: fixed or oracle ROI
split:
  - train
  - holdout
```

### 측정

```yaml
metrics:
  - actual_base_train_score
  - actual_base_holdout_score
  - per_taxonomy_base_score
  - answer_text_samples
  - failure_type_summary
```

### 통과 조건

```yaml
pass_if:
  base_holdout_score:
    lower: 0.30
    upper: 0.85

too_easy_if:
  base_holdout_score: ">= 0.90"

too_hard_if:
  base_holdout_score: "< 0.20"
```

### 실패 시 조치

```yaml
if_too_easy:
  - add distractors
  - lower contrast
  - smaller font
  - harder label binding

if_too_hard:
  - check ROI visibility
  - simplify prompt
  - check answer normalization
```

---

## M4. Offline Curriculum Compiler

### 목표

Teacher/rule/manual annotation을 curriculum manifest로 컴파일한다.

### 입력

```yaml
inputs:
  - route traces if available
  - generated dataset rows
  - teacher annotations
  - hard negatives
```

### 출력

```yaml
outputs:
  - curriculum_manifest.jsonl
  - adapter_candidate_plan.yaml
  - certification_eval_plan.yaml
```

### Teacher handling

```yaml
teacher_modes:
  gemma:
    status: optional
    notes: "Gemma annotation is candidate supervision only."

  rule:
    status: allowed
    notes: "Allowed for MVP if clearly marked."

  manual:
    status: allowed
    notes: "Useful for small MVP."
```

### 통과 조건

```yaml
pass_if:
  - curriculum rows point to taxonomy
  - train/holdout split is preserved
  - teacher_label_is_candidate is true
  - certification plan requires actual eval
```

---

## M5. Gate 1 — Single LoRA Learns

### 목표

각 taxonomy별 correct LoRA가 실제로 학습되는지 확인한다.

### 비교

```yaml
comparisons:
  - base_no_adapter
  - correct_lora
```

### 실행

```yaml
train:
  taxonomies:
    - document_field_bind
    - chart_table_cell
  label_mask_mode: answer_only
  target_modules: ["q_proj", "v_proj"]
  rank: 4 or 8
  steps: 50~200 initially
```

### 통과 조건

```yaml
pass_if:
  correct_lora_train_score: "> actual_base_train_score"
  correct_lora_holdout_score: ">= actual_base_holdout_score"

soft_pass:
  - train overfit allowed for first diagnosis
  - holdout no-gain allowed only if failure_reason is recorded
```

### 실패 유형

```yaml
failure_reasons:
  - no_learning_signal
  - base_too_strong
  - task_too_hard
  - bad_label_mask
  - bad_roi_or_prompt
```

---

## M6. Gate 2 — Actual Certification: Correct Beats Wrong

### 목표

Track A의 핵심 검증.  
correct LoRA가 wrong/random보다 나은지 actual evaluation으로 확인한다.

### 비교

동일 holdout sample set에서 모두 평가한다.

```yaml
comparisons:
  base_no_adapter:
    adapter: none

  correct_lora:
    adapter: taxonomy_matched_adapter

  wrong_lora:
    adapter: other_taxonomy_adapter

  random_untrained_lora:
    adapter: same-rank random adapter
```

### 통과 조건

```yaml
pass_if:
  correct_score: "> base_score"
  correct_score: "> wrong_score + 0.05"
  correct_score: "> random_score + 0.05"

strong_pass_if:
  correct_score: "> wrong_score + 0.10"
```

### 결과 상태

```yaml
status:
  actual_certified:
    if: all gates pass

  actual_failed:
    if: actual eval completed but gates fail

  incomplete:
    if: any comparison missing
```

### 실패 유형

```yaml
failure_reasons:
  - no_gain_vs_base
  - no_correct_wrong_margin
  - correct_worse_than_base
  - wrong_adapter_same_as_correct
  - random_adapter_same_as_correct
  - all_scores_low
  - base_too_high
```

### 산출물

```text
results/actual_certification_result.json
docs/20_results/001_actual_certification_ko.md
configs/adapters/<adapter_id>.yaml
```

---

## M7. AdapterCard Registry Update

### 목표

actual certification 결과만 AdapterCard에 반영한다.

### 규칙

```yaml
if_status_actual_certified:
  update:
    actual_certification.status: actual_certified
    online_routing_allowed: true

if_status_actual_failed:
  update:
    actual_certification.status: actual_failed
    online_routing_allowed: false
    failure_reason: recorded

if_status_incomplete:
  update:
    actual_certification.status: incomplete
    online_routing_allowed: false
```

### 통과 조건

```yaml
pass_if:
  - all AdapterCards reflect actual certification status
  - no proxy score is in actual_certification
  - online_routing_allowed only true for actual_certified adapters
```

---

## M8. Gate 3 — Online Router Evaluation

### 선행 조건

```yaml
requires:
  - at least two actual_certified adapters
```

M6가 실패하면 router eval은 path smoke만 가능하다.

### 비교

```yaml
comparisons:
  - oracle_adapter
  - taxonomy_router
  - random_adapter
  - base_no_adapter
```

### 통과 조건

```yaml
pass_if:
  router_top1_hit: "> random_baseline"
  routed_score: "within 0.05 of oracle_adapter_score"
  wrong_adapter_damage: "bounded"
```

### 실패 시

```yaml
if_router_bad:
  - improve taxonomy classifier
  - use query + visual summary
  - add abstention
  - do not route uncertified adapters
```

---

## M9. Offline Loop Closure

### 목표

한 번의 offline loop가 실제로 돌아가는지 확인한다.

```text
failure traces
  -> curriculum compile
  -> LoRA train
  -> actual certification
  -> AdapterCard update
  -> online router eval
```

### 통과 조건

```yaml
pass_if:
  - at least one adapter candidate trained
  - actual certification completed
  - AdapterCard updated
  - route_trace from online eval stored
```

실패해도 된다. 단 실패 사유가 명확해야 한다.

---

## M10. Optional Bank Serving Cost

### 목표

LoRA bank가 full specialist model reload보다 시스템적으로 저렴한지 측정한다.

### 측정

```yaml
metrics:
  - full_model_reload_latency_ms
  - single_adapter_load_latency_ms
  - multi_adapter_bank_memory_mb
  - set_adapter_latency_mean_ms
  - set_adapter_latency_p95_ms
```

### 통과 조건

```yaml
pass_if:
  set_adapter_latency_p95_ms: "<< full_model_reload_latency_ms"
  adapter_bank_memory_mb: "<< multi_full_model_residency_estimate"
```

이 마일스톤은 accuracy claim이 아니라 system-cost claim이다.

---

## M11. Paper-Ready MVP Table

### 최소 표

#### Table 1. Dataset difficulty

```text
taxonomy / base train score / base holdout score / difficulty status
```

#### Table 2. Single LoRA learns

```text
taxonomy / base / correct LoRA train / correct LoRA holdout
```

#### Table 3. Actual certification

```text
taxonomy / base / correct / wrong / random / margin / status
```

#### Table 4. Router eval

```text
oracle / routed / random / base
```

#### Table 5. System cost

```text
full reload / adapter load / adapter bank / set_adapter switch
```

#### Table 6. Backbone migration

```text
backbone / adapter_slot / base / correct / wrong / random / margin_vs_wrong / memory / latency
```

### Paper-ready 조건

```yaml
paper_ready_mvp:
  - at least 2 taxonomy adapters evaluated
  - actual certification complete
  - at least 1 adapter actual_certified OR negative result clearly explained
  - no proxy in certification table
  - router eval only if adapter certified
```

---

## M12. Backbone Migration

### 목표

Qwen2/Qwen3 reference backbone에서 검증한 protocol을 JEPA/LeWM world-model backbone에
재실행한다.

이 단계에서 옮기는 것은 Qwen에서 학습한 LoRA weight가 아니다. 옮기는 것은 taxonomy,
curriculum, AdapterCard 구조, actual-only certification protocol이다.

### BackboneContract 요구사항

```yaml
BackboneContract:
  required:
    - backbone_id
    - backbone_type
    - input_image_query_or_roi
    - output_answer_or_structured_answer
    - output_latent_state_or_taxonomy_features
    - adapter_slots
    - memory_profile
    - actual_certification_support
```

### 비교

같은 curriculum과 holdout split을 사용해 아래를 비교한다.

```yaml
compare_backbones:
  - Qwen3 reference
  - Qwen2 lightweight
  - JEPA/LeWM world model

metrics:
  - base_score
  - correct_lora_score
  - wrong_lora_score
  - random_lora_score
  - margin_vs_wrong
  - base_memory_mb
  - adapter_memory_mb
  - route_latency_ms
  - adapter_switch_latency_ms
```

### 통과 조건

```yaml
pass_if:
  - JEPA/LeWM backbone implements BackboneContract
  - base/correct/wrong/random actual eval works
  - AdapterCard stores base_backbone and adapter_slot
  - memory/latency/score are comparable under same curriculum
  - Qwen-trained LoRA transfer is not required or claimed
```

### 초기 adapter slot 후보

```yaml
initial_jepa_adapter_slot:
  primary: projector_or_translator
  secondary: taxonomy_router_head

defer_initially:
  - full vision_encoder adaptation
  - latent_predictor adaptation
```

---

## 13. 다음 커밋 추천

```yaml
commit_1:
  name: schema_and_scaffold
  tasks:
    - add schemas
    - add repo layout
    - add README framework

commit_2:
  name: mvp_dataset
  tasks:
    - prepare_mvp_manifest.py
    - train/holdout split
    - dataset brief

commit_3:
  name: actual_base_audit
  tasks:
    - run actual base eval
    - result brief

commit_4:
  name: lora_training
  tasks:
    - train document/chart LoRA
    - evaluate train/holdout

commit_5:
  name: actual_certification
  tasks:
    - base/correct/wrong/random actual comparison
    - AdapterCard update

commit_6:
  name: backbone_contract
  tasks:
    - define BackboneContract schema
    - add Qwen reference implementation notes
    - add JEPA/LeWM migration checklist
```

---

## 14. 중단 기준

아래 중 하나면 구조를 바꾸기 전에 멈추고 원인을 분석한다.

```yaml
stop_and_diagnose_if:
  - base score >= 0.90
  - all scores <= 0.20
  - correct == wrong for both taxonomies
  - correct < base for both taxonomies
  - random adapter improves as much as correct adapter
```

이 경우 새 기능을 붙이지 말고 dataset, label mask, ROI, scoring을 먼저 확인한다.

---

## 15. 결론

새 레포의 검증은 단순하다.

```text
1. adapter-sensitive task를 만든다.
2. base가 적당히 어려워하는지 actual로 확인한다.
3. LoRA를 학습한다.
4. correct/wrong/random을 actual로 비교한다.
5. actual_certified adapter만 online routing한다.
6. 같은 protocol을 BackboneContract 구현체별로 재실행한다.
```

이 순서를 지키면 이전 프로젝트처럼 proxy, routing, Foveation, certification이 섞이지 않는다.
