# 새 레포 프레임워크: Offline-Certified Taxonomy LoRA Bank

Status: framework draft  
Target repo style: clean Track A-first research repo  
Working title: **Offline-Certified Taxonomy LoRA Banks for Shared-Backbone Vision Specialists**  
한국어 작업 제목: **공유 백본 비전 전문가 통합을 위한 오프라인 인증 Taxonomy LoRA Bank**

---

## 0. 핵심 결론

새 레포의 중심축은 **Foveation + LoRA 혼합 실험**이 아니라, 아래 하나로 고정한다.

```text
Offline-certified taxonomy LoRA bank
```

즉, 목표는 다음 질문에 답하는 것이다.

```text
오프라인 루프가 실패 trace를 taxonomy별 adapter-sensitive curriculum으로 컴파일하고,
학습된 LoRA 후보를 actual-only certification으로 검증한 뒤,
runtime online loop에서 certified adapter만 routing할 수 있는가?
```

Track B/Foveation은 메인 contribution이 아니다. 기존 FoveateR 계열 연구를 인용하고, 새 레포에서는 **visual evidence cost control module**로만 둔다.

---

## 1. 새 프레임워크의 역할 분리

### 1.1 Online LoRA Routing Loop

Online loop는 학습하지 않는다.  
선택하고, 실행하고, 실패를 기록한다.

```text
input image/query
  -> lightweight taxonomy router
  -> AdapterCard registry lookup
  -> certified LoRA top-1 selection
  -> shared VLM backbone + selected LoRA inference
  -> verifier/scorer
  -> RouteTrace 저장
```

초기 online loop는 단순하게 유지한다.

```yaml
online_loop_v0:
  adapter_selection: top_1_only
  multi_lora_mixture: false
  token_level_switching: false
  foveation_learning: false
  runtime_learning: false

inputs:
  - image
  - query
  - optional low-res visual evidence
  - optional ROI evidence from existing FoveateR/OCR pipeline

outputs:
  - selected_adapter_id
  - route_confidence
  - answer_text
  - score
  - verifier_result
  - failure_type
  - route_trace
```

초기에는 top-1 routing만 허용한다.  
multi-LoRA mixture, token-level switching, JEPA/LeWM routing은 금지한다.

---

### 1.2 Offline LoRA Learning Loop

Offline loop가 연구의 핵심이다.

```text
RouteTrace / failures / wrong answers / low-confidence cases
  -> teacher or rule annotation
  -> taxonomy assignment
  -> adapter-sensitive curriculum generation
  -> candidate LoRA training
  -> actual-only certification
  -> AdapterCard registry update
```

Simula는 runtime world model이 아니다.  
새 프레임워크에서 Simula는 **offline LoRA compiler**다.

```yaml
simula_compiler:
  inputs:
    - route_traces.jsonl
    - failed answers
    - low confidence samples
    - wrong adapter records
    - verifier rejects
    - ROI miss records
    - base model errors

  process:
    - classify failure_mode
    - assign taxonomy
    - generate adapter-sensitive samples
    - generate hard negatives
    - create train/holdout split
    - propose expected answers
    - write curriculum manifest

  outputs:
    - curriculum_manifest.jsonl
    - teacher_annotations.jsonl
    - adapter_candidate_plan.yaml
    - certification_eval_plan.yaml
```

---

### 1.3 Teacher Model

Gemma 4 26B 또는 다른 큰 모델은 teacher로만 사용한다.

```yaml
teacher_model:
  role:
    - failure annotator
    - taxonomy labeler
    - curriculum proposer
    - hard negative proposer
    - answer/rationale proposer

  not_role:
    - runtime backbone
    - final ground truth
    - certification authority
```

Teacher가 실패해도 framework가 멈추면 안 된다.  
Teacher output은 **candidate supervision**이고, 최종 판단은 actual certification이다.

```text
Teacher proposes.
Certification decides.
```

---

## 2. 절대 규칙: Certification에는 proxy 금지

새 레포의 가장 중요한 규칙이다.

```text
No proxy in certification.
No actual eval, no certification.
Failure is a valid result.
```

Certification에는 아래 네 비교가 모두 actual evaluation으로 들어가야 한다.

```yaml
actual_certification_requires:
  same_holdout_samples: true
  same_visual_policy: true
  same_roi_source: true
  comparisons:
    - base_no_adapter
    - correct_lora
    - wrong_lora
    - random_untrained_lora
```

허용되는 status는 단순하게 둔다.

```yaml
certification_status:
  incomplete:
    meaning: "actual 비교가 아직 모두 수행되지 않음"

  actual_failed:
    meaning: "actual 비교를 수행했지만 gate를 통과하지 못함"

  actual_certified:
    meaning: "actual 비교에서 correct LoRA가 base/wrong/random을 의미 있게 이김"
```

금지한다.

```yaml
forbidden_in_certification:
  - deterministic difficulty proxy
  - base_score - epsilon wrong score
  - random_score = base_score
  - mixed_actual_proxy_score
  - teacher_label_as_final_truth
  - proxy_certified
```

Proxy는 diagnostics에는 쓸 수 있지만 certification에는 쓰지 않는다.

---

## 3. Taxonomy v3

기존의 domain-only taxonomy는 너무 coarse했다.  
새 framework에서는 최소 4축 taxonomy를 사용한다.

```yaml
lora_taxonomy:
  domain:
    - document
    - scene_text
    - ui_screen
    - chart

  evidence_type:
    - small_text
    - field_value
    - table_cell
    - axis_label
    - ui_status
    - visual_symbol

  operation:
    - read
    - locate
    - bind_label_to_value
    - compare
    - normalize_answer
    - structured_output

  failure_mode:
    - missed_evidence
    - wrong_region
    - label_value_mismatch
    - distractor_confusion
    - low_res_ambiguity
    - wrong_adapter_confidence_gain
```

초기 MVP taxonomy는 2개만 선택한다.

```yaml
mvp_taxonomies:
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

처음부터 4개 domain 전체를 닫으려고 하지 않는다.  
먼저 correct/wrong adapter 차이가 실제로 나는지 확인한다.

---

## 4. AdapterCard v3

각 LoRA는 단순 weight가 아니라 card를 가진다.  
단, card의 certification 섹션은 actual-only다.

```yaml
AdapterCard:
  adapter_id: "doc_field_bind_r4_v1"
  base_backbone: "Qwen3-VL-4B"
  teacher_model: "google/gemma-4-26B-A4B-it"

  taxonomy:
    domain: "document"
    evidence_type: "field_value"
    operation: "bind_label_to_value"
    failure_mode: "distractor_confusion"

  training:
    curriculum_id: "simula_doc_field_bind_v1"
    train_manifest: "data/curricula/doc_field_bind_train.jsonl"
    holdout_manifest: "data/curricula/doc_field_bind_holdout.jsonl"
    label_mask_mode: "answer_only"

  weights:
    adapter_path: "adapters/doc_field_bind_r4_v1"
    rank: 4
    alpha: 8
    target_modules: ["q_proj", "v_proj"]

  serving:
    adapter_memory_mb: null
    load_latency_ms: null
    switch_latency_ms: null

  actual_certification:
    status: "incomplete"
    base_score: null
    correct_score: null
    wrong_score: null
    random_score: null
    gain_vs_base: null
    margin_vs_wrong: null
    margin_vs_random: null
    wrong_adapter_damage: null
    failure_reason: null

  diagnostics:
    teacher_annotation_source: "gemma | rule | manual"
    notes: []
```

---

## 5. Data Contracts

### 5.1 RouteTrace

Online loop가 남기는 최소 trace다.

```yaml
RouteTrace:
  trace_id: string
  sample_id: string
  image_id: string
  query: string

  router:
    predicted_taxonomy: {}
    selected_adapter_id: string | null
    route_confidence: float | null
    abstained: bool

  execution:
    base_backbone: string
    adapter_id: string | null
    visual_policy: string
    roi_source: string | null

  output:
    answer_text: string
    score: float | null
    verifier_pass: bool | null

  failure:
    failed: bool
    failure_type: string | null
    notes: string | null

  memory:
    base_after_load_mb: float | null
    adapter_memory_mb: float | null
    peak_mb: float | null
```

### 5.2 CurriculumManifest

Offline compiler가 만드는 학습 단위다.

```yaml
CurriculumRow:
  curriculum_id: string
  sample_id: string
  split: train | holdout
  taxonomy: {}
  prompt: string
  expected_answers: list[str]
  hard_negatives: list[str]
  full_image_path: string
  roi_box: list[float] | null
  teacher_annotation_id: string | null
  teacher_label_is_candidate: true
```

### 5.3 ActualCertificationResult

Certification runner가 만드는 결과다.

```yaml
ActualCertificationResult:
  adapter_id: string
  taxonomy: {}
  holdout_samples: int
  visual_policy: string
  roi_source: string

  scores:
    base_score: float
    correct_score: float
    wrong_score: float
    random_score: float

  margins:
    gain_vs_base: float
    margin_vs_wrong: float
    margin_vs_random: float

  gates:
    actual_fields_complete: true
    gain_gate: bool
    correct_beats_wrong_gate: bool
    correct_beats_random_gate: bool

  status: actual_certified | actual_failed
  failure_reason: string | null
```

---

## 6. Adapter-Sensitive Dataset 원칙

LoRA가 의미 있으려면 task가 adapter-sensitive해야 한다.

```yaml
dataset_goal:
  - base model은 종종 틀림
  - correct LoRA는 배울 수 있음
  - wrong LoRA는 도움이 되지 않음
  - taxonomy 차이가 score로 드러남
```

기본 task는 OCR이 아니라 label-value binding 중심으로 만든다.

```yaml
hardening:
  - small font
  - low contrast
  - multiple distractor codes
  - similar labels
  - target outside center
  - domain-specific layout
  - train/holdout code values fully disjoint
```

예시:

```yaml
document_field_bind:
  prompt: "Return the code next to WORK ORDER."
  visible_text:
    - "WORK ID: A12B"
    - "ORDER TYPE: C33X"
    - "WORK ORDER: W45Q"
    - "WORK AREA: Q91L"
  expected_answer: "W45Q"

chart_table_cell:
  prompt: "Return the code in row Q2 and column East."
  visible_table:
    rows: ["Q1", "Q2", "Q3"]
    columns: ["West", "East", "North"]
    distractor_cells: true
  expected_answer: "<target cell code>"
```

---

## 7. Online Router

초기 router는 taxonomy 기반 top-1만 한다.

```yaml
router_v0:
  input:
    - query text
    - optional low-res evidence summary
    - AdapterCard taxonomy
  output:
    - selected_adapter_id
    - predicted_taxonomy
    - route_confidence
```

Router eval은 certification 이후에만 의미가 있다.

```text
If correct LoRA does not beat wrong LoRA,
router utility is meaningless.
```

초기 router 비교:

```yaml
compare:
  - oracle_adapter
  - taxonomy_router
  - random_adapter
  - base_no_adapter
```

---

## 8. Repo Layout

새 레포는 작게 시작한다.

```text
README.md

docs/
  00_overview/
    framework_ko.md
    claim_boundary_ko.md
  10_protocols/
    validation_milestones_ko.md
    actual_certification_protocol_ko.md
  20_results/
    README.md
  30_paper_notes/
    paper_outline_ko.md
    related_work_notes_ko.md

schemas/
  adapter_card_v3.example.yaml
  route_trace.example.yaml
  curriculum_manifest.example.jsonl
  actual_certification_result.example.yaml

configs/
  track_a_mvp.yaml
  adapters.yaml

src/
  taxonomy_lora_bank/
    taxonomy.py
    adapter_card.py
    simula_compiler.py
    actual_certification.py
    router.py
    scoring.py

scripts/
  prepare_mvp_manifest.py
  compile_curriculum.py
  train_lora.py
  run_actual_certification.py
  run_online_router_eval.py
```

---

## 9. Minimal MVP

새 레포의 첫 목표는 아래만 닫는 것이다.

```yaml
mvp:
  taxonomies:
    - document_field_bind
    - chart_table_cell

  backbone:
    - Qwen3-VL-4B reference

  train_holdout:
    train_per_taxonomy: 64
    holdout_per_taxonomy: 64

  adapters:
    - document_lora
    - chart_lora

  certification:
    comparisons:
      - base_no_adapter
      - correct_lora
      - wrong_lora
      - random_untrained_lora
```

MVP success:

```yaml
mvp_success:
  - correct_lora > base
  - correct_lora > wrong_lora + 0.05
  - correct_lora > random_lora + 0.05
```

MVP failure도 결과다.

```yaml
mvp_failure_is_valid_if:
  - failure_reason is recorded
  - score table is actual-only
  - next dataset/training fix is clear
```

---

## 10. Non-goals

새 레포에서 당장 하지 않는다.

```yaml
non_goals:
  - Foveation novelty claim
  - JEPA / LeWM runtime routing
  - graph memory
  - production serving scheduler
  - broad benchmark superiority
  - multi-LoRA mixture
  - token-level adapter switching
  - teacher labels as final truth
```

---

## 11. 논문 Contribution 초안

```yaml
contributions:
  C1:
    title: "Offline-certified taxonomy LoRA bank"
    content:
      - "failure traces를 taxonomy별 adapter curriculum으로 변환"

  C2:
    title: "Actual-only adapter certification"
    content:
      - "base/correct/wrong/random actual evaluation"
      - "gain, margin, wrong-adapter damage 기록"

  C3:
    title: "Online certified-adapter routing"
    content:
      - "certified adapter만 runtime routing에 허용"

  C4_supporting:
    title: "Foveated evidence as input-cost control"
    content:
      - "기존 FoveateR-style ROI path를 input budget control로 사용"
```

---

## 12. Preferred Wording

```text
We propose an offline-certified taxonomy LoRA bank framework for consolidating vision specialists on a shared VLM backbone.
```

```text
The key idea is to compile failure traces into adapter-sensitive curricula, train candidate LoRA specialists, and certify them through actual base/correct/wrong/random evaluations before online routing.
```

한국어:

```text
본 연구는 shared VLM backbone 위에서 vision specialist를 통합하기 위한 offline-certified taxonomy LoRA bank 프레임워크를 제안한다.
핵심은 실패 trace를 adapter-sensitive curriculum으로 컴파일하고, 후보 LoRA를 학습한 뒤, base/correct/wrong/random actual evaluation을 통과한 adapter만 online routing에 허용하는 것이다.
```
