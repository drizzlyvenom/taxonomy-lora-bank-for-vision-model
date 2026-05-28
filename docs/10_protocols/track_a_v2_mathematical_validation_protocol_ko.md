# Track A v2 수학적 검증식

Status: protocol draft
Recommended path: `docs/10_protocols/track_a_v2_mathematical_validation_protocol_ko.md`
Scope: offline-certified taxonomy LoRA bank / actual-only certification

---

## 0. 목적

Track A v2 검증이 다시 흔들리지 않게 하기 위해, 모든 gate를 수식으로 고정한다.

핵심 원칙은 다음이다.

```text
No proxy in certification.
No actual eval, no certification.
Failure is a result.
No smoke test as validation.
```

즉, certification table에는 실제 모델 실행으로 얻은 값만 들어간다.
Teacher, heuristic, difficulty proxy, rule annotation은 curriculum/diagnostic에는 쓸 수 있지만 certification에는 들어갈 수 없다.
기본 검증 단위는 taxonomy별 `train 64 / holdout 64` actual run이며, 작은 smoke 결과는
검증 claim으로 사용하지 않는다.

---

## 1. 기본 표기

Taxonomy 집합을 다음과 같이 둔다.

\[
\mathcal{T}=\{\tau_1,\tau_2,\dots,\tau_K\}
\]

각 taxonomy는 네 축으로 정의된다.

\[
\tau = (d, e, o, f)
\]

여기서

```yaml
d: domain
e: evidence_type
o: operation
f: failure_mode
```

각 taxonomy \(\tau\)에 대해 train/holdout dataset을 둔다.

\[
D_\tau^{tr}=\{z_i\}_{i=1}^{n_{tr}}, \qquad
D_\tau^{ho}=\{z_i\}_{i=1}^{n_{ho}}
\]

각 sample은 다음과 같다.

\[
z_i=(I_i, x_i, y_i, \tau_i)
\]

```yaml
I_i: image or visual evidence
x_i: query / prompt
y_i: expected answer
tau_i: taxonomy label
```

Base model은 \(f_0\), taxonomy \(\tau\)의 LoRA adapter가 붙은 model은 \(f_\tau\)로 둔다.

\[
f_0 = f_{\theta}
\]

\[
f_\tau = f_{\theta, A_\tau}
\]

LoRA는 target layer \(\ell\)에서 다음처럼 표현한다.

\[
W_\ell' = W_\ell + \Delta W_{\ell}^{(\tau)}
\]

\[
\Delta W_{\ell}^{(\tau)} = B_{\ell}^{(\tau)} A_{\ell}^{(\tau)}, \qquad
\text{rank}(\Delta W_{\ell}^{(\tau)}) \le r_\tau
\]

---

## 2. Scoring 함수

정답 채점 함수는 다음처럼 둔다.

\[
s(\hat{y}, y) \in [0,1]
\]

예시는 다음과 같다.

```yaml
score_types:
  exact_match:
    output: 0 or 1

  normalized_contains:
    output: 0 or 1

  token_f1:
    output: [0,1]

  numeric_tolerance:
    output: [0,1]
```

모델 \(m\)의 dataset \(D\)에 대한 평균 점수는 다음으로 정의한다.

\[
Q(m,D)=\frac{1}{|D|}\sum_{z_i \in D} s(m(I_i,x_i),y_i)
\]

Track A v2에서 모든 비교는 가능하면 paired comparison으로 한다.
즉, 같은 sample set에 대해 base/correct/wrong/random을 모두 실행한다.

---

## 3. Actual-only complete 조건

각 taxonomy \(\tau\)에 대해 certification에 필요한 네 모델 실행을 정의한다.

```yaml
base:
  model: f_0

correct:
  model: f_tau

wrong:
  model: f_tau_prime, tau_prime != tau

random:
  model: f_random_lora
```

점수는 다음과 같이 둔다.

\[
Q_0^\tau = Q(f_0,D_\tau^{ho})
\]

\[
Q_c^\tau = Q(f_\tau,D_\tau^{ho})
\]

\[
Q_w^\tau = \max_{\tau' \ne \tau} Q(f_{\tau'},D_\tau^{ho})
\]

\[
Q_r^\tau = Q(f_{\text{random}},D_\tau^{ho})
\]

여기서 \(Q_w^\tau\)는 wrong adapter들 중 가장 강한 wrong baseline이다.
즉, correct LoRA는 가장 잘 나온 wrong LoRA보다 좋아야 한다.

Actual-only complete indicator를 다음으로 둔다.

\[
A_\tau =
\mathbf{1}[
Q_0^\tau, Q_c^\tau, Q_w^\tau, Q_r^\tau
\text{ are all actual model evaluations}
]
\]

Certification은 반드시 \(A_\tau=1\)일 때만 가능하다.

```yaml
if A_tau == 0:
  status: incomplete
  actual_certified: false
```

Proxy 값이 하나라도 있으면 certification은 중단한다.

---

## 4. Base difficulty gate

LoRA gain을 보기 위해 base model 난이도가 적당해야 한다.

\[
b_{\min} \le Q_0^\tau \le b_{\max}
\]

권장 초기값:

\[
b_{\min}=0.30, \qquad b_{\max}=0.85
\]

해석:

```yaml
if Q_0_tau >= 0.90:
  status: task_too_easy
  action:
    - add distractors
    - make label-value binding harder
    - reduce contrast or font size

if Q_0_tau < 0.20:
  status: task_too_hard
  action:
    - verify ROI
    - simplify prompt
    - check answer normalization
```

이 gate는 certification의 선행 조건이다.

---

## 5. Gate 1 — Single LoRA Learns

각 taxonomy \(\tau\)에 대해 correct LoRA가 base보다 학습 신호를 먹는지 확인한다.

Train gain:

\[
G_{tr}^{\tau} = Q(f_\tau,D_\tau^{tr}) - Q(f_0,D_\tau^{tr})
\]

Holdout gain:

\[
G_{ho}^{\tau} = Q(f_\tau,D_\tau^{ho}) - Q(f_0,D_\tau^{ho})
\]

Gate 1의 최소 조건:

\[
G_{tr}^{\tau} > 0
\]

그리고 holdout에서는 최소한 base보다 심하게 나빠지지 않아야 한다.

\[
G_{ho}^{\tau} \ge -\epsilon_{ho}
\]

권장값:

\[
\epsilon_{ho}=0.02
\]

더 강한 조건:

\[
G_{ho}^{\tau} \ge \delta_{gain}
\]

권장값:

\[
\delta_{gain}=0.05
\]

해석:

```yaml
Gate_1_pass:
  weak:
    - train_gain > 0
    - holdout_gain >= -0.02

  strong:
    - holdout_gain >= 0.05

Gate_1_fail:
  - no_learning_signal
  - task_too_easy
  - task_too_hard
  - bad_label_mask
  - bad_roi_or_prompt
```

Gate 1은 “LoRA가 배울 수 있는가”를 보는 gate다.
아직 correct-vs-wrong utility claim은 열지 않는다.

---

## 6. Gate 2 — Correct Beats Wrong

Track A v2의 핵심 gate다.

Correct LoRA의 base 대비 gain:

\[
G_\tau = Q_c^\tau - Q_0^\tau
\]

Correct LoRA의 wrong 대비 margin:

\[
M_{w}^{\tau} = Q_c^\tau - Q_w^\tau
\]

Correct LoRA의 random 대비 margin:

\[
M_{r}^{\tau} = Q_c^\tau - Q_r^\tau
\]

Actual certification 조건:

\[
A_\tau=1
\]

\[
G_\tau \ge \delta_{gain}
\]

\[
M_w^\tau \ge \delta_{wrong}
\]

\[
M_r^\tau \ge \delta_{random}
\]

권장 초기값:

\[
\delta_{gain}=0.03
\]

\[
\delta_{wrong}=0.05
\]

\[
\delta_{random}=0.05
\]

최종 판정:

```yaml
actual_certified:
  if:
    - A_tau == 1
    - G_tau >= delta_gain
    - M_w_tau >= delta_wrong
    - M_r_tau >= delta_random

actual_failed:
  if:
    - A_tau == 1
    - any gate fails

incomplete:
  if:
    - A_tau == 0
```

---

## 7. Paired confidence bound

Sample 수가 작으면 단순 평균만 보지 않는다.
같은 sample에 대해 correct와 wrong의 paired difference를 둔다.

\[
d_i^{cw} =
s(f_\tau(I_i,x_i),y_i)
-
\max_{\tau'\ne\tau} s(f_{\tau'}(I_i,x_i),y_i)
\]

평균:

\[
\bar{d}^{cw} = \frac{1}{n}\sum_{i=1}^{n}d_i^{cw}
\]

표준편차:

\[
\sigma_d = \sqrt{\frac{1}{n-1}\sum_i(d_i^{cw}-\bar{d}^{cw})^2}
\]

95% lower confidence bound:

\[
LCB_{95}(d^{cw}) =
\bar{d}^{cw} - t_{0.95,n-1}\frac{\sigma_d}{\sqrt{n}}
\]

강한 certification은 다음을 요구한다.

\[
LCB_{95}(d^{cw}) \ge \delta_{wrong}
\]

동일하게 base/random에 대해서도 paired LCB를 계산할 수 있다.

```yaml
strong_actual_certified:
  requires:
    - LCB95(correct - wrong) >= delta_wrong
    - LCB95(correct - random) >= delta_random
    - LCB95(correct - base) >= delta_gain
```

샘플 수가 작거나 score가 binary라면 bootstrap LCB를 사용해도 된다.

---

## 8. Failure reason taxonomy

Gate 2가 실패하면 실패 사유를 수식으로 분류한다.

### 8.1 No gain vs base

\[
G_\tau < \delta_{gain}
\]

```yaml
failure_reason: no_gain_vs_base
```

### 8.2 No correct-wrong margin

\[
M_w^\tau < \delta_{wrong}
\]

```yaml
failure_reason: no_correct_wrong_margin
```

### 8.3 Correct worse than base

\[
Q_c^\tau < Q_0^\tau
\]

```yaml
failure_reason: correct_worse_than_base
```

### 8.4 Wrong adapter same as correct

\[
|Q_c^\tau - Q_w^\tau| < \epsilon
\]

```yaml
failure_reason: wrong_adapter_same_as_correct
```

### 8.5 Base too high

\[
Q_0^\tau \ge b_{\max}
\]

```yaml
failure_reason: base_too_high
```

### 8.6 All scores low

\[
\max(Q_0^\tau,Q_c^\tau,Q_w^\tau,Q_r^\tau) < b_{\min}
\]

```yaml
failure_reason: all_scores_low
```

이 실패 사유는 다음 offline Simula loop의 입력으로 들어간다.

---

## 9. Wrong adapter damage와 collapse risk

Wrong adapter damage는 wrong adapter가 base보다 얼마나 나빠지는지로 정의한다.

\[
D_w^\tau = \max(0, Q_0^\tau - Q_w^\tau)
\]

하지만 damage가 크다고 무조건 나쁜 것은 아니다.
Correct와 wrong이 분리된다는 뜻일 수도 있다.
문제는 router가 wrong adapter를 고를 확률이 있을 때다.

Router의 wrong selection probability를 \(p_w^\tau\)라고 하면 routing risk는 다음으로 둔다.

\[
R_{\text{route}}^\tau = p_w^\tau \cdot D_w^\tau
\]

안전 조건:

\[
R_{\text{route}}^\tau \le \delta_{\text{risk}}
\]

초기에는 \(p_w^\tau\)를 실제 router error rate로 추정한다.

---

## 10. AdapterCard actual certification fields

AdapterCard의 certification 섹션은 다음 형식으로만 채운다.

```yaml
actual_certification:
  status: incomplete | actual_failed | actual_certified

  base_score: float | null
  correct_score: float | null
  wrong_score: float | null
  random_score: float | null

  gain_vs_base: float | null
  margin_vs_wrong: float | null
  margin_vs_random: float | null

  lcb_gain_vs_base: float | null
  lcb_margin_vs_wrong: float | null
  lcb_margin_vs_random: float | null

  actual_fields_complete: bool
  same_holdout_samples: bool
  same_visual_policy: bool
  same_roi_source: bool

  failure_reason: string | null
```

금지 필드:

```yaml
forbidden:
  - proxy_score
  - mixed_actual_proxy_score
  - difficulty_proxy_as_base_score
  - wrong_score_from_base_minus_epsilon
```

Proxy나 heuristic은 아래에만 둘 수 있다.

```yaml
diagnostics:
  difficulty_proxy_score: float | null
  teacher_annotation_source: gemma | rule | manual | fallback
  notes: []
```

---

## 11. Router gate

Router \(\pi\)는 sample \(z_i\)에서 adapter id를 선택한다.

\[
\pi(z_i) \in \mathcal{A}
\]

Oracle adapter는 ground-truth taxonomy adapter다.

\[
a_i^* = A_{\tau_i}
\]

Top-1 hit:

\[
H_{\pi}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\pi(z_i)=a_i^*]
\]

Router score:

\[
Q_{\pi}=
\frac{1}{N}
\sum_{i=1}^{N}
s(f_{\pi(z_i)}(I_i,x_i),y_i)
\]

Oracle score:

\[
Q_{\text{oracle}}=
\frac{1}{N}
\sum_{i=1}^{N}
s(f_{a_i^*}(I_i,x_i),y_i)
\]

Router regret:

\[
Regret_{\pi}=Q_{\text{oracle}}-Q_{\pi}
\]

Random adapter score:

\[
Q_{\text{random-route}}=
\frac{1}{N}
\sum_{i=1}^{N}
s(f_{a_i^{rand}}(I_i,x_i),y_i)
\]

Router gate:

\[
H_{\pi} > H_{\text{random}}
\]

\[
Regret_{\pi} \le \epsilon_{\text{route}}
\]

\[
Q_{\pi} > Q_{\text{random-route}}
\]

권장값:

\[
\epsilon_{\text{route}}=0.05
\]

중요:

```yaml
router_eval_requires:
  - at least two actual_certified adapters
```

Certified adapter가 2개 미만이면 router eval은 연기하고, router utility claim을 열지 않는다.

### 11.1 M12 perception-router 분리식

M12에서는 perception model \(P_\phi\)와 router \(\pi_\psi\)를 분리해 기록한다.

\[
e_i = P_\phi(I_i, r_i)
\]

여기서 \(r_i\)는 optional ROI이며, \(e_i\)는 visual evidence summary 또는 latent state다.

\[
\pi_\psi(e_i, x_i, \mathcal{R}) \in \mathcal{A}_{cert}
\]

\(\mathcal{R}\)은 certified AdapterCard registry이고, \(\mathcal{A}_{cert}\)는
actual_certified adapter 집합이다.

Perception-only score는 evidence가 정답에 충분한 정보를 담는지 따로 본다.

\[
Q_P(P_\phi,D)=\frac{1}{|D|}\sum_{z_i \in D}
s(g(e_i,x_i),y_i)
\]

Router-only score는 같은 \(e_i\)에서 adapter 선택이 oracle에 가까운지 본다.

\[
H_{\pi|P}=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\pi_\psi(e_i,x_i,\mathcal{R})=a_i^*]
\]

M12 grid는 smoke 없이 taxonomy별 64/64 actual split으로 실행한다.

```yaml
m12_requires:
  train_per_taxonomy: 64
  holdout_per_taxonomy: 64
  no_smoke_validation: true

compare:
  perception_models:
    - qwen_vlm
    - jepa_lewm
  router_models:
    - oracle
    - taxonomy_classifier
    - learned
    - random
```

---

## 12. Offline loop improvement

Offline loop \(k\)에서 registry를 \(\mathcal{R}^{(k)}\)라고 하자.

\[
\mathcal{R}^{(k)} = \{A_{\tau}^{(k)}\}
\]

다음 offline loop 후 registry:

\[
\mathcal{R}^{(k+1)}
\]

Certified adapter 수:

\[
C^{(k)} = |\{A \in \mathcal{R}^{(k)} : status(A)=actual\_certified\}|
\]

Route score:

\[
Q_{\pi}^{(k)} = Q(\pi^{(k)},D^{eval})
\]

Offline loop가 좋아졌다고 말하려면 다음 중 하나가 actual로 관측되어야 한다.

\[
C^{(k+1)} > C^{(k)}
\]

또는

\[
Q_{\pi}^{(k+1)} - Q_{\pi}^{(k)} \ge \delta_{\text{loop}}
\]

단, cost가 budget을 넘으면 안 된다.

\[
M_{\text{bank}}^{(k+1)} \le M_{\max}
\]

\[
L_{\text{switch}}^{(k+1)} \le L_{\max}
\]

---

## 13. System cost validation

Track A의 system claim은 accuracy와 분리해서 검증한다.

Full specialist model residency estimate:

\[
M_{\text{multi}}=\sum_{j=1}^{K}M(W_j)
\]

Shared backbone + LoRA bank:

\[
M_{\text{bank}}=M(W_0)+\sum_{\tau \in \mathcal{T}}M(A_\tau)
\]

Resident saving:

\[
S_M=1-\frac{M_{\text{bank}}}{M_{\text{multi}}}
\]

Full model reload latency:

\[
L_{\text{reload}}
\]

Adapter switch latency:

\[
L_{\text{switch}}
\]

Switch speedup:

\[
S_L=\frac{L_{\text{reload}}}{L_{\text{switch}}}
\]

System cost claim:

```yaml
system_cost_pass:
  - S_M > 0
  - S_L >> 1
```

단, 이 claim은 adapter accuracy claim과 분리한다.

---

## 14. Visual evidence cost control

Foveation/ROI는 Track A의 input cost를 통제하는 보조 모듈이다.

Full visual token count:

\[
T_{\text{full}}
\]

ROI visual token count:

\[
T_{\text{roi}}
\]

Token reduction:

\[
S_T=1-\frac{T_{\text{roi}}}{T_{\text{full}}}
\]

Visual path는 다음 budget을 만족해야 한다.

\[
T_{\text{roi}} \le T_{\max}
\]

\[
M_{\text{visual}} \le M_{\text{visual,max}}
\]

Track B result는 다음 문장까지만 허용한다.

```text
ROI evidence path controls visual input cost for Track A certification.
```

---

## 15. Paper table 기준

최소 논문 표는 다음 수식을 채워야 한다.

### Table 1. Base difficulty

\[
Q_0^\tau
\]

### Table 2. Single LoRA learns

\[
Q(f_0,D_\tau^{tr}), Q(f_\tau,D_\tau^{tr}), Q(f_0,D_\tau^{ho}), Q(f_\tau,D_\tau^{ho})
\]

### Table 3. Actual certification

\[
Q_0^\tau, Q_c^\tau, Q_w^\tau, Q_r^\tau, G_\tau, M_w^\tau, M_r^\tau
\]

### Table 4. Router evaluation

\[
H_\pi, Q_\pi, Q_{\text{oracle}}, Regret_\pi
\]

### Table 5. System cost

\[
M_{\text{bank}}, M_{\text{multi}}, S_M, L_{\text{reload}}, L_{\text{switch}}, S_L
\]

---

## 16. 최종 판정 로직

각 taxonomy에 대해:

```yaml
if actual_fields_complete == false:
  status: incomplete

elif base_score >= b_max:
  status: actual_failed
  failure_reason: base_too_high

elif max(base, correct, wrong, random) < b_min:
  status: actual_failed
  failure_reason: all_scores_low

elif gain_vs_base < delta_gain:
  status: actual_failed
  failure_reason: no_gain_vs_base

elif margin_vs_wrong < delta_wrong:
  status: actual_failed
  failure_reason: no_correct_wrong_margin

elif margin_vs_random < delta_random:
  status: actual_failed
  failure_reason: no_margin_vs_random

else:
  status: actual_certified
```

Router는 다음 조건에서만 평가한다.

```yaml
if number_of_actual_certified_adapters < 2:
  router_status: deferred_no_certified_adapter_pair
  run_router_eval: false
else:
  run_router_eval: true
```

---

## 17. 결론

이 수식 문서는 Track A v2가 다시 흔들리지 않게 하는 안전장치다.

핵심은 세 가지다.

```text
1. Certification에는 actual score만 넣는다.
2. Correct LoRA는 base/wrong/random을 모두 이겨야 한다.
3. 실패는 다음 Simula offline loop의 입력으로 기록한다.
```

이 원칙을 지키면, 실패해도 연구가 전진한다.
