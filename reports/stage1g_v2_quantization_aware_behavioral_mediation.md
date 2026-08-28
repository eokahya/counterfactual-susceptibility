# Stage 1G-v2 — Quantization-Aware Behavioral Mediation

## Terminal result and decision

```text
completed_stage1g_v2_not_supported
```

The frozen classifier had adequate prompt/crossing coverage but only 3 of 7
support criteria passed. On this small-model PLT MPS/BF16 runtime:

```text
behavioral_importance_result: not_supported_on_small_model_plt
mediation_result: not_supported_on_small_model_plt
project_decision: stop critical-alpha, finite-probe, and HVP development here
```

All observed mediation effects remained below the frozen prompt-specific BF16
behavior-resolution floors. Therefore this result supports stopping local
calibration work, but it does not identify effects smaller than the BF16
measurement floor. Any attempt to resolve such effects requires a
higher-precision/reference runtime, not another small-probe protocol.

## Git provenance and historical preservation

- Exact source branch: `stage-1g-behavioral-mediation-pilot`
- Exact base and unchanged Stage 1G-v1 head:
  `6a5bc86c24c35b6920c9682f82ee7874c80bdf58`
- V2 branch: `stage-1g-v2-quantization-aware-behavioral-mediation`
- Protocol freeze commit: `0cb28982ce44c4a20fd3c70c62339c7df6e32cbf`
- Prediction freeze and execution commit:
  `a8717bef2bd6c76ee3f926ad81e10bea02e35ac4`
- Canonical result-bundle commit:
  `4106396999bd81bbea2f251c5e42d73c02b45cb7`
- Stage 1F protected head:
  `6434e72964d8fc9d14e2a6b4cdd9109d7c29e273`
- Protected `main` head: `7aacf30d888f96a29a1cfc82d035fca489ed0c17`

Stage 1G-v1 remained `inconclusive_runtime`. Its authenticated evidence was
unchanged:

- report SHA-256:
  `ebd3f45cc6b7c6bbf5b4783ed286617bcb4d1646fb116024d349476a04cbf27d`
- preflight SHA-256:
  `0e15393e060ceb0f2ec448c19b8281a216ca0adb02304e7b5603246f766fd16f`
- G01–G20 scientific baseline calls: `0`
- scientific source-suppression calls: `0`
- scientific attempt started: `false`
- prediction manifest created: `false`

The final local/origin v2 equality is verified after the reporting commit and
reported in the handoff. No historical branch, historical report/artifact,
`main`, or `paper/` file was modified.

## Independent derivative validation

The independent reference was selected from verified pinned source before any
runtime result: signed, raw, unnormalized feature-to-answer and
feature-to-contrast logit attribution edges from `circuit-tracer==0.5.2`.
For each positive active feature:

```text
g_edge = (E_feature→answer - E_feature→contrast) / activation
```

This path did not call or consume `OutputSensitivityVJPContext.compute` and did
not use UI-normalized, absolute, pruned-display, or influence-normalized edges.
Eight Norway features across distinct layers were compared with the existing
batched output VJP.

| Metric | Observed | Frozen gate |
|---|---:|---:|
| finite values | true | true |
| feature count | 8 | >= 8 |
| sign agreement | 1.000000 | 1.000000 |
| Spearman | 1.000000 | >= 0.990000 |
| median symmetric normalized error | 0.008459 | <= 0.010000 |
| p95 symmetric normalized error | 0.028511 | <= 0.030000 |

## BF16 resolution audit

The finite intervention ladder was diagnostic only and was never presented as
derivative validation or used for prompt/pair selection, ranking, thresholds,
or terminal classification.

```text
relative activation changes: 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0
Norway behavior-resolution floor: 1.0
features with a resolvable BF16 response: 7 / 8
```

First-resolvable amplitude distribution:

| First amplitude | Feature count |
|---:|---:|
| 0.125 | 1 |
| 0.25 | 2 |
| 0.5 | 3 |
| 1.0 | 1 |
| unresolved through 2.0 | 1 |

The floor was exactly
`max(1e-4, 5*repeat_error, 4*(answer_ULP+contrast_ULP))`. No FP32 shadow
readout was implemented or used.

## Frozen scientific panel

The deterministic eligible order was:

```text
G10 G01 G02 G08 G04 G07 G19 G18
```

Twenty baseline-only model calls evaluated prompts until eight were eligible.
G16 failed top-64; G15 and G14 failed exact contextual one-token suffixes; G11
had nonpositive baseline answer-minus-contrast behavior. Panel membership was
frozen before intervention:

| Panel | Memberships | Quota shortfall |
|---|---:|---:|
| B: behavior-weighted | 32 | 0 |
| Q: inhibitory influence only | 32 | 0 |
| G: output gradient only | 32 | 0 |
| D: directional controls | 16 | 0 |

There were 111 unique execution pairs. Only one pair overlapped B and Q; B–G
and Q–G overlap was zero. Every positive-panel pair had predicted
`abs(M_hat)/BF16-floor < 1`; prompt maxima ranged from 0.0382 to 0.1229. This
metadata did not discard or rerank any pair.

## Production and call accounting

The synthetic worker → fsync journal → assembler → fresh-process hostile
validator rehearsal passed. The disjoint real Norway rehearsal also passed all
five conditions, exact multi-edit mapping, and behavior response requirements.

The canonical run used one attempt and zero scientific retries:

| Condition | Completed calls |
|---|---:|
| baseline no-op | 111 |
| baseline repeat | 111 |
| source full ablation | 111 |
| source ablation + target clamp | 73 |
| target-only injection | 73 |
| total | 479 |

Worker calls, complete journal points, and fresh-process serialized points were
all exactly `479`. There were 111 sweeps and no missing conditions or
quantization-schedule collapses.

## Behavioral and mediation results

Primary behavior was the actual BF16 answer-logit minus contrast-logit from the
pinned runtime. No zero BF16 effect was replaced by a shadow readout.

| Panel | Crossings | Prompts | Mean abs M | Median abs M | Above floor |
|---|---:|---:|---:|---:|---:|
| B | 29 / 32 | 8 | 0.088362 | 0.062500 | 0 / 29 |
| Q | 32 / 32 | 8 | 0.037109 | 0.000000 | 0 / 32 |
| G | 13 / 32 | 7 | 0.062500 | 0.062500 | 0 / 13 |
| D | 1 / 16 | 1 | n/a | n/a | n/a |

Across 73 analyzable crossing pairs, absolute signed mediation had minimum
`0`, p25 `0`, median `0.0625`, p75 `0.125`, p95 `0.125`, and maximum
`0.1875`; 46/73 were nonzero at BF16. All 73 were below their frozen floors
(`0.75` for seven prompts and `1.0` for G19), so prospective sign accuracy had
zero eligible above-floor observations. Its stored `0.0` accuracy and bootstrap
`[0.0, 0.0]` follow the frozen empty-prompt statistic and are not evidence of
sign reversal.

B target-injection sign agreement was `15/29 = 0.517241`, exact-binomial 95%
interval `[0.325315, 0.705514]`. `M` versus injection `I` Spearman was
`0.188106`. No mediated fraction was reported: no source effect exceeded the
frozen stability requirement of ten resolution floors.

Directional controls violated the movement-toward-gate constraint in
`3/16 = 0.1875`, above the frozen maximum `0.10`.

## Panel comparisons and prompt-cluster intervals

Prompt-level B minus Q mean absolute mediation was `0.049479`; its 10,000-draw
prompt-cluster bootstrap 95% interval was `[0.041016, 0.059896]`. B minus G was
`0.030599`, interval `[-0.006510, 0.059896]`. B minus Q clamp-reduction
fraction was `0.354167`, interval `[0.135417, 0.562500]`; B minus G was
`0.104167`, interval `[-0.166667, 0.385417]`.

Although B exceeded Q on raw BF16 absolute mediation, all panel effects were
below the preregistered resolution floors, injection agreement failed, and
directional-control violations exceeded the limit. The frozen terminal rule
therefore does not support behavioral mediation.

## Resolution stratification

All 95 unique positive-panel execution pairs fell in the predicted `<1x`
stratum. They produced 73 crossings over all eight prompts, mean absolute
mediation `0.059932`, median `0.0625`, above-floor fraction `0.0`, and
injection-sign agreement `0.438356`. The `1–2x`, `2–4x`, `4–8x`, and `>=8x`
strata were empty. This is the central numerical limitation of the result.

## Telemetry and safety

- Runtime: Python 3.11.13, Torch 2.6.0, Transformers 4.57.3, NNsight 0.6.1,
  circuit-tracer 0.5.2, native Apple MPS/BF16; fallback disabled.
- Exact model revision: `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`.
- Exact transcoder revision: `fada11860ac1d337c1e41e9da308798405b94c8e`.
- Immutable assets: 2,087,816,677 bytes; exact allowlist passed; no download,
  network, authentication, or credential access.
- Peak MPS driver bytes: 2,865,397,760; peak process RSS: 949,518,336; minimum
  available memory: 15,322,333,184; swap growth: 0.
- 1,437 telemetry samples, nominal thermal state, zero violations, zero
  telemetry failures.
- No model/transcoder/tokenizer payload, cache, raw graph/adjacency, dense
  activation/gradient/derivative/hidden/logit tensor, journal, private path,
  secret, or oversized binary was committed.

## Validation

- Focused Stage 1G-v2 unit/synthetic integration: passed.
- Accepted Stage 1C-v3 journal/serialization/execution regressions: 49 passed;
  the split development environment lacked PyYAML for one config test, whose
  exact assertion passed directly in the pinned runtime environment.
- Ruff and format checks: passed.
- Strict scoped MyPy for Stage 1G runtime/science/orchestration: passed.
- Development and runtime dependency checks: passed.
- Immutable asset validation: passed.
- Staged secret/private-path/large-file scan: passed.
- Final standalone hostile-input bundle validator: passed twice in separate
  fresh processes, each reporting 1,499,036 bytes, 111 pairs, and 479/479/479
  call/journal/serialization equality.

Bundle sidecar SHA-256:
`73d51db14ab0fb9788082a024dbb151746a6ecf200bcc9bf6265e4f86c3ee45e`.

## Claim boundary and recommendation

This is a negative small-model PLT result on the pinned MPS/BF16 runtime, not a
paper-ready universal claim and not evidence that sub-floor effects are exactly
zero. Stage 1F remains `completed_stage1f_e1_not_supported`; Stage 1G-v1
remains `inconclusive_runtime`; simple critical-alpha calibration remains
retired.

Do not add critical-alpha, finite-probe, HVP, or another BF16 micro-probe stage
on this runtime. Either test the previously established q-based gate-discovery
result on a stronger/reference CLT or higher-precision runtime, or narrow the
project to gate-discovery methodology and negative findings. No paper Results
claim is authorized by this experiment.
