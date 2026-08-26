# Project status

Status date: 2026-08-26

No valid Counterfactual Susceptibility intervention experiment has completed,
and the paper Results section remains pending. Stage 1C froze a valid first
prospective prediction, but its canonical intervention artifact failed closed.
Stage 1C-v2 repaired that serialization path, then stopped before prediction
freeze when the held-out baseline selection hit its frozen historical-endpoint
guard. Stage 1C-v3 corrected that exclusion policy and froze an independently
validated preregistered prediction, but its sole canonical process failed
before the first intervention API call because the frozen adapter lacked the
required baseline-remeasurement method.

## Current experiment classes

- **Stage 1A-R — Gemma 2 / reference CLT validation:** pending. The official
  native-BF16 reproduction has not completed.
- **T4/FP16 hardware adaptation:** historical but not acceptance evidence. Its
  preserved small artifacts have invalid attempt-level peak-memory provenance.
- **Gemma 2 MPS/FP16 hardware adaptation:** blocked. A real model-only forward
  passed, but replacement-runtime loading failed and measured MPS driver/swap
  peaks violated the conservative 32 GiB host budget.
- **Stage 1A-S — local small-model MPS/FP16 pilot:** `failed_runtime` at the
  first real model-only gate. Immutable upstream/model/transcoder revisions,
  the native MPS environment, operator probes, memory feasibility, and the
  exact allowlisted asset download passed. The Gemma 3 270M residual stream
  overflowed FP16 after decoder layer 7, producing non-finite logits. The
  no-retry policy stopped escalation before loaded PLT semantics, NNsight
  replacement, attribution, or intervention.
- **Stage 1A-S-BF16 — local small-model MPS/BF16 recovery:**
  `completed_small_model_mps_bf16_pilot`. The exact Gemma 3 270M model, all 18
  selected PLTs, NNsight replacement runtime, finite attribution, deterministic
  baseline-active feature selection, baseline repeat, no-op, half suppression,
  and full ablation passed on Apple MPS/BF16. The 13-file small artifact bundle
  passed the independent validator. This is engineering runtime evidence only,
  not the reference reproduction or a Counterfactual Susceptibility result.
- **Stage 1B — measurement primitives:**
  `completed_stage1b_measurement_primitives`. On the frozen Stage 1A-S-BF16
  runtime, the chunked exact loaded-JumpReLU inactive-feature scanner matched
  its ephemeral dense oracle at all three frozen chunk sizes. The independent
  targeted VJP path accepted no graph/edge input, and its reconstructed raw
  edges passed the frozen 64-pair prospective validation. This is measurement-
  tool engineering evidence only; it contains no inactive-target score,
  suppression sweep, gate crossing, behavior, mediation, Gemma 2, or CLT
  result.
- **Stage 1C — first prospective prediction:** `failed_runtime` with
  `inconclusive_runtime`. The baseline-only phase passed and was frozen at
  `6ec950d93fe1215fdcfee68c87e1f58a23a78ae8` before intervention. It selected
  12 primary, 8 near-boundary, and 8 directional-control pairs from 30,283
  eligible source-target pairs. The one allowed canonical process reported 228
  source-suppression API calls, but a frozen cleanup aliasing bug erased every
  point-level sweep row before serialization. The assembler and standalone
  validator rejected the incomplete bundle. No retry, post-outcome code
  change, susceptibility result, or gate-crossing result was accepted.
- **Stage 1C-v2 — held-out prospective prediction:** `blocked_engineering`
  with no scientific outcome. The detached serialization recovery passed a
  nonempty synthetic worker/assembler/validator chain, and the exact Germany
  prompt passed immutable offline tokenizer and MPS/BF16 preflight. A fresh
  baseline-only selection then overlapped at least one historical v1 selected
  endpoint and the frozen post-selection guard stopped publication. No pair
  was filtered or reranked after this result; no prediction manifest,
  pre-intervention commit, canonical intervention, suppression API call, or
  scientific artifact bundle exists.
- **Stage 1C-v3 — preregistered prospective prediction:** `failed_runtime`
  with `inconclusive_runtime`. The Norway prompt and authenticated 28-pair
  exact denylist were frozen before baseline. The exact-pair mask removed 16
  of 39,235 eligible pairs before ranking; all endpoint-only overlaps remained
  audit metadata. The validated manifest froze 12 primary, 8 near-boundary,
  and 8 directional rows with zero exact historical overlap. The sole
  canonical process then failed during baseline remeasurement because the
  intervention adapter lacked `measure_states`. It made zero suppression API
  calls and serialized zero points. No retry or post-freeze protocol change
  occurred.

Stage 1A-S is development runtime validation, not a replacement for Stage 1A-R.
PLT and CLT results are not interchangeable. The exact Stage 1A-S assets remain
in an ignored, project-external cache; no weight or cache file is tracked. The
failed model-forward is runtime evidence only, not an attribution,
intervention, Counterfactual Susceptibility, or readiness result.

Current readiness:

```text
stage1b_measurement_primitives: completed
stage1c_first_prediction: failed
stage1c_scientific_outcome: inconclusive_runtime
stage1c_first_prediction_readiness: false
stage1c_v2_prospective_prediction: blocked_engineering
stage1c_v2_scientific_outcome: none
stage1c_v3_prospective_prediction: failed_runtime
stage1c_v3_scientific_outcome: inconclusive_runtime
stage1b_empirical_claim_readiness: false
counterfactual_susceptibility_result: none
gate_crossing_result: none
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

See `docs/STAGE_1B_MEASUREMENT_PRIMITIVES_REPORT.md` for the completed Stage 1B
engineering evidence and
`docs/STAGE_1C_FIRST_PROSPECTIVE_PREDICTION_REPORT.md` for the Stage 1C frozen
prediction, invalidated canonical attempt, and exact claim boundary, and
`docs/STAGE_1C_V2_HELDOUT_PROSPECTIVE_PREDICTION_REPORT.md` for the v2
serialization recovery and held-out selection blocker, and
`docs/STAGE_1C_V3_PREREGISTERED_PROSPECTIVE_PREDICTION_REPORT.md` for the v3
exact-pair-masked prediction and canonical runtime blocker.
Historical reports remain unchanged and retain their original scope.
