# Project status

Status date: 2026-08-24

No Counterfactual Susceptibility experiment has completed successfully, and
the paper Results section remains pending.

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

Stage 1A-S is development runtime validation, not a replacement for Stage 1A-R.
PLT and CLT results are not interchangeable. The exact Stage 1A-S assets remain
in an ignored, project-external cache; no weight or cache file is tracked. The
failed model-forward is runtime evidence only, not an attribution,
intervention, Counterfactual Susceptibility, or readiness result.

Current readiness:

```text
stage1b_engineering_readiness: false
stage1b_empirical_claim_readiness: false
official_bf16_reproduction: pending
reference_clt_reproduction: pending
counterfactual_susceptibility_result: none
paper_results_readiness: false
```

See `docs/STAGE_1A_SMALL_MODEL_MPS_FP16_PLAN.md` for the fail-closed execution
gates. Historical reports remain unchanged and retain their original scope.
