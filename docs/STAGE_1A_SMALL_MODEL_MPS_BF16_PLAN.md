# Stage 1A-S-BF16 local small-model MPS/BF16 recovery plan

Status: `PRE_RUN_FREEZE_READY`

## Purpose and experiment identity

Stage 1A-S-BF16 is a new local runtime-validation experiment. It asks whether
the exact BF16-trained Gemma 3 270M checkpoint, the pinned 18-layer GemmaScope-2
PLT subset, and the NNsight backend can execute natively on Apple MPS/BF16 with
finite model states, source-faithful loaded JumpReLU semantics, attribution,
and absolute feature intervention.

```text
experiment_class: stage1a_small_model_mps_bf16_pilot
success_status: completed_small_model_mps_bf16_pilot
model: google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
transcoder: mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e
transcoder_subfolder: transcoder_all/width_16k_l0_small
backend: nnsight
device: mps
execution_dtype: torch.bfloat16
host: Apple M2 Max, 32 GiB unified memory, arm64, macOS
```

The exact upstream and dependency versions are inherited provisionally from
the protected FP16 implementation and must be re-audited before execution.
Immutable model and transcoder revisions may not change.

## Protected negative result and claim boundary

The prior `stage-1a-small-model-mps-fp16` result remains
`failed_runtime`. Its observed residual addition `55520 + 13408 = 68928`
exceeded the FP16 maximum `65504`, first produced positive infinity after
decoder layer 7, and propagated to non-finite logits for all three tested
prompt classes. This is protected negative evidence and will not be edited,
reclassified, or presented as a BF16 result.

Stage 1A-S-BF16 is not Stage 1A-R, an official reproduction, a Gemma 2 or CLT
result, CUDA equivalence, PLT/CLT equivalence, Counterfactual Susceptibility,
gate-crossing discovery, mediation, or paper Results evidence. A fully passing
pilot permits only:

```text
verdict: completed_small_model_mps_bf16_pilot
stage1b_engineering_readiness: true
stage1b_empirical_claim_readiness: false
official_bf16_reproduction: pending
reference_clt_reproduction: pending
counterfactual_susceptibility_result: none
paper_results_readiness: false
```

Any missing acceptance criterion keeps both Stage 1B readiness flags false.

## Git and publication boundary

| Item | Frozen identity |
| --- | --- |
| Base commit | `3baf39a5ac81e172d11d22a6de332dee80a21079` |
| Branch | `stage-1a-small-model-mps-bf16` |
| Worktree | external isolated BF16 worktree; machine-local path not recorded |
| Prior FP16 branch | protected and unchanged |
| Old T4 artifacts | protected, backup-hash verified, never staged |

Only the new BF16 branch may be committed and pushed. No existing branch may
be reset, rebased, merged, rewritten, or pushed. No PR, tag, release, or merge
is permitted. The accepted protocol must be committed and pushed before its
outputs are inspected. Accepted artifacts may be committed only after the
independent validator passes.

## Execution and fallback policy

`PYTORCH_ENABLE_MPS_FALLBACK` must be absent or false. Accepted model, dense
PLT, replacement-runtime, attribution, gradient, and intervention computation
must use MPS and `torch.bfloat16`. There is no FP16, FP32, autocast/mixed
precision, CPU model-compute, CUDA, Colab, remote, Docker, VM, alternate model,
or alternate backend fallback. Non-finite values may not be clamped, rescaled,
or hidden.

CPU is limited to documented metadata, file I/O, tokenization, checksums, JSON,
telemetry, the isolated sparse-COO metadata boundary, and a short separately
executed CPU/FP32 diagnostic reference after the MPS/BF16 model worker exits.
The FP32 diagnostic is validation only and can never rescue accepted execution.

## Ordered hard gates

1. **Repository and experiment-class gate.** Verify exact base/worktree,
   protected refs and artifacts, and document this experiment before runtime
   implementation.
2. **Source and environment gate.** Re-audit the pinned upstream/dependencies,
   hidden casts, autocast, model/PLT loading, NNsight proxy assignment,
   attribution/intervention, and sparse metadata path. Create a distinct native
   arm64 CPython 3.11 environment and exact lock.
3. **Real MPS/BF16 operator gate.** Probe every critical forward/backward,
   indexing, threshold, retained-autograd, JVP/VJP, NNsight replacement, and
   sparse-boundary operation. Reproduce FP16 overflow and require finite BF16
   plus FP32 reference under preregistered tolerances.
4. **Asset and memory gate.** Revalidate the exact existing snapshot allowlists,
   sizes, hashes, and immutable identities. Download only a missing/corrupt
   allowlisted file at the exact revision. Require a conservative memory model
   and tested child-process supervisor.
5. **Model-only recovery gate.** In a fresh MPS/BF16 worker run BOS-only,
   `Hello`, and `The capital of France is`; record every decoder state and
   logits. Layer 7 and all later states must be finite. After that process exits,
   run the separate short CPU/FP32 diagnostic under thresholds frozen in code.
6. **Loaded semantics gate.** Load one deterministic real PLT layer, verify
   bias-inclusive preactivation, raw threshold, strict `>` JumpReLU, real active
   and inactive examples, finite reconstruction, dtype/device, and independent
   BF16-aware reference agreement.
7. **Full runtime gate.** Load all 18 PLTs through the audited lazy/cache path,
   construct NNsight replacement runtime, verify feature access and absolute
   intervention semantics, and remain within memory/fallback policy.
8. **Engineering smoke gate.** Produce a nonempty finite attribution graph and
   run deterministic active-feature baseline repeat, no-op, half suppression,
   and full ablation. Smoke remains ignored/generated engineering evidence.
9. **Freeze gate.** Commit and push environment pins, plan/decisions, probes,
   config, runner, worker, supervisor, validator, tests, and artifact schema
   without accepted empirical outputs. Record this SHA as the execution commit.
10. **Accepted pilot gate.** Run exact frozen config from the execution commit,
    independently validate all controls and artifacts, then publish only small
    allowlisted derived summaries if every criterion passes.

A hard-gate failure stops escalation and produces the deepest verified
fail-closed terminal class. The only retry is a fresh-process attribution retry
after verified MPS OOM: smoke may use one preregistered smaller batch; accepted
may use `64 -> 32 -> 16`. No other failure permits retry.

## Preregistered numerical controls

The synthetic overflow regression records actual rounded operands and results:

```text
FP16: overflow/non-finite at the known scale
BF16 on MPS: finite, correct sign and expected magnitude
FP32 reference: finite
```

BF16 is not expected to match FP32 bit-for-bit. Absolute/relative tolerances,
model diagnostic thresholds, loaded-semantics tolerances, baseline/no-op
tolerances, and intervention-value tolerances must be frozen in code before
the corresponding real stage and may not be loosened after outputs are seen.

The project suppression convention remains:

\[
a_j^{\mathrm{desired}}=(1-\alpha)a_j^{\mathrm{baseline}}.
\]

Accepted conditions are baseline repeat, `alpha=0.0`, `alpha=0.5`, and
`alpha=1.0`. Feature selection is frozen before effects are viewed: highest
absolute direct contribution to the baseline top-logit node at the final token
among active non-error PLT features, with stable numeric tie-breaking; only a
predeclared API-capability fallback to highest absolute final-token baseline
activation is allowed.

## Safety limits and artifact boundary

Independent fail-closed limits remain 24 GiB MPS driver allocation, 24 GiB
process RSS, 4 GiB swap growth, 4 GiB minimum available-memory reserve, and
nominal/fair thermal state. MPS and RSS overlap and are never summed. Every real
stage runs in a child worker; stage peaks and attempt peaks are retained, and
attempt peaks must dominate every stage peak for the same metric.

The isolated accepted namespace is
`results/stage1a_small_model_mps_bf16/`. Only small JSON summaries and
`checksums.sha256` from the binding allowlist may be committed. Weights,
tokenizer payloads, caches, raw graphs, adjacency, tensor dumps, notebooks,
secrets, mutable revisions, symlinks, hardlinks, special files, and large
artifacts are forbidden.

## Completion boundary

Preparation, passing probes, model-only recovery, smoke, a clean execution
commit, or partial output is not completion. Completion requires all 23 success
criteria in the binding specification, an independent fail-closed validator,
full tests/scans, protected-history verification, a clean BF16 branch, and a
branch-only push. Otherwise the report records the exact terminal blocker and
keeps readiness false.

## Pre-run gate record (2026-08-24)

Phases 0–8 passed before the accepted protocol was frozen. This record is
engineering/preflight evidence, not the accepted pilot result.

- The real operator probe reproduced positive infinity for the known FP16
  scale and produced finite MPS/BF16 `69120` versus FP32 `68928` (relative
  error `0.0027855153`, one BF16 ULP). All frozen forward, indexing,
  retained-autograd, JVP/VJP, NNsight replacement, and sparse-boundary probes
  passed with fallback absent.
- The exact assets revalidated at 8 model files (`575454257` bytes) and 19
  transcoder-subset files (`1512362420` bytes), total `2087816677` bytes, with
  every official LFS SHA-256 matching. No download or authentication occurred.
- All three model-only prompts remained finite through every decoder layer.
  The prior layer-7 failure coordinate recovered at a finite BF16 value. The
  separate CPU/FP32 diagnostic achieved cosine `0.999969` or better, top-1
  agreement, and top-10 overlap `10/10` for each prompt.
- One real layer-0 PLT passed strict loaded JumpReLU, equality-inactive,
  reconstruction, selected-value FP32 shadow, active/inactive, device/dtype,
  and finite checks. All 18 lazy PLTs and the NNsight replacement runtime then
  loaded as MPS/BF16 with no monkeypatch and no safety violation.
- Engineering smoke produced a finite nonempty 512-feature attribution graph.
  The frozen primary rule selected a baseline-active feature; raw baseline,
  repeated frozen baseline, and alpha-zero no-op agreed exactly. Half and full
  suppression received exact absolute MPS/BF16 values and remained finite.

Three non-scientific pre-accepted failures remain preserved under ignored
generated paths: an incomplete cache argument before model loading, an
inference-tensor validation-mode error, and a validator broadcasting one
selected layer's thresholds across all layers. The latter two received narrow
regression-tested corrections; no asset, prompt, dtype, tolerance, selection
rule, or accepted scientific parameter changed. The accepted run remains
unexecuted until this implementation, validator, schema, tests, and lock are
committed and pushed as one clean execution commit.

## Second pre-run freeze correction (2026-08-24)

The first clean batch-64 accepted runtime pass was invalidated before
publication because its compact intervention artifact omitted the explicit
maximum baseline/no-op logit-difference field required by the binding
specification. The full attempt and its rejected bundle remain preserved under
ignored generated paths. No result was hand-edited or promoted.

Before a fresh accepted run, the worker now records maximum absolute logit
difference for raw baseline, baseline repeat, no-op, half suppression, and
full ablation. Raw baseline, baseline repeat, and alpha-zero no-op use a frozen
exact-zero BF16 tolerance. The independent validator requires every field and
the final attempt manifest distinguishes the invalidated pass from exactly one
canonical accepted pass. A regression fixture proves a missing field fails.
These corrections do not change the prompt, model, PLT subset, backend,
device, dtype, feature-selection rule, attribution settings, intervention
mapping, or normalized-L2 tolerance. A new pre-run commit and push are required
before the fresh accepted run.
