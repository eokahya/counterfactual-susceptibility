# Stage 1A-S local small-model MPS/FP16 pilot plan

Status: `FAILED_RUNTIME_AT_MODEL_FORWARD`

## Purpose and claim boundary

Stage 1A-S is a separate local runtime-validation experiment. Its purpose is
to establish whether a small Gemma 3 pretrained model, a selected GemmaScope-2
PLT subset, and the NNsight backend can run on Apple MPS/FP16 with trustworthy
loaded threshold semantics, attribution, and feature intervention.

It is not Stage 1A-R, a Gemma 2 or CLT reproduction, a CUDA/T4/BF16 result, a
PLT/CLT equivalence result, a Counterfactual Susceptibility result, evidence of
a gate crossing or mediation, or evidence ready for the paper Results section.
The only successful status permitted is:

```text
completed_small_model_mps_fp16_pilot
```

## Git and protected-history boundary

| Item | Identity |
| --- | --- |
| Base commit | `4ef60d2b5f8120d5671afbf8400b61d66e291f4d` |
| Branch | `stage-1a-small-model-mps-fp16` |
| Isolated worktree | sibling worktree `Antropic_Mech_Int_stage1a_small_model` |
| Protected T4 evidence | untracked `results/stage1a_t4_fp16/`, unchanged and backup-verified |

The existing `main`, Stage 1A-R/BF16, T4/FP16, MPS/FP16, and T4 final-rerun
branches are immutable for this goal. Generic safety ideas from the local T4
hardening commit may be reimplemented in new Stage 1A-S modules, but T4-specific
code and its blocker report will not be cherry-picked.

## Immutable runtime identity

Official source and Hugging Face metadata resolved the identities to:

```text
upstream: decoderesearch/circuit-tracer@8f1e2438df612464e229e44c4a00ff637bf9379b
model: google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
transcoder: mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e
subfolder: transcoder_all/width_16k_l0_small
backend: nnsight
device: mps
execution_dtype: float16
host: Apple M2 Max, 32 GiB unified memory, arm64 macOS
```

No mutable revision may be consumed by an accepted run. No alternate model,
transcoder, backend, accelerator, or dtype may silently replace this identity.

## Observed terminal gate

Phases 1-4 passed. The exact allowlisted assets were downloaded and verified
in a project-external cache. Stage 5.1 then failed its finite-logit requirement.
The exact model on MPS/FP16 first overflowed after decoder layer 7: at one
coordinate the finite residual operands were `55520` and `13408`, whose exact
sum `68928` exceeds the FP16 maximum `65504`. The resulting positive infinity
propagated to all-NaN logits. BOS-only, one-word, and the frozen-candidate prompt
all first became non-finite at hidden-state index 8.

The retry policy forbids a retry or parameter change for non-finite values.
Therefore the one-layer PLT, full PLT, NNsight replacement, attribution,
intervention, pre-run freeze, and accepted-pilot gates were not executed. The
terminal classification is `failed_runtime`, not a completed pilot.

## Hard-gated execution plan

1. **Repository and experiment-class gate.** Verify protected refs and
   artifacts, isolate the worktree, and document Stage 1A-S without changing
   historical reports or the manuscript.
2. **Current upstream and asset-metadata gate.** Audit official
   `circuit-tracer`, Gemma 3/NNSight support, attribution and intervention
   semantics, strict threshold behavior, layer mapping, lazy loading, MPS/CPU
   transfers, exact model/transcoder SHAs, and the exact selected PLT file set.
3. **Native environment and no-download MPS gate.** Create a new native-arm64
   Python 3.11 environment, pin the audited dependencies, fail if MPS fallback
   is enabled, and run the exact bounded operator/device/autograd probes before
   real assets are downloaded.
4. **Download and memory-feasibility gate.** Produce a machine-readable exact
   allowlist and conservative 32 GiB estimate. Enforce independent limits of
   24 GiB MPS driver allocation, 24 GiB process RSS, 4 GiB swap growth, a
   4 GiB available-memory reserve, and nominal/fair thermal state.
5. **Authorized immutable asset gate.** Only after gates 1-4 pass, download the
   exact model runtime files and exact selected 16K PLT runtime subset through
   official Hugging Face APIs at immutable revisions. Reject other widths,
   model sizes, PT/IT variants, feature visualizations, and full-repository
   payloads.
6. **Progressive real-runtime gate.** In fresh supervised workers, run
   model-only MPS/FP16 forward, one loaded PLT layer and semantics, the full
   selected PLT set, NNsight replacement construction, small attribution, and
   feature-intervention smoke. A critical failure stops escalation.
7. **Freeze gate.** Before accepted outputs are inspected, commit the complete
   environment/config/runner/worker/validator/tests/schema protocol as a clean
   pre-run commit. The feature is selected deterministically, never by hand.
8. **Accepted-pilot gate.** Run baseline repeat, no-op, half suppression, and
   full ablation with the frozen rule and verify independently that every API
   argument equals `(1 - alpha) * baseline_activation` within dtype tolerance.
9. **Publication gate.** Commit only allowlisted small derived artifacts after
   the independent validator, full tests, safety scans, provenance checks, and
   final diff pass. Push only the new branch; do not merge, create a PR, tag,
   or release.

## Device and fallback policy

`PYTORCH_ENABLE_MPS_FALLBACK` must be absent or disabled. Scientific model,
dense PLT, gradient, attribution, and intervention tensors must remain on MPS.
CPU is allowed only at separately declared metadata, file-I/O, tokenization,
sparse-index, or proven upstream metadata-only boundaries. Any unexpected CPU
tensor during a real stage is a hard failure. CUDA, Colab, remote execution,
Docker, VM, and CPU tensor-computation fallback are prohibited.

## Retry policy

No generic exception, unsupported operator, asset failure, non-finite value,
thermal/memory-policy breach, or runtime-loading failure permits a retry. Only
a verified recoverable MPS OOM during attribution backward batches may retry in
a fresh process, preserving all other settings:

```text
64 -> 32 -> 16
```

## Accepted protocol target

The accepted configuration is frozen only after source/API audit and smoke
evidence. Its initial target is the exact immutable model/transcoder/subfolder,
NNsight/MPS/FP16, prompt `The capital of France is`, ten logits, desired logit
probability 0.95, at most 4096 feature nodes, batch 64, and alphas
`[0.0, 0.5, 1.0]`. Any engineering-only adjustment must be justified before
the pre-run commit and may not be chosen after viewing accepted outputs.

Feature selection is preregistered as the active non-error feature at the final
token with highest absolute direct contribution to the baseline top-logit node,
with deterministic numeric tie-breaking. If the audited API cannot expose
direct contribution, the predeclared fallback is highest absolute active
feature activation with the same deterministic tie-breaking. The fallback
choice must be frozen before execution.

## Small artifact boundary

The isolated namespace is `results/stage1a_small_model_mps_fp16/`. Only the
following small derived files may be accepted:

```text
environment_manifest.json
asset_manifest.json
preflight_summary.json
operator_probe_summary.json
model_forward_summary.json
loaded_semantics_summary.json
attribution_summary.json
intervention_summary.json
memory_timing_summary.json
attempts.json
run_manifest.json
checksums.sha256
```

No weights, tokenizer payload, cache, raw graph, tensor dump, secret, private
path, large adjacency array, or mutable revision may enter the bundle.

## Completion boundary

Preparation, a model-only forward, smoke output, or a partially populated
artifact directory is not completion. Completion requires every scientific,
device, memory, thermal, provenance, checksum, and security gate plus a safe
push of only the new branch. Until then, both Stage 1B readiness flags are
false and all reference reproductions remain pending.
