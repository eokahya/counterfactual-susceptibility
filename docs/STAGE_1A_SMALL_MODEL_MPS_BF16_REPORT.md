# Stage 1A-S-BF16 local MPS/BF16 runtime-validation report

Status: `completed_small_model_mps_bf16_pilot`

Date: 2026-08-24

## Outcome

The isolated Stage 1A-S-BF16 pilot completed from clean, previously pushed
execution commit `6a5c21027fbb6b83e34c39db75987b0ce5b72d17`. The canonical
accepted attempt used attribution batch 64 and passed; no OOM retry or fallback
was used. The independent artifact validator passed the 13-file, 996 KiB bundle
under `results/stage1a_small_model_mps_bf16/`.

This is local small-model runtime-validation evidence only. It is not the
official Gemma 2/CLT BF16 reproduction, a reference CLT result, CUDA
equivalence, PLT/CLT equivalence, Counterfactual Susceptibility evidence, a
gate-crossing result, mediation evidence, or paper Results evidence.

```text
verdict: completed_small_model_mps_bf16_pilot
stage1b_engineering_readiness: true
stage1b_empirical_claim_readiness: false
official_bf16_reproduction: pending
reference_clt_reproduction: pending
counterfactual_susceptibility_result: none
paper_results_readiness: false
```

The protected Stage 1A-S MPS/FP16 `failed_runtime` result remains unchanged.
BF16 recovery is a separate experiment identity and does not invalidate or
rewrite the FP16 overflow observation.

## Frozen identity

- Base commit: `3baf39a5ac81e172d11d22a6de332dee80a21079`
- Execution commit: `6a5c21027fbb6b83e34c39db75987b0ce5b72d17`
- Branch: `stage-1a-small-model-mps-bf16`
- Upstream: `decoderesearch/circuit-tracer` v0.5.2 at
  `8f1e2438df612464e229e44c4a00ff637bf9379b`
- Model:
  `google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- Transcoder:
  `mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`
- Subfolder: `transcoder_all/width_16k_l0_small`
- Backend/device/dtype: NNsight / Apple MPS / `torch.bfloat16`
- Host: Apple M2 Max, 32 GiB unified memory, native arm64 macOS
- Runtime: CPython 3.11.13, PyTorch 2.6.0, NNsight 0.6.1,
  Transformers 4.57.3, Hugging Face Hub 0.36.2, safetensors 0.8.0
- Environment lock SHA-256:
  `84f502dccd0f1b3a686c6cf5266f3c6e1103f256a8206dd37c748413e7cffee0`
- Config SHA-256:
  `429662d9598ba24d769aca9239271183c894eb98f85c76f95a38b5b78df65c6d`

The exact model and selected transcoder snapshot contained 8 and 19 consumed
files respectively, totaling 2,087,816,677 bytes. Every expected LFS SHA-256
matched. The run used the already verified project-external cache in offline
mode; it performed no download, authentication, paid compute, or network
execution.

## Progressive gate results

### Numerical and model recovery

The operator regression reproduced positive infinity in FP16 at the known
scale. Native MPS/BF16 produced finite `69120`; the FP32 reference was `68928`,
for absolute error 192, relative error 0.002786, and one BF16 ULP. All frozen
forward, indexing, retained-autograd, JVP/VJP, NNsight replacement, and
sparse-boundary probes passed.

BOS-only, `Hello`, and `The capital of France is` remained finite through all
18 decoder layers and logits. At the prior FP16 failure coordinate
`[0,0,163]` after layer 7, BF16 operands `55040` and `13376` produced finite
observed output `68608`; the scalar FP32 diagnostic sum was `68416`.

The separate CPU/FP32 diagnostic ran only after the MPS model process exited
and never served as fallback. Across the three prompts, final-logit cosine was
0.999969 or higher, normalized L2 error was 0.015739 or lower, top-1 agreed,
and top-10 overlap was 10/10.

### Loaded runtime semantics

One real layer-0 PLT and all 18 selected PLTs loaded with MPS/BF16 runtime
parameters. The full lazy set represented 294,912 features. Source-mandated
FP32 rotary buffers were explicitly enumerated; there was no outer autocast,
FP16 conversion, hidden CPU model computation, or runtime monkeypatch.

Loaded JumpReLU was independently recomputed with each layer's own learned
threshold vector and strict `preactivation > threshold` gating. Threshold
equality was inactive, maximum gate discrepancy was zero, and real active and
inactive examples were present. The accepted selected feature had
preactivation/activation `1960`, threshold `146`; the checked inactive feature
had preactivation `-1224`, threshold `156`, activation zero.

### Attribution and deterministic selection

The accepted graph used 2,152 active/selected PLT features and had adjacency
shape `[2276,2276]`, 1,454,640 nonzero edges, 10 logit nodes, 108 error nodes,
and 6 input nodes. All scientific dense values, vectors, gradients, and
intervention values stayed on MPS/BF16. Only the audited bit-exact BF16 COO and
graph-ranking metadata crossed to CPU. No raw graph or adjacency was
persisted.

The primary frozen selection rule chose layer 17, position 5, feature 1191,
with baseline activation `1960` and absolute direct score `2.484375`. The
artifact retained a small derived audit table rather than the raw graph: all
2,152 selected features were accounted for, comprising 370 valid final-token
candidates and 1,782 non-final exclusions, with no non-finite or nonpositive
candidate exclusion. The independent validator recomputed the same winner
with the stable numeric tie-break.

### Intervention controls

Baseline, baseline repeat, alpha-zero no-op, half suppression, and full
ablation all used the same baseline-active tuple and
`freeze_attention=true`. The exact absolute value passed to upstream was

```text
desired = (1 - alpha) * baseline
```

Observed BF16 values were `1960`, `980`, and `0` for alpha `0`, `0.5`, and
`1.0`. Desired and sent values were identical on `mps:0`; all outputs were
finite. Raw-to-frozen baseline, baseline-repeat, and alpha-zero no-op
normalized-L2 errors were all zero. Half and full suppression differed from
baseline by 0.075792 and 0.158862 normalized L2 respectively. Those two values
show only that the runtime intervention path responds. Maximum absolute logit
differences from baseline were `0`, `2.03125`, and `4.03125` for no-op, half,
and full conditions; raw baseline and repeat maxima were also zero. No semantic
direction, behavioral importance, susceptibility, or causal-mechanism claim is
made.

## Resource and safety evidence

The accepted worker ran from `2026-08-24T16:50:31Z` to
`2026-08-24T16:50:43Z`; supervisor wall time was about 13.37 seconds. Attempt
peaks were:

- MPS current allocation: 689,690,112 bytes
- MPS driver allocation: 3,009,298,432 bytes
- process RSS: 1,103,904,768 bytes
- minimum available memory: 11,848,892,416 bytes
- swap growth: 0 bytes
- thermal state: nominal

MPS and RSS are overlapping unified-memory signals and were not summed.
Attempt peaks dominated every recorded stage peak. Telemetry had no sampling
failure, safety violation, timeout, thermal warning, or process-group kill.

The accepted worker recorded the exact execution SHA, branch, and a clean
worktree. `PYTORCH_ENABLE_MPS_FALLBACK` was absent, runtime was offline, and
the first batch-64 attempt passed. No retry occurred.

## Validation and preserved attempts

Before the final execution commit, Ruff check and format, strict MyPy, 391
offline tests, and 33 explicit real-MPS/BF16 target tests passed. The final bundle
passed the standalone validator, exact checksum coverage, finite JSON parser,
allowlist, regular-file/single-link/size rules, secret and machine-local path
scan, immutable snapshot identities, telemetry hierarchy, and Git ancestry.
The checksum-manifest SHA-256 is
`ea7bd6db0ceca579f4b62aba530d419a3c0bc1e3ee98abff5ccf6938062d4b95`.

All 12 empirical attempts remain represented in `attempts.json`, including
three non-scientific engineering failures: an initially incomplete cache path,
an inference-tensor validation-mode error, and an incorrectly broadcast
threshold vector in a smoke validator. The first batch-64 runtime pass from
`0560c6549cc978fa428f933cd51c817da1808ab7` is also preserved with disposition
`invalidated_missing_required_maximum_absolute_difference`; its artifact lacked
one mandatory compact control field and was never published as canonical.
The field, exact-zero no-op max-absolute tolerance, validator, and regression
fixture were frozen in the final execution commit before the fresh canonical
run. No existing tolerance or frozen scientific parameter changed, and nothing
changed after canonical accepted outputs were observed.

No model/transcoder weight, tokenizer payload, cache, secret, raw graph,
adjacency, tensor dump, archive, notebook, or large artifact is tracked. The
paper Results section was not changed.

## Decision

Stage 1A-S-BF16 has completed its narrowly scoped local runtime-validation
objective. Engineering may proceed to Stage 1B design work, but empirical
claim work remains blocked on the pending official/reference reproductions and
the separately specified Counterfactual Susceptibility experiment.
