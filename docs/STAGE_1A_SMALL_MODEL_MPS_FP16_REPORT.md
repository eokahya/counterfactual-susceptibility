# Stage 1A-S small-model MPS/FP16 report

## 1. Verdict and readiness

```text
verdict: failed_runtime
stage1b_engineering_readiness: false
stage1b_empirical_claim_readiness: false
official_bf16_reproduction: pending
reference_clt_reproduction: pending
counterfactual_susceptibility_result: none
paper_results_readiness: false
```

The first real model-only MPS/FP16 forward failed the mandatory finite-logit
gate. Nothing downstream is reported as completed.

## 2. Git provenance

- Base: `4ef60d2b5f8120d5671afbf8400b61d66e291f4d`
- Isolated worktree: sibling `Antropic_Mech_Int_stage1a_small_model`
- Branch: `stage-1a-small-model-mps-fp16`
- Preflight/worker implementation commit:
  `27826a3b77854ef478d95b7879d34484e65c8ad1`
- Pre-run execution commit: none; smoke never passed, so the accepted protocol
  was never frozen.
- Accepted execution commit: none
- Blocker report commit:
  `a99de406e200cf3731cc64ec92c5a6dec47ddf85`. The later provenance-only
  finalization commit and branch-only push are recorded in the task response.

## 3. Protected history

The original workspace remained at
`stage-1a-mps-fp16@4ef60d2b5f8120d5671afbf8400b61d66e291f4d` with only its pre-existing
untracked `results/stage1a_t4_fp16/`. `main`, BF16, T4, prior MPS, and local
final-rerun refs were not changed. The seven protected T4 files remained byte
identical to their external backup; no old artifact was copied into the new
worktree or used as evidence.

## 4. Pinned environment

The newly created runtime is native arm64 CPython 3.11.13 with PyTorch 2.6.0,
NNsight 0.6.1, `circuit-tracer` 0.5.2, Transformers 4.57.3,
Hugging Face Hub 0.36.2, and safetensors 0.8.0. MPS is built and available;
real FP16 allocation/matmul passed; `PYTORCH_ENABLE_MPS_FALLBACK` was absent.
`pip freeze --all` matched the complete lock exactly and `pip check` passed.
The lock SHA-256 is
`2ddfaebbad636911f9033f7d46236ddf4b38513215011eb4fe1214fde7f583c4`.

## 5. Immutable assets

- Upstream:
  `decoderesearch/circuit-tracer@8f1e2438df612464e229e44c4a00ff637bf9379b`
- Model:
  `google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- Transcoder:
  `mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`
- Subfolder: `transcoder_all/width_16k_l0_small`

The model allowlist contains 8 runtime files totaling `575454257` bytes. The
PLT allowlist contains `config.yaml` and `layer_0.safetensors` through
`layer_17.safetensors`, totaling `1512362420` bytes. Combined actual and
projected bytes both equal `2087816677`. Every large-file local SHA-256 matched
the official LFS identity. Other widths, `features/**`, `config_orig.yaml`,
other models, and the full repository were not downloaded. Assets remain in a
project-external cache and no authentication value was printed or stored.

## 6. Upstream semantics

The pinned release explicitly supports Gemma 3 through NNsight. PLT layer `i`
maps `mlp.hook_in` to `pre_feedforward_layernorm` output and `hook_mlp_out` to
`post_feedforward_layernorm` output. Loaded JumpReLU is strict
`x * (x > threshold)`, so equality is inactive. Feature intervention receives
an absolute post-gate activation; upstream internally forms and decodes the
delta. Therefore the only valid suppression mapping is
`desired=(1-alpha)*baseline`.

NNsight is experimental and defaults to CUDA unless MPS is explicit. Upstream
attribution also calls MPS-unsupported `to_sparse()`. The isolated local
adapter keeps scientific dense values/indices on MPS and only COO graph
metadata on CPU. It was proven on tiny real MPS tensors, but never used on a
real model graph because the preceding model gate failed.

## 7. MPS operator audit

FP16 linear, matmul, batched matmul, RMSNorm, rotary sin/cos, softmax, gather,
scatter, nonzero, `index_put_`, `index_add_`, top-k, sort, masking, retained
autograd, VJP, JVP, and NNsight proxy assignment passed on MPS. Native
`to_sparse()` failed with its expected `NotImplementedError`. The explicit
CPU-COO metadata boundary had zero activation round-trip error; its synthetic
attribution reconstruction maximum absolute error was `0.00390625`, below the
frozen `0.005` FP16 tolerance.

## 8. Memory feasibility and actual telemetry

Pre-download assets were about 1.94 GiB. The conservative runtime projection
was `11316760320` bytes, including a 4 GiB runtime reserve and 6 GiB graph cap,
below independent 24 GiB MPS-driver and RSS limits. Swap growth was limited to
4 GiB, minimum available memory to 4 GiB, and thermal state to nominal/fair.

The failed model worker collected 6 one-second samples:

- MPS current peak: `551108608` bytes
- MPS driver peak: `1126842368` bytes
- process RSS peak: `607125504` bytes
- minimum available memory: `13509787648` bytes
- swap growth: `0` bytes
- thermal states: `nominal`

No memory, swap, or thermal rule was violated. MPS and RSS counters overlap and
were not added.

## 9. Model smoke

Prompt: `The capital of France is`. Token IDs:
`[2, 818, 5279, 529, 7001, 563]`; input shape `[1, 6]`, device `mps:0`.
All model parameters were MPS/FP16.

Hidden states remained finite through index 7. After decoder layer 7,
coordinate `[0, 0, 163]` added two finite FP16 values, `55520` and `13408`.
Their exact sum `68928` exceeds the FP16 maximum `65504`, producing positive
infinity. Subsequent layers propagated non-finite values; logits shape was
`[1, 6, 262144]` with `1572864/1572864` NaNs. BOS-only and `Hello` inputs also
first failed at hidden-state index 8, excluding a prompt-specific explanation.

The specification forbids retry for non-finite values and forbids silently
changing model, dtype, backend, device, or compute path. No retry occurred.

## 10. Loaded semantics

Not run. The model-only finite-logit gate failed before any PLT layer could be
loaded as runtime evidence. Source/header facts and toy JumpReLU probes are not
presented as loaded-model semantics.

## 11. Attribution

Not run. No graph was created, saved, or committed. There is no nonempty-graph
claim and no attribution batch attempt.

## 12. Intervention

Not run. No feature was selected, no baseline/no-op/half/full condition was
executed, and no behavioral result exists. The suppression formula is verified
only in pure unit tests, not as empirical model evidence.

## 13. Artifact validation

No canonical `results/stage1a_small_model_mps_fp16/` bundle was created because
success prerequisites failed. Preflight, asset, operator, and failed-attempt
JSON stayed under ignored `results/generated/`. Consequently there is no
accepted allowlist/checksum bundle or independent success-validator pass.

## 14. Tests and checks

Before asset download, the complete offline suite reported `354 passed,
5 skipped, 1 deselected`. Skips were optional dev-environment PyTorch/PyYAML
or model-gated cases and were not counted as empirical evidence. The separate
new runtime passed exact lock comparison, `pip check`, real MPS operator probes,
and NNsight assignment. Final Ruff, Ruff format, MyPy, deterministic math,
secret/large-file scans, full offline tests, and `git diff --check` are recorded
in the task handoff after the final candidate is checked.

## 15. Files and diff

Changes are isolated to Stage 1A-S config/manifest/environment docs, the new
runtime/probe/download/worker code and tests, current status/decision/log docs,
plus the README status link and one old-test fixture portability correction.
Historical runtime modules, reports, notebook outputs, and `paper/` are
unchanged. Relative to the exact base, the blocker-report candidate changes 18
files with 3470 insertions and 5 deletions.

## 16. Scientific boundary

This run shows only that the exact Gemma 3 270M BF16-trained checkpoint, when
converted to the frozen all-FP16 MPS execution identity, overflows its residual
stream before producing finite logits. It does not establish PLT semantics,
NNsight replacement correctness on the real model, attribution, intervention,
PLT/CLT equivalence, the official BF16 reproduction, Counterfactual
Susceptibility, mediation, a gate crossing, or any paper result.

## 17. Stage 1B recommendation

Do not proceed to Stage 1B. A future experiment would require a newly specified
identity (for example a different execution-dtype policy) and fresh scientific
review; changing dtype or introducing mixed precision inside this goal would
be an invalid silent substitution.

## 18. Safety and working tree

No CPU tensor fallback, CUDA, Colab, alternate model, financial action, quota
bypass, secret exposure, weight commit, cache commit, raw graph, PR, tag,
release, merge, or protected-branch change occurred. The final clean-tree and
branch-only push status are reported after commit/push verification.
