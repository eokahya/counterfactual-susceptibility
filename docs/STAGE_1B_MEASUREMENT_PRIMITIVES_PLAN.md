# Stage 1B measurement primitives plan

Status date: 2026-08-24

This plan is the tracked implementation contract for Stage 1B. It is based on
`fb2fc158b45c842743804040e4e273776e666a48`, the completed Stage 1A-S-BF16
runtime, and the binding Goal specification. It does not report a
Counterfactual Susceptibility result, a gate crossing, behavioral importance,
mediation, a reference CLT reproduction, or a paper result.

## Frozen runtime identity

- branch: `stage-1b-measurement-primitives`
- base: `fb2fc158b45c842743804040e4e273776e666a48`
- accepted Stage 1A execution ancestor:
  `6a5c21027fbb6b83e34c39db75987b0ce5b72d17`
- circuit-tracer: version 0.5.2 at
  `8f1e2438df612464e229e44c4a00ff637bf9379b`
- model: `google/gemma-3-270m` at
  `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- PLT: `mwhanna/gemma-scope-2-270m-pt` at
  `fada11860ac1d337c1e41e9da308798405b94c8e`, subset
  `transcoder_all/width_16k_l0_small`
- NNsight 0.6.1, PyTorch 2.6.0, native arm64 CPython 3.11.13
- Apple MPS, `torch.bfloat16`, no autocast and no
  `PYTORCH_ENABLE_MPS_FALLBACK`
- accepted prompt identity: `pilot`, text `The capital of France is`

All model and PLT files must resolve from the already verified project-external
offline cache. No worker may authenticate, download, or change a revision.

## Scientific contract

For a typed feature `(layer, position, feature_id)`, the loaded PLT defines the
bias-inclusive preactivation and strict JumpReLU state

```text
z = linear(mlp.hook_in, W_enc, b_enc)
a = z * 1[z > tau]
```

Equality is inactive. Scanner eligibility requires loaded inactivity, exact
`a == 0`, finite BF16-measured values, `z <= tau`, and `margin = tau - z >= 0`.
Already measured BF16 scalars may be converted to Python floats for bounded
ordering and JSON only; projections are never recomputed on CPU or in FP32.

Pinned upstream graph matrices use target rows and source columns. A raw,
unnormalized feature edge is

```text
E[j -> i] = a_j * J_ij
J_ij = partial z_i / partial a_j
```

under the frozen attribution convention. The source is post-gate activation;
the target is pre-gate preactivation, so the target gate derivative is
excluded. The targeted path must not accept, read, or divide an adjacency or
edge value. Graph edges enter only the independent prospective validator.

The source audit is pinned to:

- `circuit_tracer/graph.py:46-68` for target-row/source-column orientation;
- `transcoder/single_layer_transcoder.py:120-125` for bias-inclusive encoding;
- `transcoder/activation_functions.py:11-35` for strict loaded JumpReLU;
- `attribution/context_nnsight.py:19-42,92-122,151-214` for reverse-mode
  contraction and activation-scaled source decoder vectors;
- `attribution/attribute_nnsight.py:227-299` for feature-row construction;
- `replacement_model/replacement_model_nnsight.py:256-288` for frozen
  attention, layer-normalization scale, and MLP/skip convention.

## Architecture

The Stage 1A runtime guards and native MPS/BF16 subclasses remain unchanged.
Stage 1B adds a narrow measurement layer:

```text
src/cfsus/backends/nnsight_plt.py
src/cfsus/scanning/near_threshold.py
src/cfsus/responses/targeted.py
src/cfsus/responses/validation.py
scripts/stage1b/preflight_stage1b.py
scripts/stage1b/run_stage1b_measurement_primitives.py
scripts/stage1b/run_stage1b_measurement_worker.py
scripts/stage1b/assemble_stage1b_artifacts.py
scripts/stage1b/validate_stage1b_artifacts.py
configs/stage1b_measurement_primitives_gemma3_270m_mps_bf16.yaml
```

The backend exposes selected state and bounded scalar response operations, not
full caches, full Jacobians, mediation, behavioral metrics, or intervention
sweeps. Capability claims are scoped to the exact frozen Gemma 3/PLT/NNsight
runtime and become supported only after the real opt-in checks pass.

## Near-threshold scanner

The scanner processes one layer and token position at a time. It slices the
loaded encoder, bias, and threshold by feature chunk, evaluates projection and
the loaded gate on MPS/BF16, converts only compact eligible scalar records, and
immediately releases chunk tensors. It maintains bounded per-group and global
top-K collections ordered exactly by
`(margin, layer, position, feature_id)`.

An independent ephemeral dense oracle is allowed only in a validation worker.
For chunk sizes 257, 1024, and 4096, acceptance requires exact equality of
candidate identity, per-group order, global order, eligibility, and recall
1.0 on the bounded oracle. No full dense preactivation array or complete
activation tensor is persisted.

## Targeted local response

The targeted response uses a dedicated reverse-mode path under the same frozen
NNsight convention. It injects the selected target encoder direction at the
target preactivation location and contracts the resulting source-output
gradient with the selected unscaled source decoder direction. This computes
the bounded VJP scalar `partial z_i / partial a_j`; it never constructs a full
Jacobian and never touches a graph object or raw edge. The implementation must
reject an inactive source, identical endpoints, non-upstream ordering, missing
gradient, nonfinite value, unexpected device/dtype, or fallback.

The independent graph-reference stage selects baseline-active feature pairs,
reads raw adjacency only after the targeted estimate exists, and compares
`a_j * J_ij` with the target-row/source-column edge. Calibration and canonical
pair IDs are disjoint and selected deterministically from a fixed hash seed.

## Validation metrics

For raw edge `x` and reconstructed edge `y`, symmetric normalized error is

```text
2 * abs(x - y) / (abs(x) + abs(y))
```

with both zero defined as 0 and exactly one zero defined as 2. Spearman uses
average ranks for ties. Sign agreement is evaluated only above the frozen edge
floor. The canonical hard floors are:

- all targeted values finite;
- Spearman at least 0.98;
- sign agreement at least 0.95 above the frozen edge floor;
- median symmetric normalized error at most 0.05;
- p95 symmetric normalized error at most 0.20;
- no orientation reversal, calibration leakage, fallback, or graph-edge
  division.

Calibration may determine whether the implementation is scientifically
capable and may freeze the edge floor, exact tolerances, selection seed, and
canonical IDs. Once the clean pre-run commit is made, canonical outputs cannot
change those definitions.

The bounded calibration completed on 2026-08-24 before the pre-run freeze.
All 90 scanner groups matched the dense oracle exactly at chunk sizes 257,
1024, and 4096. On the 16 calibration-only active pairs, Spearman and sign
agreement were both 1.0, median symmetric normalized error was
0.0022785724126932663, and p95 was 0.004240743761213505 at edge floor
0.015625. The disjoint 64-pair canonical ID set and endpoint-manifest digest
`9879064f623be1cfdac4c8a1321f293e59e9e897ed7919608157a6e63e62082c`
are frozen in the canonical config. No canonical numeric edge or targeted
response was consumed during calibration.

## Phase gates

1. Gate 0: verify exact origin/base ancestry, Stage 1A validator, immutable
   offline assets, real MPS/BF16/no-fallback access, protected refs/artifacts,
   and absence of an existing target branch/worktree.
2. Gate 1: complete independent upstream, numerical, architecture, and
   security audits; freeze this plan and decisions.
3. Gate 2: implement the backend, scanner, targeted VJP, runner/worker,
   validator, schema, and offline/adversarial tests; pass pytest, Ruff, MyPy,
   import, dependency, diff, and safety scans.
4. Gate 3: run bounded real MPS calibration, scanner/dense oracle comparison,
   known active/inactive checks, targeted active-pair calibration, memory and
   fallback checks. Do not create canonical evidence.
5. Gate 4: freeze exact prompt, selected layers/positions, chunk sizes, top-K,
   seed, calibration/canonical pair IDs, edge floor, tolerances, runner,
   worker, validator, and artifact schema in a clean pre-run commit; push only
   the Stage 1B branch.
6. Gate 5: execute one fresh canonical run from the required clean pre-run SHA.
   Scientific failures are not retried or used to tune definitions.
7. Gate 6: run the standalone validator and independent spot checks, audit the
   final diff and protected refs, publish only allowlisted compact artifacts
   and the final report, commit, and push only this branch.

A failed hard gate terminates fail-closed with the deepest verified blocker.
Only all gates passing permits
`completed_stage1b_measurement_primitives`.

## Artifact and claim boundary

The canonical directory is `results/stage1b_measurement_primitives/`. It may
contain only compact JSON summaries/tables and `checksums.sha256`, must remain
below 5 MiB, and must not contain weights, cache, raw graph, adjacency, full
activation/preactivation arrays, gradient tensors, secrets, local paths, or
temporary files.

Even on success the claim boundary remains:

```text
stage1b_measurement_primitives: completed
stage1c_first_prediction_readiness: true
stage1b_empirical_claim_readiness: false
counterfactual_susceptibility_result: none
gate_crossing_result: none
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```
