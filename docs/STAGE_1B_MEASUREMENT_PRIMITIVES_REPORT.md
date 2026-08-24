# Stage 1B measurement primitives report

Status date: 2026-08-24

Terminal status: `completed_stage1b_measurement_primitives`

## Outcome

All seven ordered phase gates passed. The accepted Stage 1A-S-BF16 Gemma 3
270M + 18-PLT + NNsight runtime was converted into a reusable bounded
measurement backend and used to validate:

1. a chunked near-threshold inactive-feature scanner using exact loaded strict
   JumpReLU semantics; and
2. an independent targeted reverse-mode calculation of
   `J_ij = partial z_i / partial a_j` from source post-gate activation to target
   pre-gate preactivation.

The targeted implementation accepts no graph, adjacency, or edge input. Raw
attribution edges were constructed separately and used only as prospective
validation references through `E_(j->i) = a_j * J_ij`.

This report contains no Counterfactual Susceptibility estimate, inactive-target
score, source-suppression sweep, gate crossing, behavioral importance,
mediation, Gemma 2 result, CLT result, official BF16 reproduction, or paper
Results claim.

## Frozen provenance

- exact base: `fb2fc158b45c842743804040e4e273776e666a48`
- branch: `stage-1b-measurement-primitives`
- clean pre-run/canonical execution commit:
  `de49bc0ee1d4ee1b2a0c15703b41e76781467ede`
- circuit-tracer 0.5.2:
  `8f1e2438df612464e229e44c4a00ff637bf9379b`
- model: `google/gemma-3-270m` at
  `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- PLT: `mwhanna/gemma-scope-2-270m-pt` at
  `fada11860ac1d337c1e41e9da308798405b94c8e`, exact subset
  `transcoder_all/width_16k_l0_small`
- runtime: native arm64 CPython 3.11.13, PyTorch 2.6.0, NNsight 0.6.1,
  Transformers 4.57.3, Apple MPS, `torch.bfloat16`
- host: Apple M2 Max, 32 GiB unified memory
- prompt: `The capital of France is`, token IDs
  `[2, 818, 5279, 529, 7001, 563]`
- frozen config SHA-256:
  `c68d5f5974a2d08b40519ad89834a5bbc37715e434bd267c3ede15affcf19369`
- frozen artifact-schema SHA-256:
  `8a88695c17a85f22e28a2c2023c98d0190a2093dbbc8b0129f79ea896a797d05`

The asset allowlist rehashed 2,087,816,677 bytes. The canonical run used the
already verified project-external offline cache with no network, download,
authentication, secret read, fallback variable, autocast, or CPU scientific
tensor computation.

## Pre-run freeze and attempt policy

Bounded calibration passed before the canonical freeze. Its 16 active pairs
were disjoint from the 64 canonical pairs. Calibration established engineering
fitness without changing the predeclared tolerances: Spearman 1.0, sign
agreement 1.0, median symmetric normalized error 0.0022785724, and p95
0.0042407438 at edge floor 0.015625. Scanner candidates matched the dense
oracle exactly at all three chunk sizes.

The prompt, tokens, layers, positions, feature width, chunk sizes, top-K,
selection seed, exact pair IDs, endpoint digest, edge floor, tolerances,
runner, worker, standalone validator, and schema were frozen in clean commit
`de49bc0ee1d4ee1b2a0c15703b41e76781467ede` and pushed before the canonical
run. Canonical numeric results were not present in calibration evidence.

The canonical run executed exactly once from that commit between
2026-08-24T19:00:22Z and 2026-08-24T19:02:22Z. It read no calibration artifact,
made no scientific retry, timed out nowhere, and was not safety-terminated.

Five failed engineering calibration attempts preceded the passing calibration:
safe traceback persistence, threshold-chunk adaptation, unsupported MPS
`index_copy.out`, NNsight gradient-proxy ordering, and strict-JSON tuple
serialization. Each was corrected and rerun before the freeze without changing
the scientific definitions, prompt, selection rule, floor, or tolerances. They
are not canonical scientific retries.

## Scanner validation

The scanner processed 18 layers by 5 non-BOS positions, 16,384 features per
group. It used the exact loaded gate `a = z * 1[z > tau]`; equality remained
inactive. Eligibility required `a == 0`, `z <= tau`, and exact
`margin = tau - z`.

| Check | Canonical result |
|---|---:|
| Groups | 90 |
| Chunk sizes against dense oracle | 257, 1024, 4096 |
| Canonical chunk size | 1024 |
| Per-group / global top-K | 8 / 128 |
| Exact candidate identity and order | passed |
| Bounded oracle recall | 1.0 |
| Maximum retained candidates | 4,104 |
| Persisted dense arrays | none |

An independent post-run check revalidated the deterministic
`(margin, layer, position, feature_id)` order and every serialized candidate's
inactive, zero-activation, `z <= tau`, and exact-margin invariants.

## Targeted local-response validation

The canonical set contained 64 calibration-disjoint baseline-active pairs,
spanning 16 target layers, all 5 selected target positions, and both raw-edge
signs. All 64 raw edges met the frozen 0.015625 floor.

| Metric | Canonical | Frozen acceptance | Result |
|---|---:|---:|---|
| Spearman | 0.9999656588 | at least 0.98 | passed |
| Sign agreement | 1.0000000000 | at least 0.95 | passed |
| Median symmetric normalized error | 0.0018869016 | at most 0.05 | passed |
| p95 symmetric normalized error | 0.0045122760 | at most 0.20 | passed |

All serialized values were finite. The targeted path reported
`graph_edge_used = false`; no graph or adjacency was persisted. The separately
reconstructed endpoint manifest had the frozen SHA-256
`9879064f623be1cfdac4c8a1321f293e59e9e897ed7919608157a6e63e62082c`.
An independent standard-library implementation reproduced pair order,
endpoint digest, `a_j * J_ij`, average-rank Spearman, sign agreement, median,
nearest-rank p95, layer/position coverage, and both-sign coverage.

## Memory, timing, and host safety

- supervisor wall time: about 119.95 seconds
- worker wall time: about 114.50 seconds
- worker MPS current-allocation peak: 641,321,728 bytes
- worker MPS driver-allocation peak: 2,865,414,144 bytes
- worker process RSS peak: 1,290,059,776 bytes
- minimum available memory: 12,009,701,376 bytes
- swap growth: 0 bytes
- telemetry failures: 0
- thermal states: `nominal`
- safety violations: none

MPS, driver, and RSS values overlap in unified memory and are not summed. All
frozen memory, swap, timeout, telemetry, and thermal gates passed.

## Verification and artifact safety

The full offline suite passed before the pre-run freeze: 397 passed, 21
skipped, and 1 deselected. Ruff lint and format checks, MyPy on all relevant
Stage 1B modules, dependency consistency, import checks, MPS/BF16/no-fallback
preflight, immutable asset hashing, hostile synthetic-bundle validation, size
checks, binary scans, and source-level circularity/safety scans passed.

After the canonical run, the standalone artifact validator returned:

```text
status: passed
verdict: completed_stage1b_measurement_primitives
artifact_count: 10
candidate_count: 128
pair_count: 64
```

The bundle contains nine JSON payloads plus `checksums.sha256`, occupies 120
KiB on disk, contains no links, and remains far below the 5 MiB limit. All nine
payload checksums passed. Checksum-manifest SHA-256:
`641abac4d7e4efb75f82b8d359b56cff1e9a7e42ddb85495abeead59c61b08ee`.
Independent secret/local-path/forbidden-extension scans were empty. No weight,
cache, raw graph, adjacency, complete activation tensor, dense preactivation
array, gradient tensor, secret, or temporary runtime output is committed.

Four independent leaf audits covered pinned upstream semantics, scanner
numerics, Stage 1A runtime integration, and artifact/security validation. Their
findings were resolved centrally before calibration and the pre-run freeze.

## Readiness boundary

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

`stage1c_first_prediction_readiness: true` means only that a separately
specified, prospectively frozen next stage may use these two measurement
primitives. It does not itself constitute an empirical prediction or result.
