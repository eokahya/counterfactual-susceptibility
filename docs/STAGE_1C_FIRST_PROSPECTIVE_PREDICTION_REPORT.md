# Stage 1C first prospective prediction report

Status date: 2026-08-26

```text
verdict: failed_runtime
scientific_outcome: inconclusive_runtime
```

The baseline-only prospective prediction completed, was independently checked,
and was frozen in a clean commit pushed before any selected inactive-target
intervention. The one permitted canonical intervention process then executed,
but its returned JSON lost every point-level sweep record because the frozen
worker cleared a shared list in its `finally` block after constructing the
return object. The frozen assembler and standalone artifact validator therefore
rejected the result. No scientific outcome from that process is accepted.

## Provenance and separation

- Exact base: `efbf70a7e462e640a0e1819a93f3b92727bbd193`
- Branch: `stage-1c-first-prospective-prediction`
- Pre-intervention and execution commit:
  `6ec950d93fe1215fdcfee68c87e1f58a23a78ae8`
- Pre-intervention commit parent: exact base above
- Prediction-manifest SHA-256:
  `43cf17f3f87ff97f9fa2aa6b827c84416add5dced2824b69c057d99a5f2b882a`
- Config SHA-256:
  `842c73e32e040a2a4576121416d867758b2e395b91e02ab5e7837406afef5332`
- Artifact-schema SHA-256:
  `f34773e34bdb6e97ebb15a0842732e402a2c616ab586c2720529e8673be60fed`
- The pre-intervention commit was clean and present at the exact origin branch
  SHA before the canonical process began.
- Prediction guards record zero source-suppression API calls, no intervention
  worker import, no prior inactive-target outcome read, and no raw graph or
  adjacency read.
- The canonical process read the tracked prediction manifest byte-for-byte and
  verified all frozen protocol hashes.
- Scientific retry count: `0`; canonical attempt count: `1`.
- No artifact-completion commit exists because no final bundle passed the
  frozen validator. The later documentation-only failure commit is identified
  in Git and in the task's final response.

## Immutable runtime identity

```text
backend: nnsight 0.6.1
device: mps:0
execution_dtype: torch.bfloat16
python: 3.11.13 (native arm64)
torch: 2.6.0
transformers: 4.57.3
circuit-tracer: 0.5.2
upstream_revision: 8f1e2438df612464e229e44c4a00ff637bf9379b
model: google/gemma-3-270m
model_revision: 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
transcoder: mwhanna/gemma-scope-2-270m-pt
transcoder_revision: fada11860ac1d337c1e41e9da308798405b94c8e
transcoder_subfolder: transcoder_all/width_16k_l0_small
host: Apple M2 Max, 32 GiB unified memory
fallback: disabled and absent
network/download/authentication: none
```

The prompt was exactly `The capital of France is`, with token IDs
`[2, 818, 5279, 529, 7001, 563]`.

## Baseline-only prediction

The exact loaded JumpReLU scanner evaluated all 90 selected layer/position
groups with feature width 16,384. It matched the ephemeral dense oracle exactly
and persisted no dense scanner array. Baseline pool counts were:

```text
scanner_candidates: 128
eligible_inactive_targets: 101
raw_active_sources: 2152
eligible_active_sources: 1908
eligible_source_target_pairs: 30283
excluded_for_causal_order: 27
definitely_crossing_predictions: 4248
boundary_ambiguous_predictions: 2952
not_crossing_predictions: 23083
q_positive: 14836
q_negative: 15447
q_zero: 0
```

The separate active-source/active-target engineering calibration compared four
disjoint many-source VJP results with the accepted Stage 1B pairwise targeted
VJP and passed exact BF16 identity. No complete derivative matrix was
persisted.

Frozen selection counts were 12 primary pairs, 8 positive-`q` near-boundary
controls, and 8 non-positive-`q` directional controls. Pair IDs were disjoint;
primary targets were unique; no source appeared in more than two primary
pairs. The primary susceptibility range was
`59.6203124999523 .. 160.61718749967878` with median
`85.57714843723858`. Primary predicted critical suppression ranged from
`0.006225983754073642` to `0.016772807086510994`, with median
`0.011720689614834677`.

The following table is the complete frozen selection. `S` and
`alpha_hat_star` are prospective baseline-only quantities. Because the
canonical sweep evidence was erased before serialization, the accepted
observation for every row is `unavailable`.

| Group | Pair ID | Source | Target | S | alpha_hat_star |
| --- | --- | --- | --- | ---: | ---: |
| primary | `1d3be832a2a95763a3df197f83295dfab228509015e9c0fc565c8f914f7db5f2` | `L14/P1/F234` | `L15/P1/F771` | 160.61718749967878 | 0.006225983754073642 |
| primary | `9fef73a2934067d0296e9eb8d72914366c15417944878360242bd18f721f407b` | `L1/P3/F111` | `L10/P3/F5755` | 129.4023437494824 | 0.007727835300510158 |
| primary | `adea8700ada5077ec76e06189b53972fb0a1a6362396a873596566967e0083f9` | `L2/P3/F582` | `L3/P3/F15884` | 128.6015624989712 | 0.007775955288257093 |
| primary | `005444146f6170c7b91125587ea2aa29264d60127ef71c69475f7584c709327a` | `L9/P1/F761` | `L10/P1/F2072` | 117.71484374952915 | 0.008495105359216857 |
| primary | `9ea8a228cbdcf78a0a0598b9d30ad703010f25bd47d68e4991448005206e7f80` | `L11/P1/F326` | `L12/P1/F11208` | 96.07812499990392 | 0.010408196454708083 |
| primary | `087c92ec67fce4defca74e6df56739d606c90fd79cfe646e03bae138fbce42bc` | `L9/P5/F2031` | `L10/P5/F7271` | 90.2753906246389 | 0.011077215984076502 |
| primary | `a3d7fa50b181b96ac5979633f52e367795aa988f707dad5858f07c2db6f31ce0` | `L9/P1/F761` | `L10/P1/F2082` | 80.87890624983825 | 0.012364163245592853 |
| primary | `afe62a2b950838c2628f73557e2446baa26adcd481a16d8fb732695bcdb62483` | `L4/P3/F324` | `L8/P3/F10215` | 80.54882812435561 | 0.01241482990228171 |
| primary | `3c0d6e9cda7622090cbecd20f26c8156ecd22688633530f2d38f06dd54bd103b` | `L2/P5/F375` | `L3/P5/F166` | 77.23749999975284 | 0.01294707881534229 |
| primary | `65d7923bca5a9736c83cf3cb6a98e6b6d5d21ee86ff313e9511cd9f83312dff4` | `L0/P1/F1386` | `L1/P1/F4617` | 71.36718749771626 | 0.014012041598248495 |
| primary | `6e09f69a54bb1baf9df376e9eaec7f51c46bf1f0bebb90e35b4946a27d7c1208` | `L10/P1/F206` | `L11/P1/F774` | 64.68124999994825 | 0.015460430959512996 |
| primary | `5e894ec732046044dc5d8c54adda097897267a14d1424ca8f5adfe0ae6df2075` | `L7/P1/F503` | `L8/P1/F30` | 59.6203124999523 | 0.016772807086510994 |
| near_boundary | `4233db1c02dff2982514ad249ac6ac9cfd907ee5e2428ec2f7c122b3d8da8bb5` | `L3/P3/F7468` | `L5/P4/F3745` | 0.9997558593736671 | 1.0002442002442002 |
| near_boundary | `e9540cc5ba5523197063ed8fc07535904498f9d15f143d04d48c2d956851afe1` | `L1/P4/F12501` | `L13/P4/F14586` | 0.9996643066396252 | 1.0003358060872485 |
| near_boundary | `ec665f470dbc1cac2fdf2ad69208fe049cb54e8f8fc566b0737e29a4f856401a` | `L5/P1/F316` | `L8/P4/F1464` | 0.999023437498002 | 1.0009775171065494 |
| near_boundary | `5fca91cc2714dc58c316c709baffd1ad1c525deb9fe645f2a24d79776cd6f352` | `L0/P2/F724` | `L9/P5/F153` | 0.999023437498668 | 1.0009775171065494 |
| near_boundary | `dfa5d5a4a02b9076ce51e1416f3bb44f388b1fdf0ebf6e486099487a849bf9e5` | `L6/P4/F1472` | `L9/P4/F3017` | 0.9986979166657789 | 1.001303780964798 |
| near_boundary | `6d451b78a705b2113ef65cc75cab330a8c527308806b78d774f425c2899e7474` | `L10/P3/F324` | `L11/P4/F9817` | 0.9978027343740021 | 1.0022021042329337 |
| near_boundary | `5229a08c747672c95d83fbdde668db7fab4f87ca3e449d71ee52d9dc7304b253` | `L2/P3/F419` | `L15/P4/F30` | 0.9973551432285017 | 1.0026518705887153 |
| near_boundary | `f7852fe6619e69b578530105be6cb181ba3ce8a2c177bd64134b460718c09ec5` | `L3/P3/F1566` | `L9/P4/F161` | 0.9973366477265473 | 1.002670464660851 |
| directional | `aa9be1a6d1625d1cd0712ec9aea85da60f65015821b29f5c0f694d73f0e41f5c` | `L3/P3/F380` | `L8/P3/F1485` | -785.999999995808 | null |
| directional | `186b91a448a9ee708ad81f5347426b43106960cc7a51df942c551c965ac0c78a` | `L3/P1/F537` | `L10/P1/F1008` | -274.99999999945004 | null |
| directional | `bb7a1840c7d896c4af223c6e2a22dfa43f123a27f9234e4ac5cd5c88ff78a273` | `L1/P1/F266` | `L2/P2/F3293` | -229.42499999853166 | null |
| directional | `23dd4740e5642e1c892a3f9c7a44d673dd3092f80100ccd7a32bb4a3fba947d7` | `L1/P5/F453` | `L2/P5/F278` | -212.49999999935238 | null |
| directional | `f333902c685ce52e8df6620c2648c37b5ed0dc0f56aada6f31ab77f0a7b5378d` | `L3/P1/F537` | `L6/P1/F4612` | -199.3749999998405 | null |
| directional | `762a7cdf35e266d3c5c7866ae40bdddbef017b34560046a2272c06daae3d3d60` | `L5/P1/F7717` | `L6/P1/F451` | -166.49999999975782 | null |
| directional | `68e49310c63d58836a2ce804e1e6c20660fb90b6ba26f59bcf6524a146f03657` | `L3/P1/F537` | `L10/P1/F4484` | -122.6041666665032 | null |
| directional | `5ae975ee251f79ea14c958111b0e88f8065952d6f1b4bfe96dd6711cfaa54e7c` | `L1/P4/F546` | `L2/P4/F542` | -113.39999999854847 | null |

## Canonical attempt and invalidation

The one canonical process ran from 2026-08-26T08:57:55Z to
2026-08-26T09:00:00Z. It remeasured 53 unique selected feature baselines and
reported 228 calls to the public source-suppression API under the frozen
regime:

```text
source_count: 1
freeze_attention: true
constrained_layers: null
desired_mapping: (1-alpha) * baseline_source_activation
target_clamp_allowed: false
```

The in-process frozen analysis was computed before cleanup. Its diagnostic
summary said 12/12 primary full-ablation crossings, 3 locally calibrated
primary pairs, 5/8 near-boundary control crossings, 0/8 directional control
violations, primary critical-suppression Spearman 0.16083916083916083, and a
provisional `mixed` label. These values are **not accepted scientific
evidence**: the point records needed to recompute them independently are
absent.

The concrete failure is an object-aliasing cleanup bug in the frozen worker.
The returned object holds references to `sweeps` and to an artifact record
derived from the same list. The function's `finally` block then calls
`sweeps.clear()` before the caller serializes the returned object. Consequently:

```text
expected selected sweep rows: 28
serialized top-level sweep rows: 0
serialized intervention_sweeps rows: 0
reported source-suppression API calls: 228
```

The frozen assembler failed with
`canonical sweeps differ from the frozen selected groups`. The standalone
validator independently rejected the incomplete result directory because it
differed from the exact ten-file allowlist. A second run is forbidden, and
changing the worker after viewing outcomes would require a new explicitly
versioned experiment class. Therefore the only valid classification is
`inconclusive_runtime`.

## Runtime, memory, swap, and thermal evidence

Prediction ran from 2026-08-26T08:49:16Z to 2026-08-26T08:50:22Z. Its worker
recorded a peak process RSS of 969,228,288 bytes, peak MPS current allocation
of 621,416,448 bytes, peak MPS driver allocation of 2,865,414,144 bytes,
minimum available memory of 12,966,510,592 bytes, zero swap growth, no
telemetry violations, and nominal thermal state.

The canonical intervention supervisor sampled for approximately 124 seconds.
It recorded peak process-group RSS of 793,853,952 bytes, minimum available
memory of 12,827,443,200 bytes, zero swap growth, no timeout, no safety
termination, no telemetry failures, and nominal thermal state. The worker's
peak MPS current and driver allocations were 587,544,832 and 2,865,414,144
bytes. Overlapping unified-memory counters are not summed.

A live sample near the first minute showed worker CPU use of 86.7%, process
RSS about 735 MiB, Apple GPU device utilization of 20% and renderer
utilization of 5%. macOS reported no thermal or performance warning.

## Validation and artifact disposition

Before intervention, Ruff format/check, `git diff --check`, strict MyPy over 42
source files, and the full production-dependency offline suite passed. The
suite result was `446 passed, 1 skipped, 1 deselected`. The prediction manifest
then passed independent standard-library validation and exact pair-score,
ordering, schedule, pair-ID, and protocol-identity recomputation.

After the canonical process:

- frozen assembler: failed closed because serialized sweep IDs did not match
  the 28 frozen selected pair IDs;
- standalone validator: failed closed because no complete allowlisted bundle
  existed;
- no final canonical checksum manifest was created;
- no worker output, raw sweep, weight, cache, graph, adjacency, derivative
  matrix, dense activation, gradient, secret, private path, or binary payload
  was committed;
- the only result artifact retained in Git is the validated prediction-only
  manifest, one file of 31,717 bytes with the SHA-256 above.

The rejected ephemeral diagnostic record hashes were:

```text
prediction_worker:   0f0cfc4679b22afc0a445fb03361b49769cf96b7ed13b4345fec482f68fef843
prediction_supervisor: 58964b11c18dedf80aac96e5a808aa97afe3485ec6b07fc0b2cd34999f9d5c93
intervention_worker: 6bb65d488c97787a715cdb0c79b257a1939543d7fd03612a0155fccf1f740bd6
intervention_supervisor: 842873857957c514f12dc20feb0547ea2cd88fc282673a22918209ac28c5e08b
```

These hashes identify invalidated runtime diagnostics only. They are not a
canonical artifact bundle and do not support an empirical claim.

## Claim boundary and readiness

```text
stage1b_measurement_primitives: completed
stage1c_first_prediction: failed
stage1c_scientific_outcome: inconclusive_runtime
counterfactual_susceptibility_result: none
gate_crossing_result: none
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

This run establishes only that a baseline-only Stage 1C prediction was frozen
prospectively and that the one canonical process reached its internal
intervention loop without a memory, thermal, fallback, or supervisor failure.
It does not establish any accepted gate crossing, Counterfactual
Susceptibility result, behavioral importance, mediation, benchmark result,
Gemma 2 result, reference-CLT reproduction, MPS/CUDA equivalence, or paper
result. The manuscript was not changed.

## Proposed next stage

Use a new explicitly versioned experiment class. Before its prediction freeze,
copy or serialize sweep evidence before cleanup, add a regression test that
asserts a returned multi-pair record remains nonempty after the worker exits,
and require the standalone validator to pass a synthetic nonempty bundle.
Then perform a new baseline-only prediction freeze and a separately authorized
single canonical run. No such rerun or new experiment class was started here.
