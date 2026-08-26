# Stage 1C-v3 preregistered prospective prediction report

Status date: 2026-08-26

```text
verdict: failed_runtime
scientific_outcome: inconclusive_runtime
```

Stage 1C-v3 successfully preregistered and froze a Norway-prompt,
baseline-only prediction after excluding the 28 exact source-target pairs from
the authenticated Stage 1C-v1 prediction manifest before ranking. The frozen
manifest passed the standalone validator and two independent audits. The one
permitted canonical intervention process then failed during selected-baseline
remeasurement because the frozen intervention adapter did not implement the
worker's required `measure_states` method. The failure occurred before the
first source-suppression API call, so no point-level intervention evidence or
scientific crossing outcome exists. The attempt is consumed and was not
retried.

## Provenance and Git separation

- Exact base: `ee9cc944fbdabaa6437b7be3c997725fce5de0a6`
- Branch: `stage-1c-v3-preregistered-prospective-prediction`
- Binding specification SHA-256:
  `1a6117bc1a195dc433b71afe1f5a0b14a55e2bc39c4deb14bb21bd4b6eb249c4`
- Initial protocol commit:
  `9953cb2cd723659a3446b64d75540020c9fcf4d0`
- Metadata-serialization protocol commit:
  `2055520bd2f4f3ebb53ec279b63328d0e3f59fe9`
- Pre-intervention prediction-freeze commit:
  `10f7234a036562e9337514fc085415a017e99102`
- Execution/artifact commit: none; the canonical run did not produce an
  accepted intervention artifact
- The initial protocol commit was not amended. Each accepted commit was pushed
  only to the v3 origin branch before the next empirical phase.
- Protected `main`, Stage 1A-S-BF16, Stage 1B, Stage 1C-v1, and Stage 1C-v2
  origin refs remained at their audited commits.

## Historical independence and prompt preregistration

The denylist was extracted only from source and target `FeatureRef` records in
the tracked, frozen baseline-only Stage 1C-v1 prediction manifest:

```text
source: results/stage1c_first_prospective_prediction/prediction_manifest.json
source SHA-256: 43cf17f3f87ff97f9fa2aa6b827c84416add5dced2824b69c057d99a5f2b882a
source Git blob: 847e9c3389097b529c0ac2861b3d519afe18d050
source freeze commit: 6ec950d93fe1215fdcfee68c87e1f58a23a78ae8
denylist SHA-256: ee31e29e2eb2be5aa5cbf72b95d75ea275098592eb54106f20aa7b3ba87405ad
exact pair count: 28
historical endpoint count: 53
```

No Stage 1C-v1 intervention outcome or Stage 1C-v2 temporary baseline output
was read. Only an exact six-coordinate source-target match was excluded.
Source-only, target-only, and both-endpoints-seen-separately overlaps remained
eligible and were recorded after selection as audit metadata; the overlap
category was not an input to ranking, quotas, schedules, or classification.

The prompt was independently derived from
`ee9cc944fbdabaa6437b7be3c997725fce5de0a6|stage1c-v3-prompt-v1`:

```text
SHA-256: 66e7d4281197efefdbc83bf369d9d317faa7641990c27fa1c3842de99c358e41
pool index: 7
prompt ID: capital_norway_preregistered_v3
prompt: The capital of Norway is
token IDs: [2, 818, 5279, 529, 32649, 563]
selected positions: [1, 2, 3, 4, 5]
```

## Immutable runtime and accepted prediction freeze

```text
model: google/gemma-3-270m
model revision: 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
transcoder: mwhanna/gemma-scope-2-270m-pt
transcoder revision: fada11860ac1d337c1e41e9da308798405b94c8e
transcoder subset: transcoder_all/width_16k_l0_small
layers / feature width: 18 / 16384
backend: nnsight 0.6.1
device / dtype: mps:0 / torch.bfloat16
python / torch / transformers: 3.11.13 / 2.6.0 / 4.57.3
circuit-tracer: 0.5.2 at 8f1e2438df612464e229e44c4a00ff637bf9379b
host: Apple M2 Max, 32 GiB unified memory
```

The accepted baseline used no source-suppression API, historical intervention
outcome, raw graph, or adjacency. Its exact scanner/dense-oracle comparison and
four-pair graph-independent VJP calibration passed. Pool and selection evidence
was:

```text
scanner candidates: 128
eligible targets / sources: 105 / 1930
eligible pairs before exact-pair mask: 39235
exact historical pairs removed before ranking: 16
eligible pairs after mask: 39219
selected primary / near-boundary / directional: 12 / 8 / 8
selected exact historical pairs: 0
endpoint overlap counts:
  neither: 15
  source-only: 2
  target-only: 9
  both-seen-separately: 2
```

Across the 28 selected rows, susceptibility ranged from
`-521.0156249833275` to `173.249999994456`. Predicted alpha was defined for 20
rows and ranged from `0.005772005772005772` to `1.003921568627451`. Primary
predicted alpha ranged from `0.005772005772005772` to
`0.017932193891846457`; near-boundary control alpha ranged from
`1.0002442002442002` to `1.003921568627451`.

The tracked prediction manifest is 45,173 bytes with SHA-256
`b2c489317852a2f54d50db783abc17dfdc08590353b0473dbab01ec3d04574cc`.
The standalone prediction validator and two independent Luna reviews
recomputed prompt selection, denylist provenance, exact-pair exclusion, pair
IDs, causal order, finite values, scores, alpha values, statuses, group order,
quotas, schedules, and endpoint-overlap categories.

## Invalidated preliminary baseline process

The first baseline-only process, run from `2026-08-26T14:54:41Z`, computed no
published prediction artifact. At the final worker publication boundary, the
strict serializer rejected `torch.__version__` because PyTorch exposes it as a
`TorchVersion` string subclass rather than an exact builtin string. No pair
artifact was published or inspected. Two independent audits classified a
minimal `str(torch.__version__)` conversion in both workers as metadata-only,
consistent with the documented v2 pre-publication precedent. The fix and
regression tests were committed and pushed before a fresh baseline. No
selection rule, mask, prompt, score, tolerance, schedule, classifier, schema,
or validator changed.

The invalidated process had 66 supervisor samples, peak process-group RSS
957,677,568 bytes, minimum available memory 12,881,625,088 bytes, zero swap
growth, nominal thermal state, and no timeout, safety termination, or
telemetry failure.

## Canonical attempt and terminal blocker

The accepted prediction was committed and pushed before intervention. The
canonical preflight passed exact Git/origin identity, tracked manifest bytes,
prompt/tokenizer identity, asset allowlists, runtime versions, native
MPS/BF16, memory, swap, thermal, no-fallback, no-autocast, offline,
no-authentication, and historical-independence gates.

The one canonical process began at `2026-08-26T15:17:11Z`. After replacement
runtime loading, the worker attempted to remeasure the 28 frozen pairs'
selected baseline states. It called
`Stage1CVersion3InterventionBackend.measure_states`, but the frozen adapter
implements only `measure_point`. Python raised `AttributeError` before the
pair-sweep loop and before the only code path that increments the instrumented
source-suppression API counter.

```text
canonical attempts: 1
scientific retries: 0
source-suppression API calls: 0
serialized sweep pairs: 0
serialized sweep points: 0
primary/control crossing outcomes: unavailable
observed critical alpha: unavailable
local-linearity metrics: unavailable
scientific outcome: inconclusive_runtime
```

No top-level or artifact sweep exists, so point-count, pair-set, crossing,
critical-alpha, local-linearity, and cross-file invariants cannot be satisfied
or recomputed. This is not `no_eligible_pairs`: the frozen manifest contains
12 eligible primary pairs. It is a canonical runtime failure. Fixing the
adapter and rerunning within v3 would violate the one-attempt rule; any retry
requires a new explicitly versioned experiment class.

The canonical supervisor sampled seven times: peak process-group RSS
668,663,808 bytes, minimum available memory 12,566,364,160 bytes, zero swap
growth, nominal thermal state, no timeout, no safety termination, and no
telemetry failure. The failed worker did not publish internal MPS allocation
telemetry, so no canonical GPU-memory peak is claimed. The accepted baseline
ran for 63.58 seconds; its internal telemetry recorded peak MPS current
726,274,048 bytes, peak MPS driver 2,865,414,144 bytes, peak process RSS
953,188,352 bytes, minimum available memory 12,544,540,672 bytes, zero swap
growth, and no violation or telemetry failure.

## Artifact and validation disposition

Only the validated prediction manifest is a scientific artifact. No canonical
intervention worker, final bundle, checksum manifest, sweep, crossing summary,
critical-alpha result, or local-linearity result was created. Temporary
preflight and supervisor diagnostics and the ignored one-attempt lock remain
outside Git and are not scientific evidence.

Pre-intervention checks included a full offline result of `595 passed, 1
skipped, 1 deselected`; the skip was the explicit opt-in real MPS/BF16 test,
which was also run separately and passed. Focused v3, inherited serialization,
Ruff, formatting, relevant strict MyPy, development/runtime `pip check`, JSON,
compile, deterministic-math, package-import, denylist, prediction-validator,
and staged secret/private-path checks passed. The strict v3 scope covered 81
source/test files. A monolithic `mypy src scripts tests` invocation remains
unusable because unchanged historical script directories contain duplicate
module basenames; no repository-wide monolithic MyPy pass is claimed. The
standalone final-bundle validator, checksum verification, and crossing/local-
linearity recomputation are inapplicable and explicitly not claimed because no
final bundle exists.

The whole-repository tracked-content scanner still reports four pre-existing
test-fixture literals outside the v3 diff: three private-path examples and one
synthetic Bearer-header example in historical Stage 1B/Stage 1C-v1 security
tests. The v3-scoped candidate scan and final staged scan contain zero
findings. All tracked entries use regular-file modes, have one hardlink, remain
under the 2 MiB per-file cap, and contain none of the forbidden model/archive
extensions.

No weight, cache, tokenizer payload, raw graph, adjacency, dense activation or
derivative tensor, gradient, emergency raw output, secret, private path,
historical intervention outcome, symlink, hardlink, FIFO, device, socket, or
oversized artifact is tracked. No merge, pull request, tag, release, or push to
`main` or a historical branch occurred.

## Claim boundary

```text
stage1b_measurement_primitives: completed
stage1c_v3_prospective_prediction: failed_runtime
stage1c_v3_scientific_outcome: inconclusive_runtime
counterfactual_susceptibility_result: none
gate_crossing_result: none
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

The frozen baseline is a valid prospective prediction artifact, but without
an accepted intervention it is not a Counterfactual Susceptibility result and
does not support any crossing, behavioral, mediation, benchmark, Gemma 2,
CLT, or paper claim.
