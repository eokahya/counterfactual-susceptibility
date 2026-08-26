# Stage 1C-v2 held-out prospective prediction report

Status date: 2026-08-26

```text
verdict: blocked_engineering
scientific_outcome: none
```

Stage 1C-v2 repaired the Stage 1C-v1 mutable-alias serialization defect and
passed its synthetic engineering gates. The new Germany prompt also passed
immutable tokenizer, asset, runtime, MPS/BF16, memory, swap, and thermal
preflight. A fresh baseline-only prediction process then reached the frozen
deterministic pair-selection gate, where it detected that at least one newly
selected source-target endpoint overlapped the historical 28-pair Stage 1C-v1
set. The predeclared post-selection guard rejected the run before publishing a
prediction manifest. Reranking or changing the guard after this observation
would be an outcome-dependent protocol change, so execution stopped. No
inactive-target intervention was attempted.

## Provenance and Git separation

- Exact base: `cc47cb604fc2422deb50aacbc7fde77499b532c5`
- Branch: `stage-1c-v2-heldout-prospective-prediction`
- Binding specification SHA-256:
  `6610f369bb2e4127a00d716e4ae4b764e67c0f216b48337e724c358f307ac636`
- Serialization-recovery commit:
  `dc71d1fbaff914f3d8fd48f9d2898cd4f13a9ba5`
- Frozen protocol commit:
  `d9e01c6412beee42b29ac9cdb130dd7afa0e9218`
- Pre-freeze supervisor diagnostic hardening commit:
  `e3f11e8bb52511072f7b2b410e265196dffb456b`
- Pre-intervention commit: none
- Canonical execution/artifact commit: none
- The exact base remains at
  `origin/stage-1c-first-prospective-prediction`; the protected Stage 1B and
  Stage 1C-v1 refs and files were not changed.

The first two commits are separated as required: the first contains detached
serialization plus synthetic end-to-end validation; the second freezes the
new v2 experiment class, held-out protocol, workers, and independent
validator. The small third commit fixes only strict-JSON-safe supervisor error
reporting after an initial prediction process produced no scientific artifact.
No commit was amended or rewritten.

## Serialization recovery

The v2 worker constructs recursively detached JSON-safe copies at the return
boundary. The top-level sweep list and
`intervention_artifacts.intervention_sweeps.pairs` are value-equal but not the
same mutable object, and their nested point rows do not alias cleanup-mutated
working state. Regression tests deliberately clear and mutate the source list
after result construction, round-trip nonempty evidence through strict JSON,
and pass it through worker, assembler, and standalone validator. Hostile
duplicate-key, nonfinite, link, path, archive, secret, oversized, missing,
extra, reordered, pair-ID, point-count, and API-call-count cases fail closed.

The final focused result was `70 passed`. An independent Luna alias/lifetime
audit found no remaining cleanup alias. This establishes an engineering fix;
because no canonical intervention ran, it does not itself establish empirical
susceptibility or gate-crossing evidence.

## Held-out identity and preflight

```text
prompt_id: capital_germany_heldout_v2
prompt: The capital of Germany is
token_ids: [2, 818, 5279, 529, 9405, 563]
selected_positions: [1, 2, 3, 4, 5]
```

The exact offline tokenizer produced the frozen IDs above. Formal preflight
verified:

```text
model: google/gemma-3-270m
model_revision: 9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1
transcoder: mwhanna/gemma-scope-2-270m-pt
transcoder_revision: fada11860ac1d337c1e41e9da308798405b94c8e
transcoder_subset: transcoder_all/width_16k_l0_small
layers/features: 18 / 16384
backend: nnsight 0.6.1
device/dtype: mps:0 / torch.bfloat16
python: 3.11.13 native arm64
torch: 2.6.0
transformers: 4.57.3
circuit-tracer: 0.5.2
upstream_revision: 8f1e2438df612464e229e44c4a00ff637bf9379b
host: Apple M2 Max, 32 GiB unified memory
```

The asset verifier checked 2,087,816,677 bytes against the existing immutable
allowlist. No download, network access, authentication, fallback variable,
outer autocast, or CPU scientific tensor path was used. The MPS/BF16 matrix
operator probe passed. At formal preflight, available memory was
16,185,819,136 bytes, swap usage was 563,546,685 bytes, and thermal state was
`nominal`.

## Invalidated baseline-only attempts and terminal blocker

The baseline-only phase made zero source-suppression API calls and did not
import or execute the intervention worker.

1. The first process wrote only its passed preflight record. The worker did
   not publish a prediction artifact, and the supervisor then rejected its own
   multiline/ANSI diagnostic tail because strict JSON forbids control
   characters. No worker, emergency, pair, or prediction-manifest file was
   produced. The diagnostic sanitizer was repaired and fully retested before
   another baseline process.
2. The second process ran from `2026-08-26T13:00:02Z` to
   `2026-08-26T13:01:03Z`. It passed preflight and reached deterministic
   selection, then stopped with
   `v2 deterministic selection overlaps a historical selected endpoint`.
   The frozen guard ran after selection and did not filter, rerank, or replace
   any pair. No worker or prediction manifest was published.

The second supervisor took 60 samples, with peak process-group RSS
976,240,640 bytes, minimum available memory 13,203,570,688 bytes, zero swap
growth, no timeout, no safety termination, no telemetry failure, and only
`nominal` thermal state. The failed worker did not serialize its internal MPS
allocation samples, so no unsupported GPU-memory peak is reported.

No pool counts, selected-group counts, susceptibility range, predicted-alpha
range, or pair IDs are accepted or reported for v2: the overlap gate fired
before the prediction publication boundary. The exact overlapping endpoint is
also not inferred from historical results. Changing the denylist semantics,
selection order, prompt, top-K, tolerances, or schedule now would use an
observed baseline selection to alter the experiment. Therefore the deepest
valid terminal class is `blocked_engineering`, not a completed scientific
outcome and not `inconclusive_runtime` from an intervention failure.

## Attempt and artifact disposition

```text
baseline prediction processes: 2 (both invalidated before publication)
valid frozen prediction manifests: 0
pre-intervention commits: 0
canonical intervention attempts: 0
scientific retries: 0
source-suppression API calls: 0
serialized sweep pairs: 0
serialized sweep points: 0
```

No final result directory or checksum bundle was created. The temporary
preflight/supervisor diagnostics remain outside the repository under the
system temporary directory and are not scientific evidence. No weight, cache,
tokenizer payload, raw graph, adjacency, complete derivative matrix, dense
activation/preactivation tensor, gradient, emergency raw output, secret, or
private absolute path is tracked.

## Validation

- Full offline suite after the final protocol fix: `516 passed, 2 skipped`.
  The skips were the explicitly opt-in Stage 1A model tests; they are not
  represented as v2 runtime evidence.
- Focused v2 suite: `70 passed`.
- Ruff check and format: passed.
- Strict MyPy over all v2 source and script files: passed, 17 files.
- Runtime and development `pip check`: passed.
- Package import smoke and deterministic-math verification: passed.
- Synthetic nonempty worker to assembler to standalone-validator chain:
  passed.
- V2-scoped candidate and staged commit-safety scans: passed with zero
  findings. A whole-repository scan still identifies four pre-existing v1
  test-fixture literals; they are outside this branch diff and are not reported
  as newly clean.
- `git diff --check`: passed.
- Standalone prediction and final-bundle validators were not applicable to a
  real run because no prediction manifest or final bundle was published.
- Independent final Luna reviews agreed that the correct terminal class is
  `blocked_engineering`, that no within-class scientific continuation is
  permissible, that all protected v1 paths are byte-identical to the base, and
  that no v2 result payload, link, special file, weight, cache, dense tensor,
  secret, or private path entered Git.

## Claim boundary

```text
stage1b_measurement_primitives: completed
stage1c_v2_prospective_prediction: blocked_engineering
stage1c_v2_scientific_outcome: none
counterfactual_susceptibility_result: none
gate_crossing_result: none
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

This Goal produced a validated serialization repair and a fail-closed held-out
selection result. It produced no accepted Counterfactual Susceptibility
prediction, intervention, crossing, behavior, mediation, benchmark, Gemma 2,
CLT, or paper result.
