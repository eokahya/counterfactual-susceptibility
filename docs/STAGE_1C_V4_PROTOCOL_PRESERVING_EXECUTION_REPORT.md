# Stage 1C-v4 Protocol-Preserving Execution Report

## Phase 0 — inspection and repair boundary

The isolated v4 worktree starts from remote v3 HEAD
`92ba35cde279c46e1907f0a48ccb56ad378ccbd5` on branch
`stage-1c-v4-protocol-preserving-execution`. The accepted prediction artifact
remains the tracked file
`results/stage1c_v3_preregistered_prospective_prediction/prediction_manifest.json`.
Its SHA-256 is
`b2c489317852a2f54d50db783abc17dfdc08590353b0473dbab01ec3d04574cc`,
matching the required prediction-freeze identity.

The failed v3 production call path was:

```text
run_stage1c_v3.py
  -> intervention preflight and early local attempt lock
  -> run_stage1c_v3_intervention_worker.py
     -> replacement runtime loading
     -> Stage1CVersion3InterventionBackend construction
     -> _verify_baselines
        -> backend.measure_states(...)  [missing method; failure]
     -> _run_pair
        -> backend.measure_point(...)
           -> model.feature_intervention(...)
     -> build_detached_worker_result
  -> assemble_stage1c_artifacts.py
  -> validate_stage1c_v3_artifacts.py
```

The exact failure was an adapter/worker contract mismatch. The production
worker legitimately batch-remeasures every selected source and target through
`measure_states`, while the production adapter exposed only `measure_point`.
The exception occurred in selected-baseline remeasurement after runtime load
and before the adapter incremented its source-suppression counter or called
`model.feature_intervention`; therefore no frozen-pair intervention outcome
was produced.

The minimal repair is to define one explicit backend protocol containing the
complete worker surface—`measure_states`, `measure_point`, and the read-only
suppression-call counter—and make the production adapter satisfy it.
`measure_states` will delegate to the already-loaded NNsight PLT runtime's
canonical endpoint measurement primitive. That primitive uses the same loaded
transcoders and exact JumpReLU state semantics as the prediction path; it does
not use graph edges or introduce a second scientific measurement
implementation. Worker startup will enforce the runtime protocol before
baseline remeasurement.

The frozen manifest, prompt, pairs, predictions, schedules, tolerances,
threshold rule, classifier, and scientific artifact fields will not change.
Because the manifest records the prediction-time protocol, its embedded file
hashes will be authenticated against prediction-freeze commit
`10f7234a036562e9337514fc085415a017e99102`, while the v4 execution repair is
separately fixed by the pushed pre-run commit.

For v4, the scientific attempt begins exactly at the first instrumented call
to the source-suppression API on any pair ID in the frozen 28-pair manifest.
Runtime loading, baseline remeasurement, quantization-only calculation, and
active-feature engineering rehearsals are pre-attempt work. Immediately at
that first frozen-pair API boundary, the worker will create an append-only
local attempt record outside Git. Every completed point will then be appended
and fsynced to an ignored local journal. A pre-call failure consumes no
scientific attempt; after the first call, no repair, protocol change, or retry
is permitted.

## Pre-run engineering freeze evidence

The production adapter now satisfies a runtime-checkable typed protocol whose
complete worker surface is `measure_states`, `measure_point`, and the
instrumented suppression-call counter. Baseline remeasurement delegates to
`NNSightPLTMeasurementBackend.measure_states` over the same loaded replacement
model and transcoders. The source edit and target gate path remains the
existing adapter implementation.

The synthetic rehearsal invoked the same `_execute_production_sweeps` function
used by the model worker with one primary, one near-boundary, and one
directional pair. It covered endpoint remeasurement, no-op and suppression
points, realized-suppression bisection, per-point append/fsync, cleanup-safe
detachment, artifact assembly, and standalone validation. The resulting test
bundle contained 20 source-suppression calls and 20 serialized points. During
this rehearsal, the independent validator exposed a pre-existing engineering
defect: after validating one bisection point, it appended that point to the
refinement list without restoring realized-suppression order. Sorting the
reconstructed refinement list after every append is the minimal fix; it does
not change the worker, schedule, bracket definition, thresholds, classifier,
or any scientific value.

The frozen-schedule quantization audit covered all 28 pairs and 168 requested
alphas without calling the intervention API. They map to 164 distinct BF16
application values. Four primary pairs each collapse one tiny positive
requested alpha into the no-op BF16 value; all 28 pairs retain at least one
distinct nonzero suppression point. The frozen schedules remain unchanged.

The real-runtime rehearsal used the exact pinned model, transcoders, NNsight,
Apple MPS, and BF16 runtime. It deterministically paired two baseline-active
source-pool endpoints—`(layer=0, position=1, feature=149)` to
`(layer=1, position=3, feature=111)`—whose exact pair is not one of the 28
frozen inactive-target pairs. Baseline remeasurement returned activations
123.5 and 314.0. The production adapter then executed a no-op and one requested
25% suppression; BF16 applied 92.5 and realized 0.2510121457489879. The
active target remained active with preactivation 310.0. A fresh standalone
process independently validated two engineering API calls, two serialized
points, zero frozen scientific-pair overlap, zero scientific intervention
calls, and no scientific attempt. Peak MPS driver allocation was
2,865,414,144 bytes, process RSS was 737,017,856 bytes, minimum available
memory was 12,829,343,744 bytes, swap growth was zero, and thermal state was
nominal.

The first rehearsal process reached only the same active engineering pair and
failed before publication because its diagnostic assembler had not copied the
representative desired value to the point's top level. No frozen target was
called and no scientific lock was created. The diagnostic-only field was
added, the rehearsal was rerun, and the complete active-only chain passed.

The pre-run commit fixes only production execution and validation wiring. It
does not modify the byte-identical frozen manifest or any scientific field.
The runner creates only the ignored parent directory needed for the attempt
boundary; it never creates the lock itself. The worker creates that lock at
the first frozen-pair call, and completion rereads the fsynced JSONL from disk
to verify exact alternating call-start/completed-point order before publishing
the worker result.

The final pre-run verification passed 30 focused production/serialization
tests and the complete inherited offline suite (599 passed, one explicit
real-hardware opt-in skipped, one historical deselection). Ruff lint and
format checks, strict MyPy over 65 source files, dependency checks in both
used environments, import checks, diff whitespace, private-path/credential
scans, and the two-megabyte file ceiling also passed. The final rehearsal and
quantization outputs remain ignored temporary diagnostics and are not part of
the commit.
