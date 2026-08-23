# Stage 1A Apple M2 Max / MPS FP16 report

## 1. Verdict

**Overall verdict: `blocked`.** The determining gate is the re-evaluated
32 GiB conservative memory-feasibility gate. The exact loading plan observed a
sampled MPS driver peak of `40,032,174,080` bytes and a system swap peak of
`34,567,031,357` bytes, invalidating the pre-run estimate and prohibiting an
identical reload on this host.

The one real worker attempt itself ended as `failed_runtime` during
`runtime_loading`: `RuntimeError`, `oom_confirmed=false`,
`retry_eligible=false`, and `cleanup_succeeded=true`. The original worker did
not retain the leaf diagnostic. A bounded postmortem reproduced a compatible,
deterministic FP16 `index_put_(..., accumulate=True)` failure at the next known
sparse-boundary check on the post-print execution path; it is the likely root
cause, but exact identity with the original RuntimeError is unconfirmed. The
coordinates came from `nonzero` and were unique, so accumulation was
unnecessary; commit `86ec118` uses replacement semantics and the live 2-by-3
MPS regression now has zero error. The large runtime was not rerun because the
independently observed memory stop remains decisive.

The official native-BF16 reproduction remains pending. The prior T4/FP16 result
remains scientifically plausible but its final bundle is invalid because its
attempt-level peak aggregation is below later stage peaks. This new MPS/FP16
adaptation did not reach loaded semantics, attribution, intervention, or the
completed-artifact validator and is not complete.

## 2. Git

- Starting/base commit: `d965e43c34a2ba408b8ae35b13b5651bf269beed`.
- New branch: `stage-1a-mps-fp16`.
- Real execution commit: `9de01b5446775f01b211acbc461f8385f9f3732a`.
- Focused branch commits before this terminal record:
  - `cff9fa0` — Add Stage 1A MPS FP16 adaptation.
  - `484e892` — Harden MPS provenance and external cache.
  - `e8dd39c` — Reject tracked ancestors of preserved T4 evidence.
  - `9de01b5` — Fail closed on external cache redirects.
  - `86ec118` — Stop unsafe MPS runtime reloads.
  - `0f1522e` — Harden MPS failure diagnostics.
- Push target: only `origin/stage-1a-mps-fp16`; local and remote equality is
  verified in the final handoff.
- `main`, `origin/main`:
  `7aacf30d888f96a29a1cfc82d035fca489ed0c17`, unchanged.
- `stage-1a-t4-fp16`, `origin/stage-1a-t4-fp16`:
  `d965e43c34a2ba408b8ae35b13b5651bf269beed`, unchanged.
- Report-authoring implementation HEAD before the terminal documentation
  commit: `0f1522e7d0ed840034eaed8ddc886a9189d33f55`. A commit cannot embed its own
  final SHA; the final handoff records and verifies the terminal full local and
  remote SHA after this report is committed.
- No merge, PR, tag, release, or push to `main` was performed.

## 3. Artifact preservation

All seven historical T4 artifact files were copied to an external backup and
verified independently by SHA-256 and byte comparison. File identity and the
known invalid attempt-peak relation were checked before MPS work. The original
`results/stage1a_t4_fp16/` directory remains unmodified, untracked, and
unstaged. The machine-local backup path is intentionally absent from tracked
files and this report.

## 4. Selected environment

- Host: Apple M2 Max, 32 GiB unified memory, Darwin arm64, macOS 26.6.2.
- Python: native CPython 3.11.13.
- PyTorch: 2.6.0; MPS built and available in the host execution process.
- TransformerLens: 3.2.1.
- Transformers: 4.57.3.
- nnsight: 0.6.1.
- huggingface-hub: 0.36.2.
- `circuit-tracer`: 0.5.2 at exact commit
  `8f1e2438df612464e229e44c4a00ff637bf9379b`.
- Fallback: disabled; `PYTORCH_ENABLE_MPS_FALLBACK` absent/false.
- MPS high-watermark override: absent; macOS guardrails were not changed.
- Full lock: `environments/stage1a_mps/requirements-lock.txt`, SHA-256
  `9adfd17bf39b20552af73eff90e659fb29c0a40adb06b8967fe7d47f853637fd`.
- The installed environment matched the lock exactly, its PEP 610/RECORD
  evidence matched the pinned source, imports passed, and `pip check` found no
  broken requirements.

## 5. Source and operator audit

The MPS implementation is separate from the official BF16 and T4 paths. Dense
model, transcoder, autograd, intervention, and numerical tensors are required
to remain on MPS/FP16. `PYTORCH_ENABLE_MPS_FALLBACK` is disabled. Native MPS
sparse COO is unsupported in this exact PyTorch build, so the only approved
execution deviation is explicit CPU storage of sparse coordinates/values while
the dense scientific payload remains on MPS. Bounded CPU comparisons verified
that boundary without changing scientific parameters.

The host preflight passed 13 required operators, transfers, backward hooks,
strict JumpReLU, a tiny hooked model, bounded graph construction, safetensors
disk-offload round-trip, finite/device checks, and the explicit sparse boundary.
The postmortem exposed a narrower deterministic operator failure missed by
preflight: FP16 MPS `index_put_` with accumulation at the next known
sparse-boundary check. This is a compatible likely explanation, not retained
proof of the original leaf error. The replacement is mathematically equivalent
because `nonzero` coordinates are unique, and a live MPS regression passed with
maximum absolute error `0.0`.

Remaining uncertainties are material: TransformerLens emitted its MPS warning
that PyTorch 2.6 may produce silently incorrect results, and the current
HF-model-plus-transcoder-plus-TransformerLens conversion plan exceeds the host
memory budget. No warning was suppressed. A future runtime would need a new,
explicitly audited loading-plan identity and renewed numerical validation; it
may not silently change PyTorch, offload mode, device, dtype, model, revision,
prompt, or any scientific parameter.

## 6. Memory feasibility

The pre-run plan used:

- physical memory: `34,359,738,368` bytes (32 GiB);
- safety fraction: 70%; conservative budget: `24,051,816,857` bytes;
- model snapshot: `10,479,239,529` bytes;
- transcoder snapshot: `7,855,395,802` bytes;
- model-resident estimate: `6,287,543,717` bytes;
- transcoder-resident estimate: `7,855,395,802` bytes;
- temporary/system headroom: `6,442,450,944` bytes;
- original estimated peak: `20,585,390,463` bytes;
- pre-run memory pressure: normal;
- pre-run swap: approximately 0.56 GB. The original runner did not persist the
  exact raw gate value, which is recorded as a diagnostic limitation.

One fresh worker was launched with configured batch label `256`. Attribution
never began, so this is not an accepted attribution batch. No `128` or `64`
worker was started. Batch size is first consumed by attribution and cannot
reduce `runtime_loading` memory. The prompt permits retry only for confirmed
MPS OOM; this failure was not classified as OOM, and the runner additionally
limits batch-based retry to the attribution stage where batch can act.

Attempt `attempt-256-yhp6ax3r` ran from `2026-08-23T11:38:05Z` to
`2026-08-23T12:46:43Z`, with `4,117.836113` seconds wall time and `3,858`
sampled observations. Sampled attempt/runtime-loading peaks were:

| Metric | Peak bytes |
| --- | ---: |
| MPS current allocated | 35,977,178,880 |
| MPS driver allocated | 40,032,174,080 |
| Worker process RSS | 5,948,440,576 |
| System-wide swap used | 34,567,031,357 |

Cleanup-stage sampled peaks were MPS current `9,172,934,656`, MPS driver
`21,725,134,848`, RSS `858,882,048`, and system swap `9,328,981,442` bytes.
`cleanup_succeeded=true` means only that the limited cleanup routine returned
without error. Because the bundle never returned, it did not call bundle close
or bundle-backed MPS cache/synchronization operations. It does not claim the
host was already recovered at that boundary. MPS current, driver, and RSS
counters can overlap in unified memory and are not added. System swap is
host-wide and is not falsely attributed entirely to the worker.

The worker targeted one-second sampling with `torch.mps` counters, process RSS,
memory pressure, and swap. The failed-attempt report retained the 3,858 count
and aggregate peaks, but discarded pressure states, recommended maximum,
method, interval, and first/final samples. Therefore a critical-pressure state
between the external five-minute checkpoints cannot be excluded. No explicit
MPS OOM was retained.

The driver peak alone is 166.4% of the 70% budget and 116.5% of physical
memory; swap exceeded the 4 GiB safety threshold by a wide margin. This
falsifies the static estimate for the identical loading plan. Commit `86ec118`
persists the observation fingerprint and fails closed before another large
load. After termination, external monitoring at 15:51, 15:56, and 16:01
Europe/Istanbul local time showed 87–88% free memory, nominal thermal pressure,
and SoC/PMU temperatures near 40–42 degrees C.

## 7. Immutable assets

- Upstream:
  `decoderesearch/circuit-tracer@8f1e2438df612464e229e44c4a00ff637bf9379b`.
- Model:
  `google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf`.
- Transcoder:
  `mwhanna/gemma-scope-transcoders@bd5773156dea09893636c801df1237d0410307d2`.

The progressive loader accepted exactly 10 required model files and 27
required transcoder files from immutable snapshots, with containment and
file-content verification, in a project-external cache. The model was resolved
and exercised before transcoder payload access; execution switched to offline
mode after both exact snapshots were resolved. No other Hugging Face model,
transcoder, or dataset snapshot was downloaded. Because the worker failed
before returning its bundle, no canonical `asset_manifest.json` was published;
none is fabricated or committed.

## 8. Model smoke test

The pinned Hugging Face Gemma model loaded on MPS in FP16 and passed the
progressive model-only forward gate before transcoder loading. The exact prompt
was `The capital of state containing Dallas is`. The tokenizer was loaded from
the same exact model revision and invoked with its default special-token
behavior; token shape was `[1, 8]` and logit shape was `[1, 8, 256000]`. The raw
token IDs were not retained. The logits were on MPS, FP16, finite, and had the
required vocabulary dimension. The in-memory record could not be published as
`model_smoke.json` because the later runtime-loading failure prevented the
bundle from returning; this observed gate is not a completed scientific result.

## 9. Loaded runtime semantics

**Status: not reached.** No loaded-runtime preactivation, raw threshold,
strict JumpReLU gate, feature activation, desired intervention mapping, repeat
error, or finite semantic summary exists. The tiny/preflight gate and the
postmortem sparse-boundary regression are engineering evidence only and do not
substitute for the required loaded Gemma/transcoder semantics.

## 10. Attribution

**Status: not reached.** Fixed configuration remained: prompt
`The capital of state containing Dallas is`, `max_n_logits=10`, desired logit
probability `0.95`, maximum feature nodes `8192`, disk offload, and initial
configured batch `256`. Accepted batch is `null`; retry count is zero. No
attribution call, graph, graph timing, node or edge count, selected logits, or
scientific result was produced.

## 11. Intervention

**Status: not reached.** Fixed configuration remained: prompt
`Hecho: Michael Jordan juega al`, feature `(layer=20, position=-1,
feature_id=341)`, alphas `0.0`, `0.5`, and `1.0`, frozen attention, and no
constrained layers. No baseline, no-op, half, full, error, or finite-output
result was produced.

## 12. Validation

- `.venv/bin/pytest -q` — PASS: 337 passed, 3 expected skips, 1 deselected.
- `.venv/bin/ruff check .` — PASS.
- `.venv/bin/ruff format --check .` — PASS: 70 files formatted.
- `.venv/bin/python -m mypy src scripts tests` — PASS: 54 source files.
- `.venv/bin/python scripts/doctor.py` — PASS, offline diagnostic.
- `.venv/bin/python scripts/verify_math.py` — PASS.
- `.venv-stage1a-mps-py311/bin/python -m pip check` — PASS: no broken
  requirements.
- `.venv-stage1a-mps-py311/bin/python -c 'import torch,transformer_lens,transformers,nnsight,huggingface_hub,circuit_tracer; print("imports: ok")'`
  — PASS.
- `.venv/bin/python scripts/stage1a/scan_commit_safety.py --mode tracked` —
  PASS: 98 paths, no findings.
- `.venv/bin/python scripts/stage1a/scan_commit_safety.py --mode candidates` —
  PASS: 105 paths, no findings.
- `git diff --check` — PASS.
- `.venv-stage1a-mps-py311/bin/python scripts/stage1a/probe_stage1a_mps.py --output results/generated/stage1a_mps_fp16/preflight-mndxhfn0/preflight_summary.json`
  — PASS at the execution commit: 13 operators plus all required gates.
- `.venv-stage1a-mps-py311/bin/python -c 'import sys; sys.path.insert(0,"src"); import torch; from cfsus.reproduction.mps_fp16 import _validate_live_sparse_metadata_boundary; result=_validate_live_sparse_metadata_boundary(torch); assert result["passed"] is True and result["maximum_absolute_error"] == 0.0; print("live MPS sparse boundary: PASS", result["maximum_absolute_error"])'`
  — PASS after the patch: maximum absolute error `0.0`.

The three skips were the two Torch-dependent sparse-boundary tests in the
lightweight `.venv` (Torch is intentionally absent there) and the sandboxed MPS
preflight test (MPS is not exposed to that test process). The deselection was
`tests/test_stage1a_model_runtime.py::test_loaded_official_feature_gate_and_cache_contract`,
excluded by the default `not model` marker. It is not called a pass. No notebook
changed on this branch, so notebook code-cell compilation was not applicable.

The real command was:

```text
.venv-stage1a-mps-py311/bin/python scripts/stage1a/run_stage1a_mps_fp16.py --config configs/stage1a_gemma2_2b_mps_fp16_reproduction.yaml --allow-download --hf-cache <PROJECT-EXTERNAL-CACHE>
```

It exited `2` after the non-retryable worker failure. The ignored preflight
record has SHA-256
`098d1cd0892cc3ef802375284aa51ab82b08079cc6cda3ebafbb19c683b5bb1c`;
the ignored attempt report has SHA-256
`bae2ee7e0061de08db8f71a6c9c0734212b4666c49d27a65bb7131cbaedfc49e`.
The strict completed-artifact validator was not run against real results
because no canonical completed bundle exists. This is `not run`, not a pass or
skip. Synthetic valid and mutation/rejection cases are covered by the passing
offline suite.

## 13. Files changed

The branch adds the separate MPS config and lock, MPS plan/report, dependency-
light runtime policy, native MPS preflight, isolated runner/worker, strict
artifact validator, MPS core adapter, and dedicated tests. The terminal update
also changes `docs/DECISIONS.md` and `docs/EXPERIMENT_LOG.md` and hardens the
worker diagnostic, unique-coordinate FP16 sparse check, and empirical loading
guard.

Final `git diff --stat d965e43c34a2ba408b8ae35b13b5651bf269beed`
at the terminal record:

```text
 .gitignore                                         |    1 +
 .../stage1a_gemma2_2b_mps_fp16_reproduction.yaml   |  106 +
 docs/DECISIONS.md                                  |   93 +
 docs/EXPERIMENT_LOG.md                             |   84 +-
 docs/STAGE_1A_MPS_FP16_PLAN.md                     |  209 ++
 docs/STAGE_1A_MPS_FP16_REPORT.md                   |  345 +++
 environments/stage1a_mps/README.md                 |   40 +
 environments/stage1a_mps/requirements-lock.txt     |  122 +
 scripts/stage1a/mps_runtime.py                     |  668 +++++
 scripts/stage1a/probe_stage1a_mps.py               |  879 ++++++
 scripts/stage1a/run_stage1a_mps_fp16.py            | 1183 +++++++++
 scripts/stage1a/run_stage1a_mps_fp16_worker.py     |  881 ++++++
 scripts/stage1a/scan_commit_safety.py              |    1 +
 scripts/stage1a/validate_mps_fp16_artifacts.py     | 2119 +++++++++++++++
 src/cfsus/reproduction/mps_fp16.py                 | 2799 ++++++++++++++++++++
 tests/test_stage1a_mps_fp16.py                     |  683 +++++
 tests/test_stage1a_mps_preflight.py                |  196 ++
 tests/test_stage1a_mps_runner.py                   | 1048 ++++++++
 tests/test_stage1a_mps_validator.py                | 1060 ++++++++
 19 files changed, 12516 insertions(+), 1 deletion(-)
```

No model weights, cache files, raw graphs, tensors, generated attempt payloads,
or the preserved T4 directory are in this diff.

## 14. Working tree

Final handoff verification of `git status --short` after committing this record
produced exactly:

```text
?? results/stage1a_t4_fp16/
```

The line is the intentionally preserved historical T4 directory. It is not
staged. Generated MPS attempts and the external asset cache are ignored and
uncommitted.

## 15. Claim boundary

This work is an Apple M2 Max/MPS/FP16 hardware adaptation. It is not the
official native-BF16 reproduction. It does not establish numerical equivalence
to BF16, CUDA, or T4, and it does not validate any Stage 1A attribution,
intervention, or susceptibility claim.

## 16. Stage 1B readiness

- Infrastructure verification: offline tests and the bounded patched live-MPS
  check passed; end-to-end runtime validity remains unestablished.
- `stage1b_engineering_readiness: false` — preflight and infrastructure alone
  cannot authorize Stage 1B; the required real MPS run did not complete.
- `stage1b_empirical_claim_readiness: false` — no validated MPS scientific
  artifact exists and the official native-BF16 reference remains pending.

## 17. Safety confirmation

No credential or secret is stored or printed in tracked evidence. No paid
compute was used. Only the two authorized, immutable Hugging Face asset
snapshots were downloaded, into a project-external cache. MPS fallback and the
high-watermark override remained disabled. There was no hidden fallback or
whole-runtime CPU/CUDA substitution; only the documented CPU sparse-metadata
boundary was used. No dtype, model, revision, prompt, batch-policy, or
scientific-parameter substitution was made.
No raw graph, activation, logits array, model/transcoder weight, cache, failed
generated payload, or external-backup path was committed. No merge, PR, tag,
release, or push to `main` occurred. The completed MPS status was not used.
