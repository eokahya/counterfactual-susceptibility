# Stage 1A Apple M2 Max / MPS FP16 hardware-adaptation plan

Status: `INFRASTRUCTURE_VALIDATED_REAL_RUN_PENDING`

## Claim boundary

This path is a separate Apple M2 Max / MPS / FP16 hardware adaptation of the
pinned Stage 1A experiment. It is not the pending official native-BF16
reproduction and it cannot establish numerical equivalence to BF16, CUDA, or
the earlier T4 run. The prior T4 outputs are historical context only: their
small artifact bundle has an invalid attempt-level peak-memory provenance
record and is not acceptance evidence for this run.

No scientific input changes in this path. Device placement, sampled MPS
telemetry, project-external cache placement, fresh-process isolation, and the
documented CPU sparse-COO metadata boundary are execution-only adaptations.

## Immutable identity

| Item | Required identity |
| --- | --- |
| Project base | `d965e43c34a2ba408b8ae35b13b5651bf269beed` |
| `circuit-tracer` | `8f1e2438df612464e229e44c4a00ff637bf9379b` (`0.5.2`) |
| Model | `google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf` |
| Transcoder | `mwhanna/gemma-scope-transcoders@bd5773156dea09893636c801df1237d0410307d2` |
| Python | native-arm64 CPython `3.11.13` |
| Preferred PyTorch | macOS arm64 `2.6.0` |
| Backend/device/dtype | TransformerLens / MPS / FP16 |
| Runtime class | `hardware_adapted_mps_fp16` |
| Completed status | `completed_hardware_adapted_mps_fp16` |

The fixed attribution prompt is `The capital of state containing Dallas is`,
with ten logits, desired logit probability `0.95`, at most `8192` feature
nodes, initial batch `256`, and disk offload. The fixed intervention prompt is
`Hecho: Michael Jordan juega al`, feature `(20, -1, 341)`, alphas
`[0.0, 0.5, 1.0]`, frozen attention, and no constrained-layer range.

## Preservation and Git isolation

Before implementation, the seven regular files in the untracked
`results/stage1a_t4_fp16/` directory were copied to a project-external backup
with metadata preservation. Independent SHA-256 and byte comparisons passed
for all seven files. The portable report records that verification but not the
machine-specific external backup path. The original directory remains
unmodified, untracked, and excluded from all MPS commits.

Work starts from the exact base commit on the separate
`stage-1a-mps-fp16` branch. `main` and `stage-1a-t4-fp16` are protected from
changes. Only the MPS branch may be pushed; no merge or pull request is part of
this task.

## Pinned source and MPS operator audit

The audit covers the exact installed upstream commit and TransformerLens
version, not mutable branches. The important execution sites are:

| Source path | Role and relevant behavior | MPS conclusion |
| --- | --- | --- |
| `replacement_model/replacement_model_transformerlens.py` | TransformerLens hooks, replacement forward, activation capture, and feature intervention | Caller must pass `device=mps` and `dtype=float16`; upstream default-device selection is not MPS-aware. |
| `attribution/attribute_transformerlens.py` | Full graph attribution, CPU graph buffers, feature batching, and final graph construction | Dense model/autograd work remains MPS; CPU graph storage is already explicit upstream behavior. |
| `attribution/context_transformerlens.py` | Backward hooks, `index_put_`, repeated backward passes, and attribution-row capture | Tiny MPS probes must cover hooks and indexed gradient replacement before weights. |
| `transcoder/single_layer_transcoder.py` | `F.linear`, strict JumpReLU, sparse COO conversion, decoder lookup, and `index_add_` reconstruction | Native MPS sparse COO conversion is unavailable in PyTorch 2.6.0; use the explicit, numerically checked CPU sparse-metadata adapter described below. |
| `utils/disk_offload.py` | Safetensors save, move to `meta`, and assignment reload to the requested device | A real tiny MPS round trip is a preflight gate; disk offload remains the preferred execution mode only if it passes. |
| TransformerLens Gemma loading/hooks | Eager attention, model conversion, device placement, hook registration | Model-only MPS forward and tensor device/dtype assertions are mandatory before transcoder access. |

The bounded preflight exercises FP16 allocation, matrix multiplication,
`einsum`, softmax, layer normalization, `topk`, gather, scatter, `index_add_`,
`index_put_`, masked `where`, sort, unique, searchsorted, explicit CPU/MPS
transfers, autograd hooks, strict JumpReLU, a tiny hooked transformer,
bounded graph construction, safetensors/disk offload, finite comparisons, and
memory counters. PyTorch may print an indexed device such as `mps:0`; device
validation uses the device type and never treats a CPU tensor as MPS.

## Explicit sparse-metadata execution deviation

PyTorch 2.6.0 does not implement the pinned upstream `acts.to_sparse()` path on
MPS. Enabling `PYTORCH_ENABLE_MPS_FALLBACK` would hide an unsupported operation
and is forbidden. The project-local adapter therefore performs these explicit
steps:

1. compute preactivations, JumpReLU activations, active encoder/decoder vectors,
   and dense reconstruction on MPS FP16;
2. obtain active coordinates and values explicitly;
3. construct/coalesce only the sparse COO indices and values on CPU for graph
   metadata operations;
4. keep dense model activations, decoder reconstruction, hooks, gradients, and
   intervention tensors on MPS;
5. compare the adapter against a bounded dense CPU reference within the
   preregistered FP16 tolerances.

This does not change feature selection, activation values, reconstruction, or
the mathematical attribution experiment. Any unexplained dense-tensor CPU move,
failed equivalence check, or required fallback makes the exact path infeasible.

## Isolated environment gate

The selected environment is `.venv-stage1a-mps-py311/`; it is ignored and
forbidden by the commit-safety scanner. The tracked observed lock is
`environments/stage1a_mps/requirements-lock.txt`. Acceptance requires:

- CPython 3.11 and `arm64`;
- PyTorch 2.6.0 with MPS built and available;
- exact installed `circuit-tracer` VCS provenance;
- `pip check` success and an exact match to the observed lock;
- no CUDA/NVIDIA runtime packages;
- fallback disabled and no zero high-watermark override;
- real FP16 MPS allocation/matmul and the complete bounded preflight passing.

PyTorch 2.7.1 may be evaluated only after a recorded 2.6.0 incompatibility and
would require a distinct immutable runtime class and lock. Python 3.13 is not a
permitted final runtime.

## Conservative 32 GiB memory gate

Metadata-only queries at the exact revisions identify approximately
`10,479,239,529` required model bytes and `7,855,395,802` required transcoder
bytes. The pre-download estimate accounts for FP16 resident weights, serialized
or conversion overlap, MPS driver allocation, sparse metadata and CPU graph
storage, disk-offload buffers, process RSS, and a six-GiB macOS/Codex reserve.

The gate records physical memory, current memory pressure, swap, conservative
budget and estimate, and the chosen execution deviation. It may return only
`feasible`, `feasible_with_explicit_execution_deviation`, or a precise blocked
status. Large downloads are authorized only for the first two. Critical memory
pressure stops a worker.

MPS memory evidence is sampled, not a CUDA allocator high-water mark. Every
stage and attempt records sample method/count/interval, current allocated MPS
bytes, driver allocated bytes, recommended maximum where available, process
RSS, pressure, swap, and wall time. Each attempt peak must be greater than or
equal to every corresponding stage peak.

## Secure progressive asset plan

Only the two exact authorized repositories and revisions may be queried or
downloaded, using an existing secure Hugging Face login without printing,
hashing, or serializing credentials. The cache must remain outside the project.
The loader follows this order:

1. resolve metadata/config/tokenizer for the exact Gemma revision;
2. resolve only the required Gemma weight shards;
3. verify contained cache links, file set, sizes, revision, and SHA-256;
4. load Gemma on MPS FP16 and run a finite model-only prompt forward;
5. only after that success, resolve and verify the 26 required transcoder
   safetensors plus `config.yaml`;
6. load the transcoders and run loaded feature/semantics smoke checks;
7. run a small engineering attribution smoke, then the full fixed reproduction.

Valid cached content is reused. Mutable revisions, incomplete shards, escaping
symlinks, unmanifested consumed files, or a manifest mismatch stop execution.

## Runtime, retry, and output isolation

Heavy work runs in a fresh worker. Batch `256` is attempted first. Only a
positively identified MPS/Metal out-of-memory condition during the attribution
stage may start a clean worker at `128`, then `64`. Generic memory errors,
operator failures, access errors, non-finite values, assertion failures,
critical pressure, invalid artifacts, and unknown process exits never authorize
a retry. Each attempt has an isolated ignored output directory; failed files
cannot enter the accepted bundle.

Loaded-runtime acceptance includes real preactivation-with-bias projection,
threshold retrieval, strict `z > threshold`, active and inactive examples,
feature `(20, -1, 341)` baseline activation, desired-value mapping, baseline
repeat, no-op, half suppression, full ablation, finite logits/probabilities,
and declared discrepancy/tolerance records. Attribution must return a nonempty
graph with valid dimensions, selected features, nonzero edges, ten logit nodes,
and finite summaries. No raw graph is committed.

## Artifact and validation boundary

The completed allowlist under `results/stage1a_mps_fp16/` is exactly:

- `preflight/preflight_summary.json`
- `feasibility_report.json`
- `environment_manifest.json`
- `asset_manifest.json`
- `attribution_summary.json`
- `intervention_summary.json`
- `semantics_summary.json`
- `memory_summary.json`
- `stage1a_mps_run_manifest.json`
- `checksums.sha256`

The validator rejects wrong pins/commit/runtime, unavailable MPS, fallback,
scientific CPU placement, non-FP16 execution, CUDA/T4-only evidence, BF16 or
equivalence overclaims, missing loaded evidence, non-finite values, empty graph,
invalid intervention/no-op records, peak/timing inconsistencies, extra or
special files, duplicate keys, oversized arrays, path traversal, secrets,
weights/caches/raw graphs, checksum mismatch, and completion/readiness labels
unsupported by the real evidence.

`stage1b_engineering_readiness` may become true only after the real run and the
strict validator pass. `stage1b_empirical_claim_readiness` remains false because
the official native-BF16 reference is still pending.

## Validation and stop rules

Offline acceptance runs pytest, Ruff, Ruff format, mypy, the repository doctor,
mathematical verification, environment import/lock/`pip check`, commit-safety
scans, duplicate/mutation validator tests, and `git diff --check`. Host-level
acceptance additionally runs the complete MPS preflight and the real artifact
validator. Protected Git refs and the preserved T4 hashes are checked again
before the final push.

Any gate named in the task's stop conditions ends the real run with an honest
blocked/failed status. Infrastructure may still be committed if it is valid,
but the completed status and experiment-log entry are forbidden unless every
real-runtime acceptance criterion passes.
