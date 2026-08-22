# Stage 1A T4/FP16 hardware-adaptation plan

Status: `PREPARED_NOT_EXECUTED`

## Claim boundary

This is a T4/FP16 hardware-adapted runtime/API reproduction using the pinned
assets; native-BF16 reference reproduction remains pending. It is not the
official BF16 reproduction, an exact numerical reproduction, evidence of dtype
invariance, or authority for final Counterfactual Susceptibility claims.

The original BF16 configuration, notebook, report, and existing artifacts remain
the reference plan and are not weakened or relabeled. The sole planned numerical
change in this path is `bfloat16 -> float16`. A reduced attribution batch is
permitted only as the declared CUDA-OOM engineering deviation below.

## Immutable identity

| Item | Required identity |
| --- | --- |
| Project base | `13b42a5debe38def14f173530bcbc81ca3f8440e` |
| `circuit-tracer` | `8f1e2438df612464e229e44c4a00ff637bf9379b` |
| Model | `google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf` |
| Transcoder | `mwhanna/gemma-scope-transcoders@bd5773156dea09893636c801df1237d0410307d2` |
| Python | `3.11` |
| Colab runtime | `2025.07` |
| Venv bootstrap | `apt-get update`; `python3.11-venv=3.11.15-1+jammy1`; `python3-pip-whl=22.0.2+dfsg-1ubuntu0.7`; `python3-setuptools-whl=68.1.2-2~jammy3` |
| PyTorch wheel | `2.6.0`, CUDA 12.4 (`cu124`) |
| Backend | `transformerlens` |
| Reference / execution dtype | `bfloat16` / `float16` |

The attribution prompt, logit target settings, node cap, intervention prompt,
feature `(20, -1, 341)`, alpha values, and intervention regime remain identical
to the BF16 plan. The T4 path uses disk attribution offload.

The pinned Colab image omits `ensurepip` from its base Python installation. The
notebook therefore refreshes the image package indexes, installs the exact
image-compatible Ubuntu packages shown above, verifies their consumed versions
with `dpkg-query`, and only then creates the isolated virtual environment. No
mutable OS-package fallback is permitted.

The T4 gate uses the observed device name and compute capability `[7, 5]`.
`torch.cuda.is_bf16_supported()` is recorded as an API observation only: its
value is not treated as proof for or against a native-BF16 reference
reproduction, which remains pending in either case.

The pinned Hugging Face model is loaded with `low_cpu_mem_usage=True` before
TransformerLens conversion. TransformerLens' destination parameters are then
allocated directly on CUDA instead of first allocating a second full model on
the 12.7-GiB Colab host. The T4 path caps TransformerLens' constant attention
buffers at 512 tokens. Every preregistered prompt is shorter than this cap, and
512 is below Gemma 2's 4096-token local-attention window, so the cap does not
change attention semantics for the executed inputs. These are host/GPU-memory
loading adaptations only: model weights, dtype, attention implementation, and
scientific settings are unchanged. The consumed context length and loader name
are recorded in runtime provenance. The Colab runner also uses unbuffered child
output so a backend failure leaves an observable stage boundary.

## Pinned upstream source audit

The audit was performed against the immutable upstream commit, not a mutable
branch:

- `ReplacementModel.from_pretrained` accepts and forwards a caller-supplied
  `torch.dtype` in
  [`replacement_model.py` lines 25–68](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model.py#L25-L68).
- The TransformerLens loader passes that dtype to both transcoder loading and
  `HookedTransformer.from_pretrained` in
  [`replacement_model_transformerlens.py` lines 123–168](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L123-L168).
- The CLI explicitly accepts `float16`/`fp16` and maps the aliases to
  `torch.float16` in
  [`__main__.py` lines 49–66](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/__main__.py#L49-L66)
  and
  [`__main__.py` lines 203–253](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/__main__.py#L203-L253).
- Hub and per-layer transcoder loading propagate/cast dtype in
  [`hf_utils.py` lines 47–118](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/hf_utils.py#L47-L118)
  and
  [`single_layer_transcoder.py` lines 530–560](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L530-L560).
  Lazy encoder inputs are cast to `W_enc.dtype` at
  [`single_layer_transcoder.py` lines 98–125](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L98-L125).
- A GemmaScope2 CLT loader initially leaves `W_skip` uncast, but the subsequent
  registered-module `.to(device, dtype)` call casts it. Runtime assertions still
  inspect `W_skip` when present. See
  [`cross_layer_transcoder.py` lines 528–529](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L528-L529)
  and the model move linked above.
- `batch_size` is documented as the number of source nodes processed per backward
  pass in
  [`attribute_transformerlens.py` lines 56–77](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L56-L77).
  Logit and feature targets are chunked at
  [`attribute_transformerlens.py` lines 193–245](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L193-L245),
  while `compute_batch` performs the corresponding injected-gradient backward
  pass at
  [`context_transformerlens.py` lines 168–232](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/context_transformerlens.py#L168-L232).
  Lowering the batch therefore preserves the prompt, target logits, objective,
  node cap, model, and transcoder. It can still change floating-point accumulation
  and feature-queue cadence, so no bitwise-equivalence claim is made.
- Upstream graph buffers are created without a dtype argument and therefore use
  the default FP32 dtype; graph normalization also acts on this storage. This is
  an expected internal storage behavior, not a model/transcoder dtype failure:
  [`attribute_transformerlens.py` lines 187–190](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L187-L190)
  and
  [`graph.py` lines 391–399](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/graph.py#L391-L399).

The pinned source supports FP16 at the API level, but upstream has no same-path
T4/FP16 empirical test. That unclosed risk is why this adaptation is fail-closed
and why the BF16 reference remains pending.

## Execution and retry policy

The tracked parent runner validates a clean checkout containing the required base
commit, resolves only immutable assets, then launches one scientific attempt in a
fresh child process. It tries batch `256` first. Only a CUDA-specific OOM type or
message can advance to `128`, then `64`. Each exited worker clears its CUDA cache;
process exit releases its CUDA context before the next worker starts.

Every attempt records batch size, outcome, exact exception class, sanitized
message, failure stage, peak CUDA allocation when available, elapsed time, and
cleanup result. Generic `MemoryError`, CPU out-of-memory text, illegal memory
access, asset mismatch, access failure, dtype failure, assertion failure,
non-finite output, or no-op/semantic failure never causes a batch retry. OOM at
64 becomes `blocked_resource`. A success at 128 or 64 records an explicit batching
deviation and makes no bitwise-equivalence claim.

## Preregistered FP16 safeguards

These tolerances are fixed in the tracked configuration before execution:

| Check | Absolute tolerance | Relative tolerance |
| --- | ---: | ---: |
| Loaded gate/cache agreement | `0.005` | not used |
| Manual encoder projection | `0.005` | not used |
| Baseline/no-op logits | `0.02` | `0.002` |
| Repeated baseline/intervention logits | `0.02` | `0.002` |

The gate/projection values match existing loaded-runtime tests; the no-op values
reuse the existing Stage 1A runtime tolerances. The runtime also requires:

- a finite CUDA FP16 matrix-multiply probe;
- deterministic samples from every loaded model parameter tensor to be finite;
- every loaded threshold tensor to be finite;
- dtype/shape checks across all transcoder layers and optional `W_skip`;
- finite preactivation and post-gate caches, baseline/intervention logits and
  probabilities, attribution effects, and summary numbers;
- strict JumpReLU equality-inactive behavior on loaded modules;
- runtime verification of `desired=(1-alpha)*baseline_activation` for `0`, `0.5`,
  and `1`;
- repeated baseline, repeated no-op, and repeated full-ablation checks;
- raw graph reload, stable checksum, and nonempty selected-feature checks.

Any non-finite value or failed no-op/runtime semantic check is
`failed_precision`; it is not patched around.

## Artifacts, statuses, and readiness

Small artifacts are isolated under `results/stage1a_t4_fp16/`. The successful
allowlist is exactly five JSON summaries/manifests, `checksums.sha256`, and
`stage1a_t4_fp16_run_manifest.json`. The raw graph remains under ignored
`results/generated/stage1a_t4_fp16/` and cannot enter the ZIP.

Allowed terminal statuses are:

- `completed_hardware_adapted_fp16`
- `blocked_access`
- `blocked_resource`
- `failed_precision`
- `failed_runtime`
- `prepared_not_executed`

The bundle validator rejects unknown members, model/tensor extensions, symlinks,
absolute paths, traversal, duplicate members, members over 5 MiB, total expanded
content over 20 MiB, malformed JSON, credentials, and private filesystem paths.
The run manifest records immutable identities, project SHA/clean state, GPU and
compute capability, CUDA/PyTorch versions, dtype provenance, batch history,
timings, peak memory, checks, readiness, artifact sizes/digests, and the BF16
pending statement.

The run manifest records sizes and SHA-256 values for the other six small files;
it cannot recursively record its own final digest. The notebook validates all
seven members and prints the final ZIP SHA-256 as the external handoff identity.

`stage1b_engineering_readiness` can become true only after all required checks and
artifact validation pass. `stage1b_empirical_claim_readiness` is always false for
this task.
