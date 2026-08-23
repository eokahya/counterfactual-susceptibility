# Stage 1A MPS FP16 environment

This directory documents the separate Apple Silicon runtime for the Stage 1A
hardware adaptation. It is not the official native-BF16 environment and it is
not a CUDA/T4 environment.

## Required runtime

- native `arm64` macOS on the Apple M2 Max (32 GiB unified memory);
- CPython 3.11.13;
- PyTorch 2.6.0 with MPS built and available;
- TransformerLens and `circuit-tracer` 0.5.2 from audited commit
  `8f1e2438df612464e229e44c4a00ff637bf9379b`;
- execution dtype `float16`, backend `transformerlens`, device `mps`;
- `PYTORCH_ENABLE_MPS_FALLBACK` unset or explicitly false.

The complete observed package lock is
`environments/stage1a_mps/requirements-lock.txt`. It was generated with
`python -m pip freeze --all`; the installed environment matched it exactly and
`pip check` reported no broken requirements. A recreated environment must pass
the same lock comparison, native-arm64 checks, exact VCS-provenance check, and
`scripts/stage1a/probe_stage1a_mps.py` before any large snapshot is accessed.

## Execution policy

The scientific inputs are fixed by
`configs/stage1a_gemma2_2b_mps_fp16_reproduction.yaml`. Disk offload is the
preferred PyTorch 2.6.0 mode and is tested with an actual safetensors round
trip. A confirmed MPS out-of-memory condition may retry in a fresh process at
batch sizes `256`, `128`, then `64`; generic errors must not trigger a retry.

The probe records MPS allocator samples, process RSS, explicit CPU transfers,
device placement, finite values, and FP16 tolerances. Native SparseMPS is not
silently replaced: if the native sparse operation is unsupported, the result
records that expected gap and requires the tested CPU sparse-metadata plus MPS
dense-payload boundary to pass.

The model and transcoder snapshots remain project-external Hugging Face cache
assets. No weights, cache files, raw activations, or graph objects belong in
this repository or in the small `results/stage1a_mps_fp16/` artifact bundle.
