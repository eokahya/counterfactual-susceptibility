# Stage 1A T4/FP16 report

Status: `PREPARED_NOT_EXECUTED`

## Outcome

The separate T4/FP16 hardware-adapted path is prepared for execution on a free
Colab T4. No empirical T4 run has been performed or imported, so this document
contains no attribution, intervention, semantic, memory, timing, or numerical
result.

The official BF16 reproduction remains pending. This preparation is not an exact
reproduction, does not demonstrate equivalence or invariance across dtypes, and
does not support final Counterfactual Susceptibility claims.

## Prepared scope

- Immutable project, upstream, model, and transcoder revisions are enforced.
- The official attribution and intervention scientific settings are preserved.
- The sole planned numerical deviation is execution dtype `float16` instead of
  the `bfloat16` reference dtype.
- Attribution begins at batch 256; fresh-process 128 and 64 retries are allowed
  only after a positively identified CUDA OOM.
- Loaded-runtime, finiteness, strict JumpReLU, desired-value, baseline repeat,
  no-op, intervention repeat, raw graph, checksum, and small-bundle checks fail
  closed.
- The tracked notebook has no execution outputs and reads `HF_TOKEN` only from
  Colab secrets into a child-process environment.
- The notebook requires Colab runtime `2025.07` and installs/verifies exact
  `python3.11-venv`, `python3-pip-whl`, and `python3-setuptools-whl` package
  versions before creating its isolated environment.
- The T4 gate is the device identity plus compute capability `[7, 5]`;
  `torch.cuda.is_bf16_supported()` is retained as observed metadata and is not
  interpreted as a native-BF16 reproduction result.
- The pinned Hugging Face model uses its low-CPU-memory loading path to avoid a
  duplicate host-RAM copy during TransformerLens conversion; this does not
  alter weights, dtype, or scientific parameters.

## Empirical fields pending execution

The following remain unknown until the returned bundle validates:

- actual GPU name, compute capability, driver/CUDA details, and BF16 support;
- selected attribution batch and retry history;
- runtime semantics outcome;
- nonempty attribution graph outcome;
- baseline/no-op/half/full intervention outcomes;
- NaN/Inf counts or other FP16 instability;
- timings and peak memory;
- final terminal status and Stage 1B engineering readiness.

Until then:

```text
stage1b_engineering_readiness: false
stage1b_empirical_claim_readiness: false
native_bf16_reference_status: pending
```

No entry has been added to `docs/EXPERIMENT_LOG.md`, because preparing an
unexecuted notebook is not an experiment.
