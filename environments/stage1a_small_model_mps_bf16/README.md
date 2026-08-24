# Stage 1A-S-BF16 native MPS environment

This environment is dedicated to the separate Gemma 3 270M, PLT, NNsight,
Apple-MPS/BF16 runtime-validation class. It does not mutate or relabel the
protected all-FP16 environment or result.

The source audit retained native arm64 CPython 3.11.13, PyTorch 2.6.0,
NNsight 0.6.1, Transformers 4.57.3, Hugging Face Hub 0.36.2,
safetensors 0.8.0, and official `circuit-tracer` v0.5.2 at exact commit
`8f1e2438df612464e229e44c4a00ff637bf9379b`. No dependency change was needed
solely for BF16.

The runtime environment is created separately at
`.venv-stage1a-small-model-mps-bf16`. Acceptance requires native `arm64`,
CPython 3.11.x, exact `pip freeze --all` agreement with the non-comment lines
of `requirements-lock.txt`, a matching circuit-tracer `direct_url` commit,
clean `pip check`, real MPS availability, and an absent/false
`PYTORCH_ENABLE_MPS_FALLBACK`.

The Gemma checkpoint stores model tensors as BF16. The selected PLT files store
five tensor families as FP32 and the audited loader converts them explicitly to
BF16 for accepted execution. Transformers performs source-mandated FP32
suboperations inside Gemma 3 RMSNorm, rotary frequency/trigonometry, and
attention softmax, then casts results back to the input dtype on MPS. These are
enumerated validation exceptions, not an outer autocast or fallback.
