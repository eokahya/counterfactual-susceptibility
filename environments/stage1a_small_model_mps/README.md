# Stage 1A-S native MPS environment

This environment is dedicated to the Gemma 3 270M, PLT, NNsight,
Apple-MPS/FP16 engineering pilot. It is separate from the Gemma 2 BF16,
T4/FP16, and earlier MPS environments.

The selected stack is CPython 3.11 on native arm64, PyTorch 2.6.0,
NNsight 0.6.1, Transformers 4.57.3, and official `circuit-tracer` v0.5.2 at
commit `8f1e2438df612464e229e44c4a00ff637bf9379b`. PyTorch 2.6.0 is retained
because it is compatible with the audited upstream dependency range and has
already passed a real local MPS FP16 allocation/matmul check on this host.

Acceptance requires a newly created environment whose `pip freeze --all`
matches `requirements-lock.txt`, an absent/false
`PYTORCH_ENABLE_MPS_FALLBACK`, native `arm64`, and passing no-download MPS
operator probes. CPU is permitted only for explicit graph/sparse metadata and
small publication summaries, never for model or PLT tensor computation.
