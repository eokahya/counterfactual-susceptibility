# Stage 1A Apple M2 Max / MPS FP16 report

Status: `PREPARED_NOT_EXECUTED`

The native-arm64 Python 3.11.13 / PyTorch 2.6.0 bounded MPS preflight passed
with fallback disabled. All thirteen operator checks, transfers, autograd
hooks, strict JumpReLU, the tiny hooked model, bounded graph construction, the
explicit sparse-metadata boundary, and the MPS safetensors/disk-offload round
trip passed. The native sparse COO operation itself was confirmed unsupported,
which activates the preregistered explicit CPU sparse-metadata execution
deviation; it does not authorize hidden fallback or dense scientific CPU work.

This report is created with the separate MPS implementation and will be
finalized only after the real pinned Gemma/transcoder run either passes the MPS
artifact validator or reaches a precise stop condition. Preflight or scaffold
tests alone are not scientific completion evidence.

The complete offline repository suite currently passes, including the
MPS-specific producer/validator mutation tests, strict static typing, lint and
format checks, deterministic mathematics verification, package-lock equality,
`pip check`, and publication-safety scanning. The real pinned assets have not
yet been downloaded or executed at this report state.

The seven historical T4 artifact files were backed up externally and verified
by independent SHA-256 and byte comparison. Their original untracked directory
remains unmodified and unstaged. The external backup path is intentionally not
stored in this portable report.

The immutable inputs, source/operator audit, explicit CPU sparse-metadata
boundary, memory plan, progressive loading order, retry policy, artifact
allowlist, and claim boundary are recorded in
`docs/STAGE_1A_MPS_FP16_PLAN.md`. The final report will record the exact
execution commit, environment, telemetry, asset hashes, model smoke, loaded
semantics, attribution, intervention, validation commands, Git state,
readiness, and safety outcome.

Claim boundary: this is an Apple M2 Max/MPS FP16 hardware adaptation. It is not
the pending official BF16 reproduction and does not establish numerical
equivalence to BF16, CUDA, or T4.
