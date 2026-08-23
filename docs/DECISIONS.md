# Decision Log

## D-001 — 2026-07-25 — Start with single-source suppression

**Decision:** The first operational problem is prediction of inactive-feature gate crossings caused by suppressing one active upstream feature.

**Reason:** It has clear ground truth, supports continuous intervention sweeps, and isolates the central active-only graph blind spot before introducing combinatorial multi-source search.

**Alternatives considered:** Begin with multi-source suppression or full graph
augmentation. Both add combinatorial and interpretive complexity before the
pairwise premise is tested.

**Revisit when:** Verified crossings are too rare, or multi-source interactions dominate prediction error.

## D-002 — 2026-07-25 — Separate activation susceptibility from behavioral salience

**Decision:** Use distinct scores for ease of gate crossing and predicted behavioral consequence.

**Reason:** A feature can be easy to activate but behaviorally irrelevant, or hard to activate but highly consequential.

**Alternatives considered:** Collapse gate proximity and downstream effect into
one headline score. This would make it difficult to diagnose whether an error
comes from gate-crossing prediction or behavioral prediction.

## D-003 — 2026-07-25 — Require mediation, not only correlation

**Decision:** High-level causal claims require source suppression plus target-clamp mediation tests.

**Reason:** A target may activate as a side effect without mediating the output change.

**Alternatives considered:** Treat an observed target activation or correlation
with an output change as sufficient evidence. Neither isolates the target's
causal contribution.

## D-004 — 2026-07-25 — Use open tooling before custom model training

**Decision:** Begin with `circuit-tracer` and available pretrained transcoders rather than training a new CLT.

**Reason:** This tests the research idea with lower engineering and compute risk. Training methodology is outside the first paper's core contribution.

**Alternatives considered:** Train a new CLT before testing the method. That
would confound representation learning with the gate-crossing hypothesis and
substantially increase Stage 0 scope.

## D-005 — 2026-08-01 — Preserve local suppression semantics at the backend boundary

**Decision:** The project API defines suppression strength as
`a_j -> (1 - alpha) a_j`, with `alpha` in `[0, 1]`. Any differing upstream
intervention convention must be translated by the backend and documented; it
must not redefine the scientific quantities.

**Reason:** A stable local convention keeps susceptibility predictions, sweeps,
and observed critical strengths comparable across backends.

**Alternatives considered:** Expose upstream intervention values directly or
change notation to match the first supported backend. Either would leak
version-specific behavior into the scientific API.

## D-006 — 2026-08-01 — Keep Stage 0 mathematics backend-independent

**Decision:** Fixed pairwise mathematics and its validation live outside the
`circuit-tracer` adapter. The Stage 0 adapter exposes only source-verified
capabilities and must fail explicitly for unsupported operations.

**Reason:** The score definitions can be tested offline, while upstream model,
transcoder, and intervention semantics are version-specific and may remain
partially unsupported.

**Alternatives considered:** Import upstream classes into the mathematical core
or implement speculative adapter behavior from names and documentation alone.
Both would make tests less deterministic and could silently encode incorrect
semantics.

## D-007 — 2026-08-01 — Do not treat deterministic verification as an experiment

**Decision:** Unit tests and the deterministic worked-example verification are
engineering checks, not empirical experiment entries. The experiment log stays
empty until a declared model/transcoder run is completed.

**Reason:** Calling formula checks experiments would blur the distinction between
validated arithmetic and evidence from an actual model intervention.

**Alternatives considered:** Record the Stage 0 math script as E0. E0 requires
upstream reproduction and intervention-semantic verification, which a pure
numerical example cannot provide.

## D-008 — 2026-08-01 — Clarify regularization, gate-boundary, and E0 scope

**Decision:** Make only the following consistency clarifications to the research
specification during Stage 0, without changing the primary definitions
`q = -a_j J_ij`, `S = q / (m_i + epsilon)`, or
`alpha_hat_star = m_i / q` for `q > 0`:

1. Add an explicit numerical worked example because the required Stage 0 test
   references one but Research Specification v0.1 did not contain one.
2. State that the reciprocal shorthand
   `S = 1 / (alpha_hat_star + epsilon_prime)` is exact only when
   `epsilon_prime = epsilon / q` for `q > 0`; the primary `q / (m + epsilon)`
   form governs implementations.
3. Treat `alpha_hat_star = 1` as a gate-semantic boundary: the local prediction
   reaches `z_i = tau_i`, but whether activation is nonzero depends on the
   loaded transcoder's strict (`>`) or non-strict (`>=`) threshold rule.
4. Defer E0's real-model attribution/intervention reproductions to Stage 1.
   Stage 0 performs the pinned source audit, typed scaffold, and offline
   validation without downloading or executing model/transcoder weights.

**Reason:** These changes reconcile internal wording and the Stage 0 validation
request while preserving the scientific quantities and the no-weight-download
scope. In particular, a threshold equality must not be mislabeled as a verified
crossing before the exact upstream gate rule is applied.

**Alternatives considered:** Redefine susceptibility to force an independent
`epsilon_prime`, classify equality as definitely crossing, omit the requested
worked example, or execute E0 model reproductions during Stage 0. Each option
would respectively change the fixed mathematics, hide an upstream-dependent
boundary, leave the acceptance test without a source example, or violate the
stage's offline safety boundary.

## D-009 — 2026-08-01 — Pin direct Stage 1A assets and reject a transitively mutable alias target

**Decision:** Select the exact model snapshot
`google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf` and the exact
transcoder snapshot
`mwhanna/gemma-scope-transcoders@bd5773156dea09893636c801df1237d0410307d2`.
The model must be consumed from an exact-SHA local snapshot because the pinned
upstream replacement-model constructor has no model-revision argument. Pass
the transcoder as an explicit `repo@sha`; do not use the bare `gemma` shortcut.

**Reason:** At the pinned `circuit-tracer` revision, the `gemma` loader alias
maps to `mwhanna/gemma-scope-transcoders`, whose selected revision directly
contains the 26 safetensors files. The upstream README instead names
`mntss/gemma-scope-transcoders@9250a2d4860ce5ed5c96c14d5882b7d8162809a3`,
but that repository's pinned configuration contains 26 unrevisioned
`hf://google/gemma-scope-2b-pt-transcoders/.../params.npz` references. Pinning
the outer `mntss` repository therefore does not transitively freeze the Google
assets, even though the currently inspected Google source revision is
`50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`.

**Alternatives considered:** Use the unrevisioned `gemma` shortcut; select the
README's outer `mntss` pin without pinning all referenced Google assets; or use
a floating model identifier. Each would leave at least one consumed artifact
mutable or make the exact runtime unrecoverable.

## D-010 — 2026-08-01 — Make device, dtype, and offload policy explicit by environment

**Decision:** The intended local path uses the TransformerLens backend, probes
MPS with a real allocation before loading assets, uses `bfloat16`, and uses CPU
offload where the upstream API supports it. If the probe fails, do not silently
run the full attribution or intervention workload on CPU; CPU is limited to
metadata, tokenizer/configuration checks, unit tests, and small semantic
checks. The prepared Colab fallback must probe CUDA explicitly, use `bfloat16`
when supported, and use the upstream official `disk` attribution offload
policy.

**Reason:** The local PyTorch environment reports MPS as built but unavailable,
and an actual MPS allocation raises an OS-version error; CUDA is also absent.
A silent full CPU fallback would turn an explicit feasibility failure into an
unbounded run. The official Colab path is the supported way to obtain CUDA and
disk offload while keeping device behavior recorded and reviewable.

**Alternatives considered:** Force MPS despite the failed allocation, silently
fall back to CPU for the full workload, use `float32` everywhere, or use disk
offload locally. These options respectively ignore a hard runtime failure,
hide a material execution change, increase memory pressure unnecessarily, or
depart from the prepared environment-specific policy.

## D-011 — 2026-08-01 — Define the E0 completion boundary by executed runtime evidence

**Decision:** E0 is complete only after the exact pinned assets are actually
consumed and all of the following pass: the full official Dallas attribution
produces a nonempty validated graph; the official Spanish intervention is
validated at baseline, no-op, half, and full strengths; loaded-runtime
preactivation, activation, raw-threshold, strict-gate, inactive-observability,
and absolute-intervention-value mapping checks pass; validation summaries and
checksums are recorded; and the Stage 0 plus new Stage 1A checks pass. Stage 1B
must not begin before this boundary is met.

**Reason:** Exact pins, access/download checks, environment preparation,
configuration, and a Colab handoff establish reproducibility prerequisites but
do not reproduce an attribution or intervention and do not verify semantics of
the loaded runtime. The current Stage 1A verdict is therefore blocked, not
complete.

**Alternatives considered:** Mark E0 complete when metadata is pinned, when
assets download, when a notebook is prepared, or after only one official run.
None supplies the combined runtime and semantic evidence required before
susceptibility work begins.

## D-012 — 2026-08-23 — Isolate the MPS/FP16 runtime and make sparse CPU metadata explicit

**Decision:** Implement Apple M2 Max/MPS/FP16 as a separate hardware-adapted
runtime and artifact class. Keep dense model, transcoder, intervention, and
autograd tensors on MPS. Because PyTorch 2.6.0 does not implement the pinned
upstream sparse-COO conversion on MPS, permit only a project-local, numerically
validated CPU boundary for sparse COO coordinates/values and the already
CPU-resident graph storage. Keep `PYTORCH_ENABLE_MPS_FALLBACK` disabled and do
not weaken the BF16 or T4 paths.

**Reason:** Silent unsupported-operator fallback would conceal device changes,
while treating MPS as a conditional branch of the CUDA path would blur runtime
identity, telemetry semantics, and scientific claims. The explicit adapter
preserves the same strict JumpReLU activations and reconstruction while making
the unavoidable metadata placement observable and testable.

**Alternatives considered:** Enable automatic MPS-to-CPU fallback; require
native MPS sparse tensors; silently use dense CPU activations; or generalize the
existing T4 validator. The first and third hide a material execution boundary,
the second makes the exact PyTorch 2.6 path infeasible despite a bounded
equivalent adapter, and the fourth risks weakening historical CUDA checks.

**Revisit when:** The selected PyTorch/runtime implements and passes the exact
sparse COO path on MPS, or numerical validation shows that the explicit adapter
does not preserve the pinned computation.

## D-013 — 2026-08-23 — Separate Hugging Face payload placement from authentication and harden Git provenance

**Decision:** Set `HF_HUB_CACHE` and `HF_XET_CACHE` beneath the checked
project-external cache when directing the two authorized pinned repositories;
preserve the caller's `HF_HOME` so the existing secure Hugging Face login
remains discoverable. Before execution,
before candidate construction, and immediately around publication, require
fully qualified protected refs, an exact symbolic branch, no replacement refs
or legacy grafts, default index flags, a case-insensitive literal T4 index
query, a clean source checkout, unchanged T4 hashes, and the same execution
commit.

**Reason:** Overriding `HF_HOME` moved the credential lookup location and caused
an authenticated gated-model request to fail before any scientific execution.
Separately, ambiguous refs, case-variant indexed paths, and Git index flags can
make shorthand or porcelain-only checks report a false clean provenance state
on macOS. Payload placement and authentication are independent concerns, and a
completed bundle must not overstate either asset access or source identity.

**Alternatives considered:** Copy or serialize the access token into the new
cache; accept shorthand refs; trust case-sensitive `ls-files`; or rely only on
`git status --porcelain`. These alternatives respectively duplicate sensitive
material or leave reproduced provenance bypasses open.
