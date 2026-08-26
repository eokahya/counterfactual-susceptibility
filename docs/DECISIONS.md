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
remains discoverable. Remove inherited Xet redirect variables, reject unsafe
entries throughout the existing cache tree, and recheck that tree between
attempts and before publication. Before execution,
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

## D-014 — 2026-08-23 — Stop the MPS reproduction after runtime-load and memory gates fail

**Decision:** Preserve the real batch-labelled-256 worker as
`failed_runtime` at `runtime_loading`; do not relabel it as MPS OOM and do not
start the 128 or 64 workers. A bounded postmortem reproduced a compatible
deterministic RuntimeError at the next known sparse-metadata check: FP16 MPS
does not support the requested `index_put_` accumulation. The original leaf
diagnostic was not retained, so exact identity is unconfirmed. Because the
coordinates originate from `nonzero` and are unique, use replacement rather
than accumulation and retain a live MPS zero-error regression. This
compatibility fix does not authorize a large rerun. Treat the exact current
loading plan as resource-blocked on the 32 GiB host because its observed
`40,032,174,080`-byte MPS-driver peak and `34,567,031,357`-byte host-wide
sampled swap peak falsify the 24,051,816,857-byte conservative budget. Persist
the observed plan fingerprint in the pre-run memory gate and fail closed before
another identical load.

**Reason:** Batch size is not consumed until attribution, so 128 or 64 cannot
reduce model/transcoder/TransformerLens construction memory. The task permits
retry only for confirmed MPS OOM; this attempt was a generic runtime error,
OOM was unconfirmed, and attribution never started. Although the operator
failure has an equivalent local fix, MPS current and driver allocations each
exceeded physical/safety budgets and system swap rose far above the 4 GiB
threshold. Current recovery to normal memory and thermal pressure does not
erase the measured peak or make the original static estimate conservative.

**Alternatives considered:** Reclassify the error as OOM and try 128/64;
silently switch to CPU or CUDA; change dtype, model, revision, offload, prompt,
or scientific parameters; suppress the TransformerLens warning; or publish
empty semantics/attribution/intervention files as a completed artifact. These
options respectively violate the narrow retry policy, conceal a runtime
change, weaken the experiment, hide an unresolved correctness uncertainty, or
fabricate evidence.

**Revisit when:** A materially new loading plan has a distinct reviewed
identity, a conservative static budget below the host reserve using this
attempt as empirical evidence, bounded MPS correctness validation, and an
isolated worker that cannot silently continue through unsafe memory. The new
plan must preserve the exact scientific inputs and produce a fully validated
real artifact before either Stage 1B readiness flag can become true.

## D-015 — 2026-08-24 — Separate Stage 1A-S development runtime validation from Stage 1A-R

**Decision:** Define Stage 1A-S as a separate local small-model runtime pilot.
Gemma 3 270M with a GemmaScope-2 PLT is a development path; Gemma 2 2B with the
reference CLT remains the pending Stage 1A-R validation path. PLT and CLT
evidence are not interchangeable.

**Reason:** The smaller model can reduce local engineering risk while testing
loaded threshold semantics, attribution, and intervention capabilities, but it
cannot answer whether the reference Gemma 2/CLT experiment reproduces.

**Alternatives considered:** Relabel the small-model pilot as the reference
reproduction or treat PLT and CLT as equivalent. Both would overstate the
scientific evidence and erase a material representation change.

## D-016 — 2026-08-24 — Give NNsight/MPS/FP16 its own fail-closed runtime identity

**Decision:** Treat `backend=nnsight`, `device=mps`, and `dtype=float16` as an
explicit Stage 1A-S runtime class. Disable automatic MPS fallback and reject
unexpected CPU scientific tensors. Any permitted CPU metadata boundary must be
enumerated and independently tested.

**Reason:** NNsight and Apple MPS have distinct tracing, device-placement,
operator, gradient, and memory behavior. Reusing TransformerLens/CUDA labels or
allowing hidden fallback would make runtime evidence ambiguous.

**Alternatives considered:** Reuse the previous Gemma 2 MPS validator, enable
`PYTORCH_ENABLE_MPS_FALLBACK`, or silently switch to CPU/CUDA. Each would hide
material backend or device differences.

## D-017 — 2026-08-24 — Keep the small-model asset identity provisional until immutable metadata audit

**Decision:** Provisionally evaluate `google/gemma-3-270m` pretrained/base and
`mwhanna/gemma-scope-2-270m-pt` subfolder
`transcoder_all/width_16k_l0_small`. Do not download payloads or accept the
identity until official upstream source and Hugging Face metadata establish
exact immutable upstream/model/transcoder revisions and an exact runtime-file
allowlist. Do not substitute another model, width, PT/IT variant, backend,
device, or dtype inside this goal.

**Reason:** Mutable names, a guessed subfolder, or an alternative selected after
a failure would make provenance and negative results irreproducible.

**Alternatives considered:** Consume repository `main`, download the whole
transcoder repository, or try a different small model after a blocker. These
options respectively leave code/assets mutable, exceed the authorized payload,
or turn a preregistered pilot into outcome-dependent model selection.

## D-018 — 2026-08-24 — Freeze the accepted Stage 1A-S protocol before outputs

**Decision:** Attribution/intervention smoke remains exploratory and ignored.
After smoke passes, commit the complete accepted config, deterministic feature
selection, runner/worker, validator, tests, schema, and dependency pins before
executing the accepted pilot. No accepted-run scientific parameter may be
changed after inspecting its outputs.

**Reason:** A separate pre-run commit prevents hand-picked features, tolerances,
or settings from converting engineering exploration into a biased accepted
result.

**Alternatives considered:** Select a semantically interesting feature by hand
or tune accepted settings after viewing effects. Both would invalidate the
pilot's deterministic and preregistered boundary.

## D-019 — 2026-08-24 — Pin the official Gemma 3 development assets and narrow PLT subset

**Decision:** Pin official `circuit-tracer` v0.5.2 at
`8f1e2438df612464e229e44c4a00ff637bf9379b`,
`google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`, and
`mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`.
For the transcoder, allow only
`transcoder_all/width_16k_l0_small/config.yaml` and the 18 files
`layer_0.safetensors` through `layer_17.safetensors`.

**Reason:** The tagged release contains explicit Gemma 3 NNsight attribution
support. Official Hugging Face metadata verifies the model architecture,
18-layer count, 16K canonical PLT tensor layout, exact byte sizes, and immutable
asset SHAs. The selected runtime subset is sufficient and totals less than the
6 GiB transcoder cap.

**Alternatives considered:** Mutable `main`, a native lower-case GemmaScope-2
layout from a different repository, all widths, feature visualizations, or an
alternate small model. Each changes provenance, loading semantics, authorized
payload, or the preregistered identity.

## D-020 — 2026-08-24 — Stop Stage 1A-S on reproducible Gemma 3 FP16 residual overflow

**Decision:** Classify Stage 1A-S as `failed_runtime` at the model-only finite
logit gate. Do not load the PLT, construct the replacement runtime, run
attribution/intervention, freeze an accepted protocol, or retry. Do not change
to BF16, FP32, mixed-precision residuals, CPU, CUDA, another model, or a new
prompt under the same experiment identity.

**Reason:** On exact MPS/FP16 execution, decoder layer 7 adds two finite values
`55520` and `13408`; their sum `68928` exceeds FP16 maximum `65504` and becomes
positive infinity. Later layers produce all-NaN logits. BOS-only, one-word,
and the planned prompt first fail at the same hidden-state index, so the issue
is not prompt-specific. Memory, swap, device, fallback, and thermal gates all
passed. The specification explicitly prohibits retries for non-finite values.

**Alternatives considered:** Treat the run as an OOM, lower attribution batch
size, suppress finite checks, choose a prompt whose observed behavior looks
better, or retain some residual operations in FP32. None addresses the observed
failure without violating the retry rule or changing the experiment's declared
dtype/scientific runtime identity.

**Revisit when:** A separately specified and reviewed experiment explicitly
permits a different numerical execution policy. It must not be presented as
this all-FP16 pilot or as the pending official BF16 reproduction.

## D-021 — 2026-08-24 — Define MPS/BF16 recovery as a new experiment class

**Decision:** Create Stage 1A-S-BF16 from exact protected commit
`3baf39a5ac81e172d11d22a6de332dee80a21079` in a separate worktree and branch.
Preserve the Stage 1A-S all-FP16 `failed_runtime` result unchanged. Native MPS
with `torch.bfloat16` is a new scientific execution identity, not a retry,
correction, or reclassification of the FP16 run.

**Reason:** The BF16-trained checkpoint exceeded FP16 dynamic range at a
reproducible residual addition. BF16 directly tests whether the checkpoint's
native exponent range recovers finite execution without erasing the valid
negative result or silently changing its declared runtime.

**Alternatives considered:** Rewrite the FP16 report, call BF16 a retry under
the same class, clamp or rescale residuals, or use mixed precision. Each would
destroy provenance or change the accepted computation without a separate
reviewed identity.

## D-022 — 2026-08-24 — Permit only a separate CPU/FP32 diagnostic reference

**Decision:** Accepted model, PLT, NNsight replacement, attribution, gradient,
and intervention computation must remain native MPS/BF16 with fallback
disabled. After a successful MPS/BF16 model-only worker exits, one short
CPU/FP32 forward may run separately as a labeled diagnostic reference under
thresholds frozen before model comparison. It is not fallback and cannot
rescue a failed MPS/BF16 gate. CPU remains otherwise limited to explicit
metadata, I/O, tokenization, checksums, telemetry, and the audited sparse-COO
metadata boundary.

**Reason:** A separately timed FP32 reference can detect catastrophic BF16
divergence while preserving the accepted hardware/dtype identity and avoiding
concurrent unified-memory pressure. Hidden CPU or FP32 model compute would make
the empirical class ambiguous.

**Alternatives considered:** Silent CPU fallback, FP32 upcast inside the
accepted model, autocast/mixed precision, or concurrent reference execution.
All either substitute the experiment or compromise resource/provenance claims.

## D-023 — 2026-08-24 — Keep development and reference tracks scientifically separate

**Decision:** Gemma 3 270M plus per-layer PLTs remains the Stage 1A-S
development runtime. Gemma 2 2B plus the reference CLT remains the pending
Stage 1A-R validation track. PLT and CLT are not equivalent; NNsight/MPS does
not establish CUDA equivalence. Even full Stage 1A-S-BF16 success permits only
engineering readiness and leaves empirical readiness, official BF16
reproduction, reference CLT reproduction, Counterfactual Susceptibility, and
paper Results pending.

**Reason:** Small-model runtime capability is useful engineering evidence but
does not answer the reference-model or proposed-algorithm questions.

**Alternatives considered:** Promote a passing small-model pilot to official
reproduction, treat PLTs and CLTs as interchangeable, or begin susceptibility
and mediation work in this Goal. Each overstates the evidence or violates the
predeclared Stage 1A boundary.

## D-024 — 2026-08-24 — Use subclassed BF16/MPS adapters, never runtime monkeypatches

**Decision:** Implement separate `MPSBF16TranscoderSet`,
`MPSBF16AttributionContext`, and `MPSBF16ReplacementModel` subclasses plus a
source-faithful local attribution function. Dense feature values,
encoder/decoder vectors, residuals, gradients, and intervention tensors remain
MPS/BF16. Only bit-exact BF16 COO and graph-ranking metadata cross to CPU. Do
not assign replacement methods onto pinned upstream instances, classes, or
modules.

**Reason:** PyTorch 2.6.0 does not implement native MPS `to_sparse()`, while the
prior FP16 workaround monkeypatched three upstream call sites and converted
sparse values to FP32. Explicit subclasses make the deviation inspectable,
typed, counted, and isolated while preserving pinned direct-effect math.

**Alternatives considered:** Enable MPS fallback, edit site-packages, reuse the
FP16 monkeypatch, or keep full scientific tensors on CPU. Each hides a runtime
change, weakens provenance, or violates the accepted device/dtype identity.

## D-025 — 2026-08-24 — Compare baseline and no-op under identical freeze semantics

**Decision:** Establish the intervention baseline, baseline repeat, alpha-zero
no-op, half suppression, and full ablation with the same baseline-active
feature tuple, `freeze_attention=true`, and no constrained layers. Also compare
the frozen no-op baseline to the raw replacement forward under the frozen
normalized-L2 tolerance.

**Reason:** Upstream does not build freeze hooks for an empty intervention
list. The prior FP16 worker therefore compared an unfrozen empty baseline with
frozen nonempty interventions. Sending the exact baseline activation as an
absolute no-op exercises the same upstream path in every condition and leaves
the model output unchanged when alpha is zero.

**Alternatives considered:** Keep the mismatched prior control, set
`freeze_attention=false` only for the baseline, or change the accepted
constraint convention after effects are observed. Those choices confound the
no-op control or make the protocol outcome-dependent.

## D-026 — 2026-08-24 — Validate each loaded PLT against its own threshold vector

**Decision:** Stack the 18 real loaded threshold vectors and independently
recompute strict JumpReLU with layerwise broadcasting. Preserve the failed
engineering attempt that incorrectly broadcast one selected layer's threshold
vector to every layer, and require a synthetic opposing-threshold regression
test before a fresh smoke run.

**Reason:** Each PLT has distinct learned thresholds. A selected-layer vector
cannot validate the full `[layer,position,feature]` cache. This was a validator
mapping bug, not a runtime or scientific failure; the correction occurred
before the execution commit and accepted pilot.

**Alternatives considered:** Loosen the gate tolerance, validate only the
selected scalar, or ignore discrepancies in other layers. Each would conceal
incorrect loaded-semantics coverage.

## D-027 — 2026-08-24 — Invalidate an accepted runtime pass missing one required compact control

**Decision:** Preserve the first batch-64 accepted runtime pass under ignored
generated provenance, but do not publish it as the canonical accepted run. It
omitted the binding specification's explicit maximum baseline/no-op logit
difference field even though normalized-L2 controls were zero. Add the field,
freeze an exact zero maximum-absolute tolerance for raw baseline, baseline
repeat, and alpha-zero no-op, regression-test the validator, commit and push a
new pre-run execution SHA, and run the complete accepted pilot again. Retain
both accepted attempts with explicit dispositions in final provenance.

**Reason:** A passing runtime is not sufficient when one mandatory compact
evidence field is absent. The specification requires every genuine validator
correction before a fresh accepted execution from a new pre-run commit.

**Alternatives considered:** Derive the value later from top-logit summaries,
edit the accepted JSON by hand, accept normalized L2 as a substitute, or omit
the first attempt. Each would weaken the frozen artifact contract or erase
attempt provenance.

## D-028 — 2026-08-24 — Separate Stage 1B measurement from prediction

**Decision:** Stage 1B implements only an exact loaded-state near-threshold
inactive scanner and an independent active-source/active-target local response
measurement. It does not score inactive targets, search for gate crossings,
run suppression sweeps, measure behavior or mediation, or change paper Results.

**Reason:** These two primitives must be validated independently before the
first prospective Counterfactual Susceptibility prediction can be trusted.

**Alternatives considered:** Combine scanning, prediction, and intervention in
one run, or treat Stage 1A attribution as sufficient. Both would make failures
circular and overstate the evidence.

## D-029 — 2026-08-24 — Preserve exact loaded JumpReLU state in the scanner

**Decision:** Compute chunk projections on native MPS/BF16 from loaded encoder,
bias, and threshold tensors; classify with the loaded strict gate
`a=z*1[z>tau]`, with equality inactive. Require exact chunk/dense-oracle
candidate identity and order at feature chunk sizes 257, 1024, and 4096.

**Reason:** CPU/FP32 recomputation, approximate thresholds, or full retained
dense caches can alter marginal BF16 feature ordering or violate memory and
artifact boundaries.

**Alternatives considered:** Use the public full dense activation cache,
recompute in FP32 on CPU, or accept approximate recall. Each changes the
measurement or weakens the exact-oracle claim.

## D-030 — 2026-08-24 — Compute targeted J independently of graph edges

**Decision:** Compute `J_ij=partial z_i/partial a_j` with a bounded reverse-mode
VJP under the frozen NNsight attribution convention, using target encoder and
unscaled source decoder directions. The targeted implementation accepts no
graph or edge input. Only the validator later compares `a_j*J_ij` with the raw
target-row/source-column adjacency edge.

**Reason:** Dividing a graph edge by activation would make the validation
circular. Target preactivation must exclude the target gate derivative.

**Alternatives considered:** Divide `E` by `a_j`, use a UI-normalized edge, or
differentiate an underlying-model intervention sweep. The first two are
circular; the last uses a different nonlinear convention.

## D-031 — 2026-08-24 — Freeze calibration and canonical evidence separately

**Decision:** Calibration may debug the implementation and freeze the prompt,
chunk sizes, top-K, hash seed, exact disjoint pair IDs, edge floor, tolerances,
runner, worker, validator, and artifact schema. Canonical outputs are produced
only from a subsequent clean pre-run commit and cannot change those choices.

**Reason:** Outcome-dependent definitions would invalidate the prospective
active-pair validation.

**Alternatives considered:** Select pairs manually, tune tolerances after the
canonical run, or reuse calibration rows as final evidence. Each leaks outcome
information into acceptance.

## D-032 — 2026-08-24 — Treat compact artifacts as hostile input

**Decision:** The standalone validator rejects duplicate JSON keys, unknown or
dense tensor-shaped fields, nonfinite values, calibration leakage, altered
pair IDs/order, missing checksum coverage, local paths, secrets, forbidden
extensions, links, raw graph/adjacency/gradient payloads, and any bundle at or
above 5 MiB. It independently recomputes scanner and local-response metrics.

**Reason:** Small JSON does not by itself guarantee scientific, provenance, or
secret safety. Claimed safety booleans must be cross-checked structurally.

**Alternatives considered:** Trust runner summaries or reuse only the Stage 1A
marker scan. Neither covers Stage 1B pair leakage, dense arrays, or adversarial
serialization.

## D-033 — 2026-08-24 — Freeze the calibrated active-pair protocol before canonical execution

**Decision:** Freeze the exact 16 calibration IDs, disjoint 64 canonical IDs,
canonical endpoint-manifest digest, edge floor 0.015625, unchanged hard
tolerances, prompt, scanner chunk sizes, top-K values, runner, worker,
validator, and artifact schema in the pre-run commit. Canonical execution may
read only this tracked frozen config and the immutable offline assets; it may
not read the calibration artifact.

**Reason:** The real MPS/BF16 calibration passed exact scanner/dense equality
and active-pair validation with Spearman 1.0, sign agreement 1.0, median SNE
0.0022785724126932663, and p95 SNE 0.004240743761213505. These results establish
that the implementation is suitable to freeze; they do not justify changing
the preregistered acceptance thresholds or making an empirical susceptibility
claim.

**Alternatives considered:** Tune the edge floor or tolerances after viewing
canonical results, reuse calibration pairs as canonical evidence, or persist
the calibration graph. Each would introduce outcome leakage, circularity, or
forbidden large evidence.

## D-034 — 2026-08-24 — Accept only the frozen one-attempt canonical evidence

**Decision:** Accept the canonical run from clean pre-run commit
`de49bc0ee1d4ee1b2a0c15703b41e76781467ede`. It used exactly one fresh process,
read no calibration artifact, made no scientific retry, passed every frozen
scanner and targeted-response threshold, and produced only the allowlisted
compact artifact bundle. The terminal class is exclusively
`completed_stage1b_measurement_primitives`.

**Reason:** The scanner matched the dense oracle exactly at all frozen chunk
sizes, while the independent 64-pair targeted path passed prospective raw-edge
validation without graph/edge input. Standalone and independent spot checks
reproduced the evidence, so all binding phase gates are satisfied without
post-outcome changes.

**Alternatives considered:** Expand the claim to susceptibility, gate crossing,
behavior, mediation, official BF16 reproduction, reference CLT reproduction,
or paper readiness. None of those outcomes was measured in this Goal.

# Stage 1C prospective prediction freeze

**Decision:** Stage 1C starts exclusively from
`efbf70a7e462e640a0e1819a93f3b92727bbd193` and separates baseline-only
prediction from all inactive-target intervention execution. The prediction
manifest, pair groups, schedule, tolerances, runtime code, validator, and
artifact schema must be committed and pushed before the single canonical
intervention attempt.

**Reason:** A prospective test is only interpretable if outcome data cannot
influence candidate selection, prediction definitions, or acceptance rules.

**Decision:** Active sources are every exact-loaded, positive post-gate PLT
feature that is strictly layer-upstream and causally positioned relative to at
least one bounded inactive target. Targeted responses are computed by a
graph-independent reverse-mode path in bounded target batches. Raw attribution
edges, adjacency, and displayed edge normalizations are forbidden prediction
inputs.

**Decision:** The canonical edit is one absolute source-feature value with
`freeze_attention=true` and `constrained_layers=null`. For each requested
suppression, the desired high-precision value is
`(1-alpha) * baseline_source_activation`; the actual MPS/BF16 value passed to
the public intervention API and its realized suppression are recorded
separately. Duplicate applied BF16 values are collapsed before execution.

**Decision:** Target crossing uses the immutable loaded JumpReLU rule
`z > threshold`; equality is inactive. Outcomes are limited to `supported`,
`mixed`, `not_supported`, `no_eligible_pairs`, and `inconclusive_runtime` under
the frozen classifier. A valid negative or mixed scientific result is not a
runtime failure.
