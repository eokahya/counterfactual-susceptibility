# Counterfactual Susceptibility

**Status:** Stage 1B measurement primitives remain completed on the pinned
local Gemma 3 270M + 18-PLT NNsight runtime on Apple MPS/BF16. After the
historical Stage 1C-v1/v2/v3 failures, Stage 1C-v4 completed the frozen Norway
development pilot with a validated `mixed` result. Stage 1D then completed an
eight-prompt held-out gate benchmark. Susceptibility beat margin-only and
random-positive ranking, but trailed influence-only; critical-alpha evidence
and monotonicity missed the frozen acceptance criteria. The project decision
is `retain_crossing_ranker_but_redesign_calibration`, not a transition to
behavioral or mediation experiments. The reference Stage 1A reproduction
remains pending, and paper Results readiness remains false. See
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for the current experiment
classes and readiness boundary.

This project studies a blind spot in sparse attribution graphs: a feature can be inactive on the baseline prompt yet become causally important after a small intervention removes an inhibitory influence. Active-only graphs cannot display such a feature before it crosses its activation threshold.

The project develops and tests **counterfactual susceptibility** scores that
rank inactive features by:

1. how close they are to activation under a specified intervention family;
2. whether an active upstream feature suppresses them;
3. how much their counterfactual activation is predicted to affect a target behavior.

For an inactive target feature `i` and active source feature `j`, the core
Stage 0 quantities are the activation margin `m_i = tau_i - z_i`, the predicted
preactivation increase under complete source suppression
`q_(j->i) = -a_j J_ij`, and the pairwise susceptibility
`S_(j->i) = q_(j->i) / (m_i + epsilon)`. Suppression strength follows
`a_j -> (1 - alpha) a_j` for `alpha` in `[0, 1]`.

The primary empirical question is not whether this local linear score looks
plausible. It is whether the score predicts actual gate crossings and output
changes under interventions, with replacement-model and underlying-model
evidence reported separately.

## Stage 1A status

Stage 1A-S-BF16 completed the narrowly scoped local Gemma 3 270M plus 18-PLT
NNsight runtime validation on Apple MPS/BF16. Finite model execution, loaded
threshold semantics, nonempty attribution, deterministic feature selection,
no-op consistency, and absolute feature suppression passed the independent
small-artifact validator. This does not establish a Gemma 2/reference-CLT
reproduction, PLT/CLT or MPS/CUDA equivalence, a susceptibility prediction,
gate crossing, mediation, behavioral importance, or paper result. See
[`docs/STAGE_1A_SMALL_MODEL_MPS_BF16_REPORT.md`](docs/STAGE_1A_SMALL_MODEL_MPS_BF16_REPORT.md).

The reference Stage 1A-R track remains pending.

## Stage 1B status

Stage 1B converted the accepted local Stage 1A-S-BF16 runtime into a reusable
measurement backend and validated two bounded primitives. The inactive-feature
scanner matched its ephemeral dense oracle exactly across chunk sizes 257,
1024, and 4096. An independent targeted reverse-mode path computed
`J_ij = partial z_i / partial a_j` without accepting graph or edge input; on
the frozen 64-pair canonical set, `a_j * J_ij` matched raw attribution edges
with Spearman 0.9999657, sign agreement 1.0, median symmetric normalized error
0.001887, and p95 error 0.004512. The compact artifact bundle passed the
standalone hostile-input validator. See
[`docs/STAGE_1B_MEASUREMENT_PRIMITIVES_REPORT.md`](docs/STAGE_1B_MEASUREMENT_PRIMITIVES_REPORT.md).

This establishes engineering readiness for a separately specified first
prediction stage only. No inactive-target susceptibility score, source
suppression, gate crossing, behavioral importance, mediation result, or paper
result was produced.

## Stage 1C status

Stage 1C completed and pushed its baseline-only prediction freeze before any
selected inactive-target intervention. The frozen manifest contains 101
eligible inactive targets, 1,908 eligible active sources, 30,283 eligible
pairs, and disjoint selections of 12 primary, 8 near-boundary, and 8
directional-control pairs. Its prediction-only guards and independent
recomputation passed.

The one permitted canonical intervention process subsequently reported 228
source-suppression API calls, but the frozen worker cleared its shared sweep
list before serialization. Both persisted sweep collections therefore had
zero rows. The frozen assembler and standalone validator failed closed, no
canonical result bundle was committed, and the scientific outcome is
`inconclusive_runtime`. See
[`docs/STAGE_1C_FIRST_PROSPECTIVE_PREDICTION_REPORT.md`](docs/STAGE_1C_FIRST_PROSPECTIVE_PREDICTION_REPORT.md).

Stage 1C-v2 repaired detached serialization but stopped before prediction
freeze when its frozen endpoint-overlap guard fired. Stage 1C-v3 corrected the
exclusion policy and froze a valid Norway prediction, then failed before its
first suppression call because the adapter lacked baseline remeasurement.
Those historical outcomes remain unchanged.

Stage 1C-v4 made the minimal production-path repair and executed the same
byte-identical Norway prediction once. Its complete 248-point bundle passed
independent reconstruction and produced an accepted `mixed` development-pilot
result. See
[`docs/STAGE_1C_V4_PROTOCOL_PRESERVING_EXECUTION_REPORT.md`](docs/STAGE_1C_V4_PROTOCOL_PRESERVING_EXECUTION_REPORT.md).

## Stage 1D status

Stage 1D reused the validated v4 production path on eight fresh prompts. The
single canonical attempt completed 438 evaluation calls and 438 durable,
serialized points over 169 unique prompt/pair identities. Full-ablation
precision@4 was 0.84375 for susceptibility, 0.50 for margin-only, 1.00 for
influence-only, and 0.3125 for deterministic random-positive selection.
Critical-alpha Spearman was 0.6941 on only 12 qualifying pairs, and 10/32
detailed positive pairs were nonmonotonic. The standalone validator therefore
accepted the experiment and its frozen decision
`retain_crossing_ranker_but_redesign_calibration`. See
[`docs/STAGE_1D_MULTIPROMPT_GATE_BENCHMARK_REPORT.md`](docs/STAGE_1D_MULTIPROMPT_GATE_BENCHMARK_REPORT.md).

This is gate-prediction evidence on the local PLT/MPS/BF16 runtime. It is not
behavioral importance, mediation, an official BF16/reference-CLT reproduction,
or a paper-ready result.

Stage 1A selected the immutable model snapshot
`google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf` and the direct
26-layer-weight download subset from the transcoder repository
`mwhanna/gemma-scope-transcoders@bd5773156dea09893636c801df1237d0410307d2`.
The pinned loader must consume the model from its exact-SHA local snapshot
because the upstream replacement-model constructor has no model-revision
argument.

The intended local path is the TransformerLens backend with an explicit MPS
probe, `bfloat16`, and CPU offload. On this machine, PyTorch reports MPS as
built but unavailable and an actual MPS allocation fails, so the full run must
not silently fall back to CPU. The Colab fallback handoff requires an explicit
final project commit, probes CUDA/BF16 and VRAM, and uses the upstream official
`disk` attribution offload policy.

E0 remains blocked because the configured Hugging Face credential received
HTTP 403 for the exact Gemma asset and no model/transcoder runtime was loaded.
Prepared pins, access checks, configuration, environment tooling, or a Colab
handoff do not constitute an empirical reproduction. No susceptibility
prediction, intervention-induced gate crossing, behavioral importance,
mediation, or underlying-model circuit has been established.

## Stage 0 scope

- typed, backend-independent representations for feature states, interventions,
  predictions, observations, behavior metadata, and backend capabilities;
- deterministic implementations and tests for the fixed pairwise mathematics;
- a conservative `circuit-tracer` adapter boundary backed by a source-level API
  audit;
- offline environment and deterministic-math verification scripts; and
- research, decision, experiment, audit, and Stage 0 reporting documents.

Stage 0 does **not** download model or transcoder weights, execute a language
model, validate a gate crossing on a real model, or establish behavioral or
mediation results.

## Local development

Python 3.11 through 3.14 is supported by the Stage 0 package metadata. From the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run the offline checks with:

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src scripts tests
python scripts/doctor.py
python scripts/verify_math.py
python -c "import cfsus"
```

The exact commands actually run and their outcomes belong in
[`docs/STAGE_0_REPORT.md`](docs/STAGE_0_REPORT.md). The example Gemma 2 2B
configuration is a future smoke-run configuration only; its presence does not
mean that the run has occurred.

## Repository map

```text
configs/                  Experiment configurations
src/cfsus/                Library code
tests/                     Unit and integration tests
notebooks/                 Exploratory notebooks only
scripts/                   Reproducible command-line entry points
docs/RESEARCH_SPEC.md      Single source of truth for the research design
docs/DECISIONS.md          Versioned scientific and engineering decisions
docs/EXPERIMENT_LOG.md     Immutable summaries of completed runs
docs/UPSTREAM_API_AUDIT.md Pinned source-level upstream semantics
docs/STAGE_0_REPORT.md     Stage 0 work, checks, and unresolved questions
paper/                     Overleaf-compatible manuscript
results/                   Small derived tables/figures; no model weights
```

## Research discipline

- Do not commit model weights, private data, caches, or large raw activations.
- Every reported figure must be generated by a tracked script from a tracked configuration.
- Exploratory notebooks may suggest hypotheses, but manuscript results must be reproduced by CLI scripts.
- Negative results and deviations from the preregistered design are recorded in `docs/EXPERIMENT_LOG.md`.
- The manuscript must not claim mechanistic faithfulness beyond what is supported by interventions in the underlying model.

## Initial model path

The first supported backend is expected to use the open-source
`decoderesearch/circuit-tracer` tooling with a small open-weight model and
pretrained transcoders. The exact upstream commit, model, dictionary, activation
convention, and intervention API must be recorded before the first empirical
run. See [`docs/UPSTREAM_API_AUDIT.md`](docs/UPSTREAM_API_AUDIT.md) for what is
verified and what remains blocked.

## Citation

A `CITATION.cff` file will be added once authorship, title, and a public release are fixed.
