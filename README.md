# Counterfactual Susceptibility

**Status:** No Counterfactual Susceptibility result exists and the reference
Stage 1A reproduction remains pending. The separate Stage 1A-S small-model
MPS/FP16 pilot stopped as `failed_runtime` at its finite-logit model gate. See
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
