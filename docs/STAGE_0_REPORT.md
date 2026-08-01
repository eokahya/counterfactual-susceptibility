# Stage 0 Report

**Date:** 2026-08-01  
**Status:** Stage 0 scientific and engineering acceptance criteria met; public
repository publication is recorded separately as an explicit user-authorized
override of the original review-state restriction.

## 1. Outcome

Stage 0 produced a typed, dependency-light Python scaffold; deterministic
backend-independent Counterfactual Susceptibility mathematics; a conservative
`circuit-tracer` adapter boundary; an exact-commit source audit; reproducibility
scripts and configuration; unit tests; and a manuscript consistency pass.

No language model or transcoder was loaded. No empirical gate crossing,
behavioral result, mediation result, or claim of mechanistic faithfulness is
reported. The empirical parts of Research Specification E0 are deliberately
deferred to Stage 1 because this stage prohibited weight downloads and real-model
execution.

Acceptance criteria 1–9 were met. The original criterion requiring an unstaged,
uncommitted, unpublished handoff was superseded by the user's later explicit
instruction to create and publish the public GitHub repository
`eokahya/counterfactual-susceptibility`. That publication changes only the Git
handoff state, not the scientific scope or validation result.

## 2. Verified workspace and environment

- Workspace: the opened repository root.
- Uploaded starter contents were normalized from the nested starter directory to
  the repository root; byte-identical duplicate root copies of the research spec
  and manuscript entry point were removed.
- Core sources verified: `docs/RESEARCH_SPEC.md` and `paper/main.tex`.
- Git initially had an unborn `main` branch and no existing tracked/user changes.
- macOS 26.5.2 (`arm64`), Apple M2 Max with 38-core Metal-capable GPU.
- CPython 3.14.6; 526 GiB free disk at initial inspection.
- PyTorch/model packages were intentionally absent from the validation virtual
  environment. No CUDA device was present; no model computation was attempted.

## 3. Working tree deliverables

- Package metadata: `pyproject.toml` for Python 3.11–3.14 with bounded, optional
  development tools and no model runtime dependency.
- Typed package: `src/cfsus/`, including scientific records, exceptions,
  backend protocol, conservative adapter, pairwise mathematics, sweep helpers,
  observed-crossing evaluation, configuration, and logging.
- Offline tests: `tests/test_pairwise.py`, `tests/test_types_and_sweeps.py`, and
  `tests/test_backend_skeleton.py`.
- Reproducibility: `scripts/doctor.py`, `scripts/verify_math.py`, and
  `configs/smoke_gemma2_2b.yaml`.
- Documentation: research specification, decision log, immutable experiment-log
  template, upstream audit, README, and this report.
- Manuscript: internally consistent `paper/` sources and corrected bibliography
  metadata, with no fabricated results.

Generated build files, virtual environments, caches, bytecode, deterministic JSON
artifacts, model formats, raw datasets, and the local download-collision task
prompt are ignored. The tracked candidate tree contains no weights, datasets,
secrets, private keys, or large files.

## 4. Checks

The lightweight development tools were installed only inside ignored `.venv/`.
No `circuit-tracer`, PyTorch, Transformers, model, or transcoder package/weight
was installed.

| Check | Exact command | Outcome |
| --- | --- | --- |
| Unit tests | `.venv/bin/python -m pytest -q` | **PASS:** 48 passed |
| Lint | `.venv/bin/python -m ruff check .` | **PASS:** all checks passed |
| Format | `.venv/bin/python -m ruff format --check .` | **PASS:** 26 files formatted |
| Static types | `.venv/bin/python -m mypy src scripts tests` | **PASS:** 19 source files |
| Environment report | `.venv/bin/python scripts/doctor.py` | **PASS:** offline schema v1 |
| Deterministic mathematics | `.venv/bin/python scripts/verify_math.py` | **PASS:** ignored JSON artifact written |
| Import smoke | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'import cfsus; print(cfsus.__version__)'` | **PASS:** `0.0.0` |
| Editable install | `.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .` | **PASS** |
| Package build | `.venv/bin/python -m build --no-isolation` | **PASS:** sdist and wheel |
| Manuscript structure/citations | local `awk`, `for`, and `rg` checks described below | **PASS** |
| LaTeX compilation | `latexmk`/`pdflatex` availability check | **SKIP:** no TeX compiler installed; none was installed |

The first validation pass correctly failed on source-path, lint, and formatting
integration issues. The package path, sequential-pair iteration, enum base,
logging marker, formatting, README Python range, and provisional license metadata
were corrected before the passing matrix above.

Manuscript checks verified that every `\\input` exists, every citation key is
defined, no bibliography entry is unused, no result-like claim pattern remains,
and no trailing whitespace remains.

## 5. Upstream findings

The complete source evidence and 95 permanent source links are in
`docs/UPSTREAM_API_AUDIT.md`.

- Official source: `https://github.com/decoderesearch/circuit-tracer`.
- Inspected commit: `8f1e2438df612464e229e44c4a00ff637bf9379b`,
  tag `v0.5.2`, inspected 2026-08-01; MIT license.
- Upstream declares Python `>=3.10` but its CI demonstrates only Python 3.11.
- JumpReLU is exactly `a = z * 1[z > tau]`; equality is inactive. `tau` is a
  raw parameter, and exposed `z` is the encoder preactivation after `b_enc`.
  Decoder bias is part of reconstruction, not `z` or the intervention value.
- Single-sequence feature caches index `[layer, token_position, feature_id]`.
  Inactive bias-inclusive preactivations are publicly obtainable with
  `get_activations(..., apply_activation_function=False)`, but the API
  materializes the dense all-layer/all-feature cache and offers no public
  feature-chunk or selected-target route.
- Attribution graphs include baseline-active targets only. A stored feature edge
  is `a_j J_ij` under upstream's frozen convention, not raw `J_ij`. There is no
  public raw inactive-target Jacobian, JVP, VJP, or virtual-weight API.
- `feature_intervention` accepts an absolute desired post-gate activation and
  adds the decoder delta to the underlying LM computation. The project mapping
  is `desired = (1 - alpha) * baseline_activation`.
- There is no distinct public replacement-model-only feature intervention.
  Default intervention freezing restores attention, while a full constrained
  layer range additionally freezes LayerNorm scales and MLP/feature outputs.
  A future nonlinear underlying-model sweep must explicitly use
  `freeze_attention=False, constrained_layers=None` and report it separately
  from an attribution-matched constrained sweep.
- Fixed-prompt interventions can return logits and a full preactivation cache,
  but there is no public selected-target-only capture.
- Likely bottlenecks are full dense inactive caches, full-layer encoder loads,
  active-only reverse passes, and the dense graph edge matrix. Upstream documents
  Gemma 2 2B on roughly 15 GB GPU memory, while some tests require over 32 GB.

## 6. Blockers and uncertainties

- Exact immutable Hugging Face revisions for the first Gemma 2 2B model and
  transcoder remain deliberately unresolved; upstream examples/aliases are
  unpinned and the README/code disagree on the Gemma PLT repository.
- Scalable near-threshold inactive-candidate access needs a local chunked encoder
  projection wrapper.
- Inactive-target `J_ij` needs a targeted backend-local VJP or a separately named
  finite-difference response, validated against represented active pairs.
- Selected target-state capture during sweeps needs a narrow local hook/projection.
- The Stage 0 adapter reports verified upstream routes but marks every local
  executable operation unsupported until these mappings are implemented and
  tested.
- This repository currently has no project license or finalized authorship/
  `CITATION.cff`; public visibility does not grant reuse rights.

## 7. Scientific and manuscript consistency

Primary definitions were not changed. Three necessary consistency clarifications
were added to `docs/RESEARCH_SPEC.md` and recorded in Decision D-008:

1. the regularized reciprocal form is exact only for
   `epsilon_prime = epsilon / q` when `q > 0`;
2. `alpha_hat_star = 1` reaches the threshold and is a gate-strictness boundary,
   not a definite crossing; and
3. the required deterministic numerical example was added explicitly.

The manuscript now states the suppression mapping, signed inhibitory convention,
local-linear caveats, epsilon/boundary conditions, and replacement-versus-
underlying distinction consistently. Two result-like future statements were
rewritten as plans. Bibliography author/key/title metadata was corrected from
primary sources. No numerical or empirical result was added.

## 8. Smallest Stage 1 proposal

After explicit approval for model-weight downloads:

1. select and record immutable model, tokenizer, and transcoder revisions;
2. reproduce one official attribution and one feature-intervention example;
3. implement TransformerLens-first, chunked layer/position preactivation access
   and compare it with upstream `encode_layer(..., False)` on small tensors;
4. implement one targeted inactive-feature VJP and validate active pairs against
   `Graph.adjacency_matrix / a_j`;
5. run one preregistered source-suppression grid for one active source and one
   inactive target, capturing target `z`, exact gate activity, and a declared
   logit metric; and
6. report nonlinear underlying-model and attribution-matched constrained sweeps
   separately.

This smoke experiment has not run.

## 9. Git and publication record

Before publication the repository was unborn, so `git diff --stat` was empty
because every candidate file was untracked. `git status --short` listed the
project tree as untracked and nothing staged. Build/test artifacts and
`CODEX_STAGE_0_PROMPT(1).md` were ignored.

Public repository: <https://github.com/eokahya/counterfactual-susceptibility>.
The initial publication is on `main`; its exact commit hash and final
clean-status confirmation are recorded in the completion handoff.

## 10. Safety confirmation

- No model/transcoder weights, datasets, paid jobs, secrets, or files over 100 MB
  were downloaded or created.
- The only upstream download was a small, filtered, depth-1 source clone in
  `/tmp` for audit; it is outside the project and contains source only.
- Lightweight test/build tools were installed only in ignored `.venv/`.
- No cloud compute, model execution, remote issue, pull request, or release was
  created.
- A public GitHub repository/commit/push is the sole remote mutation and is
  performed only because the user's latest message explicitly requested it,
  overriding the earlier local-only Git handoff restriction.
