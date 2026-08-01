# Codex Prompt — Stage 0: Repository Scaffold and Upstream API Audit

Use subagents in parallel where they materially improve speed or correctness. At minimum, delegate: (1) upstream `circuit-tracer` API/source inspection, (2) repository architecture and test design, and (3) reproducibility/documentation review. Integrate their findings yourself and resolve disagreements explicitly.

You are setting up a new research repository for the project **Counterfactual Susceptibility: Discovering Inactive and Inhibitory Circuits in Language Models**.

The scientific source of truth is `docs/RESEARCH_SPEC.md`. Read it in full before changing anything. Also read `docs/DECISIONS.md` and the starter manuscript under `paper/`.

## Goal of this stage

Create a clean, typed, testable repository scaffold and perform a source-level audit of the current open-source `decoderesearch/circuit-tracer` implementation. Do **not** yet implement or claim the full counterfactual-susceptibility method. The main deliverable is a reliable map from the scientific quantities in the spec to exact upstream APIs and tensor semantics.

## Safety and external-action constraints

- Begin by printing `pwd`, `git status --short`, current branch, Python version, OS, accelerator availability, and free disk space.
- Do not push, publish, open a pull request, create remote resources, upload data, or modify any remote repository.
- Do not commit unless explicitly instructed after the user reviews the diff.
- Do not download model weights, transcoder weights, or any file larger than 100 MB in this stage.
- Do not run paid cloud jobs.
- Do not edit an installed third-party package in place.
- Prefer an adapter layer in this repository; propose an upstream patch separately if required.
- Preserve all existing user files. If the directory is not the intended repository or has unrelated uncommitted work, stop modifications and report exactly what you found.

## Required work

### 1. Inspect and pin upstream

Inspect the current official repository and relevant source files for `decoderesearch/circuit-tracer`.

Record in `docs/UPSTREAM_API_AUDIT.md`:

- repository URL, exact commit hash, date inspected, and license;
- install method and supported Python range;
- exact classes/functions used for loading a model and transcoder;
- exact JumpReLU or thresholded activation formula;
- where encoder bias and threshold are stored;
- tensor shapes and indexing conventions for layer, token position, feature ID, and batch;
- how active feature activations are cached or returned;
- how all feature preactivations, including inactive features, can be obtained without materializing unsafe full tensors;
- intervention APIs for feature suppression, feature clamping, and residual patching;
- whether each intervention affects the replacement model, underlying model, or both;
- what nonlinearities, attention patterns, and normalization terms are frozen in local attribution;
- how virtual weights, direct effects, Jacobian-vector products, or equivalent source-to-target responses can be extracted;
- how output logits and target feature activations can be captured during an intervention sweep;
- known memory constraints and backends.

Cite file paths, symbols, and line ranges or commit permalinks wherever possible. Do not infer semantics from names alone; verify in source and tests.

### 2. Scaffold the package

Create a modern Python package under `src/cfsus/` with minimal modules and interfaces, without implementing model-specific mathematics prematurely:

```text
src/cfsus/
  __init__.py
  config.py
  types.py
  backends/
    __init__.py
    base.py
    circuit_tracer.py
  susceptibility/
    __init__.py
    pairwise.py
  interventions/
    __init__.py
    sweep.py
  evaluation/
    __init__.py
    crossings.py
  logging_utils.py
```

Define typed protocols/dataclasses for:

- `FeatureRef(layer, position, feature_id)`;
- baseline feature state containing preactivation, activation, and threshold;
- source suppression intervention with `alpha in [0, 1]`;
- predicted crossing with `q`, margin, susceptibility, and predicted critical alpha;
- observed sweep result;
- behavior metric metadata.

Implement only backend-independent pure functions whose semantics are already fixed by the research spec:

- activation margin validation;
- `q = -a_j * J_ij`;
- pairwise susceptibility;
- predicted critical alpha;
- robust classification of predicted crossing;

Use explicit handling for zero/negative margins, non-finite values, and numerical tolerances.

### 3. Synthetic unit tests

Add deterministic tests for the pure functions, including:

- inhibitory source that causes a predicted crossing;
- inhibitory source insufficient to cross;
- excitatory source suppression moving away from the gate;
- target already active, which must be rejected by the inactive-target API;
- zero margin and near-zero denominator;
- finite-value validation;
- monotonicity of predicted critical alpha with source strength;
- equivalence of `S > 1` and `alpha_hat_star < 1` away from tolerance boundaries.

Do not require GPU, model downloads, or internet for tests.

### 4. Reproducibility and configuration

Add:

- `pyproject.toml` with pinned or bounded development dependencies;
- lint/type/test configuration;
- `.gitignore` covering model caches, Hugging Face caches, notebook checkpoints, large result tensors, secrets, and local environments;
- a small YAML configuration schema for a future Gemma 2 2B smoke run, but do not execute it;
- `scripts/doctor.py` that reports environment metadata without downloading anything;
- `scripts/verify_math.py` that runs the synthetic formulas and emits a small JSON artifact.

### 5. Paper consistency check

Review the manuscript notation against `docs/RESEARCH_SPEC.md`. Fix only inconsistencies or compilation errors. Do not write results or imply that experiments have run.

### 6. Validation

Run locally available checks that require no large downloads:

- unit tests;
- type checking;
- linting/format check;
- `scripts/doctor.py`;
- `scripts/verify_math.py`;
- LaTeX compilation only if a compiler is already installed; otherwise report that it was not run.

## Required final report

Return:

1. concise working-tree summary;
2. files created or modified;
3. exact checks run and their outcomes;
4. upstream API findings, especially any blockers for inactive preactivations and underlying-model interventions;
5. assumptions that remain unverified;
6. proposed Stage 1 implementation plan;
7. `git diff --stat` and `git status --short`;
8. confirmation that no large download, remote operation, commit, or push occurred.

Do not hide partial failures. If the upstream API cannot expose a required quantity, identify the smallest adapter or upstream change needed, but do not implement a speculative invasive patch in this stage.
