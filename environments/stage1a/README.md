# Stage 1A environment

This directory describes the observed Stage 1A runtime for the pinned
official circuit-tracer reproduction. It is intentionally platform-specific:
the lock was observed on macOS arm64 with CPython 3.11.13. It is not a
universal Linux/CUDA lock.

## Observed runtime

| Component | Observed value |
| --- | --- |
| Operating system | macOS 26.5.2, arm64 |
| Python | CPython 3.11.13 |
| circuit-tracer | 0.5.2 from audited commit 8f1e2438df612464e229e44c4a00ff637bf9379b |
| PyTorch | 2.13.0 |
| TransformerLens | 3.2.1 |
| Transformers | 4.57.3 |
| NNsight | 0.6.1 |
| Local MPS | built, but unavailable; a one-element allocation failed |
| Local CUDA | unavailable |

The machine has a Metal-capable Apple M2 Max GPU, but the observed PyTorch
runtime reported MPS unavailable and rejected a one-element MPS allocation.
Consequently, local CPU use is limited to metadata, configuration, tests, and
small semantic checks. A full attribution or intervention reproduction must
not silently fall back to CPU.

The planned Colab route is CUDA with bfloat16 and disk offload. It is not
represented by the observed macOS lock and remains unobserved until it is
actually run. On Linux/CUDA, preflight labels the direct-pin file as planned
input and records a path-free exact inventory of the environment actually used.

## Files

- requirements.in lists the human-maintained direct dependency.
- constraints.txt pins runtime-sensitive packages used during resolution.
- requirements-lock-macos-arm64-py311.txt is the exact sanitized output of
  pip freeze --all from the environment that was inspected.
- requirements-colab-py311-cu124-planned.txt contains direct CUDA runtime pins
  for the prepared Colab route. It is a planned input, not an observed
  transitive lock.
- environment-schema.json declares the stable observation and lock provenance
  consumed by the environment verifier.
- scripts/stage1a/preflight.py emits a strict environment_manifest artifact
  envelope with provenance, payload, warnings, and deviations.

The lock contains the immutable circuit-tracer Git commit rather than a
mutable branch or a bare package version. It contains exact runtime versions
but no wheel hashes, so it is an observed environment lock rather than a
cross-platform artifact-integrity lock. Installed circuit-tracer metadata
must additionally show both requested_revision and commit_id equal to the
audited commit.

## Recreate

Use a clean Python 3.11 interpreter. On hosts where Python 3.11 is not on
PATH, resolve it into a session-local variable; do not commit a private
absolute interpreter path.

    python3.11 -m venv --copies .venv-stage1a
    .venv-stage1a/bin/python -m pip install \
      -r environments/stage1a/requirements-lock-macos-arm64-py311.txt

For a fresh dependency resolution rather than an exact replay:

    .venv-stage1a/bin/python -m pip install \
      -c environments/stage1a/constraints.txt \
      -r environments/stage1a/requirements.in

Refresh the observed lock only from a verified Python 3.11 macOS-arm64
environment. Inspect the generated text before committing it and reject
editable installs, file URLs, local absolute paths, credentials, or cache
locations.

Run the local-only preflight with:

    .venv-stage1a/bin/python scripts/stage1a/preflight.py

The preflight makes no network requests, does not inspect credential files,
and emits no executable, repository, cache, username, hostname, or home path.
Free disk is necessarily time-varying; the JSON shape and units are stable.
