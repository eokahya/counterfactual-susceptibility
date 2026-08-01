# Stage 1A Report

**Date:** 2026-08-01
**Verdict:** **BLOCKED**
**Base commit:** `7aacf30d888f96a29a1cfc82d035fca489ed0c17`
**Working branch:** `stage-1a-reproduction`

Stage 1A could not execute the official Gemma attribution or intervention
examples because the exact pinned model configuration remained inaccessible to
the available Hugging Face credentials. The local PyTorch runtime also reported
that MPS was built but unavailable and failed the allocation probe. The pinned
environment, public asset metadata, and a Colab handoff were prepared, but no
model or transcoder weights were fully downloaded or loaded. Consequently, no
empirical experiment or loaded-runtime semantic verification is reported.

## 1. Git and scope

Preflight verified that the current branch was created from the clean public
Stage 0 baseline. At that point local `HEAD`, `main`, and `origin/main` all
resolved to `7aacf30d888f96a29a1cfc82d035fca489ed0c17`, and `origin` was
`git@github.com:eokahya/counterfactual-susceptibility.git`.

All Stage 1A work is confined to `stage-1a-reproduction`. `main` was not reset,
rewritten, merged, or pushed. The final Stage 1A implementation commit and push
status are recorded in the completion handoff because a Git commit cannot
truthfully name itself inside its own contents.

This stage performed environment construction, immutable asset resolution,
metadata inspection, offline implementation/testing, and fallback preparation.
It did not implement Stage 1B candidate scanning, selected-feature projection,
inactive-target Jacobians, JVP/VJP operations, susceptibility ranking,
suppression sweeps, critical-strength estimation, behavioral tests, mediation,
or graph augmentation.

## 2. Pinned runtime and hardware

| Item | Observed or resolved value | Execution status |
| --- | --- | --- |
| Host | macOS 26.5.2 build 25F84, arm64 | Observed |
| Processor | Apple M2 Max, 12 CPU cores, 38 GPU cores | Observed |
| Memory | 32 GB physical RAM | Observed |
| Disk | 926 GiB total, 526 GiB free at preflight | Observed |
| Empirical environment | Dedicated ignored `.venv-stage1a` | Created |
| Python | CPython 3.11.13 | Observed in that environment |
| PyTorch | 2.13.0 | Installed and imported |
| `circuit-tracer` | 0.5.2 from commit `8f1e2438df612464e229e44c4a00ff637bf9379b` | Installed and import-tested |
| TransformerLens | 3.2.1 | Installed |
| Transformers | 4.57.3 | Installed |
| NNSight | 0.6.1 | Installed |
| CUDA | unavailable; device count 0 | Observed |
| MPS | build support `true`; runtime availability `false` | Allocation probe failed |

The installed `circuit-tracer` distribution's `direct_url.json` identifies the
official `decoderesearch/circuit-tracer` Git repository and records both the
requested revision and resolved commit as
`8f1e2438df612464e229e44c4a00ff637bf9379b`. This verifies the source package
pin independently of the displayed package version.

The MPS probe did not merely rely on `torch.backends.mps.is_available()`. It
attempted an allocation and backward operation on `device="mps"`. PyTorch
reported `is_built() == True`, `is_available() == False`, and raised a
`RuntimeError` saying that MPS requires macOS 14.0 or newer, despite the host
reporting macOS 26.5.2. This inconsistent runtime detection was preserved as a
blocker; hidden CPU fallback was not enabled and a full attribution run was not
silently moved to CPU.

Because no model was loaded, `transformerlens`, `torch.bfloat16`, and the
official CPU/disk offload choices remain reproduction configuration values, not
an observed model-execution backend, dtype, or offload result.

## 3. Immutable asset resolution

### 3.1 Model

- Repository: `google/gemma-2-2b`
- Exact revision: `c5ebcd40d208330abc697524c919956e692655cf`
- Hugging Face metadata: `private=false`, `gated="manual"`, license `gemma`
- Exact-revision configuration probe: HTTP 403 with the available token-safe
  access path
- Local immutable snapshot consumed by the loader: **no**

The model SHA was resolved from the official Hugging Face API. An exact-revision
`config.json` request through `hf_hub_download` first encountered the restricted
network sandbox; after the permitted network retry, Hugging Face returned 403.
No token was printed, serialized, or committed. Recording the SHA alone does not
satisfy immutable execution: the exact snapshot was not downloaded, and
`ReplacementModel.from_pretrained` was never allowed to fall back to mutable
`main`.

### 3.2 Official-demo transcoder

- Selected repository: `mwhanna/gemma-scope-transcoders`
- Exact revision: `bd5773156dea09893636c801df1237d0410307d2`
- Why selected: the audited commit's executable `gemma` alias resolves to this
  repository
- Selected runtime layer-weight subset: 26 direct float32 safetensor files,
  7,855,395,600 bytes (approximately 7.316 GiB)
- Full repository inventory: 56 files, 11,834,796,305 bytes; unrelated
  `features/*.bin` files are not part of the selected loader subset
- Per-layer schema from metadata/header inspection:
  `W_enc[16384,2304]`, `W_dec[16384,2304]`,
  `threshold[16384]`, `b_enc[16384]`, and `b_dec[2304]`
- Full snapshot downloaded or loaded: **no**

Only small public metadata, configuration data, and bounded safetensor ranges
were inspected. The selected-transcoder header inspection read 424 bytes. Two
64-KiB future-CLT range requests read 131,072 bytes total, of which
approximately 130,736 bytes (about 127 KiB) were leading tensor payload beyond
the headers. No complete transcoder tensor file was downloaded.

The pinned README names
`mntss/gemma-scope-transcoders@9250a2d4860ce5ed5c96c14d5882b7d8162809a3`
instead. That outer repository contains 26 external `hf://` references to
`google/gemma-scope-2b-pt-transcoders` that omit revisions. The separately
inspected current Google source revision was
`50eec2f25c60545a9a74c1c3a26a0afdd0b4b872`; it is not embedded in those
references. Pinning only the outer `mntss` revision would therefore not make
its transitive asset loads immutable. The executable alias plus direct-file
layout is the evidence for selecting the `mwhanna` asset for the future
reproduction; it is not evidence that the asset was loaded successfully in
this stage.

### 3.3 Future CLT metadata

The future Stage 1B/E2 candidate was resolved without a full download as
`mntss/clt-gemma-2-2b-426k@b1e9ab376d07c90d780bf20b5fb1a0c89bd0f5e7`.
Metadata and two bounded 64-KiB file-prefix ranges identify 26 layers, 16,384
features per layer (425,984 total), `d_model=2304`, 52 bfloat16 safetensors, and
28,463,527,576 bytes. No threshold key was present, so the pinned loader selects
ReLU. This is metadata for a future candidate only; it was not executed and is
not the PLT used by the official demo.

## 4. Official source examples and fixed parameters

The sources were read at upstream commit
`8f1e2438df612464e229e44c4a00ff637bf9379b`; mutable notebook versions were not
used.

### 4.1 Attribution

Source: `demos/attribute_demo.ipynb`, code cells 4, 6, and 8.

- Prompt: `The capital of state containing Dallas is`
- Model: `google/gemma-2-2b`
- Upstream transcoder spelling: mutable alias `gemma`, resolved above to the
  explicit repository and revision
- Backend: `transformerlens`
- Dtype: `torch.bfloat16`
- `max_n_logits=10`
- `desired_logit_prob=0.95`
- `max_feature_nodes=8192`
- `batch_size=256`
- Offload: `disk` on Colab, otherwise `cpu`

**Outcome:** not run. There is no graph, graph shape, node/edge count, raw graph
artifact, graph checksum, wall-clock measurement, or peak-memory measurement.
No attribution summary was created, because an empty or synthetic summary would
misrepresent a blocked configuration as an empirical result.

### 4.2 Feature intervention

Source: `demos/intervention_demo.ipynb`, code cells 4, 8, 10, and 12.

- Prompt: `Hecho: Michael Jordan juega al`
- Feature: `(layer=20, position=-1, feature_id=341)`
- Official intervention: absolute desired post-gate activation `0.0`
- Required Stage 1A comparisons: baseline, no-op at the measured baseline
  activation, `alpha=0.5` mapped to half the baseline activation, and
  `alpha=1.0` mapped to zero
- Official default regime to verify: underlying-model residual edit with
  `freeze_attention=True` and `constrained_layers=None`

**Outcome:** not run. There are no measured activations, logits,
probabilities, no-op discrepancies, repeat discrepancies, signed token changes,
timings, peak-memory measurements, or intervention checksums. The official
feature's loaded active/inactive state is unknown and was not inferred from the
notebook narrative.

## 5. Runtime semantic verification

The source-level semantics remain those established in
`docs/UPSTREAM_API_AUDIT.md`, but none of the mandatory loaded-asset checks ran.
In particular, this stage did **not** verify on loaded tensors:

- aligned preactivation, post-gate activation, and raw-threshold caches;
- the `[layer, token_position, feature_id]` cache shape;
- active, inactive, and near-threshold samples;
- strict JumpReLU equality behavior on the loaded activation module;
- numerical inclusion of `b_enc` and exclusion of `b_dec` in preactivation;
- an observed baseline-inactive feature reference;
- baseline-versus-no-op agreement;
- the absolute intervention-value convention at runtime; or
- `desired=(1-alpha)*baseline_activation` against an executed intervention.

Offline helpers and source-pinned tests can validate serialization, input
validation, and the algebraic mapping. They cannot substitute for the loaded
model/transcoder evidence required by Section 10 of the Stage 1A task.
Accordingly, no `semantics_summary.json` is claimed.

A tightly bounded diagnostic did execute the installed pinned JumpReLU
implementation at threshold `1.0`: inputs `[0.5, 1.0, 1.5]` produced
`[0.0, 0.0, 1.5]`. This confirms source-level strict-equality behavior only; it
does not substitute for verification on the actually loaded transcoder.

## 6. Commands actually executed

The following commands were executed. A private absolute interpreter prefix is
redacted from the first command to keep local home paths out of tracked files;
the interpreter reported Python 3.11.13.

```bash
<python-3.11.13>/bin/python -m venv .venv-stage1a
.venv-stage1a/bin/python -m pip install \
  'circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer.git@8f1e2438df612464e229e44c4a00ff637bf9379b'
.venv-stage1a/bin/python -m pip check
.venv-stage1a/bin/python scripts/stage1a/verify_environment.py
```

An inline Python/PyTorch probe then recorded Python, package versions, CUDA and
MPS flags, attempted an MPS allocation and backward operation, and captured the
runtime exception. Official Hugging Face API metadata requests used immutable
repository revisions. The exact model configuration probe was equivalent to:

```bash
.venv-stage1a/bin/python -c \
  'from huggingface_hub import hf_hub_download; hf_hub_download(repo_id="google/gemma-2-2b", revision="c5ebcd40d208330abc697524c919956e692655cf", filename="config.json")'
```

The network-restricted attempt and its approved retry were both diagnostic
attempts; only the retry reached Hugging Face and returned the access error.
Token values and authorization headers were never emitted.

The model-dependent entry points were each executed against the resolved config:

```bash
.venv-stage1a/bin/python scripts/stage1a/reproduce_attribution.py --config \
  configs/stage1a_gemma2_2b_official_reproduction.yaml
.venv-stage1a/bin/python scripts/stage1a/reproduce_intervention.py --config \
  configs/stage1a_gemma2_2b_official_reproduction.yaml
.venv-stage1a/bin/python scripts/stage1a/verify_runtime_semantics.py --config \
  configs/stage1a_gemma2_2b_official_reproduction.yaml
```

All three failed closed with exit status 2 before loading assets because the
configured MPS device was built but unavailable. They wrote no empirical
summary. This diagnostic execution must not be mistaken for a completed model
run or loaded-runtime semantic verification.

The final metadata refresh and integrated offline checks were:

```bash
.venv-stage1a/bin/python scripts/stage1a/resolve_assets.py --output \
  results/stage1a/asset_manifest.json
.venv-stage1a/bin/python scripts/stage1a/preflight.py --output \
  results/stage1a/environment_manifest.json
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy src scripts tests
.venv/bin/python scripts/doctor.py
.venv/bin/python scripts/verify_math.py
.venv/bin/python -c 'import cfsus'
.venv-stage1a/bin/python -m pip check
.venv-stage1a/bin/python scripts/stage1a/verify_environment.py
.venv/bin/python scripts/stage1a/validate_artifacts.py --write-checksums
.venv/bin/python scripts/stage1a/validate_artifacts.py
.venv/bin/python scripts/stage1a/scan_commit_safety.py
.venv/bin/python -m pytest -q -m model
git diff --check
git status --short
git diff --exit-code HEAD -- docs/EXPERIMENT_LOG.md paper
```

The resolver wrote a truthful blocked manifest and returned status 2 because
the exact model probe remained HTTP 403; it requested no weight downloads.
Results were 167 passed and one model test deselected in the default suite;
Ruff lint and format passed; strict mypy passed across 39 source files; doctor,
deterministic-math verification, import smoke, the full observed environment
lock comparison, and `pip check` passed; all three metadata artifacts plus
their checksums validated; and the pre-commit candidate safety scan found zero
findings across 74 paths. The explicitly selected model test was collected but
skipped because `CFSUS_RUN_STAGE1A_MODEL_TESTS` was intentionally unset; the
required immutable snapshots and accelerator were unavailable in any case.
These are offline engineering checks and do not satisfy loaded-runtime
semantics.

## 7. Reproducibility artifacts and Colab handoff

The branch prepares a Python 3.11 environment specification, resolved
platform-specific dependency record, immutable asset metadata, configuration
validation, small deterministic artifact schemas, redaction/checksum helpers,
and a Colab-compatible handoff that invokes tracked code with the same pins.
The tracked handoff paths are
`notebooks/stage1a_official_reproduction_colab.ipynb` and
`results/stage1a/colab_handoff_manifest.json`.

The Colab handoff is a fallback execution route, not evidence that the notebook
was run. It requires the operator to paste the final 40-character
`stage-1a-reproduction` branch head; the notebook clones the branch and refuses
to continue unless its resolved `HEAD` equals that expected commit. It must
still receive authorized Gemma access, install the pinned upstream commit,
consume the same immutable model/transcoder revisions, pass the CUDA BF16 and
unverified 14-GiB-total/12-GiB-free VRAM preflight policy, execute the tracked
scripts, and return an artifact bundle that passes local validation. The VRAM
floor is not a measured requirement or OOM guarantee; the runner preserves a
sanitized failure diagnostic if execution still fails. No Hugging Face token
is embedded in the notebook or tracked configuration.

Only environment and asset metadata may be committed for this blocked result.
The following empirical files are intentionally absent:

- `results/stage1a/attribution_summary.json`
- `results/stage1a/intervention_summary.json`
- `results/stage1a/semantics_summary.json`
- any raw graph or dense activation cache

No experiment entry was appended to `docs/EXPERIMENT_LOG.md`: environment
creation, metadata inspection, a failed access probe, and notebook preparation
are not completed scientific runs.

The committed metadata checksums are:

- `asset_manifest.json`: `253b7bb84ac226b9739b489538ec548e90c6eb190c19631b4bbaa86619df753d`
- `colab_handoff_manifest.json`: `1a12471336498e13b83508d928b6c48799cddf4c14f31122cedd67b4fc1518e4`
- `environment_manifest.json`: `caa99f09ff2d7c71fb6ec2f3a7e59173dce7cc9064ffcd46276f13f352df9a72`

## 8. Acceptance-criteria accounting

| Criterion | Status | Evidence or failure |
| --- | --- | --- |
| Preserve repository state | Met | Dedicated branch from verified Stage 0 base; no reset or main mutation |
| Python 3.11 reproducible environment | Met | Python 3.11.13 dedicated environment and dependency record |
| Exact audited upstream install | Met | Version plus `direct_url.json` commit verification |
| Exact model/transcoder SHAs resolved | Met for metadata | Both exact SHAs resolved; neither full asset consumed by a loader |
| Loader consumes immutable assets | **Failed** | Model access returned 403; no local snapshot |
| Official attribution produces validated nonempty graph | **Failed** | Not run |
| Official intervention runs on `(20,-1,341)` | **Failed** | Not run |
| Baseline/no-op agreement | **Failed** | Not run |
| Loaded activation/preactivation/threshold/gate checks | **Failed** | No loaded asset |
| Runtime absolute-value and alpha mapping | **Failed** | Offline mapping only; no runtime check |
| Empirical summaries, checksums, timing, and memory | **Failed** | No empirical run or raw artifact |
| Experiment log contains only completed runs | Met | No entry added |
| Stage 0 and new offline checks | Met | 167 passed, 1 model test deselected; lint, format, mypy, doctor, math, import, environment, artifact, and safety checks passed |
| No secret, weights, cache, graph, or private path committed | Met before commit | Safety scan reported zero findings across 74 candidate paths; staged mode is rerun after staging |
| No Counterfactual Susceptibility claim | Met | Explicitly disclaimed below |
| Branch committed and pushed without touching `main` | Recorded in final handoff | Does not change blocked empirical verdict |

Because multiple mandatory empirical criteria failed, Stage 1A is not
`complete`. `partial` would also overstate the result: no model/transcoder
runtime execution succeeded. Under the task's classification rule, an access
and accelerator/runtime incompatibility that prevents actual model execution is
`blocked`.

## 9. Failures, warnings, and deviations

1. The exact model revision is manually gated, and the available access path
   received HTTP 403 for its immutable `config.json`.
2. MPS was compiled into PyTorch but unavailable at runtime; the allocation
   probe produced an OS-version error inconsistent with the reported host OS.
3. No hidden MPS-to-CPU fallback was enabled. A full CPU attribution was not
   attempted, as required by the execution policy.
4. Only metadata and the bounded ranges disclosed in Section 3 were fetched
   for public transcoder assets. The 7.316-GiB official-demo PLT subset and
   26.508-GiB future CLT were not downloaded in full.
5. No alternative model was substituted for Gemma. Such a smoke test would not
   satisfy official reproduction in any case.
6. No scientific prompt, feature, backend, revision, or attribution parameter
   was silently changed to obtain a result.
7. There are no empirical wall-clock or peak-memory values because no empirical
   run began. Diagnostic command timings must not be presented as model-run
   measurements.

## 10. Scientific interpretation and non-claims

The asset pins, package installation, metadata inspection, offline tests, and
Colab handoff are engineering and reproducibility work. They do not establish:

- that Counterfactual Susceptibility predicts any feature response;
- that source suppression caused an inactive feature to cross a gate;
- that any observed feature is semantically meaningful;
- behavioral importance or causal mediation;
- a raw inactive-target Jacobian or local response;
- mechanistic faithfulness or an underlying-model circuit; or
- the prevalence, calibration, or scalable discoverability of gate crossings.

No empirical values were copied from upstream notebook output or fabricated.

## 11. Stage 1B readiness

**Stage 1B is not ready to begin.** First, authorized access must make the exact
Gemma snapshot consumable; a working accelerator path must be established
locally or through the prepared pinned Colab route; both official examples must
complete; and every mandatory loaded-runtime semantic check and artifact
validation must pass. Only then is E0 complete.

Even after E0 completion, beginning Stage 1B would authorize only its separately
defined engineering/feasibility work. It would not by itself authorize a large
benchmark or any susceptibility, gate-crossing, behavioral, mediation, or
circuit claim. The large benchmark remains gated by E0 and by a later
candidate-generation recall audit against a brute-force subset.
