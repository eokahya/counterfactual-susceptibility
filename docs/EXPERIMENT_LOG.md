# Experiment Log

**Status:** No scientific experiment has completed successfully. One failed
real-runtime attempt is recorded below; it produced no scientific result.

Unit tests, environment inspection, and deterministic formula verification are
Stage 0 engineering checks. They are recorded in `docs/STAGE_0_REPORT.md`, not as
model evidence here.

For every completed scientific run, copy the template below and append it to the
end of this file. Entries are immutable after publication in this log. If an
interpretation changes, append a new linked entry rather than rewriting the old
one.

---

## EXP-YYYYMMDD-NNN — Short descriptive title

- **Status:** completed | failed | aborted
- **Date (UTC):** YYYY-MM-DDTHH:MM:SSZ
- **Authors/operators:**
- **Scientific question or hypothesis:**
- **Prerequisite experiment IDs:** none
- **Planned versus exploratory:**
- **Code commit:** full commit hash
- **Dirty-tree status:** clean | dirty, with diff/artifact reference
- **Configuration:** tracked path plus an archived resolved configuration
- **Random seeds:**
- **Upstream package:** repository URL, exact commit, package version
- **Model:** identifier and immutable revision
- **Transcoder/CLT:** identifier and immutable revision
- **Prompt inputs:** identifiers, tracked file, or deterministic generation recipe
- **Behavior metric:** name, direction, token/position convention, target and
  contrast where applicable
- **Hardware:** accelerator, memory, CPU, RAM
- **Software:** OS, Python, PyTorch, accelerator/runtime, relevant dependencies
- **Intervention:** source/target selection and exact `alpha` convention
- **Primary metrics and results:** signed values and uncertainty where applicable
- **Replacement-model result:** not run | summary and artifact
- **Underlying-model result:** not run | summary and artifact
- **Failures, warnings, and anomalies:**
- **Peak memory and wall-clock time:**
- **Artifacts:** paths to raw metadata, metric tables, logs, figures, and checksums
- **Deviation from plan:** none | description and rationale
- **Decision:** continue | revise | stop
- **Follow-up:**

---

Never record a planned, configured, or partially scaffolded run as completed.

---

## EXP-20260823-001 — Stage 1A MPS/FP16 progressive runtime-load attempt

- **Status:** failed
- **Date (UTC):** 2026-08-23T11:38:05Z to 2026-08-23T12:46:43Z
- **Authors/operators:** Codex App on the local Apple M2 Max host
- **Scientific question or hypothesis:** Can the exact pinned Stage 1A Gemma
  model and transcoder be loaded progressively into the separate MPS/FP16
  runtime, pass loaded semantics, and proceed to the fixed attribution and
  intervention without weakening the experiment?
- **Prerequisite experiment IDs:** none
- **Planned versus exploratory:** planned hardware-adaptation reproduction;
  bounded postmortem operator diagnosis was exploratory engineering work
- **Code commit:** `9de01b5446775f01b211acbc461f8385f9f3732a`
- **Dirty-tree status:** tracked tree clean at launch; only the preserved
  `results/stage1a_t4_fp16/` directory was untracked
- **Configuration:**
  `configs/stage1a_gemma2_2b_mps_fp16_reproduction.yaml`, SHA-256
  `b17d96a66bb307670911c5ac76b247ba5233935a3ce84887bd9cd886d98af8bc`
- **Random seeds:** Python 0, NumPy 0, Torch 0
- **Upstream package:** `https://github.com/decoderesearch/circuit-tracer`,
  commit `8f1e2438df612464e229e44c4a00ff637bf9379b`, version 0.5.2
- **Model:**
  `google/gemma-2-2b@c5ebcd40d208330abc697524c919956e692655cf`
- **Transcoder/CLT:**
  `mwhanna/gemma-scope-transcoders@bd5773156dea09893636c801df1237d0410307d2`
- **Prompt inputs:** attribution `The capital of state containing Dallas is`;
  intervention `Hecho: Michael Jordan juega al`
- **Behavior metric:** configured official Dallas target with
  `max_n_logits=10`, desired logit probability `0.95`, and a maximum of 8192
  feature nodes; not evaluated
- **Hardware:** Apple M2 Max, MPS, 32 GiB unified memory, 12-core CPU
- **Software:** macOS 26.6.2 arm64; Python 3.11.13; PyTorch 2.6.0;
  TransformerLens 3.2.1; Transformers 4.57.3; nnsight 0.6.1;
  huggingface-hub 0.36.2; fallback disabled
- **Intervention:** configured feature `(20, -1, 341)`, alphas `0.0`, `0.5`,
  and `1.0`, frozen attention, unconstrained layers; not run
- **Primary metrics and results:** the model-only MPS/FP16 forward passed with
  token shape `[1, 8]`, finite logit shape `[1, 8, 256000]`, before transcoder
  loading. The worker then failed in `runtime_loading`; loaded semantics,
  attribution, and intervention were not reached.
- **Replacement-model result:** not run; construction did not return a bundle
- **Underlying-model result:** model-only progressive gate passed; no behavior
  metric was computed
- **Failures, warnings, and anomalies:** attempt category `failed_runtime`,
  `RuntimeError`, `oom_confirmed=false`, `retry_eligible=false`, and no retry.
  The original worker over-redacted the leaf diagnostic. A bounded real-MPS
  postmortem reproduced a compatible deterministic failure from unsupported
  FP16 `index_put_` accumulation at the next known sparse-boundary check;
  because the original leaf text was not retained, exact identity is
  unconfirmed. Unique-coordinate replacement passed with zero error and was
  fixed in `86ec118`. TransformerLens also emitted its unsuppressed MPS
  correctness warning. The real loading telemetry falsified the static memory
  plan, so the large runtime was not rerun after the operator fix.
- **Peak memory and wall-clock time:** `4,117.836113` seconds; 3,858 samples;
  MPS current `35,977,178,880` bytes; MPS driver `40,032,174,080` bytes;
  process RSS `5,948,440,576` bytes; system swap `34,567,031,357` bytes.
  These overlapping unified-memory counters are not summed. The failed report
  did not retain sampled pressure states, so intervening critical pressure
  cannot be excluded.
- **Artifacts:** ignored preflight
  `results/generated/stage1a_mps_fp16/preflight-mndxhfn0/preflight_summary.json`
  (SHA-256
  `098d1cd0892cc3ef802375284aa51ab82b08079cc6cda3ebafbb19c683b5bb1c`)
  and ignored attempt
  `results/generated/stage1a_mps_fp16/attempt-256-yhp6ax3r/attempt_report.json`
  (SHA-256
  `bae2ee7e0061de08db8f71a6c9c0734212b4666c49d27a65bb7131cbaedfc49e`).
  No canonical science bundle was published.
- **Deviation from plan:** only the preregistered explicit CPU sparse-metadata
  boundary; no scientific parameter changed. The later operator replacement
  is equivalent for unique coordinates and was not used to claim a completed
  run.
- **Decision:** stop. Preserve the attempt as failed and classify the overall
  Stage 1A MPS effort as blocked at the re-evaluated conservative memory gate.
- **Follow-up:** do not launch the identical loading plan on this 32 GiB host.
  Any future attempt requires a new audited loading-plan identity, a
  conservative budget informed by the measured driver/swap peaks, renewed MPS
  numerical validation, and no silent change to scientific parameters.

---

## EXP-20260824-001 — Stage 1A-S Gemma 3 270M MPS/FP16 model gate

- **Status:** `failed_runtime`
- **Date:** 2026-08-24 (Europe/Istanbul)
- **Experiment class:** Stage 1A-S local small-model runtime validation
- **Base/branch:** `4ef60d2b5f8120d5671afbf8400b61d66e291f4d` /
  `stage-1a-small-model-mps-fp16`
- **Upstream:** official `circuit-tracer` v0.5.2 at
  `8f1e2438df612464e229e44c4a00ff637bf9379b`
- **Model:**
  `google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- **Transcoder:**
  `mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`,
  `transcoder_all/width_16k_l0_small`; downloaded and verified but not loaded
  because the preceding model gate failed
- **Environment:** native arm64 CPython 3.11.13, PyTorch 2.6.0, MPS built and
  available, NNsight 0.6.1, Transformers 4.57.3, fallback disabled; lock SHA-256
  `2ddfaebbad636911f9033f7d46236ddf4b38513215011eb4fe1214fde7f583c4`
- **Prompt:** `The capital of France is`; token IDs
  `[2, 818, 5279, 529, 7001, 563]`
- **Observed model result:** all parameters were MPS/FP16. Hidden states were
  finite through index 7. The first non-finite value occurred after decoder
  layer 7 at coordinate `[0, 0, 163]`, where finite operands `55520` and
  `13408` sum to `68928`, above FP16 maximum `65504`. Hidden-state index 8
  contained one positive infinity; later layers produced all-NaN logits
  (`1,572,864` NaNs of `1,572,864` values; shape `[1, 6, 262144]`).
- **Prompt-independence diagnosis:** BOS-only, `Hello`, and the planned prompt
  all first became non-finite at hidden-state index 8.
- **Telemetry:** 6 one-second samples; MPS current peak `551108608` bytes;
  MPS driver peak `1126842368` bytes; process RSS peak `607125504` bytes;
  minimum available memory `13509787648` bytes; swap growth `0`; thermal state
  nominal; no memory or thermal limit violated.
- **Retry:** none. The specification forbids retry for non-finite values and
  permits retry only for verified attribution MPS OOM after runtime loading.
- **Downstream stages:** one-layer semantics, full PLT, NNsight replacement,
  attribution, intervention, accepted protocol, and canonical artifact bundle
  were not run.
- **Decision:** stop. Do not substitute BF16, FP32, CPU, CUDA, another model,
  or a mixed-precision residual adapter under this experiment identity.
- **Artifacts:** exact weights remain only in a project-external ignored cache;
  diagnostic JSON remains under ignored `results/generated/`. No weights,
  cache, raw graph, or canonical science artifact was published.
