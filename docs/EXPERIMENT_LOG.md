# Experiment Log

**Status:** One local small-model runtime-validation pilot has completed. No
Counterfactual Susceptibility experiment, reference reproduction, or paper
Results experiment has completed.

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

---

## EXP-20260824-002 — Stage 1A-S-BF16 local MPS/BF16 runtime validation

- **Status:** `completed_small_model_mps_bf16_pilot`
- **Date (UTC):** 2026-08-24T16:50:30Z to 2026-08-24T16:50:44Z
- **Authors/operators:** Codex App on the local Apple M2 Max host
- **Scientific question or hypothesis:** Can the exact BF16-trained Gemma 3
  270M checkpoint, pinned 18-layer PLT subset, and NNsight replacement runtime
  execute locally on native Apple MPS/BF16 with finite attribution and correct
  absolute feature-intervention semantics?
- **Prerequisite experiment IDs:** EXP-20260824-001 is protected negative FP16
  runtime evidence, not an acceptance prerequisite
- **Planned versus exploratory:** planned local runtime-validation recovery
- **Code commit:** `6a5c21027fbb6b83e34c39db75987b0ce5b72d17`
- **Dirty-tree status:** clean execution commit, already present on the isolated
  origin branch before canonical accepted launch
- **Configuration:** `configs/stage1a_gemma3_270m_mps_bf16_pilot.yaml`,
  SHA-256 `429662d9598ba24d769aca9239271183c894eb98f85c76f95a38b5b78df65c6d`
- **Random seeds:** no stochastic feature selection or sampling; deterministic
  frozen candidate rule and stable numeric tie-break
- **Upstream package:** `https://github.com/decoderesearch/circuit-tracer`,
  commit `8f1e2438df612464e229e44c4a00ff637bf9379b`, version 0.5.2
- **Model:**
  `google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- **Transcoder/PLT:**
  `mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`,
  `transcoder_all/width_16k_l0_small`
- **Prompt inputs:** BOS-only, `Hello`, and accepted pilot
  `The capital of France is`; accepted token IDs
  `[2, 818, 5279, 529, 7001, 563]`
- **Behavior metric:** none. This experiment validates runtime response and
  control consistency, not behavior semantics or susceptibility.
- **Hardware:** Apple M2 Max, MPS, 32 GiB unified memory, 12-core CPU
- **Software:** native arm64 macOS; CPython 3.11.13; PyTorch 2.6.0;
  NNsight 0.6.1; circuit-tracer 0.5.2; Transformers 4.57.3; fallback disabled
- **Intervention:** frozen direct-contribution rule selected layer 17,
  position 5, feature 1191, baseline activation 1960. Absolute desired values
  were `(1-alpha)*baseline` for alpha 0, 0.5, and 1.0, with
  `freeze_attention=true` in baseline and every condition.
- **Primary metrics and results:** finite graph shape `[2276,2276]`, 2,152
  active/selected features, 1,454,640 nonzero edges, zero non-finite values.
  Candidate audit independently reproduced the selected feature. Raw/frozen,
  repeat, and no-op normalized-L2 and maximum-absolute differences were zero.
  Half/full response magnitudes were 0.075792/0.158862 normalized L2 and
  2.03125/4.03125 maximum absolute logit difference; no semantic-direction
  claim.
- **Replacement-model result:** all 18 PLTs and NNsight replacement runtime
  passed MPS/BF16 parameter, loaded JumpReLU, active/inactive, and finite gates.
- **Underlying-model result:** all three prompt classes remained finite; the
  separate post-MPS CPU/FP32 diagnostic passed every preregistered threshold.
- **Failures, warnings, and anomalies:** canonical accepted attempt had none
  and required no retry. Three pre-accepted engineering failures and one
  batch-64 runtime pass invalidated for a missing mandatory compact evidence
  field are retained with explicit dispositions in attempt provenance.
- **Peak memory and wall-clock time:** supervisor about 13.37 seconds; MPS
  current 689,690,112 bytes; MPS driver 3,009,298,432 bytes; process RSS
  1,103,904,768 bytes; minimum available 11,848,892,416 bytes; swap growth 0;
  thermal nominal. Overlapping unified-memory counters are not summed.
- **Artifacts:** `results/stage1a_small_model_mps_bf16/`, 13 allowlisted files,
  996 KiB total; checksum-manifest SHA-256
  `ea7bd6db0ceca579f4b62aba530d419a3c0bc1e3ee98abff5ccf6938062d4b95`
- **Deviation from plan:** one first runtime pass was correctly invalidated and
  preserved because a required compact max-absolute control field was missing.
  A new pre-run commit and fresh canonical run followed the binding correction
  policy; canonical batch 64 passed.
- **Decision:** local engineering runtime validation is complete. Keep Stage 1B
  empirical readiness false and official/reference reproductions pending.
- **Follow-up:** design Stage 1B engineering only under a separate Goal; do not
  begin Counterfactual Susceptibility or edit paper Results from this result.
