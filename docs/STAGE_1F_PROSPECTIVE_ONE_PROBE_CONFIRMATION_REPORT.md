# Stage 1F — Prospective One-Probe Secant Confirmation

## Result

```text
terminal_class: completed_stage1f_e1_not_supported
project_decision: retire_simple_critical_alpha_calibration
canonical_attempt_count: 1
scientific_retry_count: 0
```

This is a valid prospective scientific negative, not a runtime failure. The
exact Stage 1E E1 estimator did not achieve the frozen median-error improvement
threshold on the fresh confirmatory panel. The historical Stage 1E status
remains `completed_stage1e_offline_negative`.

## Immutable provenance

```text
base_commit: f7aae1f3ce3b1b8d98e850093a3cb5ca480277ea
required_stage1e_ancestor: f1cbaa29ba4d7ee0133a4b6c5011709f723e8980
protocol_freeze_commit: 588c1f233c072b89520693f556f25f470d852e8b
prediction_freeze_and_execution_commit: 4193151108e6ad86e775ffa347a42e9e70c3f311
artifact_commit: c1bb6a3bbab3de945767eded4503b17343ba88e6
branch: stage-1f-prospective-one-probe-confirmation
```

The base was the fetched remote head of
`origin/stage-1e-finite-probe-calibration`. The accepted Stage 1D and Stage 1E
bundles passed their preflight standalone validators and checksums. No tracked
authenticated Stage 1E Phase-B prompt list existed, so the binding fallback
list was frozen. The only historical text occurrence, F02 in a V3 candidate
pool, was metadata rather than a baseline or intervention artifact. No exact
historical prompt/source/target intervention identity required exclusion.

## Fresh prompts and frozen panel

```text
F01  The capital of Australia is
F02  The capital of Brazil is
F03  The currency of Switzerland is
F04  The chemical symbol for sodium is
F05  The author of Pride and Prejudice is
F06  The largest ocean is
F07  The boiling point of water is
F08  The planet known as the Red Planet is
F09  The language primarily spoken in Brazil is
F10  The square root of 81 is
```

Every non-BOS token position was used. Candidate discovery used only descending
positive inhibitory influence `q`; `q/m` was not used for ranking. Every prompt
filled all frozen quotas without backfill:

| Stratum | E0 alpha interval | Pairs per prompt | Total |
|---|---:|---:|---:|
| B1 | `[0.02, 0.10)` | 4 | 40 |
| B2 | `[0.10, 0.40)` | 4 | 40 |
| B3 | `[0.40, 0.95]` | 4 | 40 |
| **All** | — | **12** | **120** |

The prediction manifest passed a standalone fresh-process validator before it
was committed and pushed. Fresh source-suppression calls before prediction
freeze were zero.

## Production-path rehearsal and execution

The synthetic worker → fsync journal → assembler → standalone-validator chain
passed. A real MPS/BF16 active-feature rehearsal used an exact identity disjoint
from the confirmatory panel and passed with two engineering calls, two completed
journal points, detached serialization, zero evaluation calls, and no scientific
attempt lock.

The canonical worker then ran once at the pushed prediction-freeze commit.
Every frozen pair received no-op, the first nonzero BF16-realized probe from
`[0.125, 0.1875, 0.25]`, and full ablation. Eligible crossing pairs received at
most six bisections. Results were assembled only from the durable journal.

```text
instrumented API calls: 512
journal call-start records: 512
journal completed-point records: 512
serialized unique point rows: 512
frozen pair sweeps: 120
required calls: 360
refinement calls: 152
```

## Prospective E0 versus E1 result

The paired primary set contains the same E1-accepted, quantization-resolvable,
monotonic, observed-crossing pairs for both estimators.

| Metric | E0 | E1 |
|---|---:|---:|
| Paired reference pairs | 63 | 63 |
| Median absolute error | 0.03099849 | 0.03041363 |
| p95 absolute error | 0.31794872 | 0.32062147 |
| Median bracket distance | 0.00905469 | 0.00887311 |
| Spearman | 0.76491557 | 0.85293871 |
| Full-panel classification accuracy | 0.66666667 | 0.81666667 |

The paired median absolute-error reduction was `0.00058487`, or only `1.8868%`.
The frozen 10,000-resample prompt-cluster bootstrap interval for that reduction
was `[-0.00859353, 0.02554239]`; its lower bound was not positive. E1 therefore
failed both the confirmed median ratio gate (`E1 <= 0.75 * E0`) and the mixed
minimum improvement gate (at least 15%). Under the frozen precedence, 63 paired
references make the result adequately powered and the terminal class is
`completed_stage1f_e1_not_supported`.

E0 full-panel confusion counts were TP 80, FP 40, TN 0, FN 0. E1, with
abstention treated as no crossing, had TP 62, FP 4, TN 36, FN 18. These better
classification and Spearman values do not override the preregistered error gate.

## Coverage, abstention, strata, and cost

```text
observed-crossing monotonic reference denominator: 80
paired E1-accepted references: 63
E1 reference coverage: 0.7875
E1 accepted on full panel: 67 / 120 (0.55833333)
E1 abstentions: 53 / 120 (0.44166667)
abstention reason: nonpositive_or_nonfinite_secant_drive (53)
nonmonotonic pairs: 0 / 120
total calls per accepted E1 prediction: 7.64179104
probe calls expended per accepted E1 prediction: 1.79104478
```

Paired counts by stratum were B1 18, B2 24, and B3 21. Median absolute errors
were respectively E0/E1 `0.01005/0.01330`, `0.02088/0.02171`, and
`0.19939/0.08712`.

## Runtime, telemetry, and validation

The immutable Gemma 3 270M + 18×16K PLT + NNsight runtime ran on Apple MPS in
BF16 with no CPU fallback, CUDA, autocast, network substitution, or retry.

Canonical telemetry:

```text
MPS driver peak bytes: 2865414144
MPS current peak bytes: 591941120
process RSS peak bytes: 1096122368
minimum available memory bytes: 12914573312
swap growth bytes: 0
thermal states: [nominal]
telemetry violations: 0
telemetry failures: 0
```

The final bundle is 1,569,141 bytes; its largest file is 1,237,721 bytes, below
the frozen 2 MiB per-file cap. All six JSON checksums passed. Two independent
fresh-process standalone validations recomputed the same terminal class and
confirmed `512 API = 512 journal completions = 512 serialized points`.
Commit-safety scanning at the frozen 2 MiB review cap found no secret, private
path, cache, weight, graph, dense tensor, derivative, gradient, or forbidden
binary artifact.

## Claim boundary

```text
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

This result rejects this simple prospective critical-alpha calibration under the
frozen Stage 1F rule. It does not establish or refute behavioral importance,
mediation, a reference CLT, Gemma 2, or a paper-ready result.
