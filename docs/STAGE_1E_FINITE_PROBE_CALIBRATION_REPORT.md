# Stage 1E Finite-Probe Calibration Report

Status date: 2026-08-27

## Terminal outcome

Stage 1E completed in Phase A as:

```text
completed_stage1e_offline_negative
project_decision: simple_finite_probe_calibration_not_supported
```

Neither finite-probe estimator passed every frozen development gate. No
estimator was selected, no protocol or prediction freeze for Phase B was
created, and none of the ten fresh confirmatory prompts was run. Phase A and
Phase B both used zero model calls and zero intervention API calls.

## 1. Stage 1D development reanalysis

The analysis used only the accepted, tracked Stage 1D serialized artifacts.
The Stage 1D bundle first passed its original standalone validator and all nine
covered JSON checksums. Ignored temporary outputs, journals, model caches,
NNsight runtime objects, and historical in-memory worker results were not read.

Exact provenance:

- Remote Stage 1D base:
  `b71df55fdeb2fb66601af56207b6fbe5238e57d8`.
- Audited Stage 1D execution/artifact ancestor:
  `2a5c3e63a838e7547fe1b30fe888610ec21ee46e`.
- Stage 1E code/artifact commit:
  `f1cbaa29ba4d7ee0133a4b6c5011709f723e8980`.
- Offline analysis SHA-256:
  `696fcc97bf757b643f6bd428e85bfa6f5b210722a70920449567ed29801e8e26`.
- Run manifest SHA-256:
  `1296eb0f97f4dedf5a3a27254310f041fd73b4c2ee5c9eb8f7750a2f32c4b116`.

All 32 Stage 1D detailed positive-pair trajectories were reconstructed from
serialized point rows. Every calculation used the BF16-realized suppression,
not the requested alpha. The primary error/correlation reference was the same
12 B1/B2/B3 pairs that were Stage 1D quantization-resolvable, had an observed
crossing bracket, and were monotonic. The near-boundary rows remained in the
32-pair full-ablation classification and abstention analyses but were not
silently promoted into the critical-alpha reference set.

For historical offline replay, a nominal probe was usable only when its
BF16-applied source value existed in the accepted serialized trajectory. The
frozen E1 grid was `[0.125, 0.1875, 0.25]`; the first available nonzero applied
value was used. The E2 second-probe grid was `[0.5, 0.625]`. All 32 trajectories
had an available E1 and E2 probe value, although the estimators could still
abstain under their frozen mathematical rules.

## 2. Estimator comparison

| Metric | E0: zero probe | E1: one-probe secant | E2: two-probe quadratic |
| --- | ---: | ---: | ---: |
| Eligible critical pairs | 12 | 9 | 7 |
| Coverage | 1.0000 | 0.7500 | 0.5833 |
| Spearman | 0.6941 | 0.7983 | 0.6727 |
| Median absolute error | 0.1069 | 0.0587 | 0.0787 |
| p95 absolute error | 0.2926 | 0.1015 | 0.1015 |
| Median bracket distance | 0.1041 | 0.0549 | 0.0767 |
| p95 bracket distance | 0.2906 | 0.1000 | 0.1000 |
| Full-ablation classification accuracy | 0.6563 | 0.7500 | 0.7188 |
| Abstention rate over all 32 pairs | 0.0000 | 0.5000 | 0.6250 |
| Nonmonotonic rejection rate | 0.0000 | 0.6000 | 0.6000 |
| Finite probes per accepted prediction | 0 | 1 | 2 |

E1 reduced median absolute error by 45.1% relative to E0 and improved the
point-estimate Spearman correlation, but it produced only 9 accepted reference
predictions. E2 reduced median error by 26.4%, but had lower coverage and lower
Spearman than E0.

Prompt-level 10,000-resample bootstrap intervals were:

| Estimator | Spearman 95% interval | Median-error 95% interval |
| --- | ---: | ---: |
| E0 | [0.1916, 0.9450] | [0.0163, 0.1802] |
| E1 | [0.5190, 1.0000] | [0.0087, 0.0847] |
| E2 | [−0.1765, 1.0000] | [0.0105, 0.0962] |

These are development-set, method-enriched panel diagnostics. They are not
population estimates or confirmatory evidence.

## 3. Frozen offline decision

E1 passed four of five requirements:

```text
eligible pairs >= 10:                            false (9)
coverage >= 0.60:                                true  (0.75)
median error <= 0.80 * E0 median error:           true
Spearman >= max(0.70, E0 Spearman):               true
full-ablation classification no worse than E0:   true
```

E2 passed two of five requirements:

```text
eligible pairs >= 10:                            false (7)
coverage >= 0.60:                                false (0.5833)
median error <= 0.80 * E0 median error:           true
Spearman >= max(0.70, E0 Spearman):               false (0.6727)
full-ablation classification no worse than E0:   true
```

The preference order was E1 before E2, but neither estimator passed all
criteria. The valid terminal decision is therefore
`simple_finite_probe_calibration_not_supported`. E1's in-development error
reduction is worth recording, but the frozen minimum-eligibility failure means
it cannot authorize confirmatory model calls.

## 4. Fresh confirmatory phase

Phase B did not run. In particular:

```text
fresh confirmatory prompts executed: 0 / 10
baseline model calls:                 0
finite-probe intervention calls:      0
full-ablation calls:                  0
critical-alpha refinement calls:      0
canonical attempts:                   0
scientific retries:                   0
```

Because no scientific model call was permitted after the offline gate failed,
the synthetic and real MPS production rehearsals that were conditional on a
future model call were also unnecessary. No Stage 1E attempt lock or journal
was created.

## 5. Scientific interpretation

Inhibitory influence `q` remains the candidate-discovery ranker. Stage 1E does
not rehabilitate `q/m` as a full-ablation discovery ranker. A single finite
secant probe showed useful development-set error reduction among the pairs on
which it produced an estimate, but abstention reduced the accepted reference
set below the preregistered minimum. The quadratic estimator did not recover
coverage or rank calibration despite using twice the probe cost.

The next experiment should not proceed directly to behavioral importance or
mediation. A later, separately specified redesign may evaluate second-order/HVP
information or multi-source effects, with new frozen thresholds and fresh
evidence. The current simple finite-probe family should not be scaled as if it
were confirmed.

## 6. Validation, artifacts, and resource cost

The focused Stage 1E plus inherited Stage 1D set passed 9 tests; after the
result-generic gate hardening, the Stage 1E focused set passed again with 3
tests. The complete inherited offline suite passed with 601 passed, 8 skipped,
and 1 deselected. Repository-wide Ruff lint/format, strict scoped MyPy,
dependency checks, import checks, diff checks, and secret/private-path scans
passed.

The standalone Stage 1E validator ran in a separate process. It revalidated
the accepted Stage 1D bundle, reconstructed all 32 trajectories and all three
estimator metric sets from serialized inputs, reproduced the two failed gates,
and verified both Stage 1E checksums. The Stage 1E bundle is 249,897 bytes;
each artifact is well below the 2 MiB cap.

This was CPU-only JSON/statistical analysis. There was no model load, NNsight
runtime, MPS allocation, intervention API use, network access, or new thermal
load to report. No model/transcoder weight, cache, graph, adjacency, dense
activation, derivative matrix, gradient, journal, secret, or private absolute
path is tracked.

## 7. Claim boundary

```text
q_ranker_status: retained_for_candidate_discovery
finite_probe_calibration_status: not_supported
stage1f_behavioral_readiness: false
counterfactual_behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

The final branch SHA and local/origin equality are reported in the immutable
Goal handoff after the documentation commit; a commit cannot contain its own
SHA without changing it.
