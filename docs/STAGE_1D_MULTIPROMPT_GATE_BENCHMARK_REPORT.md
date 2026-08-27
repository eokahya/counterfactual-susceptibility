# Stage 1D Multiprompt Gate Benchmark Report

Status date: 2026-08-27

## Outcome

The canonical benchmark completed as
`completed_stage1d_multiprompt_gate_benchmark`. Its frozen project decision is
`retain_crossing_ranker_but_redesign_calibration`.

Across the eight fresh prompts, Counterfactual Susceptibility (`S = q/m`) added
clear full-ablation crossing-discovery value beyond margin alone, but not
beyond inhibitory influence (`q`) alone. Mean prompt precision@4 was 0.84375
for S, 0.50 for margin-only, 1.00 for influence-only, and 0.3125 for the
deterministic random-positive panel. S therefore exceeded margin-only by
0.34375, but trailed influence-only by 0.15625.

Critical-suppression rank calibration was promising but not accepted:
Spearman was 0.6941, but only 12 monotonic, quantization-resolvable crossing
pairs were available versus the frozen minimum of 20. The detailed positive
panel was also nonmonotonic for 10/32 pairs (0.3125), above the frozen maximum
of 0.20. Directional controls passed with 0/16 violations. This is a valid
mixed/negative benchmark result, not a runtime failure.

## Frozen protocol and provenance

- Exact base: `d4fdcc2c2f0040654af17e21f396f1d26072aa0e` from
  `stage-1c-v4-protocol-preserving-execution`.
- Isolated branch: `stage-1d-multiprompt-gate-benchmark`.
- Protocol freeze: `bad3d4f01a427068c212de5c7d6b99c5e94a9cd0`.
- Protocol map SHA-256:
  `8bc1f35de153222a0de7d7971becbf9c5f80eadaea0ba78abdd586d2cf4a6b8b`.
- Prediction/pre-run freeze:
  `d0cb2b9645aeef4f2ea6ccda11143a03bf26852d`.
- Prediction manifest SHA-256:
  `56ed1466309813c6f42443b69eb1932c9c02cf83721dd2ee592be12a442aec52`.
- Execution/artifact commit:
  `2a5c3e63a838e7547fe1b30fe888610ec21ee46e`.
- Canonical journal SHA-256:
  `b0d7d340bd42de72f25e51ee584cb1e471b5ee5c2266b831a8ccb2f6ef840b83`.

The protocol was pushed before any evaluation-prompt baseline model call. The
combined prediction manifest was produced using baseline-only scanner and
graph-independent targeted-VJP measurements, independently validated, and
pushed before intervention. It contains eight prompts and 169 unique
prompt/source/target pair identities. Norway/v4 remained a development pilot
and was excluded from all Stage 1D metrics.

The runtime remained the pinned Gemma 3 270M model at
`9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`, the 18-layer 16K PLT subset at
transcoder revision `fada11860ac1d337c1e41e9da308798405b94c8e`, NNsight
0.6.1, PyTorch 2.6.0, and MPS/BF16. There was no CPU model-compute fallback,
network access, revision change, or historical intervention-outcome input.

## Production-path evidence

Before the scientific boundary, the frozen production path passed a synthetic
worker-to-journal-to-assembler-to-validator rehearsal and a real Norway
active-feature MPS/BF16 rehearsal. The real rehearsal used pair identities
disjoint from all evaluation pairs, executed one no-op and one nonzero
suppression, serialized two completed points, passed the standalone validator,
and created no Stage 1D scientific-attempt lock.

The single canonical attempt began at the first instrumented suppression call
on an evaluation pair. It used zero scientific retries and completed:

```text
instrumented evaluation API calls: 438
completed fsync-journal points:     438
serialized unique point rows:      438
prompt count:                        8
unique evaluated pair count:       169
```

The append-only journal contained 438 call-start and 438 completed-point
records. A fresh process rebuilt the final artifacts only from that journal;
a second standalone process independently reconstructed the metrics and
decision from serialized point rows. The exact mapping
`desired = (1 - requested_alpha) * baseline_source_activation`, applied BF16
activation, realized suppression, and strict `z_i > tau_i` crossing rule were
validated at point level.

## Full-ablation discovery

All precision figures use the preregistered denominator of four per prompt;
this is a method-enriched panel diagnostic, not population AUPRC.

| Method | Crossings | Pooled precision@4 | Prompt-bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| S | 27/32 | 0.84375 | [0.78125, 0.93750] |
| Margin-only | 16/32 | 0.50000 | [0.37500, 0.62500] |
| Influence-only | 32/32 | 1.00000 | [1.00000, 1.00000] |
| Random-positive | 10/32 | 0.31250 | [0.18750, 0.40625] |

Paired prompt-level differences used 10,000 frozen prompt-bootstrap resamples:

| Contrast | Estimate | 95% interval |
| --- | ---: | ---: |
| S − margin-only | +0.34375 | [+0.21875, +0.46875] |
| S − influence-only | −0.15625 | [−0.21875, −0.06250] |
| S − random-positive | +0.53125 | [+0.37500, +0.68750] |

Prompt-level precision@4 shows that S was stable but never exceeded the
influence-only panel:

| Prompt | S | Margin | Influence | Random |
| --- | ---: | ---: | ---: | ---: |
| P01 Sweden capital | 1.00 | 0.50 | 1.00 | 0.25 |
| P02 Canada capital | 0.75 | 0.25 | 1.00 | 0.50 |
| P03 Japan capital | 1.00 | 0.50 | 1.00 | 0.25 |
| P04 Japan currency | 0.75 | 0.25 | 1.00 | 0.25 |
| P05 oxygen symbol | 0.75 | 0.50 | 1.00 | 0.00 |
| P06 largest planet | 0.75 | 0.75 | 1.00 | 0.50 |
| P07 Hamlet author | 1.00 | 0.75 | 1.00 | 0.25 |
| P08 water freezing | 0.75 | 0.50 | 1.00 | 0.50 |

Directional controls had 0/16 violations, fraction 0.0, with exact-binomial
95% interval [0.0, 0.20591].

## Critical suppression and movement

On the 12 monotonic, quantization-resolvable crossing pairs, predicted versus
observed critical suppression had Spearman 0.6941 with prompt-bootstrap 95%
interval [0.1897, 0.9562]. Median bracket distance was 0.1041, p95 bracket
distance was 0.2906, and median midpoint absolute error was 0.1069.

| Predicted-alpha bin | Pair count | Spearman | Median absolute error |
| --- | ---: | ---: | ---: |
| B1 `[0.02, 0.10)` | 6 | 1.0000 | 0.0617 |
| B2 `[0.10, 0.40)` | 2 | 1.0000 | 0.0561 |
| B3 `[0.40, 0.95]` | 4 | 0.8165 | 0.2730 |

The detailed panel's mean movement-sign agreement was 0.6475. Point-level
median symmetric normalized error was 1.0944 and p95 was 2.0. Ten of 32
detailed positive pairs showed nonmonotonic gates. Across selected execution
pairs, the frozen quantization audit recorded 43 collapsed requested values
and 51 predicted-alpha BF16 no-ops; these audit counts are not calibration
successes and were not hidden by schedule substitution.

## Frozen decision and interpretation

The independently reconstructed decision inputs were:

```text
S beats margin-only by >= 0.10:       true
S beats influence-only by >= 0.10:    false
critical calibration criterion:       false
directional-control criterion:        true
nonmonotonic-fraction criterion:       false
```

Accordingly, the frozen classifier returned
`retain_crossing_ranker_but_redesign_calibration`. S supplies value beyond
threshold margin alone, but the benchmark provides no evidence that dividing
inhibitory influence by margin improves full-ablation candidate selection over
influence alone. The influence-only panel was the strongest crossing ranker in
this benchmark. Critical-alpha calibration should be redesigned and tested on
a new preregistered experiment before scaling. The project should not proceed
directly to behavioral importance or mediation from this result.

## Validation, resource safety, and claim boundary

Before canonical execution, 9 focused Stage 1D/v4 tests passed; the complete
inherited offline suite passed with 598 passed, 8 skipped, and 1 deselected.
Ruff lint/format, strict scoped MyPy over 15 source files, dependency checks in
both used environments, import/compile smoke, diff checks, immutable-asset
verification, and staged secret/private-path scans passed. After execution,
the standalone hostile-input validator passed again with 9 verified checksums
and exact 438/438/438 call/journal/serialization equality.

Baseline prediction peaked at 2,865,414,144 bytes of MPS driver allocation and
1,783,660,544 bytes process RSS. The canonical intervention attempt peaked at
2,865,414,144 bytes MPS driver allocation and 966,033,408 bytes process RSS;
minimum available memory was 12,916,719,616 bytes. Swap growth was zero,
thermal state remained nominal, and no memory, telemetry, timeout, or safety
violation occurred.

The ten-file bundle is 2,346,107 bytes, every file is below 2 MiB, all nine
covered JSON checksums pass, and the content scan found no secret or private
absolute path. The result establishes a decision-relevant local MPS/BF16 gate
benchmark only. It does not establish behavioral importance, mediation, an
official BF16 reproduction, a reference CLT reproduction, or paper Results
readiness:

```text
behavioral_importance_result: none
mediation_result: none
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```
