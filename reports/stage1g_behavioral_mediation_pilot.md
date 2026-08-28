# Stage 1G — Behavioral Mediation Pilot

## Terminal result

```text
inconclusive_runtime
```

The frozen target-to-behavior output-sensitivity engineering gate failed before
scientific prediction. This is not a positive, mixed, or negative behavioral
mediation result. No Stage 1G scientific prompt panel was selected and no
scientific source-suppression call occurred.

## Provenance and Git boundaries

- Source branch: `stage-1f-prospective-one-probe-confirmation`
- Exact base: `c1bb6a3bbab3de945767eded4503b17343ba88e6`
- Stage 1G branch: `stage-1g-behavioral-mediation-pilot`
- Protocol-freeze commit: `75ee2e2b12e174676a3fbf74360a132835ba0230`
- Protocol manifest SHA-256:
  `d2c36906a545718e7c8750ec7b0e30cea6cf8fbdb3b2de5bf2fbb5a38d30d4d0`
- The protocol commit was pushed and local/origin identity was verified before
  the engineering MPS run.
- Historical branches, historical artifacts, `main`, and `paper/` were not
  modified.

The deterministic prompt order was frozen as:

```text
G06 G10 G16 G01 G07 G15 G12 G17 G20 G13
G11 G09 G14 G08 G04 G05 G03 G18 G19 G02
```

No prompt in this order was evaluated for Stage 1G eligibility because the
required output-sensitivity gate stopped execution first. Consequently there
are no eligible prompt/token IDs, candidate pools, panel memberships,
crossing rates, behavioral-enrichment statistics, or scientific effects to
report.

## Offline and production-path preparation

- Immutable model/transcoder assets passed the existing exact-byte allowlist
  validator: 2,087,816,677 bytes; no download, authentication, or network
  access occurred.
- Model revision:
  `google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- Transcoder revision:
  `mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`
- Exact subset: `transcoder_all/width_16k_l0_small`
- Backend/device/dtype: NNsight 0.6.1, Apple MPS, BF16; CPU fallback disabled.
- Ruff passed, strict mypy passed for the new runtime/scientific modules, and
  the full offline suite passed with 610 tests and 4 expected hardware skips.
- The synthetic production worker → fsync journal → assembler → fresh-process
  hostile validator chain passed, including simultaneous source/target edits
  and detached point serialization.

## Deepest verified blocker

The sacrificial engineering prompt was:

```text
The capital of Norway is | answer " Oslo" | contrast " Stockholm"
```

Eight baseline-active sacrificial features were probed using the frozen
relative half-width `0.0625`. The independent output VJP returned finite,
nonconstant derivatives:

```text
 0.000965118408203125
 0.00083160400390625
 0.000762939453125
 0.000637054443359375
 0.000560760498046875
 0.00054931640625
-0.000530242919921875
 0.0005035400390625
```

Every BF16-realized finite secant was exactly `0.0`. The observed frozen-gate
metrics were therefore:

```text
all finite:                          true
sign agreement:                     0.0       (required >= 0.90)
Spearman:                            undefined (required >= 0.90)
median symmetric normalized error:  2.0       (required <= 0.05)
```

A single engineering-only diagnostic repetition on the same sacrificial
Norway setup reproduced the nonconstant VJP values and all-zero BF16 finite
responses. It did not use a G01–G20 scientific prompt or pair and did not start
the scientific attempt. The result identifies BF16 response quantization under
the frozen small-probe protocol as the deepest verified blocker; it does not
establish that the mathematical VJP is correct or incorrect.

The frozen tolerances, probe size, selection rule, scientific protocol, and
classifier were not changed after this observation. The specification
requires `inconclusive_runtime` at this gate, so real-runtime rehearsal,
baseline-only scientific prediction, prediction freeze, and canonical
intervention were not run.

## Call and safety accounting

- Stage 1G scientific baseline calls on G01–G20: `0`
- Stage 1G scientific source-suppression calls: `0`
- Scientific canonical attempts: `0`
- Scientific retries: `0`
- Prediction manifest: not created
- Canonical point journal/bundle: not created
- Telemetry safety violation emitted: none
- Canonical artifact validator: not applicable because the preflight gate
  stopped before a canonical bundle; the synthetic standalone validator passed.

## Scientific claim boundary and next decision

```text
behavioral_mediation_result: none
historical_stage1f_terminal_class: completed_stage1f_e1_not_supported
simple_critical_alpha_calibration: retired
official_bf16_reproduction: pending
reference_clt_reproduction: pending
paper_results_readiness: false
```

Do not interpret this runtime/preflight result as evidence for or against
behavioral mediation. A future, separately preregistered Goal would need to
design a BF16-resolvable output-sensitivity validation protocol before any
behavior-weighted scientific selection can be attempted. This Goal does not
reopen critical-alpha calibration and does not modify paper Results.
