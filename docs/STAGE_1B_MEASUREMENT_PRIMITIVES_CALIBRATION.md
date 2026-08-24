# Stage 1B measurement-primitives calibration

Status date: 2026-08-24

This is a calibration-only engineering record. It is not canonical evidence,
does not report Counterfactual Susceptibility, and does not establish a gate
crossing, behavior, mediation, reference CLT reproduction, or paper result.

## Frozen inputs

- exact Stage 1B base: `fb2fc158b45c842743804040e4e273776e666a48`
- model: `google/gemma-3-270m` at
  `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`
- PLT: `mwhanna/gemma-scope-2-270m-pt` at
  `fada11860ac1d337c1e41e9da308798405b94c8e`, subset
  `transcoder_all/width_16k_l0_small`
- circuit-tracer: 0.5.2 at
  `8f1e2438df612464e229e44c4a00ff637bf9379b`
- runtime: NNsight 0.6.1, PyTorch 2.6.0, native Apple MPS/BF16,
  CPython 3.11.13, fallback absent
- prompt: `The capital of France is`, token IDs
  `[2, 818, 5279, 529, 7001, 563]`

The offline asset verifier rehashed the exact allowlist: 2,087,816,677 bytes,
with no authentication, network access, or download.

## Calibration result

The scanner evaluated 18 layers × 5 non-BOS positions. Its bounded candidate
identity and order matched the ephemeral dense oracle exactly for chunk sizes
257, 1024, and 4096; bounded recall was 1.0 and no dense array was persisted.

The graph-reference stage found 8,413 eligible active pairs. It retained no
graph. Deterministic hash selection produced 16 calibration IDs and 64
disjoint canonical IDs with the required target-layer, target-position, and
edge-sign coverage. Targeted VJP read no graph edge.

At the predeclared edge floor 0.015625, all 16 calibration pairs were above the
floor:

| Metric | Calibration | Hard acceptance |
|---|---:|---:|
| Spearman | 1.000000 | ≥ 0.98 |
| Sign agreement | 1.000000 | ≥ 0.95 |
| Median symmetric normalized error | 0.0022785724 | ≤ 0.05 |
| p95 symmetric normalized error | 0.0042407438 | ≤ 0.20 |

The canonical endpoint-manifest digest is
`9879064f623be1cfdac4c8a1321f293e59e9e897ed7919608157a6e63e62082c`.
Canonical numeric edges and targeted responses were not present in the
calibration pair-freeze record.

The frozen canonical config SHA-256 is
`c68d5f5974a2d08b40519ad89834a5bbc37715e434bd267c3ede15affcf19369`;
the frozen artifact-schema SHA-256 is
`8a88695c17a85f22e28a2c2023c98d0190a2093dbbc8b0129f79ea896a797d05`.

## Safety evidence

- peak MPS current allocation: 641,208,064 bytes
- peak MPS driver allocation: 2,865,414,144 bytes
- peak process RSS: 1,322,450,944 bytes
- minimum available memory: 13,710,082,048 bytes
- swap growth: 0 bytes
- telemetry failures: 0
- thermal state: nominal

MPS and RSS are overlapping unified-memory signals and are not summed.

## Engineering failures before the passing calibration

The failed calibration attempts were not canonical scientific retries:

1. The supervisor initially refused to persist an unredacted local traceback.
   Its safe-tail redaction was fixed without changing measurement definitions.
2. The chunk adapter passed a full threshold vector to a chunk activation.
   It was corrected to call the pinned loaded JumpReLU function with the exact
   threshold slice and unchanged bandwidth.
3. Native MPS lacks `index_copy.out`; the unique result write was replaced by
   equivalent native-MPS `index_put_`, verified without fallback.
4. Targeted gradient proxies were read out of NNsight execution order. The
   reverse-layer access was aligned with the pinned upstream attribution
   context.
5. A passing calibration result was rejected at serialization because a tuple
   was not strict JSON. Capability evidence is now emitted as an explicit
   list, and the calibration was rerun rather than repaired by hand.

None of these changes altered the prompt, pair seed, pair counts, edge floor,
tolerances, scanner settings, mathematical definitions, model/PLT identity, or
runtime dtype/device. The final passing calibration output remained outside
the repository and is not canonical evidence.
