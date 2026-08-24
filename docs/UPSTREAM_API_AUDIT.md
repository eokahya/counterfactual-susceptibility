# Upstream API Audit: `decoderesearch/circuit-tracer`

**Status:** source-level audit complete for Stage 0; no model or transcoder weights were downloaded and no real-model call was run.

## 1. Audit pin and scope

- Official repository: <https://github.com/decoderesearch/circuit-tracer>.
- Inspected commit: [`8f1e2438df612464e229e44c4a00ff637bf9379b`](https://github.com/decoderesearch/circuit-tracer/commit/8f1e2438df612464e229e44c4a00ff637bf9379b), tagged `v0.5.2`.
- Upstream commit timestamp: `2026-07-17T20:46:35-07:00`.
- Inspection date: `2026-08-01`.
- Method: filtered source clone in a disposable `/tmp` directory; source, tests, demos, CI, packaging metadata, and README were inspected. No weights, datasets, or generated graphs were fetched.
- License: MIT; see [`LICENSE`, lines 1-19](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/LICENSE#L1-L19).

All GitHub links below are permanent links to this commit. Upstream warns that its API is under active development and may break ([`CONTRIBUTING.md`, lines 16-18](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/CONTRIBUTING.md#L16-L18)); the local adapter must therefore capability-check rather than assume later releases are compatible.

## 2. Installation and compatibility

The documented install is clone plus `pip install .` ([`README.md`, lines 22-23](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/README.md#L22-L23)); editable development install is `pip install -e ".[dev]"` ([`CONTRIBUTING.md`, lines 28-32](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/CONTRIBUTING.md#L28-L32)). A source-pinned future environment should install the audited commit, not an unpinned branch:

```bash
python -m pip install \
  "circuit-tracer @ git+https://github.com/decoderesearch/circuit-tracer.git@8f1e2438df612464e229e44c4a00ff637bf9379b"
```

This command was **not** run in Stage 0.

- Declared Python floor: `>=3.10` ([`pyproject.toml`, lines 1-7](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/pyproject.toml#L1-L7)).
- CI tests only Python 3.11 on `ubuntu-latest`; there is no version/OS matrix ([`.github/workflows/ci.yaml`, lines 7-24](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/.github/workflows/ci.yaml#L7-L24)). Thus 3.10+ is declared, but only 3.11 is demonstrated by CI.
- Runtime dependency bounds at the pin are: `einops>=0.8.0`, `huggingface_hub<1.0.0`, `ipykernel>=6.29.5,<7`, `ipywidgets>=8.1.7`, `nnsight==0.6.1`, `numpy>=1.24.0`, `pydantic>=2`, `safetensors>=0.5.0`, `seaborn>=0.13.2`, `tokenizers>=0.21.0`, `torch>=2.0.0`, `tqdm>=4.60.0`, `transformer-lens>=2.16.0`, and `transformers>=4.56.0,<=4.57.3` ([`pyproject.toml`, lines 7-22](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/pyproject.toml#L7-L22)). Development bounds are at [`pyproject.toml`, lines 24-25](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/pyproject.toml#L24-L25).
- This is not a lockfile: most dependencies have only lower bounds. Reproducing an empirical run requires a resolved environment lock in addition to the upstream commit.

## 3. Loading the model and transcoder

### Public entry points

The lazy top-level exports are `ReplacementModel`, `Graph`, and `attribute` ([`circuit_tracer/__init__.py`, lines 8-25](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/__init__.py#L8-L25)). The primary factory is:

```python
ReplacementModel.from_pretrained(
    model_name: str,
    transcoder_set: str,
    backend: Literal["nnsight", "transformerlens"] = "transformerlens",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
)
```

It calls `load_transcoder_from_hub(...)`, then routes through `from_pretrained_and_transcoders(...)` to `NNSightReplacementModel` or `TransformerLensReplacementModel` ([`replacement_model.py`, lines 24-69](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model.py#L24-L69), [`replacement_model.py`, lines 71-122](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model.py#L71-L122)). `load_transcoder_from_hub` parses `repo@revision`, reads `config.yaml`, records the revision and dispatches to a PLT `TranscoderSet` or `CrossLayerTranscoder` loader ([`hf_utils.py`, lines 20-44](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/hf_utils.py#L20-L44), [`hf_utils.py`, lines 47-118](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/hf_utils.py#L47-L118), [`hf_utils.py`, lines 121-202](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/hf_utils.py#L121-L202)). Local cached/path loading is also supported.

The concrete dictionary loaders are `load_transcoder` / `load_transcoder_set` for per-layer transcoders ([`single_layer_transcoder.py`, lines 447-492](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L447-L492), [`single_layer_transcoder.py`, lines 564-644](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L564-L644)) and `load_clt` for a cross-layer transcoder ([`cross_layer_transcoder.py`, lines 391-447](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L391-L447)). GemmaScope-specific conversion loaders are also present ([`single_layer_transcoder.py`, lines 408-444](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L408-L444), [`single_layer_transcoder.py`, lines 495-561](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L495-L561), [`cross_layer_transcoder.py`, lines 450-549](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L450-L549)).

### Identifier and revision findings

- Official examples use `model_name="google/gemma-2-2b"` and `transcoder_set="gemma"` ([`tests/test_interventions.py`, lines 16-24](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/tests/test_interventions.py#L16-L24)).
- The code maps shortcut `gemma` to the **unrevisioned** `mwhanna/gemma-scope-transcoders`, and `llama` to `mntss/transcoder-Llama-3.2-1B` ([`hf_utils.py`, lines 87-92](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/hf_utils.py#L87-L92)).
- The README instead lists Gemma-2 PLTs as `mntss/gemma-scope-transcoders` ([`README.md`, lines 43-51](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/README.md#L43-L51)). This documentation/code mismatch must not be guessed away.
- A transcoder reference can be pinned with `repo@revision`, but the official shortcuts and examples do not pin it. Model and transcoder weight revisions were therefore **not verified or fixed by this source audit**. Stage 1 must choose exact Hugging Face commits, record them in configuration, and verify their tensor schema before downloading. The present `TO_BE_PINNED` placeholders are intentional.

## 4. Exact feature equation and parameter storage

For both PLTs and CLTs, upstream computes the encoder preactivation **after encoder bias**:

\[
z_{l,p,f}=W^{\mathrm{enc}}_{l,f,:}x_{l,p,:}+b^{\mathrm{enc}}_{l,f}.
\]

For a JumpReLU dictionary it then computes

\[
a_{l,p,f}=z_{l,p,f}\,\mathbf 1[z_{l,p,f}>\tau_{l,f}].
\]

The comparison is strictly `>`, not `>=`; equality is inactive. The implementation is `x * (x > threshold)` ([`activation_functions.py`, lines 11-15](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/activation_functions.py#L11-L15)), and the test explicitly makes values equal to threshold zero ([`test_activation_functions.py`, lines 16-27](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/tests/transcoder/test_activation_functions.py#L16-L27)). Its backward pass uses the same strict mask for the input derivative and a rectangular surrogate only for a trainable threshold derivative ([`activation_functions.py`, lines 17-35](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/activation_functions.py#L17-L35)); inference does not use that surrogate.

Storage and conventions:

- `JumpReLU.threshold` is a raw `nn.Parameter`; there is no `exp`, log-threshold transform, or offset in the activation module ([`activation_functions.py`, lines 38-51](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/activation_functions.py#L38-L51)). The comment saying “log-thresholds” in the PLT set loader is inconsistent with the actual loaders, which copy checkpoint `threshold` directly into `activation_function.threshold` ([`single_layer_transcoder.py`, lines 428-443](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L428-L443), [`single_layer_transcoder.py`, lines 530-560](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L530-L560)). Treat the raw stored tensor as `tau` after schema verification.
- PLT tensors are `W_enc[d_transcoder,d_model]`, `W_dec[d_transcoder,d_model]`, `b_enc[d_transcoder]`, and `b_dec[d_model]`; optional `W_skip[d_model,d_model]` ([`single_layer_transcoder.py`, lines 58-86](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L58-L86)). `SingleLayerTranscoder.encode(..., apply_activation_function=False)` returns `F.linear(input, W_enc, b_enc)`, proving that exposed preactivation includes `b_enc` ([`single_layer_transcoder.py`, lines 120-125](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L120-L125)). `b_dec` is added only during reconstruction ([`single_layer_transcoder.py`, lines 127-135](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L127-L135)).
- CLT tensors are `W_enc[n_layers,d_transcoder,d_model]`, `b_enc[n_layers,d_transcoder]`, `b_dec[n_layers,d_model]`, and one `W_dec[source_layer]` of shape `[d_transcoder,n_layers-source_layer,d_model]`; JumpReLU threshold is `[n_layers,1,d_transcoder]` for broadcasting ([`cross_layer_transcoder.py`, lines 35-47](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L35-L47), [`cross_layer_transcoder.py`, lines 83-125](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L83-L125)). `encode_layer(..., False)` likewise returns `W_enc x + b_enc` before the gate ([`cross_layer_transcoder.py`, lines 163-182](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L163-L182)).
- `b_dec` is a reconstruction baseline, not part of `z`, `tau`, or the per-feature intervention value. For CLTs it is added once per output layer after summing active decoder contributions ([`cross_layer_transcoder.py`, lines 284-301](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L284-L301)).

This matches the research specification’s intended `z_i` only if `z_i` is defined as the bias-inclusive encoder preactivation. The local project should make that convention explicit.

## 5. Tensor, indexing, dtype, and device contract

- A single-sequence activation cache has shape `[layer, token_position, feature_id] = [n_layers,n_pos,d_transcoder]`. PLT `TranscoderSet.compute_attribution_components` states and returns this sparse shape ([`single_layer_transcoder.py`, lines 341-385](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L341-L385)); CLT `encode_sparse` documents and builds the same order ([`cross_layer_transcoder.py`, lines 184-219](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L184-L219)). Sparse indices are therefore `(layer, position, feature)`.
- Low-level PLT encode accepts any leading dimensions before `d_model`; model caching supplies `[batch,position,d_model]` and squeezes the single batch dimension. `get_activations` finally stacks layers ([`replacement_model_transformerlens.py`, lines 275-341](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L275-L341)). The attribution setup is explicitly single-sequence and asserts one-dimensional token IDs ([`replacement_model_transformerlens.py`, lines 424-440](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L424-L440)).
- `Graph.active_features` has shape `[n_active,3]` with rows `(layer,position,feature_id)`. Graph nodes are active features, error nodes, input-token nodes, then logits; adjacency rows are targets and columns are sources ([`graph.py`, lines 46-68](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/graph.py#L46-L68)).
- The model stream is chosen by transcoder config strings `feature_input_hook` and `feature_output_hook`, commonly `hook_resid_mid` and `hook_mlp_out`; it is not a fourth cache axis ([`single_layer_transcoder.py`, lines 231-277](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L231-L277)). Stage 1 must record these hook names with each checkpoint.
- The unified loader defaults to `float32`, moves transcoders to the model’s device/dtype, and chooses CUDA when available, otherwise CPU ([`replacement_model.py`, lines 24-59](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model.py#L24-L59), [`replacement_model_transformerlens.py`, lines 165-178](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L165-L178), [`utils/__init__.py`, lines 5-7](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/__init__.py#L5-L7)). The CLI accepts fp32, bf16, and fp16 and defaults to fp32 ([`__main__.py`, lines 49-67](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/__main__.py#L49-L67)).
- Encoder input is explicitly cast to `W_enc.dtype`; caches and thresholds therefore use the loaded transcoder/model dtype rather than an independent scientific dtype ([`single_layer_transcoder.py`, lines 120-125](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L120-L125)). Candidate margins should be promoted to at least float32 locally before subtraction/ranking, while retaining the source dtype in metadata.
- `zero_positions` is forcibly set to zero in public activation caches: normally position 0, or the first four positions for Gemma-3-IT ([`replacement_model_transformerlens.py`, lines 169-172](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L169-L172), [`replacement_model_transformerlens.py`, lines 285-303](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L285-L303)). Public caches cannot recover the true preactivation at those ignored positions.

## 6. Active and inactive feature access

### Active activations

`model.get_activations(inputs, sparse=False, apply_activation_function=True)` returns `(logits, activation_cache)`. `sparse=True` converts the post-gate cache to a sparse COO tensor ([`replacement_model_transformerlens.py`, lines 314-341](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L314-L341); NNSight equivalent: [`replacement_model_nnsight.py`, lines 360-385](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L360-L385)). Attribution uses `encode_sparse`, so only nonzero features and their encoder/activation-scaled decoder vectors enter the graph ([`cross_layer_transcoder.py`, lines 323-350](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L323-L350)).

### Inactive preactivations

Inactive, bias-inclusive preactivations **are exposed** by:

```python
logits, z = model.get_activations(
    inputs,
    sparse=False,
    apply_activation_function=False,
)
z_i = z[layer, position, feature_id]
```

The caching hook passes `apply_activation_function=False` to `encode_layer`, whose implementation returns before gating ([`replacement_model_transformerlens.py`, lines 275-303](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L275-L303), [`single_layer_transcoder.py`, lines 120-125](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L120-L125)). This is sufficient for a small dictionary or small number of layers.

It is **not** sufficient for scalable candidate generation:

- `get_activations` materializes and stacks every feature for every layer/position. `sparse=True` is useful after a sparse gate, but applying COO conversion to ungated preactivations does not avoid the dense encoder result and will normally store almost every value ([`replacement_model_transformerlens.py`, lines 285-340](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L285-L340)).
- `encode_layer` limits computation to one layer, but still multiplies by the full layer encoder and returns all `d_transcoder` features ([`cross_layer_transcoder.py`, lines 176-182](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L176-L182)). There is no public feature-slice, threshold-margin top-k, feature chunk, or streaming callback.
- Lazy encoder loading saves resident weight memory but `_get_encoder_weights(layer_id)` still loads the entire layer’s encoder tensor ([`cross_layer_transcoder.py`, lines 137-161](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L137-L161)).
- Upstream `TopK` is an alternative activation nonlinearity that retains the largest `k` values; it is not a query primitive for JumpReLU margins ([`activation_functions.py`, lines 54-63](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/activation_functions.py#L54-L63)).

**Smallest safe Stage 1 adapter:** capture only the selected layer/position residual input using backend-native read-only hooks, then compute requested feature chunks as `F.linear(x, W_enc[chunk], b_enc[chunk])` and pair them with `threshold[chunk]`. This uses checkpoint tensors without monkey-patching upstream. It must be implemented separately for PLT and CLT lazy layouts and tested against `encode_layer(..., False)` on small tensors.

## 7. Attribution, virtual weights, and local responses

The public entry point is `attribute(prompt, model, ..., batch_size, max_feature_nodes, offload)` ([`attribute.py`, lines 20-60](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute.py#L20-L60)). It builds a graph only from baseline-active features: the sparse activation matrix is enumerated, and `Graph.active_features` receives those indices and values ([`attribute_transformerlens.py`, lines 134-185](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L134-L185), [`attribute_transformerlens.py`, lines 265-277](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L265-L277)). Inactive targets cannot be requested through this public graph API.

### What an upstream feature edge means

Upstream first multiplies every active source decoder by its activation `a_j` ([`cross_layer_transcoder.py`, lines 235-282](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L235-L282); PLT: [`single_layer_transcoder.py`, lines 174-201](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L174-L201)). For a target feature, attribution injects its encoder vector as the custom reverse-mode gradient direction ([`attribute_transformerlens.py`, lines 229-244](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L229-L244)) and contracts the resulting residual gradient with the activation-scaled source decoder ([`context_transformerlens.py`, lines 93-132](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/context_transformerlens.py#L93-L132)). Since graph rows are targets and columns are sources, for active features the stored direct edge is

\[
A_{i,j}=a_j\,J^{\mathrm{frozen}}_{ij},
\]

not the raw Jacobian entry. Under the research notation, `q_{j->i} = -A_{i,j}` and `J_ij = A_{i,j}/a_j` only when the target is represented, `a_j` is safely nonzero, and the identical freeze convention is desired. This scaling is a source-level derivation, not an upstream-named `J_ij` API.

### Freeze convention

Attribution is a prompt-local linearized model, not the full nonlinear Jacobian:

- TransformerLens permanently detaches attention patterns and all LayerNorm scale tensors, freezes parameters, and makes embeddings differentiable ([`replacement_model_transformerlens.py`, lines 189-213](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L189-L213)).
- The original MLP forward value is preserved, but its gradient is detached; only an optional learned linear skip path remains differentiable ([`replacement_model_transformerlens.py`, lines 215-250](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L215-L250)). Thus MLP nonlinearities are bypassed in attribution.
- NNSight applies the corresponding attention/LN detaches and skip-gradient construction ([`replacement_model_nnsight.py`, lines 256-288](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L256-L288)).
- Feature gates/active sets are not differentiated. They are selected once from the baseline sparse activation matrix. Upstream tests describe the fully frozen model as linear and test zero second derivatives ([`test_freeze_points_hessian.py`, lines 44-55](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/tests/test_freeze_points_hessian.py#L44-L55)).
- For models with final logit softcapping, `zero_softcap()` temporarily disables that nonlinearity ([`replacement_model_transformerlens.py`, lines 343-350](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L343-L350)). This is not automatically entered by `attribute`; experiments must declare whether the metric uses pre- or post-softcap logits.

### JVP/VJP and targeting capabilities

There is no public virtual-weight, raw Jacobian, JVP, VJP, or targeted-inactive-feature method. Internally, `AttributionContext.compute_batch` performs batched reverse-mode custom-gradient injections and returns direct-effect rows ([`context_transformerlens.py`, lines 168-232](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/context_transformerlens.py#L168-L232)); `batch_size` chunks active target rows and `max_feature_nodes` truncates them by influence ([`attribute_transformerlens.py`, lines 184-249](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L184-L249)). Those mechanisms do not admit inactive targets.

**Smallest safe Stage 1 response adapter:** for a selected inactive target, inject its encoder row as the target preactivation direction into a backend-local VJP over the same declared freeze convention, while contracting only selected active-source decoder directions. Prefer a local wrapper around public PyTorch/TransformerLens hooks; do not depend on `AttributionContext` private buffers. Cross-check active pairs against `Graph.adjacency_matrix / a_j`. A separate, explicitly named “full nonlinear intervention response” must be measured by finite differences; it is not the same object.

## 8. Intervention API and semantics

### Public methods and tuple format

Both backends implement:

```python
model.feature_intervention(
    inputs,
    interventions,                 # (layer, position, feature_id, value)
    constrained_layers=None,
    freeze_attention=True,
    apply_activation_function=True,
    sparse=False,
    return_activations=True,
) -> (logits, activation_cache_or_none)
```

The annotated tuple permits scalar/tensor layer and feature IDs, integer/slice/tensor positions, and scalar/tensor values ([`replacement_model_transformerlens.py`, lines 20-26](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L20-L26)); the public function and return contract are at [`replacement_model_transformerlens.py`, lines 736-789](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L736-L789). There is little explicit bounds/shape validation, so the local adapter must validate every index, value, and alpha.

### Exact meaning of `value`

`value` is an **absolute desired post-gate feature activation**, not a preactivation, threshold displacement, multiplier, decoder contribution, or residual vector. Upstream computes

\[
\Delta a_f=\text{value}-a_f^{\mathrm{current}},
\qquad
\Delta r=\Delta a_f W^{\mathrm{dec}}_f,
\]

then adds `Delta r` at the configured underlying model feature-output/MLP-output location. The calculation is explicit at [`replacement_model_transformerlens.py`, lines 648-689](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L648-L689), and the residual addition is at [`replacement_model_transformerlens.py`, lines 691-717](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L691-L717). NNSight performs the same decoder scaling and in-place addition ([`replacement_model_nnsight.py`, lines 681-738](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L681-L738)). For a CLT, that decoder delta is distributed over every remaining output layer; for a PLT it writes only at its own layer.

Even when `apply_activation_function=False`, that flag controls the **returned cache**. Before calculating the intervention delta, upstream re-applies the feature nonlinearity ([`replacement_model_transformerlens.py`, lines 648-669](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L648-L669)). Intervention values therefore remain post-gate.

The project’s suppression convention maps exactly as:

```python
desired_value = (1.0 - alpha) * baseline_post_gate_activation
intervention = (layer, position, feature_id, desired_value)
```

where the local adapter validates `0 <= alpha <= 1`. `alpha=1` is zero ablation; a positive value can force an inactive feature on without making its encoder preactivation cross threshold. Clamping a target to zero uses the same post-gate/decoder-delta mechanism.

### Underlying model versus “replacement model”

Despite the class name, the normal forward pass does not replace the MLP output by transcoder reconstruction. The wrapper returns `skip + (actual_mlp_output - skip).detach()`, numerically the actual MLP output, and only changes gradient flow ([`replacement_model_transformerlens.py`, lines 215-250](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L215-L250)). Attribution separately computes reconstruction error as `actual_mlp_output - transcoder_reconstruction` ([`replacement_model_transformerlens.py`, lines 442-472](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L442-L472)).

Therefore `feature_intervention` is best described as a **feature-coordinate decoder edit to the underlying LM residual computation**. There is no distinct public “replacement-model-only” feature intervention. With `constrained_layers=None`, downstream real MLPs, normalizations, and gates can recompute from the edited residual; with constrained layers, cached baseline feature-output/MLP-output tensors are restored and intervention deltas are added to the selected range ([`replacement_model_transformerlens.py`, lines 475-566](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L475-L566)).

There is no unified first-class residual-stream patching method in `circuit-tracer`. TransformerLens hooks and NNSight traces can patch arbitrary residual locations, but that is backend-specific upstream functionality and should be wrapped as a separate capability rather than conflated with feature replacement.

### What is frozen during interventions

- The default `freeze_attention=True` causes baseline attention patterns/attention locations to be cached and restored. Source code adds LayerNorm scale locations only when `constrained_layers` covers all model layers, and adds feature-output/MLP-output freezes whenever a constrained range is provided ([`replacement_model_transformerlens.py`, lines 492-536](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L492-L536); NNSight: [`replacement_model_nnsight.py`, lines 559-569](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L559-L569)).
- Consequently, the docstring phrase “frozen attention + LayerNorm (default)” is misleading: with no constrained range, the implementation freezes attention locations but does not add LayerNorm scale locations. Source behavior must take precedence.
- `constrained_layers=range(n_layers)` is the direct-effect-like regime: attention, all LayerNorm scales, and all feature-output/MLP-output locations are frozen to baseline, with decoder deltas inserted. A partial range freezes outputs in that range but LayerNorm scales only when the range covers every layer.
- For empirical “underlying-model response” sweeps, use `freeze_attention=False, constrained_layers=None` and label that convention. For comparison to upstream local attribution, separately run the fully constrained convention. Do not mix them in one calibration curve.

One capture caveat follows from hook order: the activation cache is encoded at a layer’s feature **input**, while the intervention decoder delta is added at its feature **output**. The intervened source’s own cache entry remains its encoder-derived current/baseline value rather than being overwritten with `value`; later-layer target entries can reflect propagated edits. Use the declared intervention value for the source and the returned cache only for downstream targets.

### Generation

`feature_intervention_generate` supports open-ended position slices and returns `(text, logits, activations)`; generation logits are `[seq_len,vocab]` with batch size one ([`replacement_model_transformerlens.py`, lines 791-860](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L791-L860)). With KV caching, only the new token is processed per step and numerical equality with a full forward pass is not guaranteed ([`replacement_model_transformerlens.py`, lines 827-836](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L827-L836)). Critical-alpha sweeps should use non-generation `feature_intervention` on fixed prompts.

## 9. Capturing logits and selected target states

For a fixed-prompt sweep, one call can return both full logits and all feature states:

```python
logits, z = model.feature_intervention(
    token_ids,
    [(j_layer, j_pos, j_feature, desired_value)],
    freeze_attention=False,
    constrained_layers=None,
    apply_activation_function=False,
    sparse=False,
    return_activations=True,
)
target_z = z[i_layer, i_pos, i_feature]
```

`feature_intervention` returns model logits and stacks the requested activation/preactivation cache ([`replacement_model_transformerlens.py`, lines 774-789](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_transformerlens.py#L774-L789)). NNSight directly saves `self.output.logits` and the cache ([`replacement_model_nnsight.py`, lines 793-822](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L793-L822)). For selected logit metrics, index the returned `[batch,position,vocab]` tensor and record whether softcapping is active.

The snippet illustrates semantics and was not executed. The Stage 1 adapter should derive `target_a` with its verified scalar JumpReLU helper and the selected raw threshold; it should not rely on checkpoint-shape-sensitive broadcasting through the full upstream activation module.

There is no public selected-target-only capture. `return_activations=False` avoids returning the full cache, but then no target state is returned. NNSight can internally limit activation computation to intervention layers when no cache is requested ([`replacement_model_nnsight.py`, lines 785-803](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L785-L803)); this is not a selected feature-state API. A Stage 1 adapter should hook/calculate only requested targets during sweeps.

## 10. Memory, performance, and backend limitations

- Backends are `transformerlens` (default) and `nnsight` ([`replacement_model.py`, lines 13-16](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model.py#L13-L16)). TransformerLens supports only architectures it implements; NNSight supports more Hugging Face models but is documented as experimental, slower, less memory-efficient, and potentially incomplete ([`README.md`, lines 59-62](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/README.md#L59-L62)).
- The README says Gemma-2 2B can run on a 15 GB Colab GPU, with more memory enabling less offload and larger backward batches ([`README.md`, lines 13-20](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/README.md#L13-L20)). This is not a guarantee for all dictionary widths or for dense inactive-preactivation scans.
- Several NNSight/offload tests explicitly require over 32 GB VRAM ([`tests/conftest.py`, lines 4-5](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/tests/conftest.py#L4-L5), [`test_offload.py`, lines 52-90](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/tests/test_offload.py#L52-L90)). Real tests are predominantly CUDA-gated; CPU correctness of small transcoder operations exists, but practical full-model CPU performance is not established.
- Attribution can batch target rows, limit active feature nodes, lazy-load encoders/decoders, and offload modules to CPU or temporary safetensors on disk ([`attribute.py`, lines 20-60](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute.py#L20-L60), [`disk_offload.py`, lines 31-80](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/utils/disk_offload.py#L31-L80)). These optimizations address weight/backward memory, not the full dense inactive preactivation tensor.
- The graph allocates a dense edge matrix whose row count is selected active features plus logits and whose columns include all selected feature/error/token/logit nodes ([`attribute_transformerlens.py`, lines 180-190](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/attribution/attribute_transformerlens.py#L180-L190)). This and repeated backward passes are likely attribution bottlenecks. Inactive-candidate discovery should not construct a full graph.
- Lazy loading requires circuit-tracer-compatible checkpoint layout; GemmaScope-2 native format disables lazy loading ([`cross_layer_transcoder.py`, lines 450-485](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/cross_layer_transcoder.py#L450-L485), [`single_layer_transcoder.py`, lines 495-528](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/transcoder/single_layer_transcoder.py#L495-L528)).

## 11. Source-level blockers and Stage 1 decisions

| Requirement | Upstream status at audited commit | Consequence / smallest safe next step |
|---|---|---|
| Bias-inclusive `z_i`, raw `tau_i`, strict gate | Verified | Directly usable after checkpoint schema/dtype check. |
| Inactive `z_i` | Public, but only full cache/all features | Add chunked layer/position encoder projection in local adapter. |
| Near-threshold top-k inactive candidates | Not public | Local chunked `(tau-z)` selection without full all-layer materialization. |
| Raw `J_ij` for inactive target | Not public | Add targeted VJP/finite-difference capability; validate on active pairs. |
| Existing graph edge | Only active targets; stores `a_j J_ij` | Do not call it raw `J`; for represented pairs `q=-edge`. |
| JVP/VJP/virtual-weight API | Internal reverse-mode buffers only | Wrap backend hooks locally; avoid private buffer dependency. |
| Suppression/clamp | Absolute post-gate decoder edit | Adapter maps `alpha` to `(1-alpha)a_j`; distinguish from preactivation clamp. |
| Underlying-model sweep | Available with `freeze_attention=False`, no constraints | Make this the declared nonlinear ground-truth regime. |
| Attribution-matched sweep | Full constrained range | Report separately; it freezes attention/LN/MLP outputs. |
| Replacement-only sweep | No distinct public mode | Do not claim one; upstream forward retains real MLP outputs. |
| Residual patching | No unified circuit-tracer method | Optional backend-specific capability only. |
| Selected target capture | No public target-only cache | Add target hook/projection to avoid dense all-feature sweep caches. |
| Exact model/dictionary revisions | Not pinned by examples/aliases | Block real run until immutable HF revisions and tensor schema are recorded. |

The central source-level blocker is therefore **not access to the mathematical preactivation itself**. It is scalable, targeted access to inactive margins and the absence of a public inactive-target local-response API. The scientific design in `docs/RESEARCH_SPEC.md` need not change, but the adapter must not equate upstream active-only graph edges with raw inactive-target Jacobians.

## 12. Minimal future commands (not executed in Stage 0)

After immutable model and transcoder revisions have been selected and access approved, the minimal upstream attribution shape is:

```python
import torch
from circuit_tracer import ReplacementModel, attribute

MODEL_ID = "google/gemma-2-2b"  # record exact HF commit separately
TRANSCODER_REF = "mwhanna/gemma-scope-transcoders@<HF_COMMIT>"

model = ReplacementModel.from_pretrained(
    MODEL_ID,
    TRANSCODER_REF,
    backend="transformerlens",
    dtype=torch.bfloat16,
)
with model.zero_softcap():
    graph = attribute(
        "The capital of France is",
        model,
        batch_size=256,
        max_feature_nodes=7500,
        offload=None,
    )
```

The minimal absolute feature intervention, with the research suppression mapping, is:

```python
_, baseline_a = model.get_activations(prompt, sparse=False)
a_j = baseline_a[j_layer, j_pos, j_feature]
desired = (1.0 - alpha) * a_j

logits, z = model.feature_intervention(
    prompt,
    [(j_layer, j_pos, j_feature, desired)],
    freeze_attention=False,
    constrained_layers=None,
    apply_activation_function=False,
    return_activations=True,
)
observed_z_i = z[i_layer, i_pos, i_feature]
```

These examples intentionally retain a model-revision warning and a transcoder revision placeholder. They document API shape only and are not reproducible empirical commands until both weight revisions, tokenizer revision, hook names, dtype, device, prompt token IDs, and upstream/local commits are locked.

## 13. Remaining uncertainties

1. The exact immutable Hugging Face revisions for the first Gemma-2 2B model and dictionary were not established because Stage 0 forbids weight downloads and upstream shortcuts are unpinned.
2. The README/code disagreement over the Gemma PLT repository must be resolved by inspecting the chosen repository metadata and `config.yaml` in Stage 1, before any large file download.
3. The scalar and batched behavior of tensor-valued intervention indices is weakly validated upstream; the local API should initially support validated scalar `(layer,position,feature)` references only.
4. No current public contract promises target-only activation capture, inactive-target VJPs, or feature-chunked encoders. These must remain capability failures until the local adapter implements and tests the narrow wrappers described above.
5. NNSight is explicitly experimental and its implementation uses wrapped/private module access for transcoder vectors ([`replacement_model_nnsight.py`, lines 711-713](https://github.com/decoderesearch/circuit-tracer/blob/8f1e2438df612464e229e44c4a00ff637bf9379b/circuit_tracer/replacement_model/replacement_model_nnsight.py#L711-L713)). TransformerLens should be the first adapter target for Gemma-2 2B unless the selected model/checkpoint forces NNSight.

## 14. Stage 1A-S Gemma 3 / NNsight addendum (2026-08-24)

This addendum is separate from the Gemma 2 / CLT audit above.

- The newest tagged official release containing explicit Gemma 3 NNsight tests
  is `circuit-tracer` v0.5.2 at
  `8f1e2438df612464e229e44c4a00ff637bf9379b`.
- `Gemma3ForCausalLM` maps `mlp.hook_in` to the output of
  `model.layers[i].pre_feedforward_layernorm` and `hook_mlp_out` to the output
  of `model.layers[i].post_feedforward_layernorm`.
- The selected `mwhanna` files use canonical `W_enc`, `W_dec`, `b_enc`,
  `b_dec`, and `activation_function.threshold` keys. Consequently the generic
  PLT loader supports lazy encoder and decoder access; the lower-case native
  GemmaScope-2 special loader and its non-lazy warning do not apply.
- Loaded JumpReLU is strict: `x * (x > threshold)`. Equality is inactive.
- NNsight intervention tuples carry an absolute post-gate activation. Upstream
  subtracts the current activation internally, decodes the delta, and applies
  it at the mapped MLP output. The project suppression mapping is therefore
  exactly `desired=(1-alpha)*baseline`, never a delta argument.
- NNsight defaults to CUDA unless a device is explicit. The audited constructor
  maps explicit `mps` to `device_map={"": "mps"}`. Stage 1A-S never uses the
  default.
- Upstream NNsight attribution calls MPS-unsupported `to_sparse()` and mixes
  CPU graph metadata with device indices. The isolated adapter retains dense
  scientific values and indices on MPS while storing only COO graph metadata
  on CPU. Tiny real-MPS equivalence tests observed zero activation error and
  `0.00390625` maximum reconstruction error, inside the frozen FP16 tolerance.
- The adapter was not used on a real attribution graph. The model-only gate
  failed first, so no NNsight replacement, attribution, or intervention claim
  is made.

Immutable Hugging Face identities were verified as
`google/gemma-3-270m@9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1` and
`mwhanna/gemma-scope-2-270m-pt@fada11860ac1d337c1e41e9da308798405b94c8e`.
The selected subfolder contains exactly 18 layer safetensors plus the runtime
`config.yaml`; feature-visualization data and other widths are excluded.

## 15. Stage 1A-S-BF16 recovery addendum (2026-08-24)

This addendum applies only to the separate native MPS/BF16 recovery class.

- The smallest valid dependency starting point remains native arm64 CPython
  3.11.13, PyTorch 2.6.0, NNsight 0.6.1, Transformers 4.57.3,
  Hugging Face Hub 0.36.2, safetensors 0.8.0, and circuit-tracer v0.5.2 at
  `8f1e2438df612464e229e44c4a00ff637bf9379b`. Source inspection showed no
  reason to upgrade. A new isolated venv reproduced all 118 lock entries,
  passed `pip check`, and retained the exact circuit-tracer VCS commit.
- The exact model safetensors header contains 236 BF16 tensors. The selected
  PLT layer headers each contain canonical `W_enc`, `W_dec`, `b_enc`, `b_dec`,
  and `activation_function.threshold` tensors stored as FP32. The canonical
  loader constructs/moves every PLT tensor to its explicit requested dtype;
  accepted execution requests BF16 and verifies the observed result.
- PyTorch 2.6 maps BF16 to native MPSGraph BF16 on supported macOS releases,
  but source support is not universal operator evidence. Every critical
  forward, indexing, gradient, sparse-boundary, and NNsight operation remains
  gated by a real MPS/BF16 probe.
- Transformers 4.57.3 performs three source-mandated FP32 subcomputations on
  the same MPS device: Gemma 3 RMSNorm accumulation/weight multiplication,
  rotary frequency/trigonometry, and attention softmax. Each casts back to the
  input/query dtype. These exceptions are enumerated in the BF16 config. No
  outer autocast, hidden FP16 conversion, or FP32 residual/model fallback is
  permitted.
- NNsight receives explicit `device_map={"": "mps"}` and the requested dtype
  from circuit-tracer; its defaults must never select CUDA/CPU implicitly.
  Transcoders are moved to the model device/dtype during replacement-model
  configuration.
- Native MPS `.to_sparse()` remains unsupported in the audited path. The prior
  FP16 runtime adapter cannot be reused verbatim because it hard-codes FP16,
  converts values to CPU FP32, and monkey-patches upstream objects at runtime.
  The BF16 path must use isolated local classes/functions: BF16 scientific
  values and dense reconstruction stay on MPS, only COO graph metadata moves
  to CPU, and attribution context/index placement is explicit. No installed
  third-party package is edited or monkey-patched.
- Upstream intervention consumes an absolute post-gate activation and computes
  the decoder delta internally. Baseline, repeat, no-op, half, and full must use
  the same frozen attention/constraint convention. The prior FP16 worker used
  different conventions for baseline and intervention; the BF16 path corrects
  that control before any accepted run.

Before the accepted execution commit, the audited local subclasses passed real
MPS/BF16 model, one-layer PLT, full 18-PLT, NNsight replacement, attribution,
and intervention engineering gates. The sparse adapter stored BF16 values and
COO coordinates on CPU while keeping direct-effect vectors and gradients on
MPS/BF16; adapter counters showed one component call, multiple batch calls,
CPU-only partial ranking, and zero runtime monkeypatches. Full loaded semantics
required each PLT layer's own threshold vector. All intervention controls used
the same `freeze_attention=true` convention and passed the exact absolute
activation `(1-alpha)*baseline` to upstream.

These are pre-accepted engineering results only. They establish capability for
freezing the accepted protocol; they do not establish the accepted pilot,
CUDA equivalence, PLT/CLT equivalence, reference reproduction, Counterfactual
Susceptibility, or paper Results evidence.
