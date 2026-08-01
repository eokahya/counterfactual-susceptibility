#!/usr/bin/env python3
"""Reproduce the pinned official ``circuit-tracer`` attribution example.

The model and transcoder are resolved at immutable Hugging Face revisions before
they are loaded.  Optional runtime dependencies are imported only after CLI and
configuration validation, so ``--help`` remains usable in the lightweight
Stage 0 environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
MODEL_ID = "google/gemma-2-2b"
MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
TRANSCODER_ID = "mwhanna/gemma-scope-transcoders"
TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"
OFFICIAL_PROMPT = "The capital of state containing Dallas is"
OFFICIAL_ARGUMENTS = {
    "max_n_logits": 10,
    "desired_logit_prob": 0.95,
    "max_feature_nodes": 8192,
    "batch_size": 256,
}

MODEL_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
TRANSCODER_REQUIRED_FILES = (
    "config.yaml",
    *(f"layer_{layer}.safetensors" for layer in range(26)),
)


class Stage1ABlocked(RuntimeError):
    """A prerequisite prevented an empirical run from starting or completing."""


@dataclass(slots=True)
class RuntimeBundle:
    """Loaded immutable runtime plus sanitized provenance."""

    model: Any
    torch: Any
    config: dict[str, Any]
    provenance: dict[str, Any]
    device: str
    dtype: str


def repository_root() -> Path:
    return REPOSITORY_ROOT


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        yaml: Any = importlib.import_module("yaml")
    except ModuleNotFoundError as exc:
        raise Stage1ABlocked(
            "PyYAML is unavailable in the Stage 1A environment"
        ) from exc

    if not path.is_file():
        raise Stage1ABlocked("resolved configuration file does not exist")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise Stage1ABlocked(
            f"resolved configuration could not be read: {type(exc).__name__}"
        ) from exc
    if not isinstance(loaded, dict):
        raise Stage1ABlocked("resolved configuration must be a YAML mapping")
    from cfsus.reproduction.config import Stage1AConfig, Stage1AConfigError

    try:
        Stage1AConfig.from_mapping(loaded)
    except Stage1AConfigError as exc:
        raise Stage1ABlocked(f"resolved configuration is invalid: {exc}") from exc
    return loaded


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise Stage1ABlocked(f"configuration section {name!r} must be a mapping")
    return value


def _value(mapping: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _require_sha(name: str, value: object, expected: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise Stage1ABlocked(f"{name} must be an exact 40-character revision")
    try:
        int(value, 16)
    except ValueError as exc:
        raise Stage1ABlocked(f"{name} must be hexadecimal") from exc
    if value != expected:
        raise Stage1ABlocked(f"{name} is {value}, expected pinned revision {expected}")
    return value


def validate_official_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate immutable provenance and the official scientific constants."""

    upstream_section = _mapping(config.get("upstream"), "upstream")
    runtime = _mapping(config.get("runtime"), "runtime")
    model = _mapping(config.get("model"), "model")
    transcoder = _mapping(config.get("transcoder"), "transcoder")
    attribution = _mapping(
        _value(config, "attribution", "official_attribution"), "attribution"
    )

    _require_sha(
        "upstream.revision", upstream_section.get("revision"), UPSTREAM_REVISION
    )
    if model.get("identifier") != MODEL_ID:
        raise Stage1ABlocked(f"model.identifier must be {MODEL_ID!r}")
    _require_sha("model.revision", model.get("revision"), MODEL_REVISION)
    if transcoder.get("identifier") != TRANSCODER_ID:
        raise Stage1ABlocked(
            f"transcoder.identifier must be {TRANSCODER_ID!r}; mutable aliases "
            "are forbidden"
        )
    _require_sha("transcoder.revision", transcoder.get("revision"), TRANSCODER_REVISION)

    implementation = runtime.get("backend")
    if implementation != "transformerlens":
        raise Stage1ABlocked(
            "the official reproduction requires the transformerlens backend"
        )

    prompt = _value(attribution, "prompt", "prompt_text")
    if isinstance(prompt, dict):
        prompt = prompt.get("text")
    if prompt != OFFICIAL_PROMPT:
        raise Stage1ABlocked(
            "attribution prompt differs from the pinned official notebook"
        )
    configured_arguments = {
        "max_n_logits": attribution.get("max_n_logits"),
        "desired_logit_prob": attribution.get("desired_logit_probability"),
        "max_feature_nodes": attribution.get("max_feature_nodes"),
        "batch_size": attribution.get("batch_size"),
    }
    for key, expected in OFFICIAL_ARGUMENTS.items():
        actual = configured_arguments[key]
        if actual != expected:
            raise Stage1ABlocked(
                f"attribution.{key} is {actual!r}, expected official value {expected!r}"
            )

    dtype = runtime.get("dtype")
    if dtype not in {"bfloat16", "bf16"}:
        raise Stage1ABlocked("official reproduction requires bfloat16")

    asset_policy = _mapping(config.get("asset_policy"), "asset_policy")
    if asset_policy.get("require_offline_execution") is not True:
        raise Stage1ABlocked("resolved execution must require offline model execution")
    return attribution


def _installed_upstream_revision() -> str:
    try:
        distribution = importlib.metadata.distribution("circuit-tracer")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Stage1ABlocked(
            "the pinned circuit-tracer distribution is not installed"
        ) from exc

    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise Stage1ABlocked("circuit-tracer has no direct_url.json provenance")
    try:
        direct_url = json.loads(raw)
        vcs_info = direct_url["vcs_info"]
        revision = vcs_info["commit_id"]
        requested_revision = vcs_info["requested_revision"]
        vcs = vcs_info["vcs"]
        source_url = direct_url["url"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise Stage1ABlocked(
            "circuit-tracer was not installed from a verifiable pinned VCS revision"
        ) from exc
    if (
        source_url != "https://github.com/decoderesearch/circuit-tracer.git"
        or requested_revision != UPSTREAM_REVISION
        or vcs != "git"
    ):
        raise Stage1ABlocked(
            "installed circuit-tracer provenance does not match the canonical "
            "audited Git revision"
        )
    return _require_sha(
        "installed circuit-tracer revision", revision, UPSTREAM_REVISION
    )


def _resolve_snapshot(
    *,
    identifier: str,
    revision: str,
    override: Path | None,
    allow_download: bool,
    required_files: tuple[str, ...],
) -> Path:
    if override is not None:
        snapshot = override.expanduser().resolve()
    else:
        try:
            huggingface_hub: Any = importlib.import_module("huggingface_hub")
        except ModuleNotFoundError as exc:
            raise Stage1ABlocked("huggingface_hub is unavailable") from exc
        try:
            snapshot = Path(
                huggingface_hub.snapshot_download(
                    repo_id=identifier,
                    revision=revision,
                    allow_patterns=list(required_files),
                    local_files_only=not allow_download,
                )
            ).resolve()
        except Exception as exc:
            mode = "download-enabled" if allow_download else "offline cache-only"
            raise Stage1ABlocked(
                f"could not resolve {identifier}@{revision} in {mode} mode: "
                f"{type(exc).__name__}"
            ) from exc

    if not snapshot.is_dir():
        raise Stage1ABlocked("resolved immutable snapshot is not a directory")
    return snapshot


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise Stage1ABlocked("asset manifest contains a duplicate JSON key")
        value[key] = item
    return value


def _asset_file_inventory(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load exact-file identities from the tracked rich asset manifest."""

    from cfsus.reproduction.artifacts import (
        ArtifactValidationError,
        validate_artifact_envelope,
    )

    artifacts = _mapping(config.get("artifacts"), "artifacts")
    manifest_path = repository_root() / str(artifacts["asset_manifest"])
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise Stage1ABlocked("rich asset manifest is missing or is a symlink")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
        validate_artifact_envelope(manifest, expected_type="asset_manifest")
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ArtifactValidationError,
    ) as exc:
        raise Stage1ABlocked(
            f"rich asset manifest is invalid: {type(exc).__name__}"
        ) from exc
    if not isinstance(manifest, dict):
        raise Stage1ABlocked("rich asset manifest must be a JSON object")
    provenance = manifest.get("provenance")
    payload = manifest.get("payload")
    if not isinstance(provenance, dict) or not isinstance(payload, dict):
        raise Stage1ABlocked("rich asset manifest has invalid metadata sections")
    expected_pins = {
        "model_repo_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transcoder_repo_id": TRANSCODER_ID,
        "transcoder_revision": TRANSCODER_REVISION,
    }
    if any(provenance.get(key) != value for key, value in expected_pins.items()):
        raise Stage1ABlocked("rich asset manifest does not match the runtime pins")
    assets = payload.get("assets")
    if not isinstance(assets, dict):
        raise Stage1ABlocked("rich asset manifest has no asset inventory")
    result: dict[str, dict[str, Any]] = {}
    expected_assets = {
        "model": (MODEL_ID, MODEL_REVISION),
        "selected_transcoder": (TRANSCODER_ID, TRANSCODER_REVISION),
    }
    for key, (repo_id, revision) in expected_assets.items():
        asset = assets.get(key)
        if not isinstance(asset, dict):
            raise Stage1ABlocked(f"rich asset manifest omits {key}")
        if (
            asset.get("repo_id") != repo_id
            or asset.get("verified_revision") != revision
        ):
            raise Stage1ABlocked(f"rich asset manifest has a pin mismatch for {key}")
        files = asset.get("files")
        if not isinstance(files, list) or any(
            not isinstance(item, dict) for item in files
        ):
            raise Stage1ABlocked(f"rich asset manifest has invalid files for {key}")
        names = [item.get("name") for item in files]
        if any(not isinstance(name, str) for name in names) or len(names) != len(
            set(names)
        ):
            raise Stage1ABlocked(f"rich asset manifest has duplicate files for {key}")
        result[key] = {
            str(item["name"]): item for item in files if isinstance(item, dict)
        }
    return result


def _stream_hash(path: Path, algorithm: str, *, git_blob: bool = False) -> str:
    digest = hashlib.new(algorithm)
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_snapshot_files(
    *,
    snapshot: Path,
    inventory: dict[str, Any],
    required_files: tuple[str, ...],
    role: str,
) -> int:
    """Hash every consumed file against the exact-revision Hub inventory."""

    missing_metadata = sorted(set(required_files) - set(inventory))
    if missing_metadata:
        raise Stage1ABlocked(f"rich asset manifest omits required {role} files")
    for filename in required_files:
        record = inventory[filename]
        if not isinstance(record, dict):
            raise Stage1ABlocked(f"invalid rich metadata for {role} file {filename}")
        path = snapshot / filename
        if not path.is_file():
            raise Stage1ABlocked(f"immutable {role} snapshot is missing {filename}")
        expected_size = record.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or path.stat().st_size != expected_size
        ):
            raise Stage1ABlocked(f"immutable {role} file size mismatch: {filename}")
        lfs_sha256 = record.get("lfs_sha256")
        git_blob_id = record.get("git_blob_id")
        if isinstance(lfs_sha256, str) and len(lfs_sha256) == 64:
            actual = _stream_hash(path, "sha256")
            expected = lfs_sha256
        elif isinstance(git_blob_id, str) and len(git_blob_id) == 40:
            actual = _stream_hash(path, "sha1", git_blob=True)
            expected = git_blob_id
        else:
            raise Stage1ABlocked(
                f"immutable {role} file has no usable hash: {filename}"
            )
        if actual != expected:
            raise Stage1ABlocked(f"immutable {role} file hash mismatch: {filename}")
    return len(required_files)


def _reject_unmanifested_snapshot_entries(
    *, snapshot: Path, inventory: dict[str, Any], role: str
) -> None:
    """Keep loader precedence from selecting an unverified override file."""

    expected_files = set(inventory)
    expected_directories = {
        parent
        for filename in expected_files
        for parent in Path(filename).parents
        if parent != Path(".")
    }
    for entry in snapshot.rglob("*"):
        relative = entry.relative_to(snapshot)
        name = relative.as_posix()
        if entry.is_dir() and not entry.is_symlink():
            if relative not in expected_directories:
                raise Stage1ABlocked(
                    f"immutable {role} snapshot contains an unmanifested directory: "
                    f"{name}"
                )
        elif name not in expected_files:
            raise Stage1ABlocked(
                f"immutable {role} snapshot contains an unmanifested entry: {name}"
            )


def _numeric_layer_files(snapshot: Path) -> dict[int, str]:
    expected_names = {f"layer_{layer}.safetensors" for layer in range(26)}
    actual_names = {path.name for path in snapshot.glob("layer_*.safetensors")}
    if actual_names != expected_names:
        raise Stage1ABlocked(
            "pinned transcoder snapshot must contain only canonical layer_0 through "
            "layer_25 safetensors"
        )
    return {layer: str(snapshot / f"layer_{layer}.safetensors") for layer in range(26)}


def _torch_device(torch: Any, configured: str) -> Any:
    device = configured.lower()
    fallback_value = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    fallback_enabled = fallback_value.strip().casefold() not in {
        "",
        "0",
        "off",
        "false",
        "no",
    }
    if device == "mps" and fallback_enabled:
        raise Stage1ABlocked(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled; hidden CPU fallback is forbidden"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise Stage1ABlocked("CUDA was requested but is unavailable")
    if device == "mps" and (
        not torch.backends.mps.is_built() or not torch.backends.mps.is_available()
    ):
        raise Stage1ABlocked(
            "MPS was requested but is unusable "
            f"(built={torch.backends.mps.is_built()}, "
            f"available={torch.backends.mps.is_available()})"
        )
    if device == "cpu":
        raise Stage1ABlocked(
            "full model execution on CPU is prohibited; use CUDA or a verified MPS path"
        )
    if device not in {"cuda", "mps"}:
        raise Stage1ABlocked(f"unsupported Stage 1A device {configured!r}")
    return torch.device(device)


def _probe_bfloat16(torch: Any, device: Any) -> None:
    """Fail before model loading unless the selected accelerator executes bfloat16."""

    if device.type == "cuda":
        supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if not supported:
            raise Stage1ABlocked("selected CUDA device does not support bfloat16")
    try:
        operand = torch.ones((2, 2), device=device, dtype=torch.bfloat16)
        result = operand @ operand
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("bfloat16 probe returned non-finite values")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
    except Exception as exc:
        raise Stage1ABlocked(
            f"{device.type} bfloat16 execution probe failed: {type(exc).__name__}"
        ) from exc


def load_runtime(
    config: dict[str, Any],
    *,
    allow_download: bool,
    model_snapshot: Path | None = None,
    transcoder_snapshot: Path | None = None,
) -> RuntimeBundle:
    """Load only exact immutable model/transcoder snapshots."""

    validate_official_config(config)
    installed_revision = _installed_upstream_revision()

    try:
        torch: Any = importlib.import_module("torch")
        yaml: Any = importlib.import_module("yaml")
        ReplacementModel: Any = importlib.import_module(
            "circuit_tracer"
        ).ReplacementModel
        load_transcoder_set: Any = importlib.import_module(
            "circuit_tracer.transcoder.single_layer_transcoder"
        ).load_transcoder_set
        transformers: Any = importlib.import_module("transformers")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer
    except ModuleNotFoundError as exc:
        raise Stage1ABlocked(
            f"optional model runtime import failed: {exc.name}"
        ) from exc

    model_section = _mapping(config["model"], "model")
    transcoder_section = _mapping(config["transcoder"], "transcoder")
    runtime = _mapping(config.get("runtime"), "runtime")
    configured_device = str(runtime.get("device"))
    device = _torch_device(torch, configured_device)
    dtype = torch.bfloat16
    _probe_bfloat16(torch, device)

    configured_model_path = repository_root() / str(model_section["snapshot_path"])
    configured_transcoder_path = repository_root() / str(
        transcoder_section["snapshot_path"]
    )
    model_path = _resolve_snapshot(
        identifier=MODEL_ID,
        revision=MODEL_REVISION,
        override=model_snapshot
        or (configured_model_path if configured_model_path.is_dir() else None),
        allow_download=allow_download,
        required_files=MODEL_REQUIRED_FILES,
    )
    transcoder_path = _resolve_snapshot(
        identifier=TRANSCODER_ID,
        revision=TRANSCODER_REVISION,
        override=transcoder_snapshot
        or (
            configured_transcoder_path if configured_transcoder_path.is_dir() else None
        ),
        allow_download=allow_download,
        required_files=TRANSCODER_REQUIRED_FILES,
    )
    inventories = _asset_file_inventory(config)
    _reject_unmanifested_snapshot_entries(
        snapshot=model_path,
        inventory=inventories["model"],
        role="model",
    )
    _reject_unmanifested_snapshot_entries(
        snapshot=transcoder_path,
        inventory=inventories["selected_transcoder"],
        role="transcoder",
    )
    model_verified_files = _verify_snapshot_files(
        snapshot=model_path,
        inventory=inventories["model"],
        required_files=MODEL_REQUIRED_FILES,
        role="model",
    )
    transcoder_verified_files = _verify_snapshot_files(
        snapshot=transcoder_path,
        inventory=inventories["selected_transcoder"],
        required_files=TRANSCODER_REQUIRED_FILES,
        role="transcoder",
    )
    # Network access, if explicitly granted, ends at immutable snapshot resolution.
    # Scientific execution below is always local-only.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config_path = transcoder_path / "config.yaml"
    if not config_path.is_file():
        raise Stage1ABlocked("pinned transcoder snapshot has no config.yaml")
    transcoder_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(transcoder_config, dict):
        raise Stage1ABlocked("transcoder config.yaml is not a mapping")
    if transcoder_config.get("model_name") not in {None, MODEL_ID}:
        raise Stage1ABlocked("transcoder config targets a different language model")
    if transcoder_config.get("model_kind") != "transcoder_set":
        raise Stage1ABlocked("official reproduction requires per-layer transcoders")

    layer_files = _numeric_layer_files(transcoder_path)
    transcoders = load_transcoder_set(
        layer_files,
        scan_name=f"{TRANSCODER_ID}@{TRANSCODER_REVISION}",
        feature_input_hook=transcoder_config["feature_input_hook"],
        feature_output_hook=transcoder_config["feature_output_hook"],
        activation=transcoder_config.get("activation"),
        k=transcoder_config.get("k"),
        device=device,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=bool(transcoder_section.get("lazy_decoder", True)),
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        model = ReplacementModel.from_pretrained_and_transcoders(
            model_name=MODEL_ID,
            transcoders=transcoders,
            backend="transformerlens",
            device=device,
            dtype=dtype,
            hf_model=hf_model,
            tokenizer=tokenizer,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    except Exception as exc:
        raise Stage1ABlocked(
            f"immutable runtime load failed: {type(exc).__name__}"
        ) from exc
    model.eval()

    if model.cfg.n_layers != 26:
        raise Stage1ABlocked("loaded model/transcoder layer count is not 26")
    expected_shapes = {
        "W_enc": (16384, 2304),
        "W_dec": (16384, 2304),
        "b_enc": (16384,),
        "b_dec": (2304,),
        "threshold": (16384,),
    }
    for layer_index in range(26):
        transcoder = model.transcoders[layer_index]
        tensors = {
            "W_enc": transcoder.W_enc,
            "W_dec": transcoder.W_dec,
            "b_enc": transcoder.b_enc,
            "b_dec": transcoder.b_dec,
            "threshold": transcoder.activation_function.threshold,
        }
        if type(transcoder.activation_function).__name__ != "JumpReLU":
            raise Stage1ABlocked("loaded transcoder activation is not JumpReLU")
        for tensor_name, expected_shape in expected_shapes.items():
            tensor = tensors[tensor_name]
            if tuple(tensor.shape) != expected_shape or tensor.dtype != dtype:
                raise Stage1ABlocked(
                    f"layer {layer_index} {tensor_name} schema is incompatible"
                )

    provenance = {
        "upstream_repository": "https://github.com/decoderesearch/circuit-tracer",
        "upstream_revision": installed_revision,
        "model_identifier": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transcoder_identifier": TRANSCODER_ID,
        "transcoder_revision": TRANSCODER_REVISION,
        "backend": "transformerlens",
        "device": str(device),
        "dtype": "bfloat16",
        "asset_integrity": {
            "verification": "exact_file_content_hashes_matched",
            "model_verified_files": model_verified_files,
            "transcoder_verified_files": transcoder_verified_files,
        },
        "dtype_probe": {
            "device_type": device.type,
            "dtype": "bfloat16",
            "operation": "2x2_matmul",
            "passed": True,
        },
    }
    return RuntimeBundle(
        model=model,
        torch=torch,
        config=config,
        provenance=provenance,
        device=str(device),
        dtype="bfloat16",
    )


def _safe_output(path: str, *, generated: bool) -> Path:
    root = repository_root()
    candidate = (root / path).resolve()
    allowed = (
        root / ("results/generated" if generated else "results/stage1a")
    ).resolve()
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        kind = "raw" if generated else "summary"
        raise Stage1ABlocked(
            f"{kind} output must stay under {allowed.relative_to(root)}"
        ) from exc
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _seed_runtime(config: dict[str, Any], torch: Any) -> int:
    seeds = _mapping(config.get("seeds"), "seeds")
    python_seed = int(seeds["python"])
    numpy_seed = int(seeds["numpy"])
    torch_seed = int(seeds["torch"])
    random.seed(python_seed)
    numpy: Any = importlib.import_module("numpy")
    numpy.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
        torch.cuda.reset_peak_memory_stats()
    return torch_seed


def _write_summary(path: Path, envelope: dict[str, Any]) -> None:
    from cfsus.reproduction.artifacts import write_json_atomic

    write_json_atomic(path, envelope)


def _graph_summary_signature(graph: Any, torch: Any, tokenizer: Any) -> dict[str, Any]:
    """Return every raw-graph field used by the committed small summary."""

    adjacency = graph.adjacency_matrix
    probabilities = graph.logit_probabilities.detach().cpu()
    if not bool(torch.isfinite(adjacency).all().item()) or not bool(
        torch.isfinite(probabilities).all().item()
    ):
        raise Stage1ABlocked("attribution graph contains non-finite summary values")
    token_ids = [int(value) for value in graph.input_tokens.detach().cpu().tolist()]
    tokens = [str(tokenizer.convert_ids_to_tokens(value)) for value in token_ids]
    probability_values = probabilities.tolist()
    if len(graph.logit_targets) != len(probability_values):
        raise Stage1ABlocked(
            "attribution logit targets and probabilities are misaligned"
        )
    logit_targets = [
        {
            "token_id": int(target.vocab_idx),
            "token": str(target.token_str),
            "probability_weight": float(probability),
        }
        for target, probability in zip(
            graph.logit_targets,
            probability_values,
            strict=True,
        )
    ]
    return {
        "token_ids": token_ids,
        "tokens": tokens,
        "logit_targets": logit_targets,
        "graph": {
            "adjacency_shape": [int(value) for value in adjacency.shape],
            "active_feature_count": len(graph.active_features),
            "selected_feature_count": len(graph.selected_features),
            "error_node_count": int(graph.cfg.n_layers * graph.n_pos),
            "input_node_count": int(graph.n_pos),
            "logit_node_count": len(graph.logit_targets),
            "nonzero_edge_count": int(torch.count_nonzero(adjacency).item()),
            "finite": True,
        },
    }


def reproduce_attribution(
    bundle: RuntimeBundle,
    *,
    raw_output: str | None = None,
    summary_output: str | None = None,
) -> dict[str, Any]:
    """Run the official attribution without inference mode (gradients are required)."""

    try:
        circuit_tracer: Any = importlib.import_module("circuit_tracer")
    except ModuleNotFoundError as exc:
        raise Stage1ABlocked("circuit-tracer is unavailable") from exc
    attribute = circuit_tracer.attribute
    Graph = importlib.import_module("circuit_tracer.graph").Graph

    config = bundle.config
    attribution_config = validate_official_config(config)
    artifacts = _mapping(config.get("artifacts"), "artifacts")
    raw_name = raw_output or str(artifacts["raw_graph"])
    summary_name = summary_output or str(artifacts["attribution_summary"])
    raw_path = _safe_output(raw_name, generated=True)
    summary_path = _safe_output(summary_name, generated=False)

    torch = bundle.torch
    seed = _seed_runtime(config, torch)

    offload = str(attribution_config.get("offload", "cpu"))
    if offload not in {"cpu", "disk", "None", "none"}:
        raise Stage1ABlocked(f"invalid attribution offload mode {offload!r}")
    offload_value = None if offload.lower() == "none" else offload

    started = time.perf_counter()
    try:
        graph = attribute(
            prompt=OFFICIAL_PROMPT,
            model=bundle.model,
            max_n_logits=OFFICIAL_ARGUMENTS["max_n_logits"],
            desired_logit_prob=OFFICIAL_ARGUMENTS["desired_logit_prob"],
            max_feature_nodes=OFFICIAL_ARGUMENTS["max_feature_nodes"],
            batch_size=OFFICIAL_ARGUMENTS["batch_size"],
            offload=offload_value,
            verbose=bool(attribution_config.get("verbose", True)),
        )
    except Exception as exc:
        raise Stage1ABlocked(
            f"official attribution failed: {type(exc).__name__}"
        ) from exc
    wall_seconds = time.perf_counter() - started

    adjacency = graph.adjacency_matrix
    if adjacency.numel() == 0 or len(graph.selected_features) == 0:
        raise Stage1ABlocked("attribution returned an empty graph")
    in_memory_summary = _graph_summary_signature(
        graph,
        torch,
        bundle.model.tokenizer,
    )
    graph.to_pt(str(raw_path))
    raw_digest = _sha256(raw_path)
    try:
        reloaded = Graph.from_pt(str(raw_path), map_location="cpu")
    except Exception as exc:
        raise Stage1ABlocked(
            f"saved attribution graph could not be reloaded: {type(exc).__name__}"
        ) from exc
    reloaded_summary = _graph_summary_signature(
        reloaded,
        torch,
        bundle.model.tokenizer,
    )
    digest_after_reload = _sha256(raw_path)
    if reloaded_summary != in_memory_summary or digest_after_reload != raw_digest:
        raise Stage1ABlocked("saved attribution graph failed regeneration checks")

    raw_relative = raw_path.relative_to(repository_root()).as_posix()
    payload = {
        "source_notebook": {
            "path": "demos/attribute_demo.ipynb",
            "code_cells_zero_based": [4, 6, 8],
        },
        "prompt": OFFICIAL_PROMPT,
        "token_ids": in_memory_summary["token_ids"],
        "tokens": in_memory_summary["tokens"],
        "parameters": {**OFFICIAL_ARGUMENTS, "offload": offload_value},
        "graph": in_memory_summary["graph"],
        "raw_validation": {
            "passed": True,
            "loader": "Graph.from_pt",
            "map_location": "cpu",
            "regenerated_summary_fields": reloaded_summary,
            "checksum_stable_after_reload": True,
        },
        "logit_targets": in_memory_summary["logit_targets"],
        "raw_artifact": {
            "path": raw_relative,
            "sha256": raw_digest,
            "size_bytes": raw_path.stat().st_size,
        },
        "timing": {
            "wall_seconds": wall_seconds,
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
        },
        "seed": seed,
        "classification": "exact",
        "claim_boundary": (
            "Upstream E0 reproduction only; this does not test Counterfactual "
            "Susceptibility or predict gate crossings."
        ),
    }
    run_id = "stage1a-official-attribution"
    from cfsus.reproduction.artifacts import make_artifact_envelope

    envelope: dict[str, Any] = make_artifact_envelope(
        artifact_type="attribution_summary",
        run_id=run_id,
        status="completed",
        provenance=bundle.provenance,
        payload=payload,
    )
    _write_summary(summary_path, envelope)
    return envelope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=repository_root()
        / "configs/stage1a_gemma2_2b_official_reproduction.yaml",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--model-snapshot", type=Path)
    parser.add_argument("--transcoder-snapshot", type=Path)
    parser.add_argument("--raw-output")
    parser.add_argument("--summary-output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_yaml(args.config.resolve())
        bundle = load_runtime(
            config,
            allow_download=args.allow_download,
            model_snapshot=args.model_snapshot,
            transcoder_snapshot=args.transcoder_snapshot,
        )
        reproduce_attribution(
            bundle,
            raw_output=args.raw_output,
            summary_output=args.summary_output,
        )
    except Stage1ABlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print("Stage 1A official attribution reproduction completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
