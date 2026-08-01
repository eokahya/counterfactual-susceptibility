"""Strict, dependency-light configuration for the Stage 1A reproduction.

The typed model can be constructed from ordinary mappings without model-runtime
dependencies.  YAML support is deliberately optional and imported only by
``load_stage1a_config``.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Protocol, TextIO, TypeVar, cast

SCHEMA_VERSION = 1
OFFICIAL_UPSTREAM_REPOSITORY = "https://github.com/decoderesearch/circuit-tracer"
OFFICIAL_UPSTREAM_REVISION = "8f1e2438df612464e229e44c4a00ff637bf9379b"
OFFICIAL_MODEL_ID = "google/gemma-2-2b"
OFFICIAL_MODEL_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
OFFICIAL_TRANSCODER_ID = "mwhanna/gemma-scope-transcoders"
OFFICIAL_TRANSCODER_REVISION = "bd5773156dea09893636c801df1237d0410307d2"
OFFICIAL_ATTRIBUTION_PROMPT = "The capital of state containing Dallas is"
OFFICIAL_INTERVENTION_PROMPT = "Hecho: Michael Jordan juega al"

_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_REPOSITORY_ID_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
_PLACEHOLDER_PARTS = ("TO_BE_", "PLACEHOLDER", "<", ">")
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class Stage1AConfigError(ValueError):
    """Raised when a resolved Stage 1A configuration is invalid."""


class ConfigDependencyError(RuntimeError):
    """Raised when optional YAML loading was requested without PyYAML."""


class BackendName(StrEnum):
    """Backend accepted for the pinned official reproduction."""

    TRANSFORMERLENS = "transformerlens"


class DeviceName(StrEnum):
    """Explicit execution devices; ambiguous automatic selection is forbidden."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class DTypeName(StrEnum):
    """Explicit dtypes supported by the audited upstream interface."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class OffloadName(StrEnum):
    """Explicit attribution offload choices."""

    NONE = "none"
    CPU = "cpu"
    DISK = "disk"


class _YamlModule(Protocol):
    def safe_load(self, stream: TextIO) -> object:
        """Return parsed YAML data."""


def _as_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage1AConfigError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _expect_keys(value: Mapping[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise Stage1AConfigError(
            f"{context} keys are not exact; missing={missing}, unknown={unknown}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise Stage1AConfigError(f"{context} must be a non-empty trimmed string")
    if any(part in value for part in _PLACEHOLDER_PARTS):
        raise Stage1AConfigError(f"{context} contains a placeholder")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Stage1AConfigError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1AConfigError(f"{context} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise Stage1AConfigError(f"{context} must be finite")
    return result


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise Stage1AConfigError(f"{context} must be a boolean")
    return value


def _enum_value(enum_type: type[_EnumT], value: object, context: str) -> _EnumT:
    text = _string(value, context)
    try:
        return enum_type(text)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise Stage1AConfigError(
            f"{context} must be one of: {choices}; got {text!r}"
        ) from error


def _sha(value: object, context: str) -> str:
    revision = _string(value, context)
    if _SHA_PATTERN.fullmatch(revision) is None:
        raise Stage1AConfigError(
            f"{context} must be an exact 40-character lowercase hexadecimal SHA"
        )
    if revision == "0" * 40:
        raise Stage1AConfigError(f"{context} must not be the all-zero SHA")
    return revision


def _repository_id(value: object, context: str) -> str:
    repository_id = _string(value, context)
    if _REPOSITORY_ID_PATTERN.fullmatch(repository_id) is None:
        raise Stage1AConfigError(
            f"{context} must be an explicit owner/repository identifier without @"
        )
    return repository_id


def _relative_path(value: object, context: str) -> str:
    text = _string(value, context)
    if "\\" in text or text.startswith("~"):
        raise Stage1AConfigError(f"{context} must use repository-relative POSIX syntax")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != text
    ):
        raise Stage1AConfigError(
            f"{context} must be a normalized repository-relative path"
        )
    return path.as_posix()


def _require_under(path_text: str, prefix: tuple[str, ...], context: str) -> None:
    parts = PurePosixPath(path_text).parts
    if parts[: len(prefix)] != prefix:
        raise Stage1AConfigError(f"{context} must remain under {'/'.join(prefix)}/")


@dataclass(frozen=True, slots=True)
class PinnedUpstream:
    repository: str
    revision: str

    @classmethod
    def from_mapping(cls, value: object) -> PinnedUpstream:
        data = _as_mapping(value, "upstream")
        _expect_keys(data, {"repository", "revision"}, "upstream")
        repository = _string(data["repository"], "upstream.repository")
        revision = _sha(data["revision"], "upstream.revision")
        if repository != OFFICIAL_UPSTREAM_REPOSITORY:
            raise Stage1AConfigError("upstream.repository must be the audited project")
        if revision != OFFICIAL_UPSTREAM_REVISION:
            raise Stage1AConfigError("upstream.revision must equal the audited commit")
        return cls(repository=repository, revision=revision)


@dataclass(frozen=True, slots=True)
class PinnedAsset:
    identifier: str
    revision: str
    snapshot_path: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        context: str,
        required_identifier: str,
        required_revision: str,
    ) -> PinnedAsset:
        data = _as_mapping(value, context)
        _expect_keys(data, {"identifier", "revision", "snapshot_path"}, context)
        identifier = _repository_id(data["identifier"], f"{context}.identifier")
        revision = _sha(data["revision"], f"{context}.revision")
        snapshot_path = _relative_path(
            data["snapshot_path"], f"{context}.snapshot_path"
        )
        if identifier != required_identifier:
            raise Stage1AConfigError(
                f"{context}.identifier must be {required_identifier!r}"
            )
        if revision != required_revision:
            raise Stage1AConfigError(
                f"{context}.revision must equal the resolved immutable revision"
            )
        _require_under(
            snapshot_path,
            ("results", "generated", "stage1a", "assets"),
            f"{context}.snapshot_path",
        )
        return cls(
            identifier=identifier,
            revision=revision,
            snapshot_path=snapshot_path,
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    backend: BackendName
    device: DeviceName
    dtype: DTypeName

    @classmethod
    def from_mapping(cls, value: object) -> RuntimeConfig:
        data = _as_mapping(value, "runtime")
        _expect_keys(data, {"backend", "device", "dtype"}, "runtime")
        runtime = cls(
            backend=_enum_value(BackendName, data["backend"], "runtime.backend"),
            device=_enum_value(DeviceName, data["device"], "runtime.device"),
            dtype=_enum_value(DTypeName, data["dtype"], "runtime.dtype"),
        )
        if runtime.dtype is not DTypeName.BFLOAT16:
            raise Stage1AConfigError(
                "runtime.dtype must be bfloat16 for the official reproduction"
            )
        return runtime


@dataclass(frozen=True, slots=True)
class SeedConfig:
    python: int
    numpy: int
    torch: int

    @classmethod
    def from_mapping(cls, value: object) -> SeedConfig:
        data = _as_mapping(value, "seeds")
        _expect_keys(data, {"python", "numpy", "torch"}, "seeds")
        upper_bound = 2**32 - 1
        parsed = {
            key: _integer(data[key], f"seeds.{key}")
            for key in ("python", "numpy", "torch")
        }
        if any(seed > upper_bound for seed in parsed.values()):
            raise Stage1AConfigError(f"seeds must not exceed {upper_bound}")
        return cls(**parsed)


@dataclass(frozen=True, slots=True)
class AssetPolicy:
    allow_download: bool
    require_offline_execution: bool

    @classmethod
    def from_mapping(cls, value: object) -> AssetPolicy:
        data = _as_mapping(value, "asset_policy")
        _expect_keys(
            data,
            {"allow_download", "require_offline_execution"},
            "asset_policy",
        )
        allow_download = _boolean(data["allow_download"], "asset_policy.allow_download")
        require_offline = _boolean(
            data["require_offline_execution"],
            "asset_policy.require_offline_execution",
        )
        if not require_offline:
            raise Stage1AConfigError(
                "asset_policy.require_offline_execution must be true for a resolved run"
            )
        return cls(
            allow_download=allow_download,
            require_offline_execution=require_offline,
        )


@dataclass(frozen=True, slots=True)
class AttributionConfig:
    prompt: str
    max_n_logits: int
    desired_logit_probability: float
    max_feature_nodes: int
    batch_size: int
    offload: OffloadName

    @classmethod
    def from_mapping(cls, value: object) -> AttributionConfig:
        data = _as_mapping(value, "attribution")
        _expect_keys(
            data,
            {
                "prompt",
                "max_n_logits",
                "desired_logit_probability",
                "max_feature_nodes",
                "batch_size",
                "offload",
            },
            "attribution",
        )
        prompt = _string(data["prompt"], "attribution.prompt")
        max_n_logits = _integer(
            data["max_n_logits"], "attribution.max_n_logits", minimum=1
        )
        probability = _number(
            data["desired_logit_probability"],
            "attribution.desired_logit_probability",
        )
        max_feature_nodes = _integer(
            data["max_feature_nodes"],
            "attribution.max_feature_nodes",
            minimum=1,
        )
        batch_size = _integer(data["batch_size"], "attribution.batch_size", minimum=1)
        if prompt != OFFICIAL_ATTRIBUTION_PROMPT:
            raise Stage1AConfigError("attribution.prompt must match the official demo")
        if max_n_logits != 10 or probability != 0.95 or max_feature_nodes != 8192:
            raise Stage1AConfigError(
                "attribution scientific target parameters must be 10, 0.95, and 8192"
            )
        return cls(
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_probability=probability,
            max_feature_nodes=max_feature_nodes,
            batch_size=batch_size,
            offload=_enum_value(OffloadName, data["offload"], "attribution.offload"),
        )


@dataclass(frozen=True, slots=True)
class FeatureCoordinates:
    layer: int
    position: int
    feature_id: int

    @classmethod
    def from_mapping(cls, value: object) -> FeatureCoordinates:
        data = _as_mapping(value, "intervention.feature")
        _expect_keys(data, {"layer", "position", "feature_id"}, "intervention.feature")
        layer = _integer(data["layer"], "intervention.feature.layer")
        feature_id = _integer(data["feature_id"], "intervention.feature.feature_id")
        position_raw = data["position"]
        if isinstance(position_raw, bool) or not isinstance(position_raw, int):
            raise Stage1AConfigError("intervention.feature.position must be an integer")
        if (layer, position_raw, feature_id) != (20, -1, 341):
            raise Stage1AConfigError(
                "intervention.feature must match official coordinates (20, -1, 341)"
            )
        return cls(layer=layer, position=position_raw, feature_id=feature_id)


@dataclass(frozen=True, slots=True)
class InterventionConfig:
    prompt: str
    feature: FeatureCoordinates
    alphas: tuple[float, ...]
    freeze_attention: bool
    constrained_layers: None

    @classmethod
    def from_mapping(cls, value: object) -> InterventionConfig:
        data = _as_mapping(value, "intervention")
        _expect_keys(
            data,
            {
                "prompt",
                "feature",
                "alphas",
                "freeze_attention",
                "constrained_layers",
            },
            "intervention",
        )
        prompt = _string(data["prompt"], "intervention.prompt")
        if prompt != OFFICIAL_INTERVENTION_PROMPT:
            raise Stage1AConfigError("intervention.prompt must match the official demo")
        raw_alphas = data["alphas"]
        if isinstance(raw_alphas, (str, bytes)) or not isinstance(raw_alphas, Sequence):
            raise Stage1AConfigError("intervention.alphas must be a sequence")
        alphas = tuple(
            _number(alpha, f"intervention.alphas[{index}]")
            for index, alpha in enumerate(raw_alphas)
        )
        if alphas != (0.0, 0.5, 1.0):
            raise Stage1AConfigError(
                "intervention.alphas must be exactly [0.0, 0.5, 1.0]"
            )
        freeze_attention = _boolean(
            data["freeze_attention"], "intervention.freeze_attention"
        )
        if not freeze_attention:
            raise Stage1AConfigError(
                "the official intervention reproduction requires frozen attention"
            )
        if data["constrained_layers"] is not None:
            raise Stage1AConfigError(
                "the official intervention reproduction requires "
                "constrained_layers=null"
            )
        return cls(
            prompt=prompt,
            feature=FeatureCoordinates.from_mapping(data["feature"]),
            alphas=alphas,
            freeze_attention=freeze_attention,
            constrained_layers=None,
        )


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    raw_graph: str
    environment_manifest: str
    asset_manifest: str
    attribution_summary: str
    intervention_summary: str
    semantics_summary: str
    checksums: str

    @classmethod
    def from_mapping(cls, value: object) -> ArtifactPaths:
        data = _as_mapping(value, "artifacts")
        fields = {
            "raw_graph",
            "environment_manifest",
            "asset_manifest",
            "attribution_summary",
            "intervention_summary",
            "semantics_summary",
            "checksums",
        }
        _expect_keys(data, fields, "artifacts")
        paths = {key: _relative_path(data[key], f"artifacts.{key}") for key in fields}
        _require_under(
            paths["raw_graph"],
            ("results", "generated", "stage1a"),
            "artifacts.raw_graph",
        )
        if not paths["raw_graph"].endswith(".pt"):
            raise Stage1AConfigError("artifacts.raw_graph must use the .pt suffix")
        for key in fields - {"raw_graph", "checksums"}:
            _require_under(paths[key], ("results", "stage1a"), f"artifacts.{key}")
            if not paths[key].endswith(".json"):
                raise Stage1AConfigError(f"artifacts.{key} must use the .json suffix")
        _require_under(
            paths["checksums"], ("results", "stage1a"), "artifacts.checksums"
        )
        if not paths["checksums"].endswith(".sha256"):
            raise Stage1AConfigError("artifacts.checksums must use the .sha256 suffix")
        if len(set(paths.values())) != len(paths):
            raise Stage1AConfigError("artifact paths must be unique")
        return cls(**paths)


@dataclass(frozen=True, slots=True)
class Stage1AConfig:
    """Fully resolved configuration for the official Stage 1A reproduction."""

    schema_version: int
    experiment_name: str
    upstream: PinnedUpstream
    model: PinnedAsset
    transcoder: PinnedAsset
    runtime: RuntimeConfig
    seeds: SeedConfig
    asset_policy: AssetPolicy
    attribution: AttributionConfig
    intervention: InterventionConfig
    artifacts: ArtifactPaths

    @classmethod
    def from_mapping(cls, value: object) -> Stage1AConfig:
        data = _as_mapping(value, "configuration")
        _expect_keys(
            data,
            {
                "schema_version",
                "experiment_name",
                "upstream",
                "model",
                "transcoder",
                "runtime",
                "seeds",
                "asset_policy",
                "attribution",
                "intervention",
                "artifacts",
            },
            "configuration",
        )
        schema_version = _integer(data["schema_version"], "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise Stage1AConfigError(
                f"schema_version must equal {SCHEMA_VERSION}, got {schema_version}"
            )
        experiment_name = _string(data["experiment_name"], "experiment_name")
        if experiment_name != "stage1a_gemma2_2b_official_reproduction":
            raise Stage1AConfigError("experiment_name must identify the official run")
        return cls(
            schema_version=schema_version,
            experiment_name=experiment_name,
            upstream=PinnedUpstream.from_mapping(data["upstream"]),
            model=PinnedAsset.from_mapping(
                data["model"],
                context="model",
                required_identifier=OFFICIAL_MODEL_ID,
                required_revision=OFFICIAL_MODEL_REVISION,
            ),
            transcoder=PinnedAsset.from_mapping(
                data["transcoder"],
                context="transcoder",
                required_identifier=OFFICIAL_TRANSCODER_ID,
                required_revision=OFFICIAL_TRANSCODER_REVISION,
            ),
            runtime=RuntimeConfig.from_mapping(data["runtime"]),
            seeds=SeedConfig.from_mapping(data["seeds"]),
            asset_policy=AssetPolicy.from_mapping(data["asset_policy"]),
            attribution=AttributionConfig.from_mapping(data["attribution"]),
            intervention=InterventionConfig.from_mapping(data["intervention"]),
            artifacts=ArtifactPaths.from_mapping(data["artifacts"]),
        )


def load_stage1a_config(path: str | Path) -> Stage1AConfig:
    """Load and validate YAML, importing optional PyYAML only on demand."""

    try:
        yaml_module = cast(_YamlModule, importlib.import_module("yaml"))
    except ModuleNotFoundError as error:
        raise ConfigDependencyError(
            "PyYAML is required only to load a Stage 1A YAML file; install it in "
            "the dedicated Stage 1A environment"
        ) from error

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml_module.safe_load(stream)
    except OSError as error:
        raise Stage1AConfigError(
            f"unable to read Stage 1A configuration: {config_path}"
        ) from error
    return Stage1AConfig.from_mapping(raw)
