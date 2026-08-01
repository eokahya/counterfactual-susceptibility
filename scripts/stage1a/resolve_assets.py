#!/usr/bin/env python3
"""Resolve Stage 1A Hugging Face assets without downloading them by default.

The resolver talks only to official ``huggingface.co`` metadata and immutable
file endpoints.  It never includes credentials or local cache paths in its
JSON output.  Full snapshots are fetched only when ``--download`` is supplied
explicitly, and even then every request uses a verified 40-character commit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

HF_BASE_URL = "https://huggingface.co"
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CONFIG_BYTES = 64 * 1024
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
ARTIFACT_TYPE = "asset_manifest"
RUN_ID = "stage1a-assets-20260801-c5ebcd40-bd577315"
UPSTREAM_COMMIT = "8f1e2438df612464e229e44c4a00ff637bf9379b"


class ResolutionError(RuntimeError):
    """Raised when official metadata does not match the declared asset pin."""


@dataclass(frozen=True)
class AssetSpec:
    key: str
    repo_id: str
    revision: str
    role: str
    config_name: str | None = "config.yaml"

    def __post_init__(self) -> None:
        if not SHA_PATTERN.fullmatch(self.revision):
            raise ValueError(f"invalid immutable revision for {self.key}")
        if self.repo_id.count("/") != 1:
            raise ValueError(f"invalid Hugging Face repository ID for {self.key}")


MODEL = AssetSpec(
    key="model",
    repo_id="google/gemma-2-2b",
    revision="c5ebcd40d208330abc697524c919956e692655cf",
    role="official_stage1a_model",
    config_name="config.json",
)
TRANSCODER = AssetSpec(
    key="selected_transcoder",
    repo_id="mwhanna/gemma-scope-transcoders",
    revision="bd5773156dea09893636c801df1237d0410307d2",
    role="pinned_upstream_alias_target",
)
README_TRANSCODER = AssetSpec(
    key="readme_transcoder",
    repo_id="mntss/gemma-scope-transcoders",
    revision="9250a2d4860ce5ed5c96c14d5882b7d8162809a3",
    role="pinned_readme_comparison_only",
)
ORIGINAL_TRANSCODERS = AssetSpec(
    key="external_original_transcoders",
    repo_id="google/gemma-scope-2b-pt-transcoders",
    revision="50eec2f25c60545a9a74c1c3a26a0afdd0b4b872",
    role="current_source_of_readme_transcoder_references",
    config_name=None,
)
FUTURE_CLT = AssetSpec(
    key="future_clt",
    repo_id="mntss/clt-gemma-2-2b-426k",
    revision="b1e9ab376d07c90d780bf20b5fb1a0c89bd0f5e7",
    role="stage1b_metadata_candidate_only",
)

SPECS = (MODEL, TRANSCODER, README_TRANSCODER, ORIGINAL_TRANSCODERS, FUTURE_CLT)

MODEL_DOWNLOAD_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
TRANSCODER_DOWNLOAD_PATTERNS = ("config.yaml", "layer_*.safetensors")


def _repo_url(repo_id: str, revision: str | None = None) -> str:
    encoded_repo = urllib.parse.quote(repo_id, safe="/")
    url = f"{HF_BASE_URL}/api/models/{encoded_repo}"
    if revision is not None:
        url += f"/revision/{urllib.parse.quote(revision, safe='')}"
    return f"{url}?blobs=true"


def _file_url(spec: AssetSpec, filename: str) -> str:
    encoded_repo = urllib.parse.quote(spec.repo_id, safe="/")
    encoded_name = urllib.parse.quote(filename, safe="/")
    return f"{HF_BASE_URL}/{encoded_repo}/resolve/{spec.revision}/{encoded_name}"


def _request_bytes(
    url: str,
    *,
    limit: int,
    token: str | None = None,
) -> bytes:
    headers = {"Accept": "application/json", "User-Agent": "cfsus-stage1a/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = cast(bytes, response.read(limit + 1))
    except urllib.error.HTTPError as error:
        raise ResolutionError(f"official API returned HTTP {error.code}") from None
    except urllib.error.URLError:
        raise ResolutionError("official API request failed") from None
    if len(payload) > limit:
        raise ResolutionError("official API response exceeded the safety limit")
    return payload


def _request_json(url: str) -> Mapping[str, Any]:
    payload = _request_bytes(url, limit=MAX_METADATA_BYTES)
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ResolutionError("official API returned invalid JSON") from None
    if not isinstance(parsed, dict):
        raise ResolutionError("official API returned a non-object response")
    return parsed


def _request_text(url: str) -> str:
    payload = _request_bytes(url, limit=MAX_CONFIG_BYTES)
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        raise ResolutionError("immutable config was not UTF-8") from None


def _safe_file_metadata(raw: Mapping[str, Any]) -> dict[str, Any]:
    lfs = raw.get("lfs")
    lfs_map = lfs if isinstance(lfs, dict) else {}
    return {
        "name": raw.get("rfilename"),
        "size_bytes": raw.get("size"),
        "git_blob_id": raw.get("blobId"),
        "lfs_sha256": lfs_map.get("sha256"),
    }


def _safe_repo_metadata(spec: AssetSpec) -> dict[str, Any]:
    current = _request_json(_repo_url(spec.repo_id))
    pinned = _request_json(_repo_url(spec.repo_id, spec.revision))
    resolved_revision = pinned.get("sha")
    if resolved_revision != spec.revision:
        raise ResolutionError(f"immutable revision mismatch for {spec.key}")

    raw_siblings = pinned.get("siblings")
    if not isinstance(raw_siblings, list):
        raise ResolutionError(f"official API omitted files for {spec.key}")
    files = [
        _safe_file_metadata(item) for item in raw_siblings if isinstance(item, dict)
    ]
    sizes = [item["size_bytes"] for item in files]
    snapshot_bytes = sum(size for size in sizes if isinstance(size, int))
    return {
        "repo_id": spec.repo_id,
        "role": spec.role,
        "verified_revision": spec.revision,
        "current_revision": current.get("sha"),
        "current_matches_verified": current.get("sha") == spec.revision,
        "private": pinned.get("private"),
        "gated": pinned.get("gated"),
        "disabled": pinned.get("disabled"),
        "last_modified": pinned.get("lastModified"),
        "snapshot_file_bytes": snapshot_bytes,
        "files": files,
    }


def _quoted_hf_references(config: str) -> list[str]:
    references: list[str] = []
    for line in config.splitlines():
        stripped = line.strip()
        if not stripped.startswith('- "hf://') or not stripped.endswith('"'):
            continue
        references.append(stripped[3:-1])
    return references


def _file_names(asset: Mapping[str, Any]) -> set[str]:
    raw_files = asset.get("files")
    if not isinstance(raw_files, list):
        return set()
    return {
        str(item["name"])
        for item in raw_files
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }


def _comparison(assets: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    selected_config = _request_text(_file_url(TRANSCODER, "config.yaml"))
    readme_config = _request_text(_file_url(README_TRANSCODER, "config.yaml"))
    future_config = _request_text(_file_url(FUTURE_CLT, "config.yaml"))
    selected_files = _file_names(assets[TRANSCODER.key])
    readme_files = _file_names(assets[README_TRANSCODER.key])
    clt_files = _file_names(assets[FUTURE_CLT.key])
    external_refs = _quoted_hf_references(readme_config)
    return {
        "selected_transcoder": {
            "explicit_reference": f"{TRANSCODER.repo_id}@{TRANSCODER.revision}",
            "model_name": "google/gemma-2-2b",
            "model_kind": "transcoder_set",
            "feature_input_hook": "ln2.hook_normalized",
            "feature_output_hook": "hook_mlp_out",
            "direct_layer_safetensors": len(
                [name for name in selected_files if name.startswith("layer_")]
            ),
            "external_hf_references": _quoted_hf_references(selected_config),
            "transitively_immutable": True,
        },
        "readme_transcoder": {
            "explicit_reference": (
                f"{README_TRANSCODER.repo_id}@{README_TRANSCODER.revision}"
            ),
            "model_name": "google/gemma-2-2b",
            "model_kind": "transcoder_set",
            "feature_input_hook": "ln2.hook_normalized",
            "feature_output_hook": "hook_mlp_out",
            "direct_layer_safetensors": len(
                [name for name in readme_files if name.startswith("layer_")]
            ),
            "external_hf_reference_count": len(external_refs),
            "external_references_have_revisions": all(
                "?revision=" in reference for reference in external_refs
            ),
            "transitively_immutable": False,
        },
        "future_clt": {
            "explicit_reference": f"{FUTURE_CLT.repo_id}@{FUTURE_CLT.revision}",
            "model_name": "google/gemma-2-2b",
            "model_kind": "cross_layer_transcoder",
            "feature_input_hook": "hook_resid_mid",
            "feature_output_hook": "hook_mlp_out",
            "encoder_safetensors": len(
                [name for name in clt_files if name.startswith("W_enc_")]
            ),
            "decoder_safetensors": len(
                [name for name in clt_files if name.startswith("W_dec_")]
            ),
            "config_verified": "cross_layer_transcoder" in future_config,
            "metadata_only": True,
        },
    }


def _available_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            return token
    try:
        from huggingface_hub import get_token  # type: ignore[import-not-found]
    except ImportError:
        return None
    return cast(str | None, get_token())


def _probe_exact_config(token: str | None) -> dict[str, Any]:
    headers = {"User-Agent": "cfsus-stage1a/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        _file_url(MODEL, "config.json"), headers=headers, method="HEAD"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    except urllib.error.URLError:
        status = None
    return {
        "repo_id": MODEL.repo_id,
        "revision": MODEL.revision,
        "filename": "config.json",
        "authentication_present": token is not None,
        "http_status": status,
        "access_granted": status is not None and 200 <= status < 300,
        "response_body_recorded": False,
    }


def _download_selected(
    targets: Sequence[str], token: str | None
) -> list[dict[str, Any]]:
    if not targets:
        return []
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ResolutionError("--download requires huggingface_hub") from None

    results: list[dict[str, Any]] = []
    target_specs = {
        "model": (MODEL, MODEL_DOWNLOAD_PATTERNS),
        "transcoder": (TRANSCODER, TRANSCODER_DOWNLOAD_PATTERNS),
    }
    for index, target in enumerate(targets):
        spec, patterns = target_specs[target]
        try:
            snapshot_download(
                repo_id=spec.repo_id,
                revision=spec.revision,
                allow_patterns=list(patterns),
                token=token or False,
            )
        except Exception as error:  # third-party clients expose many exception types
            results.append(
                {
                    "target": target,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
            )
            for skipped_target in targets[index + 1 :]:
                skipped_spec, _ = target_specs[skipped_target]
                results.append(
                    {
                        "target": skipped_target,
                        "repo_id": skipped_spec.repo_id,
                        "revision": skipped_spec.revision,
                        "status": "skipped",
                        "reason": "earlier_requested_download_failed",
                    }
                )
            break
        else:
            results.append(
                {
                    "target": target,
                    "repo_id": spec.repo_id,
                    "revision": spec.revision,
                    "status": "completed",
                }
            )
    return results


def resolve_assets(downloads: Sequence[str] = ()) -> dict[str, Any]:
    """Resolve and validate pinned assets using official Hub metadata only."""

    assets = {spec.key: _safe_repo_metadata(spec) for spec in SPECS}
    token = _available_token()
    access_probe = _probe_exact_config(token)
    if downloads and not access_probe["access_granted"]:
        target_specs = {"model": MODEL, "transcoder": TRANSCODER}
        download_results = [
            {
                "target": target,
                "repo_id": target_specs[target].repo_id,
                "revision": target_specs[target].revision,
                "status": "skipped",
                "reason": "exact_model_access_not_granted",
            }
            for target in downloads
        ]
    else:
        download_results = _download_selected(downloads, token)
    download_failed = any(result["status"] == "failed" for result in download_results)
    if not access_probe["access_granted"]:
        status = "blocked"
        resolution_status = "blocked_by_model_access"
    elif download_failed:
        status = "failed"
        resolution_status = "requested_download_failed"
    else:
        status = "resolved"
        resolution_status = "immutable_assets_resolved"

    warnings = [
        "The README-listed transcoder repository is not transitively immutable "
        "because its external Hugging Face references omit revisions.",
        "The future 426K CLT is metadata-only and the pinned loader selects ReLU "
        "because its encoder files contain no threshold tensor.",
    ]
    if status == "blocked":
        warnings.append(
            "The exact Gemma model config probe did not grant access; no model "
            "snapshot was downloaded."
        )
    deviations = []
    if not downloads:
        deviations.append(
            "No complete asset snapshot was requested; this invocation resolved "
            "metadata only."
        )

    payload = {
        "resolution_status": resolution_status,
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "source": "official_hugging_face_hub_api",
        "default_mode": "metadata_only",
        "assets": assets,
        "transcoder_comparison": _comparison(assets),
        "access_probe": access_probe,
        "downloads_requested": list(downloads),
        "download_results": download_results,
        "metadata_range_audit": {
            "complete_weight_files_downloaded": bool(download_results)
            and all(result["status"] == "completed" for result in download_results),
            "completed_snapshot_targets": [
                result["target"]
                for result in download_results
                if result["status"] == "completed"
            ],
            "incomplete_snapshot_targets": [
                result["target"]
                for result in download_results
                if result["status"] != "completed"
            ],
            "mwhanna_safetensors_header_bytes": 424,
            "future_clt_range_request_bytes": 131072,
            "future_clt_approximate_tensor_payload_bytes": 130736,
            "disclosure": (
                "Earlier schema inspection used bounded HTTP Range requests. "
                "The two 64-KiB future-CLT prefixes included approximately "
                "127 KiB of leading tensor payload beyond their headers."
            ),
        },
        "safety": {
            "tokens_serialized": False,
            "local_paths_serialized": False,
            "future_clt_download_supported": False,
            "download_requires_explicit_flag": True,
        },
    }
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": RUN_ID,
        "status": status,
        "provenance": {
            "upstream_commit": UPSTREAM_COMMIT,
            "model_repo_id": MODEL.repo_id,
            "model_revision": MODEL.revision,
            "transcoder_repo_id": TRANSCODER.repo_id,
            "transcoder_revision": TRANSCODER.revision,
            "metadata_source": "official_hugging_face_hub_api",
        },
        "payload": payload,
        "warnings": warnings,
        "deviations": deviations,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="append",
        choices=("model", "transcoder"),
        default=[],
        help="Explicitly download one pinned Stage 1A snapshot; may be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON atomically to this path instead of standard output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = resolve_assets(args.download)
        if args.output is None:
            sys.stdout.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            _atomic_write_json(args.output, manifest)
    except Exception as error:
        sys.stderr.write(f"asset resolution failed ({type(error).__name__})\n")
        return 1
    return 0 if manifest["status"] in {"completed", "resolved"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
