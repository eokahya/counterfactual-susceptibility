"""Stage 1C-v3 publication-safe serialization helpers."""

from cfsus.stage1c_v3.serialization import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_DEPTH,
    SerializationError,
    detach_json,
    detached_sweep_copies,
    read_json_strict,
    write_json_new,
)
from cfsus.stage1c_v3.worker_result import (
    build_detached_worker_result,
    construct_then_cleanup,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_DEPTH",
    "SerializationError",
    "build_detached_worker_result",
    "construct_then_cleanup",
    "detach_json",
    "detached_sweep_copies",
    "read_json_strict",
    "write_json_new",
]
