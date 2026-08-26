from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from cfsus.stage1c_v2.serialization import (
    SerializationError,
    detach_json,
    detached_sweep_copies,
    read_json_strict,
    write_json_new,
)


def _sweeps() -> list[dict[str, object]]:
    return [
        {
            "pair_id": "a" * 64,
            "group": "primary",
            "points": [
                {
                    "realized_suppression": 0.0,
                    "requested_mappings": [
                        {
                            "requested_alpha": 0.0,
                            "metadata": {"labels": ["zero", "baseline"]},
                        }
                    ],
                    "nested": {"history": [{"stage": "grid", "values": [1, 2]}]},
                },
                {
                    "realized_suppression": 1.0,
                    "requested_mappings": [],
                    "nested": {"history": [{"stage": "grid", "values": [3, 4]}]},
                },
            ],
        },
        {
            "pair_id": "b" * 64,
            "group": "near_boundary",
            "points": [
                {
                    "realized_suppression": 0.0,
                    "requested_mappings": [],
                    "nested": {"history": []},
                }
            ],
        },
    ]


def _container_ids(value: object) -> set[int]:
    if isinstance(value, dict):
        result = {id(value)}
        for item in value.values():
            result.update(_container_ids(item))
        return result
    if isinstance(value, list):
        result = {id(value)}
        for item in value:
            result.update(_container_ids(item))
        return result
    return set()


def _root(tmp_path: Path) -> Path:
    """Use a canonical absolute test root (macOS /var is commonly aliased)."""

    return tmp_path.resolve()


def test_nested_sweeps_survive_working_clear_and_mutation() -> None:
    working = _sweeps()
    first, second = detached_sweep_copies(working)

    assert first == second == working
    assert first is not second
    assert _container_ids(first).isdisjoint(_container_ids(second))
    assert _container_ids(first).isdisjoint(_container_ids(working))
    assert _container_ids(second).isdisjoint(_container_ids(working))

    working[0]["points"] = []
    nested = working[1]["points"]
    assert isinstance(nested, list)
    nested.clear()
    working.clear()
    assert len(first) == 2
    assert len(second) == 2
    assert len(first[0]["points"]) == 2
    assert first == second

    first[0]["points"][0]["nested"]["history"].clear()
    assert second[0]["points"][0]["nested"]["history"]
    second[0]["points"][0]["nested"]["history"].append({"stage": "second-copy"})
    assert first[0]["points"][0]["nested"]["history"] == []


def test_strict_write_read_preserves_pair_ids_and_point_counts(tmp_path: Path) -> None:
    path = _root(tmp_path) / "sweeps.json"
    first, _ = detached_sweep_copies(_sweeps())
    digest = write_json_new(path, {"pairs": first})

    loaded = read_json_strict(path)
    assert isinstance(loaded, dict)
    pairs = loaded["pairs"]
    assert isinstance(pairs, list)
    assert all(isinstance(item, dict) for item in pairs)
    typed_pairs = cast(list[dict[str, Any]], pairs)
    assert [item["pair_id"] for item in typed_pairs] == [
        "a" * 64,
        "b" * 64,
    ]
    assert [len(item["points"]) for item in typed_pairs] == [2, 1]
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_writer_uses_canonical_strict_json_bytes(tmp_path: Path) -> None:
    path = _root(tmp_path) / "canonical.json"
    write_json_new(path, {"z": 1, "a": [True, None]})
    assert path.read_bytes() == b'{\n  "a": [\n    true,\n    null\n  ],\n  "z": 1\n}\n'
    assert read_json_strict(path) == {"a": [True, None], "z": 1}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"value": float("nan")}, "non-finite"),
        ({"value": float("inf")}, "non-finite"),
        (("tuple",), "unsupported"),
        ({"secret": "do-not-write"}, "sensitive"),
        # commit-safety: allow-test-fixture
        ({"note": "/Users/example/private/file.json"}, "private path"),
        ({"note": "line\nfeed"}, "control"),
    ],
)
def test_detachment_rejects_unsafe_values(value: object, message: str) -> None:
    with pytest.raises(SerializationError, match=message):
        detach_json(value)


def test_detachment_rejects_cycles() -> None:
    value: list[object] = []
    value.append(value)
    with pytest.raises(SerializationError, match="cyclic"):
        detach_json(value)


def test_reader_rejects_duplicate_keys_nonfinite_and_controls(tmp_path: Path) -> None:
    for name, content, message in (
        ("duplicate.json", '{"x": 1, "x": 2}', "duplicate"),
        ("nan.json", '{"x": NaN}', "non-finite"),
        ("control.json", '{"x": "\\u0001"}', "control"),
    ):
        path = _root(tmp_path) / name
        path.write_text(content, encoding="utf-8")
        with pytest.raises(SerializationError, match=message):
            read_json_strict(path)


def test_new_writer_rejects_existing_path_and_preserves_bytes(tmp_path: Path) -> None:
    root = _root(tmp_path)
    path = root / "existing.json"
    path.write_bytes(b"previous\n")
    with pytest.raises(SerializationError, match="already exists"):
        write_json_new(path, {"new": True})
    assert path.read_bytes() == b"previous\n"
    assert not tuple(root.glob(".existing.json.*.tmp"))


def test_new_writer_rejects_symlinked_parent_and_output(tmp_path: Path) -> None:
    root = _root(tmp_path)
    real_parent = root / "real"
    real_parent.mkdir()
    linked_parent = root / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SerializationError, match="symlinked parent"):
        write_json_new(linked_parent / "artifact.json", {"ok": True})

    output = root / "output.json"
    output.symlink_to(real_parent / "target.json")
    with pytest.raises(SerializationError, match="symlink output"):
        write_json_new(output, {"ok": True})


def test_new_writer_rejects_intermediate_symlinked_ancestor(tmp_path: Path) -> None:
    root = _root(tmp_path)
    real_parent = root / "real"
    (real_parent / "nested").mkdir(parents=True)
    linked_ancestor = root / "linked"
    linked_ancestor.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(SerializationError, match="symlinked parent"):
        write_json_new(linked_ancestor / "nested" / "artifact.json", {"ok": True})

    real_artifact = real_parent / "nested" / "readable.json"
    real_artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SerializationError, match="symlinked parent"):
        read_json_strict(linked_ancestor / "nested" / "readable.json")


def test_new_writer_and_reader_reject_hardlinks_and_special_files(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    source = root / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    hardlink = root / "hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(SerializationError, match=r"already exists|hardlinked"):
        write_json_new(hardlink, {"ok": True})
    with pytest.raises(SerializationError, match="single-link"):
        read_json_strict(hardlink)

    fifo = root / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(SerializationError):
        read_json_strict(fifo)


def test_failed_publication_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    path = root / "artifact.json"
    original_link = os.link

    def fail_link(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(SerializationError, match="publication failed"):
        write_json_new(path, {"ok": True})
    assert not path.exists()
    assert not tuple(root.glob(".artifact.json.*.tmp"))
    monkeypatch.setattr(os, "link", original_link)


def test_destination_race_never_overwrites_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    path = root / "raced.json"
    original_link = os.link

    def race_link(source: Any, destination: Any, **kwargs: Any) -> None:
        del kwargs
        Path(destination).write_bytes(b"race-winner\n")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", race_link)
    with pytest.raises(SerializationError, match="already exists"):
        write_json_new(path, {"should": "not win"})
    assert path.read_bytes() == b"race-winner\n"
    assert not tuple(root.glob(".raced.json.*.tmp"))


def test_depth_and_size_limits_are_fail_closed() -> None:
    nested: object = {"leaf": True}
    for _ in range(5):
        nested = [nested]
    with pytest.raises(SerializationError, match="depth"):
        detach_json(nested, maximum_depth=3)
    with pytest.raises(SerializationError, match="encoded JSON"):
        detach_json({"x": "a" * 100}, maximum_bytes=32)


def test_reader_rejects_private_and_sensitive_fields(tmp_path: Path) -> None:
    root = _root(tmp_path)
    for name, value, message in (
        ("secret.json", {"api_key": "abc"}, "sensitive"),
        # commit-safety: allow-test-fixture
        ("private.json", {"path": "file:///Users/emre/private"}, "private path"),
    ):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(SerializationError, match=message):
            read_json_strict(path)


def test_written_file_is_single_link_regular_and_durable(tmp_path: Path) -> None:
    path = _root(tmp_path) / "durable.json"
    write_json_new(path, {"ok": True})
    info = path.stat()
    assert stat.S_ISREG(info.st_mode)
    assert info.st_nlink == 1
