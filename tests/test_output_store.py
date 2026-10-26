import hashlib
from types import SimpleNamespace

import pytest
import torch

from megatoken.model.media import MediaOutputSpec
from megatoken.multitrack_service import _output_response
from megatoken.output_store import FileOutputStore


def test_immutable_scoped_chunk_reads_and_cleanup(tmp_path):
    store = FileOutputStore(tmp_path, max_bytes=32)
    payload = bytes(range(32))
    ref = store.put("session:4", payload)
    assert ref["sha256"] == hashlib.sha256(payload).hexdigest()
    assert store.put("session:4", payload) == ref
    other_ref = store.put("other", payload)
    assert other_ref["handle"] != ref["handle"]
    with pytest.raises(ValueError, match="this session"):
        store.read("other", ref["handle"], 0, 8)
    parts = [store.read("session:4", ref["handle"], i, 7) for i in range(0, len(payload), 7)]
    assert b"".join(parts) == payload
    with pytest.raises(ValueError, match="offset"):
        store.read("session:4", ref["handle"], len(payload), 1)
    with pytest.raises(ValueError, match="offset"):
        store.read("session:4", ref["handle"], -1, 1)
    with pytest.raises(ValueError, match="maximum chunk"):
        store.read("session:4", ref["handle"], 0, 1 << 21)
    store.delete_owner("session:4")
    with pytest.raises(ValueError, match="this session"):
        store.read("session:4", ref["handle"], 0, 1)


def test_store_detects_missing_or_modified_content(tmp_path):
    store = FileOutputStore(tmp_path)
    ref = store.put("session:revision", b"original")
    path = tmp_path / ref["sha256"]
    path.chmod(0o644)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        store.read("session:revision", ref["handle"], 0, 3)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        store.read("session:revision", ref["handle"], 0, 3)


def test_repeated_output_poll_reuses_cached_serialized_descriptor(tmp_path):
    store = FileOutputStore(tmp_path)
    writes = []
    original_put = store.put

    def put(owner, data):
        writes.append(data)
        return original_put(owner, data)

    store.put = put
    group = SimpleNamespace(
        tape=SimpleNamespace(
            tracks=("voice",),
            event_ids=("voice-1",),
            modalities=("audio",),
            valid=(True,),
            payloads=(b'{"kind":"media-sha256-v1"}',),
            emitted_ids=((),),
            terminal=(True,),
        ),
        media_outputs=(torch.arange(4, dtype=torch.float32)[None],),
    )
    cache = {}
    spec = MediaOutputSpec("audio", sample_count=4, sample_rate=24000)
    first = _output_response(group, (spec,), store, "session:7", cache)
    expected_ref = {**first[0]["media_ref"], "shape": list(first[0]["media_ref"]["shape"])}
    first[0]["media_ref"]["shape"][0] = 99
    second = _output_response(group, (spec,), store, "session:7", cache)
    assert len(writes) == 1
    assert second[0]["media_ref"] == expected_ref


def test_store_reservations_preflight_budget_and_close_cleans_custom_root(tmp_path):
    store = FileOutputStore(tmp_path, max_bytes=10)
    store.reserve("session:1", 8)
    with pytest.raises(ValueError, match="before generation"):
        store.reserve("session:2", 3)
    ref = store.put("session:1", b"12345678")
    with pytest.raises(ValueError, match="byte budget"):
        store.reserve("session:2", 3)
    with pytest.raises(ValueError, match="nonempty"):
        store.put("session:2", b"")
    store.close()
    assert not (tmp_path / ref["sha256"]).exists()
