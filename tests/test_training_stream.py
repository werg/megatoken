"""Distributed stream contracts with real prepared readers and thread collectives."""

from __future__ import annotations

import copy
import sqlite3
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from megatoken.data import (
    DedupConfig,
    DedupStore,
    Episode,
    Lineage,
    PhysicalRecord,
    PreparedCorpus,
    PreparedDataset,
    WindowConfig,
)
from megatoken.data import storage as storage_module
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.training.stream import MixtureWindowStream, WindowStream

WINDOWS = WindowConfig(max_window_tokens=4096, max_decisions=99, max_records=1)


class ThreadReduction:
    """A strict sum collective; a missing participant fails within five seconds."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.values = [None] * world_size
        self.arrived = threading.Barrier(world_size, timeout=5)
        self.copied = threading.Barrier(world_size, timeout=5)

    def callback(self, rank):
        def reduce(counts):
            self.values[rank] = dict(counts)
            self.arrived.wait()
            keys = self.values[0].keys()
            assert all(value.keys() == keys for value in self.values)
            result = {key: sum(value[key] for value in self.values) for key in keys}
            self.copied.wait()
            return result

        return reduce


def parallel(world_size, worker):
    reduction = ThreadReduction(world_size)
    with ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = [
            pool.submit(worker, rank, reduction.callback(rank)) for rank in range(world_size)
        ]
        return [future.result(timeout=20) for future in futures]


@pytest.fixture
def prepared(tmp_path):
    codec = ByteCodec.byte_fallback()
    store = DedupStore(
        tmp_path / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    )
    try:
        with store.transaction():
            for name, counts in {"alpha": (1, 3, 2), "beta": (2, 1, 4), "singleton": (2,)}.items():
                for episode_index, count in enumerate(counts):
                    records = []
                    for index in range(count):
                        scored = name != "beta" or index % 2 == 1
                        identity = f"{name}-{episode_index}:{index}"
                        records.append(
                            PhysicalRecord(
                                identity,
                                "assistant" if scored else "user",
                                identity.encode() + bytes([65 + index]) * (1 + 11 * episode_index),
                                turn_index=index,
                                logical_id=identity,
                                policy_mask=scored,
                                provenance="natural",
                            )
                        )
                    store.add(
                        Episode(
                            f"{name}-{episode_index}",
                            tuple(records),
                            Lineage("fixture", "v1", f"family:{name}-{episode_index}"),
                            stream=name,
                        ),
                        codec,
                    )
        store.freeze()
        root = tmp_path / "prepared"
        export_prepared(
            store, root, codec=codec, source_receipts=[], pipeline_fingerprint="stream-fixture"
        )
    finally:
        store.close()
    return root


def item_identity(item):
    if item is None:
        return None
    return {
        "episode": item.episode.episode_id,
        "stream": item.episode.stream,
        "index": item.window.index,
        "records": tuple(entry.record.record_id for entry in item.window.records),
        "prefix": tuple(entry.record.record_id for entry in item.window.prefix),
        "D": item.window.D,
        "I": item.window.I,
        "P": item.window.P,
    }


def test_actual_source_sampling_is_restartable_and_excludes_padding(prepared):
    from megatoken.training.stream import source_sampling_report

    codec = ByteCodec.byte_fallback()
    original = WindowStream(PreparedCorpus([prepared]), codec, WINDOWS)
    restarted = WindowStream(PreparedCorpus([prepared]), codec, WINDOWS)
    try:
        for _ in range(4):
            original.next()
        cursor = original.state_dict()
        assert source_sampling_report([cursor]) == {
            "source_windows": {"fixture": 4},
            "complete_history": True,
        }
        restarted.load_state_dict(cursor)
        assert item_identity(original.next()) == item_identity(restarted.next())
        assert restarted.state_dict() == original.state_dict()
        from dataclasses import asdict

        from megatoken.io import content_hash

        obsolete = content_hash(
            {
                "dataset": restarted.dataset.identity,
                "scope": restarted.dataset.scope,
                "window": asdict(WINDOWS),
                "packing_contract": "meridian-atomic-decisions-tail-v3",
            }
        )
        with pytest.raises(ValueError, match="dataset_fingerprint"):
            restarted.load_state_dict({**cursor, "dataset_fingerprint": obsolete})
        with pytest.raises(ValueError, match="dataset_fingerprint"):
            restarted.load_state_dict(
                {**cursor, "dataset_fingerprint": "unrelated-dataset-or-packing"}
            )
        bad = {**cursor, "sampled_source_windows": {"fixture": -1}}
        before = restarted.state_dict()
        with pytest.raises(ValueError, match="sampled source"):
            restarted.load_state_dict(bad)
        assert restarted.state_dict() == before
        legacy = dict(cursor)
        legacy.pop("sampled_source_windows")
        legacy.pop("sampled_history_complete")
        with pytest.raises(ValueError, match="Missing required sampled source history"):
            restarted.load_state_dict(legacy)
    finally:
        original.close()
        restarted.close()


def test_whole_episode_budget_counts_once_after_fragmented_windows_and_restart(prepared):
    stream = WindowStream(
        PreparedCorpus([prepared], stream="alpha"), ByteCodec.byte_fallback(), WINDOWS, seed=0
    )
    replacement = WindowStream(
        PreparedCorpus([prepared], stream="alpha"), ByteCodec.byte_fallback(), WINDOWS, seed=0
    )
    completed, records = [], []
    try:
        for _ in range(2):
            item = stream.next()
            records.append(item.episode.episode_id)
            if item.episode_end:
                completed.append(item.episode.episode_id)
        replacement.load_state_dict(stream.state_dict())
        while replacement.epoch == 0:
            item = replacement.next()
            if item is not None:
                records.append(item.episode.episode_id)
                if item.episode_end:
                    completed.append(item.episode.episode_id)
        assert len(records) == 6
        assert len(completed) == len(set(completed)) == 3
    finally:
        stream.close()
        replacement.close()


def test_inventory_separates_preparation_from_actual_source_sampling(prepared, tmp_path):
    from megatoken.data.inventory import source_inventory
    from megatoken.io import write_json

    checkpoint = tmp_path / "metadata.json"
    write_json(
        checkpoint,
        {
            "global_step": 2,
            "source_sampling": {"source_windows": {"fixture": 3}, "complete_history": True},
        },
    )
    report = source_inventory([], prepared=[prepared], checkpoints=[checkpoint])
    row = report["sources"]["fixture"]
    assert row["listed"] == row["access"] == []
    assert row["prepared"] and row["sampled"][0]["windows"] == 3
    assert {x["stream"] for x in row["prepared"]} == {"alpha", "beta", "singleton"}
    with pytest.raises(ValueError, match="locally checked"):
        source_inventory(
            [],
            access_receipts=[
                {
                    "source_id": "fixture",
                    "revision": "v1",
                    "result": "downloaded",
                    "checked_at": "2026-10-23",
                    "evidence": "some download claimed success",
                }
            ],
        )


def window_run(root, epochs=2, cursors=None):
    def worker(rank, reduce):
        stream = WindowStream(
            PreparedCorpus([root], stream="alpha"),
            ByteCodec.byte_fallback(),
            WINDOWS,
            seed=0,
            rank=rank,
            world_size=2,
            reduce_counts=reduce,
        )
        try:
            if cursors is not None:
                stream.load_state_dict(cursors[rank])
            trace = []
            while stream.epoch < epochs:
                before = stream.state_dict()
                item = stream.next()
                trace.append(
                    {"before": before, "item": item_identity(item), "after": stream.state_dict()}
                )
            return trace
        finally:
            stream.close()

    return parallel(2, worker)


def synchronized_coordinates(state):
    return tuple(state[key] for key in ("epoch", "block_start", "slot", "block_length"))


def test_unequal_episode_windows_visit_every_window_once_per_synchronized_epoch(prepared):
    ranks = window_run(prepared)
    assert len(ranks[0]) == len(ranks[1])
    expected = {
        (f"alpha-{episode}", index)
        for episode, count in enumerate((1, 3, 2))
        for index in range(count)
    }
    for left, right in zip(*ranks, strict=True):
        assert synchronized_coordinates(left["before"]) == synchronized_coordinates(right["before"])
        assert synchronized_coordinates(left["after"]) == synchronized_coordinates(right["after"])
    for epoch in range(2):
        visits = Counter(
            (step["item"]["episode"], step["item"]["index"])
            for trace in ranks
            for step in trace
            if step["before"]["epoch"] == epoch and step["item"] is not None
        )
        assert visits == Counter(dict.fromkeys(expected, 1))
    assert any(step["item"] is None for trace in ranks for step in trace)
    for trace in ranks:
        assert trace[-1]["after"]["epoch"] == 2
        assert trace[-1]["after"]["block_start"] == trace[-1]["after"]["slot"] == 0


def test_resume_inside_unassigned_rank_tail_preserves_next_windows_and_exposure(prepared):
    full = window_run(prepared)
    checkpoint_index = next(
        index
        for index, step in enumerate(full[0])
        if step["after"]["epoch"] == 0
        and step["after"]["block_start"] == 2
        and 0 < step["after"]["slot"] < step["after"]["block_length"]
    )
    cursors = [trace[checkpoint_index]["after"] for trace in full]
    assert full[1][checkpoint_index]["item"] is None
    resumed = window_run(prepared, cursors=cursors)
    for rank in range(2):
        assert resumed[rank] == full[rank][checkpoint_index + 1 :]


def test_cursor_rejects_scope_window_topology_and_invalid_positions_without_mutation(prepared):
    codec = ByteCodec.byte_fallback()
    stream = WindowStream(PreparedCorpus([prepared], stream="alpha"), codec, WINDOWS)
    try:
        stream.next()
        original = stream.state_dict()
        for overrides in (
            {"version": 1},
            {"rank": 1},
            {"world_size": 2},
            {"epoch": -1},
            {"epoch": 0.5},
            {"slot": True},
            {"block_start": 3},
            {"slot": 1, "block_length": 0},
        ):
            with pytest.raises(ValueError, match="cursor"):
                stream.load_state_dict(original | overrides)
            assert stream.state_dict() == original
        alternatives = [
            WindowStream(PreparedCorpus([prepared], split=None, stream="alpha"), codec, WINDOWS),
            WindowStream(
                PreparedCorpus([prepared], stream="alpha"), codec, replace(WINDOWS, max_records=2)
            ),
        ]
        for other in alternatives:
            try:
                with pytest.raises(ValueError, match="dataset_fingerprint"):
                    other.load_state_dict(original)
            finally:
                other.close()
    finally:
        stream.close()


def mixture_run(root, meter, steps=30, cursors=None, shares=None):
    shares = shares or {"alpha": 0.35, "beta": 0.65}

    def worker(rank, reduce):
        stream = MixtureWindowStream(
            [root],
            ByteCodec.byte_fallback(),
            WINDOWS,
            dict(reversed(list(shares.items()))) if rank else shares,
            meter,
            seed=0,
            rank=rank,
            world_size=2,
            reduce_counts=reduce,
        )
        try:
            if cursors is not None:
                stream.load_state_dict(cursors[rank])
            trace = []
            for _ in range(steps):
                before = stream.state_dict()
                item = stream.next()
                after = stream.state_dict()
                changed = [
                    name for name in shares if before["exposure"][name] != after["exposure"][name]
                ]
                assert len(changed) == 1
                trace.append(
                    {
                        "before": before,
                        "item": item_identity(item),
                        "after": after,
                        "choice": changed[0],
                    }
                )
            return trace
        finally:
            stream.close()

    return parallel(2, worker)


@pytest.mark.parametrize("meter", ["D", "I", "P"])
def test_mixture_uses_global_rank_charges_for_identical_choices_and_resume(prepared, meter):
    traces = mixture_run(prepared, meter)
    different_local_charges = False
    zero_charge_steps = 0
    for left, right in zip(*traces, strict=True):
        name = left["choice"]
        assert name == right["choice"]
        assert left["before"]["exposure"] == right["before"]["exposure"]
        assert left["after"]["exposure"] == right["after"]["exposure"]
        charges = [step["item"][meter] if step["item"] else 0 for step in (left, right)]
        different_local_charges |= charges[0] != charges[1]
        zero_charge_steps += sum(charges) == 0
        assert left["after"]["exposure"][name] - left["before"]["exposure"][name] == max(
            1, sum(charges)
        )
        for step in (left, right):
            if step["item"]:
                assert step["item"]["stream"] == name
        for stream_name in ("alpha", "beta"):
            assert synchronized_coordinates(
                left["after"]["streams"][stream_name]
            ) == synchronized_coordinates(right["after"]["streams"][stream_name])
    assert different_local_charges
    if meter == "P":
        assert zero_charge_steps
    checkpoint = 10
    resumed = mixture_run(
        prepared, meter, steps=19, cursors=[trace[checkpoint]["after"] for trace in traces]
    )
    assert resumed == [trace[checkpoint + 1 :] for trace in traces]


def test_nonempty_mixture_stream_smaller_than_world_size_uses_idle_ranks(prepared):
    traces = mixture_run(prepared, "D", steps=7, shares={"singleton": 1.0})
    assert all(step["item"] is None for step in traces[1])
    assert all(step["item"] is not None for step in traces[0])
    for left, right in zip(*traces, strict=True):
        assert left["after"]["exposure"] == right["after"]["exposure"]
        assert synchronized_coordinates(
            left["after"]["streams"]["singleton"]
        ) == synchronized_coordinates(right["after"]["streams"]["singleton"])


def test_one_rank_window_preparation_failure_reaches_every_rank(prepared):
    class UnavailableCodec:
        fingerprint = ByteCodec.byte_fallback().fingerprint

        def encode(self, payload):
            raise OSError("fixture tokenizer unavailable")

    def worker(rank, reduce):
        stream = WindowStream(
            PreparedCorpus([prepared], stream="alpha"),
            UnavailableCodec() if rank == 0 else ByteCodec.byte_fallback(),
            WINDOWS,
            rank=rank,
            world_size=2,
            reduce_counts=reduce,
        )
        try:
            before = stream.state_dict()
            with pytest.raises(ValueError, match="Distributed window preparation failed") as error:
                stream.next()
            assert stream.state_dict() == before
            return str(error.value), error.value.__cause__
        finally:
            stream.close()

    errors = parallel(2, worker)
    assert "fixture tokenizer unavailable" in errors[0][0]
    assert isinstance(errors[0][1], OSError)
    assert "another rank" in errors[1][0] and errors[1][1] is None


@pytest.mark.parametrize("overrides", [{"epoch": 1}, {"block_length": 99}])
def test_resumed_rank_cursor_drift_is_rejected_collectively(prepared, overrides):
    def worker(rank, reduce):
        stream = WindowStream(
            PreparedCorpus([prepared], stream="alpha"),
            ByteCodec.byte_fallback(),
            WINDOWS,
            rank=rank,
            world_size=2,
            reduce_counts=reduce,
        )
        try:
            cursor = stream.state_dict()
            stream.load_state_dict(cursor | overrides if rank else cursor)
            with pytest.raises(ValueError, match="differs across ranks"):
                stream.next()
        finally:
            stream.close()

    parallel(2, worker)


def test_mixture_restore_is_atomic_and_configuration_is_not_aliased(prepared):
    shares = {"alpha": 0.5, "beta": 0.5}
    stream = MixtureWindowStream(
        [prepared], ByteCodec.byte_fallback(), WINDOWS, shares, "P", seed=0, rank=0, world_size=1
    )
    try:
        shares["alpha"] = 0.1
        stream.next()
        original = stream.state_dict()
        assert original["mixture"] == {"alpha": 0.5, "beta": 0.5}
        modified = copy.deepcopy(original)
        modified["streams"]["alpha"]["epoch"] += 1
        modified["streams"]["beta"]["dataset_fingerprint"] = "changed"
        with pytest.raises(ValueError, match="dataset_fingerprint"):
            stream.load_state_dict(modified)
        assert stream.state_dict() == original
        modified["streams"]["beta"] = None
        with pytest.raises(ValueError, match="expected an object"):
            stream.load_state_dict(modified)
        assert stream.state_dict() == original
        for count in (-1, 0.5, float("nan"), True):
            modified = copy.deepcopy(original)
            modified["exposure"]["alpha"] = count
            with pytest.raises(ValueError, match="exposure"):
                stream.load_state_dict(modified)
            assert stream.state_dict() == original
        snapshot = stream.state_dict()
        snapshot["mixture"]["alpha"] = 0.25
        assert stream.state_dict() == original
    finally:
        stream.close()


def test_invalid_mixture_closes_opened_readers_and_rejects_invalid_meters(prepared, monkeypatch):
    opened = []

    def track_reader(*args, **kwargs):
        reader = PreparedDataset(*args, **kwargs)
        opened.append(reader)
        return reader

    monkeypatch.setattr(storage_module, "PreparedDataset", track_reader)
    with pytest.raises(ValueError, match="at least one episode"):
        MixtureWindowStream(
            [prepared],
            ByteCodec.byte_fallback(),
            WINDOWS,
            {"alpha": 0.5, "missing": 0.5},
            "D",
            seed=0,
            rank=0,
            world_size=1,
        )
    assert len(opened) == 2
    for reader in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            reader.connection.execute("SELECT 1")
    with pytest.raises(ValueError, match="finite and positive"):
        MixtureWindowStream(
            [prepared],
            ByteCodec.byte_fallback(),
            WINDOWS,
            {"alpha": float("nan")},
            "D",
            seed=0,
            rank=0,
            world_size=1,
        )
    with pytest.raises(ValueError, match="meter"):
        MixtureWindowStream(
            [prepared],
            ByteCodec.byte_fallback(),
            WINDOWS,
            {"alpha": 1.0},
            "Z",
            seed=0,
            rank=0,
            world_size=1,
        )


def test_a_block_encodes_its_episode_once_and_yields_exactly_the_planned_windows(
    prepared, monkeypatch
):
    from megatoken.data import iter_windows
    from megatoken.training import stream as stream_module

    codec = ByteCodec.byte_fallback()
    passes = []

    def counting(episode, codec_, config):
        passes.append(episode.episode_id)
        return iter_windows(episode, codec_, config)

    monkeypatch.setattr(stream_module, "iter_windows", counting)
    stream = WindowStream(PreparedCorpus([prepared], stream="alpha"), codec, WINDOWS, seed=0)
    try:
        seen = []
        for _ in range(6):  # all three episodes: 1 + 3 + 2 windows
            item = stream.next()
            seen.append(item)
        assert len(passes) == 3 and len(set(passes)) == 3  # one pass per episode, not two
        for item in seen:
            expected = list(iter_windows(item.episode, codec, WINDOWS))[item.window.index]
            assert item.window == expected
        # Resuming in the middle of a block is one pass too.
        cursor = stream.state_dict()
        passes.clear()
        resumed = WindowStream(PreparedCorpus([prepared], stream="alpha"), codec, WINDOWS, seed=0)
        resumed.load_state_dict(cursor)
        resumed.next()
        assert len(passes) == 1
        resumed.close()
    finally:
        stream.close()
