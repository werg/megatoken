"""Integration checks for concurrent execution, source publication and delivery."""

from __future__ import annotations

import json
from dataclasses import replace
from threading import Event

import numpy as np
import torch
from test_multitrack_runtime import _runtime

from megatoken.controls import build_control_payload, separate_event_references
from megatoken.data.schema import canonical_json
from megatoken.environments import FIXTURE_TOOLS, SnapshotEnvironment, SnapshotToolAdapter
from megatoken.execution import SessionExecutor
from megatoken.generation.structure import ActionSpec
from megatoken.model.media import MediaOutputSpec
from megatoken.model.multitrack import ObservedEvent
from megatoken.multitrack_runtime import ConcurrentToolRunner
from megatoken.retrieval.collection import SourceCollection, text_item_id
from megatoken.retrieval.index import unit
from megatoken.teacher import PINNED_TEACHER
from megatoken.tool_schema import compile_tool_spec


def test_execution_keeps_deciding_while_an_unrelated_actual_read_is_blocked():
    _, runtime = _runtime()
    session = runtime.open(effort=1)
    entered, unblock = Event(), Event()
    environment = SnapshotEnvironment()
    initial = environment.reset(
        {"initial_files": {"left.txt": "42"}, "expected_files": {"left.txt": "42"}}
    )
    adapter = SnapshotToolAdapter(environment, initial.state_id)
    runner = ConcurrentToolRunner(max_workers=1)
    executor = SessionExecutor(runtime, session, max_selections=2, runner=runner)
    schema = {
        "name": "read",
        "parameters": {
            "type": "object",
            "const": {"path": "left.txt"},
            "properties": {"path": {"type": "string", "enum": ["left.txt"]}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }

    class BlockedRead:
        def execute_selected(self, **kwargs):
            entered.set()
            if not unblock.wait(10):
                raise TimeoutError("Read was not released")
            return adapter.execute_selected(**kwargs)

    try:
        selected = runtime.generate(
            session,
            tracks=("reader",),
            event_ids=("slow-io",),
            specs=(compile_tool_spec(json.dumps([schema]), max_tokens=128),),
            controls=(build_control_payload(tools_json=json.dumps([schema])),),
            effort=1,
        )
        assert all(selected.tape.valid)
        runtime.dispatch_tools(
            session,
            selected,
            runner,
            {"read": BlockedRead()},
            schemas={"read": schema["parameters"]},
        )
        assert entered.wait(5)
        result = executor.tick(effort=1, seed=13)
        assert result["status"] == "selected"
        assert runner.pending and not next(iter(runner.pending.values())).done()
        assert "result:0:slow-io" not in session.completed_events
    finally:
        unblock.set()
        executor.close()
        runtime.close()


def test_execution_keeps_deciding_while_an_unrelated_voice_render_is_blocked():
    _, runtime = _runtime()
    session = runtime.open(effort=1)
    entered, unblock = Event(), Event()
    render = runtime.generator.render_selected

    def blocked(selected):
        if selected.tape.event_ids == ("slow-voice",):
            entered.set()
            if not unblock.wait(10):
                raise TimeoutError("Voice was never released")
        return render(selected)

    runtime.generator.render_selected = blocked
    executor = SessionExecutor(runtime, session, max_selections=2)
    try:
        revision = runtime.start_generation(
            session,
            tracks=("voice",),
            event_ids=("slow-voice",),
            specs=(MediaOutputSpec("audio", sample_count=48, sample_rate=24000),),
            effort=1,
        )
        assert entered.wait(5)
        result = executor.tick(effort=1, seed=13)
        assert result["status"] == "selected"
        assert result["selection_revision"] > revision
        assert not session.rendering[revision].future.done()
    finally:
        unblock.set()
        executor.close()
        runtime.close()


def _await_render(runtime, session, revision):
    session.rendering[revision].future.result(timeout=10)
    return runtime.poll_generation(session, revision)


def test_full_duplex_publication_snapshot_reads_and_late_cancelled_render(tmp_path):
    _, runtime = _runtime()
    runtime.max_retained_outputs = 3
    session = runtime.open(effort=1)
    entered, unblock = Event(), Event()
    render = runtime.generator.render_selected

    def blocked_render(selected):
        if selected.tape.event_ids == ("blocked-voice",):
            entered.set()
            if not unblock.wait(10):
                raise TimeoutError("test did not release blocked renderer")
        return render(selected)

    runtime.generator.render_selected = blocked_render
    source_path = tmp_path / "collection"
    key = unit(np.arange(1, 769, dtype=np.float32))
    try:
        pending_revision = runtime.start_generation(
            session,
            tracks=("voice",),
            event_ids=("blocked-voice",),
            specs=(MediaOutputSpec("audio", sample_count=48, sample_rate=24000),),
            effort=1,
            generator=torch.Generator().manual_seed(11),
        )
        assert entered.wait(5)
        selected = session.rendering[pending_revision].selection
        runtime.observe(
            session,
            (
                ObservedEvent(
                    "user", "text", b"New input arrived during rendering", event_id="new-user-input"
                ),
            ),
            effort=1,
        )
        assert "new-user-input" in session.completed_events
        assert session.rendering[pending_revision].selection is selected

        with SourceCollection.create(source_path, name="live", teacher=PINNED_TEACHER) as writer:
            runner = ConcurrentToolRunner(max_workers=2)
            try:
                before_revision = writer.revision

                def prepare_source():
                    row = writer.add_text(b"newly prepared evidence", key, origin="fixture/live")
                    assert row is not None
                    writer.reindex(cells=1)
                    return writer.publication(row)

                action_id = runtime.prepare_source(
                    session,
                    track="source-worker",
                    event_id="source-ready-event",
                    runner=runner,
                    prepare=prepare_source,
                )
                runner.pending[action_id].result(timeout=10)
                assert "result:" + action_id not in session.completed_events
                assert "source-worker" not in [
                    track for _, tracks in session.events for track in tracks
                ]
                assert writer.revision > before_revision
                current = writer.search(key, 5)
                assert any(
                    hit.item_id == text_item_id(b"newly prepared evidence") for hit in current
                )
                old_snapshot = writer.search(key, 5, max_revision=before_revision)
                assert all(
                    hit.item_id != text_item_id(b"newly prepared evidence") for hit in old_snapshot
                )

                ready, failures = runtime.collect_tools(session, runner, effort=1)
                assert action_id in ready and not failures
                assert "result:" + action_id in session.completed_events
                assert session.events[-1][1] == ("source-worker",)
                published = json.loads(ready[action_id])
                assert published["source_ready"]["revision"] == writer.revision
                assert published["source_ready"]["item_id"] == text_item_id(
                    b"newly prepared evidence"
                )
                row = writer.find((published["source_ready"]["item_id"],))[
                    text_item_id(b"newly prepared evidence")
                ]
                assert writer.item(row).origin == "fixture/live"

                stopped = runtime.stop_delivery(session, pending_revision)
                assert stopped.status == "cancelled"
                assert not stopped.future.cancelled()
                unblock.set()
                stopped.future.result(timeout=10)
                assert runtime.poll_generation(session, pending_revision).status == "cancelled"
                assert pending_revision not in session.selected_groups
                runtime.release_generation(session, pending_revision)

                referenced = next(
                    event
                    for event in session.state.history_event_ids
                    if event.startswith("result:" + action_id + ":part:")
                )
                next_revision = runtime.start_generation(
                    session,
                    tracks=("answer",),
                    event_ids=("answer-after-source",),
                    specs=(ActionSpec(max_tokens=4),),
                    controls=(build_control_payload(reference_event_ids=(referenced,)),),
                    effort=1,
                    generator=torch.Generator().manual_seed(13),
                )
                next_group = _await_render(runtime, session, next_revision).result
                _, references = separate_event_references(next_group.tape.controls[0])
                assert references == (referenced,)
                assert referenced in session.completed_events
                assert next_revision in session.selected_groups
                runtime.delivered(session, next_revision, next_group.tape.event_ids)
                runtime.release_generation(session, next_revision)
                assert next_revision not in session.rendering
                assert next_revision not in session.selected_groups

                # These are actual external calls through the runtime dispatcher;
                # both bind to one immutable environment version and dispatch twice
                # must not create duplicate effects.
                env = SnapshotEnvironment()
                initial = env.reset({"initial_files": {"a": "0", "b": "0"}})
                adapter = SnapshotToolAdapter(env, initial.state_id)
                read_schema = next(t["parameters"] for t in FIXTURE_TOOLS if t["name"] == "read")
                tool_revision = runtime.start_generation(
                    session,
                    tracks=("reader-a", "reader-b"),
                    event_ids=("read-event-a", "read-event-b"),
                    specs=(ActionSpec(max_tokens=4), ActionSpec(max_tokens=4)),
                    effort=1,
                    generator=torch.Generator().manual_seed(12),
                )
                group = _await_render(runtime, session, tool_revision).result
                actions = (
                    canonical_json({"name": "read", "arguments": {"path": "a"}}),
                    canonical_json({"name": "read", "arguments": {"path": "b"}}),
                )
                crafted_tape = replace(
                    group.tape,
                    payloads=tuple(action.encode() for action in actions),
                    valid=(True, True),
                    modalities=("text", "text"),
                )
                group = replace(group, tape=crafted_tape)
                session.selected_groups[tool_revision] = group
                tools = {"read": adapter}
                schemas = {"read": read_schema}
                runtime.dispatch_tools(session, group, runner, tools, schemas=schemas)
                runtime.dispatch_tools(session, group, runner, tools, schemas=schemas)
                assert len(runner.pending) == 2
                for future in runner.pending.values():
                    future.result(timeout=10)
                results, failures = runtime.collect_tools(session, runner, effort=1)
                assert len(results) == 2 and not failures
                assert env.sequence == 2
                assert {
                    json.loads(value)["observation"]["content"] for value in results.values()
                } == {"0"}

            finally:
                unblock.set()
                runner.close()
    finally:
        unblock.set()
        runtime.close()
