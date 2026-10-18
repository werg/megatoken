"""Small, actually executed dictionary tasks for offline integration checks."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from megatoken.io import atomic_binary_writer

from .schema import (
    Dependency,
    Episode,
    Lineage,
    LogicalAction,
    Supervision,
    canonical_json,
    digest_bytes,
    make_records,
    stable_id,
)
from .sources import SourceSpec, TermsReview, pin_local_source


def make_demo_episodes(
    count: int = 32, *, max_record_bytes: int = 128, stream: str = "bootstrap"
) -> Iterator[Episode]:
    """This is a smoke corpus, not evidence of public-source quality or scale."""
    if count <= 0:
        raise ValueError("Demo episode count must be positive")
    for index in range(count):
        episode_id = f"fixture-{index:06d}"
        key, value = f"key_{index}", f"Wert_{index}_é_🙂"
        task = f"Remember the exact value under {key}.".encode()
        arguments = {"key": key, "value": value}
        action_bytes = canonical_json({"name": "store.set", "arguments": arguments}).encode()
        action_id = stable_id("demo-action", episode_id, action_bytes)
        execution_id = stable_id("demo-execution", action_id)
        first = make_records(
            task,
            logical_id=episode_id + ":task",
            role="user",
            turn_index=0,
            provenance="natural",
            max_bytes=max_record_bytes,
        )
        action_records = make_records(
            action_bytes,
            logical_id=episode_id + ":action",
            role="assistant",
            turn_index=1,
            provenance="teacher",
            policy_mask=True,
            max_bytes=max_record_bytes,
            action_id=action_id,
        )
        # Execute the operation in fresh per-episode state and verify its result.
        state: dict[str, str] = {}
        parsed = __import__("json").loads(action_bytes)
        state[parsed["arguments"]["key"]] = parsed["arguments"]["value"]
        observed = canonical_json({"stored": state[key]}).encode()
        result = make_records(
            observed,
            logical_id=episode_id + ":result",
            role="tool",
            turn_index=2,
            provenance="executed",
            max_bytes=max_record_bytes,
            result_of=action_id,
            execution_id=execution_id,
        )
        followup = make_records(
            f"What exact value did the tool store in {key}?".encode(),
            logical_id=episode_id + ":followup",
            role="user",
            turn_index=3,
            provenance="natural",
            max_bytes=max_record_bytes,
        )
        answer = make_records(
            state[key].encode(),
            logical_id=episode_id + ":answer",
            role="assistant",
            turn_index=4,
            provenance="teacher",
            policy_mask=True,
            max_bytes=max_record_bytes,
        )
        assert b"".join(record.payload for record in answer) == value.encode()
        action = LogicalAction(
            action_id=action_id,
            call_id=f"call-{index}",
            tool_name="store.set",
            arguments_json=canonical_json(arguments),
            record_ids=tuple(record.record_id for record in action_records),
            serialized_sha256=digest_bytes(action_bytes),
            execution_id=execution_id,
            schema_id="demo-store-v1",
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "store.set",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                        "required": ["key", "value"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        yield Episode(
            episode_id=episode_id,
            records=first + action_records + result + followup + answer,
            lineage=Lineage(
                "local-demo", "fixture-v1", f"fixture:{index}", task_id=f"fixture:{index}"
            ),
            stream=stream,
            actions=(action,),
            tools_json=canonical_json(tools),
            supervision=Supervision(
                outcome="success",
                verification="local_assertions",
                reward=1.0,
                annotations_json='{"fixture_only":true}',
            ),
            dependencies=(Dependency(result[0].record_id, answer[-1].record_id, "exact_value"),),
        )


def write_demo_source(
    path: str | Path, *, count: int = 32, max_record_bytes: int = 128, stream: str = "bootstrap"
) -> SourceSpec:
    with atomic_binary_writer(path) as handle:
        for episode in make_demo_episodes(count, max_record_bytes=max_record_bytes, stream=stream):
            handle.write(canonical_json(episode.to_dict()).encode() + b"\n")
    return pin_local_source(
        path,
        source_id="local-demo",
        adapter="native",
        stream=stream,
        terms=TermsReview(
            approved=True,
            license="repository-test-fixture",
            reviewed_by="fixture-generator",
            reviewed_at="2026-10-17",
            notes="Locally constructed integration cases; no external corpus.",
        ),
    )
