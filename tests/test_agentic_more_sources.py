"""The second agentic and chat manifest: pinned, licence-recorded, and every source readable by its adapter."""

import json
from pathlib import Path

import pytest
from test_agentic_sources import (
    CONFIGS,
    context,
    first,
    put,
    row_passing_filters,
    sharegpt,
)

from megatoken import cloud
from megatoken.data.adapters import PROFILES, adapt_row
from megatoken.data.quality import validate_row_filters
from megatoken.data.recipes import stage_recipe
from megatoken.data.sources import load_source_manifest

MORE = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-more.json")}
MORE_GATED = {
    s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-more-gated.json")
}
EVERY = {**MORE, **MORE_GATED}
EARLIER = {
    s.source_id
    for name in (
        "agentic",
        "agentic-gated",
        "agentic-coding",
        "open",
        "v13",
    )
    for s in load_source_manifest(CONFIGS / f"sources.{name}.json")
}


def satisfy(row, source):
    """Set every filtered field to a value its row filter accepts."""
    for rule in source.options.get("row_filters", []):
        value = rule.get("value")
        put(
            row,
            rule["path"],
            value[0] if rule["op"] == "in" else "x" if rule["op"] == "exists" else value,
        )
    return row


def fixture_row(source):
    """One row in the shape the source's adapter reads, including the shapes added with this manifest."""
    adapter, fm = source.adapter, source.field_map
    if adapter == "human_assistant_text":
        row = {fm["messages"]: "\n\nHuman: q\n\nAssistant: a"}
    elif adapter == "alternating_turns":
        row = {fm.get("messages", "data"): ["q", "a", "q2", "a2"]}
    elif adapter == "prompt_and_reply":
        row = {
            fm.get("messages", "conversation"): [{"role": "user", "content": "q"}],
            "output": "a",
        }
    elif adapter == "serialized_chat":
        row = {
            fm.get(
                "text", "text"
            ): "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\na<|im_end|>\n"
        }
    elif adapter == "document":
        row = {fm.get("text", "text"): "A complete document."}
    elif adapter == "artifact":
        row = {}
        for key in ("prompt", "target"):
            put(row, first(fm.get(key, key)), "text")
    elif adapter == "smoltalk2":
        row = {
            "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        }
    elif adapter == "glaive_tools":
        function = '{"name": "f", "description": "d", "parameters": {"type": "object"}}'
        chat = (
            'USER: go\nASSISTANT: <functioncall> {"name": "f", "arguments": \'{"a": 1}\'} <|endoftext|>\n'
            'FUNCTION RESPONSE: {"v": 1}\nASSISTANT: done <|endoftext|>'
        )
        system = f"SYSTEM: You have functions.\n{function}\n"
        if "messages" in fm:
            row = {fm["messages"]: system + chat}
        else:
            row = {"system": system, "chat": chat}
    elif adapter == "react_agent" and fm.get("messages") not in (None, "conversations"):
        row = {}
        put(
            row,
            fm["messages"],
            sharegpt(
                [
                    ("human", "task"),
                    ("gpt", "Think: look."),
                    ("human", "output: a"),
                    ("gpt", "done"),
                ]
            ),
        )
        for key in ("family", "id", "task"):
            if key in fm:
                put(row, first(fm[key]), "k1")
    else:
        row = row_passing_filters(source)
        if adapter == "xlam_calls":
            row["tools"] = json.dumps(
                [{"name": "f", "description": "d", "parameters": {"a": {"type": "int"}}}]
            )
        return satisfy(row, source)
    for key in ("family", "id", "task"):
        if key in fm:
            put(row, first(fm[key]), "k1")
    return satisfy(row, source)


def test_every_source_is_pinned_licensed_approved_and_new():
    assert len(MORE) + len(MORE_GATED) == len(EVERY)
    assert not set(EVERY) & EARLIER, "a source id repeats one of the earlier manifests"
    for source in EVERY.values():
        assert len(source.revision) == 40 and source.format == "hf", source.source_id
        assert source.adapter in PROFILES and source.max_rows, source.source_id
        assert source.terms.license not in ("", "unreviewed"), source.source_id
        assert source.terms.license_url.startswith("https://"), source.source_id
        assert (
            source.terms.reviewed_by == "Gabriel Pickard"
            and source.terms.reviewed_at == "2026-10-22"
        )
        source.terms.require_approval()


def test_every_source_streams_into_a_stream_the_stage_plan_reads():
    wanted = set()
    for phase in ("P1", "M1", "M2", "LC", "S"):
        wanted |= set(stage_recipe(phase).mixture)
    assert {s.stream for s in EVERY.values()} <= wanted


@pytest.mark.parametrize("source_id", sorted(EVERY))
def test_fixture_row_becomes_an_episode_in_the_declared_stream(source_id):
    source = EVERY[source_id]
    row = fixture_row(source)
    validate_row_filters(row, source.options.get("row_filters", []))
    episode = adapt_row(row, source.adapter, context(source))
    assert episode.stream == source.stream
    assert any(record.policy_mask for record in episode.records), "nothing to imitate"
    for record in episode.records:
        assert not record.policy_mask or record.role in {"assistant", "document"}
        assert not record.result_of or (record.role == "tool" and not record.policy_mask)
    if source.stream == "verified_trajectories":
        assert episode.supervision.verified_success
    provenance = {r.provenance for r in episode.records if r.result_of}
    assert provenance <= {"executed", "simulated"}


def test_executed_streams_say_executed_and_simulated_dialects_say_simulated_unless_told():
    for source in EVERY.values():
        provenance = source.options.get(
            "observation_provenance", PROFILES[source.adapter].observation_provenance
        )
        if source.stream in {"verified_trajectories", "executed_trajectories"}:
            assert provenance == "executed", source.source_id
        if source.adapter in {"hermes_tools", "glaive_tools", "python_call_tools", "xlam_calls"}:
            # A source may say its results were really executed (smolagents search traces).
            assert provenance in {"simulated", "executed"}, source.source_id
            assert "observation_provenance" in source.options or provenance == "simulated"


def test_verified_streams_only_take_sources_that_filter_to_a_recorded_test_verdict():
    for source in EVERY.values():
        if source.stream != "verified_trajectories":
            continue
        assert PROFILES[source.adapter].outcome_verification == "publisher_tests", source.source_id
        assert not source.options.get("assume_outcome"), source.source_id
        assert source.options.get("row_filters"), source.source_id


def test_the_cloud_job_reads_the_manifest_it_names_and_needs_no_gpu():
    job = cloud.validate(
        json.loads(
            (Path(__file__).parents[1] / "configs/cloud/prepare-agentic-more.json").read_text()
        )
    )
    assert "configs/data/sources.agentic-more.json" in " ".join(job["command"])
    assert "configs/data/parser-review.agentic.json" in " ".join(job["command"])
    assert job["resources"]["gpu"] == 0
