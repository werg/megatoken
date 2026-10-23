"""The agentic-chat manifests: pinned, licence-recorded, listed for full-volume runs, and every
source readable by its adapter."""

import json
from pathlib import Path

import pytest
from test_agentic_sources import (
    CONFIGS,
    TOOLS,
    context,
    first,
    openai_messages,
    put,
    row_for,
    sharegpt,
)

from megatoken.data.adapters import PROFILES, adapt_row
from megatoken.data.quality import validate_row_filters
from megatoken.data.recipes import stage_recipe
from megatoken.data.sources import load_source_manifest

ROOT = Path(__file__).parents[1]
CHAT = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-chat.json")}
GATED = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-chat-gated.json")}
EVERY = {**CHAT, **GATED}
EARLIER = {
    s.source_id
    for name in (
        "agentic",
        "agentic-gated",
        "agentic-more",
        "agentic-more-gated",
        "later-stages",
        "open",
        "v13",
    )
    for s in load_source_manifest(CONFIGS / f"sources.{name}.json")
}


def messages_key(source, default="messages"):
    defaults = dict(PROFILES[source.adapter].fields)
    return source.field_map.get("messages") or defaults.get("messages", (default,))[0]


def tools_key(source):
    return first(source.field_map.get("tools", "tools"))


def system_prompt_with_call_list():
    tool = {"name": "f", "description": "d", "parameters": {"x": {"type": "string"}}}
    return f"Tools:\n<tool>{json.dumps([tool])}</tool>"


def new_shape_row(source):
    """One row in the shape of an adapter added with this manifest; None for the older adapters."""
    adapter, key = source.adapter, messages_key(source)
    row = {}
    call = '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'
    if adapter in {"hermes_tools", "hermes_user_result_tools"}:
        bare = adapter == "hermes_user_result_tools"
        put(row, tools_key(source), json.dumps(TOOLS))
        return {
            **row,
            **_nested(
                key,
                sharegpt(
                    [
                        ("system", "You may call functions."),
                        ("human", "go"),
                        ("gpt", call),
                        (
                            "human" if bare else "tool",
                            '{"v": 1}' if bare else "<tool_response>\n1\n</tool_response>",
                        ),
                        ("gpt", "done"),
                    ]
                ),
            ),
        }
    if adapter == "simulated_tool_roles":
        put(row, tools_key(source), json.dumps(TOOLS))
        messages = [
            {"from": "human", "value": "go"},
            {"from": "function_call", "value": '{"name": "f", "arguments": {"a": 1}}'},
            {"from": "observation", "value": "out"},
            {"from": "gpt", "value": "done"},
        ]
        return {**row, **_nested(key, messages)}
    if adapter == "python_call_tools":
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "[f(a=1)]"},
            {"role": "tool", "content": "out"},
            {"role": "assistant", "content": "done"},
        ]
        return _nested(key, messages)
    if adapter in {"react_agent", "react_function_agent"}:
        reply = "function" if adapter == "react_function_agent" else "human"
        turns = [
            ("human", "task"),
            ("gpt", "Think: look.\nAct: bash"),
            (reply, "out"),
            ("gpt", "done"),
        ]
        return _nested(key, sharegpt(turns))
    if adapter == "harmony_tools":

        def message(role, text, **rest):
            return {
                "role": role,
                "content": [{"text": text}],
                "channel": None,
                "recipient": None,
            } | rest

        return _nested(
            key,
            [
                message("user", "Q"),
                message(
                    "assistant", '{"query": "a"}', channel="analysis", recipient="browser.search"
                ),
                message("tool", "result", name="browser.search"),
                message("assistant", "Answer", channel="final"),
            ],
        )
    if adapter == "tool_dict_tools":
        messages = [
            {
                "role": "assistant",
                "content": "look",
                "tool_call": {"tool": "search", "args": {"q": 1}},
            },
            {"role": "tool", "tool": "search", "content": "found"},
            {"role": "assistant", "content": "answer"},
        ]
        return {"question": "Q?", **_nested(key, messages)}
    if adapter == "device_call_tools":
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "name": "f", "arguments": {}}],
            },
            {"role": "tool", "content": "out"},
            {"role": "assistant", "content": "done"},
        ]
        return _nested(key, messages)
    if adapter == "call_list_tools":
        messages = [
            {"role": "system", "content": system_prompt_with_call_list()},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": 'Plan.\n<call>[{"name": "f", "arguments": {"x": "1"}}]</call>',
            },
            {"role": "tool", "content": json.dumps([{"name": "f", "results": 1}])},
            {"role": "assistant", "content": "<final>done</final>"},
        ]
        return _nested(key, messages)
    if adapter == "responses_tools":
        return {
            "responses_create_params": {
                "tools": TOOLS,
                "input": [
                    {"role": "system", "content": "Help."},
                    {"role": "user", "content": "go"},
                    {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c", "output": "ok"},
                ],
            },
            "expected_action": {"type": "message", "content": "done"},
        }
    if adapter == "tag_trace_tools":
        output = "<think>x</think><search>q</search>\n<result>r</result><answer>a</answer>"
        return {"instruction": "Search.", "input": "Q?", "output": output}
    if adapter == "glaive_message_tools":
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": '<functioncall> {"name": "f", "arguments": \'{"a": 1}\'}',
            },
            {"role": "tool", "content": "out"},
            {"role": "assistant", "content": "done"},
        ]
        return _nested(key, messages)
    if adapter == "reasoning_role_tools":
        messages = [
            {"role": "user", "content": "Q"},
            {"role": "reasoning", "content": "<think>look</think>"},
            {"role": "tool_call", "content": call},
            {"role": "tool_output", "content": "<tool_response>1</tool_response>"},
            {"role": "answer", "content": "<answer>A</answer>"},
        ]
        return _nested(key, messages)
    return None


def _nested(key, value):
    row = {}
    put(row, key, value)
    return row


def chat_row(source):
    row = _nested(
        messages_key(source), openai_messages(source.options.get("positional_call_ids", False))
    )
    put(row, tools_key(source), TOOLS)
    return row


def fixture_row(source):
    row = new_shape_row(source)
    if row is None:
        row = chat_row(source) if source.adapter == "chat" else row_for(source)
    for key in ("family", "id", "task"):
        if key in source.field_map:
            put(row, first(source.field_map[key]), "k1")
    for rule in source.options.get("row_filters", []):
        value = rule.get("value")
        put(
            row,
            rule["path"],
            value[0] if rule["op"] == "in" else "x" if rule["op"] == "exists" else value,
        )
    return row


def test_every_source_is_pinned_licensed_approved_and_new():
    assert len(CHAT) + len(GATED) == len(EVERY)
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
    # Tool and agent conversations without a recorded test verdict never take a verified stream.
    assert {s.stream for s in EVERY.values()}.isdisjoint(
        {"verified_trajectories", "verified_successful_multiturn"}
    )


def test_gated_sources_say_which_page_to_agree_on():
    for source in GATED.values():
        assert "Gated" in source.terms.notes and "Agree" in source.terms.notes, source.source_id
    assert not set(CHAT) & set(GATED)


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
    provenance = {r.provenance for r in episode.records if r.result_of}
    assert provenance <= {"executed", "simulated", "unknown"}


def test_executed_streams_say_executed_and_text_dialects_say_simulated_unless_told():
    for source in EVERY.values():
        provenance = source.options.get(
            "observation_provenance", PROFILES[source.adapter].observation_provenance
        )
        assert provenance in {"executed", "simulated", "unknown"}, source.source_id
        if source.adapter in {"glaive_tools", "glaive_message_tools", "xlam_calls"}:
            assert provenance == "simulated", source.source_id


def test_full_volume_runs_can_list_every_open_source_and_no_listing_is_empty():
    listing = json.loads((CONFIGS / "files.agentic-chat.json").read_text())
    assert set(CHAT) <= set(listing)
    for source_id in CHAT:
        assert listing[source_id], f"{source_id} lists no files"
        assert all(size > 0 for _, size in listing[source_id]), source_id


def test_the_cloud_job_reads_the_manifest_and_listing_it_names():
    spec = json.loads((ROOT / "configs/cloud/prepare-full-agentic-chat.json").read_text())
    assert spec["sources"] == "configs/data/sources.agentic-chat.json"
    assert spec["listing"] == "configs/data/files.agentic-chat.json"
    assert spec["review_config"] == "configs/data/parser-review.agentic.json"
    assert spec["resources"]["gpu"] == 0
