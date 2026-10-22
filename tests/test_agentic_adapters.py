"""Tool-use and agent-trajectory profiles: calls and results become their own linked records."""

import datetime
import json
from dataclasses import replace

import pytest

from megatoken.data import AdapterContext, SchemaError, adapt_row
from megatoken.generation.codec import ByteCodec


def context(stream="tool_use", **options):
    return AdapterContext(
        source_id="fixture",
        revision="rev",
        uri="hf://fixture",
        stream=stream,
        options=options,
        codec=ByteCodec.byte_fallback(),
        max_record_tokens=1024,
    )


def adapt(row, adapter, stream="tool_use", **options):
    return adapt_row(row, adapter, context(stream, **options))


def kinds(episode):
    """Role, scored flag and what each record is, in order."""
    result = []
    for record in episode.records:
        kind = "action" if record.action_id else "result" if record.result_of else "text"
        result.append((record.role, record.policy_mask, kind, record.provenance))
    return result


def linked(episode):
    """Every result names an action that precedes it and carries its execution identity."""
    first = {}
    for index, record in enumerate(episode.records):
        if record.action_id:
            first.setdefault(record.action_id, index)
    executions = {action.action_id: action.execution_id for action in episode.actions}
    for index, record in enumerate(episode.records):
        if record.result_of:
            assert first[record.result_of] < index
            assert record.execution_id == executions[record.result_of]
    return True


HERMES_TOOLS = json.dumps(
    [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
)


def hermes_row(assistant, tool=None):
    messages = [
        {"from": "system", "value": "You may call functions."},
        {"from": "human", "value": "do it"},
        {"from": "gpt", "value": assistant},
    ]
    if tool is not None:
        messages += [{"from": "tool", "value": tool}, {"from": "gpt", "value": "done"}]
    return {"id": "h1", "tools": HERMES_TOOLS, "conversations": messages}


def test_hermes_parallel_calls_and_results_pair_in_order_with_simulated_results():
    row = hermes_row(
        'Checking.\n<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>\n'
        "<tool_call>\n{'name': 'f', 'arguments': {'a': 2}}\n</tool_call>",
        '<tool_response>\n{"v": 1}\n</tool_response>\n<tool_response>\n{"v": 2}\n</tool_response>',
    )
    episode = adapt(row, "hermes_tools")
    assert [(r.role, r.policy_mask, r.provenance) for r in episode.records[2:]] == [
        ("assistant", True, "teacher"),
        ("assistant", True, "teacher"),
        ("assistant", True, "teacher"),
        ("tool", False, "simulated"),
        ("tool", False, "simulated"),
        ("assistant", True, "teacher"),
    ]
    assert [json.loads(a.arguments_json) for a in episode.actions] == [{"a": 1}, {"a": 2}]
    assert [r.payload for r in episode.records if r.result_of] == [b'{"v": 1}', b'{"v": 2}']
    assert linked(episode)
    assert not episode.supervision.verified_success


def test_hermes_result_count_must_match_the_calls():
    row = hermes_row(
        '<tool_call>{"name": "f", "arguments": {}}</tool_call>',
        "<tool_response>1</tool_response><tool_response>2</tool_response>",
    )
    with pytest.raises(SchemaError, match="pending"):
        adapt(row, "hermes_tools")


def test_results_sent_one_tool_message_each_answer_parallel_calls_in_turn():
    calls = "".join(
        f'<tool_call>{{"name": "f", "arguments": {{"a": {i}}}}}</tool_call>' for i in range(2)
    )
    row = hermes_row(calls)
    row["conversations"] += [
        {"from": "tool", "value": "<tool_response>1</tool_response>"},
        {"from": "tool", "value": "<tool_response>2</tool_response>"},
        {"from": "gpt", "value": "both"},
    ]
    episode = adapt(row, "hermes_tools")
    assert [r.payload for r in episode.records if r.result_of] == [b"1", b"2"]
    assert linked(episode)
    row["conversations"].pop(-2)  # the second result never arrives
    with pytest.raises(SchemaError, match="partly answered"):
        adapt(row, "hermes_tools")


def test_python_call_syntax_in_content_and_in_a_field_with_one_result_per_call():
    toolace = {
        "system": "You compose functions.",
        "conversations": [
            {"from": "user", "value": "q"},
            {
                "from": "assistant",
                "value": '[Market Trends API(trend="US", n=3), Other(x=[1, "a, b"])]',
            },
            {
                "from": "tool",
                "value": json.dumps(
                    [{"name": "Market Trends API", "results": 1}, {"name": "Other", "results": 2}]
                ),
            },
            {"from": "assistant", "value": "Here you go."},
        ],
    }
    episode = adapt(toolace, "python_call_tools")
    assert [a.tool_name for a in episode.actions] == ["Market Trends API", "Other"]
    assert json.loads(episode.actions[1].arguments_json) == {"x": [1, "a, b"]}
    assert sum(1 for r in episode.records if r.result_of) == 2 and linked(episode)
    dolci = {
        "messages": [
            {"role": "system", "content": "tools", "functions": HERMES_TOOLS},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "function_calls": "f(a=1)\nf(a=2)"},
            {"role": "environment", "content": '{"r": 1}\n{"r": 2}'},
            {"role": "assistant", "content": "ok"},
        ]
    }
    episode = adapt(dolci, "python_call_tools")
    assert len(episode.actions) == 2 and linked(episode)
    assert json.loads(episode.tools_json)[0]["function"]["name"] == "f"


def test_prose_that_merely_starts_with_a_bracket_is_not_a_call():
    row = {
        "conversations": [
            {"from": "user", "value": "q"},
            {"from": "assistant", "value": "[1] First, note that..."},
        ]
    }
    episode = adapt(row, "python_call_tools")
    assert not episode.actions
    broken = {
        "conversations": [
            {"from": "user", "value": "q"},
            {"from": "assistant", "value": "[F(a=1+)]"},
        ]
    }
    with pytest.raises(SchemaError):
        adapt(broken, "python_call_tools")


def test_roles_for_calls_and_results_executed_mcp_tools():
    row = {
        "uuid": "t1",
        "tools": HERMES_TOOLS,
        "messages": json.dumps(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "calling both"},
                {"role": "tool_call", "content": {"name": "f", "arguments": '{"a": 1}'}},
                {"role": "tool_call", "content": '{"name": "f", "arguments": {"a": 2}}'},
                {"role": "tool_response", "content": "one"},
                {"role": "tool_response", "content": "two"},
                {"role": "assistant", "content": "both done"},
            ]
        ),
    }
    episode = adapt(row, "mcp_tools")
    assert [k for _, _, k, _ in kinds(episode)] == [
        "text",
        "text",
        "action",
        "action",
        "result",
        "result",
        "text",
    ]
    assert {p for _, _, k, p in kinds(episode) if k == "result"} == {"executed"}
    assert linked(episode)
    # The first result answers the first call, not the last.
    first = episode.actions[0]
    assert next(r for r in episode.records if r.result_of).result_of == first.action_id


def test_glaive_dumps_with_single_quoted_arguments():
    row = {
        "system": 'SYSTEM: You have access to -\n{"name": "get_rate", "description": "d", '
        '"parameters": {"type": "object", "properties": {"c": {"type": "string"}}}}\n',
        "chat": 'USER: rate?\n\n\nASSISTANT: <functioncall> {"name": "get_rate", "arguments": '
        '\'{"c": "EUR"}\'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {"rate": 1.1}\n\n\n'
        "ASSISTANT: It is 1.1. <|endoftext|>\n\n\n",
    }
    episode = adapt(row, "glaive_tools")
    assert json.loads(episode.actions[0].arguments_json) == {"c": "EUR"}
    assert [k for _, _, k, _ in kinds(episode)] == ["text", "text", "action", "result", "text"]
    assert json.loads(episode.tools_json)[0]["name"] == "get_rate"
    refusal = {
        "system": "SYSTEM: none",
        "chat": "USER: hi\n\n\nASSISTANT: hello <|endoftext|>\n\n\n",
    }
    assert not adapt(refusal, "glaive_tools").actions


def test_xlam_calls_have_no_result_and_parameters_become_json_schema():
    row = {
        "id": 3,
        "query": "weather in Paris and Rome?",
        "answers": json.dumps(
            [
                {"name": "w", "arguments": {"city": "Paris"}},
                {"name": "w", "arguments": {"city": "Rome"}},
            ]
        ),
        "tools": json.dumps(
            [
                {
                    "name": "w",
                    "description": "d",
                    "parameters": {
                        "city": {"type": "str", "description": "c"},
                        "days": {"type": "int", "default": 1},
                    },
                }
            ]
        ),
    }
    with pytest.raises(SchemaError, match="observation is missing"):
        adapt(row, "xlam_calls")
    episode = adapt(row, "xlam_calls", allow_pending_final_action=True)
    assert len(episode.actions) == 2 and not any(r.result_of for r in episode.records)
    schema = json.loads(episode.tools_json)[0]["parameters"]
    assert schema["required"] == ["city"] and schema["properties"]["days"]["type"] == "integer"


OPENHANDS_CALL = (
    "Let me look.\n\n<function=execute_bash>\n<parameter=command>ls /repo</parameter>\n</function>"
)


def openhands_row(**extra):
    return {
        "instance_id": "a__b-1",
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": OPENHANDS_CALL},
            {"role": "user", "content": "OBSERVATION:\nREADME.md"},
            {"role": "assistant", "content": "<function=submit>\n</function>"},
        ],
        **extra,
    }


def test_text_function_scaffold_is_an_executed_action_and_result():
    episode = adapt(
        openhands_row(resolved=True),
        "openhands_xml",
        "verified_trajectories",
        allow_pending_final_action=True,
    )
    assert [k for _, _, k, _ in kinds(episode)] == ["text", "text", "action", "result", "action"]
    assert episode.actions[0].tool_name == "execute_bash"
    assert json.loads(episode.actions[0].arguments_json) == {"command": "ls /repo"}
    assert episode.records[3].provenance == "executed" and episode.records[3].role == "tool"
    assert episode.supervision.verified_success
    assert [r.policy_mask for r in episode.records] == [False, False, True, False, True]


def test_failed_scaffold_trace_is_context_only_and_curated_success_is_not_verification():
    failed = adapt(openhands_row(resolved=False), "openhands_xml", allow_pending_final_action=True)
    assert not any(r.policy_mask for r in failed.records)
    curated = adapt(
        openhands_row(), "openhands_xml", allow_pending_final_action=True, assume_outcome="success"
    )
    assert curated.supervision.outcome == "success"
    assert curated.supervision.verification == "publisher_curated"
    assert not curated.supervision.verified_success
    assert sum(r.policy_mask for r in curated.records) == 2
    unlabelled = adapt(openhands_row(), "openhands_xml", allow_pending_final_action=True)
    assert unlabelled.supervision.outcome == "unknown"
    with pytest.raises(SchemaError, match="only be 'success'"):
        adapt(
            openhands_row(),
            "openhands_xml",
            allow_pending_final_action=True,
            assume_outcome="failure",
        )


def test_fenced_command_scaffold_reads_its_own_roles_masks_and_system_prompt():
    row = {
        "instance_id": "o__r-2",
        "target": True,
        "trajectory": [
            {"role": "system", "system_prompt": "SETTING: agent", "text": None, "mask": False},
            {"role": "user", "text": "Issue text", "mask": False},
            {"role": "ai", "text": "DISCUSSION\nlook\n```\nls\n```", "mask": True},
            {"role": "user", "text": "file.py", "mask": False},
            {"role": "ai", "text": "DISCUSSION\ndone\n```\nsubmit\n```", "mask": False},
        ],
    }
    episode = adapt(
        row, "swe_agent_fenced", "verified_trajectories", allow_pending_final_action=True
    )
    assert episode.records[0].payload == b"SETTING: agent"
    assert [k for _, _, k, _ in kinds(episode)] == ["text", "text", "action", "result", "action"]
    # The publisher's own per-message mask is honoured: the second action is not trained.
    assert [r.policy_mask for r in episode.records] == [False, False, True, False, False]
    assert episode.supervision.verified_success


def test_mini_swe_agent_actions_are_single_bash_blocks():
    row = {
        "instance_id": "x__y-3",
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "THOUGHT: look\n\n```bash\nls\n```"},
            {"role": "user", "content": "<returncode>0</returncode>\n<output>a</output>"},
        ],
    }
    episode = adapt(row, "mini_swe_agent", assume_outcome="success")
    assert episode.actions[0].tool_name == "bash" and linked(episode)
    assert episode.records[-1].provenance == "executed"


def test_react_text_turns_after_the_task_are_environment_records():
    row = {
        "id": "os_0",
        "conversations": [
            {"from": "human", "value": "task", "loss": None},
            {"from": "gpt", "value": "Think: list.\nAct: bash\n```bash\nls\n```", "loss": True},
            {"from": "human", "value": "The output of the OS:\na b", "loss": None},
            {"from": "gpt", "value": "Think: done.\nAct: answer(2)", "loss": False},
        ],
    }
    episode = adapt(row, "react_agent", "environment_trajectories")
    assert [(r.role, r.policy_mask, r.provenance) for r in episode.records] == [
        ("user", False, "natural"),
        ("assistant", True, "teacher"),
        ("environment", False, "executed"),
        ("assistant", False, "teacher"),
    ]
    simulated = adapt(
        row, "react_agent", "environment_trajectories", observation_provenance="simulated"
    )
    assert simulated.records[2].provenance == "simulated"
    assert not episode.actions


def test_turn_pairs_and_a_withheld_prompt():
    row = {"conversation": [{"input": "hi", "output": "hello"}, {"input": "bye", "output": "ciao"}]}
    episode = adapt(row, "turn_pairs", "chat")
    assert [r.role for r in episode.records] == ["user", "assistant", "user", "assistant"]
    for withheld_prompt in (None, "", "  "):
        withheld = {
            "messages": [
                {"role": "user", "content": withheld_prompt},
                {"role": "assistant", "content": "answer"},
            ]
        }
        with pytest.raises(SchemaError, match="no content"):
            adapt(withheld, "chat", "chat")


def test_timestamps_in_a_columnar_row_do_not_stop_the_row_hash():
    row = {
        "timestamp": datetime.datetime(2026, 1, 1, 12, 0),
        "conversation": [
            {"role": "user", "content": "hi", "timestamp": datetime.datetime(2026, 1, 1)},
            {"role": "assistant", "content": "hello"},
        ],
    }
    wild = AdapterContext(
        "wild",
        "rev",
        "hf://wild",
        "chat",
        field_map={"messages": "conversation"},
        codec=ByteCodec.byte_fallback(),
        max_record_tokens=1024,
    )
    episode = adapt_row(row, "chat", wild)
    assert [r.role for r in episode.records] == ["user", "assistant"]
    assert adapt_row(row, "chat", wild).episode_id == episode.episode_id


def test_structured_results_null_schema_keys_and_string_tool_calls():
    row = {
        "repo": "o/r",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "string", "enum": None, "required": True},
                            "b": None,
                            "c": {"type": "List[float]"},
                            "d": {"type": "int, optional"},
                            "e": {"type": "weird"},
                        },
                        "additionalProperties": None,
                    },
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": json.dumps(
                    [
                        {
                            "id": "c1",
                            "type": "function",
                            "index": None,
                            "function": {"name": "f", "arguments": '{"a": "x"}'},
                        }
                    ]
                ),
            },
            {"role": "tool", "tool_call_id": "c1", "content": {"ok": True}},
            {"role": "assistant", "content": "done"},
        ],
    }
    episode = adapt(row, "open_swe", "executed_trajectories", assume_outcome="success")
    assert json.loads(episode.tools_json)[0]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "c": {"type": "array"},
            "d": {"type": "integer"},
            "e": {},
        },
    }
    result = next(r for r in episode.records if r.result_of)
    assert result.payload == b'{"ok":true}' and result.source_span.transform == "canonical-json-v1"
    action = next(r for r in episode.records if r.action_id)
    assert json.loads(action.payload) == {
        "function": {"arguments": '{"a": "x"}', "name": "f"},
        "id": "c1",
        "type": "function",
    }


def test_results_find_their_call_under_each_publisher_spelling():
    def row(result_key):
        return {
            "repo": "o/r",
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {}}}],
                },
                {"role": "tool", **result_key, "content": [{"type": "text", "text": "out"}]},
                {"role": "assistant", "content": "done"},
            ],
        }

    for key in ({"tool_call_id": "c1"}, {"tool_call_ids": '["c1"]'}, {"id": "c1"}, {}):
        episode = adapt(row(key), "open_swe", "executed_trajectories")
        assert any(r.result_of for r in episode.records) and linked(episode)


def test_calls_without_ids_pair_by_position_only_when_the_source_opts_in():
    row = {
        "repo": "o/r",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}},
                    {"type": "function", "function": {"name": "f", "arguments": '{"a": 2}'}},
                ],
            },
            {"role": "tool", "content": "one"},
            {"role": "tool", "content": "two"},
            {"role": "assistant", "content": "done"},
        ],
    }
    with pytest.raises(SchemaError, match="ID"):
        adapt(row, "open_swe", "executed_trajectories")
    episode = adapt(
        row, "open_swe", "executed_trajectories", positional_call_ids=True, assume_outcome="success"
    )
    first, second = episode.actions
    answers = [r for r in episode.records if r.result_of]
    assert [r.result_of for r in answers] == [first.action_id, second.action_id]
    assert [r.payload for r in answers] == [b"one", b"two"]


def test_a_source_with_no_outcome_is_admitted_as_unknown_only_when_it_says_so():
    row = openhands_row()
    context_free = adapt(row, "openhands_xml", allow_pending_final_action=True)
    assert not any(r.policy_mask for r in context_free.records)
    admitted = adapt(
        row, "openhands_xml", allow_pending_final_action=True, admit_unknown_outcome=True
    )
    assert admitted.supervision.outcome == "unknown" and not admitted.supervision.verified_success
    assert sum(r.policy_mask for r in admitted.records) == 2
    # A recorded failure is never admitted, whatever the option says.
    failed = adapt(
        openhands_row(resolved=False),
        "openhands_xml",
        allow_pending_final_action=True,
        admit_unknown_outcome=True,
    )
    assert not any(r.policy_mask for r in failed.records)


def test_terminal_batches_written_after_a_thinking_block_are_still_actions():
    call = (
        '{"analysis": "look", "plan": "ls", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]}'
    )
    row = {
        "task": "t1",
        "messages": [
            {"role": "user", "content": "Solve the task."},
            {"role": "assistant", "content": f"<think>\nList the files first.\n</think>\n\n{call}"},
            {"role": "user", "content": "New Terminal Output:\na.py"},
            {
                "role": "assistant",
                "content": '<think>done</think>\n\n{"commands": [], "task_complete": true}',
            },
        ],
    }
    episode = adapt_row(
        row,
        "nemotron_terminal",
        replace(
            context(
                "executed_trajectories", allow_pending_final_action=True, admit_unknown_outcome=True
            ),
            field_map={"family": "task"},
        ),
    )
    assert [kind for _, _, kind, _ in kinds(episode)] == ["text", "action", "result", "text"]
    assert linked(episode)
    # Prose around the call is not parsed as a JSON document.
    assert not any("low_syntax_coverage" in r.chunk_diagnostics for r in episode.records)


def test_a_row_level_language_does_not_turn_every_message_into_that_language():
    prose = "You are a careful agent. Read the issue, look at the code, then fix it. " * 8
    row = openhands_row(language="python", resolved=True)
    row["messages"][0]["content"] = prose
    episode = adapt(row, "openhands_xml", "verified_trajectories", allow_pending_final_action=True)
    assert all("python" not in r.chunk_languages for r in episode.records if r.role == "system")
    assert not any("low_syntax_coverage" in r.chunk_diagnostics for r in episode.records)
    # A message that names its own language still sets it.
    row["messages"][1] = {"role": "user", "content": "x = 1", "language": "python"}
    tagged = adapt(row, "openhands_xml", "verified_trajectories", allow_pending_final_action=True)
    assert any("python" in r.chunk_languages for r in tagged.records if r.role == "user")


def adapt_mapped(row, adapter, stream, field_map, **options):
    return adapt_row(row, adapter, replace(context(stream, **options), field_map=field_map))


def test_capitalised_roles_mean_the_same_as_lowercase_ones():
    row = {
        "conversations": [
            {"from": "User", "value": "hi"},
            {"from": "Assistant", "value": "hello"},
        ]
    }
    episode = adapt(row, "chat", "chat")
    assert [r.role for r in episode.records] == ["user", "assistant"]
    assert [r.policy_mask for r in episode.records] == [False, True]


def test_anthropic_text_and_alternating_strings_become_messages():
    chosen = "\n\nHuman: Hi\n\nAssistant: Hello there\n\nHuman: Bye\n\nAssistant: Goodbye"
    mapped = {"messages": "chosen"}
    episode = adapt_mapped({"chosen": chosen}, "human_assistant_text", "chat", mapped)
    assert [(r.role, r.policy_mask) for r in episode.records] == [
        ("user", False),
        ("assistant", True),
        ("user", False),
        ("assistant", True),
    ]
    with pytest.raises(SchemaError, match="before the first"):
        adapt_mapped(
            {"chosen": "preamble\n\nHuman: Hi\n\nAssistant: ok"},
            "human_assistant_text",
            "chat",
            mapped,
        )
    episode = adapt({"data": ["q", "a", "q2", "a2"], "id": "u1"}, "alternating_turns", "chat")
    assert [r.role for r in episode.records] == ["user", "assistant", "user", "assistant"]
    with pytest.raises(SchemaError, match="list of strings"):
        adapt({"data": [{"role": "user"}]}, "alternating_turns", "chat")


def test_a_reply_kept_apart_from_its_prompt_turns_is_appended_with_its_reasoning():
    split = {
        "conversation": [{"role": "user", "content": "q"}],
        "generated_message": {
            "role": "assistant",
            "content": "the answer",
            "reasoning_content": "think first",
            "tool_calls": [],
        },
    }
    reply = adapt(split, "prompt_and_reply", "chat").records[-1]
    assert reply.policy_mask and reply.payload == b"<think>\nthink first\n</think>\n\nthe answer"
    columns = {
        "conversation": [{"role": "user", "content": "q"}],
        "reasoning": "r",
        "answer": "a",
    }
    assert adapt(columns, "prompt_and_reply", "chat").records[-1].payload.endswith(b"\n\na")
    flagged = {"input": [{"role": "user", "content": "q"}], "output": "a", "reasoning": "on"}
    plain = adapt_mapped(flagged, "prompt_and_reply", "chat", {"messages": "input"})
    assert plain.records[-1].payload == b"a"
    both = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [{"role": "assistant", "content": "a"}],
    }
    assert [r.role for r in adapt(both, "prompt_completion", "chat").records] == [
        "user",
        "assistant",
    ]


def test_reasoning_kept_in_its_own_message_field_is_spelled_inline_and_scored():
    row = {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "plan the call",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done", "reasoning_content": " "},
        ]
    }
    episode = adapt(row, "chat", "tool_use", observation_provenance="executed")
    texts = [r for r in episode.records if not r.action_id and not r.result_of]
    assert b"<think>\nplan the call\n</think>\n\n" in [r.payload for r in texts]
    # A blank trace adds nothing.
    assert texts[-1].role == "assistant" and texts[-1].payload == b"done"


def test_openai_wrapped_tools_and_declined_queries_in_the_xlam_shape():
    tools = json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
                },
            }
        ]
    )
    answered = {"query": "q", "answers": '[{"name": "f", "arguments": {"a": "x"}}]', "tools": tools}
    episode = adapt(answered, "xlam_calls", "tool_use", allow_pending_final_action=True)
    assert json.loads(episode.tools_json)[0]["parameters"]["properties"] == {
        "a": {"type": "string"}
    }
    declined = {"query": "q", "answers": "The query cannot be answered.", "tools": tools}
    episode = adapt(declined, "xlam_calls", "recovery_clarification")
    assert episode.records[-1].payload == b"The query cannot be answered."
    assert not any(r.action_id for r in episode.records)


def test_glaive_first_release_keeps_system_prompt_and_chat_in_one_string():
    function = '{"name": "f", "description": "d", "parameters": {"type": "object"}}'
    sample = (
        f"SYSTEM: You have functions.\n{function}\n"
        "USER: go\n"
        'ASSISTANT: <functioncall> {"name": "f", "arguments": \'{"a": 1}\'} <|endoftext|>\n'
        'FUNCTION RESPONSE: {"v": 1}\n'
        "ASSISTANT: done <|endoftext|>"
    )
    episode = adapt_mapped({"sample": sample}, "glaive_tools", "tool_use", {"messages": "sample"})
    assert any(r.action_id for r in episode.records)
    assert linked(episode)


def test_python_calls_with_odd_tool_names_json_literals_and_keyword_parameters():
    # Shapes seen in ToolACE and Dolci: parentheses and ampersands in tool names, a parameter
    # called ``from``, ``true``/``null`` inside Python call syntax, and a ``"null"`` call field.
    toolace = {
        "conversations": [
            {"from": "user", "value": "q"},
            {
                "from": "assistant",
                "value": '[User Feed (Video Posts) V2(username="a"), '
                'Q&A Notifications(from="2025-01-01", n=3)]',
            },
            {"from": "tool", "value": json.dumps([{"results": 1}, {"results": 2}])},
            {"from": "assistant", "value": "done"},
        ]
    }
    episode = adapt(toolace, "python_call_tools")
    assert [a.tool_name for a in episode.actions] == [
        "User Feed (Video Posts) V2",
        "Q&A Notifications",
    ]
    assert json.loads(episode.actions[1].arguments_json) == {"from": "2025-01-01", "n": 3}
    dolci = {
        "messages": [
            {"role": "user", "content": "q", "function_calls": "null"},
            {
                "role": "assistant",
                "content": None,
                "function_calls": 'api.v1_search(keyword=null, strict=true, code="3000")',
            },
            {"role": "environment", "content": '{"r": 1}'},
            {"role": "assistant", "content": "ok", "function_calls": "null"},
        ]
    }
    episode = adapt(dolci, "python_call_tools")
    assert json.loads(episode.actions[0].arguments_json) == {
        "keyword": None,
        "strict": True,
        "code": "3000",
    }
    assert linked(episode)
    # Bracketed prose that is not a call is still prose.
    prose = {
        "conversations": [
            {"from": "user", "value": "q"},
            {"from": "assistant", "value": "[Tip & trick (see docs)]"},
        ]
    }
    assert not adapt(prose, "python_call_tools").actions


def test_qwen_style_search_traces_put_results_in_user_turns_and_the_prompt_in_a_column():
    call = '{"name": "search", "arguments": {"query": "x"}}'
    row = {
        "system_prompt": "You are a deep search assistant.",
        "messages": [
            {"role": "user", "content": "Question?"},
            {
                "role": "assistant",
                "content": f"<think>hm</think>\n\n<tool_call>\n{call}\n</tool_call>",
            },
            {"role": "user", "content": "<tool_response>\nresult text\n</tool_response>"},
            {"role": "assistant", "content": "<think>ok</think>\n\n<answer>42</answer>"},
        ],
    }
    episode = adapt(row, "hermes_tools", observation_provenance="executed")
    assert [a.tool_name for a in episode.actions] == ["search"] and linked(episode)
    assert episode.records[0].role == "system"
    assert [r.provenance for r in episode.records if r.result_of] == ["executed"]
    # A user turn that merely mentions the tag is not a result.
    chatty = {"messages": [{"role": "user", "content": "what is <tool_response> for?"}]}
    assert not adapt(chatty, "hermes_tools").actions


def test_harmony_traces_pair_parallel_calls_in_order_and_drop_empty_system_configuration():
    def message(role, text, **rest):
        return {
            "role": role,
            "content": [{"text": text}],
            "channel": None,
            "recipient": None,
        } | rest

    schema = {"type": "object", "properties": {"query": {"type": "string"}, "cursor": None}}
    tools = {"browser": {"tools": [{"name": "search", "description": "d", "parameters": schema}]}}
    row = {
        "messages": [
            {"role": "system", "content": [{"text": None, "tools": tools}]},
            message("developer", "You are a research agent."),
            message("user", "Question"),
            message("assistant", "thinking", channel="analysis"),
            message("assistant", '{"query": "a"}', channel="analysis", recipient="browser.search"),
            message("assistant", '{"query": "b"}', channel="analysis", recipient="browser.search"),
            message("tool", "result a", name="browser.search", recipient="assistant"),
            message("tool", "result b", name="browser.search", recipient="assistant"),
            message("assistant", "Answer", channel="final"),
        ]
    }
    episode = adapt(row, "harmony_tools")
    assert [a.tool_name for a in episode.actions] == ["browser.search", "browser.search"]
    assert [json.loads(a.arguments_json)["query"] for a in episode.actions] == ["a", "b"]
    assert [r.payload for r in episode.records if r.result_of] == [b"result a", b"result b"]
    assert linked(episode) and json.loads(episode.tools_json)[0]["name"] == "browser.search"
    broken = {"messages": row["messages"][:3] + [message("tool", "stray", name="browser.open")]}
    with pytest.raises(SchemaError, match="no pending call"):
        adapt(broken, "harmony_tools")


def test_response_items_become_turns_and_the_next_step_closes_the_trace():
    tool = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}
    row = {
        "responses_create_params": {
            "tools": [tool],
            "input": [
                {"role": "system", "content": "Help."},
                {"role": "user", "content": "Where is it?"},
                {
                    "type": "reasoning",
                    "id": "r",
                    "summary": [{"type": "summary_text", "text": "look"}],
                },
                {
                    "role": "assistant",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "One sec."}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "lookup",
                    "arguments": '{"q": "it"}',
                },
                {"type": "function_call_output", "call_id": "c1", "output": "Paris"},
            ],
        },
        "expected_action": {"type": "message", "content": "It is in Paris."},
    }
    episode = adapt(row, "responses_tools")
    assert [(r.role, r.policy_mask) for r in episode.records][-1] == ("assistant", True)
    (action,) = episode.actions
    assert action.tool_name == "lookup" and json.loads(action.arguments_json) == {"q": "it"}
    assert json.loads(episode.tools_json)[0]["name"] == "lookup"
    assert [r.payload for r in episode.records if r.result_of] == [b"Paris"]
    assert linked(episode)


def test_call_lists_pair_one_result_entry_per_call_and_tools_come_from_the_prompt():
    system = (
        'Tools:\n<tool>[{"name": "a", "description": "d", "parameters": '
        '{"x": {"type": "string"}}, "required": ["x"]}]</tool>\nAnswer in <final></final>.'
    )
    calls = '[{"name": "a", "arguments": {"x": "1"}}, {"name": "a", "arguments": {"x": "2"}}]'
    results = json.dumps(
        [{"name": "a", "arguments": {"x": "1"}, "results": 1}, {"name": "a", "results": {"y": 2}}]
    )
    row = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": f"Plan.\n<call>{calls}</call>"},
            {"role": "tool", "content": results},
            {"role": "assistant", "content": "<final>Done</final>"},
        ]
    }
    episode = adapt(row, "call_list_tools")
    assert [json.loads(a.arguments_json)["x"] for a in episode.actions] == ["1", "2"]
    assert [r.payload for r in episode.records if r.result_of] == [b"1", b'{"y": 2}']
    tool = json.loads(episode.tools_json)[0]
    assert tool["parameters"]["properties"]["x"] == {"type": "string"}
    assert tool["parameters"]["required"] == ["x"] and linked(episode)


def test_the_customers_own_tool_calls_stay_customer_text():
    row = {
        "messages": [
            {"role": "system", "content": "Policy."},
            {"role": "user", "content": "Help"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c", "name": "lookup", "arguments": {}, "requestor": "assistant"}
                ],
            },
            {"role": "tool", "content": "found"},
            {
                "role": "user",
                "content": "Checking my phone.",
                "tool_calls": [{"id": "d", "name": "status", "arguments": {}, "requestor": "user"}],
            },
            {"role": "tool", "content": "Airplane Mode: ON"},
            {"role": "assistant", "content": "Turn it off."},
        ]
    }
    episode = adapt(row, "device_call_tools")
    assert [a.tool_name for a in episode.actions] == ["lookup"]
    texts = [r.payload for r in episode.records if r.role == "user"]
    assert any(b"Airplane Mode: ON" in text for text in texts) and linked(episode)


def test_a_dialogue_prefix_with_its_next_step_trains_only_the_next_step():
    row = {
        "messages": [
            {"role": "system", "content": "Policy."},
            {"role": "user", "content": "Help"},
            {"role": "assistant", "content": "Sure."},
            {"role": "user", "content": "Thanks"},
        ],
        "answer": {
            "role": "assistant",
            "content": "",
            "thinking": "plan",
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    }
    episode = adapt(row, "device_call_tools", allow_pending_final_action=True)
    scored = [r for r in episode.records if r.policy_mask]
    assert len(scored) == 2 and all(r.role == "assistant" for r in scored)
    assert [a.tool_name for a in episode.actions] == ["f"]
    assert b"<think>" in scored[0].payload


def test_blank_user_turns_json_message_strings_and_empty_call_fields():
    blank = {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Real question"},
            {"role": "assistant", "content": "Answer", "tool_calls": ""},
        ]
    }
    with pytest.raises(SchemaError, match="no content"):
        adapt(blank, "chat")
    episode = adapt(blank, "chat", drop_empty_user_turns=True)
    assert [r.role for r in episode.records] == ["user", "assistant"]
    strings = {"messages": [json.dumps(m) for m in blank["messages"][1:]]}
    assert len(adapt(strings, "chat").records) == 2


def test_function_role_replies_in_react_logs_are_the_environment_turn():
    row = {
        "conversations": [
            {"from": "system", "value": "You are AutoGPT."},
            {"from": "user", "value": "Find a meme."},
            {"from": "assistant", "value": "Thought: search\nAction: memes\nAction Input: {}"},
            {"from": "function", "value": '{"error": "", "response": "ok"}'},
            {"from": "assistant", "value": "Thought: done\nAction: Finish\nAction Input: {}"},
        ]
    }
    episode = adapt(row, "react_function_agent", stream="environment_trajectories")
    assert [r.role for r in episode.records].count("environment") == 1


def test_call_roles_spelled_with_a_space_and_calls_listed_in_one_message():
    row = {
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "tool call",
                "content": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}],
            },
            {"role": "tool_response", "content": "1"},
            {"role": "tool_response", "content": "2"},
            {"role": "assistant", "content": "done"},
        ]
    }
    episode = adapt(row, "simulated_tool_roles")
    assert [a.tool_name for a in episode.actions] == ["a", "b"] and linked(episode)


def test_a_reply_that_is_one_fenced_block_of_calls_is_a_call_and_other_code_is_text():
    code = "```python\nbook(when=datetime(2023, 5, 12, 14, 0), note=...)\n```"
    row = {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": code},
            {"role": "tool", "content": "ok"},
            {"role": "assistant", "content": "```python\nx = 1\nprint(x)\n```"},
        ]
    }
    episode = adapt(row, "python_call_tools")
    (action,) = episode.actions
    assert json.loads(action.arguments_json) == {
        "when": "datetime(2023, 5, 12, 14, 0)",
        "note": "...",
    }
    assert linked(episode)


def test_llamafactory_calls_keep_their_thinking_and_results_lose_their_tags():
    row = {
        "system": "You are an agent.",
        "tools": json.dumps(
            [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": json.dumps({"type": "object"})},
                }
            ]
        ),
        "conversations": [
            {"from": "human", "value": "go"},
            {
                "from": "function_call",
                "value": '<think>\nplan\n</think>\n<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>',
            },
            {"from": "observation", "value": "<tool_response>\nok\n</tool_response>"},
            {"from": "gpt", "value": "done"},
        ],
    }
    episode = adapt(row, "simulated_tool_roles")
    assert [r.payload for r in episode.records if r.result_of] == [b"ok"]
    assert any(b"plan" in r.payload for r in episode.records if r.policy_mask and not r.action_id)
    assert json.loads(episode.tools_json)[0]["function"]["parameters"] == {"type": "object"}
    assert linked(episode)


def test_one_response_holding_every_parallel_result_is_split_per_call():
    calls = "".join(
        f'<tool_call>{{"name": "f", "arguments": {{"a": {i}}}}}</tool_call>' for i in range(2)
    )
    packed = json.dumps([{"name": "f", "results": {"v": 1}}, {"name": "f", "results": "two"}])
    episode = adapt(hermes_row(calls, f"<tool_response>{packed}</tool_response>"), "hermes_tools")
    assert [r.payload for r in episode.records if r.result_of] == [b'{"v": 1}', b'"two"']


def test_tool_calls_written_inside_one_assistant_output_are_split_out_with_their_results():
    row = {
        "instruction": "Think, search and answer.",
        "input": "Who wrote it?",
        "output": (
            "<think> look it up </think><search>who wrote it</search>\n\n"
            "<result>Ann wrote it.</result> <think>done</think><answer>Ann</answer>"
        ),
    }
    episode = adapt(row, "tag_trace_tools")
    (action,) = episode.actions
    assert action.tool_name == "search" and json.loads(action.arguments_json) == {
        "query": "who wrote it"
    }
    assert [r.payload for r in episode.records if r.result_of] == [b"Ann wrote it."]
    assert episode.records[-1].role == "assistant" and episode.records[-1].policy_mask
    assert linked(episode)


def test_xml_function_calls_keep_parameter_types():
    call = (
        "<tool_call>\n<function=book>\n<parameter=city>\nParis\n</parameter>\n"
        "<parameter=nights>\n3\n</parameter>\n<parameter=rooms>\n[1, 2]\n</parameter>\n"
        "<parameter=pets>\nfalse\n</parameter>\n</function>\n</tool_call>"
    )
    episode = adapt(hermes_row(call, "<tool_response>\nok\n</tool_response>"), "hermes_tools")
    (action,) = episode.actions
    assert action.tool_name == "book"
    assert json.loads(action.arguments_json) == {
        "city": "Paris",
        "nights": 3,
        "rooms": [1, 2],
        "pets": False,
    }
    assert linked(episode)
