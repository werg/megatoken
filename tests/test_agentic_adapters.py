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


def test_hyphenated_tool_result_blocks_with_output_values_pair_with_their_call():
    call = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
    row = {
        "instance_id": "i1",
        "repo": "o/r",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "reading", "tool_calls": [call]},
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "c1",
                        "toolName": "read",
                        "output": {"type": "text", "value": "file text"},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ],
    }
    episode = adapt(row, "open_swe", "executed_trajectories", assume_outcome="success")
    result = next(r for r in episode.records if r.result_of)
    assert result.payload == b"file text" and result.provenance == "executed"
    assert linked(episode)


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
