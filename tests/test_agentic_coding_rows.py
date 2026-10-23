"""Agentic coding rows in the shapes publishers actually release them."""

import json

import pytest

from megatoken.data import AdapterContext, SchemaError, adapt_row
from megatoken.generation.codec import ByteCodec


def adapt(row, adapter, stream="executed_trajectories", **options):
    return adapt_row(
        row,
        adapter,
        AdapterContext(
            source_id="fixture",
            revision="rev",
            uri="hf://fixture",
            stream=stream,
            options=options,
            codec=ByteCodec.byte_fallback(),
            max_record_tokens=1024,
        ),
    )


def kinds(episode):
    return [
        "action" if r.action_id else "result" if r.result_of else "text" for r in episode.records
    ]


def linked(episode):
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


def open_swe_call(name, **arguments):
    return {
        "id": f"chatcmpl-tool-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_open_swe_traces_results_carry_no_ids_and_the_last_call_has_no_result():
    # The published shape: columnar null fields on every message, results without a
    # tool_call_id, and a final finish call that never gets an observation.
    def message(role, content, calls=None):
        return {
            "role": role,
            "content": content,
            "reasoning_content": "Let me look." if role == "assistant" else None,
            "tool_calls": calls,
        }

    row = {
        "instance_id": "more-itertools_more-itertools_pr566",
        "repo": "more-itertools/more-itertools",
        "language": "python",
        "trajectory_id": "63bea7c5-1e45-4c47-b4d0-f5aa3700fc9e",
        "resolved": 1,
        "metadata": {"category": "feature-request"},
        "messages": [
            message("system", "You are OpenHands agent."),
            message("user", "We are addressing the following issue."),
            message("assistant", "", [open_swe_call("execute_bash", command="ls")]),
            message("tool", "a.py\nb.py"),
            message("assistant", "", [open_swe_call("finish", message="Done.")]),
        ],
    }
    episode = adapt(row, "open_swe")
    assert kinds(episode) == ["text", "text", "text", "action", "result", "text", "action"]
    assert linked(episode) and len(episode.actions) == 2
    assert episode.supervision.verified_success
    assert episode.lineage.family_id == "repository:more-itertools/more-itertools"
    # A source that wants traces to end in a result can still require it.
    with pytest.raises(SchemaError, match="observation is missing"):
        adapt(row, "open_swe", allow_pending_final_action=False)


def test_a_blank_follow_up_user_turn_is_skipped_but_a_blank_task_is_still_refused():
    def row(first):
        return {
            "instance_id": "t",
            "messages": [
                {"role": "user", "content": first},
                {"role": "assistant", "content": "Looking.", "tool_calls": [open_swe_call("ls")]},
                {"role": "tool", "content": "a.py"},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Done."},
            ],
        }

    episode = adapt(row("Fix it."), "open_swe")
    assert kinds(episode).count("text") == 3  # task, the thought, the closing answer
    with pytest.raises(SchemaError, match="has no content"):
        adapt(row(""), "open_swe")


def call(call_id, name="f", **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def tool_use_row(*messages):
    return {
        "id": "t",
        "messages": [{"role": "user", "content": "Do it."}, *messages],
    }


def test_call_ids_that_restart_every_turn_are_separate_calls():
    row = tool_use_row(
        {"role": "assistant", "content": "", "tool_calls": [call("call_1"), call("call_2")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "a"},
        {"role": "tool", "tool_call_id": "call_2", "content": "b"},
        {"role": "assistant", "content": "", "tool_calls": [call("call_1"), call("call_2")]},
        {"role": "tool", "tool_call_id": "call_2", "content": "d"},
        {"role": "tool", "tool_call_id": "call_1", "content": "c"},
        {"role": "assistant", "content": "Done."},
    )
    episode = adapt(row, "chat", stream="tool_use")
    assert len(episode.actions) == 4 and sum(1 for r in episode.records if r.result_of) == 4
    assert linked(episode)
    # An ID reused while its first call is still open is still an error.
    clash = tool_use_row(
        {"role": "assistant", "content": "", "tool_calls": [call("c"), call("c")]},
    )
    with pytest.raises(SchemaError, match="reused"):
        adapt(clash, "chat", stream="tool_use")


def test_one_untagged_result_answers_a_batch_of_parallel_calls_but_separate_ones_pair_in_order():
    calls = {"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]}
    one = tool_use_row(
        calls,
        {"role": "tool", "content": "both"},
        {"role": "assistant", "content": "Done."},
    )
    episode = adapt(one, "chat", stream="tool_use")
    assert len(episode.actions) == 2 and sum(1 for r in episode.records if r.result_of) == 1
    assert linked(episode)
    two = tool_use_row(
        calls,
        {"role": "tool", "content": "first"},
        {"role": "tool", "content": "second"},
        {"role": "assistant", "content": "Done."},
    )
    episode = adapt(two, "chat", stream="tool_use")
    assert sum(1 for r in episode.records if r.result_of) == 2 and linked(episode)


def test_a_result_naming_an_unknown_call_answers_the_only_open_one():
    renamed = tool_use_row(
        {"role": "assistant", "content": "", "tool_calls": [call("a")]},
        {"role": "tool", "tool_call_id": "zzz", "content": "out"},
        {"role": "assistant", "content": "Done."},
    )
    episode = adapt(renamed, "chat", stream="tool_use")
    assert sum(1 for r in episode.records if r.result_of) == 1 and linked(episode)
    ambiguous = tool_use_row(
        {"role": "assistant", "content": "", "tool_calls": [call("a"), call("b")]},
        {"role": "tool", "tool_call_id": "zzz", "content": "out"},
    )
    with pytest.raises(SchemaError, match="no preceding"):
        adapt(ambiguous, "chat", stream="tool_use")


def test_a_chat_dialogue_may_end_on_a_call_unless_the_source_says_otherwise():
    row = tool_use_row({"role": "assistant", "content": "", "tool_calls": [call("a")]})
    assert len(adapt(row, "chat", stream="tool_use").actions) == 1
    with pytest.raises(SchemaError, match="observation is missing"):
        adapt(row, "chat", stream="tool_use", allow_pending_final_action=False)


TERMINUS_CALL = (
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]}'
)


def test_terminus_rows_without_a_task_field_group_their_trials_and_one_screen_closes_a_batch():
    row = {
        "task": None,
        "trial_name": "code_contests-0000__c8ZdrpP",
        "run_id": "b84cca06-f1ef-4191-9c45-87ddf92df94a",
        "original_source": "code_contests",
        "conversations": [
            {"role": "user", "content": "Solve the task. Current terminal state:\nroot@x:/app#"},
            {"role": "assistant", "content": TERMINUS_CALL},
            # Two replies in a row issue two batches; the next screen answers both.
            {"role": "assistant", "content": TERMINUS_CALL.replace("ls", "pwd")},
            {"role": "user", "content": "New Terminal Output:\n\nroot@x:/app# ls"},
            {"role": "assistant", "content": '{"commands": [], "task_complete": true}'},
        ],
    }
    episode = adapt(row, "agent_trove", admit_unknown_outcome=True)
    assert kinds(episode) == ["text", "action", "action", "result", "text"]
    assert linked(episode)
    answered = next(r for r in episode.records if r.result_of)
    assert answered.result_of == episode.actions[-1].action_id
    # The trial's random suffix is not part of its family.
    assert episode.lineage.family_id == "code_contests-0000"
    sibling = {**row, "trial_name": "code_contests-0000__ZZZZZZZ", "run_id": "other"}
    assert (
        adapt(sibling, "agent_trove", admit_unknown_outcome=True).lineage.family_id
        == episode.lineage.family_id
    )


def test_a_graded_terminal_trial_is_verified_only_through_the_graded_profile():
    row = {
        "task": "task_1",
        "trial": "t",
        "reward": 1.0,
        "messages": [
            {"role": "user", "content": "Solve the task. Current terminal state:\nroot@x:/app#"},
            {"role": "assistant", "content": TERMINUS_CALL},
            {"role": "user", "content": "New Terminal Output:\n\nroot@x:/app# ls"},
            {"role": "assistant", "content": '{"commands": [], "task_complete": true}'},
        ],
    }
    stream = "verified_trajectories"
    assert adapt(row, "agent_trove_graded", stream=stream).supervision.verified_success
    # The plain profile records a publisher outcome, not a test verdict.
    assert not adapt(
        row, "agent_trove", stream="executed_trajectories"
    ).supervision.verified_success
    failed = {**row, "reward": 0.0}
    assert not adapt(
        failed, "agent_trove_graded", stream="executed_trajectories"
    ).supervision.verified_success


def session(adapter, events_key, events, **options):
    row = {"source_file": "s/1.jsonl", events_key: events}
    return adapt(row, adapter, admit_unknown_outcome=True, **options)


def test_claude_code_log_lines_become_one_dialogue_with_linked_results():
    usage = {"input_tokens": 1}
    events = [
        {"type": "mode", "mode": "normal"},
        {"type": "user", "uuid": "u1", "message": {"role": "user", "content": "fix the bug"}},
        {"type": "attachment", "attachment": {"type": "skill_listing"}},
        # One model reply, written as one line per block.
        {
            "type": "assistant",
            "message": {
                "id": "m1",
                "role": "assistant",
                "usage": usage,
                "content": [{"type": "thinking", "thinking": "Look first.", "signature": "x"}],
            },
        },
        {
            "type": "assistant",
            "message": {
                "id": "m1",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
                ],
            },
        },
        {
            "type": "assistant",
            "isSidechain": True,
            "message": {
                "id": "s1",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "side", "name": "Bash", "input": {"command": "x"}}
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}],
            },
        },
        # A replayed call and a result whose call was compacted away.
        {
            "type": "assistant",
            "message": {
                "id": "m2",
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                    {"type": "text", "text": "Done."},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "gone",
                        "content": [{"type": "text", "text": "old"}],
                    }
                ],
            },
        },
    ]
    episode = session("claude_code_session", "events", events)
    assert kinds(episode) == ["text", "text", "action", "result", "text", "text"]
    assert linked(episode) and len(episode.actions) == 1
    assert any(r.payload == b"<think>\nLook first.\n</think>" for r in episode.records)


def test_pi_codex_and_opencode_sessions_pair_results_by_call_id():
    pi = [
        {"type": "session", "id": "s", "cwd": "/w"},
        {
            "type": "message",
            "message": {"role": "developer", "content": [{"type": "text", "text": "You are pi."}]},
        },
        {
            "type": "message",
            "message": {"role": "user", "content": [{"type": "text", "text": "build it"}]},
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hm"},
                    {
                        "type": "toolCall",
                        "id": "c1",
                        "name": "bash",
                        "arguments": {"command": "pwd"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "toolResult",
                "toolCallId": "c1",
                "content": [{"type": "text", "text": "/w"}],
            },
        },
    ]
    episode = session("pi_session", "events", pi)
    assert kinds(episode) == ["text", "text", "text", "action", "result"] and linked(episode)

    def item(payload):
        return {"type": "response_item", "payload": payload}

    codex = [
        {"type": "event_msg", "payload": {"type": "user_message", "message": "dup"}},
        item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "add a flag"}],
            }
        ),
        item(
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"ls"}',
                "call_id": "a",
            }
        ),
        item(
            {
                "type": "custom_tool_call",
                "name": "apply_patch",
                "input": "*** Begin Patch\n*** End Patch",
                "call_id": "b",
            }
        ),
        item({"type": "function_call", "name": "", "arguments": "{}"}),  # partial: no id
        item({"type": "function_call_output", "call_id": "a", "output": "x.py"}),
        item({"type": "custom_tool_call_output", "call_id": "b", "output": "Done!"}),
        item(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Added."}],
            }
        ),
    ]
    episode = session("codex_session", "events", codex)
    assert kinds(episode) == ["text", "action", "action", "result", "result", "text"]
    assert linked(episode)
    assert json.loads(episode.actions[1].arguments_json) == {
        "input": "*** Begin Patch\n*** End Patch"
    }

    document = {
        "messages": [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "run tests"}]},
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "step-start"},
                    {
                        "type": "tool",
                        "callID": "k1",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {"command": "pytest"},
                            "output": "1 passed",
                        },
                    },
                    {"type": "text", "text": "All green."},
                ],
            },
        ]
    }
    episode = session("opencode_session", "document", document)
    assert kinds(episode) == ["text", "action", "result", "text"] and linked(episode)


def test_mini_swe_agent_v2_rows_with_a_named_function_and_a_closing_exit_record():
    def call(call_id, command):
        return {"id": call_id, "function": "bash", "arguments": {"command": command}}

    row = {
        "instance_id": "astropy__astropy-12907",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the PR."},
            {"role": "assistant", "content": "Look.", "tool_calls": [call("t1", "ls")]},
            {"role": "tool", "content": "a.py", "tool_call_id": "t1", "function": "bash"},
            {"role": "assistant", "content": "Submit.", "tool_calls": [call("t2", "echo DONE")]},
            {"role": "tool", "content": "DONE", "tool_call_id": "t2"},
            {"role": "exit", "content": "diff --git a/a.py b/a.py", "extra": {}},
        ],
    }
    episode = adapt(row, "open_swe", admit_unknown_outcome=True)
    assert kinds(episode) == [
        "text",
        "text",
        "text",
        "action",
        "result",
        "text",
        "action",
        "result",
    ]
    assert linked(episode)


def test_a_call_id_reused_after_its_result_is_a_new_call_and_parallel_calls_are_one_batch():
    def call(command):
        return {
            "id": "bash:1",
            "type": "function",
            "function": {"name": "bash", "arguments": {"command": command}},
        }

    row = {
        "instance_id": "django__django-1",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [call("ls")]},
            {"role": "tool", "tool_call_id": "bash:1", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [call("pwd")]},
            {"role": "tool", "tool_call_id": "bash:1", "content": "/"},
        ],
    }
    episode = adapt(row, "open_swe", admit_unknown_outcome=True)
    assert kinds(episode) == ["text", "action", "result", "action", "result"] and linked(episode)
    assert len({a.call_id for a in episode.actions}) == 2

    # MEnvData: one reply holds several calls, then each call's output comes as a tool message.
    reply = (
        "Let me look.\n\n<functions>\n<function=execute_bash>\n<parameter=command>ls</parameter>\n"
        "</function>\n<function=execute_bash>\n<parameter=command>pwd</parameter>\n</function>"
    )
    row = {
        "instance_id": "aws__s2n-1",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": reply},
            {"role": "tool", "content": "Exit code: 0\n[STDOUT]\na.c"},
            {"role": "tool", "content": "Exit code: 0\n[STDOUT]\n/testbed"},
            {"role": "assistant", "content": "Done."},
        ],
    }
    episode = adapt(row, "openhands_xml", admit_unknown_outcome=True)
    assert kinds(episode) == ["text", "action", "result", "text", "text"] and linked(episode)
    assert json.loads(episode.actions[0].arguments_json)["calls"][1]["arguments"] == {
        "command": "pwd"
    }


def test_calls_held_in_parts_and_nested_runs_in_one_task_row():
    call = json.dumps({"name": "bash", "arguments": {"command": "ls"}})
    messages = [
        {"role": "user", "content": "Fix the issue."},
        {
            "role": "assistant",
            "content": "",
            "parts": [{"type": "text", "text": "Let me look."}, {"type": "python", "text": call}],
        },
        {"role": "user", "content": "<returncode>0</returncode>\n<output>\na.py\n</output>"},
        {"role": "assistant", "content": "", "parts": [{"type": "text", "text": "Done."}]},
    ]
    row = {"instance_id": "x__y.1", "resolved": True, "messages": json.dumps(messages)}
    episode = adapt(row, "parts_agent", "verified_trajectories")
    assert kinds(episode) == ["text", "text", "action", "result", "text"] and linked(episode)
    assert episode.supervision.verified_success

    from megatoken.data.sources import _exploded

    task = {
        "instance_id": "paramiko__1",
        "patch": "gold",
        "trajectories": json.dumps(
            [
                {"messages": messages, "is_correct": True, "patch": "run0"},
                {"messages": messages, "is_correct": False},
            ]
        ),
    }
    rows = list(_exploded(iter([task]), "trajectories"))
    assert [(r["explode_index"], r["is_correct"], r["patch"]) for r in rows] == [
        (0, True, "run0"),
        (1, False, "gold"),
    ]
    assert rows[0]["instance_id"] == "paramiko__1" and "trajectories" not in rows[0]


def test_the_agentic_review_policy_waives_only_the_recursion_depth_and_overlap_diagnostics():
    from pathlib import Path

    from megatoken.data.cli import _parser_review_policy

    root = Path(__file__).parents[1] / "configs/data"
    agentic = _parser_review_policy(str(root / "parser-review.agentic.json"))
    default = _parser_review_policy(str(root / "parser-review.json"))
    codes = [
        "depth_budget_exhausted",
        "tree_depth_budget_exhausted",
        "node_budget_exhausted",
        "crossing_protected_regions",
    ]
    assert agentic.classify(codes) == ("node_budget_exhausted",)
    assert default.classify(codes) == tuple(sorted(codes))


def test_a_missing_token_marked_on_a_decoded_string_seam_stays_inside_the_source():
    # A tool call whose argument is TypeScript without its last semicolon: the parser's
    # zero-width "missing ;" mark sits on the seam of the JSON string it was decoded from.
    from megatoken.data.parsing.engine import parse_content

    code = 'export function createOverlayHandler(options: OverlayServerOptions = {}): (req: Request) => Response {\n  const cursorPath = options.cursorPath ?? wwmCursorPath();\n  const html = renderCursorOverlayHtml({ label: options.label });\n\n  return (req: Request): Response => {\n    const url = new URL(req.url);\n    if (url.pathname === "/" || url.pathname === "/index.html") {\n      return new Response(html, { headers: { "content-type": "text/html; charset=utf-8" } });\n    }\n    if (url.pathname === "/cursor") {\n      // The agent turn writes the live position to cursorPath; the overlay polls here.\n      const pos = readCursor(cursorPath);\n      return new Response(JSON.stringify(pos ?? null), {\n        headers: { "content-type": "application/json", "cache-control": "no-store" },\n      });\n    }\n    return new Response("not found", { status: 404 });\n  };\n}'
    payload = json.dumps(
        {
            "function": {
                "arguments": {"file_path": "/w/a.ts", "new_string": code},
                "name": "write_file",
            }
        }
    ).encode()
    result = parse_content(payload)
    assert all(0 <= d.start <= d.end <= len(payload) for d in result.diagnostics)


def test_each_listed_hub_parquet_file_is_read_with_its_own_columns_in_small_batches(
    monkeypatch, tmp_path
):
    # One file's messages carry {role, content, loss}, the next {role, content, type, loss}:
    # reading them as one dataset casts the second to the first file's columns and fails. Rows
    # are decoded in byte-budgeted batches, so row groups past Arrow's 2 GiB limit still read.
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq

    import megatoken.data.sources as sources
    from megatoken.data.sources import SourceFile, SourceSpec, TermsReview, iter_source_rows

    message = {"role": "user", "content": "hi", "loss": False}
    local = {"a.parquet": tmp_path / "a.parquet", "b.parquet": tmp_path / "b.parquet"}
    pq.write_table(
        pa.Table.from_pylist([{"m": [message], "n": i} for i in range(2)]), local["a.parquet"]
    )
    pq.write_table(
        pa.Table.from_pylist([{"m": [{**message, "type": "text"}], "n": i} for i in range(2)]),
        local["b.parquet"],
    )
    opened = []

    class FakeFileSystem:
        def open(self, address, mode, block_size=None):
            opened.append(address)
            return local[address.rsplit("/", 1)[-1]].open(mode)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFileSystem)
    monkeypatch.setattr(sources, "PARQUET_BATCH_BYTES", 1)
    source = SourceSpec(
        source_id="s",
        uri="hf://datasets/o/r",
        format="hf",
        adapter="chat",
        stream="chat",
        revision="a" * 40,
        files=(SourceFile("a.parquet"), SourceFile("b.parquet")),
        max_rows=3,
        terms=TermsReview(approved=True, license="mit", reviewed_by="t", reviewed_at="2026-10-22"),
    )
    rows = list(iter_source_rows(source))
    assert [(i, r["n"], "type" in r["m"][0]) for i, r in rows] == [
        (0, 0, False),
        (1, 1, False),
        (2, 0, True),
    ]
    assert opened == [
        f"hf://datasets/o/r@{'a' * 40}/a.parquet",
        f"hf://datasets/o/r@{'a' * 40}/b.parquet",
    ]


@pytest.mark.parametrize("suffix", [".jsonl", ".jsonl.gz", ".jsonl.zst"])
def test_json_lines_preserve_changing_nested_fields_and_resume(monkeypatch, suffix):
    import gzip
    import io

    import huggingface_hub
    import zstandard

    from megatoken.data.sources import SourceFile, SourceSpec, TermsReview, iter_source_rows

    rows = [
        {"report": {"passed": 1, "per_file": {"a.py": True}}},
        {"report": {"passed": 0, "error": "failed", "per_file": "unavailable"}},
        {"messages": [{"role": "user", "content": "hi", "type": "text"}]},
    ]
    data = ("\n".join(json.dumps(row) for row in rows) + "\n\n").encode()
    if suffix.endswith(".gz"):
        data = gzip.compress(data)
    elif suffix.endswith(".zst"):
        data = zstandard.ZstdCompressor().compress(data)
    opened = []

    class FakeFileSystem:
        def open(self, address, mode, block_size=None):
            opened.append(address)
            return io.BytesIO(data)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFileSystem)
    source = SourceSpec(
        source_id="s",
        uri="hf://datasets/o/r",
        format="hf",
        adapter="chat",
        stream="chat",
        revision="b" * 40,
        files=(SourceFile("data/x" + suffix),),
        terms=TermsReview(approved=True, license="mit", reviewed_by="t", reviewed_at="2026-10-22"),
    )
    assert list(iter_source_rows(source)) == list(enumerate(rows))
    assert list(iter_source_rows(source, start_row=1)) == list(enumerate(rows[1:], 1))
    assert opened == [f"hf://datasets/o/r@{'b' * 40}/data/x{suffix}"] * 2


def test_a_row_filter_can_require_a_nonempty_conversation():
    from megatoken.data.quality import RowFilterReject, validate_row_filters

    rule = [{"path": "messages", "op": "nonempty"}]
    validate_row_filters({"messages": [{"role": "user"}]}, rule)
    validate_row_filters({"messages": "text"}, rule)
    for row in ({"messages": []}, {"messages": ""}, {"messages": None}, {}, {"messages": 3}):
        with pytest.raises(RowFilterReject):
            validate_row_filters(row, rule)


def test_codex_rollout_skips_stray_lines_and_null_content():
    def item(payload):
        return {"type": "response_item", "payload": payload}

    events = [
        "not an event",
        None,
        item({"type": "message", "role": "user", "content": None}),
        item({"type": "reasoning", "summary": None}),
        item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "add a flag"}],
            }
        ),
        item(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Added."}],
            }
        ),
    ]
    episode = session("codex_session", "events", events)
    assert kinds(episode) == ["text", "text"]
