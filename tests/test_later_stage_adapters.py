"""Adapter shapes that the later stages need: file lists, observed context, derived evidence."""

from __future__ import annotations

import json

import pytest

from megatoken.data.adapters import AdapterContext, SchemaError, adapt_row
from megatoken.data.schema import Lineage
from megatoken.generation.codec import ByteCodec


def context(stream: str = "chat", **options) -> AdapterContext:
    return AdapterContext(
        "fixture", "rev", "local", stream, codec=ByteCodec.byte_fallback(), options=options
    )


def roles(episode):
    return [(record.role, record.policy_mask) for record in episode.records]


def test_a_repository_row_reads_the_issue_the_discussion_the_files_then_scores_each_patch():
    row = {
        "repo": "o/r",
        "problem_statement": "Crash on empty input",
        "hints_text": "Reproduced on 2.1",
        "before": [
            {"path": "src/a.py", "content": "def f(x):\n    return x[0]\n"},
            "plain note",
            {"path": "empty.txt", "content": ""},
        ],
        "patch": [
            {"path": "src/a.py", "patch": "--- a/src/a.py\n+++ b/src/a.py\n"},
            {"path": "tests/t.py", "patch": "--- a/tests/t.py\n+++ b/tests/t.py\n"},
        ],
    }
    episode = adapt_row(row, "repository", context("issue_pr_diffs"))
    assert roles(episode) == [
        ("user", False),
        ("document", False),
        ("document", False),
        ("document", False),
        ("assistant", True),
        ("assistant", True),
    ]
    assert [record.source_span.field for record in episode.records] == [
        "problem_statement",
        "hints_text",
        "before[0]",
        "before[1]",
        "patch[0]",
        "patch[1]",
    ]
    assert [record.provenance for record in episode.records[1:4]] == ["natural"] * 3
    assert episode.records[2].payload == b"def f(x):\n    return x[0]\n"
    assert episode.lineage.family_id == "repository:o/r"


def test_a_repository_row_without_files_still_places_the_issue_before_the_patch():
    row = {"repo": "o/r", "instance_id": "o__r-1", "issue": "bug", "patch": "diff --git a/x b/x\n"}
    episode = adapt_row(row, "repository", context("issue_pr_diffs"))
    assert roles(episode) == [("user", False), ("assistant", True)]


def test_a_repository_row_needs_a_patch():
    with pytest.raises(SchemaError, match="patch"):
        adapt_row({"repo": "o/r", "issue": "bug"}, "repository", context())
    with pytest.raises(SchemaError, match="empty"):
        adapt_row({"repo": "o/r", "issue": "bug", "patch": []}, "repository", context())


def test_a_file_list_item_without_content_is_an_error_not_a_silent_gap():
    row = {"repo": "o/r", "issue": "bug", "before": [{"path": "a.py"}], "patch": "p"}
    with pytest.raises(SchemaError, match="before\\[0\\]"):
        adapt_row(row, "repository", context())


def test_fim_reads_retrieved_context_before_the_prefix_and_scores_only_the_middle():
    row = {
        "context": [{"file_path": "util.py", "content": "def helper():\n    pass\n"}],
        "prefix": "def g():\n    ",
        "suffix": "\n",
        "middle": "return helper()",
        "repo": "o/r",
    }
    episode = adapt_row(row, "fim", context("long_context_fim"))
    assert roles(episode) == [
        ("document", False),
        ("document", False),
        ("document", False),
        ("assistant", True),
    ]
    assert episode.records[0].payload.startswith(b"def helper")


def test_an_artifact_may_omit_its_request_when_the_profile_says_so():
    row = {"source_document": "x=1", "target": "x = 1\n", "repo": "o/r"}
    episode = adapt_row(row, "code_rewrite", context("synthetic_code_rewrite"))
    assert roles(episode) == [("document", False), ("assistant", True)]
    assert episode.records[-1].provenance == "teacher"
    natural = adapt_row(row, "human_code_rewrite", context("synthetic_code_rewrite"))
    assert natural.records[-1].provenance == "natural"
    with pytest.raises(SchemaError, match="prompt"):
        adapt_row(row, "artifact", context())


@pytest.mark.parametrize("flag", [True, "True", 1.0, "pass"])
def test_a_recorded_pass_makes_a_tested_artifact_a_verified_success(flag):
    row = {"prompt": "add", "target": "def add(a, b): return a + b", "correct": flag}
    episode = adapt_row(
        row,
        "tested_artifact",
        AdapterContext("k", "rev", "local", "tested_artifacts", field_map={"outcome": "correct"}),
    )
    assert episode.supervision.verified_success
    assert episode.supervision.verification == "publisher_tests"


@pytest.mark.parametrize("flag", [False, "False", 0.0])
def test_a_recorded_failure_is_never_promoted(flag):
    row = {"prompt": "add", "target": "x", "correct": flag}
    episode = adapt_row(
        row,
        "tested_artifact",
        AdapterContext("k", "rev", "local", "tested_artifacts", field_map={"outcome": "correct"}),
    )
    assert episode.supervision.outcome == "failure" and not episode.supervision.verified_success


def test_a_tool_result_block_in_the_ai_sdk_spelling_answers_its_call():
    row = {
        "messages": [
            {"role": "user", "content": "Which focus?"},
            {
                "role": "assistant",
                "content": "I will ask.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "question", "arguments": '{"q": "which?"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "c1",
                        "toolName": "question",
                        "output": {"type": "text", "value": "Both."},
                    }
                ],
            },
            {"role": "assistant", "content": "Then I cover both."},
        ]
    }
    episode = adapt_row(row, "chat", context("recovery_clarification"))
    (result,) = [record for record in episode.records if record.result_of]
    assert result.payload == b"Both." and result.role == "tool" and not result.policy_mask


def editor_trajectory(*, outcome=True, shell_only=False):
    def call(name, **arguments):
        return {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    steps = [
        ("find", call("execute_bash", command="grep -rn socket .")),
        ("view", call("str_replace_editor", command="view", path="/w/a.py")),
        (
            "edit",
            call(
                "str_replace_editor",
                command="str_replace",
                path="/w/a.py",
                old_str="a",
                new_str="b",
            ),
        ),
        (
            "again",
            call(
                "str_replace_editor",
                command="str_replace",
                path="/w/a.py",
                old_str="b",
                new_str="c",
            ),
        ),
        ("new", call("str_replace_editor", command="create", path="/w/b.py", file_text="x")),
    ]
    if shell_only:
        steps = [step for step in steps if step[1]["function"]["name"] == "execute_bash"]
    messages = [{"role": "user", "content": "Fix it."}]
    for note, tool_call in steps:
        messages.append({"role": "assistant", "content": note, "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": f"result of {note}"})
    messages.append({"role": "assistant", "content": "Done."})
    return {"messages": messages, "resolved": outcome, "instance_id": "o__r-1"}


def evidence_context(**options):
    return context(
        "long_verified_dependencies", positional_call_ids=True, path_dependencies=True, **options
    )


def test_an_edit_depends_on_the_first_result_that_showed_its_file():
    episode = adapt_row(editor_trajectory(), "open_swe", evidence_context())
    records = {record.record_id: record for record in episode.records}
    pairs = [
        (records[d.writer_id].payload, records[d.reader_id].payload[:40], d.kind)
        for d in episode.dependencies
    ]
    # The edit and the second edit both read the view; the new file was never shown.
    assert [writer for writer, _, _ in pairs] == [b"result of view"] * 2
    assert all(kind == "file_read_to_edit" for _, _, kind in pairs)
    assert all(
        records[d.reader_id].policy_mask and records[d.reader_id].role == "assistant"
        for d in episode.dependencies
    )
    assert episode.supervision.verified_success


def test_shell_commands_and_failed_runs_create_no_derived_dependencies():
    shell = adapt_row(editor_trajectory(shell_only=True), "open_swe", evidence_context())
    assert shell.dependencies == ()
    failed = adapt_row(
        editor_trajectory(outcome=False),
        "open_swe",
        evidence_context(),
    )
    assert failed.dependencies == ()


def test_derived_dependencies_are_off_unless_requested():
    episode = adapt_row(
        editor_trajectory(), "open_swe", context("verified_trajectories", positional_call_ids=True)
    )
    assert episode.dependencies == ()


@pytest.mark.parametrize("spelling", ["o/r", "github.com/o/r", "https://github.com/O/R.git"])
def test_one_repository_has_one_lineage_key_however_a_source_spells_it(spelling):
    lineage = Lineage(source_id="s", revision="r", family_id="f", repository=spelling)
    assert "repository:o/r" in lineage.family_keys


def test_ai_sdk_tool_definitions_are_read_as_function_tools():
    row = {
        "messages": [
            {"role": "user", "content": "Ask me."},
            {
                "role": "assistant",
                "content": "Asking.",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "question", "arguments": '{"q": "which?"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "Both."},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": [
            {
                "id": "question",
                "description": "Ask the user",
                "inputSchema": {"jsonSchema": {"type": "object", "properties": {"q": {}}}},
            }
        ],
    }
    episode = adapt_row(row, "chat", context("recovery_clarification"))
    (tool,) = json.loads(episode.tools_json)
    assert tool["function"]["name"] == "question"
    assert tool["function"]["parameters"]["properties"] == {"q": {}}
    assert [action.tool_name for action in episode.actions] == ["question"]
