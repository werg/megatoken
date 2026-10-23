"""The later-stage manifest: pinned, licence-recorded, and every source readable by its adapter.

Each fixture row below has the shape of the real pinned file (field names and nesting taken
from rows read on 2026-10-22), cut down to a few lines. Nothing here touches the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from megatoken import cloud
from megatoken.data.adapters import PROFILES, AdapterContext, adapt_row
from megatoken.data.quality import QualityPolicy, validate_row_filters
from megatoken.data.recipes import stage_recipe
from megatoken.data.sharding import plan_shards, row_cost
from megatoken.data.sources import load_source_manifest
from megatoken.generation.codec import ByteCodec

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs/data"
SOURCES = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.later-stages.json")}
OTHERS = [
    s
    for name in ("sources.agentic.json", "sources.agentic-gated.json", "sources.open.json")
    for s in load_source_manifest(CONFIGS / name)
]

STREAMS = {
    "M1": (
        "issue_pr_sequences",
        "connected_repositories",
        "code_evolution",
        "build_logs",
        "long_context",
    ),
    "M2": ("grounded_artifacts", "issue_pr_diffs", "long_context_fim"),
    "LC": ("long_verified_dependencies", "long_natural_code_replay", "mixed_quality_episodes"),
    "S": (
        "recovery_clarification",
        "tested_artifacts",
        "verified_successful_multiturn",
        "natural_document_replay",
    ),
    "P1": ("synthetic_code_rewrite", "environment_trajectories"),
}

CODE = "def f(x):\n    return x * 2\n"
PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
EDITOR_TOOLS = json.dumps(
    [
        {
            "type": "function",
            "function": {"name": "str_replace_editor", "parameters": {"type": "object"}},
        }
    ]
)


def editor_call(**arguments):
    return {"function": {"name": "str_replace_editor", "arguments": json.dumps(arguments)}}


def editor_messages(*, with_ids=False):
    """A resolved run: view a file, edit it, finish. Calls carry no ids in most releases."""
    calls = [
        editor_call(command="view", path="/w/a.py"),
        editor_call(command="str_replace", path="/w/a.py", old_str="x", new_str="y"),
    ]
    messages = [{"role": "user", "content": "<issue_description>Fix a.py</issue_description>"}]
    for number, call in enumerate(calls):
        if with_ids:
            call = {"id": f"c{number}", "type": "function", **call}
        messages.append({"role": "assistant", "content": f"step {number}", "tool_calls": [call]})
        result = {"role": "tool", "content": f"result {number}"}
        messages.append({**result, "tool_call_id": f"c{number}"} if with_ids else result)
    messages.append({"role": "assistant", "content": "Done."})
    return messages


def openhands_text_messages(first="user"):
    call = "<function=execute_bash>\n<parameter=command>ls</parameter>\n</function>"
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": first, "content": "<uploaded_files>/w</uploaded_files> fix it"},
        {"role": "assistant", "content": f"Looking.\n\n{call}"},
        {"role": "user", "content": "EXECUTION RESULT: a.py"},
        {"role": "assistant", "content": "Done."},
    ]


def sharegpt(*pairs):
    return [{"from": role, "value": text} for role, text in pairs]


ADP_CONVERSATION = sharegpt(
    ("human", "Find the page. [Current URL: None] ===== accessibility tree ====="),
    (
        "gpt",
        'Click it.\n\n<function=browser>\n<parameter=code>click(bid="1")</parameter>\n</function>',
    ),
    ("human", "[Current URL: x] ===== accessibility tree ====="),
    ("gpt", "Done."),
)

ROWS = {
    "swallow-code-v2-rewrite": {
        "old_text": CODE,
        "text": 'def f(x: int) -> int:\n    """Double x."""\n    return x * 2\n',
        "repo_name": "o/r",
        "revision_date": "2024-05-01T00:00:00",
        "is_vendor": False,
        "is_generated": False,
    },
    "pie-perf-slow-to-fast": {
        "input": "print(sum([i for i in range(10)]))",
        "target": "print(45)",
        "problem_id": "p00001",
        "status_v0": "Accepted",
        "status_v1": "Accepted",
        "improvement_frac": 40.0,
    },
    "kodcode-sft-4o": {
        "question": "Add two numbers.",
        "4o_solution": "def add(a, b):\n    return a + b\n",
        "4o_correctness": "True",
        "question_id": "Leetcode_3_I",
    },
    "kodcode-sft-r1": {
        "question": "Add two numbers.",
        "r1_solution": "def add(a, b):\n    return a + b\n",
        "r1_correctness": "True",
        "question_id": "Leetcode_3_I",
    },
    "openr1-verifiable-python-tested": {
        "problem": "Print the sum of two integers.",
        "gold_standard_solution": "```python\nprint(sum(map(int, input().split())))\n```",
        "test_reward": 1.0,
        "problem_id": "vfc_0",
    },
    "s-tested-opencodeinstruct": {
        "input": "Write a function that doubles x.",
        "output": CODE,
        "average_test_score": "1",
    },
    "scale-swe-distilled-v4-pro": {
        "messages": editor_messages(),
        "tools": EDITOR_TOOLS,
        "extra_info": {"instance_id": "o_r_pr1", "success": True, "score": 1.0},
    },
    "swe-lego-real-resolved": {
        "messages": openhands_text_messages(),
        "resolved": True,
        "instance_id": "o__r-1",
        "repo": "o/r",
    },
    "ifeval-like-filtered": {
        "key": 7,
        "prompt": "Write two words.",
        "response": "Hello there.",
        "prompt_level_strict_acc": True,
    },
    "if-multi-constraints-sft": {
        "key": "k1",
        "prompt": "Answer in lowercase.",
        "response": "ok",
        "strict_score": 1.0,
    },
    "openmathinstruct1-correct": {
        "question": "What is 2 + 2?",
        "generated_solution": "Adding gives 4. \\boxed{4}",
        "is_correct": True,
    },
    "gharchive-threads": {
        "id": "1",
        "text": "Title: crash on start\n\nIt crashes.\n\nusername_0: reproduced",
        "source": "gharchive/issue",
        "created": "2023-06-19T06:12:12+00:00",
        "metadata": {"repo": "o/r", "url": "https://github.com/o/r/issues/1", "license": "MIT"},
    },
    "pr-resolve-issue-to-patch": {
        "repo": "o/r",
        "pr_number": 7,
        "task_type": "issue_to_patch",
        "issue_text": "",
        "pr_title": "Fix crash",
        "pr_body": "Closes the crash on start.",
        "gold_diff": PATCH,
    },
    "pr-resolve-comment-to-fix": {
        "repo": "o/r",
        "pr_number": 8,
        "task_type": "comment_to_fix",
        "pr_title": "Add cache",
        "pr_body": "Adds a cache.",
        "gold_diff": PATCH,
        "review_threads": [
            {
                "comment": "Please bound the cache.",
                "path": "a.py",
                "hunk": "@@ -1 +1 @@",
                "resolving_sha": "abc",
                "resolving_diff": PATCH,
            }
        ],
    },
    "pr-traces-10k": {
        "repo_name": "o/r",
        "pr_number": 336,
        "pr": {"title": "Fix max", "body": "Closes #329", "created_at": "2025-05-01T00:00:00Z"},
        "linked_issue": {"title": "Max is wrong", "body": "It is 100."},
        "linked_issue_comments": [],
        "pr_comments": [{"body": "LGTM"}],
        "reviews": [{"body": None}, {"body": "Looks good"}],
        "review_comments": [{"body": "nit", "path": "a.py", "diff_hunk": "@@"}],
        "full_diff": PATCH,
    },
    "commit-chronicle": {
        "repo": "o/r",
        "hash": "662e971e",
        "date": "04.01.2017 17:46:47",
        "message": "support generators",
        "mods": [
            {"change_type": "MODIFY", "old_path": "a.py", "new_path": "a.py", "diff": PATCH},
            {"change_type": "ADD", "old_path": None, "new_path": "b.py", "diff": PATCH},
        ],
    },
    "locus-commit-pool": {
        "payload": {
            "message": "Mark package as abandoned",
            "changed_files": [
                {"path": "README.md", "before_content": "# Pkg\n", "unified_patch": PATCH},
                {"path": "composer.json", "before_content": "{}\n", "unified_patch": PATCH},
            ],
        },
        "meta": {
            "repository_id": "github.com/o/r",
            "commit_oid": "b08a9a30",
            "identity": {"committed_at": "2020-04-02T21:47:21+02:00"},
        },
        "annotations": {"possible_ai_authorship": {"signal": "no_named_signal_detected"}},
    },
    "repository-v2-long": {
        "document_id": "repository_e057f8d1",
        "text": "`src/App.js`\n```javascript\nconsole.log(1);\n```\n`README.md`\n```markdown\n# App\n```\n",
        "part_ends": [10, 20],
        "part_trainable": [True, True],
        "must_not_split": False,
    },
    "code-world-traces-affected": {
        "instance_id": "o__r-1",
        "test_id": "tests/test_a.py::test_x",
        "affected": True,
        "text": "def test_x():\n    assert f(1) == 2\n# trace\nline 1: x = 1\n# fix\n" + PATCH,
        "license": "MIT",
    },
    "swe-gym-failure-logs": {
        "instance_id": "o__r-1",
        "problem_statement": "FAILED tests/test_a.py::test_x - AssertionError",
        "hints_text": "",
        "patch": PATCH,
        "repo": "o/r",
    },
    "swe-synth-failure-logs": {
        "instance_id": "o__r.abc12345.func_basic__x",
        "problem_statement": "E   AssertionError: assert 1 == 2",
        "_metadata": {"reverse_patch_diff": PATCH},
    },
    "longmino-science-pdfs": {
        "id": "doc-1",
        "source": "olmocr_science_pdfs",
        "text": "Chapter 1. " + "Findings follow. " * 30,
        "metadata": {"Source-Url": "s3://x/1.pdf", "olmocr_version": "1"},
    },
    "pg19-books": {
        "short_book_title": "A Book",
        "url": "http://x/1",
        "text": "Chapter I. It was. " * 20,
    },
    "pre-1929-books": {
        "id": "1",
        "source": "x",
        "text": "CHAPTER I. It was. " * 20,
        "metadata": {},
    },
    "s-replay-pg19": {
        "short_book_title": "A Book",
        "url": "http://x/2",
        "text": "Chapter I. It was. " * 20,
    },
    "github-patches": {
        "in_source_id": "Cloud-CV__EvalAI-1233",
        "issue": "Crash when x is empty",
        "before_files": [{"path": "a.py", "content": CODE}],
        "after_files": [{"path": "a.py", "content": CODE}],
        "pr_diff": PATCH,
    },
    "swe-fixer-train-110k": {
        "repo": "o/r",
        "instance_id": "o__r-1",
        "problem_statement": "Crash when x is empty",
        "hints_text": "I can reproduce.",
        "patch": PATCH,
        "test_patch": PATCH,
    },
    "swe-gym-raw": {
        "repo": "o/r",
        "instance_id": "o__r-2",
        "problem_statement": "Crash when x is empty",
        "hints_text": "",
        "patch": PATCH,
    },
    "swe-rebench-v2-prs": {
        "repo": "o/r",
        "instance_id": "o__r-3",
        "problem_statement": "**Title** Fix crash",
        "hints_text": "",
        "pr_description": "fix: crash on empty input",
        "patch": PATCH,
        "FAIL_TO_PASS": ["t"],
    },
    "multi-swe-rl-reupload": {
        "repo": "OpenBLAS",
        "instance_id": "OpenMathLib__OpenBLAS-4729",
        "title": "Fix INF handling in SCAL",
        "body": "fixes #4728",
        "resolved_issues": {"title": ["inf"], "body": ["Result is wrong."], "number": [4728]},
        "fix_patch": PATCH,
    },
    "commitpackft": {
        "commit": "9f7a229a",
        "repos": "o/r",
        "message": "Add retry",
        "old_contents": CODE,
        "new_contents": CODE + "# retry\n",
    },
    "longcite-45k": {
        "prompt": "Answer with citations.\n[Document Start]\n<C0>First fact.<C1>Second fact.\n[Document End]\nWhat?",
        "response": "<statement>Second.<cite>[1-1]</cite></statement>",
    },
    "imported-context-files": {
        "repo_name": "o/r",
        "path": "pkg/util.py",
        "context": "# pkg/helpers.py\ndef helper():\n    pass\n",
        "content": "from pkg.helpers import helper\n\n\ndef run():\n    helper()\n",
    },
    "sourcegraph-context-fim": {
        "repo_name": "o/r",
        "lang": "py",
        "prefix": "def g():\n    ",
        "suffix": "\n",
        "middle": "return helper()",
        "context_items": [
            {
                "content": "def helper():\n    pass\n",
                "file_path": "util.py",
                "rank": 0,
                "score": 1.0,
            }
        ],
    },
    "tiger-fim-midtraining": {
        "metadata": {"repository_url": "https://github.com/o/r", "sample_id": 1},
        "messages": [
            {"role": "user", "content": "Complete the masked function.\n\ndef f():\n    <MASK>\n"},
            {"role": "assistant", "content": "### Reasoning\nReturn one.\n\n    return 1"},
        ],
    },
    "scale-swe-distilled-v32": {
        "data_source": "python-semver_python-semver_pr198",
        "messages": [
            {"role": "system", "content": "You are an agent.", "loss_mask": 0},
            {"role": "user", "content": "<uploaded_files>/w</uploaded_files> fix", "loss_mask": 0},
            {
                "role": "assistant",
                "content": "Looking.\n\n<function=execute_bash>\n<parameter=command>ls</parameter>\n</function>",
                "loss_mask": 1,
            },
            {"role": "tool", "content": "a.py", "loss_mask": 0},
            {"role": "assistant", "content": "Done.", "loss_mask": 1},
        ],
    },
    "opencode-question-tool": {
        "uuid": "u1",
        "tools": [
            {
                "id": "question",
                "description": "Ask the user a question",
                "inputSchema": {
                    "jsonSchema": {
                        "type": "object",
                        "properties": {"questions": {"type": "array"}},
                    }
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Explain this hook."},
            {
                "role": "assistant",
                "content": "I will ask.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "question", "arguments": '{"questions": []}'},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "call_1",
                        "toolName": "question",
                        "output": {"type": "text", "value": "Both aspects."},
                    }
                ],
            },
            {"role": "assistant", "content": "Then I cover both."},
        ],
    },
    "code-feedback": {
        "id": 1,
        "messages": [
            {"role": "user", "content": "Write a function that divides."},
            {"role": "assistant", "content": "def div(a, b): return a / b"},
            {"role": "user", "content": "Execution result: ZeroDivisionError"},
            {"role": "assistant", "content": "Guard the zero case."},
        ],
    },
    "adp-go-browse-wa": {"id": "webarena_1", "conversations": ADP_CONVERSATION, "system": "x"},
    "adp-nnetnav-wa": {"id": "nnetnav_1", "conversations": ADP_CONVERSATION, "system": "x"},
    "adp-mind2web": {
        "id": "m2w_1",
        "system": "x",
        "conversations": sharegpt(
            ("human", "Book a table. [Current URL: None] ===== tree ====="),
            (
                "function_call",
                '<function=browser>\n<parameter=code>goto(url="x")</parameter>\n</function>',
            ),
            ("human", "[Current URL: x] ===== tree ====="),
            (
                "function_call",
                '<function=browser>\n<parameter=code>click(bid="1")</parameter>\n</function>',
            ),
            ("human", "[Current URL: y] ===== tree ====="),
        ),
    },
    "agentrefine-gpt4o": {
        "id": "a1",
        "conversations": [
            {"from": "human", "value": "You are in a kitchen. Task: heat the egg.", "loss": False},
            {"from": "gpt", "value": "Thought: find egg.\nAction: go to fridge", "loss": True},
            {"from": "human", "value": "You see an egg.", "loss": False},
            {"from": "gpt", "value": "Thought: take it.\nAction: take egg", "loss": True},
        ],
    },
}

# The four long-dependency views re-read verified trajectory sources in their own shapes.
ROWS["ldep-scale-swe-v4-pro"] = ROWS["scale-swe-distilled-v4-pro"]
ROWS["ldep-rebench-openhands"] = {
    "trajectory": editor_messages(with_ids=True),
    "tools": EDITOR_TOOLS,
    "instance_id": "o__r-1",
    "resolved": True,
}
ROWS["ldep-coderforge"] = {
    "messages": editor_messages(with_ids=True),
    "tools": EDITOR_TOOLS,
    "trajectory_id": "t1",
    "reward": 1.0,
}
ROWS["opencode-agent-skills-question-tool"] = ROWS["opencode-question-tool"]
ROWS["repository-v2-long-replay"] = ROWS["repository-v2-long"]


def context(source):
    return AdapterContext(
        source_id=source.source_id,
        revision=source.revision,
        uri=source.uri,
        stream=source.stream,
        field_map=source.field_map,
        options=source.options,
        codec=ByteCodec.byte_fallback(),
        max_record_tokens=1024,
    )


def test_the_fixtures_cover_exactly_the_manifest():
    assert set(ROWS) == set(SOURCES)


def test_every_source_is_pinned_licensed_and_approved_by_the_maintainer():
    for source in SOURCES.values():
        assert len(source.revision) == 40 and source.format == "hf", source.source_id
        assert source.adapter in PROFILES and source.max_rows, source.source_id
        assert source.terms.license not in ("", "unreviewed"), source.source_id
        assert source.terms.license_url.startswith("https://"), source.source_id
        assert (
            source.terms.reviewed_by == "Gabriel Pickard"
            and source.terms.reviewed_at == "2026-10-22"
        )
        source.terms.require_approval()


def test_ids_are_new_and_the_only_repeated_rows_are_declared_second_views():
    notes = {
        s["source_id"]: s["notes"]
        for s in json.loads((CONFIGS / "sources.later-stages.json").read_text())["sources"]
    }
    taken = {s.source_id for s in OTHERS}
    assert not taken & set(SOURCES)
    seen = {}
    for source in SOURCES.values():
        key = (source.uri, source.subset, source.split, tuple(f.path for f in source.files))
        if key in seen:
            # Two entries over the same pinned rows must differ in stream and say why.
            assert source.stream != seen[key].stream, (source.source_id, seen[key].source_id)
            assert "second view" in notes[source.source_id].lower(), source.source_id
        seen[key] = source


def test_every_stream_the_later_stages_name_has_a_source_and_the_plan_reads_it():
    produced = {s.stream for s in SOURCES.values()}
    for phase, streams in STREAMS.items():
        mixture = stage_recipe(phase).mixture
        for stream in streams:
            assert stream in mixture, (phase, stream)
            assert stream in produced, (phase, stream)
    wanted = set()
    for phase in ("P1", "M1", "M2", "LC", "S"):
        wanted |= set(stage_recipe(phase).mixture)
    assert produced <= wanted


@pytest.mark.parametrize("source_id", sorted(SOURCES))
def test_fixture_row_becomes_an_episode_in_the_declared_stream(source_id):
    source = SOURCES[source_id]
    row = ROWS[source_id]
    # The fixture must pass the source's own filters (so the filters name real fields).
    validate_row_filters(row, source.options.get("row_filters", []))
    episode = adapt_row(row, source.adapter, context(source))
    assert episode.stream == source.stream
    assert any(record.policy_mask for record in episode.records), "nothing to imitate"
    for record in episode.records:
        assert not record.policy_mask or record.role in {"assistant", "document"}
        assert not record.result_of or (record.role == "tool" and not record.policy_mask)
    quality = QualityPolicy(**source.options.get("quality", {}))
    if quality.required_verified_success:
        assert episode.supervision.verified_success
    if quality.required_executed_observation:
        assert any(r.result_of and r.provenance == "executed" for r in episode.records)


def test_streams_the_engine_trains_only_on_verified_success_keep_a_recorded_verdict():
    for source in SOURCES.values():
        if source.stream in {"tested_artifacts", "verified_successful_multiturn"}:
            assert PROFILES[source.adapter].outcome_verification == "publisher_tests"
            assert QualityPolicy(**source.options["quality"]).required_verified_success
            assert not source.options.get("assume_outcome"), source.source_id
            # A source that mixes outcomes keeps only the rows whose verdict is a pass.
            mapped = source.field_map.get("outcome")
            filtered = {rule["path"] for rule in source.options.get("row_filters", [])}
            assert mapped in filtered, source.source_id
            episode = adapt_row(ROWS[source.source_id], source.adapter, context(source))
            assert episode.supervision.verified_success, source.source_id


def test_long_dependency_views_link_edits_to_the_results_that_showed_the_file():
    views = [s for s in SOURCES.values() if s.stream == "long_verified_dependencies"]
    assert len(views) >= 3
    for source in views:
        assert source.options["path_dependencies"] is True
        assert source.options["quality"]["min_tokens"] >= 32768
        assert source.options["quality"]["required_verified_success"] is True
        episode = adapt_row(ROWS[source.source_id], source.adapter, context(source))
        assert episode.supervision.verified_success
        assert len(episode.dependencies) == 1, source.source_id
        (dependency,) = episode.dependencies
        records = {r.record_id: r for r in episode.records}
        assert records[dependency.writer_id].role == "tool"
        assert records[dependency.reader_id].policy_mask
        assert dependency.kind == "file_read_to_edit"


def test_unknown_outcome_runs_are_admitted_but_never_counted_as_verified():
    source = SOURCES["scale-swe-distilled-v32"]
    episode = adapt_row(ROWS[source.source_id], source.adapter, context(source))
    assert episode.supervision.outcome == "unknown" and not episode.supervision.verified_success
    assert any(record.policy_mask for record in episode.records)


def test_repository_sources_place_files_before_the_scored_change():
    source = SOURCES["github-patches"]
    episode = adapt_row(ROWS[source.source_id], source.adapter, context(source))
    roles = [record.role for record in episode.records]
    assert roles[0] == "user" and roles.index("document") < roles.index("assistant")
    assert episode.records[1].payload == CODE.encode()
    thread = SOURCES["pr-traces-10k"]
    episode = adapt_row(ROWS[thread.source_id], thread.adapter, context(thread))
    texts = [r.payload for r in episode.records if r.role == "document"]
    # Linked issue, PR body, a conversation comment, the one review with text, an inline comment.
    assert texts[:2] == [b"Max is wrong", b"It is 100."] and len(texts) == 6
    assert [r.payload for r in episode.records if r.policy_mask] == [PATCH.encode()]


def test_the_fix_after_a_review_comment_is_the_scored_target():
    source = SOURCES["pr-resolve-comment-to-fix"]
    episode = adapt_row(ROWS[source.source_id], source.adapter, context(source))
    scored = [r for r in episode.records if r.policy_mask]
    assert [r.payload for r in scored] == [PATCH.encode()]
    assert b"Please bound the cache." in b"".join(r.payload for r in episode.records[:-1])


def test_a_rewrite_reads_the_original_then_scores_the_rewrite():
    for name, provenance in (
        ("swallow-code-v2-rewrite", "teacher"),
        ("pie-perf-slow-to-fast", "natural"),
    ):
        source = SOURCES[name]
        episode = adapt_row(ROWS[name], source.adapter, context(source))
        assert [r.role for r in episode.records][0] == "document"
        assert episode.records[-1].policy_mask and episode.records[-1].provenance == provenance


def test_row_filters_drop_what_the_stream_must_not_contain():
    reject = {
        "swallow-code-v2-rewrite": {"is_vendor": True},
        "pie-perf-slow-to-fast": {"status_v1": "Wrong Answer"},
        "kodcode-sft-4o": {"4o_correctness": "False"},
        "openr1-verifiable-python-tested": {"test_reward": 0.0},
        "s-tested-opencodeinstruct": {"average_test_score": "0.8"},
        "if-multi-constraints-sft": {"strict_score": 0.8},
        "openmathinstruct1-correct": {"is_correct": False},
        "pr-resolve-issue-to-patch": {"task_type": "comment_to_fix"},
        "code-world-traces-affected": {"affected": False},
        "scale-swe-distilled-v4-pro": {"extra_info": {"success": False}},
    }
    for name, change in reject.items():
        row = {**ROWS[name], **change}
        with pytest.raises(Exception, match="source_filter"):
            validate_row_filters(row, SOURCES[name].options["row_filters"])


def test_the_manifest_splits_into_balanced_shards_that_partition_it():
    manifest = json.loads((CONFIGS / "sources.later-stages.json").read_text())
    shards = plan_shards(manifest, 8, None, scale=2)
    ids = [s["source_id"] for shard in shards for s in shard["sources"]]
    assert sorted(ids) == sorted(SOURCES)
    cost = [sum(row_cost(s) for s in shard["sources"]) for shard in shards]
    assert min(cost) > 0 and max(cost) <= 2 * sum(cost) / len(cost)


def test_the_cloud_job_reads_the_manifest_it_names_and_needs_no_gpu():
    job = cloud.validate(json.loads((ROOT / "configs/cloud/prepare-later-stages.json").read_text()))
    command = " ".join(job["command"])
    assert "configs/data/sources.later-stages.json" in command
    assert "configs/data/parser-review.agentic.json" in command
    assert job["resources"]["gpu"] == 0
