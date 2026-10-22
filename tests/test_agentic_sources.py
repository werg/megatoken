"""The agentic and chat manifests: pinned, licence-recorded, and every source readable by its adapter."""

import importlib.util
import json
from pathlib import Path

import pytest

from megatoken import cloud
from megatoken.data.adapters import PROFILES, AdapterContext, adapt_row
from megatoken.data.quality import validate_row_filters
from megatoken.data.recipes import stage_recipe
from megatoken.data.review import ParserReviewPolicy
from megatoken.data.sources import load_source_manifest
from megatoken.generation.codec import ByteCodec

CONFIGS = Path(__file__).parents[1] / "configs/data"
SOURCES = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic.json")}
GATED = {s.source_id: s for s in load_source_manifest(CONFIGS / "sources.agentic-gated.json")}
EVERY = {**SOURCES, **GATED}

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}


def put(row, path, value):
    *parents, leaf = path.split(".")
    for key in parents:
        row = row.setdefault(key, {})
    row[leaf] = value


def first(value):
    return value[0] if isinstance(value, (list, tuple)) else value


def openai_messages(positional):
    call = {k: v for k, v in CALL.items() if not (positional and k == "id")}
    result = (
        {"role": "tool", "content": "out"}
        if positional
        else {"role": "tool", "tool_call_id": "c1", "content": "out"}
    )
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "content": "Looking.", "tool_calls": [call]},
        result,
        {"role": "assistant", "content": "Done."},
    ]


def text_scaffold(action):
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "content": action},
        {"role": "user", "content": "OBSERVATION: ok"},
        {"role": "assistant", "content": "Finished."},
    ]


def sharegpt(pairs):
    return [{"from": role, "value": text} for role, text in pairs]


def conversation(source, row, messages):
    defaults = dict(PROFILES[source.adapter].fields)
    key = source.field_map.get("messages") or defaults.get("messages", ("messages",))[0]
    return conversation_of(row, key, messages)


def conversation_of(row, key, messages):
    put(row, key, messages)
    return row


# One row in each adapter's input shape; field_map names decide where each field goes.
def row_for(source):
    adapter, fm = source.adapter, source.field_map
    row = {}
    for key in ("family", "id", "task"):
        if key in fm:
            put(row, first(fm[key]), "k1")
    if "mask" in fm:
        put(row, fm["mask"], [False, False, True])
    if adapter in {"open_swe", "orchard_swe"}:
        row["tools"] = TOOLS
        row.setdefault("repo", "o/r")
        return conversation(
            source, row, openai_messages(source.options.get("positional_call_ids", False))
        )
    if adapter in {"nemotron_agentic_v4"}:
        row["tools"] = TOOLS
        return conversation(
            source, row, openai_messages(source.options.get("positional_call_ids", False))
        )
    if adapter in {"agent_trove", "nemotron_terminal"}:
        row.setdefault("task", "t1")
        command = (
            '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n", "duration": 0.1}]}'
        )
        return conversation(source, row, text_scaffold(command))
    if adapter == "openhands_xml":
        return conversation(
            source,
            row,
            text_scaffold(
                "Look.\n\n<function=execute_bash>\n<parameter=command>ls</parameter>\n</function>"
            ),
        )
    if adapter == "swe_agent_fenced":
        messages = [
            {"role": "system", "system_prompt": "SETTING", "text": None},
            {"role": "user", "text": "Issue"},
            {"role": "ai", "text": "look\n```\nls\n```"},
            {"role": "user", "text": "a.py"},
            {"role": "ai", "text": "done"},
        ]
        return conversation(source, row, messages)
    if adapter == "mini_swe_agent":
        return conversation(source, row, text_scaffold("THOUGHT: look\n\n```bash\nls\n```"))
    if adapter == "codeact_agent":
        return conversation(source, row, text_scaffold("Run it.\n<execute>\nprint(1)\n</execute>"))
    if adapter == "react_agent":
        messages = sharegpt(
            [
                ("human", "task"),
                ("gpt", "Think: look.\nAct: bash"),
                ("human", "output: a"),
                ("gpt", "Think: done."),
            ]
        )
        put(row, fm.get("messages", "conversations"), messages)
        return row
    if adapter == "hermes_tools":
        row["tools"] = json.dumps(TOOLS)
        messages = sharegpt(
            [
                ("system", "You may call functions."),
                ("human", "go"),
                ("gpt", '<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call>'),
                ("tool", '<tool_response>\n{"v": 1}\n</tool_response>'),
                ("gpt", "done"),
            ]
        )
        put(row, "conversations", messages)
        return row
    if adapter == "python_call_tools":
        if "toolace" in source.source_id:
            return conversation_of(
                row,
                "conversations",
                sharegpt(
                    [
                        ("human", "go"),
                        ("gpt", "[f(a=1)]"),
                        ("tool", '[{"name": "f", "results": 1}]'),
                        ("gpt", "done"),
                    ]
                ),
            )
        # Ai2 Dolci: the call has its own field and the result comes from an "environment" turn.
        row["functions"] = json.dumps([TOOLS[0]["function"]])
        return conversation_of(
            row,
            "messages",
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "function_calls": "f(a=1)"},
                {"role": "environment", "content": '{"r": 1}'},
                {"role": "assistant", "content": "done"},
            ],
        )
    if adapter == "glaive_tools":
        row["system"] = (
            'SYSTEM: functions -\n{"name": "f", "description": "d", "parameters": {"type": "object"}}\n'
        )
        row["chat"] = (
            'USER: go\n\n\nASSISTANT: <functioncall> {"name": "f", "arguments": \'{"a": 1}\'} <|endoftext|>\n\n\n'
            'FUNCTION RESPONSE: {"v": 1}\n\n\nASSISTANT: done <|endoftext|>\n\n\n'
        )
        return row
    if adapter == "xlam_calls":
        row.update(
            query="q",
            answers=json.dumps([{"name": "f", "arguments": {"a": 1}}]),
            tools=json.dumps(
                [{"name": "f", "description": "d", "parameters": {"a": {"type": "int"}}}]
            ),
        )
        return row
    if adapter in {"simulated_tool_roles", "mcp_tools"}:
        row["tools"] = json.dumps(TOOLS)
        messages = [
            {"from": "human", "value": "go"},
            {"from": "function_call", "value": '{"name": "f", "arguments": {"a": 1}}'},
            {"from": "observation", "value": "out"},
            {"from": "gpt", "value": "done"},
        ]
        if adapter == "mcp_tools":
            messages = [
                {"role": "user", "content": "go"},
                {"role": "tool_call", "content": {"name": "f", "arguments": '{"a": 1}'}},
                {"role": "tool_response", "content": "out"},
                {"role": "assistant", "content": "done"},
            ]
            row["messages"] = json.dumps(messages)
        else:
            row["conversations"] = messages
        return row
    if adapter == "turn_pairs":
        row["conversation"] = [{"input": "hi", "output": "hello"}]
        return row
    if adapter == "nemotron_chat":
        row["messages"] = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        row["metadata"] = {"train_turns": [False, False, True]}
        return row
    if adapter == "artifact":
        for key in ("prompt", "target"):
            put(row, fm.get(key, key), "text")
        return row
    if adapter == "chat":
        ours = (
            "conversation"
            if fm.get("messages") == "conversation"
            else fm.get("messages", "messages")
        )
        messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        if "conversations" in source.field_map.get(
            "messages", "conversations"
        ) or source.source_id in {
            "openthoughts3",
            "bespoke-stratos-17k",
            "slimorca-dedup",
            "am-thinking-other",
            "am-thinking-multiturn",
            "am-thinking-if",
        }:
            put(row, "conversations", sharegpt([("human", "q"), ("gpt", "a")]))
            return row
        if source.source_id.startswith("orca-agentinstruct"):
            row["messages"] = json.dumps(messages)
            return row
        if source.source_id == "when2call-sft":
            row["tools"] = [json.dumps(TOOLS[0]["function"])]
        put(row, ours, messages)
        return row
    raise AssertionError(f"No fixture for adapter {adapter}")


def row_passing_filters(source):
    row = row_for(source)
    for rule in source.options.get("row_filters", []):
        put(row, rule["path"], rule.get("value"))
    # A recorded test verdict is what lets the engine train on a trajectory; unknown outcomes are not imitated.
    defaults = dict(PROFILES[source.adapter].fields)
    outcome = first(source.field_map.get("outcome") or defaults.get("outcome", "resolved"))
    filtered = {rule["path"] for rule in source.options.get("row_filters", [])}
    if outcome not in filtered and not source.options.get("assume_outcome"):
        put(row, outcome, outcome in {"resolved", "target"} or 1.0)
    return row


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


def test_every_source_is_pinned_licensed_and_approved_by_the_maintainer():
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
    assert not set(SOURCES) & set(GATED)


def test_every_source_streams_into_a_stream_the_stage_plan_or_its_own_stage_reads():
    wanted = set()
    for phase in ("P1", "M1", "M2", "LC", "S"):
        wanted |= set(stage_recipe(phase).mixture)
    # Archival sources keep their own stream; the plan reads the rest.
    assert {s.stream for s in EVERY.values()} <= wanted


@pytest.mark.parametrize("source_id", sorted(EVERY))
def test_fixture_row_becomes_an_episode_in_the_declared_stream(source_id):
    source = EVERY[source_id]
    row = row_passing_filters(source)
    validate_row_filters(row, source.options.get("row_filters", []))
    episode = adapt_row(row, source.adapter, context(source))
    assert episode.stream == source.stream
    assert any(record.policy_mask for record in episode.records), "nothing to imitate"
    for record in episode.records:
        # Only assistant emission can be trained; calls are assistant records, results are not.
        assert not record.policy_mask or record.role in {"assistant", "document"}
        assert not record.result_of or (record.role == "tool" and not record.policy_mask)
    results = [r for r in episode.records if r.result_of]
    if source.stream in {"verified_trajectories", "verified_successful_multiturn"}:
        # The training engine refuses these streams unless every episode is a verified success.
        assert episode.supervision.verified_success
    if source.options.get("assume_outcome"):
        assert episode.supervision.verification == "publisher_curated"
        assert not episode.supervision.verified_success
    if results:
        provenance = {r.provenance for r in results}
        assert provenance <= {"executed", "simulated"}


def test_executed_and_simulated_backends_are_labelled_by_profile_not_guessed():
    for source in EVERY.values():
        provenance = source.options.get(
            "observation_provenance", PROFILES[source.adapter].observation_provenance
        )
        if source.stream in {
            "verified_trajectories",
            "verified_successful_multiturn",
            "executed_trajectories",
        }:
            assert provenance == "executed", source.source_id
        if source.adapter in {
            "hermes_tools",
            "glaive_tools",
            "python_call_tools",
            "xlam_calls",
            "simulated_tool_roles",
        }:
            assert provenance == "simulated", source.source_id
        if source.adapter == "mcp_tools":
            assert provenance == "executed"


def test_verified_streams_only_take_sources_that_filter_to_a_recorded_test_verdict():
    for source in EVERY.values():
        if source.stream not in {"verified_trajectories", "verified_successful_multiturn"}:
            continue
        assert PROFILES[source.adapter].outcome_verification == "publisher_tests", source.source_id
        assert not source.options.get("assume_outcome"), source.source_id
        # A mixed-outcome source keeps only the rows whose verdict is a pass.
        mapped = source.field_map.get("outcome")
        filtered = {rule["path"] for rule in source.options.get("row_filters", [])}
        assert mapped in {None, "reward", "resolved", "target"} or mapped in filtered, (
            source.source_id
        )


def test_gated_sources_are_kept_apart_from_the_ones_the_account_can_read():
    # The first four need the maintainer to accept the publisher's terms on the Hub; the last two have
    # their first prompts withheld until restored (configs/cloud/restore-chat-v3-prompts.json for v3).
    assert set(GATED) == {
        "lmsys-chat-1m",
        "infinity-instruct-7m",
        "infinity-instruct-gen",
        "nemotron-pt-v2-chat",
        "chat-v3-chat",
        "nemotron-pt-v1-chat",
    }
    for source in GATED.values():
        assert "gated" in source.terms.notes.lower() or "withheld" in source.terms.notes.lower()


def load_scale_script():
    spec = importlib.util.spec_from_file_location(
        "scale_source_caps", Path(__file__).parents[1] / "scripts/scale_source_caps.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaled_caps_keep_the_leading_rows_and_shards_partition_the_manifest():
    script = load_scale_script()
    manifest = json.loads((CONFIGS / "sources.agentic.json").read_text())
    doubled = script.scale(manifest, 2)
    assert [s["max_rows"] for s in doubled["sources"]] == [
        2 * s["max_rows"] for s in manifest["sources"]
    ]
    shards = [script.shard(doubled, i, 8) for i in range(8)]
    ids = [s["source_id"] for shard in shards for s in shard["sources"]]
    assert sorted(ids) == sorted(s["source_id"] for s in manifest["sources"])
    # Balanced by estimated text: no shard carries more than twice the average.
    cost = [sum(script.cost(s) for s in shard["sources"]) for shard in shards]
    assert max(cost) <= 2 * sum(cost) / len(cost)
    with pytest.raises(ValueError):
        script.shard(doubled, 8, 8)


def test_cloud_jobs_read_the_manifests_they_name():
    for name, manifest in (
        ("prepare-agentic-mix", "sources.agentic.json"),
        ("restore-chat-v3-prompts", "sources.agentic-gated.json"),
    ):
        job = cloud.validate(
            json.loads((Path(__file__).parents[1] / f"configs/cloud/{name}.json").read_text())
        )
        assert f"configs/data/{manifest}" in " ".join(job["command"])
        assert job["resources"]["gpu"] == 0


def test_the_agentic_job_holds_back_parser_faults_but_not_prose_around_code():
    config = json.loads((CONFIGS / "parser-review.agentic.json").read_text())
    policy = ParserReviewPolicy(**{**config, "hard_codes": tuple(config["hard_codes"])})
    default = ParserReviewPolicy()
    # A conversation mixes prose and code in every turn; one badly parsed chunk must not drop
    # a whole trajectory. Every other fault is still held out.
    assert set(default.hard_codes) - set(policy.hard_codes) == {"low_syntax_coverage"}
    assert policy.classify(["low_syntax_coverage", "recovered_syntax"]) == ()
    assert policy.classify(["native_parser_crash"]) == ("native_parser_crash",)
    job = json.loads(
        (Path(__file__).parents[1] / "configs/cloud/prepare-agentic-mix.json").read_text()
    )
    assert "configs/data/parser-review.agentic.json" in " ".join(job["command"])
