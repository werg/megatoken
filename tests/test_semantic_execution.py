"""Execution export admission, causal visibility and native-meridian conversion."""

import copy
import json

import pytest

from megatoken.data import Episode, WindowConfig, iter_windows
from megatoken.data.semantic_execution import (
    SemanticExecutionError,
    adapt_semantic_execution,
)
from megatoken.generation.codec import ByteCodec


def line(kind, fields):
    return kind + " " + json.dumps(fields, separators=(",", ":"), ensure_ascii=True)


def fixture():
    source = '"use strict"; function f(x){if(x)return x+1;return 0;}'
    groups = [
        ("observed", "code", {"sourceId": "src1", "text": source, "sites": [], "codes": []}),
        ("observed", "value", {"entry": "f", "graph": {"roots": [["num", 3]], "objects": []}}),
        ("generated", "enter", {"invocation": "f1", "entry": "f", "arguments": [["num", 3]]}),
        ("generated", "value", {"id": "e1", "invocation": "f1", "result": ["num", 3]}),
        ("generated", "branch", {"id": "e2", "dependsOn": ["e1"], "test": ["bool", True], "path": "then"}),
        ("generated", "return", {"id": "e3", "dependsOn": ["e2"], "result": ["num", 4]}),
        ("generated", "end", {"invocation": "f1", "kind": "return"}),
    ]
    return {
        "public": {
            "schema": "semantic-execution/2",
            "profile": "training-js/1",
            "traceFormat": "semantic-trace/2",
            "contextFormat": "semantic-context/2",
            "completionPolicy": "immediate-return",
            "groups": [
                {
                    "id": "m" + str(i + 1),
                    "mode": mode,
                    "loss": "none" if mode == "observed" else "semantic",
                    "events": [{"track": "t0", "record": line(kind, values)}],
                }
                for i, (mode, kind, values) in enumerate(groups)
            ],
        },
        "sidecar": {
            "schema": "semantic-supervision/1",
            "sourceLineage": "family-A",
            "referenceStatus": "agreed",
            "split": "train",
            "admission": "locally_verified",
            "archive": {"private": "never model-visible"},
        },
    }


def test_unadmitted_data_rejected_by_default():
    with pytest.raises(SemanticExecutionError, match="Unadmitted"):
        adapt_semantic_execution(fixture())


def test_adapter_builds_causal_episode_without_oracle_sidecar_leakage():
    source = fixture()
    ep, receipt = adapt_semantic_execution(source, allow_local=True)
    assert isinstance(ep, Episode)
    assert ep.stream == "semantic_execution"
    assert receipt.generated == 5
    assert receipt.observed == 2
    assert receipt.dependencies == 2
    assert ep.supervision.outcome == "unknown"
    assert ep.lineage.family_id == "family-A"
    assert "repository:werg/tracing-js-self" not in ep.lineage.family_keys
    assert ep.lineage.frozen_split == "train"
    assert len(ep.meridians) == 7
    assert {track.track_id for track in ep.tracks} == {"t" + str(i) for i in range(8)}
    assert [row.policy_mask for row in ep.records] == [False, False, True, True, True, True, True]
    assert all(record.role == "assistant" for record in ep.records[2:])
    assert all(b"private" not in record.payload for record in ep.records)
    assert all(b"referenceStatus" not in record.payload for record in ep.records)
    assert {(d.writer_id, d.reader_id) for d in ep.dependencies} == {
        ("semantic:r4", "semantic:r5"),
        ("semantic:r5", "semantic:r6"),
    }
    codec = ByteCodec.byte_fallback()
    windows = list(iter_windows(ep, codec, WindowConfig(max_records=16, max_decisions=16)))
    assert windows


@pytest.mark.parametrize(
    "bad",
    [
        lambda e: e["public"]["groups"][4].update(mode="observed"),
        lambda e: e["public"]["groups"][4]["events"][0].update(track="t8"),
        lambda e: e["public"]["groups"][5]["events"][0].update(
            record=line("return", {"id": "e3", "dependsOn": ["e77"], "result": ["num", 4]})
        ),
        lambda e: e["public"]["groups"][3]["events"][0].update(
            record=line("value", {"id": "e2", "result": ["num", 3]})
        ),
        lambda e: e["public"]["groups"][2]["events"][0].update(
            record=line("observe", {"id": "e100", "effect": "fx1"})
        ),
        lambda e: e["public"]["groups"].append(
            {
                "id": "m8",
                "mode": "generated",
                "loss": "semantic",
                "events": [{"track": "t0", "record": line("value", {"id": "e4"})}],
            }
        ),
        lambda e: e["public"].update(traceFormat="semantic-trace/999"),
    ],
)
def test_invalid_semantic_meridians_fail_closed(bad):
    candidate = copy.deepcopy(fixture())
    bad(candidate)
    with pytest.raises(SemanticExecutionError):
        adapt_semantic_execution(candidate, allow_local=True)


def test_observed_results_require_prior_committed_effect_with_matching_identity():
    document = fixture()
    group = {
        "id": "m" + str(len(document["public"]["groups"])),
        "mode": "observed",
        "loss": "none",
        "events": [{"track": "t0", "record": line("observe", {"effect": "fx", "result": 3})}],
    }
    document["public"]["groups"].insert(-1, group)
    for i, row in enumerate(document["public"]["groups"]):
        row["id"] = "m" + str(i + 1)
    with pytest.raises(SemanticExecutionError, match="matching request"):
        adapt_semantic_execution(document, allow_local=True)


def test_fragments_commit_only_at_close_before_dependent_read():
    candidate = fixture()
    body = candidate["public"]["groups"]
    value = body.pop(3)
    literal = value["events"][0]["record"]
    parts = [
        line("value-open", {"id": "e1", "type": "semantic-record"}),
        line("value-part", {"id": "e1", "data": literal[:12]}),
        line("value-part", {"id": "e1", "data": literal[12:]}),
        line("value-close", {"id": "e1"}),
    ]
    for offset, text in enumerate(parts):
        body.insert(
            3 + offset,
            {
                "id": "",
                "mode": "generated",
                "loss": "semantic",
                "events": [{"track": "t0", "record": text}],
            },
        )
    for i, row in enumerate(body):
        row["id"] = "m" + str(i + 1)
    episode, receipt = adapt_semantic_execution(candidate, allow_local=True)
    assert len(episode.records) == len(body)
    assert receipt.generated == 5  # value fragments count as one committed semantic event
    assert ("semantic:r7", "semantic:r8") in {
        (d.writer_id, d.reader_id) for d in episode.dependencies
    }


def test_actual_codec_hard_capacity_is_checked():
    example = fixture()
    with pytest.raises(SemanticExecutionError, match="output-position"):
        adapt_semantic_execution(
            example, allow_local=True, codec=ByteCodec.byte_fallback(), max_positions=5
        )


def test_local_diagnostics_never_impersonate_independently_admitted_data():
    example = fixture()
    example["sidecar"]["admission"] = "admitted"
    example["sidecar"]["referenceStatus"] = "not-checked"
    with pytest.raises(SemanticExecutionError, match="agreement"):
        adapt_semantic_execution(example)
    example["sidecar"]["referenceStatus"] = "agreed"
    with pytest.raises(SemanticExecutionError, match="external admission verifier"):
        adapt_semantic_execution(example)


def test_jsonl_converter_never_labels_diagnostic_episodes_admitted(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    from megatoken.data.schema import Episode

    source = tmp_path / "tracer.jsonl"
    output = tmp_path / "native.jsonl"
    source.write_text(json.dumps(fixture()) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/import_semantic_execution.py"
    command = [
        sys.executable,
        str(script),
        "--input",
        str(source),
        "--output",
        str(output),
        "--diagnostic-byte",
        "--allow-local",
    ]
    complete = subprocess.run(command, capture_output=True, text=True, check=True)
    assert '"admitted": 0' in complete.stdout
    episode = Episode.from_dict(json.loads(output.read_text(encoding="utf-8")))
    assert episode.stream == "semantic_execution"
    assert episode.supervision.outcome == "unknown"
    assert all(b"private" not in record.payload for record in episode.records)


def test_native_preparation_preserves_semantic_groups_and_exact_target_bytes():
    from megatoken.data.adapters import AdapterContext, adapt_row

    episode, _ = adapt_semantic_execution(fixture(), allow_local=True)
    context = AdapterContext(
        source_id="tracer",
        revision="pinned-current",
        uri="local://semantic-execution",
        stream="semantic_execution",
        options={"preserve_native_stream": True},
        codec=ByteCodec.byte_fallback(),
    )
    prepared = adapt_row(episode.to_dict(), "native", context)
    assert prepared.stream == "semantic_execution"
    assert prepared.meridians == episode.meridians
    assert prepared.dependencies == episode.dependencies
    assert [x.payload for x in prepared.records] == [x.payload for x in episode.records]
    assert prepared.supervision.outcome == "unknown"


def _reindex(groups):
    for index, group in enumerate(groups):
        group["id"] = "m" + str(index + 1)


def _pieces(kind, fields, *, label, mode):
    target = line(kind, fields)
    records = [
        line("value-open", {"id": label, "type": "semantic-record"}),
        line("value-part", {"id": label, "data": target[: len(target) // 2]}),
        line("value-part", {"id": label, "data": target[len(target) // 2 :]}),
        line("value-close", {"id": label}),
    ]
    return [
        {
            "id": "",
            "mode": mode,
            "loss": "semantic" if mode == "generated" else "none",
            "events": [{"track": "t0", "record": text}],
        }
        for text in records
    ]


def test_fragmented_source_and_input_are_still_observed_user_context():
    candidate = fixture()
    original = candidate["public"]["groups"]
    source = original[0]["events"][0]["record"]
    kind, body = source.split(" ", 1)
    assert kind == "code"
    original[:1] = _pieces(kind, json.loads(body), label="record:code", mode="observed")
    _reindex(original)

    episode, receipt = adapt_semantic_execution(candidate, allow_local=True)
    assert receipt.observed == 2
    assert [record.role for record in episode.records[:5]] == ["user"] * 5
    assert episode.records[5].role == "assistant"
    assert [record.part_index for record in episode.records[:4]] == [0, 1, 2, 3]
    assert all(record.part_count == 4 for record in episode.records[:4])
    assert len({record.logical_id for record in episode.records[:4]}) == 1
    assert Episode.from_dict(episode.to_dict()) == episode


def test_fragmented_json_request_stays_one_logical_action_and_delivery():
    candidate = fixture()
    groups = candidate["public"]["groups"]
    request = {
        "id": "e1",
        "effect": "fx1",
        "operation": "demo.store",
        "signature": "demo.store/1",
        "codec": "canonical-json/1",
        "payload": '["answer",{"n":2}]',
        "arguments": ["answer", {"n": 2}],
    }
    groups[3:] = [
        *_pieces("request", request, label="e1", mode="generated"),
        {
            "id": "",
            "mode": "observed",
            "loss": "none",
            "events": [
                {
                    "track": "t0",
                    "record": line("observe", {"id": "e2", "effect": "fx1", "result": ["num", 2]}),
                }
            ],
        },
        {
            "id": "",
            "mode": "generated",
            "loss": "semantic",
            "events": [
                {
                    "track": "t0",
                    "record": line(
                        "return", {"id": "e3", "dependsOn": ["e2"], "result": ["num", 2]}
                    ),
                }
            ],
        },
        {
            "id": "",
            "mode": "generated",
            "loss": "semantic",
            "events": [
                {"track": "t0", "record": line("end", {"invocation": "f1", "kind": "return"})}
            ],
        },
    ]
    _reindex(groups)
    episode, receipt = adapt_semantic_execution(candidate, allow_local=True)
    assert receipt.actions == 1
    assert receipt.generated == 4
    assert receipt.observed == 3
    (action,) = episode.actions
    assert action.action_id == "semantic:fx1"
    assert action.schema_id == "demo.store/1"
    assert action.record_ids == tuple("semantic:r" + str(i) for i in range(4, 8))
    assert all(record.action_id == action.action_id for record in episode.records[3:7])
    assert episode.records[7].role == "environment"
    assert episode.records[7].result_of == action.action_id
    assert episode.records[7].execution_id == action.execution_id
    assert [record.part_index for record in episode.records[3:7]] == [0, 1, 2, 3]
    assert ("semantic:r8", "semantic:r9") in {
        (dep.writer_id, dep.reader_id) for dep in episode.dependencies
    }
    assert Episode.from_dict(episode.to_dict()) == episode

    tampered = copy.deepcopy(candidate)
    invalid_request = {**request, "payload": '["forged",{"n":2}]'}
    tampered["public"]["groups"][3:7] = _pieces(
        "request", invalid_request, label="e1", mode="generated"
    )
    _reindex(tampered["public"]["groups"])
    with pytest.raises(SemanticExecutionError, match="bytes disagree"):
        adapt_semantic_execution(tampered, allow_local=True)


def test_wrong_external_request_codec_or_shape_is_rejected():
    document = fixture()
    requests = [
        {
            "id": "e1",
            "effect": "fx1",
            "operation": "demo.readAsync",
            "signature": "demo.readAsync/1",
            "codec": "canonical-json/1",
            "payload": '["key"]',
            "arguments": ["key"],
        },
        {
            "id": "e1",
            "effect": "fx1",
            "operation": "demo.readAsync",
            "signature": "demo.readAsync/1",
            "codec": "graph/999",
            "payload": '["key"]',
            "arguments": ["key"],
        },
    ]
    for request in requests[1:]:
        candidate = copy.deepcopy(document)
        candidate["public"]["groups"][3]["events"][0]["record"] = line("request", request)
        with pytest.raises(SemanticExecutionError, match="codec"):
            adapt_semantic_execution(candidate, allow_local=True)
