"""Paired native Episode import contract for typed semantic-execution/3."""
from __future__ import annotations

import copy
import json

import pytest

from megatoken.data import WindowConfig, iter_windows
from megatoken.data.schema import Episode
from megatoken.data.semantic_execution import (
    SemanticExecutionError,
    adapt_semantic_execution,
)
from megatoken.generation.codec import ByteCodec


def line(kind, payload):
    return kind + " " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def fixture():
    code = {"entry": "run", "source": "function run(x){return x*2+1;}", "sites": []}
    graph = {"graph": {"roots": [["num", 7]], "objects": []}}
    records = [
        ("observed", "code", code),
        ("observed", "input", graph),
        ("generated", "compute", {
            "expr": {"op": "binary", "operator": "*",
                     "left": {"op": "literal", "value": ["num", 7]},
                     "right": {"op": "literal", "value": ["num", 2]}},
            "result": ["num", 14]}),
        ("generated", "compute", {
            "expr": {"op": "binary", "operator": "+",
                     "left": {"op": "literal", "value": ["num", 14]},
                     "right": {"op": "literal", "value": ["num", 1]}},
            "result": ["num", 15]}),
        ("generated", "return", {"value": ["num", 15]}),
        ("generated", "end", {"kind": "return"}),
    ]
    return {
        "public": {
            "schema": "semantic-execution/3",
            "traceFormat": "semantic-trace/3",
            "contextFormat": "semantic-context/3",
            "profile": "training-js/1",
            "completionPolicy": "immediate-root",
            "groups": [
                {"id": f"m{i + 1}", "mode": mode,
                 "loss": "none" if mode == "observed" else "semantic",
                 "events": [{"track": "t0", "record": line(kind, payload)}]}
                for i, (mode, kind, payload) in enumerate(records)
            ],
        },
        "sidecar": {
            "schema": "semantic-supervision/2",
            "admission": "locally_verified",
            "referenceStatus": "agreed",
            "archive": {"private": "never delivered"},
        }
    }


def test_native_version_three_import_keeps_exact_bytes_and_loss():
    document = fixture()
    episode, receipt = adapt_semantic_execution(
        document, allow_local=True, codec=ByteCodec.byte_fallback()
    )
    assert isinstance(episode, Episode)
    assert receipt.generated == 4
    assert receipt.observed == 2
    assert [record.payload for record in episode.records] == [
        row["events"][0]["record"].encode("utf-8")
        for row in document["public"]["groups"]
    ]
    assert [record.policy_mask for record in episode.records] == [
        False, False, True, True, True, True,
    ]
    assert Episode.from_dict(episode.to_dict()) == episode
    assert list(iter_windows(
        episode, ByteCodec.byte_fallback(),
        WindowConfig(max_records=16, max_decisions=16)
    ))
    assert episode.supervision.outcome == "unknown"
    assert all(b"private" not in record.payload for record in episode.records)


@pytest.mark.parametrize("mutator", [
    lambda doc: doc["public"].update(traceFormat="semantic-trace/2"),
    lambda doc: doc["public"].update(contextFormat="semantic-context/2"),
    lambda doc: doc["sidecar"].update(schema="semantic-supervision/1"),
    lambda doc: doc["public"]["groups"][3]["events"][0].update(
        record=line("compute", {"result": ["num", 2]})),
    lambda doc: doc["public"]["groups"][3]["events"][0].update(
        record=line("read", {"object": "o1", "key": "n", "present": True, "value": ["num", "1"]})),
    lambda doc: doc["public"]["groups"][1]["events"][0].update(
        record=line("input", {"graph": {"roots": [["ref", "o100"]], "objects": []}})),
    lambda doc: doc["public"]["groups"][2].update(mode="observed"),
    lambda doc: doc["public"]["groups"][-1]["events"][0].update(
        record=line("end", {"kind": "throw"})),
])
def test_v3_rejects_mixed_versions_and_bad_witnesses(mutator):
    document = copy.deepcopy(fixture())
    mutator(document)
    with pytest.raises(SemanticExecutionError):
        adapt_semantic_execution(document, allow_local=True)


def test_v3_sparse_input_and_special_tags_survive_native_import():
    doc = fixture()
    input_record = {
        "graph": {
            "roots": [["ref", "o1"], ["num", "-0"], ["num", "NaN"], ["bigint", "42"]],
            "objects": [
                {"id": "o1", "kind": "array", "properties": [
                    {"key": "1", "value": ["undefined"]},
                    {"key": "length", "value": ["num", 3]}
                ]}
            ]
        }
    }
    doc["public"]["groups"][1]["events"][0]["record"] = line("input", input_record)
    episode, receipt = adapt_semantic_execution(doc, allow_local=True)
    assert receipt.observed == 2
    assert b'"NaN"' in episode.records[1].payload
    assert b'"-0"' in episode.records[1].payload


def test_v3_rejects_noncanonical_duplicate_or_unknown_expr_keys():
    document = fixture()
    record = 'compute {"expr":{"op":"literal","value":["num",1],"value":["num",2]},"result":["num",2]}'
    document["public"]["groups"][2]["events"][0]["record"] = record
    with pytest.raises(SemanticExecutionError):
        adapt_semantic_execution(document, allow_local=True)
