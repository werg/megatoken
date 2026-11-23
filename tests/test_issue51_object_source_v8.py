"""Independent native checking of the complete issue 51 original source cohort."""

import base64
import copy
import gzip
import json
from pathlib import Path

import pytest

from megatoken.data.native_v8 import (
    NativeV8Error,
    _native_checkpoint,
    _restore_native_checkpoint,
    validate_v8_artifact,
)
from megatoken.data.protocol_v8 import (
    ProtocolV8Error,
    _internal_machine,
    apply_protocol_record,
    decode_protocol_record,
    encode_protocol_record,
)
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.data.source_families_v8 import verify_region_family_registry

FIXTURE = Path(__file__).parent / "fixtures/semantic-issue51-objects-v8.json.gz.b64"
DOCUMENT = json.loads(gzip.decompress(base64.b64decode(FIXTURE.read_bytes())))
CASES = DOCUMENT["cases"]
ACCEPTED = [row for row in CASES if row["artifact"] is not None]
BY_ID = {row["id"]: row for row in ACCEPTED}


def test_original_population_includes_all_refusals_and_one_lineage():
    assert DOCUMENT["schema"] == "issue51-object-evidence/1"
    assert DOCUMENT["producerTreeClean"] is True
    assert len(CASES) == 23
    assert len(ACCEPTED) == 20
    assert len({row["id"] for row in CASES}) == 23
    registry = DOCUMENT["registry"]
    verify_region_family_registry(registry)
    assert len(registry["components"]) == 1
    assert set(registry["components"][0]["sourceIds"]) == {row["id"] for row in CASES}
    assert all(row["ancestors"] == ["issue51-original-objects"] for row in registry["sources"])
    sources = {row["id"]: row for row in registry["sources"]}
    assert set(sources) == {row["id"] for row in CASES}
    dispositions = {row["id"]: row for row in DOCUMENT["dispositions"]}
    assert set(dispositions) == set(sources)
    assert {row["id"] for row in CASES if row["artifact"] is None} == {
        "issue51-control-for-in-prototype-replacement",
        "issue51-control-for-in-proxy",
        "issue51-control-in-invalid-rhs-before-key-coercion",
    }
    for row in CASES:
        identity = sources[row["id"]]
        assert row["fixture"]["source"] == identity["source"]
        assert row["fixture"]["entry"] == identity["entry"]
        disposition = dispositions[row["id"]]
        if row["artifact"] is None:
            assert disposition["expected"] == "refused"
            assert disposition["status"] == "refused"
            assert disposition["reason"]
        else:
            assert disposition["expected"] == "supported"
            assert disposition["status"] == "passed"
            binding = row["artifact"]["sidecar"]["datasetBinding"]
            assert binding["sourceId"] == row["id"]
            assert binding["registryDigest"] == registry["digest"]
            assert binding["split"] == registry["components"][0]["split"]


@pytest.mark.parametrize("case", ACCEPTED, ids=lambda row: row["id"])
def test_native_replay_every_original_context_and_continuation(case):
    artifact = case["artifact"]
    checked = validate_v8_artifact(artifact)
    assert checked["admitted"] is False  # no live source/tokenizer check here
    assert all(checked["checks"][key] == "passed" for key in ("replay", "contexts", "continuation"))
    assert checked["counts"]["contexts"] == len(artifact["contexts"])


def _operation(case_id, kind):
    artifact = BY_ID[case_id]["artifact"]
    generated = [row for row in artifact["public"]["timeline"] if row["mode"] == "generated"]
    for row, context in zip(generated, artifact["contexts"], strict=True):
        line = row["fields"].get("line", "")
        if line.startswith("protocol "):
            record = decode_protocol_record(line)
            if record["kind"] == kind:
                return artifact, context, record
    raise AssertionError(f"{case_id} has no {kind} operation")


@pytest.mark.parametrize(
    ("case_id", "kind", "changes"),
    [
        (
            "issue51-has-in-reflect-symbol-accessor",
            "property.has",
            [{"object": "forged"}, {"key": ["str", "__absent__"]}, {"result": ["bool", False]}],
        ),
        (
            "issue51-delete-own-and-inherited",
            "property.delete",
            [{"object": "forged"}, {"success": False}],
        ),
        (
            "issue51-for-in-sparse-array",
            "enumeration.next",
            [{"object": "forged"}, {"result": ["str", "forged"]}, {"done": True}],
        ),
    ],
)
def test_original_operation_restores_and_rejects_forged_records_atomically(case_id, kind, changes):
    artifact, context, record = _operation(case_id, kind)
    prefix = artifact["public"]["timeline"][: context["visibleThrough"]]
    checkpoint = _native_checkpoint(prefix, context)
    restored = _restore_native_checkpoint(
        checkpoint,
        expected_prefix=prefix,
        expected_state=context["state"],
        expected_cursor=context["regionCursor"],
        expected_obligations=context["obligations"],
    )
    machine = _internal_machine(restored)
    before = copy.deepcopy(machine)
    for change in changes:
        forged = decode_protocol_record(encode_protocol_record({**record, **change}))
        with pytest.raises(ProtocolV8Error):
            apply_protocol_record(machine, forged)
        assert machine == before
    assert apply_protocol_record(machine, record)["status"] == "committed"


def test_rehashed_original_enumeration_checkpoint_cannot_invent_cursor():
    artifact, context, _ = _operation("issue51-for-in-sparse-array", "enumeration.next")
    prefix = artifact["public"]["timeline"][: context["visibleThrough"]]
    checkpoint = _native_checkpoint(prefix, context)
    cursor = next(node for _, node in checkpoint["state"]["heap"] if node["kind"] == "enumerator")
    cursor["nextIndex"] = len(cursor["keys"]) + 1
    checkpoint["integrity"] = stable_algorithm_hash(
        {key: value for key, value in checkpoint.items() if key != "integrity"}
    )
    with pytest.raises(NativeV8Error, match="verified replay boundary"):
        _restore_native_checkpoint(
            checkpoint, expected_prefix=prefix, expected_state=context["state"]
        )
