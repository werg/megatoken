import copy
import gzip
import json
from pathlib import Path

import pytest

from megatoken.data.algorithm_v7 import stable_algorithm_hash, stable_algorithm_json
from megatoken.data.detailed_context_v7 import (
    DetailedContextV7Error,
    validate_detailed_context,
    validate_detailed_continuation_v7,
)
from megatoken.data.detailed_harness_v7 import DetailedHarnessError, adapt_detailed_harness_v7
from megatoken.data.detailed_v7 import (
    DetailedV7Error,
    encode_record,
    parse_record,
)
from megatoken.data.semantic_contract_v7 import (
    CONTEXT_SCHEMA,
    CONTINUATION_SCHEMA,
    CONTRACT,
    FIXTURE_DIGEST,
    PUBLIC_SCHEMA,
    REPRESENTATION,
    SCHEMA_DIGEST,
    TRACE_SCHEMA,
)


def _minimal_context():
    source = "function main(){return 1;}"
    source_digest = stable_algorithm_hash(source)
    program = {
        "schema": PUBLIC_SCHEMA,
        "session": "session1",
        "sequence": 1,
        "ownerSequence": 1,
        "track": "host",
        "mode": "observed",
        "kind": "program",
        "programId": "program1",
        "digest": source_digest,
        "entry": "main",
        "source": source,
        "sites": [],
        "functions": [],
        "capabilities": [],
        "dependencies": [],
        "sourceMap": None,
        "languageProfile": "owned-javascript/1",
        "module": False,
        "sourceMetadata": None,
    }
    invocation = {
        "schema": PUBLIC_SCHEMA,
        "session": "session1",
        "sequence": 2,
        "ownerSequence": 1,
        "track": "worker1",
        "mode": "observed",
        "kind": "invocation",
        "programId": "program1",
        "invocationId": "invocation1",
        "realmId": "realm1",
        "triggerId": "trigger1",
        "completionPolicy": "call-return",
        "realmPolicy": "fresh",
        "inputGraph": {"roots": [], "objects": []},
        "entry": "main",
        "sourceDigest": source_digest,
        "receiver": ["undefined"],
        "closureId": None,
        "activation": "call",
        "parentInvocationId": None,
        "taskRevision": 1,
        "deadline": None,
        "bindings": {},
        "ingressKey": "ingress1",
    }
    state = {
        "schema": CONTINUATION_SCHEMA,
        "heap": [],
        "locals": [],
        "captureAliases": [],
        "stack": ["invocation1"],
        "continuations": [],
        "ready": [],
        "callReturn": None,
        "pending": None,
        "completion": None,
        "randomState": None,
    }
    prefix = {
        "schema": "selected-inference-prefix/3",
        "representation": REPRESENTATION,
        "traceFormat": TRACE_SCHEMA,
        "programId": "program1",
        "entry": "main",
        "source": source,
        "sourceDigest": source_digest,
        "invocationId": "invocation1",
        "trackId": "worker1",
        "realmId": "realm1",
        "committed": [program, invocation],
        "realmCommitted": [program, invocation],
        "completionPolicy": "call-return",
        "maxOutstanding": 64,
        "taskRevision": 1,
        "parentInvocationId": None,
        "receiver": ["undefined"],
        "closureId": None,
        "activation": "call",
        "selectedState": state,
    }
    context = {
        "schema": CONTEXT_SCHEMA,
        "representation": REPRESENTATION,
        "prefix": prefix,
        "visibleThrough": 2,
        "prefixDigest": stable_algorithm_hash(prefix),
    }
    line = encode_record("return", {"value": ["num", 1]}, role="generated")
    target = {
        "schema": PUBLIC_SCHEMA,
        "session": "session1",
        "sequence": 3,
        "ownerSequence": 2,
        "track": "worker1",
        "mode": "generated",
        "kind": "semantic",
        "invocationId": "invocation1",
        "traceFormat": TRACE_SCHEMA,
        "line": line,
    }
    return [program, invocation], context, target


def test_contract_digests_and_all_canonical_detailed_vectors():
    assert stable_algorithm_hash(CONTRACT["definition"]) == SCHEMA_DIGEST
    assert stable_algorithm_hash(CONTRACT["fixtures"]) == FIXTURE_DIGEST
    fixtures = CONTRACT["fixtures"]["detailed"]
    assert len(fixtures["canonical"]) == 83
    for vector in fixtures["canonical"]:
        role = vector["role"]
        decoded = parse_record(vector["line"], role=role)
        assert decoded == (vector["record"]["kind"], vector["record"]["fields"])
        assert encode_record(*decoded, role=role) == vector["line"]


def test_malformed_and_authority_mismatch_vectors_are_rejected():
    fixtures = CONTRACT["fixtures"]["detailed"]
    for line in fixtures["invalid"]:
        with pytest.raises(DetailedV7Error):
            parse_record(line)
    for line in fixtures["invalidGenerated"]:
        with pytest.raises(DetailedV7Error):
            parse_record(line, role="generated")
    for line in fixtures["invalidObserved"]:
        with pytest.raises(DetailedV7Error):
            parse_record(line, role="observed")
    for kind, fields in (
        ("code", {"entry": "main", "source": "function main(){}", "sites": []}),
        ("input", {"graph": {"roots": [], "objects": []}}),
        ("observe", {"effect": "e1", "result": ["num", 1], "status": "fulfilled"}),
    ):
        with pytest.raises(DetailedV7Error):
            encode_record(kind, fields, role="generated")
    for kind, fields in (
        ("return", {"value": ["num", 1]}),
        ("compute", {"expr": {"op": "literal", "value": ["num", 1]}, "result": ["num", 1]}),
    ):
        with pytest.raises(DetailedV7Error):
            encode_record(kind, fields, role="observed")


@pytest.mark.parametrize(
    "record",
    [
        ("value-open", {"id": "r1", "type": "detail"}),
        ("value-part", {"id": "r1", "data": "return 1"}),
        ("value-close", {"id": "r1"}),
    ],
)
def test_logical_fragments_are_rejected_for_every_role(record):
    kind, fields = record
    for role in ("any", "generated", "observed"):
        with pytest.raises(DetailedV7Error):
            encode_record(kind, fields, role=role)


def test_numeric_object_keys_follow_javascript_enumeration_after_sorting():
    assert stable_algorithm_json({"01": "ordinary", "1": "index", "2": "index"}) == (
        '{"1":"index","2":"index","01":"ordinary"}'
    )


def test_detailed_context_restores_exact_current_public_prefix():
    prior, context, target = _minimal_context()
    assert validate_detailed_context(context, prior, target)
    restored = validate_detailed_continuation_v7(context, prior_timeline=prior, target=target)
    assert restored == context["prefix"]["selectedState"]


def test_context_rejects_missing_public_history_future_rows_and_mutated_state():
    prior, context, target = _minimal_context()
    missing = copy.deepcopy(context)
    missing["prefix"]["committed"].pop()
    missing["prefixDigest"] = stable_algorithm_hash(missing["prefix"])
    with pytest.raises(DetailedContextV7Error, match="committed prefix"):
        validate_detailed_context(missing, prior, target)

    future = copy.deepcopy(context)
    future["prefix"]["visibleThrough"] = 3
    future["prefixDigest"] = stable_algorithm_hash(future["prefix"])
    future["visibleThrough"] = 3
    with pytest.raises(DetailedContextV7Error):
        validate_detailed_context(future, prior, target)

    expected_state = copy.deepcopy(context["prefix"]["selectedState"])
    changed = copy.deepcopy(context)
    changed["prefix"]["selectedState"]["stack"] = ["forged"]
    changed["prefixDigest"] = stable_algorithm_hash(changed["prefix"])
    with pytest.raises(DetailedContextV7Error, match="independent public-prefix replay"):
        validate_detailed_context(changed, prior, target, expected_state=expected_state)


def _interop_exports():
    path = Path(__file__).parent / "fixtures" / "semantic-v7-detailed-interoperability.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def test_all_five_fresh_detailed_exports_replay_with_exact_contexts():
    exports = _interop_exports()
    assert {item["case"] for item in exports} == {
        "branch-1",
        "throw-finally",
        "rejection-finally",
        "callback-captured-write",
        "persistent-external-trigger",
    }
    for item in exports:
        case = item["case"]
        export = item["export"]
        assert export["public"]["schema"] == PUBLIC_SCHEMA
        assert export["public"]["representation"] == REPRESENTATION
        assert len(export["contexts"]) == sum(
            row["kind"] == "semantic" for row in export["public"]["timeline"]
        )
        episode, receipt = adapt_detailed_harness_v7(export, allow_local=True)
        assert episode.records
        assert {track.track_id for track in episode.tracks} == {"assistant"}
        assert all(
            event.track_id == "assistant" for group in episode.meridians for event in group.events
        )
        assert episode.track_revisions == ()
        assert not any(
            record.record_id.startswith("semantic:lifecycle:") for record in episode.records
        )
        assert sum(record.payload.startswith(b'{"prefix":') for record in episode.records) == len(
            export["contexts"]
        )
        assert receipt.admission == "locally_verified"
        assert receipt.context_digest == stable_algorithm_hash(export["contexts"]), case
        assert receipt.public_digest == stable_algorithm_hash(export["public"]), case
        assert receipt.actions >= 0
        assert receipt.completions
        with pytest.raises(DetailedHarnessError, match="allow_local=True"):
            adapt_detailed_harness_v7(export)


def test_all_five_exports_reject_context_tampering_even_after_rehashing():
    for item in _interop_exports():
        mutated = copy.deepcopy(item["export"])
        context = mutated["contexts"][0]
        context["prefix"]["realmCommitted"].pop()
        context["prefixDigest"] = stable_algorithm_hash(context["prefix"])
        mutated["sidecar"]["contextDigest"] = stable_algorithm_hash(mutated["contexts"])
        with pytest.raises(DetailedHarnessError, match="realm prefix"):
            adapt_detailed_harness_v7(mutated, allow_local=True)

        timeline = mutated["public"]["timeline"]
        target = next(row for row in timeline if row["kind"] == "semantic")
        prior = timeline[: target["sequence"] - 1]
        with pytest.raises(DetailedContextV7Error):
            validate_detailed_continuation_v7(context, prior_timeline=prior, target=target)
