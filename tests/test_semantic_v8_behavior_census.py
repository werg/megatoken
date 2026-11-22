"""V8 coverage comes from canonical source-bound public records."""

import base64
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from megatoken.data.detailed_v8 import parse_record
from megatoken.data.native_v8 import validate_v8_artifact
from megatoken.data.semantic_corpus_v8 import semantic_behaviors_v8


def fixture_examples(name):
    payload = (Path(__file__).parent / "fixtures" / name).read_bytes()
    document = json.loads(gzip.decompress(base64.b64decode(payload)))
    return document.get("examples", document.get("cases", [document]))


def census_for(document):
    annotations = {
        "semantic_execution": {
            "schema": "semantic-execution/8",
            "publicArtifact": {
                "public": document["public"],
                "contexts": document.get("contexts", []),
            },
        }
    }
    episode = SimpleNamespace(supervision=SimpleNamespace(annotations_json=json.dumps(annotations)))
    return semantic_behaviors_v8(episode)


def detail_rows(document, record_kind):
    return [
        row
        for row in document["public"]["timeline"]
        if row.get("kind") == "semantic"
        and isinstance(row.get("fields", {}).get("line"), str)
        and row["fields"]["line"].startswith("detail ")
        and parse_record(row["fields"]["line"], role="generated")[0] == record_kind
    ]


def test_v8_repeated_invocation_counts_distinct_root_calls_only():
    repeated = fixture_examples("semantic-retained-invocations-v8.json.gz.b64")[0]
    assert "semantic_repeated_invocation" in census_for(repeated)

    callback_turns = fixture_examples("semantic-recovery-v8.json.gz.b64")[4]
    behaviors = census_for(callback_turns)
    assert "semantic_callback" in behaviors
    assert "semantic_repeated_invocation" not in behaviors
    # A callback's parent edge is not orchestration child work. V8 SE coverage
    # requires persisted create/join/completion receipts, which this trace lacks.
    assert "semantic_child" not in behaviors


def test_v8_census_requires_observed_source_before_observed_invocation():
    repeated = fixture_examples("semantic-retained-invocations-v8.json.gz.b64")[0]

    unpublished = copy.deepcopy(repeated)
    program = next(row for row in unpublished["public"]["timeline"] if row["kind"] == "program")
    program["mode"] = "generated"
    assert "semantic_code_invocation" not in census_for(unpublished)
    assert "semantic_repeated_invocation" not in census_for(unpublished)

    published_after_calls = copy.deepcopy(repeated)
    program = next(
        row for row in published_after_calls["public"]["timeline"] if row["kind"] == "program"
    )
    program["sequence"] = 100
    assert "semantic_code_invocation" not in census_for(published_after_calls)
    assert "semantic_repeated_invocation" not in census_for(published_after_calls)

    generated_invocation = copy.deepcopy(repeated)
    for row in generated_invocation["public"]["timeline"]:
        if row["kind"] == "invocation":
            row["mode"] = "generated"
    assert "semantic_code_invocation" not in census_for(generated_invocation)
    assert "semantic_repeated_invocation" not in census_for(generated_invocation)


def test_v8_async_and_tool_feedback_parse_typed_detail_rows():
    document = fixture_examples("semantic-regions-v8.json.gz.b64")[2]
    behaviors = census_for(document)
    assert "semantic_async" in behaviors
    assert "semantic_tool_feedback" in behaviors

    for field, value in (("session", "foreign-session"), ("track", "foreign-track")):
        changed = copy.deepcopy(document)
        resume = detail_rows(changed, "resume")[0]
        resume["fields"][field] = value
        assert "semantic_async" not in census_for(changed)

    changed_result_mode = copy.deepcopy(document)
    for row in changed_result_mode["public"]["timeline"]:
        if row["kind"] == "result":
            row["mode"] = "generated"
    assert "semantic_tool_feedback" not in census_for(changed_result_mode)

    changed_result_session = copy.deepcopy(document)
    for row in changed_result_session["public"]["timeline"]:
        if row["kind"] == "result":
            row["fields"]["session"] = "foreign-session"
    assert "semantic_tool_feedback" not in census_for(changed_result_session)

    missing_invocation_track = copy.deepcopy(document)
    for row in missing_invocation_track["public"]["timeline"]:
        if row["kind"] == "invocation":
            row["fields"]["trackId"] = row["fields"].pop("track")
    behaviors = census_for(missing_invocation_track)
    assert "semantic_code_invocation" not in behaviors
    assert "semantic_async" not in behaviors
    assert "semantic_tool_feedback" not in behaviors


def test_v8_source_reread_requires_a_bound_eviction_and_exact_reread():
    artifact = json.loads(
        (Path(__file__).parent / "fixtures" / "semantic-v8-source-reread-census.json").read_text()
    )
    receipt = validate_v8_artifact(artifact)
    assert all(
        receipt["checks"][name] == "passed" for name in ("replay", "contexts", "continuation")
    )
    assert [row["kind"] for row in artifact["public"]["timeline"]].count("source-evict") == 1
    assert [row["kind"] for row in artifact["public"]["timeline"]].count("source-read") == 1
    assert "semantic_source_reread" in census_for(artifact)

    changed = copy.deepcopy(artifact)
    read = next(row for row in changed["public"]["timeline"] if row["kind"] == "source-read")
    read["fields"]["source"] += " "
    assert "semantic_source_reread" not in census_for(changed)

    unpaired = copy.deepcopy(artifact)
    unpaired["public"]["timeline"] = [
        row for row in unpaired["public"]["timeline"] if row["kind"] != "source-evict"
    ]
    assert "semantic_source_reread" not in census_for(unpaired)

    evicted_before_invocation = copy.deepcopy(artifact)
    eviction = next(
        row
        for row in evicted_before_invocation["public"]["timeline"]
        if row["kind"] == "source-evict"
    )
    eviction["sequence"] = 1
    assert "semantic_source_reread" not in census_for(evicted_before_invocation)

    read_before_eviction = copy.deepcopy(artifact)
    reread = next(
        row for row in read_before_eviction["public"]["timeline"] if row["kind"] == "source-read"
    )
    reread["sequence"] = 2
    assert "semantic_source_reread" not in census_for(read_before_eviction)

    generated_eviction = copy.deepcopy(artifact)
    eviction = next(
        row for row in generated_eviction["public"]["timeline"] if row["kind"] == "source-evict"
    )
    eviction["mode"] = "generated"
    assert "semantic_source_reread" not in census_for(generated_eviction)

    alternate_wire_names = copy.deepcopy(artifact)
    for row in alternate_wire_names["public"]["timeline"]:
        row["seq"] = row.pop("sequence")
        fields = row["fields"]
        if "track" in fields:
            fields["trackId"] = fields.pop("track")
    assert "semantic_source_reread" not in census_for(alternate_wire_names)

    missing_canonical_track = copy.deepcopy(artifact)
    for row in missing_canonical_track["public"]["timeline"]:
        if "track" in row["fields"]:
            row["fields"]["trackId"] = row["fields"].pop("track")
    assert "semantic_source_reread" not in census_for(missing_canonical_track)


def test_v8_module_initialization_requires_observed_source_bound_module_publication():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "semantic-module-init-census-v8.json").read_text()
    )
    # This exact V8 public-harness control proves census binding behavior only.
    assert (
        fixture["fixtureAuthority"]
        == "model-free-v8-public-harness-control-only; no independent source or reference qualification"
    )
    artifact = fixture["artifact"]
    program = next(row for row in artifact["timeline"] if row["kind"] == "program")
    invocation = next(row for row in artifact["timeline"] if row["kind"] == "invocation")
    assert program["mode"] == "observed"
    assert program["fields"]["module"] is True
    assert invocation["mode"] == "observed"
    assert invocation["fields"]["activation"] == "module-init"
    assert invocation["fields"]["sourceDigest"] == program["fields"]["digest"]
    assert "semantic_initialization" in census_for({"public": artifact})

    changed_module_flag = copy.deepcopy(artifact)
    next(row for row in changed_module_flag["timeline"] if row["kind"] == "program")["fields"][
        "module"
    ] = False
    assert "semantic_initialization" not in census_for({"public": changed_module_flag})

    unpublished_module = copy.deepcopy(artifact)
    next(row for row in unpublished_module["timeline"] if row["kind"] == "program")["mode"] = (
        "generated"
    )
    assert "semantic_initialization" not in census_for({"public": unpublished_module})

    substituted_source = copy.deepcopy(artifact)
    next(row for row in substituted_source["timeline"] if row["kind"] == "program")["fields"][
        "source"
    ] += " "
    assert "semantic_initialization" not in census_for({"public": substituted_source})


def test_v8_recovery_requires_exact_source_effect_reconciliation_and_result():
    artifact = fixture_examples("semantic-source-recovery-census-v8.json.gz.b64")[0]
    receipt = validate_v8_artifact(artifact)
    assert all(
        receipt["checks"][name] == "passed" for name in ("replay", "contexts", "continuation")
    )
    assert "semantic_recovery" in census_for(artifact)
    assert "semantic_source_reread" in census_for(artifact)

    missing_request = copy.deepcopy(artifact)
    missing_request["public"]["timeline"] = [
        row
        for row in missing_request["public"]["timeline"]
        if not (
            row["kind"] == "semantic"
            and parse_record(row["fields"]["line"], role="generated")[0] == "request"
        )
    ]
    assert "semantic_recovery" not in census_for(missing_request)

    foreign_dispatch = copy.deepcopy(artifact)
    next(row for row in foreign_dispatch["public"]["timeline"] if row["kind"] == "dispatch")[
        "fields"
    ]["effectId"] = "another-effect"
    assert "semantic_recovery" not in census_for(foreign_dispatch)

    missing_first_attempt = copy.deepcopy(artifact)
    next(row for row in missing_first_attempt["public"]["timeline"] if row["kind"] == "dispatch")[
        "fields"
    ]["attempt"] = 2
    assert "semantic_recovery" not in census_for(missing_first_attempt)

    dispatch_after_reconcile = copy.deepcopy(artifact)
    reconcile = next(
        row for row in dispatch_after_reconcile["public"]["timeline"] if row["kind"] == "reconcile"
    )
    dispatch = next(
        row for row in dispatch_after_reconcile["public"]["timeline"] if row["kind"] == "dispatch"
    )
    dispatch["sequence"] = reconcile["sequence"] + 2
    assert "semantic_recovery" not in census_for(dispatch_after_reconcile)

    mismatched_result = copy.deepcopy(artifact)
    result = next(row for row in mismatched_result["public"]["timeline"] if row["kind"] == "result")
    result["fields"]["status"] = "rejected"
    assert "semantic_recovery" not in census_for(mismatched_result)

    # A not-dispatched reconciliation only counts when the exact effect is
    # safely retried and an observed result arrives after that retry.
    not_dispatched = fixture_examples("semantic-recovery-v8.json.gz.b64")[3]
    assert "semantic_recovery" in census_for(not_dispatched)
    lost = fixture_examples("semantic-recovery-v8.json.gz.b64")[2]
    assert "semantic_recovery" not in census_for(lost)

    # The public timeline has no authoritative handler-lost transition, so a
    # callback reattach/evidenceId pair alone cannot establish recovery.
    reattached = fixture_examples("semantic-recovery-v8.json.gz.b64")[4]
    assert "semantic_recovery" not in census_for(reattached)
