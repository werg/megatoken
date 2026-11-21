"""Coverage comes from native-checked source histories and actual training bytes.

The recovery fixture retains a fresh original-source recheck at its recorded
producer. Byte fallback here exercises framing only; it is not token admission.
"""

import base64
import copy
import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from megatoken.data.native_v8 import prepare_native_v8_training, validate_v8_artifact
from megatoken.data.schema import encode_episode
from megatoken.data.semantic_corpus_v8 import semantic_behaviors_v8
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import ByteCodec


def packet(name):
    raw = (Path(__file__).parent / "fixtures" / name).read_bytes()
    return json.loads(gzip.decompress(base64.b64decode(raw)))


def recovery():
    return packet("semantic-source-recovery-v8.json.gz.b64")["cases"][0]["artifact"]


def episode(document, *, max_positions=1024):
    prepared, _ = prepare_native_v8_training(
        document, codec=ByteCodec.byte_fallback(), max_positions=max_positions, max_history=128
    )
    annotations = json.loads(prepared.supervision.annotations_json)
    annotations["semantic_execution"]["publicArtifact"] = {
        key: document[key] for key in ("public", "contexts")
    }
    return replace(
        prepared,
        supervision=replace(prepared.supervision, annotations_json=json.dumps(annotations)),
    )


def prefix(document, through):
    document = copy.deepcopy(document)
    document["public"]["timeline"] = document["public"]["timeline"][:through]
    count = sum(row["mode"] == "generated" for row in document["public"]["timeline"])
    document["contexts"] = document["contexts"][:count]
    return document


def test_original_source_recovery_counts_real_feedback_async_and_reread():
    assert semantic_behaviors_v8(episode(recovery())) == {
        "semantic_v8",
        "semantic_code_invocation",
        "semantic_tool_feedback",
        "semantic_async",
        "semantic_source_reread",
        "semantic_recovery",
    }


@pytest.mark.parametrize(
    ("through", "present", "absent"),
    [
        (7, set(), {"semantic_async", "semantic_tool_feedback", "semantic_source_reread"}),
        (8, set(), {"semantic_async", "semantic_tool_feedback", "semantic_source_reread"}),
        (9, {"semantic_source_reread"}, {"semantic_async", "semantic_tool_feedback"}),
        (
            11,
            {"semantic_source_reread", "semantic_recovery"},
            {"semantic_async", "semantic_tool_feedback"},
        ),
        (12, {"semantic_source_reread", "semantic_async", "semantic_tool_feedback"}, set()),
    ],
)
def test_coverage_does_not_anticipate_later_causal_work(through, present, absent):
    coverage = semantic_behaviors_v8(episode(prefix(recovery(), through)))
    assert present <= coverage
    assert not absent & coverage


def test_source_callbacks_are_repeated_invocations_but_first_reads_are_not_rereads():
    source = next(
        row["artifact"]
        for row in packet("semantic-callback-return-v8.json.gz.b64")["cases"]
        if row["id"] == "return-callback-receiver-alias"
    )
    kinds = [row["kind"] for row in source["public"]["timeline"]]
    assert kinds.count("invocation") == 3
    assert kinds.count("source-read") == 3
    assert "source-evict" not in kinds
    coverage = semantic_behaviors_v8(episode(source))
    assert {"semantic_callback", "semantic_repeated_invocation"} <= coverage
    assert not {"semantic_child", "semantic_source_reread"} & coverage


def test_census_is_independent_of_physical_chunk_geometry():
    document = recovery()
    small, large = episode(document, max_positions=512), episode(document, max_positions=2048)
    assert len(small.records) > len(large.records)
    assert semantic_behaviors_v8(small) == semantic_behaviors_v8(large)


@pytest.mark.parametrize("logical_prefix", ["observation", "context", "target"])
def test_altered_training_bytes_cannot_retain_public_coverage(logical_prefix):
    prepared = episode(recovery())
    index = next(
        i
        for i, row in enumerate(prepared.records)
        if row.logical_id.startswith("semantic:v8:" + logical_prefix + ":")
    )
    records = list(prepared.records)
    records[index] = replace(records[index], payload=records[index].payload[:-1] + b"X")
    changed = replace(prepared, records=tuple(records))
    with pytest.raises(ValueError, match="differs from training records"):
        semantic_behaviors_v8(changed)


@pytest.mark.parametrize("change", ["mask", "role", "identity", "part_index", "part_count"])
def test_training_authority_and_framing_are_bound(change):
    prepared = episode(recovery())
    records = list(prepared.records)
    index = next(i for i, row in enumerate(records) if row.policy_mask) if change == "mask" else 0
    first = records[index]
    mutations = {
        "mask": {"policy_mask": False},
        "role": {"role": "assistant"},
        "identity": {"logical_id": "semantic:v8:observation:999"},
        "part_index": {"part_index": 1},
        "part_count": {"part_count": first.part_count + 1},
    }
    # Some malformed framing already fails at the ordinary episode boundary;
    # the remaining valid physical records must fail their public-history bind.
    with pytest.raises(ValueError):
        records[index] = replace(first, **mutations[change])
        semantic_behaviors_v8(replace(prepared, records=tuple(records)))


def test_valid_but_different_retained_history_cannot_label_original_rows():
    prepared = episode(recovery())
    changed = json.loads(prepared.supervision.annotations_json)
    changed["semantic_execution"]["publicArtifact"] = prefix(recovery(), 12)
    with pytest.raises(ValueError, match="differs from training records"):
        semantic_behaviors_v8(
            replace(
                prepared,
                supervision=replace(prepared.supervision, annotations_json=json.dumps(changed)),
            )
        )


def test_census_replays_the_retained_history_instead_of_trusting_annotations():
    prepared = episode(recovery())
    changed = json.loads(prepared.supervision.annotations_json)
    changed["semantic_execution"]["publicArtifact"]["public"]["timeline"][10]["fields"][
        "effectId"
    ] = "unrequested"
    with pytest.raises(ValueError):
        semantic_behaviors_v8(
            replace(
                prepared,
                supervision=replace(prepared.supervision, annotations_json=json.dumps(changed)),
            )
        )


def test_declared_curriculum_labels_do_not_create_child_coverage():
    prepared = episode(recovery())
    changed = json.loads(prepared.supervision.annotations_json)
    changed["semantic_execution"]["behaviors"] = ["semantic_child"]
    changed["semantic_execution"]["phase"] = "SE"
    changed["semantic_execution"]["orchestration"] = {"completedChildren": 1}
    assert "semantic_child" not in semantic_behaviors_v8(
        replace(
            prepared,
            supervision=replace(prepared.supervision, annotations_json=json.dumps(changed)),
        )
    )


@pytest.mark.parametrize("mask_kind", ["free_spans", "emitted_free_mask"])
def test_disabled_target_positions_cannot_earn_training_coverage(mask_kind):
    prepared = episode(recovery())
    codec = ByteCodec.byte_fallback()
    assert sum(sum(row.free_mask) for row in encode_episode(prepared, codec)) > 0
    records = []
    for row in prepared.records:
        if row.policy_mask:
            if mask_kind == "free_spans":
                row = replace(row, free_spans=())
            else:
                row = replace(
                    row,
                    emitted_ids=row.cached_ids,
                    emitted_free_mask=(False,) * len(row.cached_ids),
                )
        records.append(row)
    disabled = replace(prepared, records=tuple(records))
    assert sum(sum(row.free_mask) for row in encode_episode(disabled, codec)) == 0
    with pytest.raises(ValueError, match="differs from training records"):
        semantic_behaviors_v8(disabled)


@pytest.mark.parametrize("change", ["source", "program", "digest", "read_type", "digest_type"])
def test_native_replay_rejects_false_source_cache_observations_after_rehashing(change):
    changed = copy.deepcopy(recovery())
    for row in changed["public"]["timeline"]:
        if row["kind"] not in {"source-read", "source-evict"}:
            continue
        if change == "source" and row["kind"] == "source-read":
            row["fields"]["source"] += "\n/* changed */"
        elif change == "program":
            row["fields"]["programId"] = "unpublished"
        elif change == "digest":
            row["fields"]["digest"] = "0" * 64
        elif change == "read_type" and row["kind"] == "source-read":
            row["fields"]["source"] = {"source": row["fields"]["source"]}
        elif change == "digest_type":
            row["fields"]["digest"] = [row["fields"]["digest"]]
    for context in changed["contexts"]:
        context["prefixDigest"] = stable_algorithm_hash(
            changed["public"]["timeline"][: context["visibleThrough"]]
        )
    with pytest.raises(ValueError, match="[Ss]ource"):
        validate_v8_artifact(changed)
    with pytest.raises(ValueError, match="[Ss]ource"):
        prepare_native_v8_training(changed, codec=ByteCodec.byte_fallback())


def test_only_observed_inputs_do_not_establish_training_behavior():
    with pytest.raises(ValueError, match="requires checked generated training targets"):
        semantic_behaviors_v8(episode(prefix(recovery(), 2)))
