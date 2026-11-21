"""Native public replay of freshly source-checked shared conversion inputs.

The five retained source refusals remain in the denominator. These tests prove
native replay and continuation, not tokenizer admission or model execution.
"""

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
from megatoken.data.prefix_v8 import validate_v8_prediction_prefix
from megatoken.data.semantic_json import stable_algorithm_hash

FIXTURE = Path(__file__).parent / "fixtures/semantic-primitive-conversion-v8.json.gz.b64"
DOCUMENT = json.loads(gzip.decompress(base64.b64decode(FIXTURE.read_bytes())))
ACCEPTED = [row for row in DOCUMENT["cases"] if "artifact" in row]


def test_complete_original_population_and_refusals_are_preserved():
    original = [
        "key-read",
        "key-getter",
        "key-symbol-hook",
        "key-fallback",
        "key-live-fallback",
        "key-getter-throw",
        "key-method-throw-identity",
        "key-noncallable",
        "key-exhaustion",
        "key-number-nested",
        "number-getter",
        "number-existing",
        "key-plain-write",
        "key-compound-divergence",
        "key-logical-divergence",
        "key-update-divergence",
    ]
    assert DOCUMENT["schema"] == "semantic-primitive-conversion-fixtures/1"
    assert DOCUMENT["originalIds"] == original
    assert [row["id"] for row in DOCUMENT["cases"][:16]] == original
    assert len(DOCUMENT["cases"]) == 24
    assert len({row["id"] for row in DOCUMENT["cases"]}) == 24
    assert len(ACCEPTED) == 19
    assert DOCUMENT["qualification"] == {
        "source": "checked-during-generation",
        "public": "replayed",
        "native": "untested",
        "model": "unmeasured",
    }
    boundaries = {}
    for row in DOCUMENT["cases"]:
        assert row["sourceInputDigest"] == stable_algorithm_hash(row["fixture"])
        if "artifact" not in row:
            assert row["status"] == "source-unqualified"
            assert row["reason"]
            boundaries[row["id"]] = row["boundary"]
        else:
            assert row["status"] == "source-qualified"
    assert boundaries == {
        "key-noncallable": "intrinsic-error-source-metadata",
        "key-exhaustion": "intrinsic-error-source-metadata",
        "key-compound-divergence": "node24-reference-disagreement",
        "key-logical-divergence": "node24-reference-disagreement",
        "key-update-divergence": "node24-reference-disagreement",
    }


@pytest.mark.parametrize("case", ACCEPTED, ids=lambda row: row["id"])
def test_native_source_artifact_replay_and_every_checkpoint(case):
    artifact = case["artifact"]
    checked = validate_v8_artifact(artifact)
    assert checked["admitted"] is False
    assert all(checked["checks"][key] == "passed" for key in ("replay", "contexts", "continuation"))
    for context in artifact["contexts"]:
        prefix = artifact["public"]["timeline"][: context["visibleThrough"]]
        checkpoint = _native_checkpoint(prefix, context)
        restored = _restore_native_checkpoint(
            checkpoint,
            expected_prefix=prefix,
            expected_state=context["state"],
            expected_cursor=context["regionCursor"],
            expected_obligations=context["obligations"],
        )
        assert restored == context["state"]


@pytest.mark.parametrize("case", ACCEPTED, ids=lambda row: row["id"])
def test_prediction_prefix_restores_each_conversion_phase_without_future_outputs(case):
    artifact = case["artifact"]
    observed = set()
    for index, context in enumerate(artifact["contexts"]):
        protocol = context["state"]["protocol"]
        conversions = (protocol or {}).get("primitiveConversions")
        rows = conversions["operations"] if conversions else []
        # Each distinct retained phase, goal and nesting in this original source
        # is checked with only already committed records and prior contexts.
        signature = tuple((row["goal"], row["phase"]) for row in rows)
        if signature in observed:
            continue
        observed.add(signature)
        public = {
            **artifact["public"],
            "timeline": artifact["public"]["timeline"][: context["visibleThrough"]],
        }
        checked = validate_v8_prediction_prefix(public, artifact["contexts"][:index], context)
        assert checked["context"] == context
    assert any(signature for signature in observed), (
        "Source must execute the registered conversion machine"
    )


def test_fresh_hashes_cannot_authorize_forged_conversion_checkpoint_state():
    artifact = next(row["artifact"] for row in ACCEPTED if row["id"] == "key-getter")
    context = next(
        context
        for context in artifact["contexts"]
        if any(
            row["phase"] == "getter-waiting"
            for row in ((context["state"]["protocol"] or {}).get("primitiveConversions") or {}).get(
                "operations", []
            )
        )
    )
    prefix = artifact["public"]["timeline"][: context["visibleThrough"]]
    checkpoint = _native_checkpoint(prefix, context)
    for field, value in (
        ("goal", "Number"),
        ("method", ["undefined"]),
        ("child", None),
        ("candidate", ["str", "invented"]),
        ("phase", "complete"),
    ):
        forged = copy.deepcopy(checkpoint)
        forged["state"]["protocol"]["primitiveConversions"]["operations"][0][field] = value
        forged["integrity"] = stable_algorithm_hash(
            {key: item for key, item in forged.items() if key != "integrity"}
        )
        with pytest.raises(NativeV8Error, match="verified replay boundary"):
            _restore_native_checkpoint(
                forged, expected_prefix=prefix, expected_state=context["state"]
            )


def test_rehashed_context_cannot_substitute_a_conversion_goal():
    artifact = next(row["artifact"] for row in ACCEPTED if row["id"] == "key-read")
    index, context = next(
        (index, context)
        for index, context in enumerate(artifact["contexts"])
        if (context["state"]["protocol"] or {}).get("primitiveConversions")
    )
    public = {
        **artifact["public"],
        "timeline": artifact["public"]["timeline"][: context["visibleThrough"]],
    }
    forged = copy.deepcopy(context)
    forged["state"]["protocol"]["primitiveConversions"]["operations"][0]["goal"] = "Number"
    forged["prefixDigest"] = stable_algorithm_hash(public["timeline"])
    with pytest.raises(NativeV8Error, match="context differs from independent replay"):
        validate_v8_prediction_prefix(public, artifact["contexts"][:index], forged)
