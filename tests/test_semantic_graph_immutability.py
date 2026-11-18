"""Guest mutations must never rewrite earlier observed input/result evidence."""

import json
import os
from pathlib import Path

import pytest

from megatoken.data.semantic_admission import native_admission
from megatoken.data.semantic_harness_v5 import adapt_selected_harness_v5
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec

FIXTURES = ("input-mutation-short", "input-mutation-context", "result-mutation-context")


def document(name):
    return json.loads(
        (Path(__file__).parent / "fixtures/semantic-v5" / (name + ".json")).read_text()
    )


def serialized(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


@pytest.mark.parametrize("name", FIXTURES)
def test_selected_graph_import_preserves_original_document_and_record_bytes(name):
    original = document(name)
    before = serialized(original)
    episode, _ = adapt_selected_harness_v5(original, allow_local=True)
    assert serialized(original) == before
    logical = [
        parts
        for parts in episode.iter_logical_records()
        if not parts[0].logical_id.startswith(("semantic:context:", "semantic:lifecycle:"))
    ]
    assert len(logical) == len(original["public"]["timeline"])
    for item, parts in zip(original["public"]["timeline"], logical, strict=True):
        expected = item["line"] if item["kind"] == "semantic" else serialized(item)
        assert b"".join(part.payload for part in parts) == expected.encode()
    if name.endswith("context"):
        assert any(item.get("selectedState") for item in original["public"]["timeline"])


def test_rejected_context_preserves_original_evidence_after_prior_guest_mutations():
    original = document("input-mutation-context")
    context = next(item for item in original["public"]["timeline"] if item.get("selectedState"))
    context["realmPrefixDigest"] = "0" * 64
    before = serialized(original)
    with pytest.raises(ValueError, match="committed realm prefix"):
        adapt_selected_harness_v5(original, allow_local=True)
    assert serialized(original) == before


@pytest.fixture
def production_codec():
    path = Path(os.environ.get("MEGATOKEN_TOKENIZER_JSON", ".validation/tokenizer.json"))
    if not path.is_file():
        pytest.skip("Dedicated semantic CI supplies the pinned production tokenizer")
    return ByteCodec.from_tokenizer_json(path, expected_sha256=LING_TOKENIZER_SHA256)


@pytest.mark.parametrize("name", FIXTURES)
def test_production_native_admission_preserves_mutated_graph_observations(name, production_codec):
    original = document(name)
    before = serialized(original)
    receipt = native_admission(original, codec=production_codec, consumer_revision="a" * 40)
    assert receipt["admitted"] and receipt["fragmentationValidated"]
    assert receipt["visibilityValidated"] and receipt["lossMaskValidated"]
    assert receipt["generatedTokens"] > 0 and receipt["observedTokens"] > 0
    assert serialized(original) == before
