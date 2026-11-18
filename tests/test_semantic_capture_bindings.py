"""Source capture aliases are JavaScript names, not heap-reference identities."""

import json
import os
from pathlib import Path

import pytest

from megatoken.data.semantic_admission import native_admission
from megatoken.data.witness_v5 import WitnessV5Error, parse_record, value
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec


def allocation(captures):
    fields = {
        "name": "o1",
        "kind": "closure",
        "properties": [],
        "code": "function:1",
        "captures": captures,
    }
    return "allocate " + json.dumps(fields, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize(
    "name", ["keyFn", "_key", "key_fn", "keyFn_2", "$key", "key$Fn", "$"]
)
def test_capture_aliases_and_cells_use_the_existing_source_name_grammar(name):
    captures = [
        {"binding": name, "cell": "outerCell"},
        {"binding": "alias", "cell": name},
    ]
    assert parse_record(allocation(captures))[1]["captures"] == captures
    assert (
        parse_record(
            "bind " + json.dumps({"name": name, "from": name, "value": ["num", 1]})
        )[0]
        == "bind"
    )


@pytest.mark.parametrize("field", ["binding", "cell"])
@pytest.mark.parametrize(
    "invalid", ["", "1key", "key-fn", "key fn", "key.fn", "o:1", None, 3, ["ref", "o1"]]
)
def test_malformed_capture_aliases_and_cells_are_rejected(field, invalid):
    relation = {"binding": "keyFn", "cell": "outerCell"}
    relation[field] = invalid
    with pytest.raises(WitnessV5Error, match="Malformed shared cell capture relation"):
        parse_record(allocation([relation]))


def test_distinct_aliases_can_share_one_cell_but_duplicate_aliases_are_rejected():
    with pytest.raises(WitnessV5Error, match="Malformed shared cell capture relation"):
        parse_record(
            allocation(
                [
                    {"binding": "keyFn", "cell": "firstCell"},
                    {"binding": "keyFn", "cell": "secondCell"},
                ]
            )
        )
    captures = [
        {"binding": "left", "cell": "sharedCell"},
        {"binding": "right", "cell": "sharedCell"},
    ]
    assert parse_record(allocation(captures))[1]["captures"] == captures


@pytest.mark.parametrize("name", ["keyFn", "_key", "$key", "o:1"])
def test_source_names_do_not_relax_heap_reference_identity(name):
    value(["ref", "o1"])
    with pytest.raises(WitnessV5Error):
        value(["ref", name])
    with pytest.raises(WitnessV5Error, match="Invalid witness identity"):
        parse_record(
            "read "
            + json.dumps(
                {"object": name, "key": "x", "value": ["num", 1], "present": True}
            )
        )


def test_pinned_vscode_distinct_capture_passes_production_tokenizer_admission():
    tokenizer = Path(
        os.environ.get("MEGATOKEN_TOKENIZER_JSON", ".validation/tokenizer.json")
    )
    if not tokenizer.is_file():
        pytest.skip("Dedicated semantic CI supplies the pinned production tokenizer")
    document = json.loads(
        (
            Path(__file__).parent / "fixtures/semantic-v5/source-binding-capture.json"
        ).read_text()
    )
    before = json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    assert any(
        {"binding": "keyFn", "cell": "keyFn"} in fields.get("captures", [])
        for row in document["public"]["timeline"]
        if row["kind"] == "semantic"
        for _, fields in [parse_record(row["line"])]
    )
    codec = ByteCodec.from_tokenizer_json(
        tokenizer, expected_sha256=LING_TOKENIZER_SHA256
    )
    receipt = native_admission(document, codec=codec, consumer_revision="a" * 40)
    assert receipt["admitted"] and receipt["visibilityValidated"]
    assert receipt["lossMaskValidated"] and receipt["fragmentationValidated"]
    assert receipt["generatedTokens"] > 0
    assert receipt["optimizerSteps"] == 0
    assert json.dumps(document, ensure_ascii=False, separators=(",", ":")) == before
