"""Shared declared field parity plus independent causal validation and adversaries."""

import copy
import json
from pathlib import Path

import pytest

from megatoken.data.semantic_contract import CONTRACT, DEFINITION, validate_envelope_record
from megatoken.data.semantic_harness_v4 import adapt_selected_harness_v4
from megatoken.data.witness_v4 import FIELDS, REQUIRED, parse_record


def test_independent_parser_matches_frozen_shared_field_declaration():
    assert FIELDS == {key: set(value) for key, value in DEFINITION["witness"]["allowed"].items()}
    assert REQUIRED == {key: set(value) for key, value in DEFINITION["witness"]["required"].items()}
    for line in CONTRACT["fixtures"]["witness"]["valid"]:
        parse_record(line)
    for line in CONTRACT["fixtures"]["witness"]["invalid"]:
        with pytest.raises(ValueError):
            parse_record(line)
    for document in CONTRACT["fixtures"]["envelope"]["valid"]:
        for item in document["public"]["timeline"]:
            assert validate_envelope_record(item)
        adapt_selected_harness_v4(document, allow_local=True)
    for document in CONTRACT["fixtures"]["envelope"]["invalid"]:
        with pytest.raises(ValueError):
            adapt_selected_harness_v4(document, allow_local=True)


@pytest.mark.parametrize(
    "name", ["async-pair", "checkpoint-recovery", "promise-finally", "retained-events"]
)
def test_actual_js_controlled_continuity_fixtures_import(name):
    document = json.loads(
        (Path(__file__).parent / "fixtures/semantic-v4" / (name + ".json")).read_text()
    )
    episode, receipt = adapt_selected_harness_v4(document, allow_local=True)
    assert receipt.invocations >= 1
    assert not episode.supervision.verified_success
    assert all(not record.policy_mask for record in episode.records if record.role != "assistant")
    wrong = copy.deepcopy(document)
    generated = next(item for item in wrong["public"]["timeline"] if item["mode"] == "generated")
    generated["mode"] = "observed"
    with pytest.raises(ValueError, match="authority|contract"):
        adapt_selected_harness_v4(wrong, allow_local=True)


def test_job_owner_and_promise_authority_cannot_be_forged():
    document = json.loads(
        (Path(__file__).parent / "fixtures/semantic-v4/promise-finally.json").read_text()
    )
    request = next(
        item
        for item in document["public"]["timeline"]
        if item.get("line", "").startswith("request ")
    )
    request["line"] = request["line"].replace('"invocation":"job:cleanup"', '"invocation":"call"')
    with pytest.raises(ValueError, match="invocation"):
        adapt_selected_harness_v4(document, allow_local=True)
