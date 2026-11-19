import json
from pathlib import Path

import pytest

from megatoken.data.witness_v6 import WitnessV6Error, encode_record, parse_record

FIXTURE = Path(__file__).parent / "fixtures/semantic-v6/trace.json"


def _fixture():
    return json.loads(FIXTURE.read_text())


def test_v6_canonical_fixture_parses_and_round_trips_exactly():
    rows = _fixture()["canonical"]
    assert len(rows) >= 70
    for row in rows:
        kind, fields = parse_record(row["line"])
        assert kind == row["kind"]
        assert fields == row["fields"]
        assert encode_record(kind, fields) == row["line"]


def test_v6_rejects_every_noncanonical_fixture_string():
    invalid = _fixture()["invalid"]
    assert len(invalid) >= 70
    for line in invalid:
        with pytest.raises(WitnessV6Error):
            parse_record(line)


@pytest.mark.parametrize(
    "line",
    [
        'return {"value":["num",1]}',
        'bind {"name":"x","value":["num",1]}',
        "evaluate main(1) => 1 choosing 1..2:then",
    ],
)
def test_v6_never_accepts_v5_json_or_unregistered_summary_branch_names(line):
    with pytest.raises(WitnessV6Error):
        parse_record(line)


def test_summary_profiles_keep_only_their_declared_claims():
    line = "evaluate main(2, 3) => 5"
    assert parse_record(line) == (
        "summarize",
        {
            "entry": "main",
            "arguments": [["num", 2], ["num", 3]],
            "result": ["num", 5],
            "profile": "pure-helper/1",
        },
    )
