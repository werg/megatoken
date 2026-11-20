"""Direct V8 codecs and observations work with legacy wire imports unavailable."""

import base64
import copy
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken.data.detailed_v8 import DetailedV8Error, encode_record, parse_record
from megatoken.data.observations_v8 import validate_observation_v8


def examples():
    fixture = Path(__file__).parent / "fixtures" / "semantic-regions-v8.json.gz.b64"
    return json.loads(gzip.decompress(base64.b64decode(fixture.read_bytes())))["examples"]


def test_retained_v8_detail_bytes_and_observations_validate_directly():
    count = 0
    for example in examples():
        for row in example["public"]["timeline"]:
            if row["mode"] == "observed":
                validate_observation_v8(row)
            line = row["fields"].get("line", "")
            if line.startswith("detail "):
                kind, fields = parse_record(line, role=row["mode"])
                assert encode_record(kind, fields, role=row["mode"]) == line
                count += 1
    assert count > 20


@pytest.mark.parametrize(
    "bad",
    [
        'detail {"kind":"return","kind":"return","fields":{"value":["num",1]}}',
        'detail {"fields":{"value":["num",NaN]},"kind":"return"}',
        'detail {"fields":{"value":["num",-0]},"kind":"return"}',
        'detail {"fields":{"value":["num",1]},"kind":"return","extra":1}',
        "return 1",
    ],
)
def test_v8_rejects_noncanonical_or_legacy_bytes(bad):
    with pytest.raises(DetailedV8Error):
        parse_record(bad)


def test_observation_routing_and_completion_are_checked_without_legacy_conversion():
    observed = [
        row
        for case in examples()
        for row in case["public"]["timeline"]
        if row["mode"] == "observed"
    ]
    for row in observed:
        changed = copy.deepcopy(row)
        changed["fields"]["ownerSequence"] = 1
        with pytest.raises(ValueError, match="fields"):
            validate_observation_v8(changed)
    result = next((row for row in observed if row["kind"] == "result"), None)
    if result is not None:
        changed = copy.deepcopy(result)
        changed["fields"]["result"] = ["str", "forged"]
        with pytest.raises(ValueError, match="differs"):
            validate_observation_v8(changed)


def test_detail_and_observation_modules_do_not_import_legacy_wire_code():
    source = """
import importlib.abc
import sys
class BlockLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit('.', 1)[-1] in {
            'witness_v6', 'detailed_v7', 'semantic_contract', 'algorithm_v7'
        }:
            raise AssertionError('legacy wire dependency: ' + fullname)
sys.meta_path.insert(0, BlockLegacy())
from megatoken.data.detailed_v8 import encode_record, parse_record
from megatoken.data.observations_v8 import validate_observation_v8
assert parse_record(encode_record('return', {'value': ['num', 7]}))[1]['value'] == ['num', 7]
"""
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
