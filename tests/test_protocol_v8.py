import unittest
import json
import base64
import gzip
from pathlib import Path

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.native_v8 import NativeV8Error, validate_v8_artifact
from megatoken.data.protocol_v8 import ProtocolV8Error, decode_protocol_record, encode_protocol_record, replay_protocol_case, validate_protocol_state

FIXTURE=Path(__file__).parent/"fixtures"/"semantic-protocol-v8.json"
REGION_FIXTURE=Path(__file__).parent/"fixtures"/"semantic-protocol-region-v8.json.gz.b64"


class ProtocolV8CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.document=json.loads(FIXTURE.read_text())

    def test_interoperability_machine_cases_replay_and_restore_checkpoints(self):
        self.assertEqual(self.document["schema"],"semantic-protocol-fixtures/1")
        for case in self.document["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual(case["qualification"],"controller-contract-fixture")
                self.assertEqual(replay_protocol_case(case),case["expected"])

    def test_roundtrips_symbol_property_and_completion_records(self):
        records = [
            {"kind":"symbol.create","id":"o3","description":["str","tag"]},
            {"kind":"symbol.for","id":"o4","key":"shared"},
            {"kind":"property.define","object":"o1","key":["symbol","o3"],
             "descriptor":{"kind":"data","value":["num",4],"writable":True,"enumerable":False,"configurable":True},"success":True},
            {"kind":"property.finish","id":"g1","completion":{"kind":"throw","value":["str","boom"]}},
        ]
        for record in records:
            with self.subTest(kind=record["kind"]):
                line=encode_protocol_record(record)
                self.assertTrue(line.startswith("protocol "))
                self.assertEqual(decode_protocol_record(line),record)

    def test_rejects_symbol_values_in_legacy_shape_and_malformed_protocol(self):
        for record in (
            {"kind":"bind","name":"s","value":["symbol","o3","extra"]},
            {"kind":"property.get","object":"o1","key":["symbol"],"receiver":["ref","o1"],"result":["undefined"]},
            {"kind":"symbol.read","symbol":["ref","o3"],"operation":"description","result":["undefined"],"bind":None},
            {"kind":"end","completion":"normal"},
        ):
            with self.subTest(record=record), self.assertRaises(ProtocolV8Error): encode_protocol_record(record)

    def test_rejects_noncanonical_or_oversized_protocol_lines(self):
        record={"kind":"bind","name":"s","value":["symbol","o3"]}
        line=encode_protocol_record(record)
        with self.assertRaises(ProtocolV8Error): decode_protocol_record("protocol { \"kind\": \"bind\", \"name\": \"s\", \"value\": [\"symbol\",\"o3\"] }")
        with self.assertRaises(ProtocolV8Error): decode_protocol_record(line+" ")
        with self.assertRaises(ProtocolV8Error): decode_protocol_record("protocol "+" "*65540)

    def test_protocol_snapshot_requires_full_symbols_and_exact_state_fields(self):
        snapshot=self.document["cases"][0]["expected"]["protocol"]
        self.assertTrue(validate_protocol_state(snapshot,dict(self.document["cases"][0]["expected"]["heap"])))
        bad=json.loads(json.dumps(snapshot));bad["symbols"]=[row for row in bad["symbols"] if row.get("id")!="well-known:unscopables"]
        with self.assertRaises(ProtocolV8Error): validate_protocol_state(bad)
        bad=json.loads(json.dumps(snapshot));bad["unchecked"]=True
        with self.assertRaises(ProtocolV8Error): validate_protocol_state(bad)


class ProtocolV8NativeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=gzip.decompress(base64.b64decode(REGION_FIXTURE.read_bytes()))
        cls.artifact=json.loads(raw)

    def test_source_qualified_protocol_region_public_golden_replays_and_continues(self):
        self.assertEqual(self.artifact["schema"],"semantic-protocol-region-fixture/1")
        self.assertEqual(stable_algorithm_hash(self.artifact["public"]),"7c8071a7861d80003a8ce020b5dca313e8af1935177a36eeab9b435473d43ddf")
        self.assertEqual(stable_algorithm_hash(self.artifact["contexts"]),"40a30688f85533e69f2090a99ff89b5fc9e315111451ee517078203c5c3ddee9")
        result=validate_v8_artifact(self.artifact)
        self.assertTrue(all(result["checks"][name]=="passed" for name in ("replay","contexts","continuation")))
        self.assertEqual(result["counts"]["protocolRecords"],11)
        self.assertEqual(result["counts"]["regionRecords"],3)
        self.assertEqual(result["counts"]["detailedRecords"],8)

    def test_protocol_context_tampering_is_rejected(self):
        bad=json.loads(json.dumps(self.artifact))
        first=next(c for c in bad["contexts"] if c["state"]["protocol"] is not None)
        first["state"]["protocol"]["sequence"]+=1
        with self.assertRaises(NativeV8Error): validate_v8_artifact(bad)

    def test_protocol_value_cannot_be_relabelled_as_legacy_detail(self):
        bad=json.loads(json.dumps(self.artifact))
        generated=bad["public"]["timeline"]
        target=next(r for r in generated if r["kind"]=="protocol")
        target["kind"]="semantic"
        target["fields"]["line"]="detail end {}"
        with self.assertRaises(NativeV8Error): validate_v8_artifact(bad)


if __name__ == "__main__": unittest.main()
