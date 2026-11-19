import unittest
import json
from pathlib import Path

from megatoken.data.protocol_v8 import ProtocolV8Error, decode_protocol_record, encode_protocol_record, replay_protocol_case, validate_protocol_state

FIXTURE=Path(__file__).parent/"fixtures"/"semantic-protocol-v8.json"


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


if __name__ == "__main__": unittest.main()
