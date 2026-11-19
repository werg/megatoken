import unittest

from megatoken.data.protocol_v8 import ProtocolV8Error, apply_protocol_record, create_protocol_state


class BinaryMemoryConversionTests(unittest.TestCase):
    def setUp(self):
        self.machine = {
            "protocol": create_protocol_state(),
            "heap": {},
            "stack": ["root"],
            "locals": {},
            "captureAliases": {},
            "frameFunctions": {"root": "root"},
            "receiver": ["undefined"],
            "pending": None,
            "lastReturn": None,
            "completion": None,
        }

    def construct(self, constructor, name, arguments):
        apply_protocol_record(
            self.machine,
            {
                "kind": "memory.construct",
                "constructor": constructor,
                "name": name,
                "backing": None,
                "arguments": arguments,
            },
        )

    def set(self, object_id, key, value):
        apply_protocol_record(
            self.machine,
            {
                "kind": "memory.set",
                "object": object_id,
                "key": ["str", key],
                "value": value,
            },
        )

    def get(self, object_id, key, result):
        apply_protocol_record(
            self.machine,
            {
                "kind": "memory.get",
                "object": object_id,
                "key": ["str", key],
                "result": result,
            },
        )

    def call(self, method, receiver, arguments, result):
        apply_protocol_record(
            self.machine,
            {
                "kind": "memory.call",
                "method": method,
                "receiver": receiver,
                "arguments": arguments,
                "result": result,
                "name": None,
            },
        )

    def test_nonfinite_integer_typed_and_data_view_writes_store_zero(self):
        self.construct("ArrayBuffer", "buffer", [["num", 4]])
        self.construct("Int16Array", "typed", [["ref", "buffer"]])
        self.construct("DataView", "view", [["ref", "buffer"]])

        for value in (
            ["num", "NaN"],
            ["num", "+Infinity"],
            ["num", "-Infinity"],
            ["str", "inf"],
            ["undefined"],
        ):
            self.set("typed", "0", value)
            self.get("typed", "0", ["num", 0])
            self.call(
                "DataView.prototype.setInt16",
                ["ref", "view"],
                [["num", 0], value, ["bool", True]],
                ["undefined"],
            )
            self.call(
                "DataView.prototype.getInt16",
                ["ref", "view"],
                [["num", 0], ["bool", True]],
                ["num", 0],
            )
            self.assertEqual(self.machine["heap"]["buffer"]["bytes"][:2], [0, 0])

    def test_compare_exchange_converts_expected_to_element_width(self):
        self.construct("SharedArrayBuffer", "shared", [["num", 4]])
        self.construct("Int8Array", "view", [["ref", "shared"]])
        self.set("view", "0", ["num", 1])

        # 257 converts to the stored Int8 value 1 before the comparison.
        self.call(
            "Atomics.compareExchange",
            None,
            [["ref", "view"], ["num", 0], ["num", 257], ["num", 9]],
            ["num", 1],
        )
        self.assertEqual(self.machine["heap"]["shared"]["bytes"][0], 9)

    def test_integer_writes_modulo_and_float32_overflow(self):
        self.construct("ArrayBuffer", "buffer", [["num", 8]])
        self.construct("ArrayBuffer", "float-buffer", [["num", 4]])
        self.construct("Uint8Array", "bytes", [["ref", "buffer"]])
        self.construct("Float32Array", "floats", [["ref", "float-buffer"]])
        self.set("bytes", "0", ["num", "+Infinity"])
        self.set("bytes", "1", ["num", 257])
        self.set("floats", "0", ["num", 1e100])
        self.assertEqual(self.machine["heap"]["buffer"]["bytes"][0:2], [0, 1])
        self.assertEqual(self.machine["heap"]["float-buffer"]["bytes"], [0, 0, 128, 127])

    def test_atomics_store_returns_unwrapped_conversion_but_stores_wrapped_value(self):
        self.construct("SharedArrayBuffer", "shared", [["num", 8]])
        self.construct("Uint8Array", "byte", [["ref", "shared"], ["num", 0], ["num", 1]])

        for operand, result, stored in (
            (["num", 257], ["num", 257], ["num", 1]),
            (["num", 1.9], ["num", 1], ["num", 1]),
            (["num", "-0"], ["num", 0], ["num", 0]),
            (["num", "NaN"], ["num", 0], ["num", 0]),
            (["undefined"], ["num", 0], ["num", 0]),
            (["num", "+Infinity"], ["num", "+Infinity"], ["num", 0]),
            (["num", "-Infinity"], ["num", "-Infinity"], ["num", 0]),
        ):
            self.call("Atomics.store", None, [["ref", "byte"], ["num", 0], operand], result)
            self.get("byte", "0", stored)

    def test_bigint_atomics_store_returns_unwrapped_bigint(self):
        self.construct("SharedArrayBuffer", "shared", [["num", 8]])
        self.construct("BigUint64Array", "words", [["ref", "shared"]])
        original = str((1 << 64) + 7)
        self.call(
            "Atomics.store",
            None,
            [["ref", "words"], ["num", 0], ["bigint", original]],
            ["bigint", original],
        )
        self.get("words", "0", ["bigint", "7"])

    def test_atomics_wait_expected_uses_int32_width_conversion(self):
        self.construct("SharedArrayBuffer", "shared", [["num", 4]])
        self.construct("Int32Array", "words", [["ref", "shared"]])
        self.set("words", "0", ["num", 1])
        self.call(
            "Atomics.wait",
            None,
            [["ref", "words"], ["num", 0], ["num", (1 << 32) + 1], ["num", 0]],
            ["str", "timed-out"],
        )

    def test_data_view_endianness_uses_bigint_truthiness(self):
        self.construct("ArrayBuffer", "buffer", [["num", 2]])
        self.construct("DataView", "data", [["ref", "buffer"]])
        self.call(
            "DataView.prototype.setInt16",
            ["ref", "data"],
            [["num", 0], ["num", 258], ["bigint", "0"]],
            ["undefined"],
        )
        self.assertEqual(self.machine["heap"]["buffer"]["bytes"], [1, 2])
        self.call(
            "DataView.prototype.setInt16",
            ["ref", "data"],
            [["num", 0], ["num", 258], ["bigint", "1"]],
            ["undefined"],
        )
        self.assertEqual(self.machine["heap"]["buffer"]["bytes"], [2, 1])

    def test_bigint_strings_use_ecmascript_grammar_in_all_storage_paths(self):
        self.construct("SharedArrayBuffer", "shared", [["num", 8]])
        self.construct("BigInt64Array", "words", [["ref", "shared"]])
        self.construct("DataView", "data", [["ref", "shared"]])
        for text, expected in (
            ("", "0"),
            ("  ", "0"),
            ("0x10", "16"),
            ("0b11", "3"),
            ("0o12", "10"),
            ("-12", "-12"),
        ):
            with self.subTest(text=text):
                self.set("words", "0", ["str", text])
                self.get("words", "0", ["bigint", expected])
                self.call(
                    "DataView.prototype.setBigInt64",
                    ["ref", "data"],
                    [["num", 0], ["str", text], ["bool", True]],
                    ["undefined"],
                )
                self.call(
                    "DataView.prototype.getBigInt64",
                    ["ref", "data"],
                    [["num", 0], ["bool", True]],
                    ["bigint", expected],
                )
                self.call(
                    "Atomics.store",
                    None,
                    [["ref", "words"], ["num", 0], ["str", text]],
                    ["bigint", expected],
                )
        for text in ("1_0", "1.0", "+0x10"):
            with (
                self.subTest(invalid=text),
                self.assertRaisesRegex(ProtocolV8Error, "BigInt conversion"),
            ):
                self.set("words", "0", ["str", text])


if __name__ == "__main__":
    unittest.main()
