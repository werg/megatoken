import hashlib
import math
import unittest

from megatoken.data.algorithm_v7 import (
    ALGORITHM_PROFILE,
    ALGORITHM_VERSION,
    OPERATIONS,
    algorithm_digest,
    algorithm_heap_from_graph,
    canonical_algorithm_graph,
    encode_algorithm_record,
    encode_algorithm_term,
    evaluate_algorithm,
    evaluate_term,
    parse_algorithm_record,
    stable_algorithm_hash,
    stable_algorithm_json,
    validate_algorithm,
    validate_algorithm_record,
    validate_term,
)


def lit(value):
    return {"op": "literal", "value": value}


def inp(identity):
    return {"op": "input", "id": identity}


def ref(identity):
    return {"op": "ref", "id": identity}


def op(name, *args):
    return {"op": name, "args": list(args)}


def core(inputs, steps):
    return {"schema": ALGORITHM_VERSION, "profile": ALGORITHM_PROFILE,
            "inputs": inputs, "steps": steps}


class AlgorithmV7CodecTests(unittest.TestCase):
    def test_all_record_kinds_and_typed_scalar_edges_round_trip(self):
        add = op("add", inp("a0"), lit(["num", 2]))
        records = [
            ({"kind": "compute", "id": "v0", "term": add, "result": ["num", 3]}, "generated"),
            ({"kind": "request", "id": "r0", "boundary": "b0", "operation": "demo.readAsync",
              "arguments": [["str", "counter"], ["num", "-0"]]}, "generated"),
            ({"kind": "observe", "request": "r0", "id": "v0", "status": "fulfilled",
              "value": ["num", "NaN"]}, "observed"),
            ({"kind": "observe", "request": "r0", "id": "v0", "status": "completed"}, "observed"),
            ({"kind": "write", "object": ["ref", "object:1"], "key": "n", "value": ["undefined"],
              "present": False}, "generated"),
            ({"kind": "return", "value": ["bigint", "12345678901234567890"]}, "generated"),
            ({"kind": "throw", "value": ["str", "bad\nvalue"]}, "generated"),
            ({"kind": "suspend", "requests": ["r0", "r:1"], "live": ["v0"]}, "generated"),
        ]
        for record, role in records:
            with self.subTest(record=record):
                line = encode_algorithm_record(record, role)
                self.assertEqual(parse_algorithm_record(line, role), record)
                self.assertEqual(encode_algorithm_record(parse_algorithm_record(line, role), role), line)

        scalar_records = [
            ["undefined"], ["null"], ["bool", True], ["bool", False], ["str", "a\n\"\\é"],
            ["num", 0], ["num", 1e-6], ["num", 1e-7], ["num", 1e21],
            ["num", "-0"], ["num", "NaN"], ["num", "+Infinity"], ["num", "-Infinity"],
            ["bigint", "-12345678901234567890"], ["ref", "scope:object-1"],
        ]
        for value in scalar_records:
            line = encode_algorithm_record({"kind": "return", "value": value})
            self.assertEqual(parse_algorithm_record(line)["value"], value)

    def test_every_operation_uses_its_registered_arity(self):
        for name, (arity, _) in OPERATIONS.items():
            term = op(name, *[lit(["num", index]) for index in range(arity)])
            record = {"kind": "compute", "id": "v0", "term": term, "result": ["num", 0]}
            line = encode_algorithm_record(record)
            self.assertEqual(parse_algorithm_record(line), record, name)
            invalid = dict(record, term=op(name, *[lit(["num", 0]) for _ in range(arity + 1)]))
            with self.assertRaises(TypeError, msg=name):
                encode_algorithm_record(invalid)

    def test_parser_rejects_noncanonical_malformed_and_v6_records(self):
        invalid = [
            "return 1.0", "return 01", "return 1e0", "return 1E+21", "return -0n",
            "return 1e999", "return +1", "return true ", " return true", "return \"\\u0061\"",
            "v0=add($a0, 1) => 2", "v0 = add($a0,1) => 2", "v0 = add(1) => 2",
            "v0 = unknown(1) => 2", "v0 = fold_left($a0, 0, multiply) => 0",
            "observe r0 as v0: fulfilled", "observe r0 as v0: completed undefined",
            "request r0 b0: demo.store(\"counter\", 8) payload [\"counter\",8]",
            'evaluate run(1) => 2', 'compute (a + 1) => 2', '{"kind":"return","value":["num",1]}',
            "suspend [] live []", "suspend [r0, r0] live []", "return 1\nreturn 2",
        ]
        for line in invalid:
            with self.subTest(line=line), self.assertRaises(TypeError):
                parse_algorithm_record(line)
        with self.assertRaises(TypeError):
            parse_algorithm_record("observe r0 as v0: fulfilled 7", "generated")
        with self.assertRaises(TypeError):
            parse_algorithm_record("return 7", "observed")
        with self.assertRaises(TypeError):
            encode_algorithm_record({"kind": "return", "value": ["num", 1], "payload": []})
        with self.assertRaises(TypeError):
            encode_algorithm_record({"kind": "write", "object": ["ref", "o0"], "key": "__proto__",
                                     "value": ["null"], "present": True})

    def test_record_and_term_budgets_are_enforced(self):
        deep = lit(["num", 0])
        for _ in range(65):
            deep = {"op": "negate", "args": [deep]}
        with self.assertRaises(TypeError):
            validate_term(deep)
        with self.assertRaises(TypeError):
            parse_algorithm_record("return \"" + "x" * 65536 + "\"")
        with self.assertRaises(TypeError):
            encode_algorithm_record({"kind": "request", "id": "r0", "boundary": "b0",
                                     "operation": "demo.call", "arguments": [["num", 1]] * 129})


class AlgorithmV7CoreTests(unittest.TestCase):
    def test_core_validation_rejects_bad_bindings_authority_and_budgets(self):
        valid = core([{"id": "a0", "type": "number"}], [
            {"kind": "compute", "id": "v0", "term": op("add", inp("a0"), lit(["num", 1]))},
            {"kind": "return", "value": ref("v0")},
        ])
        self.assertEqual(validate_algorithm(valid), {"inputs": 1, "values": 1, "requests": 0, "steps": 2})
        self.assertEqual(algorithm_digest(valid), hashlib.sha256(stable_algorithm_json(valid).encode()).hexdigest())
        invalids = []
        invalids.append(dict(valid, extra=True))
        invalids.append(core([{"id": "a0", "type": "number"}], [
            {"kind": "compute", "id": "v0", "term": op("add", inp("future"), lit(["num", 1]))},
            {"kind": "return", "value": ref("v0")},
        ]))
        invalids.append(core([], [{"kind": "compute", "id": "x", "term": op("add", lit(["num", 1]))},
                                  {"kind": "return", "value": ref("x")}]))
        invalids.append(core([], [{"kind": "return", "value": lit(["num", 1])},
                                  {"kind": "throw", "value": lit(["num", 2])}]))
        invalids.append(core([], [{"kind": "suspend", "requests": ["r0"], "live": ["future"]}]))
        invalids.append(core([], [{"kind": "observe", "id": "v0", "request": "r0"},
                                  {"kind": "return", "value": ref("v0")}]))
        invalids.append(core([], [{"kind": "request", "id": "r0", "boundary": "b0", "operation": "x",
                                  "arguments": []},
                                 {"kind": "observe", "id": "v0", "request": "r0"},
                                 {"kind": "observe", "id": "v1", "request": "r0"},
                                 {"kind": "return", "value": lit(["num", 0])}]))
        for invalid in invalids:
            with self.subTest(core=invalid), self.assertRaises(TypeError):
                validate_algorithm(invalid)

    def test_primitive_binary64_string_bigint_and_lazy_semantics(self):
        run = lambda term: evaluate_term(term)
        self.assertEqual(run(op("add", lit(["str", "2"]), lit(["num", 3]))), ["str", "23"])
        self.assertEqual(run(op("add", lit(["bool", True]), lit(["str", "x"]))), ["str", "truex"])
        self.assertEqual(run(op("divide", lit(["num", 1]), lit(["num", 0]))), ["num", "+Infinity"])
        self.assertEqual(run(op("divide", lit(["num", 0]), lit(["num", 0]))), ["num", "NaN"])
        self.assertEqual(run(op("remainder", lit(["num", -5]), lit(["num", 2]))), ["num", -1.0])
        self.assertEqual(run(op("multiply", lit(["num", "-0"]), lit(["num", 2]))), ["num", "-0"])
        self.assertEqual(run(op("equal", lit(["num", "NaN"]), lit(["num", "NaN"]))), ["bool", False])
        self.assertEqual(run(op("loose_equal", lit(["num", 1]), lit(["str", "1"]))), ["bool", True])
        self.assertEqual(run(op("type_of", lit(["null"]))), ["str", "object"])
        self.assertEqual(run(op("divide", lit(["bigint", "-7"]), lit(["bigint", "3"]))),
                         ["bigint", "-2"])
        self.assertEqual(run(op("remainder", lit(["bigint", "-7"]), lit(["bigint", "3"]))),
                         ["bigint", "-1"])
        error_term = op("divide", lit(["bigint", "1"]), lit(["bigint", "0"]))
        with self.assertRaises(ZeroDivisionError):
            run(error_term)
        with self.assertRaises(TypeError):
            run(op("add", lit(["bigint", "1"]), lit(["num", 1])))
        self.assertEqual(run(op("select", lit(["bool", False]), error_term, lit(["num", 4]))),
                         ["num", 4])
        self.assertEqual(run(op("and", lit(["num", 0]), error_term)), ["num", 0])
        self.assertEqual(run(op("or", lit(["str", "ready"]), error_term)), ["str", "ready"])
        self.assertEqual(run(op("coalesce", lit(["num", 0]), error_term)), ["num", 0])
        self.assertEqual(run(op("coalesce", lit(["undefined"]), lit(["num", 9]))), ["num", 9])

    def test_ordered_fold_preserves_binary64_rounding_and_requires_dense_finite_arrays(self):
        fold = {"op": "fold_left", "input": inp("numbers"), "initial": lit(["num", 0]),
                "step": "add", "order": "ascending", "profile": "dense-number-array/1"}
        program = core([{"id": "numbers", "type": "dense-number-array"}],
                       [{"kind": "return", "value": fold}])

        def array(values):
            properties = [["length", ["num", len(values)]]]
            properties.extend([[str(i), value] for i, value in enumerate(values)])
            return {"objects": [{"id": "a0", "kind": "array", "properties": properties}], "roots": [["ref", "a0"]]}

        self.assertEqual(evaluate_algorithm(program, [["ref", "a0"]], heap=array([
            ["num", 1e16], ["num", 1], ["num", -1e16],
        ]))["completion"], {"kind": "return", "value": ["num", 0.0]})
        self.assertEqual(evaluate_algorithm(program, [["ref", "a0"]], heap=array([
            ["num", 1e16], ["num", -1e16], ["num", 1],
        ]))["completion"], {"kind": "return", "value": ["num", 1.0]})
        for properties in (
            [["length", ["num", 2]], ["0", ["num", 1]]],
            [["length", ["num", 1]], ["0", ["num", 1]], ["extra", ["num", 2]]],
            [["length", ["num", 1]], ["0", ["num", math.inf]]],
        ):
            bad_heap = {"a0": {"kind": "array", "properties": properties}}
            with self.assertRaises(TypeError):
                evaluate_algorithm(program, [["ref", "a0"]], heap=bad_heap)

    def test_projection_writes_aliases_and_cycles_are_kept_in_rooted_graphs(self):
        graph = {"roots": [["ref", "box"], ["ref", "box"]], "objects": [
            {"id": "box", "kind": "object", "properties": [
                {"key": "first", "value": ["ref", "child"], "version": 1},
                {"key": "again", "value": ["ref", "child"], "version": 1},
            ]},
            {"id": "child", "kind": "object", "properties": [["parent", ["ref", "box"]],
                                                                     ["n", ["num", 2]]]},
            {"id": "private", "kind": "object", "properties": []},
        ]}
        canonical = canonical_algorithm_graph(graph, input_ids=["box"])
        self.assertEqual(canonical["roots"], [["ref", "i0"], ["ref", "i0"]])
        self.assertEqual([row["id"] for row in canonical["objects"]], ["i0", "g0"])
        self.assertEqual(canonical["objects"][0]["properties"][0][1], ["ref", "g0"])
        self.assertEqual(canonical["objects"][0]["properties"][1][1], ["ref", "g0"])
        self.assertEqual(canonical["objects"][1]["properties"][0][1], ["ref", "i0"])
        heap = algorithm_heap_from_graph(graph)
        env = {"inputs": {"boxArg": ["ref", "box"]}, "values": {}, "heap": heap}
        projected = {"op": "project", "object": inp("boxArg"), "key": "first"}
        self.assertEqual(evaluate_term({"op": "project", "object": projected, "key": "n"}, env), ["num", 2])
        self.assertEqual(evaluate_term(op("equal", projected,
                                            {"op": "project", "object": inp("boxArg"), "key": "again"}), env),
                         ["bool", True])

        program = core([{"id": "boxArg", "type": "data"}], [
            {"kind": "write", "object": inp("boxArg"), "key": "n", "value": lit(["num", 9]),
             "present": True},
            {"kind": "return", "value": {"op": "project", "object": inp("boxArg"), "key": "n"}},
        ])
        result = evaluate_algorithm(program, [["ref", "box"]], heap=heap)
        self.assertEqual(result["completion"], {"kind": "return", "value": ["num", 9]})
        self.assertEqual(result["heap"]["box"]["properties"]["n"], ["num", 9])

    def test_requests_observations_and_suspension_keep_causal_ownership(self):
        program = core([{"id": "key", "type": "scalar"}], [
            {"kind": "request", "id": "r0", "boundary": "b0", "operation": "demo.readAsync",
             "arguments": [inp("key")]},
            {"kind": "observe", "id": "v0", "request": "r0"},
            {"kind": "compute", "id": "v1", "term": op("add", ref("v0"), lit(["num", 1]))},
            {"kind": "return", "value": ref("v1")},
        ])
        result = evaluate_algorithm(program, [["str", "counter"]],
                                    observations=[{"request": "r0", "status": "fulfilled",
                                                   "result": ["num", 7], "mode": "observed"}])
        self.assertEqual(result["completion"], {"kind": "return", "value": ["num", 8.0]})
        self.assertEqual(result["requests"][0]["arguments"], [["str", "counter"]])
        self.assertEqual(result["observations"][0]["status"], "fulfilled")
        suspended = evaluate_algorithm(program, [["str", "counter"]])
        self.assertEqual(suspended["completion"], {"kind": "suspend", "requests": ["r0"],
                                                   "live": [], "step": 1})
        rejected = evaluate_algorithm(program, [["str", "counter"]], observations=[
            {"request": "r0", "status": "rejected", "result": ["str", "denied"]},
        ])
        self.assertEqual(rejected["completion"], {"kind": "throw", "value": ["str", "denied"]})
        with self.assertRaises(TypeError):
            evaluate_algorithm(program, [["str", "counter"]], observations=[
                {"request": "r1", "status": "fulfilled", "result": ["num", 7]},
            ])
        reference_result = core([], [
            {"kind": "request", "id": "r0", "boundary": "b0", "operation": "demo.readAsync",
             "arguments": []},
            {"kind": "observe", "id": "v0", "request": "r0"},
            {"kind": "return", "value": ref("v0")},
        ])
        graph_heap = {"box": {"kind": "object", "properties": []}}
        observed_reference = evaluate_algorithm(reference_result, [], heap=graph_heap, observations=[
            {"request": "r0", "status": "fulfilled", "result": ["ref", "box"]},
        ])
        self.assertEqual(observed_reference["completion"], {"kind": "return", "value": ["ref", "box"]})



class StableAlgorithmJsonTests(unittest.TestCase):
    def test_js_number_spelling_utf16_key_order_and_hash(self):
        value = {"upper": 1e21, "tiny": 1e-7, "micro": 1e-6, "a": 1,
                 "\ue000": 2, "😀": 3, "max": 1.7976931348623157e308, "min": 5e-324}
        self.assertEqual(stable_algorithm_json(value),
                         '{"a":1,"max":1.7976931348623157e+308,"micro":0.000001,'
                         '"min":5e-324,"tiny":1e-7,"upper":1e+21,"😀":3,"\ue000":2}')
        self.assertEqual(stable_algorithm_hash({}), "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a")
        self.assertEqual(stable_algorithm_hash("a"), hashlib.sha256(b"a").hexdigest())

    def test_stable_json_refuses_cycles_nonfinite_and_signed_zero(self):
        cycle = []
        cycle.append(cycle)
        for value in (cycle, {"x": math.nan}, {"x": math.inf}, {"x": -0.0}, {1: "bad"}):
            with self.subTest(value=value), self.assertRaises(TypeError):
                stable_algorithm_json(value)


if __name__ == "__main__":
    unittest.main()
