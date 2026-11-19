import unittest

from megatoken.data.algorithm_v8 import AlgorithmV8Error, evaluate_algorithm, replay_algorithm_line, validate_algorithm


def callback_term(mode="map"):
    return {
        "op": "traverse", "mode": mode, "input": {"op": "input", "id": "a0"},
        "initial": None,
        "callback": {"arity": 3, "body": {"op": "multiply", "args": [
            {"op": "bound", "depth": 0, "index": 0},
            {"op": "literal", "value": ["num", 2]},
        ]}},
        "profile": "dense-primitive-array/1", "order": "ascending",
    }


class AlgorithmV8Tests(unittest.TestCase):
    def setUp(self):
        self.heap = {"o2": {"kind": "array", "properties": [
            ["0", ["num", 2]], ["1", ["num", 3]], ["2", ["num", 4]], ["length", ["num", 3]],
        ]}}
        self.core = {"schema": "semantic-algorithm/2", "profile": "ecmascript-pure-regions/1",
                     "inputs": [{"id": "a0", "type": "dense-number-array"}],
                     "steps": [{"kind": "compute", "id": "v0", "term": callback_term()},
                               {"kind": "return", "value": {"op": "ref", "id": "v0"}}]}

    def test_map_preserves_input_identity_and_allocates_result(self):
        result = evaluate_algorithm(self.core, [["ref", "o2"]], heap=self.heap)
        self.assertEqual(result["completion"], {"kind": "return", "value": ["ref", "v8:0"]})
        self.assertEqual(result["heap"]["o2"]["properties"]["2"], ["num", 4])
        self.assertEqual(result["heap"]["v8:0"]["properties"]["2"], ["num", 8])

    def test_callback_bound_depth_and_index_are_checked(self):
        self.core["steps"][0]["term"]["callback"]["body"]["args"][0] = {"op": "bound", "depth": 0, "index": 3}
        with self.assertRaises(AlgorithmV8Error): validate_algorithm(self.core)

    def test_sparse_or_accessor_like_array_data_is_refused(self):
        del self.heap["o2"]["properties"][1]
        with self.assertRaises(ValueError): evaluate_algorithm(self.core, [["ref", "o2"]], heap=self.heap)

    def test_old_algorithm_version_is_not_relabelled(self):
        self.core["schema"] = "semantic-algorithm/1"
        with self.assertRaises(AlgorithmV8Error): validate_algorithm(self.core)

    def test_generated_claims_and_allocations_are_replayed(self):
        line = 'algorithm {"allocations":[{"id":"o0","kind":"array","properties":[{"key":"0","value":["num",4]},{"key":"1","value":["num",6]},{"key":"2","value":["num",8]},{"key":"length","value":["num",3]}]}],"claims":[{"id":"v0","kind":"compute","result":["ref","o0"],"step":0},{"kind":"return","step":1,"value":["ref","o0"]}],"completion":{"kind":"return","value":["ref","o0"]}}'
        graph = {"roots": [["ref", "o2"]], "objects": [{"id":"o2","kind":"array","properties":[
            {"key":"0","value":["num",2]},{"key":"1","value":["num",3]},{"key":"2","value":["num",4]},{"key":"length","value":["num",3]},
        ]}]}
        self.assertEqual(replay_algorithm_line(self.core, graph, line)["completion"]["value"], ["ref", "o0"])
        forged = line.replace('["num",8]', '["num",9]')
        with self.assertRaises(AlgorithmV8Error): replay_algorithm_line(self.core, graph, forged)


if __name__ == "__main__": unittest.main()
