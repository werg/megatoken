"""Independent validator for the generated, pinned host-operation contract.

The declaration bytes are shared with the JS producer. Materialization is
independent Python code and never calls a guest object, getter, or native service.
"""

from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal
from importlib.resources import files

from .witness_v4 import graph, value

CONTRACT_BYTES = files("megatoken.data").joinpath("contracts/host-operations-v4.json").read_bytes()
CONTRACT_DIGEST = hashlib.sha256(CONTRACT_BYTES).hexdigest()
_contract = json.loads(CONTRACT_BYTES)
if _contract.get("schema") != "host-operations/4":
    raise ValueError("Incompatible native capability catalog")
HOST_OPERATIONS = {item["operation"]: item for item in _contract["operations"]}
if len(HOST_OPERATIONS) != len(_contract["operations"]):
    raise ValueError("Repeated native capability descriptor")


def canonical_json(item):
    """JSON.stringify-compatible finite JSON data, with no object callbacks."""
    if item is None or type(item) in (bool, str):
        return json.dumps(item, ensure_ascii=False, separators=(",", ":"))
    if type(item) in (int, float):
        number = float(item)
        if not math.isfinite(number) or (type(item) is int and int(number) != item):
            raise ValueError("Noncanonical ECMAScript JSON number")
        if number == 0:
            return "0"
        text = repr(number).lower()
        if 1e-6 <= abs(number) < 1e21:
            text = format(Decimal(text), "f")
            return text.rstrip("0").rstrip(".") if "." in text else text
        if "e" in text:
            mantissa, exponent = text.split("e")
            exponent = int(exponent)
            return (
                mantissa.removesuffix(".0")
                + "e"
                + ("+" if exponent >= 0 else "-")
                + str(abs(exponent))
            )
        return text.removesuffix(".0")
    if isinstance(item, list):
        return "[" + ",".join(canonical_json(part) for part in item) + "]"
    if isinstance(item, dict) and all(isinstance(key, str) for key in item):
        integers = sorted(
            (
                key
                for key in item
                if key.isascii()
                and key.isdecimal()
                and str(int(key)) == key
                and int(key) < 4294967295
            ),
            key=int,
        )
        keys = integers + [key for key in item if key not in integers]
        return (
            "{"
            + ",".join(canonical_json(key) + ":" + canonical_json(item[key]) for key in keys)
            + "}"
        )
    raise ValueError("External payload contains non-data values")


def _scalar(tag):
    value(tag)
    if tag == ["null"]:
        return None
    if tag[0] in {"str", "bool"}:
        return tag[1]
    if tag[0] == "num" and type(tag[1]) in (int, float):
        return tag[1]
    raise ValueError("Non-JSON scalar requires the declared owned graph codec")


def _property_order(node):
    """OwnPropertyKeys order: array-index strings first, then insertion order."""
    indexed, ordinary = [], []
    for prop in node["properties"]:
        key = prop["key"]
        if key.isascii() and key.isdecimal() and str(int(key)) == key and int(key) < 4294967295:
            indexed.append(prop)
        else:
            ordinary.append(prop)
    return {**node, "properties": sorted(indexed, key=lambda prop: int(prop["key"])) + ordinary}


def materialize_request(fields: dict, *, heap: dict | None = None, require_promise=False):
    operation = HOST_OPERATIONS.get(fields.get("operation"))
    if operation is None:
        raise ValueError("Unregistered v4 effect capability")
    arguments = fields.get("arguments")
    if not isinstance(arguments, list) or len(arguments) != operation["arity"]:
        raise ValueError("Invalid v4 effect argument arity")
    key = operation["key"]
    if key is not None and (
        not isinstance(arguments[key], list)
        or len(arguments[key]) != 2
        or arguments[key][0] != "str"
    ):
        raise ValueError("Invalid external operation key")
    if require_promise and ("promise" in fields) != (operation["kind"] == "promise"):
        raise ValueError("Request must preserve the operation's declared Promise return")
    payload = fields.get("payload")
    if not isinstance(payload, str) or len(payload.encode()) > operation["limits"]["payloadBytes"]:
        raise ValueError("External payload exceeds its declared byte budget")
    supplied = json.loads(payload)
    if canonical_json(supplied) != payload:
        raise ValueError("External payload must preserve canonical JSON bytes")
    callbacks = set(operation["callbackIndices"])
    if operation["codec"] == "owned-graph/1":
        if (
            not isinstance(supplied, dict)
            or set(supplied) != {"schema", "roots", "objects"}
            or supplied["schema"] != "owned-graph/1"
            or not isinstance(supplied["roots"], list)
            or not isinstance(supplied["objects"], list)
            or len(supplied["roots"]) != operation["arity"]
            or len(supplied["objects"]) > operation["limits"]["graphNodes"]
        ):
            raise ValueError("External graph requires its explicit versioned envelope")
        properties = 0
        for node in supplied["objects"]:
            if not isinstance(node, dict) or not isinstance(node.get("properties"), list):
                raise ValueError("External graph requires explicit data properties")
            properties += len(node["properties"])
            if any(
                not isinstance(prop, dict) or "value" not in prop for prop in node["properties"]
            ):
                raise ValueError("External graph has a malformed data property")
        if properties > operation["limits"]["graphProperties"]:
            raise ValueError("External graph exceeds its total property budget")
        callback_ids = set()

        def normalize(tag, *, slot=False):
            if isinstance(tag, list) and tag and tag[0] == "callback":
                if (
                    not (slot or operation.get("callbackGraph"))
                    or len(tag) != 2
                    or not isinstance(tag[1], str)
                ):
                    raise ValueError("Callback marker lacks a declared owned callback slot")
                value(["ref", tag[1]])
                if heap is not None and (tag[1] not in heap or heap[tag[1]]["kind"] != "closure"):
                    raise ValueError("Graph callback is not a committed owned closure")
                callback_ids.add(tag[1])
                return ["ref", tag[1]]
            return tag

        normalized = {
            "roots": [
                normalize(tag, slot=index in callbacks)
                for index, tag in enumerate(supplied["roots"])
            ],
            "objects": [
                {
                    **node,
                    "properties": [
                        {**prop, "value": normalize(prop["value"])} for prop in node["properties"]
                    ],
                }
                for node in supplied["objects"]
            ],
        }
        graph(normalized, retained=callback_ids)
        if any(node["id"] in callback_ids for node in normalized["objects"]):
            raise ValueError("A native data object cannot impersonate an owned callback")
        if normalized["roots"] != arguments:
            raise ValueError("External graph roots differ from evaluated arguments")
        if heap is not None:
            reachable = set()

            def visit(tag):
                if tag[0] == "ref" and tag[1] not in reachable:
                    if tag[1] in callback_ids:
                        return
                    reachable.add(tag[1])
                    if tag[1] not in heap:
                        raise ValueError("Unknown owned graph reference")
                    for prop in heap[tag[1]]["properties"]:
                        visit(prop["value"])

            for argument in arguments:
                visit(argument)
            nodes = {node["id"]: node for node in normalized["objects"]}
            if set(nodes) != reachable or any(
                nodes[name] != _property_order(heap[name]) for name in reachable
            ):
                raise ValueError("External graph snapshot differs from current selected state")
        return supplied, operation["signature"]
    seen = set()

    def resolve(tag, depth=0):
        if depth > 64:
            raise ValueError("External JSON graph exceeds depth limit")
        if tag[0] != "ref":
            return _scalar(tag)
        name = tag[1]
        if heap is None or name not in heap or name in seen:
            raise ValueError("JSON references require an independent, nonaliased selected snapshot")
        seen.add(name)
        node = heap[name]
        properties = {prop["key"]: prop["value"] for prop in node["properties"]}
        if node["kind"] == "array":
            length = properties.pop("length", ["num", -1])[1]
            if (
                type(length) is not int
                or not 0 <= length <= 4096
                or set(properties) != {str(i) for i in range(length)}
            ):
                raise ValueError("JSON arrays require dense indices and no extra properties")
            return [resolve(properties[str(i)], depth + 1) for i in range(length)]
        if node["kind"] != "object" or "__proto__" in properties:
            raise ValueError("JSON transport only accepts ordinary owned data")
        return {key: resolve(tag, depth + 1) for key, tag in properties.items()}

    expected = []
    for index, tag in enumerate(arguments):
        if index in callbacks:
            if not isinstance(tag, list) or len(tag) != 2 or tag[0] != "ref":
                raise ValueError("Callback slot must reference an owned guest closure")
            if heap is not None and (tag[1] not in heap or heap[tag[1]]["kind"] != "closure"):
                raise ValueError("Callback reference is not a committed owned closure")
            expected.append({"$ownedCallback": tag[1]})
        else:
            expected.append(resolve(tag))
    if canonical_json(expected) != payload:
        raise ValueError("Grounded request payload differs from evaluated arguments")
    return expected, operation["signature"]
