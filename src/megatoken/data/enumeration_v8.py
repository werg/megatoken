"""Independent replay for bounded for-in cursors over ordinary prototype chains."""

from __future__ import annotations

import copy

FIELDS = {
    "enumeration.create": {"kind", "name", "source"},
    "enumeration.next": {"kind", "object", "name", "result", "done"},
}
BOUND = 4096
CHAIN_BOUND = 128
NODE_FIELDS = {"kind", "source", "chain", "keys", "nextIndex", "done", "properties"}


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid for-in enumeration protocol: " + message)


def validate_record(record):
    from .protocol_v8 import _identity, validate_protocol_value

    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None:
        return False
    if set(record) != fields:
        _fail("record fields")
    _identity(record["name"])
    if kind == "enumeration.create":
        validate_protocol_value(record["source"])
    else:
        _identity(record["object"])
        validate_protocol_value(record["result"])
        if record["result"][0] not in ("str", "undefined") or type(record["done"]) is not bool:
            _fail("next result")
    return True


def _ordinary_node(machine, identity):
    node = machine["heap"].get(identity)
    if type(node) is not dict or node.get("kind") not in ("object", "array"):
        _fail("ordinary source or prototype required")
    return node


def _primitive_keys(source):
    if source[0] != "str":
        return []
    length = len(source[1].encode("utf-16-le", "surrogatepass")) // 2
    if length > BOUND:
        _fail("string enumeration bound")
    return [str(index) for index in range(length)]


def _keys_from_chain(chain):
    visited = set()
    keys = []
    for row in chain:
        for key, enumerable in row["keys"]:
            if key in visited:
                continue
            visited.add(key)
            if len(visited) > BOUND:
                _fail("enumeration key bound")
            if enumerable:
                keys.append(key)
    return keys


def _snapshot(machine, source):
    from .protocol_v8 import _descriptor, _known, _ordered

    _known(machine, source)
    if source[0] != "ref":
        return [], _primitive_keys(source)
    identity = source[1]
    chain = []
    visited = set()
    while identity is not None:
        if identity in visited or len(chain) >= CHAIN_BOUND:
            _fail("prototype cycle or chain bound")
        visited.add(identity)
        node = _ordinary_node(machine, identity)
        rows = [
            [key, _descriptor(node, key)["enumerable"]]
            for key in _ordered(node)
            if isinstance(key, str)
        ]
        if len(rows) > BOUND:
            _fail("own key bound")
        prototype = node.get("prototype", "ordinary")
        if prototype is not None and prototype != "ordinary":
            if not isinstance(prototype, list) or len(prototype) != 2 or prototype[0] != "ref":
                _fail("prototype profile")
            _known(machine, prototype)
        chain.append({"id": identity, "prototype": copy.deepcopy(prototype), "keys": rows})
        identity = prototype[1] if isinstance(prototype, list) else None
    return chain, _keys_from_chain(chain)


def _validate_chain(machine, source, chain):
    from .protocol_v8 import _array_index, _identity

    if (
        not isinstance(chain, list)
        or len(chain) > CHAIN_BOUND
        or any(type(row) is not dict for row in chain)
    ):
        _fail("chain shape or bound")
    if source[0] != "ref":
        if chain:
            _fail("primitive prototype chain")
        return _primitive_keys(source)
    if not chain or chain[0].get("id") != source[1]:
        _fail("source chain identity")
    visited = set()
    for index, row in enumerate(chain):
        if type(row) is not dict or set(row) != {"id", "prototype", "keys"}:
            _fail("chain row fields")
        _identity(row["id"])
        if row["id"] in visited:
            _fail("duplicate chain identity")
        visited.add(row["id"])
        _ordinary_node(machine, row["id"])
        prototype = row["prototype"]
        if index + 1 < len(chain):
            if prototype != ["ref", chain[index + 1].get("id")]:
                _fail("chain prototype linkage")
        elif prototype is not None and prototype != "ordinary":
            _fail("unterminated prototype chain")
        rows = row["keys"]
        if not isinstance(rows, list) or len(rows) > BOUND:
            _fail("chain keys bound")
        own = set()
        last_index = -1
        ordinary_seen = False
        for pair in rows:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
                or type(pair[1]) is not bool
                or pair[0] in own
            ):
                _fail("chain key rows")
            key = pair[0]
            own.add(key)
            number = _array_index(key)
            if number is None:
                ordinary_seen = True
            elif ordinary_seen or number <= last_index:
                _fail("chain numeric key order")
            else:
                last_index = number
    return _keys_from_chain(chain)


def validate_state(machine):
    from .protocol_v8 import _known

    count = 0
    for node in machine["heap"].values():
        if node.get("kind") != "enumerator":
            continue
        count += 1
        if (
            set(node) != NODE_FIELDS
            or node["properties"] != {}
            or type(node["nextIndex"]) is not int
            or type(node["done"]) is not bool
            or not isinstance(node["keys"], list)
        ):
            _fail("cursor state fields")
        _known(machine, node["source"])
        expected = _validate_chain(machine, node["source"], node["chain"])
        if node["keys"] != expected:
            _fail("cursor keys disagree with captured chain")
        if not 0 <= node["nextIndex"] <= len(expected):
            _fail("cursor index")
        if node["done"] and node["nextIndex"] != len(expected):
            _fail("completed cursor index")
    return {"valid": True, "enumerators": count}


def apply_record(machine, record):
    from .iterator_v8 import _allocate
    from .protocol_v8 import _assert_fresh_heap_identity

    if not validate_record(record):
        return False
    if record["kind"] == "enumeration.create":
        _assert_fresh_heap_identity(machine, record["name"])
        chain, keys = _snapshot(machine, record["source"])
        machine["heap"][record["name"]] = {
            "kind": "enumerator",
            "source": copy.deepcopy(record["source"]),
            "chain": chain,
            "keys": keys,
            "nextIndex": 0,
            "done": False,
            "properties": {},
        }
    else:
        validate_state(machine)
        node = machine["heap"].get(record["object"])
        if type(node) is not dict or node.get("kind") != "enumerator":
            _fail("cursor identity")
        chain, keys = _snapshot(machine, node["source"])
        if chain != node["chain"] or keys != node["keys"]:
            _fail("source structure changed during enumeration")
        index = node["nextIndex"]
        done = node["done"] or index == len(keys)
        value = ["undefined"] if done else ["str", keys[index]]
        if record["done"] != done or record["result"] != value:
            _fail("false next result")
        if not done:
            node["nextIndex"] += 1
        node["done"] = done
        _allocate(
            machine,
            {
                "name": record["name"],
                "objectKind": "object",
                "length": None,
                "properties": [
                    {"key": ["str", "value"], "value": value},
                    {"key": ["str", "done"], "value": ["bool", done]},
                ],
            },
        )
    validate_state(machine)
    return True
