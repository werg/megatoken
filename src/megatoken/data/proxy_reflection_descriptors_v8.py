"""Independent descriptor and own-key invariants for Proxy reflection."""

import copy
import math

from . import semantic_primitives as scalar
from .proxy_v8 import _eq, _truthy


def _p():
    from . import protocol_v8

    return protocol_v8


def _require(test, message):
    if not test:
        _p()._fail("Proxy reflection: " + message)


def _read(machine, reference, name):
    _p()._known(machine, reference)
    _require(reference[0] == "ref", "descriptor/list object identity")
    _, descriptor = _p()._lookup_descriptor(machine, reference[1], name, for_has=True)
    _require(
        descriptor is None or descriptor["kind"] == "data",
        "descriptor/list getter requires continuation",
    )
    return (descriptor is not None, descriptor["value"] if descriptor else ["undefined"])


def _display(machine, key):
    if key[0] == "str":
        return key[1]
    symbol = next((row for row in machine["protocol"]["symbols"] if row["id"] == key[1]), None)
    description = (
        "Symbol." + symbol["name"]
        if symbol and symbol["kind"] == "well-known"
        else (symbol or {}).get("description", ["str", ""])[1]
    )
    return "Symbol(" + description + ")"


def _descriptor(machine, value):
    fields = {}
    for name in ("enumerable", "configurable", "value", "writable", "get", "set"):
        present, selected = _read(machine, value, name)
        if not present:
            continue
        if name in ("get", "set") and selected[0] != "undefined":
            _require(
                selected[0] == "ref"
                and machine["heap"].get(selected[1], {}).get("kind") in ("closure", "intrinsic"),
                "descriptor invalid callable needs intrinsic fault",
            )
        fields[name] = (
            _truthy(selected) if name in ("enumerable", "configurable", "writable") else selected
        )
    accessor = "get" in fields or "set" in fields
    data = "value" in fields or "writable" in fields
    _require(not (accessor and data), "mixed descriptor needs intrinsic fault")
    result = {
        "kind": "accessor" if accessor else "data",
        "enumerable": fields.get("enumerable", False),
        "configurable": fields.get("configurable", False),
    }
    if accessor:
        result.update(get=fields.get("get", ["undefined"]), set=fields.get("set", ["undefined"]))
    else:
        result.update(
            value=fields.get("value", ["undefined"]), writable=fields.get("writable", False)
        )
    return result


def _compatible(current, descriptor, extensible):
    if current is None:
        return extensible
    if current["configurable"]:
        return True
    if (
        descriptor["configurable"]
        or descriptor["enumerable"] != current["enumerable"]
        or descriptor["kind"] != current["kind"]
    ):
        return False
    if current["kind"] == "accessor":
        return _eq(current["get"], descriptor["get"]) and _eq(current["set"], descriptor["set"])
    return (
        current["writable"]
        or not descriptor["writable"]
        and _eq(current["value"], descriptor["value"])
    )


def _descriptor_result(machine, row, value, target):
    key = row["arguments"][0]
    display, prefix = _display(machine, key), "'getOwnPropertyDescriptor' on proxy: "
    if value[0] not in ("ref", "undefined"):
        return {
            "fault": prefix
            + "trap returned neither object nor undefined for property '"
            + display
            + "'"
        }
    current = _p()._descriptor(target, _p()._key(key))
    if value[0] == "undefined":
        if current is None:
            return {"value": ["undefined"]}
        if not current["configurable"]:
            return {
                "fault": prefix
                + "trap returned undefined for property '"
                + display
                + "' which is non-configurable in the proxy target"
            }
        if not target.get("extensible", True):
            return {
                "fault": prefix
                + "trap returned undefined for property '"
                + display
                + "' which exists in the non-extensible proxy target"
            }
        return {"value": ["undefined"]}
    extensible = target.get("extensible", True)
    descriptor = _descriptor(machine, value)
    if not _compatible(current, descriptor, extensible):
        return {
            "fault": prefix
            + "trap returned descriptor for property '"
            + display
            + "' that is incompatible with the existing property in the proxy target"
        }
    if not descriptor["configurable"]:
        if current is None or current["configurable"]:
            return {
                "fault": prefix
                + "trap reported non-configurability for property '"
                + display
                + "' which is either non-existent or configurable in the proxy target"
            }
        if descriptor["kind"] == "data" and not descriptor["writable"] and current["writable"]:
            return {
                "fault": prefix
                + "trap reported non-configurable and non-writable for property '"
                + display
                + "' which is non-configurable, writable in the proxy target"
            }
    return {"descriptor": descriptor}


def _keys_result(machine, row, value, target):
    if value[0] != "ref":
        return {"fault": "CreateListFromArrayLike called on non-object"}
    _, length_value = _read(machine, value, "length")
    _require(
        length_value[0] not in ("ref", "symbol", "bigint"),
        "ownKeys length conversion requires continuation",
    )
    number = scalar._to_number(length_value)
    length = (
        0
        if math.isnan(number) or number <= 0
        else min(math.floor(number), 9007199254740991)
        if math.isfinite(number)
        else 9007199254740991
    )
    _require(length <= 4096, "ownKeys list bound")
    keys = []
    for index in range(length):
        _, key = _read(machine, value, str(index))
        if key[0] not in ("str", "symbol"):
            _require(
                key[0] != "ref", "ownKeys invalid object key diagnostic requires fault profile"
            )
            return {"fault": scalar._to_string(key) + " is not a valid property name"}
        keys.append(key)
    if any(any(_eq(key, other) for other in keys[:index]) for index, key in enumerate(keys)):
        return {"fault": "'ownKeys' on proxy: trap returned duplicate entries"}
    extensible = target.get("extensible", True)
    target_keys = [_p()._public(key) for key in _p()._ordered(target)]
    fixed, configurable = [], []
    for key in target_keys:
        descriptor = _p()._descriptor(target, _p()._key(key))
        (fixed if descriptor and not descriptor["configurable"] else configurable).append(key)
    unchecked = copy.deepcopy(keys)
    for key in fixed + ([] if extensible else configurable):
        index = next((index for index, other in enumerate(unchecked) if _eq(key, other)), None)
        if index is None:
            return {
                "fault": "'ownKeys' on proxy: trap result did not include '"
                + _display(machine, key)
                + "'"
            }
        unchecked.pop(index)
    if not extensible and unchecked:
        return {
            "fault": "'ownKeys' on proxy: trap returned extra keys but proxy target is non-extensible"
        }
    return {"keys": keys}


def describe_result(machine, row, value=None, direct=False):
    target = machine["heap"].get(row["target"][1])
    _require(target and target["kind"] == "object", "ordinary target internal method")
    if row["operation"] == "getOwnPropertyDescriptor":
        return (
            {"descriptor": _p()._descriptor(target, _p()._key(row["arguments"][0]))}
            if direct
            else _descriptor_result(machine, row, value, target)
        )
    return (
        {"keys": [_p()._public(key) for key in _p()._ordered(target)]}
        if direct
        else _keys_result(machine, row, value, target)
    )


def materialize_result(machine, row, value, result, direct=False):
    computed = describe_result(machine, row, value, direct)
    _require(not computed.get("fault"), "derived reflection fault required")
    if row["operation"] == "getOwnPropertyDescriptor" and not computed.get("descriptor"):
        _require(result == ["undefined"], "false absent descriptor")
        return ["undefined"]
    _require(result[0] == "ref", "fresh descriptor/list identity")
    array = row["operation"] == "ownKeys"
    if array:
        fields = [(str(index), value) for index, value in enumerate(computed["keys"])]
    else:
        descriptor = computed["descriptor"]
        fields = [
            (name, descriptor[name])
            for name in (("get", "set") if descriptor["kind"] == "accessor" else ("value",))
        ]
        if descriptor["kind"] == "data":
            fields.append(("writable", ["bool", descriptor["writable"]]))
        fields.extend((name, ["bool", descriptor[name]]) for name in ("enumerable", "configurable"))
    record = {
        "kind": "allocate",
        "name": result[1],
        "objectKind": "array" if array else "object",
        "length": len(computed["keys"]) if array else None,
        "properties": [{"key": ["str", key], "value": value} for key, value in fields],
        "code": None,
        "captures": [],
        "functionKind": None,
        "lexicalReceiver": None,
    }
    prior = machine.get("_internalProtocolTransition")
    machine["_internalProtocolTransition"] = True
    try:
        _p()._apply(machine, record)
    finally:
        if prior is None:
            machine.pop("_internalProtocolTransition", None)
        else:
            machine["_internalProtocolTransition"] = prior
    return copy.deepcopy(result)
