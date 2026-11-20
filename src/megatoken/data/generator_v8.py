"""Independent bounded replay for synchronous generator protocol records."""

from __future__ import annotations

import copy

PROFILE = "synchronous-generator/1"
INTRINSICS = {"GeneratorFunction.prototype", "Generator.prototype", "Iterator.prototype"}
METHODS = {"next", "return", "throw"}
BUILTINS = {"Generator.next", "Generator.return", "Generator.throw", "Iterator.identity"}
BOUND = 4096
FIELDS = {
    "generator.function": {"kind", "object", "prototype", "name", "length"},
    "generator.create": {"kind", "name", "child", "callee", "receiver", "arguments"},
    "generator.start": {"kind", "object"},
    "generator.resume": {"kind", "object", "method", "argument"},
    "generator.receive": {"kind", "object", "bind"},
    "generator.handler.enter": {"kind", "object", "name", "catch", "finally"},
    "generator.handler.route": {"kind", "object", "completion"},
    "generator.handler.catch": {"kind", "object", "name", "bind"},
    "generator.handler.leave": {"kind", "object", "name"},
    "generator.delegate.enter": {"kind", "object", "iterator", "next"},
    "generator.delegate.call": {"kind", "object", "method", "call"},
    "generator.delegate.result": {"kind", "object", "result", "done"},
    "generator.delegate.yield": {"kind", "object", "result"},
    "generator.delegate.receive": {"kind", "object"},
    "generator.delegate.end": {"kind", "object"},
    "generator.delegate.absent": {"kind", "object", "method"},
    "generator.delegate.abrupt": {"kind", "object", "value"},
    "generator.yield": {"kind", "object", "value", "name"},
    "generator.finish": {"kind", "object", "completion", "name"},
    "generator.completed": {"kind", "object", "method", "argument", "name"},
    "generator.lookup": {"kind", "receiver", "key", "method"},
    "generator.read": {"kind", "object", "key", "result"},
    "generator.identity": {"kind", "object", "result"},
}


def _protocol():
    from . import protocol_v8

    return protocol_v8


def _fail(message):
    _protocol()._fail("Generator: " + message)


def _id(value, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > 256:
        _fail("identity")


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None:
        return False
    p = _protocol()
    p._exact(record, fields, "generator record")
    for name in ("object", "child", "name", "prototype", "bind"):
        if name in record:
            _id(record[name], nullable=True)
    for name in ("callee", "receiver", "argument", "result", "value", "iterator"):
        if name in record:
            p.validate_protocol_value(record[name])
    if "key" in record:
        p.validate_protocol_key(record["key"])
    if kind == "generator.function":
        if (
            not isinstance(record["name"], str)
            or len(record["name"]) > BOUND
            or type(record["length"]) is not int
            or not 0 <= record["length"] <= BOUND
        ):
            _fail("function metadata")
    if kind == "generator.create":
        if (
            record["callee"][0] != "ref"
            or type(record["arguments"]) is not list
            or len(record["arguments"]) > BOUND
        ):
            _fail("generator call arguments")
        for value in record["arguments"]:
            p.validate_protocol_value(value)
    if kind in ("generator.resume", "generator.completed") and record["method"] not in METHODS:
        _fail("resume method")
    if kind == "generator.delegate.enter":
        if (
            record["next"] not in ("Iterator.next", "Generator.next")
            or record["iterator"][0] != "ref"
        ):
            _fail("delegate iterator or cached next")
    if (
        kind in ("generator.delegate.call", "generator.delegate.absent")
        and record["method"] not in METHODS
    ):
        _fail("delegated method")
    if kind == "generator.delegate.result" and type(record["done"]) is not bool:
        _fail("delegated done flag")
    if kind == "generator.delegate.call":
        call = record["call"]
        if call.get("kind") == "iterator.next":
            from .iterator_v8 import validate_record as validate_iterator_record

            validate_iterator_record(call)
        elif call.get("kind") in ("generator.resume", "generator.completed"):
            if not validate_record(call):
                _fail("nested generator operation")
        else:
            _fail("delegated call operation")
    if kind == "generator.lookup" and record["method"] not in BUILTINS:
        _fail("builtin selection")
    if kind in ("generator.finish", "generator.handler.route"):
        completion = record["completion"]
        p._exact(completion, {"kind", "value"}, "generator completion")
        allowed = ("return", "throw", "normal") if kind.endswith("route") else ("return", "throw")
        if (
            completion["kind"] not in allowed
            or completion["kind"] == "normal"
            and completion["value"] != ["undefined"]
        ):
            _fail("completion kind")
        p.validate_protocol_value(completion["value"])
    if kind == "generator.handler.enter" and (
        type(record["catch"]) is not bool
        or type(record["finally"]) is not bool
        or not (record["catch"] or record["finally"])
    ):
        _fail("handler requires catch or finally")
    return True


def _value(machine, value):
    return _protocol()._known(machine, value)


def _node(machine, identity):
    node = machine["heap"].get(identity)
    if not node or node.get("kind") != "generator":
        _fail("generator identity")
    return node


def _active(machine, node):
    if not machine.get("stack") or machine["stack"][-1] != node["invocation"]:
        _fail("generator body is not active")


def _completion(machine, record, kind=None):
    if record.get("kind") not in ((kind,) if kind else ("return", "throw")):
        _fail("completion kind")
    _value(machine, record["value"])


def _api_call(machine, call):
    _protocol()._apply(machine, call)


def _allocate(machine, name, properties):
    _api_call(
        machine,
        {
            "kind": "allocate",
            "name": name,
            "objectKind": "object",
            "length": None,
            "properties": [{"key": ["str", key], "value": value} for key, value in properties],
            "code": None,
            "captures": [],
            "functionKind": None,
            "lexicalReceiver": None,
        },
    )


def _result(machine, name, value, done):
    _id(name)
    _allocate(machine, name, [("value", value), ("done", ["bool", done])])
    return ["ref", name]


def _descriptor(node, key):
    return _protocol()._descriptor(node, _protocol()._key(key))


def _lookup(machine, receiver, key):
    _value(machine, receiver)
    if receiver[0] != "ref":
        _fail("receiver must be owned object")
    current = receiver[1]
    seen = set()
    while True:
        if current in seen:
            _fail("prototype cycle")
        seen.add(current)
        node = machine["heap"].get(current)
        if not node or node.get("kind") not in ("object", "array", "closure", "generator"):
            _fail("ordinary prototype object")
        descriptor = _descriptor(node, key)
        if descriptor is not None:
            return {"descriptor": descriptor}
        prototype = node.get("prototype", "ordinary")
        if isinstance(prototype, str) and prototype in INTRINSICS:
            if prototype == "Generator.prototype":
                if key == ["symbol", "well-known:iterator"]:
                    return {"builtin": "Iterator.identity"}
                if key[0] == "str" and key[1] in METHODS:
                    return {"builtin": "Generator." + key[1]}
                if key[0] == "symbol" and key[1] == "well-known:toStringTag":
                    return {
                        "descriptor": {
                            "kind": "data",
                            "value": ["str", "Generator"],
                            "writable": False,
                            "enumerable": False,
                            "configurable": True,
                        }
                    }
                if key[0] == "str" and key[1] == "constructor":
                    return {"unsupported": True}
            if prototype == "GeneratorFunction.prototype" and key == [
                "symbol",
                "well-known:toStringTag",
            ]:
                return {
                    "descriptor": {
                        "kind": "data",
                        "value": ["str", "GeneratorFunction"],
                        "writable": False,
                        "enumerable": False,
                        "configurable": True,
                    }
                }
            if (
                prototype == "GeneratorFunction.prototype"
                and key[0] == "str"
                and key[1]
                in (
                    "prototype",
                    "constructor",
                    "call",
                    "apply",
                    "bind",
                    "caller",
                    "arguments",
                    "toString",
                    "valueOf",
                    "__proto__",
                )
            ):
                return {"unsupported": True}
            if (
                key[0] == "symbol"
                and key[1] == "well-known:iterator"
                and prototype == "Iterator.prototype"
            ):
                return {"builtin": "Iterator.identity"}
            if (
                prototype == "Iterator.prototype"
                and key[0] == "str"
                and key[1]
                in (
                    "constructor",
                    "__proto__",
                    "toString",
                    "toLocaleString",
                    "valueOf",
                    "hasOwnProperty",
                    "isPrototypeOf",
                    "propertyIsEnumerable",
                )
            ):
                return {"unsupported": True}
            if key[0] == "str" and key[1] in (
                "constructor",
                "__proto__",
                "toString",
                "toLocaleString",
                "valueOf",
                "hasOwnProperty",
                "isPrototypeOf",
                "propertyIsEnumerable",
                "__defineGetter__",
                "__defineSetter__",
                "__lookupGetter__",
                "__lookupSetter__",
            ):
                return {"unsupported": True}
            return {"descriptor": None}
        if prototype in (None, "ordinary"):
            return {"descriptor": None}
        _value(machine, prototype)
        if prototype[0] != "ref":
            _fail("prototype identity")
        current = prototype[1]


def _set_local(machine, name, value):
    if name is None:
        return
    aliases = machine.get("captureAliases", {})
    if isinstance(aliases, list):
        aliases = dict(aliases)
    machine["locals"][aliases.get(name, name)] = copy.deepcopy(value)


def _resume_call(machine, node, method, argument):
    invocation = node["invocation"]
    if invocation in machine["stack"] or invocation not in machine.get("frameFunctions", {}):
        _fail("retained generator invocation")
    if not machine.get("stack"):
        _fail("resume requires caller frame")
    closure = machine["heap"][node["callee"][1]]
    if any(capture["cell"] not in machine["locals"] for capture in closure.get("captures", [])):
        _fail("generator capture is not initialized")
    proto = machine["protocol"]
    proto["callFrames"].append(
        {
            "child": invocation,
            "parent": machine["stack"][-1],
            "priorReceiver": copy.deepcopy(machine.get("receiver", ["undefined"])),
            "priorAliases": [
                [key, value] for key, value in machine.get("captureAliases", {}).items()
            ],
            "arguments": copy.deepcopy(node["arguments"]),
        }
    )
    machine["receiver"] = copy.deepcopy(
        closure.get("lexicalReceiver")
        if closure.get("functionKind") == "arrow"
        else node["receiver"]
    )
    machine["captureAliases"] = {row["binding"]: row["cell"] for row in closure.get("captures", [])}
    machine["locals"][invocation + ":this"] = copy.deepcopy(machine["receiver"])
    for index, value in enumerate(node["arguments"]):
        machine["locals"][invocation + ":argument:" + str(index)] = copy.deepcopy(value)
    machine["stack"].append(invocation)
    node["resume"] = {
        "kind": method,
        "from": node["phase"],
        "value": ["undefined"]
        if method == "next" and node["phase"] == "suspendedStart"
        else copy.deepcopy(argument),
    }
    node["phase"] = "executing"


def _complete_frame(machine, invocation, kind, value):
    if not machine.get("stack") or machine["stack"][-1] != invocation:
        _fail("completion invocation")
    if len(machine["stack"]) > 1:
        frame = machine["protocol"]["callFrames"].pop()
        if frame["child"] != invocation:
            _fail("call frame completion")
        machine["stack"].pop()
        machine["receiver"] = frame["priorReceiver"]
        machine["captureAliases"] = dict(frame["priorAliases"])
        machine["lastReturn"] = {"kind": kind, "value": copy.deepcopy(value)}
    else:
        machine["stack"].pop()
        machine["pending"] = {"kind": kind, "value": copy.deepcopy(value)}


def _route(machine, node, completion):
    value = completion["value"]
    _value(machine, value)
    if completion["kind"] == "normal":
        if not node["handlers"]:
            _fail("normal handler completion requires a handler")
        handler = node["handlers"][-1]
        if handler["phase"] == "finally" or handler["caught"] is not None:
            _fail("normal handler completion")
        if handler["finally"]:
            handler["phase"], handler["pending"] = "finally", copy.deepcopy(completion)
        else:
            node["handlers"].pop()
        return
    while node["handlers"]:
        handler = node["handlers"][-1]
        if completion["kind"] == "throw" and handler["phase"] == "try" and handler["catch"]:
            handler["phase"], handler["caught"] = "catch", copy.deepcopy(value)
            return
        if handler["phase"] != "finally" and handler["finally"]:
            handler["phase"], handler["caught"], handler["pending"] = (
                "finally",
                None,
                copy.deepcopy(completion),
            )
            return
        node["handlers"].pop()
    node["pending"] = copy.deepcopy(completion)


def _delegate_method(machine, iterator, method):
    _value(machine, iterator)
    target = machine["heap"].get(iterator[1]) if iterator[0] == "ref" else None
    if target and target.get("kind") == "iterator":
        return "Iterator.next" if method == "next" else None
    if not target or target.get("kind") != "generator":
        _fail("delegate requires an owned iterator or generator")
    selected = _lookup(machine, iterator, ["str", method])
    if selected.get("builtin"):
        return selected["builtin"]
    descriptor = selected.get("descriptor")
    if (
        selected.get("unsupported")
        or descriptor
        and (
            descriptor.get("kind") != "data"
            or descriptor.get("value", [None])[0] not in ("undefined", "null")
        )
    ):
        _fail("custom delegate methods require their callable continuation")
    return None


def _delegated_result(machine, value):
    _value(machine, value)
    result = machine["heap"].get(value[1]) if value[0] == "ref" else None
    if not result or result.get("kind") != "object":
        _fail("delegated result object")
    descriptor = _descriptor(result, ["str", "done"])
    if not descriptor or descriptor["kind"] != "data" or descriptor["value"][0] != "bool":
        _fail("delegated done property must be a data boolean")
    return result, descriptor["value"][1]


def _apply_delegation(machine, node, record):
    _active(machine, node)
    if node["phase"] != "executing" or node["pending"] is not None:
        _fail("delegation is not executing")
    kind = record["kind"]
    if kind == "generator.delegate.enter":
        if (
            node["delegation"] is not None
            or node["resume"] is not None
            or record["iterator"][1] == record["object"]
        ):
            _fail("existing delegation or unconsumed resume")
        if _delegate_method(machine, record["iterator"], "next") != record["next"]:
            _fail("false cached delegate method")
        node["delegation"] = {
            "iterator": copy.deepcopy(record["iterator"]),
            "next": record["next"],
            "request": {"kind": "next", "value": ["undefined"]},
            "phase": "ready",
            "result": None,
        }
        return
    delegate = node.get("delegation")
    if delegate is None:
        _fail("missing delegated iterator")
    if kind == "generator.delegate.receive":
        if (
            delegate["phase"] != "yielded"
            or not node["resume"]
            or node["resume"]["from"] != "suspendedYield"
        ):
            _fail("missing delegated resume input")
        delegate["request"] = {
            "kind": node["resume"]["kind"],
            "value": copy.deepcopy(node["resume"]["value"]),
        }
        node["resume"], delegate["phase"], delegate["result"] = None, "ready", None
        return
    if node["resume"] is not None:
        _fail("delegation input must be consumed first")
    if kind == "generator.delegate.call":
        call = record["call"]
        if (
            delegate["phase"] != "ready"
            or record["method"] != delegate["request"]["kind"]
            or call.get("object") != delegate["iterator"][1]
        ):
            _fail("delegate call changed request or receiver")
        selected = (
            delegate["next"]
            if record["method"] == "next"
            else _delegate_method(machine, delegate["iterator"], record["method"])
        )
        delegate["phase"] = "calling"
        if selected == "Iterator.next":
            if record["method"] != "next" or call["kind"] != "iterator.next":
                _fail("cached iterator next call kind")
            from .iterator_v8 import apply_record as apply_iterator_record

            machine["_internalProtocolTransition"] = True
            try:
                apply_iterator_record(machine, call)
            finally:
                machine.pop("_internalProtocolTransition", None)
            delegate["result"] = ["ref", call["name"]]
        else:
            if (
                selected != "Generator." + record["method"]
                or call["kind"] not in ("generator.resume", "generator.completed")
                or call["method"] != record["method"]
                or call["argument"] != delegate["request"]["value"]
            ):
                _fail("selected generator method or sent value changed")
            machine["_internalProtocolTransition"] = True
            try:
                apply_record(machine, call)
            finally:
                machine.pop("_internalProtocolTransition", None)
        return
    if kind == "generator.delegate.result":
        if delegate["phase"] != "calling":
            _fail("no pending delegate call")
        returned = machine.get("lastReturn")
        expected = delegate["result"] or (
            returned["value"] if returned and returned["kind"] == "return" else None
        )
        if expected is None or expected != record["result"]:
            _fail("delegated result identity changed")
        _, done = _delegated_result(machine, record["result"])
        if done != record["done"]:
            _fail("delegated completion claim")
        delegate["result"], delegate["phase"] = (
            copy.deepcopy(record["result"]),
            "complete" if done else "result",
        )
        return
    if kind == "generator.delegate.yield":
        if (
            delegate["phase"] != "result"
            or delegate["result"] != record["result"]
            or _delegated_result(machine, record["result"])[1]
        ):
            _fail("delegated yield result")
        if node["yielded"] >= BOUND:
            _fail("yield bound")
        node["phase"], node["yielded"], delegate["phase"] = (
            "suspendedYield",
            node["yielded"] + 1,
            "yielded",
        )
        _complete_frame(machine, node["invocation"], "return", record["result"])
        return
    if kind == "generator.delegate.end":
        if delegate["phase"] != "complete":
            _fail("delegate is not complete")
        result, done = _delegated_result(machine, delegate["result"])
        if not done:
            _fail("delegate completion changed")
        value = _descriptor(result, ["str", "value"])
        if value and value["kind"] != "data":
            _fail("delegated value getter requires scoped call")
        node["delegation"] = None
        if delegate["request"]["kind"] == "return":
            _route(
                machine,
                node,
                {"kind": "return", "value": value["value"] if value else ["undefined"]},
            )
        return
    if kind == "generator.delegate.absent":
        if (
            delegate["phase"] != "ready"
            or record["method"] != "return"
            or delegate["request"]["kind"] != "return"
            or _delegate_method(machine, delegate["iterator"], "return") is not None
        ):
            _fail("delegate return method is not absent")
        node["delegation"] = None
        _route(machine, node, {"kind": "return", "value": delegate["request"]["value"]})
        return
    if kind == "generator.delegate.abrupt":
        returned = machine.get("lastReturn")
        target = machine["heap"].get(delegate["iterator"][1])
        if (
            delegate["phase"] != "calling"
            or delegate["result"] is not None
            or not target
            or target.get("kind") != "generator"
            or not returned
            or returned["kind"] != "throw"
            or returned["value"] != record["value"]
        ):
            _fail("delegated abrupt completion changed")
        node["delegation"] = None
        _route(machine, node, {"kind": "throw", "value": record["value"]})


def apply_record(machine, record):
    if not validate_record(record):
        return False
    p = _protocol()
    kind = record["kind"]
    if kind == "generator.function":
        fn = machine["heap"].get(record["object"])
        if (
            not fn
            or fn.get("kind") != "closure"
            or fn.get("functionKind") != "ordinary"
            or fn.get("generatorFunction")
            or fn.get("callable")
        ):
            _fail("generator declaration requires ordinary closure")
        if fn.get("properties") or fn.get("descriptors"):
            _fail("generator closure already has descriptors")
        _id(record["prototype"])
        _allocate(machine, record["prototype"], [])
        proto = machine["heap"][record["prototype"]]
        proto["prototype"] = "Generator.prototype"
        fn.setdefault("properties", {})
        fn.setdefault("descriptors", {})
        fn.setdefault("keyOrder", [])
        fn.setdefault("extensible", True)
        fn["generatorFunction"], fn["prototype"] = True, "GeneratorFunction.prototype"
        p._install(
            fn,
            "length",
            {
                "kind": "data",
                "value": ["num", record["length"]],
                "writable": False,
                "enumerable": False,
                "configurable": True,
            },
        )
        p._install(
            fn,
            "name",
            {
                "kind": "data",
                "value": ["str", record["name"]],
                "writable": False,
                "enumerable": False,
                "configurable": True,
            },
        )
        p._install(
            fn,
            "prototype",
            {
                "kind": "data",
                "value": ["ref", record["prototype"]],
                "writable": True,
                "enumerable": False,
                "configurable": False,
            },
        )
        return True
    if kind == "generator.create":
        callee = _value(machine, record["callee"])
        fn = machine["heap"].get(callee[1]) if callee[0] == "ref" else None
        if not fn or fn.get("kind") != "closure" or fn.get("generatorFunction") is not True:
            _fail("declared generator function required")
        p._assert_fresh_heap_identity(machine, record["name"])
        machine["_internalProtocolTransition"] = True
        try:
            _api_call(
                machine,
                {
                    "kind": "call",
                    "child": record["child"],
                    "callee": record["callee"],
                    "receiver": record["receiver"],
                    "arguments": record["arguments"],
                },
            )
        finally:
            machine.pop("_internalProtocolTransition", None)
        prototype = _descriptor(fn, ["str", "prototype"])
        if not prototype or prototype["kind"] != "data":
            _fail("generator prototype descriptor")
        value = prototype["value"]
        proto = machine["heap"].get(value[1]) if value[0] == "ref" else None
        machine["heap"][record["name"]] = {
            "kind": "generator",
            "callee": copy.deepcopy(record["callee"]),
            "invocation": record["child"],
            "receiver": copy.deepcopy(record["receiver"]),
            "arguments": copy.deepcopy(record["arguments"]),
            "phase": "initializing",
            "resume": None,
            "pending": None,
            "yielded": 0,
            "handlers": [],
            "delegation": None,
            "prototype": value if proto else "Generator.prototype",
            "properties": {},
        }
        return True
    if kind == "generator.lookup":
        selected = _lookup(machine, record["receiver"], record["key"])
        if selected.get("builtin") != record["method"]:
            _fail("false generator builtin selection")
        return True
    node = _node(machine, record["object"])
    if kind == "generator.read":
        selected = _lookup(machine, ["ref", record["object"]], record["key"])
        descriptor = selected.get("descriptor")
        if (
            selected.get("builtin")
            or selected.get("unsupported")
            or descriptor
            and descriptor["kind"] != "data"
            and descriptor["get"] != ["undefined"]
        ):
            _fail("generator read requires callable profile")
        actual = (
            descriptor["value"] if descriptor and descriptor["kind"] == "data" else ["undefined"]
        )
        if _value(machine, record["result"]) != actual:
            _fail("false generator property read")
    elif kind == "generator.identity":
        if _value(machine, record["result"]) != ["ref", record["object"]]:
            _fail("generator iterator identity")
    elif kind == "generator.start":
        _active(machine, node)
        if node["phase"] != "initializing" or node["resume"] is not None:
            _fail("generator start state")
        node["phase"] = "suspendedStart"
        _complete_frame(machine, node["invocation"], "return", ["ref", record["object"]])
    elif kind == "generator.resume":
        if node["phase"] not in ("suspendedStart", "suspendedYield"):
            _fail("generator is executing or completed")
        _value(machine, record["argument"])
        _resume_call(machine, node, record["method"], record["argument"])
    elif kind == "generator.receive":
        _active(machine, node)
        resume = node.get("resume")
        if (
            node["phase"] != "executing"
            or not resume
            or resume["kind"] != "next"
            or record["bind"] is not None
            and resume["from"] != "suspendedYield"
        ):
            _fail("generator next input")
        if record["bind"] is not None:
            _set_local(machine, record["bind"], resume["value"])
        node["resume"] = None
    elif kind == "generator.handler.enter":
        _active(machine, node)
        if (
            node["phase"] != "executing"
            or node["resume"] is not None
            or node["pending"] is not None
            or len(node["handlers"]) >= 64
            or any(h["name"] == record["name"] for h in node["handlers"])
        ):
            _fail("generator handler entry")
        node["handlers"].append(
            {
                "name": record["name"],
                "catch": record["catch"],
                "finally": record["finally"],
                "phase": "try",
                "caught": None,
                "pending": None,
            }
        )
    elif kind == "generator.handler.route":
        _active(machine, node)
        if node["phase"] != "executing" or node["pending"] is not None:
            _fail("completion routing is not active")
        completion = record["completion"]
        if node["resume"] is not None:
            if node["resume"]["kind"] == "next" or completion != {
                "kind": node["resume"]["kind"],
                "value": node["resume"]["value"],
            }:
                _fail("abrupt resume completion changed")
            node["resume"] = None
        _route(machine, node, completion)
    elif kind == "generator.handler.catch":
        _active(machine, node)
        handler = node["handlers"][-1] if node["handlers"] else None
        if (
            not handler
            or handler["name"] != record["name"]
            or handler["phase"] != "catch"
            or handler["caught"] is None
        ):
            _fail("generator catch state")
        _set_local(machine, record["bind"], handler["caught"])
        handler["caught"] = None
    elif kind == "generator.handler.leave":
        _active(machine, node)
        handler = node["handlers"][-1] if node["handlers"] else None
        if (
            not handler
            or handler["name"] != record["name"]
            or handler["phase"] != "finally"
            or handler["pending"] is None
            or node["pending"] is not None
        ):
            _fail("generator cleanup state")
        node["handlers"].pop()
        if handler["pending"]["kind"] != "normal":
            _route(machine, node, handler["pending"])
    elif kind.startswith("generator.delegate."):
        _apply_delegation(machine, node, record)
    elif kind == "generator.yield":
        _active(machine, node)
        if (
            node["phase"] != "executing"
            or node["resume"] is not None
            or node["pending"] is not None
            or node["delegation"] is not None
        ):
            _fail("generator suspension state")
        if node["yielded"] >= BOUND:
            _fail("generator yield bound")
        value = _value(machine, record["value"])
        node["phase"] = "suspendedYield"
        node["yielded"] += 1
        _complete_frame(
            machine, node["invocation"], "return", _result(machine, record["name"], value, False)
        )
    elif kind == "generator.finish":
        _active(machine, node)
        completion = record["completion"]
        _completion(machine, completion)
        if (
            node["phase"] not in ("initializing", "executing")
            or node["handlers"]
            or node["delegation"] is not None
            or node["pending"] is not None
            and node["pending"] != completion
        ):
            _fail("generator completion state")
        if node["phase"] == "initializing" and completion["kind"] != "throw":
            _fail("parameter initialization cannot return")
        if node["resume"] is not None and (
            node["resume"]["kind"] == "next"
            or completion != {"kind": node["resume"]["kind"], "value": node["resume"]["value"]}
        ):
            _fail("abrupt resume changed")
        node["phase"], node["resume"], node["pending"] = "completed", None, None
        if completion["kind"] == "throw":
            if record["name"] is not None:
                _fail("throw completion cannot allocate result")
            _complete_frame(machine, node["invocation"], "throw", completion["value"])
        else:
            _complete_frame(
                machine,
                node["invocation"],
                "return",
                _result(machine, record["name"], completion["value"], True),
            )
    elif kind == "generator.completed":
        _value(machine, record["argument"])
        if node["phase"] != "completed" and not (
            node["phase"] == "suspendedStart" and record["method"] != "next"
        ):
            _fail("generator body still needs resume")
        if node["invocation"] in machine["stack"]:
            _fail("completed generator frame is active")
        node["phase"], node["resume"], node["pending"] = "completed", None, None
        if record["method"] == "throw":
            if record["name"] is not None:
                _fail("throw cannot allocate result")
            machine["lastReturn"] = {"kind": "throw", "value": copy.deepcopy(record["argument"])}
        else:
            machine["lastReturn"] = {
                "kind": "return",
                "value": _result(
                    machine,
                    record["name"],
                    record["argument"] if record["method"] == "return" else ["undefined"],
                    True,
                ),
            }
    return True


def assert_next(machine, record):
    current_id = machine.get("stack", [])[-1] if machine.get("stack") else None
    node = next(
        (
            (identity, value)
            for identity, value in machine["heap"].items()
            if value.get("kind") == "generator" and value.get("invocation") == current_id
        ),
        None,
    )
    if node is None:
        return True
    identity, current = node
    kind = record.get("kind")
    if kind in ("return", "throw"):
        _fail("generator completion requires its retained continuation protocol")
    resume = current.get("resume") or {}
    if resume and current.get("delegation") is not None:
        if kind != "generator.delegate.receive" or record.get("object") != identity:
            _fail("delegated resume input must be consumed")
    elif resume.get("kind") == "next" and (
        kind != "generator.receive" or record.get("object") != identity
    ):
        _fail("normal resume input must be consumed")
    elif (
        resume
        and resume["kind"] != "next"
        and (
            kind not in ("generator.finish", "generator.handler.route")
            or record.get("object") != identity
        )
    ):
        _fail("abrupt resume requires handler or completion")
    if current.get("pending") and (kind != "generator.finish" or record.get("object") != identity):
        _fail("pending generator completion cannot be discarded")
    if (
        current.get("delegation")
        and current["delegation"]["phase"] == "calling"
        and kind not in ("generator.delegate.result", "generator.delegate.abrupt")
    ):
        _fail("delegated call must consume its exact result")
    if (
        current.get("handlers")
        and current["handlers"][-1].get("caught") is not None
        and (kind != "generator.handler.catch" or record.get("object") != identity)
    ):
        _fail("catch must consume exact thrown value")
    return True


def validate_state(machine):
    p = _protocol()

    def completion(value, *, normal=False):
        p._exact(value, {"kind", "value"}, "generator saved completion")
        if value["kind"] not in (("normal", "return", "throw") if normal else ("return", "throw")):
            _fail("saved completion kind")
        _value(machine, value["value"])
        if value["kind"] == "normal" and value["value"] != ["undefined"]:
            _fail("normal saved completion value")

    count = 0
    invocations = set()
    for identity, node in machine.get("heap", {}).items():
        if "generatorFunction" in node:
            if (
                node.get("kind") != "closure"
                or node["generatorFunction"] is not True
                or node.get("functionKind") != "ordinary"
                or node.get("callable") is not None
            ):
                _fail("generator function capability")
            descriptor = _descriptor(node, ["str", "prototype"])
            if (
                not descriptor
                or descriptor["kind"] != "data"
                or descriptor["enumerable"]
                or descriptor["configurable"]
            ):
                _fail("generator function prototype descriptor")
        if node.get("kind") != "generator":
            continue
        count += 1
        p._exact(
            node,
            {
                "kind",
                "callee",
                "invocation",
                "receiver",
                "arguments",
                "phase",
                "resume",
                "pending",
                "yielded",
                "handlers",
                "delegation",
                "prototype",
                "properties",
            }
            | (set(node) & {"descriptors", "keyOrder", "extensible"}),
            "generator heap node",
        )
        _id(node["invocation"])
        _value(machine, node["receiver"])
        if type(node["arguments"]) is not list or len(node["arguments"]) > BOUND:
            _fail("saved arguments")
        for value in node["arguments"]:
            _value(machine, value)
        if (
            node.get("phase")
            not in ("initializing", "suspendedStart", "executing", "suspendedYield", "completed")
            or type(node.get("yielded")) is not int
            or not 0 <= node["yielded"] <= BOUND
            or type(node.get("handlers")) is not list
            or len(node["handlers"]) > 64
            or type(node["properties"]) is not dict
        ):
            _fail("generator restored state")
        if node.get("invocation") in invocations:
            _fail("generator invocation reuse")
        invocations.add(node["invocation"])
        if (
            node.get("callee", [None])[0] != "ref"
            or machine["heap"].get(node["callee"][1], {}).get("generatorFunction") is not True
        ):
            _fail("generator retained callable")
        if (
            machine.get("frameFunctions", {}).get(node["invocation"])
            != machine["heap"][node["callee"][1]]["code"]
        ):
            _fail("generator retained invocation")
        handler_names = set()
        for handler in node["handlers"]:
            p._exact(
                handler,
                {"name", "catch", "finally", "phase", "caught", "pending"},
                "generator handler",
            )
            _id(handler["name"])
            if (
                handler["name"] in handler_names
                or type(handler["catch"]) is not bool
                or type(handler["finally"]) is not bool
                or not (handler["catch"] or handler["finally"])
                or handler["phase"] not in ("try", "catch", "finally")
                or handler["phase"] == "catch"
                and not handler["catch"]
                or handler["phase"] == "finally"
                and (not handler["finally"] or handler["pending"] is None)
            ):
                _fail("generator handler state")
            handler_names.add(handler["name"])
            if handler["caught"] is not None:
                _value(machine, handler["caught"])
                if handler["phase"] != "catch":
                    _fail("caught value outside catch")
            if handler["pending"] is not None:
                completion(handler["pending"], normal=True)
                if handler["phase"] != "finally":
                    _fail("pending completion outside cleanup")
        if node["pending"] is not None:
            completion(node["pending"])
            if (
                node["phase"] != "executing"
                or node["handlers"]
                or node["resume"] is not None
                or node["delegation"] is not None
            ):
                _fail("terminal completion state")
        if node["resume"] is not None:
            resume = node["resume"]
            p._exact(resume, {"kind", "from", "value"}, "generator resume")
            if (
                node["phase"] != "executing"
                or not isinstance(resume["kind"], str)
                or resume["kind"] not in METHODS
                or resume["from"] not in ("suspendedStart", "suspendedYield")
            ):
                _fail("generator resume state")
            _value(machine, resume["value"])
            if (
                resume["kind"] == "next"
                and resume["from"] == "suspendedStart"
                and resume["value"] != ["undefined"]
            ):
                _fail("initial next input was not ignored")
        delegation = node.get("delegation")
        if delegation is not None:
            p._exact(
                delegation,
                {"iterator", "next", "request", "phase", "result"},
                "generator delegation",
            )
            _value(machine, delegation["iterator"])
            p._exact(delegation["request"], {"kind", "value"}, "generator delegate request")
            _value(machine, delegation["request"]["value"])
            if (
                delegation["iterator"][0] != "ref"
                or delegation["iterator"][1] == identity
                or delegation["next"] not in ("Iterator.next", "Generator.next")
                or delegation["phase"] not in ("ready", "calling", "result", "complete", "yielded")
                or delegation["request"].get("kind") not in METHODS
            ):
                _fail("delegation state")
            target = machine["heap"].get(delegation["iterator"][1])
            if not target or target.get("kind") not in ("iterator", "generator"):
                _fail("delegated target")
            if delegation["next"] != (
                "Iterator.next" if target["kind"] == "iterator" else "Generator.next"
            ):
                _fail("cached delegate method")
            phase = delegation["phase"]
            if phase == "ready" and (
                delegation["result"] is not None or node["resume"] is not None
            ):
                _fail("delegate request state")
            if (
                phase == "calling"
                and target["kind"] == "generator"
                and delegation["result"] is not None
            ):
                _fail("delegate cannot predeclare its call result")
            if phase == "yielded":
                if not (
                    node["phase"] == "suspendedYield"
                    and node["resume"] is None
                    or node["phase"] == "executing"
                    and (node["resume"] or {}).get("from") == "suspendedYield"
                ):
                    _fail("suspended delegate state")
            elif node["phase"] != "executing" or node["resume"] is not None:
                _fail("active delegate state")
            if delegation["result"] is not None:
                if delegation["phase"] == "yielded":
                    # The result is already published to the caller. Its consumed
                    # done flag must not be re-read after an ordinary mutation.
                    result = _value(machine, delegation["result"])
                    if result[0] != "ref" or machine["heap"][result[1]]["kind"] != "object":
                        _fail("retained delegate result object")
                else:
                    _, done = _delegated_result(machine, delegation["result"])
                    if delegation["phase"] in ("result", "complete") and done != (
                        delegation["phase"] == "complete"
                    ):
                        _fail("saved delegate completion")
            elif phase not in ("ready", "calling"):
                _fail("delegate lacks its result identity")
        if node["phase"] in ("suspendedStart", "completed") and node["handlers"]:
            _fail("inactive generator has entered handler obligations")
        active = node["invocation"] in machine.get("stack", [])
        if active != (node["phase"] in ("initializing", "executing")):
            _fail("generator stack/phase mismatch")
        prototype = node["prototype"]
        if (
            prototype is not None
            and prototype != "ordinary"
            and not (isinstance(prototype, str) and prototype in INTRINSICS)
        ):
            _value(machine, prototype)
            if prototype[0] != "ref":
                _fail("generator prototype identity")
    if count > BOUND:
        _fail("generator allocation bound")
    return True
