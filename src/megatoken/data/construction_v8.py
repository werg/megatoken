"""Independent native checker for v8 class and private-element records.

Construction records extend the existing protocol machine. They share its
heap and protocol namespace; this module does not execute JavaScript source.
"""
from __future__ import annotations

import copy
import re

STATE_SCHEMA = "semantic-construction-state/1"
STATE_FIELDS = {"schema", "privateNames", "frames", "lastCompletion"}
CALLABLE_FIELDS = {"constructorKind", "classConstructor", "homeObject", "instanceInitializer", "staticInitializer"}
FIELDS = {
    "callable.define": {"kind", "object", "constructorKind", "classConstructor"},
    "callable.home": {"kind", "object", "home"},
    "class.link": {"kind", "object", "prototype", "heritage"},
    "class.initializer": {"kind", "object", "placement", "initializer"},
    "private.declare": {"kind", "id", "owner", "name"},
    "private.define": {"kind", "object", "privateId", "elementKind", "value"},
    "private.get": {"kind", "object", "privateId", "result"},
    "private.set": {"kind", "object", "privateId", "value"},
    "private.has": {"kind", "object", "privateId", "result"},
    "construct.enter": {"kind", "id", "child", "callee", "newTarget", "arguments", "receiver", "mode", "owner"},
    "construct.body": {"kind", "id", "child"},
    "construct.finish": {"kind", "id", "completion"},
    "construct.target": {"kind", "invocation", "result"},
    "construct.this": {"kind", "invocation", "result"},
    "super.select": {"kind", "owner", "callee"},
    "super.bind": {"kind", "owner", "receiver", "child"},
    "super.finish": {"kind", "owner", "result"},
}
FRAME_FIELDS = {"id", "parent", "bodyInvocation", "activeChild", "callee", "newTarget", "arguments", "receiver", "initialized", "phase", "superParent", "selectedSuper"}


def _fail(message):
    from .protocol_v8 import ProtocolV8Error
    raise ProtocolV8Error("Invalid construction protocol: " + message)


def validate_protocol_value(value):
    from .protocol_v8 import validate_protocol_value as validate
    return validate(value)


def _exact(value, fields, label):
    if type(value) is not dict or set(value) != set(fields): _fail(label + " fields")


def _id(value):
    if not isinstance(value, str) or not value or len(value) > 256: _fail("identity")


def _ref(value):
    validate_protocol_value(value)
    if value[0] != "ref": _fail("owned reference required")


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None: return False
    _exact(record, fields, "record")
    for key in ("object", "id", "owner", "privateId", "child", "invocation"):
        if key in record and record[key] is not None: _id(record[key])
    if kind in ("construct.enter","construct.body") and record["child"] is None: _fail("constructor child invocation")
    if kind in ("construct.target","construct.this") and record["invocation"] is None: _fail("constructor invocation")
    if kind == "callable.define":
        if record["constructorKind"] not in (None, "base", "derived") or type(record["classConstructor"]) is not bool:
            _fail("callable construction capability")
        if record["classConstructor"] and record["constructorKind"] is None: _fail("class constructor kind")
        if not record["classConstructor"] and record["constructorKind"] == "derived": _fail("nonclass derived constructor")
    elif kind == "callable.home": _ref(record["home"])
    elif kind == "class.link":
        _ref(record["prototype"]); validate_protocol_value(record["heritage"])
        if record["heritage"][0] not in ("ref", "null", "undefined"): _fail("class heritage")
    elif kind == "class.initializer":
        if record["placement"] not in ("instance", "static"): _fail("initializer placement")
        _ref(record["initializer"])
    elif kind == "private.declare":
        if not isinstance(record["name"], str) or not record["name"] or len(record["name"]) > 256: _fail("private source name")
    elif kind == "private.define":
        if record["elementKind"] not in ("field", "method"): _fail("private element kind")
        validate_protocol_value(record["value"])
        if record["elementKind"] == "method": _ref(record["value"])
    elif kind == "private.has":
        if type(record["result"]) is not bool: _fail("private brand result")
    elif kind == "construct.enter":
        _ref(record["callee"]); _ref(record["newTarget"]); validate_protocol_value(record["receiver"])
        if record["receiver"][0] not in ("ref", "undefined") or record["mode"] not in ("ordinary", "super") or (record["mode"] == "super") != (record["owner"] is not None): _fail("constructor entry linkage")
        if not isinstance(record["arguments"], list) or len(record["arguments"]) > 4096: _fail("constructor argument bound")
        for value in record["arguments"]: validate_protocol_value(value)
    elif kind == "construct.finish":
        completion=record["completion"]; _exact(completion,{"kind","value"},"constructor completion")
        if completion["kind"] not in ("normal","throw"): _fail("constructor completion kind")
        validate_protocol_value(completion["value"])
    elif kind == "super.select": validate_protocol_value(record["callee"])
    elif kind == "super.bind": _ref(record["receiver"])
    elif kind in ("private.get", "construct.target", "construct.this", "super.finish"):
        validate_protocol_value(record["result"])
    elif kind == "private.set": validate_protocol_value(record["value"])
    return True


def create_state():
    return {"schema": STATE_SCHEMA, "privateNames": [], "frames": [], "lastCompletion": None}


def _construct_state(machine):
    protocol = machine["protocol"]
    if "construction" not in protocol: protocol["construction"] = create_state()
    return protocol["construction"]


def _frame(machine,identity):
    row=next((item for item in _construct_state(machine)["frames"] if item["id"]==identity),None)
    if row is None: _fail("unknown constructor continuation")
    return row


def _node(machine, identity):
    node = machine["heap"].get(identity)
    if not isinstance(node, dict) or node.get("kind") not in ("object", "array", "closure"):
        _fail("owned object required")
    return node


def _callable(machine, identity):
    node = _node(machine, identity)
    if node["kind"] != "closure": _fail("callable must be an owned closure")
    return node


def _metadata(machine, identity):
    node = _callable(machine, identity)
    if not isinstance(node.get("callable"), dict): _fail("callable construction capability is undeclared")
    return node["callable"]


def _known(machine, value):
    validate_protocol_value(value)
    if value[0] == "ref" and value[1] not in machine["heap"]: _fail("unallocated reference")
    if value[0] == "symbol" and not any(row["id"] == value[1] for row in machine["protocol"]["symbols"]): _fail("undeclared Symbol")
    return value


def _private_name(machine, identity):
    state = _construct_state(machine)
    row = next((row for row in state["privateNames"] if row["id"] == identity), None)
    if row is None: _fail("undeclared private identity")
    return row


def _desc(node, key):
    # Heap public representation stores ordinary values in `properties` and
    # descriptor flags separately, using protocol_v8's internal key shape.
    prop = node.setdefault("properties", {})
    descriptors = node.setdefault("descriptors", {})
    meta = descriptors.get(key)
    if meta and meta.get("kind") == "accessor": return copy.deepcopy(meta)
    if key not in prop: return None
    default = {"kind": "data", "writable": True, "enumerable": True, "configurable": True}
    if node["kind"] == "array" and key == "length": default = {"kind": "data", "writable": True, "enumerable": False, "configurable": False}
    return {**default, **copy.deepcopy(meta or {}), "value": copy.deepcopy(prop[key])}


def _install(machine, node, key, descriptor):
    from .protocol_v8 import _install as protocol_install
    protocol_install(node, key, descriptor)


def _apply_private(machine, record):
    state = _construct_state(machine)
    kind = record["kind"]
    if kind == "private.declare":
        identity = record["id"]
        occupied = identity in machine["heap"] or any(row["id"] == identity for row in machine["protocol"]["symbols"])
        if occupied or not re.fullmatch(r"o[1-9][0-9]*", identity): _fail("private identity collision or format")
        if any(row["id"] == identity for row in state["privateNames"]) or any(row["owner"] == record["owner"] and row["name"] == record["name"] for row in state["privateNames"]):
            _fail("duplicate private identity or class name")
        if len(state["privateNames"]) >= 4096: _fail("private identity bound")
        if not _metadata(machine, record["owner"])["classConstructor"]: _fail("private owner is not a class")
        state["privateNames"].append({"id": identity, "owner": record["owner"], "name": record["name"]})
        return
    _private_name(machine, record["privateId"])
    node = _node(machine, record["object"])
    values = node.setdefault("privateElements", {})
    element = values.get(record["privateId"])
    if kind == "private.define":
        if element is not None: _fail("private element already installed")
        _known(machine, record["value"])
        if record["elementKind"] == "method":
            if _metadata(machine, record["value"][1])["constructorKind"] is not None: _fail("private method is constructible")
        values[record["privateId"]] = {"kind": record["elementKind"], "value": copy.deepcopy(record["value"]), "version": 1}
    elif kind == "private.has":
        if record["result"] != (element is not None): _fail("false private brand claim")
    elif kind == "private.get":
        if element is None: _fail("private brand is absent")
        _known(machine, record["result"])
        if record["result"] != element["value"]: _fail("false private read")
    else:
        if element is None: _fail("private brand is absent")
        if element["kind"] != "field": _fail("private method is readonly")
        _known(machine, record["value"])
        element["value"] = copy.deepcopy(record["value"]); element["version"] += 1


def _construction_next(machine, record):
    state=machine.get("protocol",{}).get("construction")
    if not state: return
    pending=next((row for row in reversed(state["frames"]) if row["phase"]!="complete" and row["activeChild"] not in machine["stack"]),None)
    if pending:
        expected={"initialize-base":"construct.body","initialize-derived":"super.finish"}.get(pending["phase"],"construct.finish")
        if record["kind"]!=expected or record.get("id",record.get("owner"))!=pending["id"]: _fail("pending constructor continuation requires "+expected)
    completed=state.get("lastCompletion")
    if completed and completed.get("superParent") is not None and completed["completion"]["kind"]=="normal":
        if record["kind"]!="super.bind" or record.get("owner")!=completed["superParent"]: _fail("super result must bind its derived receiver")


def assert_next(machine, record):
    _construction_next(machine,record)


def _call(machine, child, callee, receiver, arguments, *, construction=False):
    node=_node(machine,callee[1]); info=node.get("callable")
    if node["kind"]!="closure": _fail("owned callable required")
    if info and info["classConstructor"] and not construction: _fail("class constructor cannot be called without construction")
    if child in machine["frameFunctions"]: _fail("duplicate invocation identity")
    protocol=machine["protocol"]
    protocol["callFrames"].append({"child":child,"parent":machine["stack"][-1],"priorReceiver":copy.deepcopy(machine.get("receiver",["undefined"])),"priorAliases":[[key,value] for key,value in machine.get("captureAliases",{}).items()],"arguments":copy.deepcopy(arguments)})
    machine["receiver"]=copy.deepcopy(node.get("lexicalReceiver") if node.get("functionKind")=="arrow" else receiver)
    machine["captureAliases"]={row["binding"]:row["cell"] for row in node.get("captures",[])}
    machine["locals"][child+":this"]=copy.deepcopy(machine["receiver"])
    for index,value in enumerate(arguments): machine["locals"][child+":argument:"+str(index)]=copy.deepcopy(value)
    machine["frameFunctions"][child]=node.get("code"); machine["stack"].append(child)


def _apply_continuation(machine, record):
    kind=record["kind"]; state=_construct_state(machine)
    if kind=="construct.enter":
        for value in (record["callee"],record["newTarget"],*record["arguments"]): _known(machine,value)
        info=_metadata(machine,record["callee"][1]);target=_metadata(machine,record["newTarget"][1])
        if info["constructorKind"] is None or target["constructorKind"] is None: _fail("construction requires constructible targets")
        if len(state["frames"])>=4096 or any(row["id"]==record["id"] for row in state["frames"]): _fail("constructor identity or frame bound")
        parent=machine["stack"][-1]
        if record["mode"]=="super":
            owner=_frame(machine,record["owner"])
            if owner["phase"]!="body" or owner["bodyInvocation"]!=parent or _metadata(machine,owner["callee"][1])["constructorKind"]!="derived": _fail("super call has no active derived owner")
            if owner["selectedSuper"]!=record["callee"] or owner["newTarget"]!=record["newTarget"]: _fail("super constructor or newTarget changed")
            owner["selectedSuper"]=None
        derived=info["constructorKind"]=="derived"
        if derived and record["receiver"][0]!="undefined": _fail("derived this must begin uninitialized")
        if not derived:
            if record["receiver"][0]!="ref": _fail("base constructor requires fresh receiver")
            rid=record["receiver"][1]
            if rid in machine["heap"] or any(row["id"]==rid for row in machine["protocol"]["symbols"]): _fail("receiver identity collision")
            prototype=_desc(_node(machine,record["newTarget"][1]),"prototype")
            if prototype and prototype["kind"]!="data": _fail("newTarget prototype accessor unsupported")
            parent_proto="ordinary"
            if prototype and prototype["value"][0]=="ref":
                parent_proto=copy.deepcopy(prototype["value"])
                if _node(machine,parent_proto[1])["kind"] not in ("object","array"): _fail("unsupported constructor prototype")
            machine["heap"][rid]={"kind":"object","code":None,"captures":[],"prototype":parent_proto,"properties":{},"descriptors":{},"keyOrder":[],"extensible":True}
        entry={"id":record["id"],"parent":parent,"bodyInvocation":None,"activeChild":record["child"],"callee":copy.deepcopy(record["callee"]),"newTarget":copy.deepcopy(record["newTarget"]),"arguments":copy.deepcopy(record["arguments"]),"receiver":copy.deepcopy(record["receiver"]),"initialized":not derived,"phase":"body","superParent":record["owner"],"selectedSuper":None}
        state["frames"].append(entry);state["lastCompletion"]=None
        init=info["instanceInitializer"]
        if not derived and init:
            entry["phase"]="initialize-base";_call(machine,record["child"],init,entry["receiver"],[],construction=True)
        else:
            _call(machine,record["child"],entry["callee"],entry["receiver"],entry["arguments"],construction=True)
            entry["phase"]="body";entry["activeChild"]=record["child"];entry["bodyInvocation"]=record["child"]
        return
    if kind=="construct.body":
        entry=_frame(machine,record["id"])
        if entry["phase"]!="initialize-base" or machine.get("lastReturn",{}).get("kind")!="return" or machine["stack"][-1]!=entry["parent"]: _fail("base initializer must return before body entry")
        _call(machine,record["child"],entry["callee"],entry["receiver"],entry["arguments"],construction=True)
        entry.update(phase="body",activeChild=record["child"],bodyInvocation=record["child"]);return
    if kind=="construct.finish":
        entry=_frame(machine,record["id"]);returned=machine.get("lastReturn")
        if entry["phase"]!="body" or machine["stack"][-1]!=entry["parent"] or returned is None: _fail("constructor body is not complete")
        _known(machine,record["completion"]["value"])
        if returned["kind"]=="throw": expected={"kind":"throw","value":returned["value"]}
        elif returned["value"][0]=="ref": expected={"kind":"normal","value":returned["value"]}
        else:
            if _metadata(machine,entry["callee"][1])["constructorKind"]!="base" and (returned["value"][0]!="undefined" or not entry["initialized"]): _fail("derived primitive/uninitialized return needs abrupt construction")
            expected={"kind":"normal","value":entry["receiver"]}
        if expected!=record["completion"]: _fail("false constructor completion")
        entry["phase"]="complete";state["lastCompletion"]={"id":entry["id"],"parent":entry["parent"],"superParent":entry["superParent"],"completion":copy.deepcopy(expected)}
        machine["lastReturn"]={"kind":"return" if expected["kind"]=="normal" else "throw","value":copy.deepcopy(expected["value"])};return
    if kind in ("construct.target","construct.this"):
        if machine["stack"][-1]!=record["invocation"]: _fail("receiver/newTarget read invocation")
        entry=next((row for row in reversed(state["frames"]) if row["bodyInvocation"]==record["invocation"] and row["phase"]!="complete"),None)
        expected=(entry["newTarget"] if entry else ["undefined"]) if kind=="construct.target" else machine.get("receiver",["undefined"])
        if kind=="construct.this" and entry and not entry["initialized"]: _fail("uninitialized this")
        _known(machine,record["result"])
        if record["result"]!=expected: _fail("false receiver/newTarget read")
        return
    if kind=="super.select":
        owner=_frame(machine,record["owner"])
        if owner["phase"]!="body" or owner["bodyInvocation"]!=machine["stack"][-1] or _metadata(machine,owner["callee"][1])["constructorKind"]!="derived" or owner["selectedSuper"] is not None: _fail("super selection requires active derived constructor")
        expected=_node(machine,owner["callee"][1]).get("prototype","ordinary"); expected=expected if isinstance(expected,list) else ["undefined"]
        _known(machine,record["callee"])
        if record["callee"]!=expected: _fail("false super constructor selection")
        owner["selectedSuper"]=copy.deepcopy(expected);return
    if kind=="super.bind":
        owner=_frame(machine,record["owner"]);completed=state.get("lastCompletion")
        if owner["phase"]!="body" or owner["bodyInvocation"]!=machine["stack"][-1] or owner["initialized"] or not completed or completed["superParent"]!=owner["id"] or completed["parent"]!=owner["bodyInvocation"] or completed["completion"]["kind"]!="normal" or completed["completion"]["value"]!=record["receiver"]: _fail("super binding lacks successful superclass result")
        _known(machine,record["receiver"]);owner["receiver"]=copy.deepcopy(record["receiver"]);owner["initialized"]=True;state["lastCompletion"]=None
        machine["receiver"]=copy.deepcopy(record["receiver"]);machine["locals"][owner["bodyInvocation"]+":this"]=copy.deepcopy(record["receiver"])
        initializer=_metadata(machine,owner["callee"][1])["instanceInitializer"]
        if initializer:
            if record["child"] is None: _fail("derived initializer invocation required")
            owner["phase"]="initialize-derived";owner["activeChild"]=record["child"]
            _call(machine,record["child"],initializer,record["receiver"],[],construction=True)
        elif record["child"] is not None: _fail("invented derived initializer")
        return
    if kind=="super.finish":
        owner=_frame(machine,record["owner"])
        if owner["phase"]!="initialize-derived" or machine["stack"][-1]!=owner["bodyInvocation"] or machine.get("lastReturn",{}).get("kind")!="return": _fail("derived initializer has not completed normally")
        if record["result"]!=owner["receiver"] or machine.get("receiver")!=owner["receiver"]: _fail("derived initializer lost receiver")
        owner["phase"]="body";owner["activeChild"]=owner["bodyInvocation"]


def apply_record(machine, record):
    """Apply validated construction metadata or private storage transition."""
    if not validate_record(record): return False
    kind = record["kind"]
    _construction_next(machine,record)
    state = _construct_state(machine)
    if kind.startswith("construct.") or kind.startswith("super."):
        _apply_continuation(machine,record)
        return True
    if kind.startswith("private."):
        _apply_private(machine, record)
        return True
    if kind == "callable.define":
        node = _callable(machine, record["object"])
        if node.get("callable") is not None: _fail("callable already declared")
        if node.get("functionKind") == "arrow" and (record["constructorKind"] is not None or record["classConstructor"]): _fail("arrow cannot be a constructor")
        node["callable"] = {"constructorKind": record["constructorKind"], "classConstructor": record["classConstructor"], "homeObject": None, "instanceInitializer": None, "staticInitializer": None}
        return True
    if kind == "callable.home":
        info = _metadata(machine, record["object"])
        _known(machine, record["home"]); _node(machine, record["home"][1])
        if _node(machine, record["object"]).get("functionKind") == "arrow" or info["constructorKind"] is not None or info["homeObject"] is not None: _fail("home object immutable or nonconstructible method required")
        info["homeObject"] = copy.deepcopy(record["home"])
        return True
    if kind == "class.link":
        info = _metadata(machine, record["object"]); proto = _node(machine, record["prototype"][1]); node = _node(machine, record["object"])
        if not info["classConstructor"] or info["homeObject"] is not None: _fail("class already linked or not a class")
        if proto["kind"] != "object" or record["object"] == record["prototype"][1]: _fail("class prototype must be a distinct ordinary object")
        instance_parent = "ordinary"; constructor_parent = "ordinary"
        heritage = record["heritage"]
        if info["constructorKind"] == "base":
            if heritage[0] != "undefined": _fail("base class cannot have heritage")
        else:
            if heritage[0] == "undefined": _fail("derived class requires heritage")
            if heritage[0] == "null": instance_parent = None
            else:
                parent_info = _metadata(machine, heritage[1])
                if parent_info["constructorKind"] is None: _fail("heritage is not constructible")
                pnode = _node(machine, heritage[1]); d = _desc(pnode, "prototype")
                if not d or d["kind"] != "data": _fail("heritage prototype must be an owned data property")
                _known(machine, d["value"])
                if d["value"][0] == "ref": _node(machine, d["value"][1])
                if d["value"][0] not in ("null", "ref"): _fail("heritage prototype must be object or null")
                instance_parent = None if d["value"][0] == "null" else copy.deepcopy(d["value"])
                constructor_parent = copy.deepcopy(heritage)
        existing = _desc(node, "prototype"); back = _desc(proto, "constructor")
        if existing and (existing["kind"] != "data" or existing["value"] != record["prototype"]): _fail("class prototype identity changed")
        if back and (back["kind"] != "data" or back["value"] != ["ref", record["object"]]): _fail("class constructor backlink changed")
        node["prototype"] = copy.deepcopy(constructor_parent); proto["prototype"] = copy.deepcopy(instance_parent); info["homeObject"] = copy.deepcopy(record["prototype"])
        _install(machine, node, "prototype", {"kind": "data", "value": copy.deepcopy(record["prototype"]), "writable": False, "enumerable": False, "configurable": False})
        _install(machine, proto, "constructor", {"kind": "data", "value": ["ref", record["object"]], "writable": True, "enumerable": False, "configurable": True})
        return True
    if kind == "class.initializer":
        info = _metadata(machine, record["object"]); initializer = record["initializer"][1]; init = _metadata(machine, initializer)
        field = record["placement"] + "Initializer"
        expected_home = ["ref", record["object"]] if record["placement"] == "static" else info["homeObject"]
        if not info["classConstructor"] or info["homeObject"] is None or info[field] is not None: _fail("initializer already declared or class unlinked")
        if init["constructorKind"] is not None or init["classConstructor"] or init["homeObject"] != expected_home: _fail("initializer capability/home mismatch")
        info[field] = copy.deepcopy(record["initializer"])
        return True
    _fail("unsupported construction transition")


def validate_state(snapshot, machine):
    _exact(snapshot, STATE_FIELDS, "construction state")
    if snapshot["schema"] != STATE_SCHEMA or not isinstance(snapshot["privateNames"], list) or not isinstance(snapshot["frames"], list): _fail("state schema or arrays")
    if len(snapshot["privateNames"]) > 4096: _fail("private identity bound")
    identities=set(); names=set()
    for row in snapshot["privateNames"]:
        _exact(row, {"id", "owner", "name"}, "private identity")
        for field in ("id", "owner", "name"): _id(row[field])
        if row["id"] in identities or (row["owner"],row["name"]) in names: _fail("duplicate private identity/name")
        identities.add(row["id"]); names.add((row["owner"],row["name"]))
        if row["id"] in machine["heap"] or any(x["id"] == row["id"] for x in machine["protocol"]["symbols"]): _fail("private identity collision")
        if not _metadata(machine,row["owner"])["classConstructor"]: _fail("private owner is not a class")
    for object_id,node in machine["heap"].items():
        if "callable" in node:
            info=node["callable"]
            if node.get("kind")!="closure": _fail("callable metadata requires closure")
            _exact(info,CALLABLE_FIELDS,"callable state")
            if info["constructorKind"] not in (None,"base","derived") or type(info["classConstructor"]) is not bool: _fail("callable state capability")
            if info["classConstructor"] and info["constructorKind"] is None or not info["classConstructor"] and info["constructorKind"]=="derived": _fail("callable state class/kind mismatch")
            if node.get("functionKind")=="arrow" and (info["constructorKind"] is not None or info["classConstructor"]): _fail("arrow constructor metadata")
            if info["homeObject"] is not None:
                _known(machine,info["homeObject"])
                if info["homeObject"][0]!="ref": _fail("home object reference")
                _node(machine,info["homeObject"][1])
            for placement in ("instance","static"):
                initializer=info[placement+"Initializer"]
                if initializer is not None:
                    _known(machine,initializer)
                    if initializer[0]!="ref": _fail("initializer reference")
                    init=_metadata(machine,initializer[1])
                    expected_home=["ref",object_id] if placement=="static" else info["homeObject"]
                    if not info["classConstructor"] or init["constructorKind"] is not None or init["classConstructor"] or init["homeObject"]!=expected_home: _fail("initializer ownership")
        if "privateElements" in node:
            entries=node["privateElements"]
            if not isinstance(entries,dict): _fail("private elements map")
            for private_id,element in entries.items():
                if private_id not in identities: _fail("private storage has undeclared identity")
                _exact(element,{"kind","value","version"},"private element")
                if element["kind"] not in ("field","method") or type(element["version"]) is not int or element["version"]<1: _fail("private element kind/version")
                _known(machine,element["value"])
                if element["kind"]=="method" and (element["value"][0]!="ref" or element["version"]!=1 or _metadata(machine,element["value"][1])["constructorKind"] is not None): _fail("private method storage")
    if len(snapshot["frames"]) > 4096: _fail("construction frame bound")
    frames={}
    for row in snapshot["frames"]:
        _exact(row,FRAME_FIELDS,"construction frame")
        for key in ("id","parent","activeChild"): _id(row[key])
        if row["bodyInvocation"] is not None: _id(row["bodyInvocation"])
        if row["id"] in frames: _fail("duplicate construction frame")
        frames[row["id"]]=row
        _known(machine,row["callee"]);_known(machine,row["newTarget"]);_known(machine,row["receiver"])
        if row["callee"][0]!="ref" or row["newTarget"][0]!="ref": _fail("constructor references")
        info=_metadata(machine,row["callee"][1])
        if info["constructorKind"] is None or _metadata(machine,row["newTarget"][1])["constructorKind"] is None: _fail("frame has nonconstructible target")
        if not isinstance(row["arguments"],list) or len(row["arguments"])>4096: _fail("retained arguments")
        for value in row["arguments"]:_known(machine,value)
        if type(row["initialized"]) is not bool or row["phase"] not in ("initialize-base","body","initialize-derived","complete"): _fail("constructor phase")
        if row["initialized"] != (row["receiver"][0]=="ref") or (not row["initialized"] and info["constructorKind"]!="derived"): _fail("invalid retained receiver")
        if row["parent"] not in machine.get("frameFunctions",{}) or row["activeChild"] not in machine.get("frameFunctions",{}): _fail("constructor invocation identity absent")
        if row["bodyInvocation"] is not None and machine["frameFunctions"].get(row["bodyInvocation"])!=_node(machine,row["callee"][1]).get("code"): _fail("constructor body identity mismatch")
        if row["phase"]=="initialize-base" and (info["constructorKind"]!="base" or row["bodyInvocation"] is not None or info["instanceInitializer"] is None): _fail("invalid base initializer continuation")
        if row["phase"]=="initialize-derived" and (info["constructorKind"]!="derived" or row["bodyInvocation"] is None or not row["initialized"] or info["instanceInitializer"] is None): _fail("invalid derived initializer continuation")
        if row["phase"]=="body" and row["bodyInvocation"]!=row["activeChild"]: _fail("constructor active body mismatch")
        if row["phase"]=="complete" and (row["activeChild"] in machine["stack"] or row["bodyInvocation"] in machine["stack"]): _fail("completed constructor remains active")
        elif row["activeChild"] in machine["stack"]:
            parent=row["bodyInvocation"] if row["phase"]=="initialize-derived" else row["parent"]
            if machine["stack"].index(row["activeChild"])!=machine["stack"].index(parent)+1: _fail("constructor continuation stack mismatch")
        if row["selectedSuper"] is not None:
            _known(machine,row["selectedSuper"])
            if info["constructorKind"]!="derived" or row["phase"]!="body": _fail("super selected outside derived body")
    for row in snapshot["frames"]:
        if row["superParent"] is not None:
            owner=frames.get(row["superParent"])
            if not owner or owner["bodyInvocation"]!=row["parent"] or owner["newTarget"]!=row["newTarget"]: _fail("super owner/newTarget mismatch")
    if snapshot["lastCompletion"] is not None:
        completed=snapshot["lastCompletion"];_exact(completed,{"id","parent","superParent","completion"},"last constructor completion")
        row=frames.get(completed["id"])
        if not row or row["phase"]!="complete" or row["parent"]!=completed["parent"] or row["superParent"]!=completed["superParent"]: _fail("last completion identity")
        _exact(completed["completion"],{"kind","value"},"constructor completion")
        if completed["completion"]["kind"] not in ("normal","throw"): _fail("constructor completion kind")
        _known(machine,completed["completion"]["value"])
    return True
