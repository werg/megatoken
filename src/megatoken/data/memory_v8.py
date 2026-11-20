"""Independent bounded Python replay for semantic v8 owned binary memory."""

from __future__ import annotations

import copy
import math
import struct

from . import algorithm_v7

PROFILE = "owned-binary-memory/1"
TYPE_INFO = {
    "Int8Array": (1, "b", False, True),
    "Uint8Array": (1, "B", False, True),
    "Int16Array": (2, "h", False, True),
    "Uint16Array": (2, "H", False, True),
    "Int32Array": (4, "i", False, True),
    "Uint32Array": (4, "I", False, True),
    "BigInt64Array": (8, "q", True, True),
    "BigUint64Array": (8, "Q", True, True),
    "Float32Array": (4, "f", False, False),
    "Float64Array": (8, "d", False, False),
}
DATA_TYPES = {name[:-5]: spec for name, spec in TYPE_INFO.items()}
CONSTRUCTORS = {"ArrayBuffer", "SharedArrayBuffer", "DataView", *TYPE_INFO}
METHODS = {
    "ArrayBuffer.isView",
    "ArrayBuffer.prototype.slice",
    "SharedArrayBuffer.prototype.slice",
    "SharedArrayBuffer.prototype.grow",
    "TypedArray.prototype.subarray",
    *{"DataView.prototype.get" + name for name in DATA_TYPES},
    *{"DataView.prototype.set" + name for name in DATA_TYPES},
    *{
        "Atomics." + name
        for name in (
            "load",
            "store",
            "add",
            "sub",
            "and",
            "or",
            "xor",
            "exchange",
            "compareExchange",
            "wait",
            "notify",
        )
    },
}
FIELDS = {
    "memory.construct": {"kind", "constructor", "name", "backing", "arguments"},
    "memory.get": {"kind", "object", "key", "result"},
    "memory.set": {"kind", "object", "key", "value"},
    "memory.call": {"kind", "method", "receiver", "arguments", "name", "result"},
}


def _fail(message):
    from .protocol_v8 import ProtocolV8Error

    raise ProtocolV8Error("Invalid binary-memory protocol: " + message)


def validate_protocol_value(value):
    from .protocol_v8 import validate_protocol_value as validate

    return validate(value)


def validate_record(record):
    kind = record.get("kind") if type(record) is dict else None
    fields = FIELDS.get(kind)
    if fields is None:
        return False
    if set(record) != fields:
        _fail("record fields")
    for key in ("name", "backing", "object"):
        if (
            key in record
            and record[key] is not None
            and (not isinstance(record[key], str) or not record[key] or len(record[key]) > 256)
        ):
            _fail("identity")
    if kind == "memory.construct":
        if record["constructor"] not in CONSTRUCTORS or record["name"] is None:
            _fail("constructor or allocation identity")
    if kind in ("memory.get", "memory.set"):
        if (
            record["object"] is None
            or not isinstance(record["key"], list)
            or len(record["key"]) != 2
            or record["key"][0] != "str"
            or not isinstance(record["key"][1], str)
        ):
            _fail("memory property key")
    if "arguments" in record:
        if not isinstance(record["arguments"], list) or len(record["arguments"]) > 64:
            _fail("argument bound")
        for value in record["arguments"]:
            validate_protocol_value(value)
    if kind == "memory.call":
        if record["method"] not in METHODS:
            _fail("unsupported memory method")
        if record["receiver"] is not None:
            validate_protocol_value(record["receiver"])
            if record["receiver"][0] != "ref":
                _fail("receiver identity")
        if record["name"] is not None and (
            not isinstance(record["name"], str) or not record["name"]
        ):
            _fail("allocation identity")
    for key in ("result", "value"):
        if key in record:
            validate_protocol_value(record[key])
    return True


def _number(value):
    validate_protocol_value(value)
    try:
        return algorithm_v7._to_number(value)
    except TypeError:
        _fail("cannot convert value to Number")


def _integer(value):
    number = _number(value)
    if math.isnan(number) or number == 0:
        return 0
    if not math.isfinite(number):
        return number
    return math.trunc(number)


def _index(value):
    integer = _integer(value)
    if not isinstance(integer, int) or integer < 0 or integer > 9007199254740991:
        _fail("invalid binary-memory index")
    return integer


def _element(value, spec):
    width, fmt, bigint, _ = spec
    if bigint:
        if value[0] == "bigint":
            return int(value[1])
        if value[0] == "bool":
            return int(value[1])
        if value[0] == "str":
            converted = algorithm_v7._bigint_string(value[1])
            if converted is None:
                _fail("invalid BigInt conversion")
            return converted
        _fail("cannot convert value to BigInt")
    if value[0] not in ("num", "str", "bool", "null", "undefined"):
        _fail("object coercion requires an explicit continuation")
    return _number(value)


def _integer_element(value, spec):
    """Apply ECMAScript's modulo conversion for integer storage elements."""
    width, fmt, bigint, _ = spec
    converted = _element(value, spec)
    if bigint:
        raw = int(converted) % (1 << (width * 8))
    else:
        if not math.isfinite(converted) or converted == 0:
            return 0
        raw = math.trunc(converted) % (1 << (width * 8))
    if fmt in ("b", "h", "i", "q") and raw >= (1 << (width * 8 - 1)):
        raw -= 1 << (width * 8)
    return raw


def _tag_number(value):
    if type(value) is int:
        value = float(value)
    return algorithm_v7._encode_scalar(value)


def _node(machine, identity):
    node = machine["heap"].get(identity)
    if not isinstance(node, dict) or node.get("kind") not in (
        "array-buffer",
        "shared-int32-buffer",
        "typed-array",
        "shared-int32-view",
        "data-view",
        "shared-data-view",
    ):
        _fail("known binary-memory object required")
    return node


def _alloc(machine, identity, node):
    if (
        not isinstance(identity, str)
        or not identity
        or identity in machine["heap"]
        or any(row["id"] == identity for row in machine["protocol"]["symbols"])
    ):
        _fail("binary-memory identity collision")
    if identity in {
        row["id"] for row in (machine["protocol"].get("construction") or {}).get("privateNames", [])
    }:
        _fail("reserved private identity")
    node = copy.deepcopy(node)
    node["properties"] = {}
    machine["heap"][identity] = node


def _buffer(kind, identity, length, max_length=None):
    if not isinstance(length, int) or length < 0 or length > 1024:
        _fail("buffer length outside bound")
    if kind == "array-buffer":
        return {
            "kind": kind,
            "byteLength": length,
            "maxByteLength": length,
            "resizable": False,
            "detached": False,
            "bytes": [0] * length,
        }
    maximum = length if max_length is None else max_length
    if not isinstance(maximum, int) or maximum < length or maximum > 1024:
        _fail("shared maximum byte length")
    return {
        "kind": kind,
        "byteLength": length,
        "maxByteLength": maximum,
        "growable": max_length is not None,
        "bytes": [0] * length,
    }


def _construct(machine, record):
    args = record["arguments"]

    def arg(n):
        return args[n] if n < len(args) else ["undefined"]

    value = arg(0)
    name = record["name"]
    ctor = record["constructor"]
    if ctor in ("ArrayBuffer", "SharedArrayBuffer"):
        if record["backing"] is not None:
            _fail("buffer allocation must not have a second backing identity")
        length = _index(value)
        if length > 1024:
            _fail("buffer size bound")
        if ctor == "ArrayBuffer":
            if arg(1)[0] != "undefined":
                _fail("resizable ArrayBuffer outside profile")
            node = _buffer("array-buffer", name, length)
        else:
            max_length = None
            options = arg(1)
            if options[0] != "undefined":
                if options[0] != "ref":
                    _fail("buffer options must be an ordinary data object")
                obj = machine["heap"].get(options[1])
                if (
                    not obj
                    or obj.get("kind") != "object"
                    or obj.get("prototype", "ordinary") not in ("ordinary", None)
                ):
                    _fail("shared option prototype")
                desc = _descriptor(obj, "maxByteLength")
                if desc and desc.get("kind") != "data":
                    _fail("option getter requires call protocol")
                if desc and desc["value"][0] != "undefined":
                    max_length = _index(desc["value"])
            node = _buffer("shared-int32-buffer", name, length, max_length)
        _alloc(machine, name, node)
        return
    backing = machine["heap"].get(value[1]) if value[0] == "ref" else None
    if ctor == "DataView":
        if (
            not backing
            or backing.get("kind") not in ("array-buffer", "shared-int32-buffer")
            or record["backing"] is not None
        ):
            _fail("DataView requires existing owned backing")
        offset = _index(arg(1))
        length = None if arg(2)[0] == "undefined" else _index(arg(2))
        if offset > backing["byteLength"]:
            _fail("DataView offset")
        tracking = length is None and backing.get("growable", False)
        size = backing["byteLength"] - offset if length is None else length
        if offset + size > backing["byteLength"]:
            _fail("DataView length")
        _alloc(
            machine,
            name,
            {
                "kind": "shared-data-view"
                if backing["kind"] == "shared-int32-buffer"
                else "data-view",
                "bufferId": value[1],
                "byteOffset": offset,
                "byteLength": size,
                "lengthTracking": tracking,
            },
        )
        return
    spec = TYPE_INFO[ctor]
    if backing and backing.get("kind") in ("array-buffer", "shared-int32-buffer"):
        if record["backing"] is not None:
            _fail("typed view with existing backing cannot allocate another")
        buffer_id = value[1]
        offset = _index(arg(1))
        if offset % spec[0]:
            _fail("typed array byte alignment")
        length = None if arg(2)[0] == "undefined" else _index(arg(2))
    else:
        if value[0] == "ref" or record["backing"] is None or record["backing"] == name:
            _fail("typed scalar constructor needs fresh backing identity")
        length = _index(value)
        offset = 0
        buffer_id = record["backing"]
        if length * spec[0] > 1024:
            _fail("typed array backing bound")
        backing = _buffer("array-buffer", buffer_id, length * spec[0])
        _alloc(machine, buffer_id, backing)
    available = backing["byteLength"] - offset
    tracking = (
        length is None
        and backing["kind"] == "shared-int32-buffer"
        and backing.get("growable", False)
    )
    if length is None:
        if not tracking and available % spec[0]:
            _fail("typed array size alignment")
        length = available // spec[0]
    if length < 0 or length * spec[0] > available:
        _fail("typed array length bounds")
    _alloc(
        machine,
        name,
        {
            "kind": "shared-int32-view"
            if backing["kind"] == "shared-int32-buffer"
            else "typed-array",
            "bufferId": buffer_id,
            "elementType": ctor,
            "byteOffset": offset,
            "length": length,
            "lengthTracking": tracking,
        },
    )


def _descriptor(node, key):
    props = node.get("properties", {})
    return {"kind": "data", "value": props[key]} if key in props else None


def _effective_length(buffer, view):
    width = TYPE_INFO[view["elementType"]][0]
    return (
        max(0, (buffer["byteLength"] - view["byteOffset"]) // width)
        if view["lengthTracking"]
        else view["length"]
    )


def _read_typed(buffer, view, index):
    spec = TYPE_INFO[view["elementType"]]
    width, fmt, big, _ = spec
    if not isinstance(index, int) or index < 0 or index >= _effective_length(buffer, view):
        _fail("typed array index bounds")
    offset = view["byteOffset"] + index * width
    try:
        value = struct.unpack_from("<" + fmt, bytes(buffer["bytes"]), offset)[0]
    except struct.error:
        _fail("corrupt backing bytes")
    return ["bigint", str(value)] if big else _tag_number(value)


def _write_typed(buffer, view, index, value):
    width, fmt, big, _ = TYPE_INFO[view["elementType"]]
    if not isinstance(index, int) or index < 0 or index >= _effective_length(buffer, view):
        _fail("typed array index bounds")
    spec = TYPE_INFO[view["elementType"]]
    converted = _integer_element(value, spec) if spec[3] else _element(value, spec)
    try:
        packed = struct.pack("<" + fmt, converted)
    except OverflowError:
        # Float32 conversion overflows to signed infinity in ECMAScript.
        if fmt != "f" or not math.isfinite(converted):
            raise
        packed = struct.pack("<f", math.copysign(math.inf, converted))
    offset = view["byteOffset"] + index * width
    buffer["bytes"][offset : offset + width] = packed


def _numeric_key(key):
    if key == "-0":
        return -0.0
    try:
        number = float(key)
    except ValueError:
        return None
    if str(int(number)) == key if math.isfinite(number) and number.is_integer() else False:
        return int(number)
    if math.isfinite(number) and str(number) == key:
        return number
    return None


def _read_property(machine, identity, key):
    obj = _node(machine, identity)
    if obj["kind"] in ("array-buffer", "shared-int32-buffer"):
        if key in ("byteLength", "maxByteLength"):
            return ["num", obj[key]]
        if obj["kind"] == "array-buffer" and key in ("resizable", "detached"):
            return ["bool", False]
        if obj["kind"] == "shared-int32-buffer" and key == "growable":
            return ["bool", obj["growable"]]
    else:
        buffer = _node(machine, obj["bufferId"])
        if key == "buffer":
            return ["ref", obj["bufferId"]]
        if key == "byteOffset":
            return ["num", obj["byteOffset"]]
        if obj["kind"] in ("data-view", "shared-data-view") and key == "byteLength":
            return [
                "num",
                buffer["byteLength"] - obj["byteOffset"]
                if obj["lengthTracking"]
                else obj["byteLength"],
            ]
        if obj["kind"] in ("typed-array", "shared-int32-view"):
            length = _effective_length(buffer, obj)
            width = TYPE_INFO[obj["elementType"]][0]
            if key == "length":
                return ["num", length]
            if key == "byteLength":
                return ["num", length * width]
            if key == "BYTES_PER_ELEMENT":
                return ["num", width]
            index = _numeric_key(key)
            if index is not None:
                if isinstance(index, float) and (
                    not index.is_integer() or math.copysign(1, index) < 0 and index == 0
                ):
                    return ["undefined"]
                return (
                    _read_typed(buffer, obj, int(index)) if 0 <= index < length else ["undefined"]
                )
    _fail("unsupported binary-memory property")


def _set_property(machine, identity, key, value):
    view = _node(machine, identity)
    if view["kind"] not in ("typed-array", "shared-int32-view"):
        _fail("indexed typed-array write required")
    _element(value, TYPE_INFO[view["elementType"]])
    index = _numeric_key(key)
    if index is None:
        _fail("ordinary typed-array properties outside profile")
    buffer = _node(machine, view["bufferId"])
    if isinstance(index, float) and (
        not index.is_integer() or index == 0 and math.copysign(1, index) < 0
    ):
        return
    index = int(index)
    if 0 <= index < _effective_length(buffer, view):
        _write_typed(buffer, view, index, value)


def _truth(value):
    return value[0] == "symbol" or algorithm_v7._truthy(value)


def _call(machine, record):
    method = record["method"]
    args = record["arguments"]
    receiver = record["receiver"]

    def argument(index):
        return args[index] if index < len(args) else ["undefined"]

    allocation = method in (
        "TypedArray.prototype.subarray",
        "ArrayBuffer.prototype.slice",
        "SharedArrayBuffer.prototype.slice",
    )
    if allocation != (record["name"] is not None):
        _fail("binary-memory allocation claim")
    if method == "ArrayBuffer.isView":
        if receiver is not None:
            _fail("isView static receiver")
        node = machine["heap"].get(argument(0)[1]) if args and argument(0)[0] == "ref" else None
        return [
            "bool",
            bool(
                node
                and node.get("kind")
                in ("typed-array", "shared-int32-view", "data-view", "shared-data-view")
            ),
        ]
    if method.startswith("Atomics."):
        if receiver is not None or not args or argument(0)[0] != "ref":
            _fail("Atomics view argument")
        view = _node(machine, argument(0)[1])
        if view["kind"] != "shared-int32-view":
            _fail("single-agent Atomics requires shared typed storage")
        if not TYPE_INFO[view["elementType"]][3]:
            _fail("Atomics requires integer element")
        buffer = _node(machine, view["bufferId"])
        index = _index(argument(1))
        if index >= _effective_length(buffer, view):
            _fail("atomic view bounds")
        op = method[8:]
        old = _read_typed(buffer, view, index)
        if op == "load":
            return old
        if op in ("notify", "wait"):
            if view["elementType"] not in ("Int32Array", "BigInt64Array"):
                _fail("wait/notify element type")
            if op == "notify":
                return ["num", 0]
            if len(args) != 4 or _number(argument(3)) != 0:
                _fail("only zero-timeout wait is owned")
            target = _integer_element(argument(2), TYPE_INFO[view["elementType"]])
            actual = int(old[1]) if old[0] == "bigint" else _number(old)
            return ["str", "timed-out" if actual == target else "not-equal"]
        spec = TYPE_INFO[view["elementType"]]
        if op == "store":
            converted = _element(argument(2), spec) if spec[2] else _integer(argument(2))
            _write_typed(buffer, view, index, argument(2))
            return ["bigint", str(converted)] if spec[2] else _tag_number(converted)
        new = _integer_element(argument(2), spec)
        old_raw = int(old[1]) if old[0] == "bigint" else _number(old)
        if op == "add":
            new = old_raw + new
        elif op == "sub":
            new = old_raw - new
        elif op == "and":
            new = int(old_raw) & int(new)
        elif op == "or":
            new = int(old_raw) | int(new)
        elif op == "xor":
            new = int(old_raw) ^ int(new)
        elif op == "exchange":
            pass
        elif op == "compareExchange":
            expected = _integer_element(argument(2), spec)
            replacement = _integer_element(argument(3), spec)
            new = replacement if old_raw == expected else old_raw
        else:
            _fail("unknown atomic operation")
        _write_typed(
            buffer,
            view,
            index,
            ["bigint", str(int(new))]
            if spec[2]
            else _tag_number(float(new) if isinstance(new, float) else new),
        )
        return old
    if receiver is None:
        _fail("memory receiver required")
    target = _node(machine, receiver[1])
    if method.startswith("DataView.prototype."):
        if target["kind"] not in ("data-view", "shared-data-view"):
            _fail("DataView receiver")
        name = method.split(".")[-1]
        setter = name.startswith("set")
        type_name = name[3:]
        spec = DATA_TYPES[type_name]
        index = _index(argument(0))
        size = (
            target["byteLength"]
            if not target["lengthTracking"]
            else _node(machine, target["bufferId"])["byteLength"] - target["byteOffset"]
        )
        if index + spec[0] > size:
            _fail("DataView index bounds")
        buffer = _node(machine, target["bufferId"])
        offset = target["byteOffset"] + index * 1
        little = _truth(args[2 if setter else 1]) if len(args) > (2 if setter else 1) else False
        endian = "<" if little else ">"
        if setter:
            converted = (
                _integer_element(argument(1), spec) if spec[3] else _element(argument(1), spec)
            )
            try:
                packed = struct.pack(endian + spec[1], converted)
            except OverflowError:
                if spec[1] != "f" or not math.isfinite(converted):
                    raise
                packed = struct.pack(endian + "f", math.copysign(math.inf, converted))
            buffer["bytes"][offset : offset + spec[0]] = packed
            return ["undefined"]
        val = struct.unpack_from(endian + spec[1], bytes(buffer["bytes"]), offset)[0]
        return ["bigint", str(val)] if spec[2] else _tag_number(val)
    if method == "TypedArray.prototype.subarray":
        if target["kind"] not in ("typed-array", "shared-int32-view"):
            _fail("subarray receiver")
        buffer = _node(machine, target["bufferId"])
        full = _effective_length(buffer, target)

        def clip(n):
            return max(0, full + n) if n < 0 else min(full, n)

        begin = clip(_integer(argument(0)))
        end = (
            full if len(args) < 2 or argument(1)[0] == "undefined" else clip(_integer(argument(1)))
        )
        width = TYPE_INFO[target["elementType"]][0]
        offset = target["byteOffset"] + begin * width
        length = (
            None
            if (len(args) < 2 or argument(1)[0] == "undefined") and target["lengthTracking"]
            else max(0, end - begin)
        )
        nlen = (
            _effective_length(
                buffer,
                {
                    "elementType": target["elementType"],
                    "byteOffset": offset,
                    "lengthTracking": length is None,
                    "length": length or 0,
                },
            )
            if length is None
            else length
        )
        node = {
            "kind": target["kind"],
            "bufferId": target["bufferId"],
            "elementType": target["elementType"],
            "byteOffset": offset,
            "length": nlen,
            "lengthTracking": length is None,
        }
        _alloc(machine, record["name"], node)
        return ["ref", record["name"]]
    if method == "SharedArrayBuffer.prototype.grow":
        if target["kind"] != "shared-int32-buffer" or not target["growable"]:
            _fail("growable SharedArrayBuffer required")
        n = _index(argument(0))
        if n < target["byteLength"] or n > target["maxByteLength"]:
            _fail("shared growth range")
        target["bytes"].extend([0] * (n - target["byteLength"]))
        target["byteLength"] = n
        return ["undefined"]
    shared = method == "SharedArrayBuffer.prototype.slice"
    expected = "shared-int32-buffer" if shared else "array-buffer"
    if target["kind"] != expected:
        _fail("buffer slice receiver")

    def clip(n):
        return max(0, target["byteLength"] + n) if n < 0 else min(target["byteLength"], n)

    begin = clip(_integer(argument(0)))
    end = (
        target["byteLength"]
        if len(args) < 2 or argument(1)[0] == "undefined"
        else clip(_integer(argument(1)))
    )
    length = max(0, end - begin)
    result = _buffer(expected, record["name"], length)
    if shared:
        result["growable"] = False
    result["bytes"] = target["bytes"][begin : begin + length]
    _alloc(machine, record["name"], result)
    return ["ref", record["name"]]


def apply_record(machine, record):
    if not validate_record(record):
        return False
    kind = record["kind"]
    if kind == "memory.construct":
        _construct(machine, record)
    elif kind == "memory.get":
        expected = _read_property(machine, record["object"], record["key"][1])
        if expected != record["result"]:
            _fail("false binary-memory read claim")
    elif kind == "memory.set":
        _set_property(machine, record["object"], record["key"][1], record["value"])
    else:
        expected = _call(machine, record)
        if expected != record["result"]:
            _fail("false binary-memory call result")
    validate_state(machine)
    return True


def validate_state(machine):
    heap = machine["heap"]
    ids = {
        identity
        for identity, node in heap.items()
        if node.get("kind")
        in (
            "array-buffer",
            "shared-int32-buffer",
            "typed-array",
            "shared-int32-view",
            "data-view",
            "shared-data-view",
        )
    }
    for identity in ids:
        node = heap[identity]
        if (
            "id" in node
            or "version" in node
            or set(node)
            - (
                {
                    "properties",
                    "kind",
                    "byteLength",
                    "maxByteLength",
                    "resizable",
                    "detached",
                    "growable",
                    "bytes",
                    "bufferId",
                    "byteOffset",
                    "lengthTracking",
                    "length",
                    "elementType",
                }
            )
        ):
            _fail("binary-memory public node fields")
        if node.get("properties") != {}:
            _fail("binary-memory ordinary properties outside profile")
        if node.get("kind") in ("array-buffer", "shared-int32-buffer"):
            fields = {"kind", "byteLength", "maxByteLength", "bytes", "properties"} | (
                {"resizable", "detached"} if node["kind"] == "array-buffer" else {"growable"}
            )
            if set(node) != fields:
                _fail("buffer node exact fields")
            n = node["byteLength"]
            m = node["maxByteLength"]
            if (
                type(n) is not int
                or n < 0
                or n > 1024
                or type(m) is not int
                or m < n
                or m > 1024
                or not isinstance(node["bytes"], list)
                or len(node["bytes"]) != n
                or any(type(x) is not int or x < 0 or x > 255 for x in node["bytes"])
            ):
                _fail("buffer bytes/length")
            if node["kind"] == "array-buffer" and (
                node["resizable"] is not False or node["detached"] is not False or m != n
            ):
                _fail("ArrayBuffer capability")
            if node["kind"] == "shared-int32-buffer" and (
                type(node["growable"]) is not bool or not node["growable"] and m != n
            ):
                _fail("shared buffer capability")
        else:
            typed = node["kind"] in ("typed-array", "shared-int32-view")
            shared = node["kind"].startswith("shared-")
            fields = {"kind", "bufferId", "byteOffset", "lengthTracking", "properties"} | (
                {"elementType", "length"} if typed else {"byteLength"}
            )
            if set(node) != fields:
                _fail("view node exact fields")
            backing = heap.get(node["bufferId"])
            expected = "shared-int32-buffer" if shared else "array-buffer"
            if not backing or backing["kind"] != expected:
                _fail("view backing identity")
            offset = node["byteOffset"]
            if (
                type(offset) is not int
                or offset < 0
                or offset > backing["byteLength"]
                or type(node["lengthTracking"]) is not bool
                or node["lengthTracking"]
                and (not shared or not backing.get("growable"))
            ):
                _fail("view offset/tracking")
            width = (
                TYPE_INFO[node["elementType"]][0]
                if typed and node["elementType"] in TYPE_INFO
                else 1
            )
            length = node["length"] if typed else node["byteLength"]
            if typed and (
                node["kind"] == "shared-int32-view"
                and node["elementType"] not in TYPE_INFO
                or node["kind"] == "typed-array"
                and node["elementType"] not in TYPE_INFO
            ):
                _fail("typed element type")
            if (
                type(length) is not int
                or length < 0
                or offset % width
                or offset + length * width > backing["byteLength"]
            ):
                _fail("view bounds")
    return {"valid": True, "objects": len(ids)}
