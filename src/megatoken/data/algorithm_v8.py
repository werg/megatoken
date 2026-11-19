"""Independent evaluator for the pure semantic-algorithm/2 language.

This deliberately does not call JavaScript or the producer.  It shares only
the established, separately implemented ECMAScript primitive operations and
tagged-value/heap rules from algorithm v7.
"""
from __future__ import annotations

import copy
import json
from typing import Any

from . import algorithm_v7 as v7

VERSION = "semantic-algorithm/2"
PROFILE = "ecmascript-pure-regions/1"
TRAVERSAL_PROFILE = "dense-primitive-array/1"
MAX_ITEMS = 128
MAX_NODES = 2048


class AlgorithmV8Error(ValueError):
    """Malformed or semantically unsupported algorithm/2 input."""


def _fail(reason: str):
    raise AlgorithmV8Error(f"Invalid {VERSION}: {reason}")


def _exact(obj, required, optional=()):
    if type(obj) is not dict or not set(required) <= set(obj) or set(obj) - set(required) - set(optional):
        _fail("record fields")


def validate_term(term: Any, *, scopes=(), inputs=None, values=None, budget=None):
    """Validate exact term shapes and lexical scopes; return node count."""
    remaining = [MAX_NODES] if budget is None else budget
    active = set()

    def visit(t, depth=0, lexical=scopes):
        if type(t) is not dict or depth > 64 or id(t) in active:
            _fail("term shape, cycle, or depth")
        remaining[0] -= 1
        if remaining[0] < 0:
            _fail("term node budget")
        active.add(id(t))
        op = t.get("op")
        if op == "literal":
            _exact(t, ("op", "value")); v7._validate_value(t["value"])
        elif op in ("input", "ref"):
            _exact(t, ("op", "id")); v7._identity(t["id"])
            allowed = inputs if op == "input" else values
            if allowed is not None and t["id"] not in allowed: _fail("unbound name")
        elif op == "bound":
            _exact(t, ("op", "depth", "index"))
            if type(t["depth"]) is not int or type(t["index"]) is not int or t["depth"] < 0 or t["depth"] >= len(lexical) or t["index"] < 0 or t["index"] >= lexical[t["depth"]]: _fail("bound reference")
        elif op == "project":
            _exact(t, ("op", "object", "key"))
            if type(t["key"]) is not str or t["key"] == "__proto__": _fail("property key")
            visit(t["object"], depth+1, lexical)
        elif op == "get":
            _exact(t, ("op", "object", "key")); visit(t["object"], depth+1, lexical); visit(t["key"], depth+1, lexical)
        elif op == "default_undefined":
            _exact(t, ("op", "args"))
            if type(t["args"]) is not list or len(t["args"]) != 2: _fail("default arity")
            for a in t["args"]: visit(a, depth+1, lexical)
        elif op in ("array", "object"):
            field = "items" if op == "array" else "properties"
            _exact(t, ("op", field)); seq=t[field]
            if type(seq) is not list or len(seq)>MAX_ITEMS: _fail("allocation shape/budget")
            if op == "array":
                for a in seq: visit(a,depth+1,lexical)
            else:
                seen=set()
                for p in seq:
                    _exact(p,("key","value"))
                    if type(p["key"]) is not str or p["key"] == "__proto__" or p["key"] in seen: _fail("object key")
                    seen.add(p["key"]); visit(p["value"],depth+1,lexical)
        elif op == "let":
            _exact(t,("op","bindings","body")); binds=t["bindings"]
            if type(binds) is not list or not binds or len(binds)>128: _fail("let bindings")
            prior = 0
            for i,b in enumerate(binds):
                _exact(b,("id","value")); v7._identity(b["id"])
                if sum(1 for x in binds[:i] if x["id"]==b["id"]): _fail("duplicate binding")
                visit(b["value"],depth+1,(prior,*lexical))
                prior += 1
            visit(t["body"],depth+1,(len(binds),*lexical))
        elif op == "traverse":
            _exact(t,("op","mode","input","initial","callback","profile","order"))
            if t["mode"] not in {"fold","scan","map","filter","find","find_index","some","every"} or t["profile"]!=TRAVERSAL_PROFILE or t["order"]!="ascending": _fail("traversal profile")
            visit(t["input"],depth+1,lexical)
            if t["mode"] in {"fold","scan"}:
                if t["initial"] is None: _fail("fold requires an explicit initial value")
                visit(t["initial"],depth+1,lexical)
            elif t["initial"] is not None: _fail("non-fold initial")
            cb=t["callback"]; _exact(cb,("arity","body")); arity=4 if t["mode"] in {"fold","scan"} else 3
            if cb["arity"]!=arity: _fail("callback arity")
            visit(cb["body"],depth+1,(arity,*lexical))
        else:
            if op not in v7.OPERATIONS: _fail("unknown operation")
            _exact(t,("op","args")); args=t["args"]
            if type(args) is not list or len(args)!=v7.OPERATIONS[op][0]: _fail("operation arity")
            for a in args: visit(a,depth+1,lexical)
        active.remove(id(t))
    visit(term)
    return MAX_NODES-remaining[0]


def validate_algorithm(core):
    _exact(core,("schema","profile","inputs","steps"))
    if core["schema"]!=VERSION or core["profile"]!=PROFILE: _fail("version/profile")
    if type(core["inputs"]) is not list or type(core["steps"]) is not list or len(core["inputs"])>128 or not core["steps"] or len(core["steps"])>256: _fail("core budget")
    ins=set(); vals=set(); allids=set()
    types={"value","scalar","number","dense-number-array","dense-primitive-array","data","nullable-data"}
    for row in core["inputs"]:
        _exact(row,("id","type")); v7._identity(row["id"])
        if row["id"] in allids or row["type"] not in types: _fail("input declaration")
        allids.add(row["id"]); ins.add(row["id"])
    done=False
    for step in core["steps"]:
        if type(step) is not dict: _fail("step")
        if step.get("kind")=="compute":
            _exact(step,("kind","id","term")); v7._identity(step["id"])
            if done or step["id"] in allids: _fail("duplicate/late compute")
            validate_term(step["term"],inputs=ins,values=vals)
            allids.add(step["id"]); vals.add(step["id"])
        elif step.get("kind") in ("return","throw"):
            _exact(step,("kind","value"))
            if done: _fail("duplicate completion")
            validate_term(step["value"],inputs=ins,values=vals); done=True
        else: _fail("pure core step kind")
    if not done: _fail("missing completion")
    return True


def _primitive_array(value, heap):
    if value[0] != "ref": _fail("traverse input identity")
    node, props = v7._heap_node(heap, value[1])
    if node.get("kind") != "array": _fail("traverse input array")
    length=props.get("length")
    if not length or length[0]!="num" or type(length[1]) not in (int,float) or int(length[1])!=length[1] or not 0<=length[1]<=MAX_ITEMS: _fail("array length")
    count=int(length[1]); result=[]
    for i in range(count):
        item=props.get(str(i))
        if item is None: _fail("sparse array")
        v7._validate_value(item)
        if item[0] == "ref": _fail("non-primitive traversal item")
        result.append(item.copy())
    if set(props)-{"length",*(str(i) for i in range(count))}: _fail("extra array property")
    return result


def _ordered_properties(properties):
    indexed=[]; ordinary=[]
    for key,value in properties.items():
        idx=v7._array_index(key)
        if idx is None: ordinary.append((key,value))
        else: indexed.append((idx,key,value))
    return {key:value for _,key,value in sorted(indexed)} | dict(ordinary)


def evaluate_term(term, *, inputs=None, values=None, heap=None, allocator=None, scopes=(), budget=None):
    inputs=inputs or {}; values=values or {}; heap=v7._snapshot_heap(heap); allocator=allocator if allocator is not None else {"prefix":"v8","next":0}; budget=budget if budget is not None else {"remaining":32768}
    if type(budget) is not dict or set(budget)!={"remaining"} or type(budget["remaining"]) is not int or not 0<=budget["remaining"]<=32768: _fail("semantic work budget")
    validate_term(term, scopes=scopes)
    def committed(value):
        v7._validate_value(value)
        if value[0]=="ref": v7._heap_node(heap,value[1])
        return value.copy()
    def alloc(kind, props):
        while True:
            if allocator["next"]>=1024: _fail("allocation budget")
            ident=f"{allocator['prefix']}:{allocator['next']}"; allocator['next']+=1
            if ident not in heap: break
        heap[ident]={"kind":kind,"properties":copy.deepcopy(props)}
        return ["ref",ident]
    def visit(t, env_inputs=inputs, env_values=values, env_scopes=scopes):
        budget["remaining"]-=1
        if budget["remaining"]<0: _fail("semantic work budget")
        op=t["op"]
        if op=="literal": return committed(t["value"])
        if op in ("input","ref"):
            src=env_inputs if op=="input" else env_values
            v=src.get(t["id"])
            if v is None: _fail("uncommitted dependency")
            return committed(v)
        if op=="bound": return env_scopes[-1-t["depth"]][t["index"]].copy()
        if op in ("project","get"):
            obj=visit(t["object"],env_inputs,env_values,env_scopes)
            key=t["key"] if op=="project" else v7._to_string(visit(t["key"],env_inputs,env_values,env_scopes))
            if obj[0]=="str":
                if key=="length": return ["num",len(obj[1].encode("utf-16-le","surrogatepass"))//2]
                if key.isascii() and key.isdecimal() and (key=="0" or not key.startswith("0")):
                    units=obj[1].encode("utf-16-le","surrogatepass"); ix=int(key)
                    if ix<len(units)//2: return ["str",units[2*ix:2*ix+2].decode("utf-16-le","surrogatepass")]
                _fail("string property outside ordinary index profile")
            if obj[0]!="ref": _fail("ordinary own-data get")
            node,props=v7._heap_node(heap,obj[1])
            inherited={"__proto__","constructor","toString","toLocaleString","valueOf","hasOwnProperty","isPrototypeOf","propertyIsEnumerable","__defineGetter__","__defineSetter__","__lookupGetter__","__lookupSetter__"}
            if key not in props and (key in inherited or node["kind"]=="array" and key!="length" and not (key.isascii() and key.isdecimal() and (key=="0" or not key.startswith("0")))): _fail("inherited property requires detailed access")
            return committed(copy.deepcopy(props.get(key,["undefined"])))
        if op=="default_undefined":
            v=visit(t["args"][0],env_inputs,env_values,env_scopes); return visit(t["args"][1],env_inputs,env_values,env_scopes) if v[0]=="undefined" else v
        if op=="let":
            scope=[]
            for b in t["bindings"]: scope.append(visit(b["value"],env_inputs,env_values,(*env_scopes,scope)))
            return visit(t["body"],env_inputs,env_values,(*env_scopes,scope))
        if op=="array": return alloc("array",{**{str(i):visit(x,env_inputs,env_values,env_scopes) for i,x in enumerate(t["items"])},"length":["num",len(t["items"])]})
        if op=="object": return alloc("object",_ordered_properties({p["key"]:visit(p["value"],env_inputs,env_values,env_scopes) for p in t["properties"]}))
        if op=="traverse":
            input_value=visit(t["input"],env_inputs,env_values,env_scopes)
            items=_primitive_array(input_value,heap); mode=t["mode"]; cb=t["callback"]; acc=visit(t["initial"],env_inputs,env_values,env_scopes) if mode in ("fold","scan") else None; out=[]
            for i,item in enumerate(items):
                argv=[acc,item,["num",i],input_value] if mode in ("fold","scan") else [item,["num",i],input_value]
                budget["remaining"]-=1
                if budget["remaining"]<0: _fail("semantic work budget")
                result=visit(cb["body"],env_inputs,env_values,(*env_scopes,argv))
                if mode=="fold": acc=result
                elif mode=="scan": acc=result; out.append(acc.copy())
                elif mode=="map": out.append(result)
                elif mode=="filter" and v7._truthy(result): out.append(item)
                elif mode=="find" and v7._truthy(result): return item
                elif mode=="find_index" and v7._truthy(result): return ["num",i]
                elif mode=="some" and v7._truthy(result): return ["bool",True]
                elif mode=="every" and not v7._truthy(result): return ["bool",False]
            if mode=="fold": return acc
            if mode in ("scan","map","filter"): return alloc("array",{**{str(i):v for i,v in enumerate(out)},"length":["num",len(out)]})
            if mode=="find": return ["undefined"]
            if mode=="find_index": return ["num",-1]
            return ["bool",mode=="every"]
        if op in ("select","and","or","coalesce"):
            left=visit(t["args"][0],env_inputs,env_values,env_scopes)
            if op=="select": return visit(t["args"][1 if v7._truthy(left) else 2],env_inputs,env_values,env_scopes)
            take=(v7._truthy(left) if op=="and" else not v7._truthy(left) if op=="or" else left[0] in ("undefined","null"))
            return visit(t["args"][1],env_inputs,env_values,env_scopes) if take else left
        if op=="fold_left": return v7.evaluate_term(t,{"inputs":env_inputs,"values":env_values,"heap":heap})
        return v7._operation(op,[visit(a,env_inputs,env_values,env_scopes) for a in t["args"]])
    return visit(term),heap,allocator


def evaluate_algorithm(core, arguments, *, heap=None, allocator=None):
    """Evaluate a pure core and retain claims, aliases and allocated objects."""
    validate_algorithm(core)
    args = v7._data_list(arguments, "algorithm/2 arguments")
    if len(args) != len(core["inputs"]): _fail("input arity")
    actual_heap = v7._snapshot_heap(heap)
    inputs = {}
    for arg, declaration in zip(args, core["inputs"], strict=True):
        v7._validate_value(arg)
        kind = declaration["type"]
        if kind in ("scalar", "number", "dense-number-array") and arg[0] == "ref" and kind != "dense-number-array": _fail("scalar input type")
        if kind == "number" and arg[0] != "num": _fail("number input type")
        if kind == "data" and arg[0] != "ref": _fail("data input type")
        if kind == "nullable-data" and arg[0] not in ("ref", "null", "undefined"): _fail("nullable-data input type")
        if arg[0] == "ref": v7._heap_node(actual_heap,arg[1])
        if kind == "dense-number-array": v7._dense_items(arg,actual_heap)
        if kind == "dense-primitive-array": _primitive_array(arg,actual_heap)
        inputs[declaration["id"]]=arg.copy()
    allocator = allocator if allocator is not None else {"prefix":"v8","next":0}
    if type(allocator) is not dict or set(allocator)!={"prefix","next"} or type(allocator["prefix"]) is not str or type(allocator["next"]) is not int or allocator["next"]<0: _fail("allocator")
    v7._identity(allocator["prefix"],"allocator prefix")
    if allocator["next"]>1024: _fail("allocation budget")
    values={}; claims=[]; completion=None; budget={"remaining":32768}
    for index,step in enumerate(core["steps"]):
        if step["kind"]=="compute":
            result,actual_heap,allocator=evaluate_term(step["term"],inputs=inputs,values=values,heap=actual_heap,allocator=allocator,budget=budget)
            values[step["id"]]=result.copy(); claims.append({"step":index,"kind":"compute","id":step["id"],"result":result.copy()})
        else:
            result,actual_heap,allocator=evaluate_term(step["value"],inputs=inputs,values=values,heap=actual_heap,allocator=allocator,budget=budget)
            completion={"kind":step["kind"],"value":result.copy()}; claims.append({"step":index,**completion}); break
    graph={"roots":[*copy.deepcopy(list(inputs.values())),completion["value"].copy()],"objects":[]}
    for identity,node in actual_heap.items():
        row={"id":identity,"kind":node["kind"],"properties":[{"key":k,"value":copy.deepcopy(v)} for k,v in _ordered_properties(node["properties"]).items()]}
        if node.get("prototype",object()) is None: row["prototype"]=None
        graph["objects"].append(row)
    return {"completion":completion,"claims":claims,"graph":graph,"heap":actual_heap,"allocator":allocator,"work":32768-budget["remaining"]}


def replay_algorithm_line(core, input_graph, line):
    """Independently derive the public algorithm target from a public input graph."""
    validate_algorithm(core)
    if type(input_graph) is not dict or set(input_graph)!={"roots","objects"} or type(input_graph["roots"]) is not list:
        _fail("input graph envelope")
    heap=v7.algorithm_heap_from_graph(input_graph)
    result=evaluate_algorithm(core,input_graph["roots"],heap=heap,allocator={"prefix":"core-v8","next":0})
    if type(line) is not str or not line.startswith("algorithm "):
        _fail("generated algorithm prefix")
    try: supplied=json.loads(line[len("algorithm "):])
    except (json.JSONDecodeError,TypeError) as e: raise AlgorithmV8Error("Malformed algorithm/2 target JSON") from e
    if type(supplied) is not dict or set(supplied)!={"claims","allocations","completion"}: _fail("algorithm target fields")
    existing=set(heap); remap={}; next_id=0
    for identity in result["heap"]:
        if identity not in existing:
            while f"o{next_id}" in existing: next_id+=1
            remap[identity]=f"o{next_id}"; existing.add(f"o{next_id}"); next_id+=1
    def map_value(value):
        if type(value) is list:
            if len(value)==2 and value[0]=="ref" and value[1] in remap: return ["ref",remap[value[1]]]
            return [map_value(x) for x in value]
        if type(value) is dict: return {k:map_value(v) for k,v in value.items()}
        return value
    expected_claims=map_value(result["claims"])
    expected_completion=map_value(result["completion"])
    allocations=[]
    for identity,node in result["heap"].items():
        if identity in remap:
            allocations.append({"id":remap[identity],"kind":node["kind"],"properties":[{"key":k,"value":map_value(v)} for k,v in node["properties"].items()]})
    expected={"claims":expected_claims,"allocations":allocations,"completion":expected_completion}
    if supplied!=expected: _fail("algorithm target differs from independent replay")
    return expected

