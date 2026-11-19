"""Public-prefix checks for semantic execution/8.

This module intentionally distinguishes public replay engineering checks from
source-bound production admission. It never promotes a document whose selected
shared continuation and private source correspondence were not independently
replayed by the full native controller.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

from .algorithm_v7 import stable_algorithm_hash
from .algorithm_v8 import replay_algorithm_line
from .detailed_v7 import parse_record
from .detailed_v7 import value as validate_value
from .semantic_contract_v8 import CONTRACT_DIGEST, SemanticContractV8Error, validate_public


class NativeV8Error(ValueError): pass


def _fail(msg): raise NativeV8Error(msg)


def _exact(obj, keys, label):
    if not isinstance(obj, Mapping) or set(obj)!=set(keys): _fail(f"Malformed v8 {label}")


def validate_v8_artifact(document):
    """Check public v8 versions, causal contexts, and each available pure target.

    Returns a cautious receipt: detailed controller continuation and source
    correspondence are explicitly untested, so this is never serving admission.
    """
    if not isinstance(document,dict) or not isinstance(document.get("public"),dict): _fail("Expected semantic-execution/8 public artifact")
    public=document["public"]
    try: timeline=validate_public(public)
    except SemanticContractV8Error as e: raise NativeV8Error(str(e)) from e
    contexts=document.get("contexts")
    if type(contexts) is not list: _fail("V8 contexts must be an ordered public list")
    generated=[(i,row) for i,row in enumerate(timeline) if row["mode"]=="generated"]
    if len(contexts)!=len(generated): _fail("Every generated v8 target requires one earlier context")
    programs={}; invocations={}; generated_count=0; algorithm_count=0; detail_count=0; region_count=0
    for context,(index,row) in zip(contexts,generated,strict=True):
        _exact(context,{"schema","representation","profile","invocationId","programs","input","state","regionCursor","obligations","visibleThrough","prefixDigest"},"context")
        if context["schema"]!="semantic-context/8" or context["representation"]!="mixed" or context["profile"]!="compositional-regions/1": _fail("V8 context version")
        cutoff=context["visibleThrough"]
        if type(cutoff) is not int or cutoff<1 or cutoff>row["sequence"]-1 or context["prefixDigest"]!=stable_algorithm_hash(timeline[:cutoff]): _fail("Context includes future information or has a false prefix digest")
        if context["invocationId"]!=row["fields"].get("invocationId"): _fail("Context invocation mismatch")
        state=context["state"]
        if type(state) is not dict or set(state)!=set(__import__("megatoken.data.semantic_contract_v8",fromlist=["DEFINITION"]).DEFINITION["sharedStateFields"]): _fail("V8 shared continuation fields")
        for name in ("heap","locals","stack","effects","jobs","ready","captureAliases","continuations","frameFunctions","jobErrors"):
            if not isinstance(state[name],list): _fail("V8 continuation ordered table")
        if any(not isinstance(row,list) or len(row)!=2 or not isinstance(row[0],str) for key in ("locals","captureAliases","continuations","frameFunctions") for row in state[key]): _fail("V8 continuation identity table")
        try:
            for _,tagged in state["locals"]: validate_value(tagged)
            validate_value(state["receiver"])
        except (TypeError,ValueError) as e: raise NativeV8Error(f"Malformed tagged v8 continuation value: {e}") from e
        if not isinstance(context["regionCursor"],dict) or set(context["regionCursor"])!={"active","consumed"} or not isinstance(context["regionCursor"]["consumed"],list): _fail("V8 region cursor")
        if not isinstance(context["obligations"],dict) or set(context["obligations"])!={"intents","handlers","callbacks","closureQueue"} or any(not isinstance(v,list) for v in context["obligations"].values()): _fail("V8 obligations")
        if not isinstance(context["programs"],list) or not isinstance(context["input"],dict) or not isinstance(context["obligations"],dict): _fail("V8 context publication/state shape")
        for published in context["programs"]:
            if type(published) is not dict or set(published)!={"id","program"}: _fail("Program publication")
            program=published["program"]
            if type(program) is not dict or program.get("schema")!="semantic-region-program/1": _fail("Region program version")
            if program.get("sourceDigest")!=stable_algorithm_hash(program.get("source")) or program.get("traceFormat")!="semantic-trace/8" or program.get("profile")!="compositional-regions/1": _fail("Program source or profile digest")
            if program.get("definitionDigest")!=stable_algorithm_hash({k:v for k,v in program.items() if k!="definitionDigest"}): _fail("Region definition digest")
            programs[published["id"]]=program
        fields=row["fields"]
        if row["kind"] in ("algorithm","region","semantic"):
            # A target must follow a program and invocation publication inside
            # the exact model-visible prefix. No importer cache can satisfy it.
            if not context["programs"]: _fail("Generated target has no publicly visible region definition")
            program=context["programs"][-1]["program"]
            if row["kind"]=="algorithm":
                core=program.get("algorithm")
                if core is None: _fail("Algorithm target lacks its published core")
                graph=context["input"].get("graph")
                if not isinstance(graph,dict): _fail("Algorithm target lacks a public input graph")
                replay_algorithm_line(core,graph,fields.get("line")); algorithm_count+=1
            elif row["kind"]=="region":
                _replay_region(program,context,fields.get("line"),fields.get("entry")); region_count+=1
            else:
                line=fields.get("line")
                if not isinstance(line,str) or not line.startswith("detail "): _fail("V8 semantic record must carry a checked detailed record")
                try: parse_record(line)
                except ValueError as e: raise NativeV8Error(f"Invalid v8 detail record: {e}") from e
                detail_count+=1
            generated_count+=1
    return {"schema":"semantic-native-admission/8","contractDigest":CONTRACT_DIGEST,
            "checks":{"replay":"passed" if generated_count and algorithm_count+region_count==generated_count else "unsupported",
                      "contexts":"passed","continuation":"untested","masks":"untested",
                      "sourceBinding":"untested","tokenizer":"untested"},
            "counts":{"contexts":len(contexts),"generatedRecords":generated_count,"algorithmRecords":algorithm_count,"regionRecords":region_count,"detailedRecords":detail_count},
            "admitted":False}


def _replay_region(program,context,line,entry):
    if not isinstance(line,str) or not line.startswith("region "): _fail("V8 region prefix")
    try: target=json.loads(line[len("region "):])
    except json.JSONDecodeError as e: raise NativeV8Error("Malformed region JSON") from e
    _exact(target,{"region","bindings","decisions","completion"},"region target")
    if not isinstance(entry,dict) or set(entry)!={"definition","region","regionDigest","visibleThrough","stateDigest"}: _fail("Trusted region entry fields")
    if entry["definition"]!=program["definitionDigest"] or entry["stateDigest"]!=stable_algorithm_hash(context["state"]) or entry["visibleThrough"]!=context["visibleThrough"]: _fail("Stale or false region entry")
    definition=next((r for r in program["regions"] if r.get("id")==target["region"]),None)
    if definition is None or entry["region"]!=definition["id"] or entry["regionDigest"]!=definition["digest"]: _fail("Unknown region definition")
    local_values={name:value for name,value in context["state"].get("locals",[])}
    locals_={name:local_values.get(cell) for name,cell in context["state"].get("captureAliases",[]) if cell in local_values}
    locals_.update({name:value for name,value in local_values.items() if name not in locals_})
    values={}; bindings=[]; decisions=[]; completion={"kind":definition["completion"]}
    def term(node):
        op=node.get("op")
        if op=="literal": return node["value"]
        if op=="local":
            name=node["name"]
            if name not in values and name not in locals_: _fail("Unavailable local at region entry")
            return values.get(name,locals_.get(name))
        if op=="select":
            c=term(node["test"]); truth=bool(c[1]) if c[0]=="bool" else c[0] not in ("null","undefined") and c[1]!=0 and c[1]!=""
            decisions.append({"condition":c,"selected":bool(truth)})
            return term(node["consequent"] if truth else node["alternate"])
        if op=="logical":
            left=term(node["left"]); operator=node["operator"]
            truth=bool(left[1]) if left[0]=="bool" else left[0] not in ("null","undefined") and left[1]!=0 and left[1]!=""
            if (operator=="&&" and not truth) or (operator=="||" and truth) or (operator=="??" and left[0] not in ("null","undefined")): return left
            return term(node["right"])
        if op in ("binary","unary"):
            operator=node["operator"]
            mappings={"+":"add","-":"subtract","*":"multiply","/":"divide","%":"remainder","**":"power","===":"equal","!==":"not_equal","<":"less","<=":"less_equal",">":"greater",">=":"greater_equal","&":"bit_and","|":"bit_or","^":"bit_xor","<<":"shift_left",">>":"shift_right",">>>":"unsigned_shift_right","!":"not","~":"bit_not","typeof":"type_of","void":"void"}
            operation=mappings.get(operator)
            if operation is None: _fail("Unsupported local primitive operator")
            from . import algorithm_v7 as v7
            args=[term(node["argument"])] if op=="unary" else [term(node["left"]),term(node["right"])]
            return v7._operation(operation,args)
        _fail("Unsupported local term")
    for op in definition["operations"]:
        if op["kind"]=="bind":
            value=term(op["term"]); values[op["binding"]]=value; bindings.append([op["binding"],value])
        elif op["kind"]=="return": completion={"kind":"return","value":term(op["term"])}
        else: _fail("Unsupported compact region operation")
    expected={"region":definition["id"],"bindings":bindings,"decisions":decisions,"completion":completion}
    if target!=expected: _fail("Region target disagrees with independent local replay")
