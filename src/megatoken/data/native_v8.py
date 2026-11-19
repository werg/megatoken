"""Public-prefix checks for semantic execution/8.

This module intentionally distinguishes public replay engineering checks from
source-bound production admission. It never promotes a document whose selected
shared continuation and private source correspondence were not independently
replayed by the full native controller.
"""
from __future__ import annotations

import json
import copy
from collections.abc import Mapping

from .algorithm_v7 import stable_algorithm_hash
from .algorithm_v8 import replay_algorithm_line
from .detailed_v7 import parse_record
from .detailed_v7 import value as validate_value
from .semantic_contract_v8 import CONTRACT_DIGEST, SemanticContractV8Error, validate_public
from megatoken.controls import canonical_semantic_observation, semantic_selection_instruction
from .schema import Dependency, Episode, Lineage, Meridian, PhysicalRecord, Supervision, TrackEvent, TrackSpec, encode_episode
from .semantic_physical import physical_semantic_episode


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
            source_rows=[r["fields"] for r in timeline[:cutoff] if r["kind"]=="program" and r["fields"].get("programId")==published["id"]]
            if not source_rows or source_rows[-1].get("regionProgram")!=program: _fail("Context program was not previously published byte-for-byte")
            programs[published["id"]]=program
        prior_invocations=[r["fields"] for r in timeline[:cutoff] if r["kind"]=="invocation" and r["fields"].get("invocationId")==context["invocationId"]]
        if not prior_invocations or context["input"].get("graph")!=prior_invocations[-1].get("inputGraph"): _fail("Context input graph was not previously observed")
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
                try: kind,detail_fields=parse_record(line)
                except ValueError as e: raise NativeV8Error(f"Invalid v8 detail record: {e}") from e
                detail_count+=1
            generated_count+=1
    final_restored=_replay_shared_state(timeline,contexts,programs,roundtrip=True)
    final_uninterrupted=_replay_shared_state(timeline,contexts,programs,roundtrip=False)
    if final_restored!=final_uninterrupted: _fail("Checkpoint restoration changed the final shared continuation")
    return {"schema":"semantic-native-admission/8","contractDigest":CONTRACT_DIGEST,
            "checks":{"replay":"passed" if generated_count and algorithm_count+region_count+detail_count==generated_count else "unsupported",
                      "contexts":"passed","continuation":"passed","masks":"untested",
                      "sourceBinding":"untested","tokenizer":"untested"},
            "counts":{"contexts":len(contexts),"generatedRecords":generated_count,"algorithmRecords":algorithm_count,"regionRecords":region_count,"detailedRecords":detail_count},
            "admitted":False}


def prepare_native_v8_training(document,*,codec,max_positions=8192,max_history=1024):
    """Build the exact native physical rows, masks, token counts and digest."""
    validate_v8_artifact(document)
    public=document["public"]; timeline=public["timeline"]
    generated=[(i,row) for i,row in enumerate(timeline) if row["mode"]=="generated"]
    contexts=document["contexts"]
    rows=[]; context_for={row["sequence"]:context for context,(_,row) in zip(contexts,generated,strict=True)}
    for sequence,row in enumerate(timeline,1):
        if row["mode"]=="generated":
            context=context_for[sequence]
            rows.append((f"semantic:v8:context:{sequence}","system",canonical_semantic_observation(context),False))
            payload=row["fields"]["line"].encode("utf-8")
            rows.append((f"semantic:v8:target:{sequence}","assistant",payload,True))
        else:
            role="user" if row["kind"] in {"program","invocation"} else "environment"
            rows.append((f"semantic:v8:observation:{sequence}",role,canonical_semantic_observation(row),False))
    logical_records=[]; meridians=[]; ids={}
    for index,(identity,role,payload,masked) in enumerate(rows):
        rec=PhysicalRecord(identity,role,payload,turn_index=index,logical_id=identity,policy_mask=masked,
                           provenance="unknown" if masked else "executed")
        logical_records.append(rec); ids[identity]=identity
        meridians.append(Meridian(index,(TrackEvent(identity,"assistant","text",index,role,
            instruction=semantic_selection_instruction() if masked else ""),)))
    dependencies=[]
    for sequence,_ in generated:
        target=f"semantic:v8:target:{timeline[sequence]["sequence"]}"
        context=f"semantic:v8:context:{timeline[sequence]["sequence"]}"
        dependencies.append(Dependency(context,target,"v8-public-context"))
    digest=stable_algorithm_hash(public)
    episode=Episode("semantic:v8:"+digest[:24],tuple(logical_records),
        Lineage("semantic-execution-v8",digest,"semantic-execution-v8"),stream="semantic_execution",
        supervision=Supervision(annotations_json=json.dumps({"session_metadata":{"schema":"semantic-execution-metadata/8"},
            "semantic_execution":{"schema":"semantic-execution/8","representation":"mixed","profile":"compositional-regions/1"}},separators=(",",":"))),
        dependencies=tuple(dependencies),tracks=(TrackSpec("assistant","assistant"),),meridians=tuple(meridians))
    episode=physical_semantic_episode(episode,codec=codec,max_positions=max_positions,max_history=max_history,
        selection_instruction=lambda continuation=False:semantic_selection_instruction(continuation=continuation),emit_context_windows=False)
    encoded=encode_episode(episode,codec)
    token_rows=[]; generated_tokens=observed_tokens=0; masked_logicals=set(); unmasked_targets=set()
    for item in encoded:
        record=item.record; labels=[token if eligible else -100 for token,eligible in zip(item.token_ids,item.free_mask,strict=True)]
        if record.policy_mask:
            generated_tokens+=item.P; masked_logicals.add(record.logical_id)
        elif record.logical_id.startswith("semantic:v8:target:"): unmasked_targets.add(record.logical_id)
        else: observed_tokens+=item.tokens
        token_rows.append({"recordId":record.record_id,"logicalId":record.logical_id,"partIndex":record.part_index,
            "partCount":record.part_count,"role":record.role,"policyMask":record.policy_mask,
            "payloadSha256":__import__("hashlib").sha256(record.payload).hexdigest(),
            "tokenIds":list(item.token_ids),"labels":labels,"codecFingerprint":item.codec_fingerprint})
    native_digest=stable_algorithm_hash({"schema":"semantic-native-training/8","codecFingerprint":codec.fingerprint,"rows":token_rows})
    expected_targets={f"semantic:v8:target:{row['sequence']}" for row in timeline if row["mode"]=="generated"}
    if masked_logicals!=expected_targets or unmasked_targets: raise NativeV8Error("Native v8 physical policy masks do not match generated targets")
    return episode,{"nativeTrainingDigest":native_digest,"generatedTokens":generated_tokens,"observedTokens":observed_tokens,
                    "physicalRecords":len(episode.records),"logicalRecords":len(timeline)+len(contexts),"contexts":len(contexts),
                    "generatedRecords":len(generated),"eosTokenId":None,"terminalTokens":0,
                    "maskValidation":True,"framing":"native-physical-record-boundaries/1"}


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
            selected="consequent" if truth else "alternate"
            decisions.append({"kind":"select","selected":selected})
            return term(node[selected])
        if op=="logical":
            left=term(node["left"]); operator=node["operator"]
            truth=bool(left[1]) if left[0]=="bool" else left[0] not in ("null","undefined") and left[1]!=0 and left[1]!=""
            take_right=(operator=="&&" and truth) or (operator=="||" and not truth) or (operator=="??" and left[0] in ("null","undefined"))
            decisions.append({"kind":operator,"selected":"right" if take_right else "left"})
            return term(node["right"]) if take_right else left
        if op in ("binary","unary"):
            operator=node["operator"]
            mappings={"+":"add","-":"subtract","*":"multiply","/":"divide","%":"remainder","**":"power","===":"equal","!==":"not_equal","==":"loose_equal","!=":"loose_not_equal","<":"less","<=":"less_equal",">":"greater",">=":"greater_equal","&":"bit_and","|":"bit_or","^":"bit_xor","<<":"shift_left",">>":"shift_right",">>>":"unsigned_shift_right","!":"not","+u":"positive","-u":"negate","~":"bit_not","typeof":"type_of","void":"void"}
            operation=mappings.get(operator)
            if operation is None: _fail("Unsupported local primitive operator")
            from . import algorithm_v7 as v7
            args=[term(node["arg"])] if op=="unary" else [term(node["left"]),term(node["right"])]
            return v7._operation(operation,args)
        _fail("Unsupported local term")
    for op in definition["operations"]:
        if op["kind"]=="bind":
            value=term(op["term"]); values[op["binding"]]=value; bindings.append([op["binding"],value])
        elif op["kind"]=="return": completion={"kind":"return","value":term(op["term"])}
        else: _fail("Unsupported compact region operation")
    expected={"region":definition["id"],"bindings":bindings,"decisions":decisions,"completion":completion}
    if target!=expected: _fail("Region target disagrees with independent local replay")
    return target


def _initial_state(program,invocation,functions=()):
    graph=invocation.get("inputGraph")
    if not isinstance(graph,dict) or set(graph)!={"roots","objects"}: _fail("Invocation input graph")
    heap=[]
    for node in graph["objects"]:
        row={"kind":node["kind"],"prototype":"ordinary","properties":[[p["key"],p["value"]] for p in node["properties"]]}
        heap.append([node["id"],row])
    fn=next((f for f in functions if f.get("name")==invocation.get("entry")),{})
    params=fn.get("params",[]); locals_=[]; aliases=[]
    for index,value in enumerate(graph["roots"]):
        cell=f"a0b{index}"; locals_.append([cell,value])
        if index<len(params): aliases.append([params[index],cell])
    for name,index in invocation.get("bindings",{}).items():
        if type(index) is int and 0<=index<len(graph["roots"]) and name not in dict(aliases): aliases.append([name,f"a0b{index}"])
    return {"heap":heap,"locals":locals_,"stack":[invocation["invocationId"]],"effects":[],"jobs":[],"ready":[],
            "randomState":None,"pending":None,"lastReturn":None,"completion":None,"receiver":invocation.get("receiver",["undefined"]),
            "captureAliases":aliases,"continuations":[],"reactionSequence":0,"blockedSync":None,"callReturn":None,
            "activeJob":None,"scheduler":{"registration":0,"ready":0},
            "frameFunctions":[[invocation["invocationId"],invocation["entry"]]],"jobErrors":[]}


def _replay_shared_state(timeline,contexts,programs_from_context,*,roundtrip):
    snapshots={row["sequence"]:ctx for ctx,(_,row) in zip(contexts,[(i,r) for i,r in enumerate(timeline) if r["mode"]=="generated"],strict=True)}
    published={}; state=None; program=None; dispatched=set(); settled=set()
    for row in timeline:
        kind=row["kind"]; fields=row["fields"]
        if kind=="program":
            program=fields.get("regionProgram")
            if not isinstance(program,dict): _fail("Program record omits its immutable region definition")
            if fields.get("digest")!=program.get("sourceDigest") or program.get("source")!=fields.get("source"): _fail("Public region definition differs from immutable source publication")
            published[fields["programId"]]=(program,fields.get("functions",[]))
            continue
        if kind=="invocation":
            published_row=published.get(fields["programId"])
            if published_row is None: _fail("Invocation has no preceding program")
            program,functions=published_row
            if fields.get("sourceDigest")!=program.get("sourceDigest"): _fail("Invocation source/definition binding")
            state=_initial_state(program,fields,functions)
            continue
        if state is None: continue
        if row["mode"]=="generated":
            context=snapshots.get(row["sequence"])
            if context is None or context["visibleThrough"]!=row["sequence"]-1: _fail("Generated prefix cutoff is not immediately causal")
            if state!=context["state"]: _fail("Public continuation differs from independently replayed shared state")
            if roundtrip:
                state=_restore_native_checkpoint(_native_checkpoint(timeline[:context["visibleThrough"]],context),
                                                 expected_prefix=timeline[:context["visibleThrough"]],expected_state=context["state"])
            program=context["programs"][-1]["program"]
            if kind=="region":
                target=_replay_region(program,context,fields["line"],fields["entry"]); _apply_region(state,target)
            elif kind=="algorithm": _apply_algorithm(state,program["algorithm"],context["input"]["graph"],fields["line"])
            elif kind=="semantic":
                k,payload=parse_record(fields["line"]); _apply_detail(state,k,payload,row)
        elif kind in ("dispatch","result"):
            _apply_observation(state,row,dispatched,settled)
    return state


def _native_checkpoint(prefix,context):
    checkpoint={"schema":"semantic-native-checkpoint/8","profile":"compositional-regions/1",
                "visibleThrough":context["visibleThrough"],"prefix":copy.deepcopy(prefix),
                "state":copy.deepcopy(context["state"]),"regionCursor":copy.deepcopy(context["regionCursor"]),
                "obligations":copy.deepcopy(context["obligations"])}
    checkpoint["integrity"]=stable_algorithm_hash(checkpoint)
    return json.loads(json.dumps(checkpoint,separators=(",",":"),ensure_ascii=False))


def _restore_native_checkpoint(checkpoint,*,expected_prefix,expected_state):
    required={"schema","profile","visibleThrough","prefix","state","regionCursor","obligations","integrity"}
    if not isinstance(checkpoint,dict) or set(checkpoint)!=required: _fail("Native v8 checkpoint fields")
    content={k:v for k,v in checkpoint.items() if k!="integrity"}
    if checkpoint["integrity"]!=stable_algorithm_hash(content): _fail("Native v8 checkpoint integrity")
    if checkpoint["schema"]!="semantic-native-checkpoint/8" or checkpoint["profile"]!="compositional-regions/1": _fail("Native v8 checkpoint version")
    if type(checkpoint["visibleThrough"]) is not int or checkpoint["visibleThrough"]!=len(checkpoint["prefix"]): _fail("Native v8 checkpoint cutoff")
    if checkpoint["prefix"]!=expected_prefix or checkpoint["state"]!=expected_state: _fail("Native v8 checkpoint differs from verified replay boundary")
    expected=set(__import__("megatoken.data.semantic_contract_v8",fromlist=["DEFINITION"]).DEFINITION["sharedStateFields"])
    if not isinstance(checkpoint["state"],dict) or set(checkpoint["state"])!=expected: _fail("Native v8 checkpoint dropped shared state")
    return copy.deepcopy(checkpoint["state"])


def _set_row(rows,key,value):
    for row in rows:
        if row[0]==key: row[1]=value; return
    rows.append([key,value])


def _table(state,field): return {key:value for key,value in state[field]}


def _apply_region(state,target):
    for name,value in target["bindings"]: _set_row(state["locals"],name,value)
    if target["completion"]["kind"]=="return":
        state["pending"]={"kind":"return","value":target["completion"]["value"]}; state["stack"]=[]


def _apply_algorithm(state,core,graph,line):
    expected=json.loads(line[10:]); heap={key:value for key,value in _table(state,"heap").items()}
    for allocation in expected["allocations"]:
        _set_row(state["heap"],allocation["id"],{"kind":allocation["kind"],"code":None,"captures":[],"prototype":"ordinary","properties":[[p["key"],p["value"]] for p in allocation["properties"]]})
    state["pending"]={"kind":expected["completion"]["kind"],"value":expected["completion"]["value"]}
    state["stack"]=[]


def _apply_detail(state,kind,fields,row):
    if kind=="bind":
        alias=dict(state["captureAliases"]).get(fields["name"])
        _set_row(state["locals"],alias or fields["name"],fields["value"])
    elif kind=="allocate":
        node={"kind":fields["kind"],"code":None,"captures":[],"prototype":"ordinary","properties":[[p["key"],p["value"]] for p in fields["properties"]]}
        if "code" in fields: node["code"]=fields["code"]
        if "captures" in fields: node["captures"]=fields["captures"]
        if "length" in fields: node["properties"].append(["length",["num",fields["length"]]])
        _set_row(state["heap"],fields["name"],node)
    elif kind=="write":
        heap=_table(state,"heap"); node=heap.get(fields["object"])
        if node is None: _fail("Write to absent object")
        props=dict(node["properties"])
        if fields["present"]: props[fields["key"]]=fields["value"]
        else: props.pop(fields["key"],None)
        node["properties"]=[[k,v] for k,v in props.items()]
    elif kind=="call":
        state["stack"].append(fields["child"]); state["frameFunctions"].append([fields["child"],fields["callee"]])
    elif kind=="return":
        value=fields["value"]; invocation=fields["invocation"]
        if len(state["stack"])>1 and state["stack"][-1]==invocation:
            state["lastReturn"]={"kind":"return","value":value}; state["stack"].pop()
        else:
            state["pending"]={"kind":"return","value":value}; state["stack"]=[]
    elif kind=="end":
        if state["stack"]: state["stack"].pop()
        state["completion"]=state["pending"] or {"kind":fields["kind"]}
        state["pending"]=None
    elif kind=="request":
        effect=fields["effect"]; promise=fields.get("promise")
        args=fields["arguments"]
        from . import algorithm_v7 as v7
        decoded=[v7._decode_scalar(v) for v in args]
        try: delivered=json.loads(fields["payload"])
        except (TypeError,json.JSONDecodeError) as e: raise NativeV8Error("Malformed host request payload") from e
        if delivered!=decoded: _fail("Host request payload differs from its tagged arguments")
        row_effect={"operation":fields["operation"],"invocation":fields["invocation"],"site":fields["site"],"payload":fields["payload"],"arguments":decoded,"argumentCodec":"canonical-json/1","status":"requested","fixture":None,"promise":promise}
        state["effects"].append([effect,row_effect])
        if promise: _set_row(state["heap"],promise,{"kind":"promise","properties":[],"state":"pending","result":None,"adopted":None,"external":effect,"reactions":[]})
    elif kind=="await":
        seq=state["scheduler"]["registration"]+1; state["scheduler"]["registration"]=seq
        _set_row(state["continuations"],fields["continuation"],{"state":"waiting","promise":fields["promise"],"invocation":fields["invocation"],"sequence":seq,"activeJob":None,"captureAliases":copy.deepcopy(state["captureAliases"]),"readySequence":None})
        state["callReturn"]={"kind":"return","value":["ref",fields["returnPromise"]]}; state["stack"]=[]
    elif kind=="resume":
        continuation=_table(state,"continuations").get(fields["continuation"])
        if continuation is None: _fail("Resume without continuation")
        continuation["state"]="resumed"; state["ready"]=[v for v in state["ready"] if v!=fields["continuation"]]
        if fields["invocation"] not in state["stack"]: state["stack"].append(fields["invocation"])
    elif kind=="promise":
        identity=fields["promise"]; heap=_table(state,"heap")
        if fields["action"]=="create": _set_row(state["heap"],identity,{"kind":"promise","properties":[],"state":"pending","result":None,"adopted":None,"external":None,"reactions":[]})
        elif fields["action"]=="resolve":
            node=heap.get(identity)
            if node is None or node["state"]!="pending": _fail("Resolve unknown or settled promise")
            node["state"]="fulfilled"; node["result"]=fields["value"]
    elif kind in ("observe",): _fail("Observed delivery cannot appear as generated detail")
    else: _fail("Unsupported detailed transition kind: "+kind)


def _apply_observation(state,row,dispatched,settled):
    fields=row["fields"]
    if row["kind"]=="dispatch":
        if fields.get("attempt")!=1 or fields.get("key")!=f"{fields.get('invocationId')}/{fields.get('effectId')}": _fail("Effect dispatch identity")
        effect=next((e for e in state["effects"] if e[0]==fields["effectId"]),None)
        if effect is None: _fail("Dispatch of unknown effect")
        if fields["effectId"] in dispatched or fields["effectId"] in settled: _fail("Duplicate effect dispatch")
        dispatched.add(fields["effectId"])
    else:
        kind,obs=parse_record(fields["line"])
        if kind!="observe": _fail("Result line must be a detailed observation")
        if fields.get("effectId")!=obs.get("effect") or fields.get("status")!=obs.get("status") or fields.get("result")!=obs.get("result"): _fail("Observed result envelope differs from its detailed observation")
        effect=next((e for e in state["effects"] if e[0]==obs["effect"]),None)
        if effect is None: _fail("Result of unknown effect")
        if obs["effect"] not in dispatched or obs["effect"] in settled: _fail("Result without one outstanding dispatch")
        settled.add(obs["effect"])
        effect[1]["status"]=obs["status"]; effect[1]["result"]=obs["result"]
        promise=effect[1]["promise"]
        if promise:
            node=_table(state,"heap").get(promise); node["state"]=obs["status"]; node["result"]=obs["result"]
        for cid,cont in state["continuations"]:
            if cont["promise"]==promise and cont["state"]=="waiting":
                seq=state["scheduler"]["ready"]+1; state["scheduler"]["ready"]=seq
                cont["state"]="ready"; cont["readySequence"]=seq; state["ready"].append(cid)
