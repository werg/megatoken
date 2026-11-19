"""Frozen shared declarations for semantic execution/8."""
import json
from importlib.resources import files

from .algorithm_v7 import stable_algorithm_hash

CONTRACT=json.loads(files("megatoken.data").joinpath("contracts/semantic-execution-v8.json").read_bytes())
DEFINITION=CONTRACT["definition"]
CONTRACT_DIGEST=CONTRACT["digest"]
FIXTURE_DIGEST=CONTRACT["fixtureDigest"]
INTEROPERABILITY_DIGEST=CONTRACT["interoperabilityDigest"]
if (stable_algorithm_hash(DEFINITION)!=CONTRACT_DIGEST or
    stable_algorithm_hash(CONTRACT["fixtures"])!=FIXTURE_DIGEST or
    stable_algorithm_hash(CONTRACT["interoperability"])!=INTEROPERABILITY_DIGEST):
    raise ValueError("Pinned semantic-execution/8 contract or fixtures are inconsistent")


class SemanticContractV8Error(ValueError): pass


def validate_public(public):
    if type(public) is not dict or set(public)!=set(DEFINITION["publicFields"]):
        raise SemanticContractV8Error("V8 public execution fields")
    if (public["schema"],public["traceFormat"],public["contextFormat"],public["representation"],public["profile"])!=(DEFINITION["execution"],DEFINITION["trace"],DEFINITION["context"],"mixed",DEFINITION["profile"]):
        raise SemanticContractV8Error("Only exact semantic execution/8 is accepted")
    timeline=public["timeline"]
    if type(timeline) is not list or not timeline: raise SemanticContractV8Error("V8 timeline")
    for sequence,row in enumerate(timeline,1):
        if type(row) is not dict or set(row)!=set(DEFINITION["envelopeFields"]): raise SemanticContractV8Error("V8 timeline envelope fields")
        if row["sequence"]!=sequence or type(row["sequence"]) is not int or row["mode"] not in ("observed","generated") or type(row["kind"]) is not str or type(row["fields"]) is not dict:
            raise SemanticContractV8Error("V8 sequence, authority, or envelope shape")
        if row["mode"] != ("generated" if row["kind"] in {"semantic","algorithm","region","protocol"} else "observed"):
            raise SemanticContractV8Error("V8 record authority mismatch")
    return timeline
