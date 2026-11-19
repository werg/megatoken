#!/usr/bin/env python3
"""Write a conservative native engineering receipt for semantic execution/8."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from megatoken.data.algorithm_v7 import stable_algorithm_hash
from megatoken.data.native_v8 import validate_v8_artifact
from megatoken.data.semantic_contract_v8 import CONTRACT, CONTRACT_DIGEST


def checked_revision(root, expected):
    actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    dirty=subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=root,text=True).strip()
    if actual!=expected or dirty:
        raise ValueError("V8 native receipt requires the exact clean consumer revision")
    if not Path(__import__("megatoken").__file__).resolve().is_relative_to(root/"src"):
        raise ValueError("Imported Python package is not this consumer checkout")
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--consumer-revision",required=True)
    parser.add_argument("--tokenizer-json",type=Path)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve(): parser.error("Input and receipt paths must differ")
    root=Path(__file__).resolve().parents[1]
    revision=checked_revision(root,args.consumer_revision)
    document=json.loads(args.input.read_text())
    result=validate_v8_artifact(document)
    sidecar=document.get("sidecar",{})
    public=document["public"]; contexts=document["contexts"]
    tokenizer={"status":"untested","identity":None,"sha256":None}
    if args.tokenizer_json is not None:
        raw=args.tokenizer_json.read_bytes(); tokenizer={"status":"untested","identity":None,"sha256":hashlib.sha256(raw).hexdigest()}
        # ByteCodec performs the repository's vocabulary and pinned tokenizer
        # verification when the tokenizer dependency is installed.
        try:
            from megatoken.generation.codec import ByteCodec, LING_TOKENIZER_SHA256
            ByteCodec.from_tokenizer_json(args.tokenizer_json,expected_sha256=LING_TOKENIZER_SHA256)
            tokenizer={"status":"passed","identity":"inclusionAI/Ling-3.0-tiny","sha256":LING_TOKENIZER_SHA256}
        except ImportError:
            tokenizer["status"]="resource_limited"
    checks=dict(result["checks"]); checks["tokenizer"]=tokenizer["status"]
    source_artifact=sidecar.get("sourceArtifact")
    receipt={"schema":"semantic-native-admission/8","traceFormat":"semantic-trace/8","contextFormat":"semantic-context/8","representation":"mixed","componentGate":"native-consumer-only",
             "publicDigest":stable_algorithm_hash(public),"contextsDigest":stable_algorithm_hash(contexts),
             "sourceArtifactDigest":stable_algorithm_hash(source_artifact) if source_artifact is not None else None,
             "datasetBindingDigest":stable_algorithm_hash(sidecar["datasetBinding"]) if isinstance(sidecar,dict) and sidecar.get("datasetBinding") is not None else None,
             "consumerRevision":revision,"contractDigest":CONTRACT_DIGEST,"fixtureDigest":CONTRACT["fixtureDigest"],
             "tokenizer":tokenizer,"checks":checks,"counts":{"records":len(public["timeline"]),"contexts":result["counts"]["contexts"],"generatedRecords":result["counts"]["generatedRecords"],"generatedTokens":None,"observedTokens":None},
             "nativeTrainingDigest":None,"optimizerSteps":0,"learningClaim":None,"admitted":False}
    # Engineering fixtures lack private source-bound records and exact physical
    # token rows. Never turn structural success into model-serving admission.
    if args.output.resolve()==args.input.resolve(): parser.error("Input and receipt paths must differ")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,ensure_ascii=False,separators=(",",":"))+"\n")
    print(json.dumps(receipt,separators=(",",":")))
    return 0


if __name__=="__main__": raise SystemExit(main())
