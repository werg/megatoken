#!/usr/bin/env python3
"""Exercise an actual initialized served model over JSONL; never create an optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from megatoken.data.chunking import ChunkingConfig
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import sha256_file, write_json
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.served import ServedBundle, export_served
from megatoken.training.checkpoint import save_checkpoint, transducer_attachment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    codec = ByteCodec.from_tokenizer_json(
        options.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    config = ModelConfig.tiny(max_history_chunks=128, max_capacity=512, vocab_size=codec.vocab_size)
    model = build_model(config, codec).eval()
    checkpoint = save_checkpoint(
        options.output / "initialization",
        "fixed-weights",
        {
            "model": model.state_dict(),
            "model_config": config.to_dict(),
            "codec_fingerprint": codec.fingerprint,
            "chunking": ChunkingConfig().with_hard_limit(config.max_capacity).identity(codec),
            "global_step": 0,
            "train_config": {
                "precision": "fp32",
                "packs": None,
                "codec": str(options.tokenizer_json.resolve()),
            },
            "packs_fingerprint": None,
        },
        {"global_step": 0, "origin": "initialized-fixed-weights", "optimizerSteps": 0},
        pinned=True,
        attachments=transducer_attachment(lambda: {"bank": model.transducer.bank.state_snapshot()}),
    )
    destination = options.output / "served"
    manifest = export_served(checkpoint, destination)
    loaded = ServedBundle(destination, device="cpu")
    if loaded.manifest["model_sha256"] != manifest["model_sha256"]:
        raise ValueError("Initialized export changed its loaded model identity")
    source = "function run(){return 7;}"
    digest = hashlib.sha256(source.encode()).hexdigest()
    common = {
        "schema": "semantic-execution/5",
        "session": "fixedweights",
        "mode": "observed",
        "ownerSequence": 1,
    }
    program = {
        **common,
        "kind": "program",
        "sequence": 1,
        "track": "host",
        "programId": "pure",
        "source": source,
        "digest": digest,
        "entry": "run",
        "sites": [],
        "functions": [],
        "capabilities": [],
        "dependencies": [],
        "sourceMap": None,
        "sourceMetadata": None,
        "languageProfile": "owned-javascript/1",
        "module": False,
    }
    invocation = {
        **common,
        "kind": "invocation",
        "sequence": 2,
        "track": "worker",
        "programId": "pure",
        "invocationId": "outside",
        "realmId": "realm",
        "triggerId": "trigger",
        "ingressKey": "trigger",
        "entry": "run",
        "sourceDigest": digest,
        "realmPolicy": "fresh",
        "completionPolicy": "call-return",
        "inputGraph": {"roots": [], "objects": []},
        "receiver": ["undefined"],
        "closureId": None,
        "activation": "call",
        "parentInvocationId": None,
        "taskRevision": 1,
        "deadline": None,
        "bindings": {},
    }
    prefix = {
        "schema": "selected-inference-prefix/2",
        "programId": "pure",
        "source": source,
        "sourceDigest": digest,
        "entry": "run",
        "invocationId": "outside",
        "trackId": "worker",
        "realmId": "realm",
        "completionPolicy": "call-return",
        "maxOutstanding": 64,
        "committed": [program, invocation],
    }
    command = [
        sys.executable,
        str(Path(__file__).with_name("serve_selected_semantic.py")),
        "--served",
        str(destination),
        "--device",
        "cpu",
        "--max-tokens",
        "8",
    ]
    messages = [
        {
            "id": "initialize",
            "op": "initialize",
            "seed": 17,
            "checkpointIdentity": manifest["model_sha256"],
        },
        {"id": "selection", "op": "select", "prefix": prefix, "representation": "compact"},
    ]
    completed = subprocess.run(
        command,
        input="".join(json.dumps(item) + "\n" for item in messages),
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    replies = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    if len(replies) != 2 or {
        key: value for key, value in replies[0].items() if key != "runtimeMetrics"
    } != {
        "id": "initialize",
        "ok": True,
        "result": {
            "schema": "semantic-inference-identity/4",
            "seed": 17,
            "checkpointIdentity": manifest["model_sha256"],
            "tokenizerIdentity": codec.fingerprint,
        },
    }:
        raise ValueError("Actual served process failed immutable identity/seed attestation")
    for reply in replies:
        metrics = reply["runtimeMetrics"]
        if (
            metrics["schema"] != "semantic-inference-metrics/4"
            or metrics["peakMemoryBytes"] <= 0
            or metrics["computeSeconds"] < 0
        ):
            raise ValueError("Actual native process omitted measured memory/time evidence")
    selected = replies[1]
    if selected["ok"]:
        if (
            selected["result"]["schema"] != "selected-model-output/2"
            or selected["result"].get("representation") != "compact"
            or not selected["result"]["selection"]["modelSelected"]
        ):
            raise ValueError(
                "Canonical semantic path did not return an actual native model selection"
            )
    elif not selected["error"].get("evidence"):
        raise ValueError("Initialized-model failure lost its actual selected output evidence")
    if sha256_file(destination / "model.safetensors") != manifest["model_sha256"]:
        raise ValueError("Inference changed the initialized served weights")
    receipt = {
        "schema": "semantic-fixed-inference-check/4",
        "origin": "initialized-fixed-weights",
        "inferenceIdentity": replies[0]["result"],
        "served": str(destination.resolve()),
        "selectedReply": selected,
        "weightsUnchanged": True,
        "optimizerSteps": 0,
        "nativeEffects": 0,
        "learnedSuccess": None,
    }
    write_json(options.output / "inference-check.json", receipt)
    print(
        json.dumps(
            {
                "receipt": str((options.output / "inference-check.json").resolve()),
                "checkpointIdentity": manifest["model_sha256"],
                "actualModelSelection": True,
                "validOutcome": selected["ok"],
                "optimizerSteps": 0,
                "learnedSuccess": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
