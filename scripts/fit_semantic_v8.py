#!/usr/bin/env python3
"""Bounded decoder training and actual v8 inference from a freshly admitted artifact.

This diagnostic freezes native E/C/P and the memory bank, captures causal
boundaries through the production grouped trainer, and trains the actual text
decoder. Serving receives only initial public observations; no target is used to
choose, repair, or force an emitted record. In-sample success is not a
generalization, external-comprehension, or production-training claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import torch

from megatoken.data.chunking import ChunkingConfig
from megatoken.data.detailed_v8 import parse_record
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.native_v8 import (
    _replay_shared_state,
    prepare_native_v8_training,
    region_entry_v8,
)
from megatoken.data.prefix_v8 import validate_v8_committed_prefix
from megatoken.data.protocol_v8 import decode_protocol_record
from megatoken.data.semantic_contract_v8 import CONTRACT_DIGEST, DEFINITION
from megatoken.data.semantic_json import stable_algorithm_hash
from megatoken.generation.codec import LING_TOKENIZER_SHA256, ByteCodec
from megatoken.io import sha256_file
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.semantic_model_v8 import NativeSemanticInferenceV8
from megatoken.served import ServedBundle, export_served
from megatoken.training.checkpoint import save_checkpoint, transducer_attachment
from megatoken.training.config import TrainConfig
from megatoken.training.engine import prepare_batches, window_config
from megatoken.training.losses import LossLedger
from megatoken.training.meridians import forward_meridian_batch
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.stream import EpisodeWindow, _WindowPlan
from megatoken.training.transducer import TransducerTrainer


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_identity(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0" + str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_admission(document, receipt, artifact_path, codec):
    """Bind all six independent checks to these exact source and model inputs."""
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip():
        raise ValueError("The admitted consumer implementation must be a clean checkout")
    required = {
        "schema": "semantic-native-admission/8",
        "admitted": True,
        "publicDigest": stable_algorithm_hash(document["public"]),
        "contextsDigest": stable_algorithm_hash(document["contexts"]),
        "sourceArtifactDigest": stable_algorithm_hash(document["sidecar"]["sourceArtifact"]),
        "consumerRevision": revision,
        "contractDigest": CONTRACT_DIGEST,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("Training requires a fresh exact-revision v8 native admission receipt")
    if any(
        receipt.get("checks", {}).get(name) != "passed"
        for name in ("replay", "contexts", "continuation", "masks", "sourceBinding", "tokenizer")
    ):
        raise ValueError("Training requires all six native admission checks")
    if receipt.get("tokenizer") != {
        "status": "passed",
        "identity": codec.fingerprint,
        "sha256": LING_TOKENIZER_SHA256,
    }:
        raise ValueError("Training receipt does not bind the loaded production tokenizer")
    source = receipt.get("sourceVerification", {})
    if source.get("artifactSha256") != sha256_file(artifact_path) or any(
        source.get(name) is not True
        for name in (
            "producerTreeClean",
            "sourceValidated",
            "projectionValidated",
            "replayValidated",
        )
    ):
        raise ValueError("Training receipt has no exact live source verification")
    _, packing = prepare_native_v8_training(document, codec=codec)
    if receipt.get("nativeTrainingDigest") != packing["nativeTrainingDigest"]:
        raise ValueError("Admitted physical training bytes differ from fresh preparation")
    return revision


def capture_boundaries(model, episode, config, output):
    """Read the real grouped trainer's pre-target states without changing its output."""
    plan = _WindowPlan(episode, model.codec, window_config(config))
    windows = [
        EpisodeWindow(episode, plan[index], TensorAssetStore(output)) for index in range(len(plan))
    ]
    batches = prepare_batches(model, windows, config)
    transducer = TransducerTrainer(
        model.transducer, bank_trainable=False, teacher_weight=0.0, navigation_weight=0.0
    )
    captured = []
    original = MultiTrackGenerator.supervised_meridian

    def observe_boundary(multi, state, **kwargs):
        captured.append((state.detach(), dict(kwargs)))
        return original(multi, state, **kwargs)

    with (
        torch.no_grad(),
        patch.object(MultiTrackGenerator, "supervised_meridian", observe_boundary),
    ):
        for batch in batches:
            ledger = LossLedger(dict(batch.counts), dict(batch.counts), config.loss_weights)
            forward_meridian_batch(
                model,
                batch,
                ledger,
                config,
                torch.device("cpu"),
                0,
                differentiate=False,
                transducer=transducer,
            )
            ledger.finish()
    if not captured:
        raise ValueError("The admitted episode has no native generated training boundaries")
    return captured, transducer


def initial_public(document):
    """Strip all teacher-generated rows and all target contexts from inference input."""
    rows = document["public"]["timeline"]
    cutoff = next(index for index, row in enumerate(rows) if row["mode"] == "generated")
    public = copy.deepcopy({**document["public"], "timeline": rows[:cutoff]})
    validate_v8_committed_prefix(public, [])
    invocations = [row["fields"] for row in public["timeline"] if row["kind"] == "invocation"]
    if len(invocations) != 1:
        raise ValueError("The bounded diagnostic requires one initial source invocation")
    return public


@torch.inference_mode()
def actual_inference(model, identity, public, *, seed, max_tokens, max_records=4):
    """Predict and independently replay actual records; no teacher target is accepted."""
    public = copy.deepcopy(public)
    contexts = []
    invocation = next(row["fields"] for row in public["timeline"] if row["kind"] == "invocation")
    session, invocation_id = invocation["session"], invocation["invocationId"]
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model.eval()))
    adapter = NativeSemanticInferenceV8(
        runtime, model_identity=identity, max_tokens=max_tokens, max_fragments=1
    )
    records = []
    outcome = {"status": "failed", "completed": False}
    started = time.perf_counter()
    try:
        adapter.initialize(
            seed=seed, checkpointIdentity=identity, tokenizerIdentity=model.codec.fingerprint
        )
        for _ in range(max_records):
            boundary = _replay_shared_state(
                public["timeline"], contexts, {}, roundtrip=False, boundary_invocation=invocation_id
            )
            context = {
                "schema": DEFINITION["context"],
                "representation": "mixed",
                "profile": DEFINITION["profile"],
                "invocationId": invocation_id,
                **boundary,
                "visibleThrough": len(public["timeline"]),
                "prefixDigest": stable_algorithm_hash(public["timeline"]),
            }
            selected = adapter.select(
                session=session, public=public, contexts=contexts, context=context
            )
            record = selected["record"]
            records.append(record)
            label = record.split(" ", 1)[0]
            kind = {
                "detail": "semantic",
                "protocol": "protocol",
                "algorithm": "algorithm",
                "region": "region",
            }.get(label)
            if kind is None:
                raise ValueError("Actual model record has no supported v8 authority")
            fields = {
                "session": session,
                "track": invocation["track"],
                "invocationId": invocation_id,
                "line": record,
            }
            if kind in {"algorithm", "region"}:
                region = "whole" if kind == "algorithm" else json.loads(record[7:])["region"]
                fields["entry"] = region_entry_v8(
                    context["programs"][-1]["program"], context, region
                )
            candidate = copy.deepcopy(public)
            candidate["timeline"].append(
                {
                    "sequence": len(public["timeline"]) + 1,
                    "mode": "generated",
                    "kind": kind,
                    "fields": fields,
                }
            )
            adapter.commit(
                session=session,
                public=candidate,
                contexts=contexts + [context],
                invocationId=invocation_id,
                record=record,
                selection=selected["selection"],
            )
            public, contexts = candidate, contexts + [context]
            terminal = (
                parse_record(record)[0] == "end"
                if kind == "semantic"
                else decode_protocol_record(record)["kind"] == "end"
                if kind == "protocol"
                else False
            )
            if terminal:
                adapter.complete(
                    session=session, public=public, contexts=contexts, invocationId=invocation_id
                )
                outcome.update(status="completed", completed=True)
                outcome["finalState"] = _replay_shared_state(
                    public["timeline"], contexts, {}, roundtrip=True
                )
                break
        else:
            raise ValueError("Actual inference exceeded the bounded record limit")
    except Exception as error:
        outcome["error"] = {"category": type(error).__name__, "message": str(error)}
        work = adapter.sessions.get(session)
        if work is not None and work.pending is not None:
            adapter.reject(
                session=session,
                selectionId=work.pending["attempt"]["selectionId"],
                reason=str(error),
            )
    finally:
        adapter.cancel(session=session, reason="diagnostic-finished")
        runtime.close()
    return {
        **outcome,
        "records": records,
        "public": public,
        "contexts": contexts,
        "seconds": time.perf_counter() - started,
        "metrics": adapter.metrics(),
    }


def save_bundle(model, optimizer, transducer, config, output, name, *, steps):
    checkpoint = save_checkpoint(
        output / "checkpoints",
        name,
        {
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "codec_fingerprint": model.codec.fingerprint,
            "chunking": ChunkingConfig()
            .with_hard_limit(model.config.max_capacity)
            .identity(model.codec),
            "global_step": 0,
            "diagnostic_optimizer_steps": steps,
            "train_config": asdict(config),
            "optimizer": optimizer.state_dict(),
            "packs_fingerprint": None,
        },
        {
            "global_step": 0,
            "diagnosticOptimizerSteps": steps,
            "scope": "bounded-v8-decoder-fit",
            "ordinaryTrainerResumeClaim": False,
        },
        pinned=True,
        keep=2,
        attachments=transducer_attachment(transducer.state_dict),
    )
    destination = output / name
    manifest = export_served(checkpoint, destination)
    loaded = ServedBundle(destination, device="cpu")
    if tensor_identity(model.served_state_dict()) != tensor_identity(
        loaded.model.served_state_dict()
    ):
        raise ValueError("Saved and reloaded served weights differ from the fitted model")
    return loaded, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, action="append", default=[])
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), default=17)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--evaluate-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    args = parser.parse_args()
    if (
        not 1 <= args.max_steps <= 256
        or not 1 <= args.max_tokens <= 512
        or not 1 <= args.evaluate_every <= 256
    ):
        parser.error(
            "Bounded fit allows 1..256 steps, 1..512 generated tokens, and 1..256 evaluation interval"
        )
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.1:
        parser.error("Learning rate must be positive, finite, and at most 0.1")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    codec = ByteCodec.from_tokenizer_json(
        args.tokenizer_json, expected_sha256=LING_TOKENIZER_SHA256
    )
    document = json.loads(args.artifact.read_text())
    admission = json.loads(args.admission.read_text())
    revision = require_admission(document, admission, args.artifact, codec)
    episode, preparation = prepare_native_v8_training(
        document, codec=codec, max_positions=512, max_history=128
    )
    episode = replace(
        episode,
        supervision=replace(episode.supervision, outcome="success", verification="external_tests"),
    )
    targets = [
        row["fields"]["line"]
        for row in document["public"]["timeline"]
        if row["mode"] == "generated"
    ]
    if not 2 <= len(targets) <= 3 or any(
        len(codec.encode(target.encode())) > args.max_tokens for target in targets
    ):
        raise ValueError("Bounded fit requires 2..3 complete one-fragment target records")
    config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        model="tiny",
        datasets=[str(args.artifact.resolve())],
        output=str(args.output),
        codec=str(args.tokenizer_json.resolve()),
        device="cpu",
        precision="fp32",
        seed=args.seed,
        max_steps=args.max_steps,
        representation_mode="bootstrap",
        effort=1,
        generated_capacity=512,
        max_window_records=128,
        max_history_tokens=16384,
        max_record_bytes=16384,
        scored_decisions=3,
        gradient_checkpointing=False,
        expert_backend="torch",
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    model = build_model(
        ModelConfig.tiny(max_capacity=512, max_history_chunks=128, vocab_size=codec.vocab_size),
        codec,
    ).eval()
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    boundaries, transducer = capture_boundaries(model, episode, config, args.output)
    if len(boundaries) != len(targets):
        raise ValueError("Diagnostic target boundaries were unexpectedly split or duplicated")
    optimizer = MasterAdamW(model, lr=args.learning_rate, weight_decay=0.0)
    decoder_owners = [name for name in optimizer.model_parameters if name.startswith("decoder.")]
    if not decoder_owners:
        raise ValueError("The native optimizer has no decoder parameter owners")
    optimizer.freeze([name for name in optimizer.model_parameters if name not in decoder_owners])
    baseline, baseline_manifest = save_bundle(
        model, optimizer, transducer, config, args.output, "baseline", steps=0
    )
    inference_inputs = [initial_public(document)] + [
        initial_public(json.loads(path.read_text())) for path in args.heldout
    ]
    evaluations = [
        {
            "optimizerSteps": 0,
            "train": actual_inference(
                baseline.model,
                baseline_manifest["model_sha256"],
                inference_inputs[0],
                seed=args.seed,
                max_tokens=args.max_tokens,
            ),
            "heldout": [
                actual_inference(
                    baseline.model,
                    baseline_manifest["model_sha256"],
                    value,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                )
                for value in inference_inputs[1:]
            ],
        }
    ]
    write_json(args.output / "baseline-evaluation.json", evaluations[0])
    del baseline
    history = []
    started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        step_started = time.perf_counter()
        torch.manual_seed(args.seed + step)
        optimizer.zero_grad()
        losses = []
        components = {}
        for state, kwargs in boundaries:
            result = MultiTrackGenerator.of(model).supervised_meridian(state, **kwargs)
            for name in ("structure", "binding", "placement", "terminal", "lexical"):
                numerator, count = result.losses[name]
                if count:
                    losses.append(numerator * (config.loss_weights[name] / count / len(boundaries)))
                    components[name] = components.get(name, 0.0) + float(
                        numerator.detach()
                    ) / count / len(boundaries)
        loss = torch.stack(losses).sum()
        if not torch.isfinite(loss):
            raise ValueError("Native diagnostic loss became nonfinite")
        loss.backward()
        update = optimizer.step(1.0)
        model.core.routing_aux_loss(clear=True)
        history.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "components": components,
                "seconds": time.perf_counter() - step_started,
                **update,
            }
        )
        if step % args.evaluate_every == 0 or step == args.max_steps:
            result = actual_inference(
                model,
                tensor_identity(model.served_state_dict()),
                inference_inputs[0],
                seed=args.seed,
                max_tokens=args.max_tokens,
            )
            evaluations.append(
                {"optimizerSteps": step, "identityKind": "in-memory-tensor-sha256", "train": result}
            )
            write_json(
                args.output / "progress.json",
                {
                    "history": history,
                    "evaluations": evaluations,
                    "seconds": time.perf_counter() - started,
                },
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": history[-1]["loss"],
                        "actualCompleted": result["completed"],
                        "exactInSampleRecords": result["records"] == targets,
                    }
                ),
                flush=True,
            )
            if result["completed"] and result["records"] == targets:
                break
    changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), initial_state[name])
    ]
    if not changed or any(not name.startswith("decoder.") for name in changed):
        raise ValueError("Bounded decoder fit changed an undeclared model owner")
    fitted, fitted_manifest = save_bundle(
        model, optimizer, transducer, config, args.output, "fitted", steps=step
    )
    final = {
        "optimizerSteps": step,
        "train": actual_inference(
            fitted.model,
            fitted_manifest["model_sha256"],
            inference_inputs[0],
            seed=args.seed,
            max_tokens=args.max_tokens,
        ),
        "heldout": [
            actual_inference(
                fitted.model,
                fitted_manifest["model_sha256"],
                value,
                seed=args.seed,
                max_tokens=args.max_tokens,
            )
            for value in inference_inputs[1:]
        ],
    }
    optimizer.close()
    report = {
        "schema": "semantic-native-bounded-fit/8",
        "scope": "Actual native grouped decoder optimization and reloaded serving on 2..3 in-sample records. No generalization, external-comprehension, or production-training claim.",
        "consumerRevision": revision,
        "scriptSha256": sha256_file(Path(__file__)),
        "admission": admission,
        "artifactSha256": sha256_file(args.artifact),
        "seed": args.seed,
        "optimizer": "MasterAdamW",
        "optimizerSteps": step,
        "learningRate": args.learning_rate,
        "tokenizerIdentity": codec.fingerprint,
        "tokenizerSha256": LING_TOKENIZER_SHA256,
        "baselineManifest": baseline_manifest,
        "fittedManifest": fitted_manifest,
        "changedModelOwners": changed,
        "generatedTrainingTokensPerStep": preparation["generatedTokens"],
        "preparation": preparation,
        "history": history,
        "evaluations": evaluations,
        "final": final,
        "inSampleExactSuccess": final["train"]["completed"]
        and final["train"]["records"] == targets,
        "heldoutCompleted": [result["completed"] for result in final["heldout"]],
        "generalizationEstablished": False,
        "externalComprehensionEstablished": False,
        "optimizerSeconds": sum(row["seconds"] for row in history),
        "fitAndEvaluationSeconds": time.perf_counter() - started,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "threads": 1,
        },
    }
    write_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "report": str(args.output / "report.json"),
                "served": str(args.output / "fitted"),
                "optimizerSteps": step,
                "inSampleExactSuccess": report["inSampleExactSuccess"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
