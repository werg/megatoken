#!/usr/bin/env python3
"""Prepare executed tasks, optimize, restart, export and exercise the actual service.

This bounded integration run is not a production-quality or media-quality claim.
The exact authored tasks, source receipts, checkpoints and generated responses
are retained for inspection. The ordinary training engine owns optimization.
"""

from __future__ import annotations

import argparse
import gc
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.dynamic_execution import SOURCE_VERSION, execution_episode
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.io import sha256_file, write_json
from megatoken.loading import load_codec
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.multitrack_service import MultiTrackService
from megatoken.served import ServedBundle, export_served
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--families", type=int, default=8)
    parser.add_argument("--model", default="tiny", choices=("tiny", "32x256", "32x512"))
    parser.add_argument("--precision", default="fp32", choices=("fp32", "bf16"))
    parser.add_argument("--codec", default="byte")
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError("Acceptance outputs are immutable; choose a fresh root")
    args.root.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "src/megatoken/data/dynamic_execution.py"
    shutil.copyfile(source, args.root / "dynamic_execution.source.py")
    codec = ByteCodec.byte_fallback() if args.codec == "byte" else load_codec(args.codec)
    chunking = ChunkingConfig().with_hard_limit(512, max_bytes=2048)
    store = DedupStore(
        args.root / "spool.sqlite",
        DedupConfig(train_fraction=0.75, validation_fraction=0.25, min_near_words=1_000_000),
    )
    try:
        for index in range(args.families):
            for delegated in (False, True):
                episode = execution_episode(
                    index, delegated=delegated, split="validation" if index % 4 == 0 else "train"
                )
                with store.transaction():
                    store.add(episode, codec)
        store.freeze()
        prepared = export_prepared(
            store,
            args.root / "data",
            codec=codec,
            source_receipts=[
                {
                    "source_id": SOURCE_VERSION,
                    "origin": "authored_executed",
                    "code_sha256": sha256_file(source),
                }
            ],
            pipeline_fingerprint=SOURCE_VERSION,
            chunking=chunking,
        )
    finally:
        store.close()
    config = TrainConfig(
        phase="B0",
        model=args.model,
        datasets=[str(args.root / "data")],
        output=str(args.root / "training"),
        device=args.device,
        precision=args.precision,
        model_overrides={
            "max_capacity": 512,
            "vocab_size": codec.vocab_size,
            **({"ngram_rows": 131072, "max_history_chunks": 64} if args.model != "tiny" else {}),
        },
        codec=args.codec,
        allow_byte_codec=True,
        max_steps=args.steps,
        microbatch_episodes=1,
        max_accumulation_windows=1,
        accumulate_scored=1,
        accumulate_ingested=2048,
        max_window_records=4,
        scored_decisions=2,
        max_history_tokens=512,
        generated_capacity=512,
        max_record_bytes=2048,
        effort=1,
        gradient_checkpointing=False,
        expert_backend="torch",
        checkpoint_every=2,
        eval_every=2,
        eval_windows=1,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    write_json(args.root / "train-config.json", asdict(config))
    first = train(config)
    gc.collect()
    checkpoint = args.root / "training/checkpoints/latest.json"
    resumed = train(replace(config, max_steps=2), resume=str(checkpoint))
    exported = export_served(checkpoint, args.root / "served", codec_path=args.codec)
    bundle = ServedBundle(args.root / "served", device=args.device)
    runtime = MultiTrackRuntime(MultiTrackGenerator(bundle.model), chunking=bundle.chunking)
    try:
        service = MultiTrackService(runtime)
        opened = service.handle({"op": "open", "effort": 1})
        if not opened["ok"]:
            raise RuntimeError(opened)
        sid = opened["session"]
        observations = service.handle(
            {
                "op": "observe_text",
                "session": sid,
                "track": "user",
                "text": "Read the independent inputs and report your next action.",
                "effort": 1,
            }
        )
        generated = service.handle(
            {
                "op": "generate",
                "session": sid,
                "effort": 1,
                "seed": 71,
                "outputs": [
                    {
                        "track": "assistant",
                        "event_id": "initial-action",
                        "modality": "text",
                        "max_tokens": 128,
                    }
                ],
            }
        )
        write_json(
            args.root / "generated.json",
            {"open": opened, "observed": observations, "generated": generated},
        )
        if not generated["ok"]:
            raise RuntimeError(generated)
    finally:
        runtime.close()
    write_json(
        args.root / "acceptance.json",
        {
            "source": SOURCE_VERSION,
            "prepared": prepared,
            "first_run": first,
            "restart": resumed,
            "export": exported,
            "generated_protocol_ok": True,
            "quality_established": False,
        },
    )


if __name__ == "__main__":
    main()
