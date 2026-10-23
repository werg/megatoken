#!/usr/bin/env python3
"""Optimize prepared context tasks, restart, export, and use the public service.

This bounded CPU-compatible integration experiment does not establish learned
delegation or task quality. Prepare its data with prepare_context_tasks.py.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from megatoken.io import write_json
from megatoken.model.context import DEFAULT_CROSS_EVENTS, DEFAULT_LOCAL_EVENTS
from megatoken.model.multitrack import MultiTrackGenerator
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.multitrack_service import MultiTrackService
from megatoken.served import ServedBundle, export_served
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--codec", default="byte")
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError("Acceptance requires a fresh output root")
    args.root.mkdir(parents=True)
    config = TrainConfig(
        purpose="diagnostic",
        phase="P1",
        model="tiny",
        datasets=[str(args.data)],
        output=str(args.root / "training"),
        codec=args.codec,
        allow_byte_codec=args.codec == "byte",
        device=args.device,
        precision="fp32",
        representation_mode="bootstrap",
        effort=1,
        model_overrides={
            "max_capacity": 512,
            "max_history_chunks": 1024,
            "context_local_events": DEFAULT_LOCAL_EVENTS,
            "context_cross_events": DEFAULT_CROSS_EVENTS,
        },
        max_steps=2,
        microbatch_episodes=1,
        max_accumulation_windows=1,
        accumulate_scored=1,
        scored_decisions=2,
        generated_capacity=512,
        max_window_records=16,
        max_history_tokens=4096,
        max_record_bytes=4096,
        checkpoint_every=1,
        eval_every=1,
        eval_windows=1,
        gradient_checkpointing=False,
        expert_backend="torch",
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    first = train(config)
    resumed = train(
        replace(config, max_steps=3), resume=args.root / "training/checkpoints/latest.json"
    )
    exported = export_served(
        args.root / "training/checkpoints/latest.json", args.root / "served", codec_path=args.codec
    )
    bundle = ServedBundle(args.root / "served", device=args.device)
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(bundle.model), chunking=bundle.chunking)
    try:
        service = MultiTrackService(runtime)
        opened = service.handle({"op": "open", "effort": 1})
        session_id = opened["session"]
        observed = service.handle(
            {
                "op": "observe",
                "session": session_id,
                "effort": 1,
                "events": [
                    {"track": "worker", "event_id": "source", "text": "The project key is violet."}
                ],
            }
        )
        configured = service.handle(
            {
                "op": "configure_context",
                "session": session_id,
                "track": "worker",
                "local_events": 4,
                "cross_events": 1,
                "pinned_events": ["source"],
            }
        )
        generated = service.handle(
            {
                "op": "generate",
                "session": session_id,
                "effort": 1,
                "outputs": [
                    {"track": "worker", "max_tokens": 4, "reference_event_ids": ["source"]}
                ],
            }
        )
        assert all(result["ok"] for result in (opened, observed, configured, generated))
    finally:
        runtime.close()
    write_json(
        args.root / "acceptance.json",
        {
            "first_run": first,
            "restart": resumed,
            "export": exported,
            "public_context_change": configured,
            "generation": generated,
            "learned_quality_established": False,
        },
    )


if __name__ == "__main__":
    main()
