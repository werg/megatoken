#!/usr/bin/env python3
"""Compare real retained KV storage and exercise context reread and policy replay.

The tiny model is untrained. This measures cache storage and execution/replay
correctness, not task quality or whole-device peak-memory savings.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from megatoken.generation.codec import ByteCodec
from megatoken.generation.structure import ActionSpec
from megatoken.io import write_json
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.context import ContextPolicy
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.model.policy_context import reconstruct
from megatoken.multitrack_runtime import MultiTrackRuntime


def run(model, *, local, cross, events):
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model), record_policy_context=True)
    try:
        session = runtime.open(effort=1)
        session.memory.state = runtime.generator.tracks.set_context_policy(
            session.state, ContextPolicy(local_events=local, cross_events=cross)
        )
        runtime.observe(
            session,
            (
                ObservedEvent(
                    "worker", "text", b"The project key is violet.", event_id="original-evidence"
                ),
            ),
            effort=1,
        )
        for i in range(events):
            runtime.observe(
                session,
                (
                    ObservedEvent(
                        "background",
                        "text",
                        f"Unrelated observation {i}".encode(),
                        event_id=f"background-{i}",
                    ),
                ),
                effort=1,
            )
        before = runtime.context_status(session, track="worker")
        assert "original-evidence" in before["visible_events"]
        selected = runtime.generator.select_group(
            session.state,
            tracks=("worker",),
            effort=1,
            specs=(ActionSpec(min_tokens=1, max_tokens=1),),
            generator=torch.Generator(device=session.state.device).manual_seed(9),
        )
        restored = reconstruct(
            runtime.generator, selected.tape.policy_context, device=session.state.device
        )
        torch.testing.assert_close(
            runtime.generator.score_group(restored, selected.tape),
            torch.tensor(selected.tape.behavior_log_prob, device=session.state.device),
        )
        fetched = runtime.fetch_context(session, track="worker", source_event="background-0")
        assert (
            fetched["event_id"] in runtime.context_status(session, track="worker")["visible_events"]
        )
        return {
            "local_events": local,
            "cross_events": cross,
            "before_explicit_reread": before,
            "reread": fetched,
            "policy_reconstruction_ok": True,
        }
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--background-events", type=int, default=24)
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError("Acceptance outputs require a fresh root")
    if not 8 <= args.background_events <= 48:
        parser.error("Use eight to forty-eight background events for the bounded comparison")
    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = build_model(
        ModelConfig.tiny(max_history_chunks=64), ByteCodec.byte_fallback(), device=args.device
    )
    model.eval()
    with torch.no_grad():
        shared = run(model, local=64, cross=64, events=args.background_events)
        asymmetric = run(model, local=4, cross=1, events=args.background_events)
    baseline = shared["before_explicit_reread"]["cache_bytes"]
    measured = asymmetric["before_explicit_reread"]["cache_bytes"]
    if measured >= baseline:
        raise AssertionError(
            "Asymmetric context did not reclaim KV storage in the controlled comparison"
        )
    args.root.mkdir(parents=True)
    write_json(
        args.root / "acceptance.json",
        {
            "device": args.device,
            "background_events": args.background_events,
            "shared": shared,
            "asymmetric": asymmetric,
            "retained_kv_bytes_saved": baseline - measured,
            "retained_kv_fraction_saved": (baseline - measured) / baseline,
            "whole_device_peak_memory_measured": False,
            "learned_quality_established": False,
        },
    )


if __name__ == "__main__":
    main()
