#!/usr/bin/env python3
"""Bounded local restart, mixed-input, and old-context recovery checks.

These diagnostic runs establish executable paths, not learned answer quality.
All outputs are written to a new directory; input corpora remain read-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from training_fixture import make_episode, tiny_train_config

from megatoken.contracts import MAIN_TRACK, ROLE_TO_ID
from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.chunking import ChunkingConfig
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
from megatoken.data.schema import sequential_meridians
from megatoken.data.storage import PreparedDataset, export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.io import write_json
from megatoken.loading import build_model
from megatoken.model import ModelConfig
from megatoken.model.multitrack import MultiTrackGenerator, ObservedEvent
from megatoken.multitrack_runtime import MultiTrackRuntime
from megatoken.multitrack_service import MultiTrackService
from megatoken.served import ServedBundle, export_served
from megatoken.training.checkpoint import load_checkpoint
from megatoken.training.config import TrainConfig
from megatoken.training.engine import train


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def restart(args):
    config = tiny_train_config(args.root / "data", device=args.device, precision="fp32")
    full = train(replace(config, output=str(args.root / "full"), max_steps=2))
    release()
    first = train(replace(config, output=str(args.root / "resumed")))
    release()
    resumed = train(replace(config, output=str(args.root / "resumed")), resume=first["checkpoint"])
    release()
    left, _ = load_checkpoint(full["checkpoint"], map_location="cpu")
    right, _ = load_checkpoint(resumed["checkpoint"], map_location="cpu")
    checked = ["model", "optimizer", "rank_state", "phase_consumed", "counters", "global_step"]
    for key in checked:
        equal_tree(left[key], right[key])
    return {"exact_match": True, "checked": checked, "global_step": right["global_step"]}


def memory(args):
    model = build_model(
        ModelConfig.tiny(max_history_chunks=4), ByteCodec.byte_fallback(), device=args.device
    ).eval()
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(model))
    fact = b"The vault code is 482."
    try:
        session = runtime.open(effort=1)
        runtime.observe(
            session,
            (
                ObservedEvent(
                    MAIN_TRACK, "text", fact, ROLE_TO_ID["document"], event_id="original-fact"
                ),
            ),
            effort=1,
        )
        for number in range(16):
            runtime.observe_text(
                session, MAIN_TRACK, f"Unrelated message {number}".encode(), effort=1
            )
        assert "original-fact" not in session.state.history_event_ids
        restored = runtime.fetch_context(
            session, track=MAIN_TRACK, source_event="original-fact", effort=1
        )
        returned = session.context_archive.get(restored["event_id"])
        assert returned.payload == fact
        assert restored["event_id"] in session.state.history_event_ids
        try:
            runtime.fetch_context(session, track=MAIN_TRACK, source_event="never-seen", effort=1)
        except ValueError:
            missing_refused = True
        else:
            raise AssertionError("Unavailable evidence was invented")
        return {
            "evicted_fact_recovered_exactly": True,
            "filler_messages": 16,
            "missing_source_refused": missing_refused,
            "learned_recall_quality_established": False,
            "retrieval": restored,
        }
    finally:
        runtime.close()


def serve_bundle(path, device):
    bundle = ServedBundle(path, device=device)
    runtime = MultiTrackRuntime(MultiTrackGenerator.of(bundle.model), chunking=bundle.chunking)
    service = MultiTrackService(runtime)
    try:
        opened = service.handle({"op": "open", "effort": 1})
        observed = service.handle(
            {
                "op": "observe_text",
                "session": opened["session"],
                "track": MAIN_TRACK,
                "text": "안녕하세요",
                "effort": 1,
            }
        )
        generated = service.handle(
            {
                "op": "generate",
                "session": opened["session"],
                "effort": 1,
                "seed": 17,
                "outputs": [{"track": MAIN_TRACK, "max_tokens": 4}],
            }
        )
        assert all(result["ok"] for result in (opened, observed, generated))
        return generated
    finally:
        runtime.close()


def mixed(args):
    if not (args.korean and args.media and args.donors):
        raise ValueError("Mixed check needs --korean, --media, and --donors")
    codec = ByteCodec.byte_fallback()
    episodes = [make_episode(778)]
    with args.korean.open() as rows:
        for line in rows:
            episode = Episode.from_dict(json.loads(line))
            if any(len(record.payload) > 512 for record in episode.records):
                continue
            if len(episodes) == 1 and not episode.actions:
                episodes.append(episode)
            elif len(episodes) == 2 and episode.actions:
                episodes.append(episode)
                break
    assert len(episodes) == 3, "Both Korean text and a linked tool episode are required"
    data = args.root / "prepared"
    destination = TensorAssetStore(data)
    with PreparedDataset(args.media, split="train") as corpus:
        found = False
        for episode in corpus.iter_episodes():
            for record in episode.records:
                if record.media_type != MEDIA_RECORD_TYPE:
                    continue
                media = MediaRecord.from_bytes(record.payload)
                corpus.assets.copy_record(media, destination)
                records = (
                    PhysicalRecord(
                        "recording", "user", media.to_bytes(), media_type=MEDIA_RECORD_TYPE
                    ),
                    PhysicalRecord(
                        "reply", "assistant", b"A recorded speech segment.", policy_mask=True
                    ),
                )
                episodes.append(
                    sequential_meridians(
                        Episode(
                            "recorded-media-smoke",
                            records,
                            Lineage("recorded-media-smoke", "v1", episode.lineage.family_id),
                        )
                    )
                )
                found = True
                break
            if found:
                break
    assert found, "A real prepared media observation is required"
    with DedupStore(
        args.root / "spool.sqlite", DedupConfig(train_fraction=1, validation_fraction=0)
    ) as spool:
        with spool.transaction():
            for episode in episodes:
                status = spool.add(episode, codec)
                assert str(status).startswith("accepted"), status
        spool.freeze()
        export_prepared(
            spool,
            data,
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint="local-mixed-smoke-v1",
            chunking=ChunkingConfig().with_hard_limit(512),
        )
    config = TrainConfig(
        purpose="diagnostic",
        model="tiny",
        phase="B0",
        datasets=[str(data)],
        codec="byte",
        allow_byte_codec=True,
        output=str(args.root / "training"),
        device=args.device,
        precision="bf16",
        max_steps=4,
        model_overrides={
            "max_capacity": 512,
            "ngram_rows": 2_500_000,
            "ngram_dim": 16,
            "exact_bigrams": True,
            "ngram_orders": (1, 2, 3, 4, 8),
        },
        engram_imports=[
            str(args.donors / name) for name in ("deepseek-v4.1-flash-1017", "qwen3.8-flash-next")
        ],
        microbatch_episodes=4,
        max_accumulation_windows=1,
        accumulate_scored=1,
        accumulate_ingested=2048,
        # Media descriptors include tensor metadata; text output is still capped
        # separately by the model's 512-position generation bound.
        max_record_bytes=8192,
        max_window_records=8,
        max_history_tokens=2048,
        generated_capacity=512,
        effort=1,
        use_stage_mixture=False,
        use_effort_mixture=False,
        checkpoint_every=0,
        eval_every=0,
        log_every=1,
    )
    first = train(config)
    release()
    resumed = train(replace(config, max_steps=1), resume=first["checkpoint"])
    release()
    _, metadata = load_checkpoint(resumed["checkpoint"], map_location="cpu")
    sampled = metadata["source_sampling"]["source_windows"]
    assert all(sampled.get(episode.lineage.source_id, 0) > 0 for episode in episodes)
    assert resumed["counters"]["media_events"] > 0
    from megatoken.assets import bake_checkpoint

    baked = bake_checkpoint(resumed["checkpoint"], args.root / "baked")
    assert baked["removed_parameters"] == 2 * 128 * 16
    exported = export_served(baked["checkpoint"], args.root / "served")
    generated = serve_bundle(args.root / "served", args.device)
    return {
        "input_episodes": len(episodes),
        "global_step": resumed["global_step"],
        "source_sampling": metadata.get("source_sampling"),
        "baked": baked,
        "export": exported,
        "served_korean_input": True,
        "generation": generated,
        "media_scope": "real recorded media observation; text prediction targets",
        "learned_quality_established": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("restart", "memory", "mixed"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--korean", type=Path)
    parser.add_argument("--media", type=Path)
    parser.add_argument("--donors", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    result = {
        "check": args.check,
        "device": args.device,
        **globals()[args.check](args),
        "passed": True,
    }
    write_json(args.root / "acceptance.json", result)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
