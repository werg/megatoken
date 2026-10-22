"""Minimal valid training setup (tiny model, CPU, fp32, byte codec).

Grouped native-meridian episodes: a document and a user request observed in
meridian 0, then a generated assistant answer in meridian 1 (and, with
``turns``, further request/answer meridians).
"""

from __future__ import annotations

from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.data.storage import export_prepared
from megatoken.generation.codec import ByteCodec
from megatoken.training.config import TrainConfig

# Model preset for every artifact; scripts may switch it (e.g. "32x512") for GPU runs.
MODEL = "tiny"
MODEL_OVERRIDES: dict = {}
# For windows of text-only episodes: nothing is ever loaded from it.
TEXT_ONLY_ASSETS = TensorAssetStore("text-only")


def make_episode(index: int, *, turns: int = 1) -> Episode:
    """Meridian 0 observes doc+request; then answer, [request2, answer2, ...]."""
    doc = f"document {index}: the key {index * 7 + 3} opens the vault".encode()
    request = f"request {index}: which key opens the vault?".encode()
    answer = f"answer {index}: key {index * 7 + 3}".encode()
    names = ["doc", "request", "answer"]
    records = [
        PhysicalRecord("doc", "document", doc),
        PhysicalRecord("request", "user", request),
        PhysicalRecord("answer", "assistant", answer, turn_index=1, policy_mask=True),
    ]
    meridians = [
        Meridian(
            0,
            tuple(
                TrackEvent(n, n, "text", 0, r.role)
                for n, r in zip(names[:2], records[:2], strict=True)
            ),
        ),
        Meridian(1, (TrackEvent("answer", "answer", "text", 1, "assistant"),)),
    ]
    for turn in range(2, turns + 1):
        ask, reply = f"request{turn}", f"answer{turn}"
        records += [
            PhysicalRecord(
                ask,
                "user",
                f"request {index}.{turn}: repeat the key?".encode(),
                turn_index=2 * turn - 2,
            ),
            PhysicalRecord(
                reply,
                "assistant",
                f"answer {index}.{turn}: {index * 7 + 3}".encode(),
                turn_index=2 * turn - 1,
                policy_mask=True,
            ),
        ]
        names += [ask, reply]
        meridians += [
            Meridian(len(meridians), (TrackEvent(ask, ask, "text", 2 * turn - 2, "user"),)),
            Meridian(
                len(meridians) + 1, (TrackEvent(reply, reply, "text", 2 * turn - 1, "assistant"),)
            ),
        ]
    records, meridians = tuple(records), tuple(meridians)
    tracks = tuple(TrackSpec(n, r.role) for n, r in zip(names, records, strict=True))
    return Episode(
        f"episode-{index}",
        records,
        Lineage("e2e", "v1", f"family-{index}"),
        tracks=tracks,
        meridians=meridians,
    )


def prepare_dataset(root: Path, episodes: list[Episode], *, heldout: bool = False) -> Path:
    """Prepared byte-codec episodes, entirely in the train or the validation split."""
    codec = ByteCodec.byte_fallback()
    store = DedupStore(
        root / "spool.sqlite",
        DedupConfig(train_fraction=0 if heldout else 1, validation_fraction=1 if heldout else 0),
    )
    try:
        with store.transaction():
            for episode in episodes:
                store.add(episode, codec)
        store.freeze()
        export_prepared(
            store,
            root / "data",
            codec=codec,
            source_receipts=[],
            pipeline_fingerprint="training-e2e-v1",
        )
    finally:
        store.close()
    return root / "data"


def tiny_train_config(root: Path, *, episodes: int = 4, **overrides) -> TrainConfig:
    """A valid tiny CPU TrainConfig; ``root`` receives the prepared data and the run."""
    root = Path(root)
    items = [make_episode(i) for i in range(episodes)]
    dataset = prepare_dataset(root, items)
    values = dict(
        phase="B0",
        datasets=[str(dataset)],
        output=str(root / "run"),
        model=MODEL,
        model_overrides=dict(MODEL_OVERRIDES),
        codec="byte",
        device="cpu",
        precision="fp32",
        max_steps=1,
        microbatch_episodes=2,
        accumulate_scored=1,
        accumulate_ingested=1000,
        max_accumulation_windows=2,
        generated_capacity=16,
        max_history_tokens=256,
        max_record_bytes=128,
        checkpoint_every=0,
        eval_every=0,
        effort=1,
        gradient_checkpointing=False,
        use_stage_mixture=False,
        use_effort_mixture=False,
    )
    values.update(overrides)
    return TrainConfig(**values)
