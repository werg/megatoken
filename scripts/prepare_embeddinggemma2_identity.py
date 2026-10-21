"""Build identity E->R media tasks from pinned hidden-cache and native target assets.

Input JSONL rows:
 {"teacher_descriptor":"/absolute/or/relative/feature.json",
  "target_asset":{... TensorAsset.asdict ...}, "family_id":"original-source-family"}

The teacher_descriptor is emitted by cache_embeddinggemma2_hidden.py with
--identity-target-sha256 equal to the target_asset SHA. Both underlying assets
must already live under --output. Optional --text-episodes-jsonl imports
schema-v3 original native text episodes into the required text training stream.
This script never generates media or teacher labels and cannot fill missing data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from megatoken.data import DedupConfig, DedupStore, Episode, Lineage, PhysicalRecord
from megatoken.data.media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAsset, TensorAssetStore
from megatoken.data.schema import Meridian, TrackEvent, TrackSpec
from megatoken.data.storage import export_prepared
from megatoken.loading import load_codec
from megatoken.omni.embeddinggemma2_hidden import HIDDEN_TAP


def read_rows(path: Path):
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            yield index, json.loads(line)


def make_identity_episode(store, record, *, source_path: Path, index: int):
    modality = record["modality"]
    if modality not in ("audio", "image", "video"):
        raise ValueError("Identity warmup supports only physical media")
    descriptor_path = (source_path.parent / record["teacher_descriptor"]).resolve()
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    target = TensorAsset.from_dict(record["target_asset"])
    features = TensorAsset.from_dict(descriptor["asset"])
    if (
        descriptor.get("format") != "megatoken.embeddinggemma2.hidden.v1"
        or descriptor.get("modality") != modality
        or descriptor.get("hidden_tap") != HIDDEN_TAP
        or not features.extractor.startswith("embeddinggemma2:")
        or features.shape[-1] != 512
        or features.kind != "features"
        or target.kind not in ("raw", "codec_latents")
        or target.modality != modality
        or features.modality != modality
        or features.source_family != target.sha256
        or descriptor.get("identity_target_sha256") != target.sha256
    ):
        raise ValueError("Identity pair must pin unpooled teacher features of the exact output")
    if target.kind == "codec_latents" and not target.codec_id:
        raise ValueError("Use pinned native codec latent targets, not anonymous legacy latents")
    # Hard-fail on absent or tampered teacher and target bytes before preparing any episode.
    store.load(features)
    store.load(target)
    family_id = record.get("family_id")
    if not isinstance(family_id, str) or not family_id:
        raise ValueError("Identity tasks require a stable original-source family_id")
    episode_id = f"embeddinggemma2-{modality}-{target.sha256[:16]}-{index}"
    note = f"Learn to reproduce this observed {modality} from its internal features.".encode()
    refs = MediaRecord(modality, features, target)
    records = (
        PhysicalRecord("prompt", "user", note),
        PhysicalRecord(
            "reproduction", "assistant", refs.to_bytes(), turn_index=1,
            policy_mask=True, media_type=MEDIA_RECORD_TYPE, provenance="teacher",
        ),
    )
    tracks = (TrackSpec("prompt", "user"), TrackSpec("reproduction", "assistant"))
    meridians = (
        Meridian(0, (TrackEvent("prompt", "prompt", "text", 0, "user"),)),
        Meridian(
            1, (
                TrackEvent(
                    "reproduction", "reproduction", modality, 1, "assistant",
                    asset_sha256=target.sha256,
                ),
            ),
        ),
    )
    teacher_rev = descriptor.get("teacher_revision", "")
    if len(teacher_rev) != 40:
        raise ValueError("Unpinned original teacher model revision")
    return Episode(
        episode_id, records, Lineage("embeddinggemma2-identity", teacher_rev, family_id),
        stream=f"embeddinggemma2_{modality}_identity",
        tracks=tracks, meridians=meridians,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", required=True, help="Exact text codec; use 'byte' for offline smoke tests")
    parser.add_argument("--text-episodes-jsonl", type=Path, default=None)
    args = parser.parse_args()
    root = args.output
    if (root / "manifest.json").exists():
        parser.error("Prepared manifest already exists; never replace immutable training data")
    root.mkdir(parents=True, exist_ok=True)
    store = TensorAssetStore(root)
    codec = load_codec(args.codec)
    dedup = DedupStore(
        root / "preparation.sqlite",
        DedupConfig(train_fraction=0.9, validation_fraction=0.1),
    )
    counters = {"audio": 0, "image": 0, "video": 0, "text": 0}
    try:
        with dedup.transaction():
            for index, record in read_rows(args.pairs_jsonl):
                episode = make_identity_episode(
                    store, record, source_path=args.pairs_jsonl, index=index
                )
                dedup.add(episode, codec)
                counters[record["modality"]] += 1
            if args.text_episodes_jsonl is not None:
                for _, record in read_rows(args.text_episodes_jsonl):
                    episode = Episode.from_dict(record)
                    if any(
                        e.modality != "text"
                        for m in episode.meridians for e in m.events
                    ):
                        raise ValueError("Native text warmup must not smuggle media through raw bytes")
                    dedup.add(replace(episode, stream="embeddinggemma2_text_pretraining"), codec)
                    counters["text"] += 1
        if not all(counters[m] for m in ("audio", "image", "video")):
            raise ValueError("All three media identity task families are required")
        if not counters["text"]:
            raise ValueError("Import native original text episodes to populate the text lane")
        dedup.freeze()
        result = export_prepared(
            dedup, root, codec=codec,
            source_receipts=[{
                "name": "embeddinggemma2_hidden_identity_pairs",
                "pairs_sha256": hashlib.sha256(args.pairs_jsonl.read_bytes()).hexdigest(),
                "text_sha256": (
                    hashlib.sha256(args.text_episodes_jsonl.read_bytes()).hexdigest()
                    if args.text_episodes_jsonl else None
                ),
                "counts": counters,
            }],
            pipeline_fingerprint="embeddinggemma2-internal-hidden-identity-v1",
        )
    finally:
        dedup.close()
    print(json.dumps({"counts": counters, "manifest": result.get("schema_version")}, indent=2))


if __name__ == "__main__":
    main()
