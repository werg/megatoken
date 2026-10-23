"""Explicit conversion of older, automatically grouped linear transcripts.

Only the former role-based linear grouping is understood here. Native parallel
groups, selected action tapes, and semantic track controls require their original
reader and an explicit translation rather than guessing a new causal structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from megatoken.io import content_hash, sha256_file, write_json

from .schema import SCHEMA_VERSION, Episode, canonical_json, sequential_meridians


def convert_linear_episode(value: dict) -> Episode:
    version = value.get("schema_version")
    if version == SCHEMA_VERSION:
        return Episode.from_dict(value)
    if type(version) is not int or version not in (3, 4):
        raise ValueError("Only legacy linear episode versions 3 and 4 can be converted")
    annotations = json.loads(value.get("supervision", {}).get("annotations_json", "{}"))
    if annotations.get("native_action_tapes") or value.get("track_revisions"):
        raise ValueError(
            "Native action tapes or track revisions need an explicit causal translation"
        )
    records = value.get("records", [])
    groups = value.get("meridians", [])
    tracks = value.get("tracks", [])
    roles = {record["role"] for record in records}
    allowed_tracks = roles | {"main"}
    for track in tracks:
        if (
            track["track_id"] not in allowed_tracks
            or track.get("description", "")
            or track.get("settings_json", "{}") != "{}"
            or track.get("default_modality", "text") != "text"
        ):
            raise ValueError("Custom track meaning cannot be inferred from a legacy transcript")
    if len(groups) != len(records):
        raise ValueError("Only one-record, sequential legacy groups can be converted")
    for index, (record, group) in enumerate(zip(records, groups, strict=True)):
        if group["meridian_id"] != index or group.get("selected_at_ns") is not None:
            raise ValueError(
                "Legacy group timing is not an automatically grouped linear transcript"
            )
        events = group.get("events", [])
        if len(events) != 1:
            raise ValueError("Simultaneous legacy events need an explicit causal translation")
        event = events[0]
        if (
            event["record_id"] != record["record_id"]
            or event["track_id"] not in allowed_tracks
            or event.get("role", record["role"]) != record["role"]
            or event.get("meridian_id", index) != index
            or event.get("request")
            or event.get("request_revision", 0)
            or event.get("reference_event_ids")
            or event.get("instruction")
            or event.get("alignment_id")
            or event.get("media_span")
            or event.get("settings_json", "{}") != "{}"
        ):
            raise ValueError("Legacy event carries causal controls that cannot be discarded")
    converted = deepcopy(value)
    converted.update(schema_version=SCHEMA_VERSION, tracks=[], meridians=[])
    converted.pop("track_revisions", None)
    return sequential_meridians(Episode.from_dict(converted))


def convert_corpus(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    partial = output / "episodes.jsonl.partial"
    input_hash, output_hash = hashlib.sha256(), hashlib.sha256()
    versions = {}
    count = 0
    with source.open("rb") as reader, partial.open("xb") as writer:
        for line in reader:
            input_hash.update(line)
            if not line.strip():
                continue
            value = json.loads(line)
            episode = convert_linear_episode(value)
            encoded = (canonical_json(episode.to_dict()) + "\n").encode()
            writer.write(encoded)
            output_hash.update(encoded)
            version = str(value.get("schema_version"))
            versions[version] = versions.get(version, 0) + 1
            count += 1
        writer.flush()
        os.fsync(writer.fileno())
    if not count:
        raise ValueError("The source corpus is empty")
    destination = output / "episodes.jsonl"
    partial.replace(destination)
    receipt = {
        "input": str(source.resolve()),
        "input_sha256": input_hash.hexdigest(),
        "input_versions": versions,
        "output_version": SCHEMA_VERSION,
        "episodes": count,
        "output_sha256": output_hash.hexdigest(),
        "output_size_bytes": destination.stat().st_size,
        "conversion": "role-grouped-linear-to-continuing-conversation-v1",
        "original_records_actions_and_lineage_preserved": True,
    }
    # Retain the original redistribution terms and quality requirements when
    # converting a previously checksummed native-source addendum.
    source_manifest = source.parent / "source-manifest.json"
    if source_manifest.is_file():
        manifest = json.loads(source_manifest.read_text())
        if len(manifest["sources"]) != 1:
            raise ValueError("A native conversion manifest needs exactly one source")
        item = manifest["sources"][0]
        declared = item.get("files", [])
        if len(declared) != 1 or declared[0].get("path") != source.name:
            raise ValueError("The original manifest must identify the input file exactly")
        if (
            declared[0].get("sha256") != receipt["input_sha256"]
            or declared[0].get("size_bytes") != source.stat().st_size
        ):
            raise ValueError("The original corpus differs from its declared checksum or size")
        item.update(
            source_id=item["source_id"] + f"-format{SCHEMA_VERSION}",
            uri=destination.name,
            format="jsonl",
            adapter="native",
            max_rows=count,
            files=[
                {
                    "path": destination.name,
                    "sha256": receipt["output_sha256"],
                    "size_bytes": receipt["output_size_bytes"],
                }
            ],
        )
        write_json(output / "source-manifest.json", manifest)
    write_json(output / "receipt.json", receipt)
    return receipt


def convert_prepared(source: Path, output: Path, codec):
    """Re-export a checksummed old prepared root with its original family splits."""
    from .chunking import ChunkingConfig
    from .dedup import DedupConfig, DedupStore
    from .media_assets import MEDIA_RECORD_TYPE, MediaRecord, TensorAssetStore
    from .storage import export_prepared

    source = source.resolve()
    if output.resolve().is_relative_to(source):
        raise ValueError("Conversion output must be separate from the immutable source")
    manifest = json.loads((source / "manifest.json").read_text())
    unsigned = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if not manifest.get("complete") or content_hash(unsigned) != manifest.get("fingerprint"):
        raise ValueError("Original prepared manifest is incomplete or has an invalid fingerprint")
    if manifest.get("schema_version") not in (3, 4, 5):
        raise ValueError("Unsupported prepared dataset version")
    if codec.fingerprint != manifest["codec_fingerprint"]:
        raise ValueError("Conversion must retain the original codec")

    def checked_path(relative):
        path = (source / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(source):
            raise ValueError("Prepared input path escapes its source root")
        return path

    for file in [manifest["index"], *manifest["shards"]]:
        if sha256_file(checked_path(file["path"])) != file["sha256"]:
            raise ValueError("Original prepared file checksum mismatch")
    output.mkdir(parents=True, exist_ok=False)
    original_assets, destination_assets = TensorAssetStore(source), TensorAssetStore(output)
    count, splits = 0, {}
    original_index = checked_path(manifest["index"]["path"])
    with sqlite3.connect(f"file:{original_index}?mode=ro&immutable=1", uri=True) as index:
        index.row_factory = sqlite3.Row
        with DedupStore(
            output / "conversion.sqlite", DedupConfig(**manifest.get("dedup", {}))
        ) as store:
            with store.transaction():
                for row in index.execute("SELECT * FROM episodes ORDER BY entry_id"):
                    with checked_path(row["shard"]).open("rb") as handle:
                        handle.seek(row["offset"])
                        line = handle.read(row["length"])
                    if hashlib.sha256(line).hexdigest() != row["sha256"]:
                        raise ValueError("Original episode differs from its indexed checksum")
                    episode = convert_linear_episode(json.loads(line))
                    if episode.episode_id != row["episode_id"]:
                        raise ValueError("Original episode differs from its indexed identity")
                    ancestors = set(episode.lineage.family_keys) - {row["family_id"]}
                    episode = replace(
                        episode,
                        lineage=replace(
                            episode.lineage,
                            family_id=row["family_id"],
                            ancestor_ids=tuple(sorted(ancestors)),
                            frozen_split=row["split"],
                        ),
                    )
                    for record in episode.records:
                        if record.media_type == MEDIA_RECORD_TYPE:
                            original_assets.copy_record(
                                MediaRecord.from_bytes(record.payload), destination_assets
                            )
                    status = store.add(episode, codec)
                    if not status.startswith("accepted"):
                        raise ValueError(
                            f"Conversion would drop an original retained episode: {status}"
                        )
                    count += 1
                    splits[row["split"]] = splits.get(row["split"], 0) + 1
            if manifest["counts"].get("*", 0) != count:
                raise ValueError("Original manifest count differs from its indexed episodes")
            for split, expected in splits.items():
                if manifest["counts"].get(f"split:{split}", 0) != expected:
                    raise ValueError("Original manifest split count differs from its index")
            store.freeze()
            identity = {
                "conversion": "legacy-linear-prepared-v1",
                "original": manifest["fingerprint"],
                "episode_version": SCHEMA_VERSION,
                "codec": codec.fingerprint,
            }
            exported = export_prepared(
                store,
                output,
                codec=codec,
                source_receipts=manifest.get("sources", []),
                pipeline_fingerprint=content_hash(identity),
                chunking=ChunkingConfig(**manifest.get("chunking", {}).get("config", {})),
            )
    if exported["counts"].get("*", 0) != count:
        (output / "manifest.json").unlink()
        raise ValueError("Conversion did not retain every indexed episode")
    for split, expected in splits.items():
        if exported["counts"].get(f"split:{split}", 0) != expected:
            (output / "manifest.json").unlink()
            raise ValueError("Conversion changed an original train/evaluation split")
    receipt = {
        **identity,
        "episodes": count,
        "splits": splits,
        "output_fingerprint": exported["fingerprint"],
    }
    write_json(output / "conversion-receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", type=Path)
    inputs.add_argument("--prepared-input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--codec", help="Original codec, required for --prepared-input")
    args = parser.parse_args()
    if args.prepared_input:
        if not args.codec:
            parser.error("--prepared-input requires its original --codec")
        from megatoken.loading import load_codec

        receipt = convert_prepared(args.prepared_input, args.output, load_codec(args.codec))
    else:
        receipt = convert_corpus(args.input, args.output)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
