"""Offline teacher sidecar for the unchanged production corpus and stage mixture."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import torch

from megatoken.io import atomic_binary_writer, canonical_json, content_hash, sha256_file, write_json

from .hindsight import (
    FORMAT,
    HindsightTargets,
    TargetEncoding,
    layout,
    suffixes,
    target_annotation,
    window_key,
    window_settings,
)
from .media_assets import TensorAssetStore
from .relations import EXTRACTOR, local_relations


def supported_text_window(episode, end):
    ids = {r.record_id for r in episode.records[:end]}
    return all(
        event.modality == "text"
        for group in episode.meridians
        for event in group.events
        if event.record_id in ids
    )


def source_windows(config, codec, model_config):
    from megatoken.data import PreparedCorpus
    from megatoken.training.engine import window_config
    from megatoken.training.stream import _WindowPlan

    with PreparedCorpus(config.datasets, split="train", verify_checksums=False) as corpus:
        for episode in corpus.iter_episodes():
            for window in _WindowPlan(episode, codec, window_config(config)):
                if not supported_text_window(episode, window.end):
                    yield episode, window, None, None
                    continue
                groups, before = layout(episode, window.end, model_config.transducer_max_lookups)
                yield episode, window, before, suffixes(groups, before)


def prepare_corpus(config, output, writer, *, fit_windows=8, encoding_from=None):
    """Publish only after every train window is indexed. Original roots remain unchanged."""
    from megatoken.data import PreparedCorpus
    from megatoken.loading import load_codec, load_model_config

    from .recipes import validate_teacher_encoding_source

    validate_teacher_encoding_source(config, encoding_from)
    model_config = load_model_config(config.model, **config.model_overrides)
    rows, width = model_config.transducer_value_rows, model_config.facet_dim
    codec = load_codec(config.codec, vocab_size=model_config.vocab_size)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from .recipes import validate_stage_supply

    with PreparedCorpus(config.datasets) as corpus:
        validate_stage_supply(config, corpus)
        corpus_identity = corpus.identity
    if config.eval_every:
        with PreparedCorpus(config.datasets, split="validation") as validation:
            if not len(validation):
                raise ValueError("Configured evaluation has no held-out validation supply")
            validate_stage_supply(config, validation, split="validation")
    previous = None
    if encoding_from is not None:
        previous = HindsightTargets(encoding_from, value_dim=rows * width, metadata_only=True)
        if previous.manifest["teacher"] != writer.identity:
            raise ValueError("Previous encoding uses a different frozen teacher")
    identity = content_hash(
        {
            "teacher": writer.identity,
            "corpus": corpus_identity,
            "settings": window_settings(config),
            "rows": rows,
            "width": width,
            "max_lookups": model_config.transducer_max_lookups,
            "navigation_extractor": EXTRACTOR,
            "fit_windows": fit_windows,
            "previous": None if previous is None else previous.manifest["encoding_identity"],
        }
    )
    intent = output / "preparation.json"
    if intent.exists() and json.loads(intent.read_text())["identity"] != identity:
        raise ValueError("Production teacher preparation identity changed")
    write_json(intent, {"identity": identity})
    if (output / "manifest.json").exists():
        targets = HindsightTargets(output, config=config, codec=codec, value_dim=rows * width)
        targets.validate_corpus(corpus_identity)
        targets.close()
        return targets.manifest
    encoding_path = output / "encoding.pt"
    if encoding_path.exists():
        encoding = TargetEncoding(**torch.load(encoding_path, weights_only=True))
    elif previous is not None:
        with atomic_binary_writer(encoding_path) as stream:
            stream.write((previous.root / "encoding.pt").read_bytes())
        encoding = TargetEncoding(**torch.load(encoding_path, weights_only=True))
    else:
        samples, seen = [], 0
        for _, _, before, views in source_windows(config, codec, model_config):
            if before is None:
                continue
            for view in views:
                if view is not None:
                    pooled, mask, _ = writer.encode(view[0], rows=rows)
                    samples.append((pooled, mask))
                    if len(samples) >= 32:
                        break
            seen += 1
            if len(samples) >= 32 or seen >= fit_windows:
                break
        if not samples:
            raise ValueError(
                "No text future groups to fit; reuse the previous stage's --encoding-from"
            )
        encoding = TargetEncoding.fit(samples, rows=rows, width=width)
        with atomic_binary_writer(encoding_path) as stream:
            torch.save(encoding.state_dict(), stream)
    if previous is not None and sha256_file(encoding_path) != sha256_file(
        previous.root / "encoding.pt"
    ):
        raise ValueError("Resumed encoding differs from the previous stage")
    encoding_identity = content_hash(
        {
            "teacher": writer.identity,
            "projection": sha256_file(encoding_path),
            "rows": rows,
            "width": width,
        }
    )
    assets = TensorAssetStore(output)
    building = output / "targets.building.sqlite"
    committed = output / "targets.sqlite"
    # An interrupted final rename remains resumable; the manifest is the commit marker.
    if committed.exists() and not building.exists():
        committed.rename(building)
    stats = dict(text_windows=0, unsupported_windows=0, target_values=0, navigation_examples=0)
    with sqlite3.connect(building) as index:
        index.execute(
            "CREATE TABLE IF NOT EXISTS windows(key TEXT PRIMARY KEY, row_json TEXT NOT NULL)"
        )
        for episode, window, before, views in source_windows(config, codec, model_config):
            key = window_key(episode, window.start, window.end)
            available = {
                r.record_id
                for r in episode.records[window.start : window.end]
                if r.media_type.startswith("text/")
            }
            links = [
                r
                for r in local_relations(episode)
                if r["left"] in available and r["right"] in available
            ]
            stats["navigation_examples"] += min(len(links), config.navigation_examples)
            stored = index.execute("SELECT row_json FROM windows WHERE key=?", (key,)).fetchone()
            if stored is not None:
                row = json.loads(stored[0])
            elif before is None:
                row = {"unsupported": "non_text_window"}
            else:
                targets = []
                for view in views:
                    if view is None:
                        targets.append(None)
                        continue
                    text, spans = view
                    pooled, mask, extent = writer.encode(text, rows=rows)
                    asset = assets.put_hindsight(
                        encoding.encode(pooled, mask),
                        extractor="hindsight:" + encoding_identity,
                        source_family=content_hash({"text": text, "byte_end": extent}),
                    )
                    spans = [
                        {**s, "end": min(s["end"], extent)} for s in spans if s["start"] < extent
                    ]
                    targets.append(target_annotation(asset, spans, extent))
                row = {"plan": before, "targets": targets}
            if row.get("unsupported"):
                stats["unsupported_windows"] += 1
            else:
                stats["text_windows"] += 1
                stats["target_values"] += sum(t is not None for t in row["targets"])
            if stored is None:
                index.execute(
                    "INSERT INTO windows VALUES(?,?)", (key, canonical_json(row).decode())
                )
                index.commit()
    building.rename(committed)
    manifest = dict(
        format=FORMAT,
        scope="text_windows",
        corpus_identity=corpus_identity,
        teacher=writer.identity,
        encoding_identity=encoding_identity,
        codec=codec.fingerprint,
        value_dim=rows * width,
        max_lookups=model_config.transducer_max_lookups,
        navigation_extractor=EXTRACTOR,
        window_settings=window_settings(config),
        memory_policy="teacher-text-only-v1",
        index={"path": committed.name, "sha256": sha256_file(committed)},
        coverage=stats,
    )
    training = {
        **asdict(config),
        "hindsight_targets": config.hindsight_targets or str(output.resolve()),
    }
    # Preserve output, mixtures, decoder packs, all roots and validation settings.
    write_json(output / "train.json", training)
    write_json(output / "manifest.json", manifest)
    return manifest
