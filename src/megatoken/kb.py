"""Building, opening and describing knowledge base files: ``megatoken kb``.

A corpus base is built once, offline: each item is read alone by the model's
ingest heads and written as one entry (``NativeTransducer.ingest_rows``). The
file ships separately from the model bundle and mounts only on the model that
wrote it. The same ``add_items`` fills a user's base while serving.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from megatoken.data.knowledge import KnowledgeItem, episode_items, episode_metadata
from megatoken.io import content_hash
from megatoken.memory.knowledge import (
    FORMAT,
    KnowledgeBase,
    geometry_identity,
    read_file,
)
from megatoken.model.ingest import IngestItem
from megatoken.model.transducer import metadata_payload

# Settings that change how a model runs, not what it computes; the weights digest
# already pins the computation, the architecture hash only names a mismatch.
RUNTIME_FIELDS = (
    "expert_backend",
    "checkpoint_stages",
    "transducer_neighbors",
    "transducer_selected",
    "transducer_exploration",
    "transducer_max_lookups",
    "transducer_max_edges",
    "transducer_byte_budget",
)


def weights_digest(model: Any) -> str:
    """SHA256 over retrieval-compatible served weights in name order.

    Hindsight weights are excluded so adding the serving writer preserves old bases.
    The same trained weights give the same digest whether the model was loaded
    from a checkpoint or from its exported bundle.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.served_state_dict().items()):
        if name.startswith("transducer.hindsight_realizer."):
            continue
        flat = tensor.detach().contiguous().cpu().reshape(-1)
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)};".encode())
        digest.update(flat.view(torch.uint8).numpy())
    return digest.hexdigest()


def model_identity(model: Any) -> dict[str, Any]:
    """What a knowledge base must share with the model that mounts it."""
    config = model.config
    return {
        "format": FORMAT,
        "geometry": geometry_identity(model.transducer.bank.geometry),
        "teacher": config.media_extractor,
        "architecture": content_hash(
            {k: v for k, v in config.to_dict().items() if k not in RUNTIME_FIELDS}
        ),
        "weights": weights_digest(model),
    }


def new_base(
    model: Any, name: str, *, identity: dict[str, Any] | None = None, writable: bool = True
) -> KnowledgeBase:
    """An empty base for this model (``identity`` skips recomputing the weights digest)."""
    return KnowledgeBase(
        model.transducer.bank.geometry,
        identity or model_identity(model),
        name=name,
        writable=writable,
        device=model.transducer.bank.device,
    )


def open_base(
    path: str | Path,
    model: Any,
    *,
    identity: dict[str, Any] | None = None,
    writable: bool = False,
) -> KnowledgeBase:
    """Open a base file for this model; refuses one written by a different model."""
    return KnowledgeBase.load(
        path,
        geometry=model.transducer.bank.geometry,
        identity=identity or model_identity(model),
        device=model.transducer.bank.device,
        writable=writable,
    )


def _groups(items: Sequence[KnowledgeItem], batch: int) -> list[list[KnowledgeItem]]:
    """Runs of whole documents (or single items) of at most ``batch`` items, in first-seen order.

    A document's parts are written in one call, so each part can depart from
    the previous part's destination.
    """
    pieces: dict[str, list[KnowledgeItem]] = {}
    for index, item in enumerate(items):
        pieces.setdefault(item.document or f"\0{index}", []).append(item)
    groups: list[list[KnowledgeItem]] = []
    for piece in pieces.values():
        if groups and len(groups[-1]) + len(piece) <= batch:
            groups[-1].extend(piece)
        else:
            groups.append(piece)
    return groups


@torch.no_grad()
def add_items(
    base: KnowledgeBase,
    model: Any,
    multi: Any,
    items: Sequence[KnowledgeItem],
    *,
    batch: int = 16,
    relations: Sequence[tuple[str, str]] = (),
) -> list[int]:
    """Read each item alone and append its entry to ``base``; the new edge IDs.

    Nothing is retrained. Items are encoded to their native event blocks the way
    the model observes them, then written by the ingest and key heads.
    """
    if not base.writable:
        raise RuntimeError("This knowledge base is read-only")
    device = model.transducer.bank.device
    staged = []
    labels = []
    for group in _groups(items, batch):
        written = []
        for item in group:
            if isinstance(item.record, bytes):
                ids = model.codec.encode(item.record)
                if not ids or len(ids) > model.config.max_capacity:
                    raise ValueError("Raw knowledge item must fit one nonempty model chunk")
                raw = model.core.encode_chunk(torch.tensor([ids], device=device))[0]
            else:
                source = item.record.source
                if source.extractor != model.config.media_extractor:
                    raise ValueError(
                        "Item features come from a different teacher than this model's"
                    )
                features = torch.from_numpy(item.assets.load(source)).to(device)[None]
                raw = model.core.encode_media(features, extractor=source.extractor)[0]
            written.append(
                IngestItem(
                    raw,
                    item.modality,
                    metadata_payload(dict(item.metadata)),
                    item.document,
                    item.part,
                )
            )
        rows = model.transducer.ingest_rows(multi, written)
        staged.append(rows)
        labels.extend(item.source for item in group)
    if not staged:
        return []
    fields = {
        name: torch.cat([getattr(rows, name) for rows in staged])
        for name in ("departure", "key", "value", "destination")
    }
    if relations:
        if len(set(labels)) != len(labels):
            raise ValueError("Navigation reference items require unique source labels")
        index = {name: i for i, name in enumerate(labels)}
        links = list(dict.fromkeys(relations))
        if len(links) > 64:
            raise ValueError("Reference navigation links exceed the per-base bound of 64")
        if any(a not in index or b not in index for a, b in links):
            raise ValueError("Reference graph has a dangling endpoint")
        source = [index[a] for a, _ in links]
        target = [index[b] for _, b in links]
        # Keep metadata entry edges as well: a graph without roots must remain reachable.
        for name in fields:
            extra = fields["destination"][source] if name == "departure" else fields[name][target]
            fields[name] = torch.cat((fields[name], extra))
        labels.extend(f"{a}->{b}" for a, b in links)
    # Finish model work before taking the lock, then publish every group together.
    with base.lock:
        return base.add(**fields, sources=labels).tolist()


def corpus_items(
    roots: Sequence[str | Path], *, split: str | None = "train"
) -> list[KnowledgeItem]:
    """Every item of the prepared roots' episodes, filed under their episode's metadata."""
    from megatoken.data import PreparedCorpus

    items: list[KnowledgeItem] = []
    with PreparedCorpus(roots, split=split) as corpus:
        for index in range(len(corpus)):
            episode, assets = corpus.fetch(index)
            items += episode_items(episode, assets, episode_metadata(episode))
    return items


def describe(path: str | Path) -> dict[str, Any]:
    """What a base file holds, without a model: its identity, size and sources."""
    data = read_file(path)
    rows = data["rows"]
    count = int(rows["ids"].numel())
    labels = data["sources"].values()
    return {
        "name": data["name"],
        "edges": count,
        "sources": len(set(labels)),
        "identity": data["identity"],
        "next_edge_id": data["next_edge_id"],
        "file_bytes": Path(path).stat().st_size,
    }


# ------------------------------------------------------------------ CLI commands


def _load_model(args: argparse.Namespace):
    """The model a base is built for: an exported bundle or a training checkpoint."""
    path = Path(args.model)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if (path / "manifest.json").exists():
        from megatoken.served import ServedBundle

        return ServedBundle(path, packs=args.packs, device=device).model
    from megatoken.loading import load_trained

    return load_trained(path, device=device, codec_path=args.codec, packs=args.packs).model.eval()


def build(args: argparse.Namespace) -> dict[str, Any]:
    from megatoken.model.multitrack import MultiTrackGenerator

    roots = [p for p in args.items if (Path(p) / "manifest.json").is_file()]
    files = [p for p in args.items if p not in roots]
    if roots and files:
        raise ValueError("Give prepared roots or files, not both")
    model = _load_model(args)
    if roots:
        items = corpus_items(roots, split=None if args.split == "all" else args.split)
    else:
        if not args.packs:
            raise ValueError("Preparing files needs the locked pack manifest: --packs")
        from megatoken.data.knowledge import prepare_files
        from megatoken.data.media_prepare import MediaPreparer

        metadata = {
            key: value
            for key in ("task", "topic", "language", "date")
            if (value := getattr(args, key)) is not None
        }
        preparer = MediaPreparer(
            args.packs, args.prepared, device=str(model.transducer.bank.device)
        )
        items = prepare_files(files, preparer, metadata=metadata)
    if not items:
        raise ValueError("No items to write")
    base = new_base(model, args.name)
    add_items(base, model, MultiTrackGenerator(model), items, batch=args.batch)
    output = Path(args.output or f"{args.name}.kb")
    base.save(output)
    return {"output": str(output), **base.summary()}


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    return describe(args.base)
