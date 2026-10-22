"""``megatoken collection``: build, extend and inspect source collections.

Torch-free until a raw file has to be prepared; the teacher is loaded only then.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .collection import DEFAULT_THRESHOLD, SourceCollection
from .ingest import FileIngest, add_prepared, expand


def make_preparer(packs: str, root: Path, device: str) -> Any:
    """The one media preparation path (teacher plus codecs) over ``root``."""
    from megatoken.data.media_prepare import MediaPreparer

    return MediaPreparer(packs, root, device=device)


def _fill(collection: SourceCollection, args: argparse.Namespace) -> dict[str, int]:
    roots, files = expand(args.items)
    added = {"prepared": add_prepared(collection, roots, split=args.split) if roots else 0}
    added["files"] = 0
    if files:
        if not args.packs:
            raise ValueError("Preparing raw files needs the locked pack manifest (--packs)")
        preparer = make_preparer(args.packs, collection.path, args.device)
        added["files"] = FileIngest(collection, preparer).add_files(files)
    return added


def _build(args: argparse.Namespace) -> dict[str, Any]:
    from megatoken.teacher import PINNED_TEACHER

    collection = SourceCollection.create(args.output, name=args.name, teacher=PINNED_TEACHER)
    try:
        added = _fill(collection, args)
        if not collection.count:
            raise ValueError("No items to put in the collection")
        collection.set_threshold(args.threshold)
        report = collection.reindex(cells=args.cells, target_recall=args.target_recall)
        return {"added": added, "index": report, **collection.info()}
    finally:
        collection.close()


def _add(args: argparse.Namespace) -> dict[str, Any]:
    with SourceCollection.open(args.path, writable=True) as collection:
        added = _fill(collection, args)
        return {"added": added, **collection.info()}


def _info(args: argparse.Namespace) -> dict[str, Any]:
    with SourceCollection.open(args.path) as collection:
        return collection.info()


def _items(item: argparse.ArgumentParser) -> None:
    item.add_argument(
        "--items",
        nargs="+",
        required=True,
        help="Prepared dataset roots (their feature items are taken as prepared), files, "
        "or directories of files (prepared here with the teacher)",
    )
    item.add_argument("--split", default="train", help="Split of the prepared roots to take")
    item.add_argument("--packs", help="Locked pack manifest, to prepare raw files")
    item.add_argument("--device", default="cuda")


def add_collection_commands(commands: argparse._SubParsersAction) -> None:
    group = commands.add_parser(
        "collection", help="Source collections: original items of any medium, found by embedding"
    )
    sub = group.add_subparsers(dest="collection_command", required=True)
    build = sub.add_parser(
        "build", help="Prepare items, fit the index cells and calibrate the probe count"
    )
    build.add_argument("--name", required=True)
    build.add_argument("--output", required=True, help="Directory of the new collection")
    _items(build)
    build.add_argument("--cells", type=int, help="Index cells (default: about sqrt of the items)")
    build.add_argument("--target-recall", type=float, default=0.97)
    build.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Cosine an item must reach with a query to be observed (default %(default)s)",
    )
    build.set_defaults(func=_build)
    add = sub.add_parser("add", help="Prepare and add items to an existing collection")
    add.add_argument("--path", required=True)
    _items(add)
    add.set_defaults(func=_add)
    info = sub.add_parser("info", help="Items, index and threshold of a collection")
    info.add_argument("--path", required=True)
    info.set_defaults(func=_info)
