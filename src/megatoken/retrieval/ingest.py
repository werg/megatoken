"""Files and prepared items into a collection, through the one preparation path.

Raw files are prepared exactly as for training: media by the media preparer
(``megatoken.data.media_prepare``: the teacher reads the file, features and the
pooled embedding are stored), text by the teacher with the document prompt, long
text cut into the teacher's own pieces and kept as features
(``megatoken.data.compression``). Items already prepared in a dataset root are
taken as they are. The preparer is anything with a ``teacher``, a ``store`` (a
``TensorAssetStore``) and ``media(modality, path, target=False)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from megatoken.data.compression import DOCUMENT_PROMPT, CompressionPolicy, DocumentCompressor
from megatoken.data.media_assets import MEDIA
from megatoken.data.storage import PreparedCorpus

from .collection import TEXT_LIMIT, SourceCollection

# The teacher's task prompt for the question side of a search (items use the document prompt).
QUERY_PROMPT = "task: search result | query: "
SUFFIXES = {
    "audio": {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"},
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
    "video": {".mp4", ".mkv", ".webm", ".mov", ".avi"},
}
SPLITS = ("train", "validation", "test")


def modality_of(path: Path) -> str:
    """``audio``, ``image`` or ``video`` by the file's suffix; anything else is read as text."""
    suffix = path.suffix.lower()
    return next((name for name, suffixes in SUFFIXES.items() if suffix in suffixes), "text")


def is_prepared_root(path: Path) -> bool:
    return (path / "manifest.json").is_file()


def expand(items: Iterable[str | Path]) -> tuple[list[Path], list[Path]]:
    """Split ``--items`` into prepared dataset roots and the files to prepare.

    A directory holding a manifest is a prepared root; any other directory
    contributes every file under it, in sorted order.
    """
    roots, files = [], []
    for item in map(Path, items):
        if item.is_dir() and is_prepared_root(item):
            roots.append(item)
        elif item.is_dir():
            files += sorted(path for path in item.rglob("*") if path.is_file())
        elif item.is_file():
            files.append(item)
        else:
            raise FileNotFoundError(f"No such file or prepared dataset: {item}")
    return roots, files


def add_prepared(
    collection: SourceCollection, roots: Iterable[Path], *, split: str = "train"
) -> int:
    """Every prepared feature item of the roots' ``split`` (compressed text, audio, images,
    video), by the original's hash. Items the collection already holds are skipped."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    added = 0
    with PreparedCorpus(list(roots), split=split) as corpus:
        for modality in ("text", *MEDIA):
            for position in range(corpus.feature_count(split, modality)):
                record, assets = corpus.feature_item(split, modality, position)
                origin = f"sha256:{record.source.source_family}"
                added += collection.add_record(record, assets, origin=origin) is not None
    return added


class FileIngest:
    """Prepares files with one preparer and adds their items to one collection."""

    def __init__(self, collection: SourceCollection, preparer: Any) -> None:
        self.collection, self.preparer = collection, preparer
        self.compressor = DocumentCompressor(
            preparer.teacher, preparer.store, CompressionPolicy(min_bytes=1)
        )

    def add_file(self, path: str | Path) -> int:
        """Prepare one file; the number of new items (a long text gives several)."""
        path = Path(path)
        modality = modality_of(path)
        if modality != "text":
            record = self.preparer.media(modality, path, target=False)
            return (
                self.collection.add_record(record, self.preparer.store, origin=str(path))
                is not None
            )
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        if not payload.strip():
            raise ValueError(f"{path} is empty")
        if len(payload) <= TEXT_LIMIT:
            embedding = self.preparer.teacher.encode("text", text, prompt=DOCUMENT_PROMPT)
            key = embedding.embedding.numpy()
            return self.collection.add_text(payload, key, origin=str(path)) is not None
        added = 0
        for offset, piece, record in self.compressor.descriptors(payload):
            origin = f"{path}#bytes={offset}-{offset + len(piece)}"
            added += (
                self.collection.add_record(record, self.preparer.store, origin=origin) is not None
            )
        return added

    def add_files(self, paths: Iterable[str | Path]) -> int:
        """Prepare and add files; the collection is indexed afterwards if it was not yet."""
        added = sum(self.add_file(path) for path in paths)
        if not self.collection.indexed and self.collection.count:
            self.collection.reindex()
        return added
