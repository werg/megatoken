"""Public media datasets -> media-identity and concordant task episodes.

``configs/data/sources.media.json`` lists every source: the pinned Hub revision, the licence
found on its card with the maintainer's recorded approval, which files to read, and which
*reader* (how the files are laid out) and *task* (what the labels mean) apply. A reader turns
one source's files into plain *records* (a row, a WebDataset sample, an archive member joined
with its table row); a task maps a record's labelled fields (``columns``) to a *clip*, a media
file on local disk plus the labels the publisher supplies, and then to an aligned bundle whose
artifacts ``MediaPreparer`` prepares (any object with the same ``media(modality, path)``
method works). ``data/concordance.py`` compiles the bundle into ordinary episodes whose stream
is the task family named in ``stages.json`` (``transcription``, ``image_edit``, ...).

Files are never mirrored first: they are read in place, streamed from the pinned Hub
revision (content addressed, so the revision is the integrity pin) or, when
``<inputs>/<source_id>`` exists, from that local directory of identically named files. Each
clip's files are deleted once its episodes are built, so a run needs scratch only for a
handful of clips. Splits are frozen here, per original source family (speaker, sentence,
photo, recording), so no family straddles train/validation.

Procedural edits (``speech_edits``, ``video_edits``) are real, verified edits: a deterministic
ffmpeg program applied to a real recording, recorded with ``executed`` provenance. They stand
in for human-made speech/video edit data, which no open dataset provides.
"""

from __future__ import annotations

import csv
import dataclasses
import glob
import hashlib
import io
import json
import re
import subprocess
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

from .concordance import (
    AlignedBundle,
    BundleArtifact,
    compile_task,
    standard_views,
)
from .media_assets import MEDIA_RECORD_TYPE, MediaRecord
from .schema import Episode, Lineage, Meridian, PhysicalRecord, TrackEvent, TrackSpec
from .sources import TermsReview

VALIDATION_PERCENT = 5
MIN_SPEECH_SECONDS = 0.5
MAX_SPEECH_SECONDS = 20.0
MAX_SIDE = 1024
_PACKAGE_TREE_MANIFEST = Path(__file__).resolve().parents[3] / "configs/data/sources.media.json"
_CHECKOUT_MANIFEST = Path.cwd() / "configs/data/sources.media.json"
MANIFEST = _PACKAGE_TREE_MANIFEST if _PACKAGE_TREE_MANIFEST.is_file() else _CHECKOUT_MANIFEST


@dataclass(frozen=True, slots=True)
class OpenSource:
    """One pinned dataset: where it is, what its terms are, and how to read it."""

    source_id: str
    repository: str
    revision: str
    reader: str
    task: str
    terms: TermsReview
    include: tuple[str, ...] = ()
    columns: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    inventory: Mapping[str, Any] = field(default_factory=dict)
    gated: bool = False
    caveat: str = ""

    @property
    def license(self) -> str:
        return self.terms.license

    @property
    def streams(self) -> tuple[str, ...]:
        """Every stage-plan stream this source can feed (procedural edits included)."""
        task = TASKS[self.task]
        return tuple(sorted(task.streams | ({task.edit_stream} if task.edit_stream else set())))

    def with_options(self, **overrides: Any) -> OpenSource:
        return dataclasses.replace(self, options={**self.options, **overrides})


def load_sources(path: Path = MANIFEST) -> dict[str, OpenSource]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError("Unknown media source manifest schema")
    sources = {}
    for row in raw["sources"]:
        source = OpenSource(
            source_id=row["source_id"],
            repository=row["repository"],
            revision=row["revision"],
            reader=row["reader"],
            task=row["task"],
            terms=TermsReview(**row["terms"]),
            include=tuple(row.get("include", ())),
            columns=row.get("columns", {}),
            options=row.get("options", {}),
            inventory=row.get("inventory", {}),
            gated=row.get("gated", False),
            caveat=row.get("caveat", ""),
        )
        if source.source_id in sources:
            raise ValueError(f"Duplicate media source {source.source_id}")
        sources[source.source_id] = source
    return sources


SOURCES = load_sources()

# The stage-plan streams the readers below emit (identity streams plus concordant task families).
STREAMS = frozenset(
    (
        "audio_identity",
        "image_identity",
        "video_identity",
        "transcription",
        "verbalization",
        "speech_translation",
        "image_description",
        "image_creation",
        "image_edit",
        "speech_edit",
        "av_creation",
        "av_edit",
    )
)


class Preparer(Protocol):
    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord: ...


class CachingPreparer:
    """Prepare each file once even when several bundles (or a voice reference) name it."""

    def __init__(self, inner: Preparer):
        self.inner = inner
        self._records: dict[tuple[str, str, bool], MediaRecord] = {}

    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord:
        key = (modality, str(path), target)
        if key not in self._records:
            self._records[key] = self.inner.media(modality, path, target=target)
        return self._records[key]


def frozen_split(family: str) -> str:
    bucket = int(hashlib.sha256(family.encode()).hexdigest()[:8], 16) % 100
    return "validation" if bucket < VALIDATION_PERCENT else "train"


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:24]


def _provenance(source: OpenSource, kind: str) -> str:
    """``natural`` unless the source declares its ``text`` labels or ``media`` machine-made (``teacher``)."""
    return source.options.get("provenance", {}).get(kind, "natural")


# --- episodes -----------------------------------------------------------------------------


def identity_episode(record: MediaRecord, family: str | None = None) -> Episode:
    """Autoencode one prepared media event: a short instruction, then the media itself.

    A ``family`` (the original source family of the recording) freezes the split with that
    family, so the clip's identity and task episodes never straddle train and validation.
    """
    if record.target is None:
        raise ValueError("Identity episodes need a realization target")
    modality = record.modality
    records = (
        PhysicalRecord("prompt", "user", f"Reproduce this {modality}.".encode()),
        PhysicalRecord(
            "reproduction",
            "assistant",
            record.to_bytes(),
            turn_index=1,
            policy_mask=True,
            media_type=MEDIA_RECORD_TYPE,
            provenance="natural",
        ),
    )
    digest = record.source.source_family
    # One ID per (media, family): the same media reached through two sources is two
    # episodes with identical content, which deduplication merges and splits as one.
    identity = hashlib.sha256(f"{digest}\0{family or digest}".encode()).hexdigest()
    return Episode(
        f"{modality}-identity-{identity[:24]}",
        records,
        Lineage(
            "media-identity",
            "v1",
            family or digest,
            frozen_split=frozen_split(family) if family else None,
        ),
        stream=f"{modality}_identity",
        tracks=(TrackSpec("prompt", "user"), TrackSpec("reproduction", "assistant")),
        meridians=(
            Meridian(0, (TrackEvent("prompt", "prompt", "text", 0, "user"),)),
            Meridian(
                1,
                (
                    TrackEvent(
                        "reproduction",
                        "reproduction",
                        modality,
                        1,
                        "assistant",
                        # Event identity is the observed source, never the R target.
                        asset_sha256=record.source.sha256,
                    ),
                ),
            ),
        ),
    )


def _text(source: OpenSource, artifact_id: str, label: str, value: str):
    return BundleArtifact(
        artifact_id,
        label,
        "text",
        payload=value.encode("utf-8"),
        provenance=_provenance(source, "text"),
    )


def _media(
    media: Preparer,
    artifact_id: str,
    label: str,
    modality: str,
    path: str | Path,
    *,
    parent_id: str | None = None,
    version: str = "v0",
    provenance: str = "natural",
) -> BundleArtifact:
    return BundleArtifact(
        artifact_id,
        label,
        modality,
        media=media.media(modality, path),
        parent_id=parent_id,
        version=version,
        provenance=provenance,
    )


def compile_bundle(
    source: OpenSource,
    bundle_id: str,
    family: str,
    artifacts: Sequence[BundleArtifact],
    *,
    preprocessing: str,
    tier: str | None = None,
) -> list[Episode]:
    """Every standard task view the bundle's labels support."""
    bundle = AlignedBundle(
        bundle_id,
        Lineage(
            source.source_id,
            source.revision,
            family,
            frozen_split=frozen_split(family),
            source_uri=f"hf://datasets/{source.repository}",
        ),
        tuple(artifacts),
        source.license,
        preprocessing,
        tier or source.options.get("tier", "publisher-annotated"),
    )
    return [compile_task(bundle, view) for view in standard_views(bundle)]


def speech_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """Speech with its verbatim transcript -> transcription episode and audio identity episode."""
    speech = _media(media, _sha(clip["id"], "speech"), "source_speech", "audio", clip["audio"])
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        (speech, _text(source, _sha(clip["id"], "text"), "verbatim_transcript", clip["text"])),
        preprocessing=f"{source.source_id}-speech-v1",
    )
    return [*episodes, identity_episode(speech.media, clip["family"])]


def verbalization_episodes(
    media: Preparer,
    source: OpenSource,
    clip: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> list[Episode]:
    """Written text -> speech (verbalization), with an earlier same-speaker voice sample."""
    speech = _media(media, _sha(clip["id"], "speech"), "speech", "audio", clip["audio"])
    artifacts = [
        _text(source, _sha(clip["id"], "written"), "written_text", clip["written"]),
        speech,
    ]
    if clip["spoken"] != clip["written"]:
        artifacts.append(_text(source, _sha(clip["id"], "spoken"), "spoken_form", clip["spoken"]))
    if reference is not None:
        artifacts.append(
            _media(
                media,
                _sha(reference["id"], "voice"),
                "voice_reference",
                "audio",
                reference["audio"],
            )
        )
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        artifacts,
        preprocessing=f"{source.source_id}-verbalization-v1",
    )
    return [*episodes, identity_episode(speech.media, clip["family"])]


def translation_pair_episodes(
    media: Preparer, source: OpenSource, pair: Mapping[str, Any]
) -> list[Episode]:
    """A sentence read in a source language and in English -> direct speech translation."""
    key = f"{pair['language']}:{pair['id']}"
    return compile_bundle(
        source,
        f"{source.source_id}:{key}",
        pair["family"],
        (
            _media(media, _sha(key, "src"), "source_speech", "audio", pair["source_audio"]),
            _text(source, _sha(key, "srct"), "verbatim_transcript", pair["source_text"]),
            _text(source, _sha(key, "tgt"), "translated_text", pair["target_text"]),
            _media(media, _sha(key, "tgts"), "translated_speech", "audio", pair["target_audio"]),
        ),
        preprocessing=f"{source.source_id}-pair-v1",
    )


def translation_text_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any]
) -> list[Episode]:
    """Speech with its transcript and a text translation -> translation, transcription, identity."""
    speech = _media(media, _sha(clip["id"], "speech"), "source_speech", "audio", clip["audio"])
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        (
            speech,
            _text(source, _sha(clip["id"], "srct"), "verbatim_transcript", clip["text"]),
            _text(source, _sha(clip["id"], "tgt"), "translated_text", clip["translation"]),
        ),
        preprocessing=f"{source.source_id}-translation-v1",
    )
    return [*episodes, identity_episode(speech.media, clip["family"])]


def image_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """A captioned photo -> image description, image creation and image identity episodes.

    The creation request is the caption's first sentence.
    """
    image = _media(
        media,
        _sha(clip["id"], "image"),
        "image",
        "image",
        clip["image"],
        provenance=_provenance(source, "media"),
    )
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        (
            image,
            _text(source, _sha(clip["id"], "caption"), "display_caption", clip["description"]),
            _text(
                source,
                _sha(clip["id"], "request"),
                "creation_request",
                first_sentence(clip["description"]),
            ),
        ),
        preprocessing=f"{source.source_id}-photo-v1",
    )
    return [*episodes, identity_episode(image.media, clip["family"])]


def edit_episodes(
    media: Preparer,
    source: OpenSource,
    kind: str,
    clip: Mapping[str, Any],
    *,
    provenance: str,
    tier: str | None = None,
) -> list[Episode]:
    """An original recording, a plain-language instruction and the verified edited version.

    ``kind`` is ``image``, ``speech`` or ``av`` (the stream is ``<kind>_edit``).
    """
    modality = {"image": "image", "speech": "audio", "av": "video"}[kind]
    original = _media(
        media, _sha(clip["id"], "original"), f"original_{kind}", modality, clip["original"]
    )
    edited = _media(
        media,
        _sha(clip["id"], "edited"),
        f"edited_{kind}",
        modality,
        clip["edited"],
        parent_id=original.artifact_id,
        version="v1",
        provenance=provenance,
    )
    return compile_bundle(
        source,
        f"{source.source_id}:{kind}-edit:{clip['id']}",
        clip["family"],
        (
            original,
            edited,
            _text(
                source, _sha(clip["id"], "edit"), f"{kind}_edit_instruction", clip["instruction"]
            ),
        ),
        preprocessing=f"{source.source_id}-{kind}-edit-v1",
        tier=tier,
    )


def sound_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """A recording of any sound -> audio identity (no open set pairs sounds with a task family)."""
    return [
        identity_episode(
            _media(media, _sha(clip["id"], "sound"), "audio", "audio", clip["audio"]).media,
            clip["family"],
        )
    ]


def picture_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """An uncaptioned image -> image identity."""
    image = _media(media, _sha(clip["id"], "image"), "image", "image", clip["image"])
    return [identity_episode(image.media, clip["family"])]


def video_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """A silent clip -> video identity."""
    video = _media(media, _sha(clip["id"], "video"), "video", "video", clip["video"])
    return [identity_episode(video.media, clip["family"])]


def av_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """A captioned clip with sound -> audiovisual creation and video identity episodes."""
    video = _media(
        media,
        _sha(clip["id"], "video"),
        "video",
        "video",
        clip["video"],
        provenance=_provenance(source, "media"),
    )
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        (
            video,
            _media(media, _sha(clip["id"], "audio"), "soundtrack", "audio", clip["audio"]),
            _text(source, _sha(clip["id"], "caption"), "display_caption", clip["caption"]),
            _text(source, _sha(clip["id"], "request"), "creation_request", clip["request"]),
        ),
        preprocessing=f"{source.source_id}-av-v1",
    )
    return [*episodes, identity_episode(video.media, clip["family"])]


# --- files: local directory or the pinned Hub revision --------------------------------------


class Shards:
    """The files one source's reader may open.

    With ``<inputs>/<source_id>`` present these are the files in it, named as in the repository;
    otherwise they are the repository's files at the pinned revision, streamed over HTTP range
    requests (the Hub's own credentials come from ``HF_TOKEN``).
    """

    def __init__(
        self, source: OpenSource, inputs: Path | None = None, shard_slice: tuple[int, int] = (0, 1)
    ):
        self.source = source
        self.shard_slice = shard_slice
        local = inputs / source.source_id if inputs else None
        self.local = local if local is not None and local.is_dir() else None
        self._fs: Any = None

    def _hub(self) -> Any:
        if self._fs is None:
            from huggingface_hub import HfFileSystem

            self._fs = HfFileSystem()
        return self._fs

    def _remote(self, name: str) -> str:
        return f"datasets/{self.source.repository}@{self.source.revision}/{name}"

    def names(self, patterns: Sequence[str] | None = None) -> list[str]:
        """Repository paths matching the source's ``include`` patterns, in a fixed order.

        ``shard_slice`` (index, count) keeps every count-th path from index, so several jobs can
        each build a disjoint part of one source.
        """
        found: list[str] = []
        for pattern in patterns or self.source.include:
            if self.local is not None:
                matches = glob.glob(pattern, root_dir=self.local, recursive=True)
            else:
                prefix = self._remote("")
                matches = [
                    path.removeprefix(prefix)
                    for path in self._hub().glob(self._remote(pattern), detail=False)
                ]
            found.extend(sorted(matches))
        index, count = self.shard_slice
        return found[index::count]

    @contextmanager
    def open(self, name: str) -> Iterator[BinaryIO]:
        if self.local is not None:
            with (self.local / name).open("rb") as handle:
                yield handle
        else:
            with self._hub().open(self._remote(name), "rb", block_size=32 << 20) as handle:
                yield handle


# --- records: rows, samples and archive members ---------------------------------------------


def locate(record: Any, path: str) -> Any:
    """Follow a dotted path through dicts and lists (list steps are integers); None if absent."""
    for part in path.split("."):
        if isinstance(record, Mapping):
            record = record.get(part)
        elif isinstance(record, (list, tuple)) and part.lstrip("-").isdigit():
            index = int(part)
            record = record[index] if -len(record) <= index < len(record) else None
        else:
            return None
        if record is None:
            return None
    return record


def field_of(record: Mapping[str, Any], spec: str | Sequence[str] | None) -> Any:
    """A labelled field: one dotted path, or several joined with "-" (a composite id)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        return locate(record, spec)
    parts = [locate(record, part) for part in spec]
    return None if any(part is None for part in parts) else "-".join(str(part) for part in parts)


def blob(value: Any) -> bytes | None:
    """The bytes of a media field (raw bytes, or the Hub's ``{"bytes": ..., "path": ...}``)."""
    if isinstance(value, Mapping):
        value = value.get("bytes")
    return bytes(value) if isinstance(value, (bytes, bytearray)) and value else None


def _top_level(source: OpenSource) -> list[str]:
    """The parquet columns a source's labelled fields, filters and uniqueness key read."""
    specs = [
        *source.columns.values(),
        *source.options.get("keep", {}),
        *source.options.get("absent", ()),
        *([source.options["unique"]] if "unique" in source.options else []),
    ]
    names = {
        part.split(".")[0] for spec in specs for part in ([spec] if isinstance(spec, str) else spec)
    }
    return sorted(names - {"__key__"})


def parquet_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as parquet

    wanted = _top_level(source)
    for name in shards.names():
        with shards.open(name) as handle:
            reader = parquet.ParquetFile(handle)
            missing = set(wanted) - set(reader.schema_arrow.names)
            if missing:
                raise ValueError(f"{source.source_id}: {name} lacks columns {sorted(missing)}")
            row = 0
            for batch in reader.iter_batches(batch_size=16, columns=wanted):
                for record in batch.to_pylist():
                    record["__key__"] = f"{PurePosixPath(name).stem}:{row}"
                    row += 1
                    yield record


_TEXT_EXTENSIONS = frozenset(("txt", "cls", "caption"))


def _sample(group: Mapping[str, bytes]) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    for extension, payload in group.items():
        if extension == "json":
            sample[extension] = json.loads(payload)
        elif extension in _TEXT_EXTENSIONS:
            sample[extension] = payload.decode("utf-8")
        else:
            sample[extension] = payload
    return sample


def webdataset_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """Samples of consecutive tar members sharing a key; fields are named by file extension."""
    for name in shards.names():
        with shards.open(name) as handle, tarfile.open(fileobj=handle, mode="r|*") as tar:
            key: str | None = None
            group: dict[str, bytes] = {}
            for member in tar:
                if not member.isfile():
                    continue
                stem, _, extension = PurePosixPath(member.name).name.partition(".")
                if stem != key and group:
                    yield {"__key__": key, **_sample(group)}
                    group = {}
                key = stem
                group[extension] = tar.extractfile(member).read()
            if group:
                yield {"__key__": key, **_sample(group)}


def _table(shards: Shards, spec: Mapping[str, Any], wanted: Iterable[str]) -> dict[str, dict]:
    """Rows of a csv/jsonl file keyed by the ``key`` template (a ``str.format`` over the row)."""
    rows: dict[str, dict] = {}
    with shards.open(spec["path"]) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
        if spec["path"].endswith(".jsonl"):
            lines: Iterable[dict] = (json.loads(line) for line in text if line.strip())
        else:
            names = spec.get("names")
            lines = csv.DictReader(text, fieldnames=names)
        for row in lines:
            for name in spec.get("ints", ()):
                row[name] = int(row[name])
            rows[spec["key"].format(**row)] = {name: row.get(name) for name in wanted}
    return rows


def archive_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """Members of zip/tar archives joined with their table row (and, for zips, a partner member).

    ``options.table`` names the csv/jsonl file and the ``key`` template that matches a row to a
    member's file name; the member's bytes are the record's ``member``. ``options.partner`` (a
    template over the archive's file name) names a second zip whose same-named member is the
    record's ``partner``, as in a source/edited pair of videos.
    """
    spec = source.options["table"]
    rows = _table(shards, spec, spec.get("columns", ()))
    partner_template = source.options.get("partner")
    for name in shards.names():
        with ExitStack() as stack:
            handle = stack.enter_context(shards.open(name))
            members = _zip_members(handle) if name.endswith(".zip") else _tar_members(handle)
            partner = None
            if partner_template:
                partner_name = partner_template.format(name=PurePosixPath(name).name)
                partner = zipfile.ZipFile(stack.enter_context(shards.open(partner_name)))
            for member_name, read in members:
                row = rows.get(PurePosixPath(member_name).name)
                if row is None:
                    continue
                record = {"__key__": PurePosixPath(member_name).name, "member": read(), **row}
                if partner is not None:
                    if member_name not in partner.NameToInfo:
                        continue
                    record["partner"] = partner.read(member_name)
                yield record


def _zip_members(handle: BinaryIO) -> Iterator[tuple[str, Callable[[], bytes]]]:
    archive = zipfile.ZipFile(handle)
    for info in archive.infolist():  # archive order: sequential reads
        if not info.is_dir():
            yield info.filename, lambda info=info: archive.read(info)


def _tar_members(handle: BinaryIO) -> Iterator[tuple[str, Callable[[], bytes]]]:
    with tarfile.open(fileobj=handle, mode="r|*") as tar:
        for member in tar:
            if member.isfile():
                yield member.name, lambda member=member: tar.extractfile(member).read()


# --- media files ----------------------------------------------------------------------------


def _stem(value: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', value)[:80]}-{_sha(value)[:8]}"


_AUDIO_SUFFIX = {"FLAC": "flac", "WAV": "wav", "OGG": "ogg", "MP3": "mp3"}


def write_audio(
    payload: bytes,
    out: Path,
    stem: str,
    *,
    min_seconds: float = MIN_SPEECH_SECONDS,
    max_seconds: float = MAX_SPEECH_SECONDS,
    crop: bool = False,
) -> Path | None:
    """Keep a decodable recording of an accepted length (``crop``: cut a longer one to a central window)."""
    import soundfile

    try:
        info = soundfile.info(io.BytesIO(payload))
    except RuntimeError:
        return None
    seconds = info.frames / info.samplerate
    if seconds < min_seconds or (seconds > max_seconds and not crop):
        return None
    out.mkdir(parents=True, exist_ok=True)
    if seconds <= max_seconds:
        path = out / f"{stem}.{_AUDIO_SUFFIX.get(info.format, 'wav')}"
        path.write_bytes(payload)
        return path
    window = int(max_seconds * info.samplerate)
    samples, rate = soundfile.read(
        io.BytesIO(payload), start=(info.frames - window) // 2, frames=window, dtype="float32"
    )
    path = out / f"{stem}.wav"
    soundfile.write(path, samples, rate)
    return path


def write_image(
    payload: bytes, out: Path, stem: str, *, max_side: int = MAX_SIDE, min_side: int = 32
) -> Path | None:
    """A decodable RGB photo scaled to at most ``max_side`` pixels (JPEG)."""
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if min(image.size) < min_side:
        return None
    image.thumbnail((max_side, max_side))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.jpg"
    image.save(path, quality=95)
    return path


def write_image_pair(
    original: bytes, edited: bytes, out: Path, stem: str, *, max_side: int = MAX_SIDE
) -> tuple[Path, Path] | None:
    """An image and its edit scaled identically (lossless, so unedited pixels stay equal)."""
    from PIL import Image

    try:
        images = [Image.open(io.BytesIO(data)).convert("RGB") for data in (original, edited)]
    except (OSError, ValueError, Image.DecompressionBombError):
        return None
    if images[0].size != images[1].size or min(images[0].size) < 32:
        return None
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, image in zip(("original", "edited"), images, strict=True):
        image.thumbnail((max_side, max_side))
        paths.append(out / f"{stem}-{name}.png")
        image.save(paths[-1])
    return paths[0], paths[1]


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def probe(path: Path) -> tuple[float, set[str]]:
    """Duration in seconds and the stream kinds (``video``, ``audio``) of a media file."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(result.stdout)
    return float(info["format"].get("duration") or 0.0), {s["codec_type"] for s in info["streams"]}


def transcode(
    source: Path,
    video: Path,
    audio: Path | None = None,
    *,
    start: float = 0.0,
    seconds: float | None = None,
) -> None:
    """A clip as small video (scaled into 640x640 at 16 frames/s, no sound) and, optionally, its
    soundtrack at 24 kHz mono (the speech realizer's rate)."""
    cut = ["-ss", str(start), *(["-t", str(seconds)] if seconds else [])]
    _ffmpeg(
        *cut,
        "-i",
        str(source),
        "-an",
        "-vf",
        "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-r",
        "16",
        str(video),
    )
    if audio is not None:
        _ffmpeg(*cut, "-i", str(source), "-vn", "-ac", "1", "-ar", "24000", str(audio))


def first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()), maxsplit=1)[0]


def _timestamp(text: str) -> float:
    """Seconds in ``HH:MM:SS.mmm`` (or ``MM:SS``, or plain seconds)."""
    seconds = 0.0
    for part in str(text).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


# --- tasks: record -> clip -> episodes --------------------------------------------------------

_GIGASPEECH_PUNCTUATION = {
    "<COMMA>": ",",
    "<PERIOD>": ".",
    "<QUESTIONMARK>": "?",
    "<EXCLAMATIONPOINT>": "!",
}


def _gigaspeech(text: str) -> str | None:
    """Spoken-form transcript with punctuation restored; None if it marks silence, music or noise."""
    for tag, mark in _GIGASPEECH_PUNCTUATION.items():
        text = text.replace(f" {tag}", mark).replace(tag, mark)
    return None if re.search(r"<[A-Z]+>", text) else text.strip()


TEXT_CLEANERS: dict[str, Callable[[str], str | None]] = {"gigaspeech": _gigaspeech}


def _label(source: OpenSource, record: Mapping[str, Any], role: str) -> Any:
    return field_of(record, source.columns.get(role))


def _text_label(source: OpenSource, record: Mapping[str, Any], role: str) -> str | None:
    """A non-empty text label, shaped by the source's options (picking one of a list, cleaning)."""
    value = _label(source, record, role)
    if isinstance(value, (list, tuple)):
        texts = [item for item in value if isinstance(item, str) and item.strip()]
        value = (
            max(texts, key=len)
            if source.options.get("pick", "longest") == "longest"
            else (texts[0] if texts else None)
        )
    if not isinstance(value, str) or not value.strip():
        return None
    cleaner = source.options.get("clean")
    value = TEXT_CLEANERS[cleaner](value) if cleaner else value
    value = value.strip() if value else value
    if not value or not source.options.get("min_chars", 1) <= len(value) <= source.options.get(
        "max_chars", 1 << 30
    ):
        return None
    return value


def _empty(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_empty(item) for item in value.values())
    return value in (None, "", [], ())


def _accepts(source: OpenSource, record: Mapping[str, Any]) -> bool:
    """``options.keep`` (field -> allowed values) and ``options.absent`` (fields that must be empty)."""
    for spec, allowed in source.options.get("keep", {}).items():
        if field_of(record, spec) not in allowed:
            return False
    return all(_empty(field_of(record, spec)) for spec in source.options.get("absent", ()))


def _identity(source: OpenSource, record: Mapping[str, Any]) -> tuple[str, str]:
    """The clip id and the source family its split is frozen with.

    The family is the ``family`` field, else the first group of ``options.family_pattern``
    matched against the id (for ids that embed their recording), else the id itself.
    """
    clip_id = str(_label(source, record, "id") or record["__key__"])
    family = _label(source, record, "family")
    if family is None and "family_pattern" in source.options:
        match = re.match(source.options["family_pattern"], clip_id)
        family = match.group(1) if match else None
    return clip_id, f"{source.source_id}:{clip_id if family is None else family}"


def _speech_file(
    source: OpenSource, record: Mapping[str, Any], out: Path, stem: str
) -> Path | None:
    payload = blob(_label(source, record, "audio"))
    return payload and write_audio(
        payload,
        out,
        stem,
        min_seconds=source.options.get("min_seconds", MIN_SPEECH_SECONDS),
        max_seconds=source.options.get("max_seconds", MAX_SPEECH_SECONDS),
        crop=source.options.get("crop", False),
    )


def speech_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    text = _text_label(source, record, "text")
    if text is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    audio = _speech_file(source, record, out, _stem(clip_id))
    return [{"id": clip_id, "family": family, "audio": audio, "text": text}] if audio else []


def verbalization_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    written = _text_label(source, record, "written")
    if written is None or not _accepts(source, record):
        return []
    spoken = _text_label(source, record, "spoken") or written
    clip_id, family = _identity(source, record)
    audio = _speech_file(source, record, out, _stem(clip_id))
    clip = {"id": clip_id, "family": family, "audio": audio, "written": written, "spoken": spoken}
    return [clip] if audio else []


def translation_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    text, translation = (
        _text_label(source, record, "text"),
        _text_label(source, record, "translation"),
    )
    if text is None or translation is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    audio = _speech_file(source, record, out, _stem(clip_id))
    clip = {
        "id": clip_id,
        "family": family,
        "audio": audio,
        "text": text,
        "translation": translation,
    }
    return [clip] if audio else []


def sound_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    if not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    audio = _speech_file(source, record, out, _stem(clip_id))
    return [{"id": clip_id, "family": family, "audio": audio}] if audio else []


def image_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    description = _text_label(source, record, "caption")
    payload = blob(_label(source, record, "image"))
    if description is None or payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    image = write_image(
        payload, out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    return (
        [{"id": clip_id, "family": family, "image": image, "description": description}]
        if image
        else []
    )


def picture_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    payload = blob(_label(source, record, "image"))
    if payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    image = write_image(
        payload, out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    return [{"id": clip_id, "family": family, "image": image}] if image else []


def image_edit_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    instruction = _text_label(source, record, "instruction")
    original = blob(_label(source, record, "original"))
    edited = blob(_label(source, record, "edited"))
    if instruction is None or original is None or edited is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    paths = write_image_pair(
        original, edited, out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    if paths is None:
        return []
    return [
        {
            "id": clip_id,
            "family": family,
            "original": paths[0],
            "edited": paths[1],
            "instruction": instruction,
        }
    ]


def _video_source(payload: bytes, out: Path, stem: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.source.mp4"
    path.write_bytes(payload)
    return path


def av_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    """A captioned recording of one clip's length: video and soundtrack as small files.

    ``options.request`` is a ``str.format`` template over ``caption`` and ``request`` (a
    second labelled field, such as an activity label).
    """
    caption = _text_label(source, record, "caption")
    payload = blob(_label(source, record, "video"))
    if caption is None or payload is None or not _accepts(source, record):
        return []
    request = _label(source, record, "request")
    if "request" in source.columns and not request:
        return []
    clip_id, family = _identity(source, record)
    low, high = source.options.get("min_seconds", 1.5), source.options.get("max_seconds", 8.0)
    original = _video_source(payload, out, _stem(clip_id))
    try:
        seconds, kinds = probe(original)
        if (
            "audio" not in kinds
            or seconds < low
            or (seconds > high and not source.options.get("trim"))
        ):
            return []
        video, audio = out / f"{_stem(clip_id)}.mp4", out / f"{_stem(clip_id)}.wav"
        transcode(original, video, audio, seconds=min(seconds, high))
    finally:
        original.unlink()
    template = source.options.get("request", "Create a short clip of: {caption}.")
    return [
        {
            "id": clip_id,
            "family": family,
            "video": video,
            "audio": audio,
            "caption": caption,
            "request": template.format(caption=caption, request=request),
        }
    ]


def scene_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    """Clips cut at the described activities of a long annotated video (FineVideo's scenes).

    Each clip is one activity's window (at most ``max_seconds``) with its description as the
    caption and its scene's title as the creation request; ``clips_per_video`` windows are
    taken, evenly spaced.
    """
    payload = blob(_label(source, record, "video"))
    meta = _label(source, record, "meta")
    if payload is None or not isinstance(meta, Mapping) or not _accepts(source, record):
        return []
    low, high = source.options.get("min_seconds", 2.0), source.options.get("max_seconds", 8.0)
    windows = []
    for scene in locate(meta, "content_metadata.scenes") or ():
        title = (scene.get("title") or "").strip()
        for activity in scene.get("activities") or ():
            description = (activity.get("description") or "").strip()
            stamps = activity.get("timestamp") or {}
            if not (title and description and stamps):
                continue
            start, end = _timestamp(stamps["start_timestamp"]), _timestamp(stamps["end_timestamp"])
            if end - start >= low:
                windows.append((start, min(end - start, high), title, description))
    count = source.options.get("clips_per_video", 3)
    windows = windows[:: max(1, len(windows) // count)][:count]
    if not windows:
        return []
    video_id, family = _identity(source, record)
    original = _video_source(payload, out, _stem(video_id))
    clips = []
    try:
        if "audio" not in probe(original)[1]:
            return []
        for start, seconds, title, description in windows:
            clip_id = f"{video_id}:{start:.1f}"
            video, audio = out / f"{_stem(clip_id)}.mp4", out / f"{_stem(clip_id)}.wav"
            transcode(original, video, audio, start=start, seconds=seconds)
            clips.append(
                {
                    "id": clip_id,
                    "family": family,
                    "video": video,
                    "audio": audio,
                    "caption": description,
                    "request": f"Create a short clip of: {title}.",
                }
            )
    finally:
        original.unlink()
    return clips


def video_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    payload = blob(_label(source, record, "video"))
    if payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    high = source.options.get("max_seconds", 8.0)
    original = _video_source(payload, out, _stem(clip_id))
    try:
        seconds, kinds = probe(original)
        if "video" not in kinds or seconds < source.options.get("min_seconds", 1.5):
            return []
        video = out / f"{_stem(clip_id)}.mp4"
        transcode(original, video, seconds=min(seconds, high))
    finally:
        original.unlink()
    return [{"id": clip_id, "family": family, "video": video}]


def video_edit_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """A video and its instructed edit, trimmed and scaled identically."""
    instruction = _text_label(source, record, "instruction")
    original, edited = (
        blob(_label(source, record, "original")),
        blob(_label(source, record, "edited")),
    )
    if instruction is None or original is None or edited is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    high = source.options.get("max_seconds", 6.0)
    sources = {
        name: _video_source(data, out, f"{_stem(clip_id)}-{name}")
        for name, data in (("original", original), ("edited", edited))
    }
    try:
        durations = {name: probe(path)[0] for name, path in sources.items()}
        if min(durations.values()) < 1.0 or abs(durations["original"] - durations["edited"]) > 0.5:
            return []
        seconds = min(min(durations.values()), high)
        paths = {name: out / f"{_stem(clip_id)}-{name}.mp4" for name in sources}
        for name, path in sources.items():
            transcode(path, paths[name], seconds=seconds)
    finally:
        for path in sources.values():
            path.unlink()
    return [{"id": clip_id, "family": family, **paths, "instruction": instruction}]


SPEECH_EDITS = (
    ("Make the speech 6 decibels louder.", ("-af", "volume=6dB,alimiter=limit=0.97")),
    ("Speak 25 percent faster without changing the pitch.", ("-af", "atempo=1.25")),
    ("Play the recording backwards.", ("-af", "areverse")),
)
VIDEO_EDITS = (
    ("Mirror the video horizontally.", ("-vf", "hflip")),
    ("Convert the video to black and white.", ("-vf", "hue=s=0")),
    ("Play the video at double speed.", ("-vf", "setpts=0.5*PTS")),
    ("Play the video in reverse.", ("-vf", "reverse")),
)


def _procedural(
    options: Sequence[tuple[str, tuple[str, ...]]],
    clip: Mapping[str, Any],
    original: Path,
    out: Path,
) -> dict[str, Any]:
    """Apply the one edit this clip id selects (deterministically) and return its clip."""
    out.mkdir(parents=True, exist_ok=True)
    instruction, arguments = options[int(_sha(clip["id"]), 16) % len(options)]
    edited = out / f"{_stem(clip['id'])}.edited{original.suffix}"
    _ffmpeg("-i", str(original), *arguments, str(edited))
    return {
        "id": clip["id"],
        "family": clip["family"],
        "original": original,
        "edited": edited,
        "instruction": instruction,
    }


def speech_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    return _procedural(SPEECH_EDITS, clip, Path(clip["audio"]), out)


def video_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    return _procedural(VIDEO_EDITS, clip, Path(clip["video"]), out)


@dataclass(frozen=True, slots=True)
class Task:
    """What a source's labels mean: how a record becomes a clip, and a clip becomes episodes."""

    streams: frozenset[str]
    episodes: Callable[
        [Preparer, OpenSource, Mapping[str, Any], Mapping[str, Any] | None], list[Episode]
    ]
    materialize: Callable[[OpenSource, Mapping[str, Any], Path], list[dict[str, Any]]] | None = None
    # Which procedural edit applies to this task's clips: "speech" or "video".
    procedural: str | None = None
    # Whether the previous clip of the same family is passed as a voice reference.
    reference: bool = False

    @property
    def edit_stream(self) -> str | None:
        return {"speech": "speech_edit", "video": "av_edit"}.get(self.procedural or "")


def _plain(episodes: Callable[[Preparer, OpenSource, Mapping[str, Any]], list[Episode]]):
    return lambda media, source, clip, reference: episodes(media, source, clip)


TASKS = {
    "transcription": Task(
        frozenset({"transcription", "audio_identity"}),
        _plain(speech_episodes),
        speech_clips,
        procedural="speech",
    ),
    "verbalization": Task(
        frozenset({"verbalization", "audio_identity"}),
        verbalization_episodes,
        verbalization_clips,
        procedural="speech",
        reference=True,
    ),
    "translation_pair": Task(
        frozenset({"speech_translation", "transcription"}), _plain(translation_pair_episodes)
    ),
    "translation_text": Task(
        frozenset({"speech_translation", "transcription", "audio_identity"}),
        _plain(translation_text_episodes),
        translation_clips,
        procedural="speech",
    ),
    "captioned_image": Task(
        frozenset({"image_description", "image_creation", "image_identity"}),
        _plain(image_episodes),
        image_clips,
    ),
    "picture": Task(frozenset({"image_identity"}), _plain(picture_episodes), picture_clips),
    "image_edit": Task(
        frozenset({"image_edit"}),
        lambda media, source, clip, reference: edit_episodes(
            media, source, "image", clip, provenance=_provenance(source, "media")
        ),
        image_edit_clips,
    ),
    "sound": Task(frozenset({"audio_identity"}), _plain(sound_episodes), sound_clips),
    "av_caption": Task(
        frozenset({"av_creation", "video_identity"}),
        _plain(av_episodes),
        av_clips,
        procedural="video",
    ),
    "av_scenes": Task(
        frozenset({"av_creation", "video_identity"}),
        _plain(av_episodes),
        scene_clips,
        procedural="video",
    ),
    "video": Task(
        frozenset({"video_identity"}), _plain(video_episodes), video_clips, procedural="video"
    ),
    "video_edit": Task(
        frozenset({"av_edit"}),
        lambda media, source, clip, reference: edit_episodes(
            media, source, "av", clip, provenance=_provenance(source, "media")
        ),
        video_edit_clips,
    ),
}


# --- readers that produce clips directly (their files are not one record per row) ---------------


def _extract(shards: Shards, name: str, wanted: Mapping[str, str], out: Path) -> dict[str, Path]:
    """Stream one tar.gz once, writing just the named members (``wanted``: member -> output name)."""
    out.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    with shards.open(name) as handle, tarfile.open(fileobj=handle, mode="r|gz") as tar:
        for member in tar:
            target = wanted.get(member.name)
            if target is None or not member.isfile():
                continue
            path = out / target
            path.write_bytes(tar.extractfile(member).read())
            found[member.name] = path
            if len(found) == len(wanted):
                break
    return found


def read_fleurs(
    source: OpenSource, shards: Shards, out: Path, limit: int
) -> Iterator[dict[str, Any]]:
    """Source-language -> English speech pairs of the same FLoRes sentence, ``limit`` in all.

    ``options.languages`` are FLEURS configs; each contributes about ``limit / len(languages)``
    sentences (by a fixed hash order), all read in English as ``options.target``.
    """
    languages, target = source.options["languages"], source.options.get("target", "en_us")
    split = source.options.get("split", "train")
    per_language = -(-limit // len(languages))
    max_samples = int(MAX_SPEECH_SECONDS * 16000)

    def rows(language: str) -> dict[str, list[str]]:
        # Columns: id, file name, raw transcription, normalized transcription, characters,
        # samples, gender.
        first: dict[str, list[str]] = {}
        with shards.open(f"data/{language}/{split}.tsv") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
            for row in csv.reader(text, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row and int(row[5]) <= max_samples:
                    first.setdefault(row[0], row)
        return first

    def archive(language: str) -> str:
        return f"data/{language}/audio/{split}.tar.gz"

    english = rows(target)
    chosen: dict[str, list[tuple[list[str], list[str]]]] = {}
    for language in languages:
        theirs = rows(language)
        shared = sorted(set(theirs) & set(english), key=_sha)
        chosen[language] = [(theirs[key], english[key]) for key in shared[:per_language]]
    wanted_english = {
        f"{split}/{en[1]}": f"{target}-{en[1]}" for pairs in chosen.values() for _, en in pairs
    }
    english_files = _extract(shards, archive(target), wanted_english, out)
    for language, pairs in chosen.items():
        wanted = {f"{split}/{row[1]}": f"{language}-{row[1]}" for row, _ in pairs}
        files = _extract(shards, archive(language), wanted, out)
        for row, en in pairs:
            yield {
                "id": row[0],
                "language": language,
                "family": f"{source.source_id}:{row[0]}",
                "source_audio": files[f"{split}/{row[1]}"],
                "source_text": row[2],
                "target_audio": english_files[f"{split}/{en[1]}"],
                "target_text": en[2],
            }


def read_docci(
    source: OpenSource, shards: Shards, out: Path, limit: int
) -> Iterator[dict[str, Any]]:
    """Training-split DOCCI photos, scaled to at most ``max_side`` pixels, with their descriptions.

    Neither file is on the Hub (its repository holds only a loading script): both are
    job inputs under ``<inputs>/docci``.
    """
    if shards.local is None:
        raise FileNotFoundError(
            f"{source.source_id}: its files are not on the Hub; provide them under <inputs>/{source.source_id}"
        )
    out.mkdir(parents=True, exist_ok=True)
    wanted = {}
    with shards.open("docci_descriptions.jsonlines") as handle:
        # JSONL lines end at "\n" only; str.splitlines would also cut at U+2028 or U+0085
        # inside a description.
        for line in handle.read().decode("utf-8").split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] == "train":
                wanted[f"images/{row['image_file']}"] = row
    kept = 0
    with (
        shards.open("docci_images.tar.gz") as handle,
        tarfile.open(fileobj=handle, mode="r|gz") as tar,
    ):
        for member in tar:
            row = wanted.get(member.name.removeprefix("./"))
            if row is None or not member.isfile():
                continue
            image = write_image(
                tar.extractfile(member).read(),
                out,
                _stem(row["example_id"]),
                max_side=source.options.get("max_side", MAX_SIDE),
            )
            if image is None:
                continue
            yield {
                "id": row["example_id"],
                "family": f"{source.source_id}:{row['example_id']}",
                "image": image,
                "description": row["description"],
            }
            kept += 1
            if kept >= limit:
                return


CONTAINERS: dict[str, Callable[[OpenSource, Shards], Iterator[dict[str, Any]]]] = {
    "parquet": parquet_records,
    "webdataset": webdataset_records,
    "archive": archive_records,
}
CLIP_READERS: dict[str, Callable[[OpenSource, Shards, Path, int], Iterator[dict[str, Any]]]] = {
    "fleurs": read_fleurs,
    "docci": read_docci,
}


def read_clips(
    source: OpenSource, shards: Shards, out: Path, limit: int
) -> Iterator[dict[str, Any]]:
    """Up to ``limit`` clips of one source, written under ``out``."""
    if limit < 1:
        return
    if source.reader in CLIP_READERS:
        clips = CLIP_READERS[source.reader](source, shards, out, limit)
    else:
        clips = _materialized(source, shards, out)
    for kept, clip in enumerate(clips, 1):
        yield clip
        if kept >= limit:
            return


def _materialized(source: OpenSource, shards: Shards, out: Path) -> Iterator[dict[str, Any]]:
    task, unique, seen = TASKS[source.task], source.options.get("unique"), set()
    for record in CONTAINERS[source.reader](source, shards):
        if unique:
            key = field_of(record, unique)
            if key in seen:
                continue
            seen.add(key)
        yield from task.materialize(source, record, out)


def discard(clip: Mapping[str, Any]) -> None:
    """Delete a clip's local media (it is prepared once its episodes exist)."""
    for value in clip.values():
        if isinstance(value, Path):
            value.unlink(missing_ok=True)


def build_episodes(
    media: Preparer,
    source: OpenSource,
    clips: Iterable[Mapping[str, Any]],
    *,
    work: Path,
    procedural: int = 0,
) -> Iterator[Episode]:
    """The episodes of each clip, then one procedural edit for each of the first ``procedural``."""
    task = TASKS[source.task]
    references: dict[str, Mapping[str, Any]] = {}
    for index, clip in enumerate(clips):
        reference = references.get(clip["family"]) if task.reference else None
        yield from task.episodes(media, source, clip, reference)
        if index < procedural and task.procedural:
            make = speech_edit if task.procedural == "speech" else video_edit
            edit = make(clip, work / "edits")
            yield from edit_episodes(
                media,
                source,
                "speech" if task.procedural == "speech" else "av",
                edit,
                provenance="executed",
                tier="procedural",
            )
            edit["edited"].unlink()
        if task.reference:
            # The next clip of this family hears this one as its voice sample.
            if clip["family"] in references:
                discard(references[clip["family"]])
            references[clip["family"]] = clip
        else:
            discard(clip)
    for clip in references.values():
        discard(clip)


def source_episodes(
    media: Preparer,
    source: OpenSource,
    *,
    count: int,
    work: Path,
    procedural: int = 0,
    inputs: Path | None = None,
    shard_slice: tuple[int, int] = (0, 1),
) -> Iterator[Episode]:
    """Every episode of ``count`` clips of one approved source."""
    source.terms.require_approval()
    shards = Shards(source, inputs, shard_slice)
    out = work / source.source_id
    yield from build_episodes(
        media, source, read_clips(source, shards, out, count), work=out, procedural=procedural
    )
