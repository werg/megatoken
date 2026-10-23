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
import functools
import glob
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

from .concordance import (
    AlignedBundle,
    BundleArtifact,
    TaskView,
    compile_task,
    edit_chain_view,
    edit_judging_view,
    edit_plan_view,
    question_chain_view,
    standard_views,
)
from .fetch import FETCH_STATS, Fetcher, download, fetched_records
from .media_assets import MEDIA_RECORD_TYPE, MediaRecord
from .schema import Episode, Lineage, Meridian, PhysicalRecord, TrackEvent, TrackSpec
from .sources import TermsReview

VALIDATION_PERCENT = 5
MIN_SPEECH_SECONDS = 0.5
MAX_SPEECH_SECONDS = 20.0
MAX_SIDE = 1024
VIDEO_FPS = 16
# The manifests, in the order they were written: the first forty-four sources, the full-volume
# additions, the additions whose repositories need the Hub account's agreement first, and the
# sources that teach judging and planning edits (some are fetched from a publisher's web site).
MANIFEST_NAMES = (
    "sources.media.json",
    "sources.media-more.json",
    "sources.media-more-gated.json",
    "sources.media-agentic.json",
)
_PACKAGE_TREE = Path(__file__).resolve().parents[3] / "configs/data"
_CONFIGS = (
    _PACKAGE_TREE if (_PACKAGE_TREE / MANIFEST_NAMES[0]).is_file() else Path.cwd() / "configs/data"
)
MANIFESTS = tuple(path for path in (_CONFIGS / name for name in MANIFEST_NAMES) if path.is_file())
MANIFEST = _CONFIGS / MANIFEST_NAMES[0]


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
        extra = (
            {"verbalization"}
            if self.task == "transcription" and self.options.get("verbalize")
            else set()
        )
        if self.options.get("sustained_translation"):
            extra.add("streaming_translation")
        return tuple(
            sorted(
                task.streams
                | extra
                | {stream for stream in (task.edit_stream, task.plan_stream) if stream}
            )
        )

    def with_options(self, **overrides: Any) -> OpenSource:
        return dataclasses.replace(self, options={**self.options, **overrides})


def load_sources(*paths: Path) -> dict[str, OpenSource]:
    """The sources of one or more manifests (default: all of ``MANIFESTS``); an id may not repeat."""
    sources: dict[str, OpenSource] = {}
    for path in paths or MANIFESTS:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("Unknown media source manifest schema")
        _add_sources(sources, raw["sources"])
    return sources


def _add_sources(sources: dict[str, OpenSource], rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
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
        "streaming_translation",
        "image_description",
        "image_creation",
        "audio_description",
        "video_description",
        "image_edit",
        "speech_edit",
        "av_creation",
        "av_edit",
        "edit_judging",
        "edit_plan",
    )
)


class Preparer(Protocol):
    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord: ...


class CachingPreparer:
    """Prepare each file once even when several bundles (or a voice reference) name it."""

    def __init__(self, inner: Preparer, *, max_entries: int = 128):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Preparation caches need a positive entry bound")
        self.inner = inner
        self.max_entries = max_entries
        self._records = OrderedDict()
        self._intervals = OrderedDict()

    def _digest(self, path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.max_entries:
            cache.popitem(last=False)
        return value

    def audio_interval(self, path, *, start_sample, end_sample, channel, target=True):
        key = (self._digest(path), start_sample, end_sample, channel, target)
        if key not in self._intervals:
            return self._remember(self._intervals, key, self.inner.audio_interval(
                path,
                start_sample=start_sample,
                end_sample=end_sample,
                channel=channel,
                target=target,
            ))
        self._intervals.move_to_end(key)
        return self._intervals[key]

    def media(self, modality: str, path: str | Path, *, target: bool = True) -> MediaRecord:
        key = (modality, self._digest(path), target)
        if key not in self._records:
            return self._remember(self._records, key, self.inner.media(modality, path, target=target))
        self._records.move_to_end(key)
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
    # The ID names the exact content: the same media reached twice is either the same
    # episode (deduplication drops the repeat) or a different one with its own ID.
    identity = hashlib.sha256(records[1].payload + f"\0{family or digest}".encode()).hexdigest()
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
    views: Sequence[TaskView] | None = None,
) -> list[Episode]:
    """Every standard task view the bundle's labels support, or exactly the given ``views``."""
    # Publishers reuse clip IDs (one clip, several captions or labels), so the ID also names
    # the content: identical rows merge in deduplication and different ones stay distinct.
    content = hashlib.sha256(family.encode())
    for artifact in artifacts:
        content.update(artifact.label.encode() + b"\0" + artifact.payload)
        if artifact.media is not None:
            content.update(artifact.media.to_bytes())
    bundle = AlignedBundle(
        f"{bundle_id}:{content.hexdigest()[:16]}",
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
    return [compile_task(bundle, view) for view in views or standard_views(bundle)]


def speech_episodes(media: Preparer, source: OpenSource, clip: Mapping[str, Any]) -> list[Episode]:
    """Speech with its verbatim transcript -> transcription episode and audio identity episode.

    A source with ``options.verbalize`` (its transcripts are cased and punctuated like written
    text) also gives the reverse: the transcript read as written text -> that speech.
    """
    speech = _media(media, _sha(clip["id"], "speech"), "source_speech", "audio", clip["audio"])
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        (speech, _text(source, _sha(clip["id"], "text"), "verbatim_transcript", clip["text"])),
        preprocessing=f"{source.source_id}-speech-v1",
    )
    if source.options.get("verbalize"):
        said = _media(media, _sha(clip["id"], "speech"), "speech", "audio", clip["audio"])
        episodes += compile_bundle(
            source,
            f"{source.source_id}:{clip['id']}:said",
            clip["family"],
            (_text(source, _sha(clip["id"], "written"), "written_text", clip["text"]), said),
            preprocessing=f"{source.source_id}-verbalization-v1",
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
    episodes = compile_bundle(
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
    if source.options.get("sustained_translation"):
        from .streaming_translation import compile_open_translation

        episodes.append(
            compile_open_translation(
                media,
                source,
                pair,
                language={
                    "source": pair["language"],
                    "target": source.options.get("target", "en_us"),
                },
            )
        )
    return episodes


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
    if source.options.get("sustained_translation"):
        from .streaming_translation import compile_open_translation

        episodes.append(compile_open_translation(media, source, clip, language=clip["language"]))
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


def chain_episodes(
    media: Preparer,
    source: OpenSource,
    kind: str,
    clip: Mapping[str, Any],
    *,
    provenance: str,
    tier: str | None = None,
) -> list[Episode]:
    """A multi-turn editing session as one episode: the original, then per turn the user's
    request and the model's edited version, each turn editing the previous result.

    ``clip["frames"]`` is the original followed by each turn's result, ``clip["instructions"]``
    the request of each turn. ``kind`` is ``image`` or ``av`` (the stream is ``<kind>_edit``).
    """
    modality = {"image": "image", "av": "video"}[kind]
    frames, instructions = clip["frames"], clip["instructions"]
    if len(frames) != len(instructions) + 1 or not instructions:
        raise ValueError("An edit chain needs one request per edited version")
    key = clip["id"]
    artifacts = [_media(media, _sha(key, "frame", "0"), f"original_{kind}", modality, frames[0])]
    steps = []
    for turn, (frame, instruction) in enumerate(zip(frames[1:], instructions, strict=True), 1):
        ask = _text(
            source, _sha(key, "ask", str(turn)), f"{kind}_edit_instruction_{turn}", instruction
        )
        result = _media(
            media,
            _sha(key, "frame", str(turn)),
            f"edited_{kind}_{turn}",
            modality,
            frame,
            parent_id=steps[-1][1] if steps else artifacts[0].artifact_id,
            version=f"v{turn}",
            provenance=provenance,
        )
        artifacts += [ask, result]
        steps.append((ask.artifact_id, result.artifact_id))
    view = edit_chain_view(
        family=f"{kind}_edit",
        view_id=f"verified-{kind}-edit-chain-{len(steps)}",
        original=artifacts[0].artifact_id,
        steps=steps,
    )
    return compile_bundle(
        source,
        f"{source.source_id}:{kind}-edit-chain:{key}",
        clip["family"],
        artifacts,
        preprocessing=f"{source.source_id}-{kind}-edit-chain-v1",
        tier=tier,
        views=(view,),
    )


def judging_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any], *, tier: str | None = None
) -> list[Episode]:
    """The original, the request and several candidate edits are shown; the verdict is written.

    ``clip["candidates"]`` are the candidate images in the order shown, ``clip["verdict"]`` the
    judgement (scores and which candidate is best, or a critique).
    """
    original = _media(
        media, _sha(clip["id"], "original"), "original_image", "image", clip["original"]
    )
    candidates = [
        _media(
            media,
            _sha(clip["id"], "candidate", str(number)),
            f"candidate_image_{number}",
            "image",
            path,
            parent_id=original.artifact_id,
            version="v1",
            provenance=_provenance(source, "media"),
        )
        for number, path in enumerate(clip["candidates"], 1)
    ]
    request = _text(
        source, _sha(clip["id"], "request"), "image_edit_instruction", clip["instruction"]
    )
    verdict = _text(source, _sha(clip["id"], "verdict"), "edit_verdict", clip["verdict"])
    view = edit_judging_view(
        view_id=f"edit-judging-{len(candidates)}",
        original=original.artifact_id,
        instruction=request.artifact_id,
        candidates=[candidate.artifact_id for candidate in candidates],
        verdict=verdict.artifact_id,
    )
    return compile_bundle(
        source,
        f"{source.source_id}:edit-judging:{clip['id']}",
        clip["family"],
        [original, request, *candidates, verdict],
        preprocessing=f"{source.source_id}-edit-judging-v1",
        tier=tier,
        views=(view,),
    )


def plan_episodes(
    media: Preparer,
    source: OpenSource,
    kind: str,
    clip: Mapping[str, Any],
    *,
    provenance: str,
    tier: str | None = None,
) -> list[Episode]:
    """The original and the request are shown; the model writes a plan (or tool call) and then
    renders the edit it describes. ``clip["frames"]`` is the original followed by each step's
    result and ``clip["plans"]`` each step's plan text; ``kind`` is ``image`` or ``av``."""
    modality = {"image": "image", "av": "video", "speech": "audio"}[kind]
    frames, plans = clip["frames"], clip["plans"]
    if len(frames) != len(plans) + 1 or not plans:
        raise ValueError("An edit plan needs one plan per edited version")
    key = clip["id"]
    original = _media(media, _sha(key, "frame", "0"), f"original_{kind}", modality, frames[0])
    request = _text(source, _sha(key, "request"), f"{kind}_edit_instruction", clip["instruction"])
    artifacts = [original, request]
    steps = []
    previous = original.artifact_id
    for turn, (frame, plan) in enumerate(zip(frames[1:], plans, strict=True), 1):
        written = _text(source, _sha(key, "plan", str(turn)), f"{kind}_edit_plan_{turn}", plan)
        result = _media(
            media,
            _sha(key, "frame", str(turn)),
            f"edited_{kind}_{turn}",
            modality,
            frame,
            parent_id=previous,
            version=f"v{turn}",
            provenance=provenance,
        )
        artifacts += [written, result]
        steps.append((written.artifact_id, result.artifact_id))
        previous = result.artifact_id
    view = edit_plan_view(
        family="edit_plan",
        view_id=f"edit-plan-{kind}-{len(steps)}",
        original=original.artifact_id,
        instruction=request.artifact_id,
        steps=steps,
    )
    return compile_bundle(
        source,
        f"{source.source_id}:{kind}-edit-plan:{key}",
        clip["family"],
        artifacts,
        preprocessing=f"{source.source_id}-{kind}-edit-plan-v1",
        tier=tier,
        views=(view,),
    )


DESCRIPTION_FAMILY = {
    "image": "image_description",
    "audio": "audio_description",
    "video": "video_description",
}


def _question_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any], modality: str
) -> list[Episode]:
    """A conversation about one image, recording or video -> one multi-turn episode, and the
    media's identity. ``clip["turns"]`` is the list of ``(question, answer)`` texts in order."""
    subject = _media(
        media,
        _sha(clip["id"], modality),
        modality,
        modality,
        clip[modality],
        provenance=_provenance(source, "media"),
    )
    artifacts = [subject]
    turns = []
    for number, (question, answer) in enumerate(clip["turns"], 1):
        asked = _text(source, _sha(clip["id"], "q", str(number)), f"question_{number}", question)
        replied = _text(source, _sha(clip["id"], "a", str(number)), f"answer_{number}", answer)
        artifacts += [asked, replied]
        turns.append((asked.artifact_id, replied.artifact_id))
    view = question_chain_view(
        family=DESCRIPTION_FAMILY[modality],
        view_id=f"grounded-{modality}-questions-{len(turns)}",
        media=subject.artifact_id,
        turns=turns,
        modality=modality,
    )
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        artifacts,
        preprocessing=f"{source.source_id}-{modality}-questions-v1",
        views=(view,),
    )
    return [*episodes, identity_episode(subject.media, clip["family"])]


def question_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any]
) -> list[Episode]:
    return _question_episodes(media, source, clip, "image")


def audio_question_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any]
) -> list[Episode]:
    return _question_episodes(media, source, clip, "audio")


def video_question_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any]
) -> list[Episode]:
    return _question_episodes(media, source, clip, "video")


def video_caption_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any]
) -> list[Episode]:
    """A (silent or not) clip with a written description -> video description and video identity."""
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
        (video, _text(source, _sha(clip["id"], "caption"), "video_caption", clip["caption"])),
        preprocessing=f"{source.source_id}-video-caption-v1",
    )
    return [*episodes, identity_episode(video.media, clip["family"])]


def audio_caption_episodes(
    media: Preparer, source: OpenSource, clip: Mapping[str, Any]
) -> list[Episode]:
    """A recording with a written description -> audio description and audio identity."""
    audio = _media(media, _sha(clip["id"], "sound"), "audio", "audio", clip["audio"])
    episodes = compile_bundle(
        source,
        f"{source.source_id}:{clip['id']}",
        clip["family"],
        (audio, _text(source, _sha(clip["id"], "caption"), "audio_caption", clip["caption"])),
        preprocessing=f"{source.source_id}-audio-caption-v1",
    )
    return [*episodes, identity_episode(audio.media, clip["family"])]


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


def is_web(source: OpenSource) -> bool:
    """Whether the source is a published site (``repository`` is its base URL), not a Hub dataset."""
    return source.repository.startswith(("https://", "http://"))


class Shards:
    """The files one source's reader may open.

    With ``<inputs>/<source_id>`` present these are the files in it, named as in the repository;
    otherwise they are the repository's files at the pinned revision, streamed over HTTP range
    requests (the Hub's own credentials come from ``HF_TOKEN``).
    """

    def __init__(
        self,
        source: OpenSource,
        inputs: Path | None = None,
        shard_slice: tuple[int, int] = (0, 1),
        files: Sequence[str] | None = None,
        stripe: tuple[int, int] | None = None,
        sizes: Mapping[str, int] | None = None,
        cache: Path | None = None,
    ):
        self.source = source
        self.cache = cache
        self.shard_slice = shard_slice
        self.files = files
        self.stripe = stripe
        self.sizes = sizes or {}
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
        each build a disjoint part of one source; ``files`` names exactly the paths to read
        (a shard of ``data/media_sharding.py``).
        """
        if self.files is not None and patterns is None:
            return list(self.files)[self.shard_slice[0] :: self.shard_slice[1]]
        if is_web(self.source) and self.local is None:
            # A published site has no file listing: ``include`` names its files.
            return list(patterns or self.source.include)[self.shard_slice[0] :: self.shard_slice[1]]
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
        elif is_web(self.source):
            # Whole file, fetched once with resume and checked against the pinned digest.
            root = self.cache or Path(tempfile.gettempdir()) / "megatoken-web-cache"
            path = download(
                f"{self.source.repository.rstrip('/')}/{name}",
                root / self.source.source_id / name,
                sha256=self.source.options.get("pins", {}).get(name),
            )
            with path.open("rb") as handle:
                yield handle
        else:
            # A file of a committed listing is opened with its known size: no Hub API call
            # (the account's calls are shared by every machine of a build).
            known = {"size": self.sizes[name]} if name in self.sizes else {}
            with self._hub().open(self._remote(name), "rb", block_size=32 << 20, **known) as handle:
                yield handle


# --- records: rows, samples and archive members ---------------------------------------------


def locate(record: Any, path: str) -> Any:
    """Follow a dotted path through dicts and lists (list steps are integers); None if absent.

    A mapping key may itself contain dots (a WebDataset member ``image.jpg``): the longest
    run of path parts that names a key is taken first.
    """
    parts = path.split(".")
    while parts:
        if isinstance(record, Mapping):
            for length in range(len(parts), 0, -1):
                key = ".".join(parts[:length])
                if key in record:
                    record, parts = record[key], parts[length:]
                    break
            else:
                return None
        elif isinstance(record, (list, tuple)) and parts[0].lstrip("-").isdigit():
            index = int(parts[0])
            record = record[index] if -len(record) <= index < len(record) else None
            parts = parts[1:]
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


def audio_blob(value: Any) -> bytes | None:
    """The bytes of an audio field: a stored file, or decoded samples (``array`` and
    ``sampling_rate``, as some publishers store them) written as 16-bit WAV."""
    payload = blob(value)
    if payload is not None or not isinstance(value, Mapping) or value.get("array") is None:
        return payload
    import numpy
    import soundfile

    buffer = io.BytesIO()
    samples = numpy.clip(numpy.asarray(value["array"], dtype="float32"), -1.0, 1.0)
    soundfile.write(buffer, samples, int(value["sampling_rate"]), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _top_level(source: OpenSource) -> list[str]:
    """The parquet columns a source's labelled fields, filters and uniqueness key read."""
    specs = [
        *source.columns.values(),
        *source.options.get("keep", {}),
        *source.options.get("absent", ()),
        *([source.options["unique"]] if "unique" in source.options else []),
        *source.options.get("session", {}).values(),
        *source.options.get("choose", {}).get("candidates", ()),
        *source.options.get("choose", {}).get("scores", ()),
        *(
            column
            for columns in source.options.get("judge", {}).get("scores", {}).values()
            for column in columns
        ),
        *source.options.get("judge", {}).get("overall", ()),
        *(
            [source.options["judge"]["vote"]["column"]]
            if "vote" in source.options.get("judge", {})
            else []
        ),
        *(
            [source.options["judge"]["reason"]]
            if "reason" in source.options.get("judge", {})
            else []
        ),
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
                    record["__key__"] = f"{name.removesuffix('.parquet')}:{row}"
                    row += 1
                    yield record


def _url(base: str, path: str, strip: str = "") -> str:
    """A published file's URL: an absolute link as it is, a relative path under ``base``."""
    if path.startswith(("https://", "http://")):
        return path
    return base.rstrip("/") + "/" + path.removeprefix(strip)


@functools.lru_cache(maxsize=2)
def _open_images_index(url: str, sha256: str | None) -> dict[str, str]:
    """Open Images' table of its pictures' original Flickr file names -> image ids.

    The table (a large CSV) is fetched once per machine into the temporary directory."""
    path = download(
        url,
        Path(tempfile.gettempdir())
        / "megatoken-web-cache"
        / "open-images"
        / PurePosixPath(url).name,
        sha256=sha256,
    )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        image_id, original = header.index("ImageID"), header.index("OriginalURL")
        return {
            row[original].rsplit("/", 1)[-1].split("?")[0]: row[image_id]
            for row in reader
            if len(row) > max(image_id, original)
        }


def _original_url(source: OpenSource, url: str) -> str:
    """Where to fetch a photo that was published as a Flickr link: Open Images keeps its own copy
    of most of them (``options.originals``: its table, its pin, and the copies' URL pattern), which
    is online when the Flickr photo is gone and is not rate limited like Flickr; a photo it does
    not hold stays a Flickr link."""
    spec = source.options.get("originals")
    if not spec:
        return url
    image_id = _open_images_index(spec["index"], spec.get("pin")).get(
        url.rsplit("/", 1)[-1].split("?")[0]
    )
    return spec["url"].format(id=image_id) if image_id else url


def _pico_session(source: OpenSource, row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Pico-Banana multi-turn row -> original, each turn's result in order, each turn's request.

    ``files`` lists ``edit_turn1`` ... ``edit_turn(n-1)``, ``final_image`` (the n-th result) and
    ``original_input_image``; every turn edits the one before it (turn 1 edits the original)."""
    base, files = source.options["base"], {item["id"]: item["url"] for item in row["files"]}
    turns = sorted(
        (name for name in files if name.startswith("edit_turn")), key=lambda n: int(n[9:])
    )
    results = [*turns, "final_image"]
    prompts = row["metadata_edit_turn_prompts"]
    if "original_input_image" not in files or len(results) != len(prompts):
        return None
    first = files[results[0]].rsplit("/", 1)[-1]
    return {
        "original": _original_url(source, _url(base, files["original_input_image"])),
        "edited": [_url(base, files[name]) for name in results],
        "instruction": list(prompts),
        "id": first.rpartition("_turn")[0] or first,
    }


def _pico_preference(source: OpenSource, row: Mapping[str, Any]) -> dict[str, Any]:
    """Pico-Banana preference row -> the original and its candidates, the chosen edit first
    (the rejected edits are listed under another folder name)."""
    base = source.options["base"]
    return {
        "original": _original_url(source, _url(base, row["open_image_input_url"])),
        "candidates": [
            _url(base, row["output_image"]),
            *(_url(base, path, "nano-banana-300k/") for path in row["rejected_images"]),
        ],
        "instruction": row["text"],
        "id": row["output_image"],
    }


def _pico_edit(source: OpenSource, row: Mapping[str, Any]) -> dict[str, Any]:
    base = source.options["base"]
    return {
        "original": _original_url(source, _url(base, row["open_image_input_url"])),
        "edited": _url(base, row["output_image"]),
        "instruction": row["text"],
        "id": row["output_image"],
    }


def table_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """Rows of table files (parquet, csv, JSON array or JSON lines), for datasets that name their
    pictures by URL (``options.fetch``) instead of holding them."""
    for name in shards.names():
        with shards.open(name) as handle:
            for number, row in enumerate(_table_rows(handle, name, None)):
                yield {**row, "__key__": f"{name}:{number}"}


_TEXT_SUFFIXES = frozenset((".txt", ".lua", ".json", ".md"))


def folders_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """One record per folder of loose files: the folders holding the files ``include`` matches,
    each record with the files ``options.members`` names (field name -> path inside the folder;
    text files as text, the rest as bytes). A folder missing a member is skipped."""
    for anchor in shards.names():
        folder = PurePosixPath(anchor).parent
        record: dict[str, Any] = {"__key__": folder.as_posix()}
        for field_name, relative in source.options["members"].items():
            try:
                with shards.open((folder / relative).as_posix()) as handle:
                    data = handle.read()
            except FileNotFoundError:
                break
            record[field_name] = (
                data.decode("utf-8", "replace")
                if PurePosixPath(relative).suffix in _TEXT_SUFFIXES
                else data
            )
        else:
            yield record


def _jarvis_plan(source: OpenSource, row: Mapping[str, Any]) -> dict[str, Any] | None:
    """JarvisArt's folder -> the retoucher's reasoning followed by the exact Lightroom settings
    (the nonzero entries of ``config.lua``) as a tool call, which renders the result."""
    settings: dict[str, Any] = {}
    for line in row["settings"].splitlines():
        name, separator, value = line.strip().rstrip(",").partition(" = ")
        if not separator:
            continue
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            settings[name] = value[1:-1]
        elif value in {"true", "false"}:
            if value == "true":
                settings[name] = True
        else:
            try:
                number = float(value)
            except ValueError:
                continue
            if number:
                settings[name] = int(number) if number.is_integer() else number
    if not settings or not row["reasoning"].strip():
        return None
    call = json.dumps({"tool": "lightroom.develop", "settings": settings}, ensure_ascii=False)
    return {
        "original": row["before"],
        "edited": row["after"],
        "instruction": row["request"],
        "plan": f"{row['reasoning'].strip()}\n\nTool call: {call}",
    }


# Rows whose layout is the publisher's own are reshaped (``options.transform``) into the columns a
# task reads; a row a transform cannot reshape is skipped.
def _wenetspeech4tts(source: OpenSource, row: Mapping[str, Any]) -> dict[str, Any] | None:
    """A WenetSpeech4TTS segment: ``wavs/<id>.wav`` and ``txts/<id>.txt`` (first line ``<id>``,
    a tab and the transcript; second line the segment's time span)."""
    first = str(row.get("txt", "")).split("\n", 1)[0]
    _, tab, text = first.partition("\t")
    if not tab or not text.strip() or "wav" not in row:
        return None
    return {"audio": row["wav"], "text": text.strip()}


ROW_TRANSFORMS: dict[str, Callable[[OpenSource, Mapping[str, Any]], dict[str, Any] | None]] = {
    "wenetspeech4tts": _wenetspeech4tts,
    "pico_session": _pico_session,
    "pico_preference": _pico_preference,
    "pico_edit": _pico_edit,
    "jarvis_plan": _jarvis_plan,
}


_TEXT_EXTENSIONS = frozenset(("txt", "cls", "caption"))


def _sample(group: Mapping[str, bytes]) -> dict[str, Any]:
    sample: dict[str, Any] = {}
    for extension, payload in group.items():
        suffix = extension.rpartition(".")[2]  # ``synthetic_caption.txt`` is text
        if suffix == "json":
            sample[extension] = json.loads(payload)
        elif suffix in _TEXT_EXTENSIONS:
            sample[extension] = payload.decode("utf-8")
        else:
            sample[extension] = payload
    return sample


def _spooled_samples(
    handle: BinaryIO, shard: str, scratch: Path | None
) -> Iterator[dict[str, Any]]:
    """The samples of one tar whose members of a sample are far apart: members are written to
    ``scratch`` while the archive streams, then each sample (the files sharing a name before
    the first dot, whatever their folder) is read back and its files deleted."""
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="spool-", dir=scratch))
    try:
        parts: dict[str, dict[str, Path]] = {}
        with tarfile.open(fileobj=handle, mode="r|*") as tar:
            for number, member in enumerate(tar):
                if not member.isfile():
                    continue
                stem, _, extension = PurePosixPath(member.name).name.partition(".")
                path = root / str(number)
                with tar.extractfile(member) as source_file, path.open("wb") as target:
                    shutil.copyfileobj(source_file, target)
                parts.setdefault(stem, {})[extension] = path
        for stem, files in parts.items():
            sample = _sample({extension: path.read_bytes() for extension, path in files.items()})
            for path in files.values():
                path.unlink()
            yield {"__key__": stem, "__shard__": shard, **sample}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def webdataset_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """Samples of consecutive tar members sharing a key; fields are named by file extension.

    ``__key__`` is the sample's name and ``__shard__`` the tar it came from: publishers restart
    their numbering in every tar, so an ID made of both (``columns.id`` may list several fields)
    is unique across the source. ``options.spool`` is for archives that list all their audio
    first and all its transcripts after (``wavs/a.wav`` ... ``txts/a.txt``): each archive is
    written to scratch disk as it streams, and its files are then paired by name.
    """
    for shard in shards.names():
        if source.options.get("spool"):
            with shards.open(shard) as handle:
                yield from _spooled_samples(handle, shard, shards.cache)
            continue
        with shards.open(shard) as handle, tarfile.open(fileobj=handle, mode="r|*") as tar:
            key: str | None = None
            group: dict[str, bytes] = {}
            for member in tar:
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name).name
                # A sample's files share the text before the first dot (``a.mp4``, ``a.json``), or
                # before the last when the key itself holds a dot (``video.3.mp4``).
                stem, _, extension = (
                    name.rpartition(".")
                    if source.options.get("key_split") == "last"
                    else name.partition(".")
                )
                if stem != key and group:
                    yield {"__key__": key, "__shard__": shard, **_sample(group)}
                    group = {}
                key = stem
                group[extension] = tar.extractfile(member).read()
            if group:
                yield {"__key__": key, "__shard__": shard, **_sample(group)}


class _Derived(dict):
    """A table row that also answers ``<field>__name`` (the last path component of a string
    field) and ``<field>__stem`` (that without its extension), for ``key`` templates."""

    def __missing__(self, key: str) -> str:
        field, _, kind = key.rpartition("__")
        if kind in ("name", "stem") and field in self:
            path = PurePosixPath(str(self[field]))
            return path.name if kind == "name" else path.stem
        raise KeyError(key)


def _table_rows(handle: BinaryIO, path: str, names: Sequence[str] | None) -> Iterable[dict]:
    """Rows of a table: a parquet file, a csv, a JSON array of objects, or JSON lines (whatever
    its name says)."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as parquet

        return (
            row
            for batch in parquet.ParquetFile(handle).iter_batches(batch_size=1024)
            for row in batch.to_pylist()
        )
    if path.endswith(".zip"):
        # A table that the publisher zipped: the archive's one file.
        archive = zipfile.ZipFile(handle)
        (member,) = [info for info in archive.infolist() if not info.is_dir()]
        return _table_rows(archive.open(member), member.filename, names)
    text = io.TextIOWrapper(handle, encoding="utf-8", newline="")
    if path.endswith(".tsv"):
        return csv.DictReader(text, fieldnames=names, delimiter="\t", quoting=csv.QUOTE_NONE)
    if not path.endswith((".json", ".jsonl")):
        return csv.DictReader(text, fieldnames=names)
    first = text.read(1)
    while first.isspace():
        first = text.read(1)
    if first == "[":
        return _json_array(text)
    rest = first + text.read()
    # JSON lines end at "\n" only; str.splitlines would also cut inside a string at U+2028.
    return (json.loads(line) for line in rest.split("\n") if line.strip())


def _json_array(text: io.TextIOBase, chunk: int = 1 << 22) -> Iterator[Any]:
    """The items of a JSON array whose "[" has been read, decoded one at a time (a table of a
    million rows is never held twice in memory)."""
    decoder = json.JSONDecoder()
    buffer, position, ended = "", 0, False
    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n,":
            position += 1
        if position >= len(buffer) and not ended:
            more = text.read(chunk)
            buffer, position, ended = more, 0, not more
            continue
        if position >= len(buffer) or buffer[position] == "]":
            return
        try:
            item, position = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            more = text.read(chunk)
            if not more:
                raise
            buffer, position = buffer[position:] + more, 0  # an item runs past the buffer
            continue
        yield item


def _table(shards: Shards, spec: Mapping[str, Any], wanted: Iterable[str]) -> dict[str, dict]:
    """Rows of a csv/json/jsonl file keyed by the ``key`` template (a ``str.format`` over the
    row, which also offers ``<field>__name`` and ``<field>__stem`` of path fields).

    ``path`` may name several files. Rows sharing a key replace one another, except that the
    fields listed in ``merge`` (lists) are joined in file order: one video asked about in many
    rows keeps all its conversations. ``prefix`` (field -> text) keeps only the rows whose field
    starts with the text: one table of several kinds of row (ids ``svit_detail_12``) is read as
    one kind at a time.
    """
    rows: dict[str, dict] = {}
    wanted = list(wanted)
    prefixes = spec.get("prefix", {})
    for path in [spec["path"]] if isinstance(spec["path"], str) else spec["path"]:
        with shards.open(path) as handle:
            for row in _table_rows(handle, path, spec.get("names")):
                if any(
                    not str(row.get(field, "")).startswith(text) for field, text in prefixes.items()
                ):
                    continue
                for name in spec.get("ints", ()):
                    row[name] = int(row[name])
                key = spec["key"].format_map(_Derived(row))
                kept = {name: row.get(name) for name in wanted}
                if key in rows and spec.get("merge"):
                    for name in spec["merge"]:
                        kept[name] = [*(rows[key][name] or ()), *(kept[name] or ())]
                rows[key] = kept
    return rows


def archive_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """Members of zip/tar archives joined with their table row (and, for zips, a partner member).

    ``options.table`` names the csv/json/jsonl file and the ``key`` template that matches a row to
    a member's file name; the member's bytes are the record's ``member``. A table with one file
    per archive says so in ``per_archive`` (``{"pattern": regex over the archive's name, "path":
    template over the regex groups}``) and is read when its archive is. ``options.partner`` (a
    template over the archive's file name) names a second zip whose same-named member is the
    record's ``partner``, as in a source/edited pair of videos.
    """
    spec = source.options["table"]
    each = spec.get("per_archive")
    rows = {} if each else _table(shards, spec, spec.get("columns", ()))
    partner_template = source.options.get("partner")
    for name in shards.names():
        if each:
            match = re.search(each["pattern"], PurePosixPath(name).name)
            if match is None:
                raise ValueError(f"{source.source_id}: {name} does not match {each['pattern']}")
            rows = _table(
                shards,
                {**spec, "path": each["path"].format(*match.groups())},
                spec.get("columns", ()),
            )
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


class Concatenated:
    """Several files read as one stream (the parts of an archive that was cut with ``split``)."""

    def __init__(self, handles: Sequence[BinaryIO]):
        self.handles = list(handles)

    def read(self, size: int = -1) -> bytes:
        chunks: list[bytes] = []
        while self.handles and size != 0:
            chunk = self.handles[0].read(size)
            if chunk:
                chunks.append(chunk)
                size -= len(chunk) if size > 0 else 0
                if size == 0:
                    break
            else:
                self.handles.pop(0)
        return b"".join(chunks)


def _numbered(row: Mapping[str, Any], templates: Sequence[str]) -> list[Any]:
    """Row values named by ``templates``: a plain (dotted) path once, a template with ``{n}`` for
    n = 1, 2, ... while the row has it."""
    values: list[Any] = []
    for template in templates:
        if "{n}" not in template:
            found = locate(row, template)
            values.extend(found if isinstance(found, list) else [found])
            continue
        number = 1
        while locate(row, template.format(n=number)) not in (None, ""):
            values.append(locate(row, template.format(n=number)))
            number += 1
    return values


def _path_key(path: str) -> str:
    return PurePosixPath(str(path).lower().replace("\\", "/")).as_posix().lstrip("./")


def _archive_streams(shards: Shards, names: Sequence[str], multipart: bool) -> Iterator[BinaryIO]:
    """The archives one after another, each open only while it is read (or all as one stream)."""
    with ExitStack() as stack:
        if multipart:
            yield Concatenated([stack.enter_context(shards.open(name)) for name in names])
            return
        for name in names:
            with shards.open(name) as handle:
                yield handle


def linked_records(source: OpenSource, shards: Shards) -> Iterator[dict[str, Any]]:
    """Table rows joined with the images they name, read from archives in one pass.

    ``include`` names the tables (parquet, JSON lines or one JSON array of rows) and ``options.linked``
    the images:
    ``strip`` is a regex removed from the row's paths first (a leading folder the archive lacks),
    ``archives`` are patterns of tar(.gz) files (``multipart``: together one archive that was cut
    into parts), ``sequence`` the row's media in order (``images``: field names, ``{n}`` standing
    for 1, 2, ... while the row has it, a list field counting as several; ``texts`` likewise). An archive member belongs to a row
    when the row's path is the member's name or its tail (case-insensitive: ``a/b.jpg`` is
    ``data/a/B.JPG``). A row is yielded, with ``images`` and ``texts`` lists, as soon as its
    last image has been read, and dropped when more than ``pending`` of its images wait unmatched
    (the archive keeps a row's images apart); a row whose image is never found is skipped.
    """
    spec = source.options["linked"]
    # An image path may be named by several rows, or twice by one: each use is a target.
    wanted: dict[str, list[tuple[int, int]]] = {}
    rows: dict[int, dict[str, Any]] = {}
    strip = re.compile(spec["strip"]) if "strip" in spec else None
    for table in shards.names(source.include):
        with shards.open(table) as handle:
            for row in _table_rows(handle, table, None):
                paths = _numbered(row, spec["sequence"]["images"])
                if strip is not None:
                    paths = [
                        strip.sub("", path) if isinstance(path, str) else path for path in paths
                    ]
                if not paths or any(not isinstance(path, str) for path in paths):
                    continue
                number = len(rows)
                rows[number] = {
                    "row": row,
                    "paths": paths,
                    "found": {},
                    "texts": _numbered(row, spec["sequence"].get("texts", ())),
                }
                for position, path in enumerate(paths):
                    wanted.setdefault(_path_key(path), []).append((number, position))
    # A shard of a source with many archives reads its own (``files``); a table row whose
    # images are in the others never completes and is dropped.
    names = list(shards.files) if shards.files is not None else shards.names(spec["archives"])
    pending_limit = spec.get("pending", 4096)
    partial: dict[int, None] = {}  # rows with some image read, oldest first
    waiting = 0  # images read for rows that are not complete
    for stream in _archive_streams(shards, names, bool(spec.get("multipart"))):
        with tarfile.open(fileobj=stream, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = _path_key(member.name).split("/")
                targets = next(
                    (
                        wanted[key]
                        for key in ("/".join(parts[start:]) for start in range(len(parts)))
                        if key in wanted
                    ),
                    (),
                )
                data = tar.extractfile(member).read() if targets else b""
                for number, position in targets:
                    entry = rows.get(number)
                    if entry is None or position in entry["found"]:
                        continue  # the row is complete, or this image is already read
                    entry["found"][position] = data
                    if len(entry["found"]) == len(entry["paths"]):
                        record = {
                            "__key__": entry["row"].get("id") or str(number),
                            **entry["row"],
                            "images": [entry["found"][i] for i in range(len(entry["paths"]))],
                            "texts": entry["texts"],
                        }
                        del rows[number]
                        partial.pop(number, None)
                        waiting -= len(entry["paths"]) - 1
                        yield record
                    else:
                        partial[number] = None
                        waiting += 1
                        if waiting > pending_limit:
                            # Too many images wait for partners that are not coming.
                            oldest = next(iter(partial))
                            waiting -= len(rows[oldest]["found"])
                            rows[oldest]["found"] = {}
                            del partial[oldest]


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


ASPECT_TOLERANCE = 0.03


def _aligned(images: Sequence[Any]) -> list[Any]:
    """The original and its successive edits at the size of the first edit.

    An editing model returns its own size (MagicBrush: 1024 px for a 500 px photo), so a
    version of another size but the same shape (aspect ratio within ``ASPECT_TOLERANCE``) is
    resized to it. A version of another shape ends the chain; if that is the original, nothing
    is left.
    """
    from PIL import Image

    if len(images) < 2:
        return []
    width, height = images[1].size
    aligned = []
    for image in images:
        if image.size != (width, height):
            if abs(image.width / image.height - width / height) > ASPECT_TOLERANCE * width / height:
                break
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        aligned.append(image)
    return aligned if len(aligned) >= 2 and min(width, height) >= 32 else []


def write_image_chain(
    frames: Sequence[bytes], out: Path, stem: str, *, max_side: int = MAX_SIDE
) -> list[Path] | None:
    """An original and each successive edit, scaled identically and saved losslessly (so
    unedited pixels stay equal). The chain stops at the first version that is undecodable or
    cannot follow its predecessor (``_aligned``)."""
    from PIL import Image

    images = []
    for data in frames:
        try:
            images.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except (OSError, ValueError, Image.DecompressionBombError):
            break
    images = _aligned(images)
    if not images:
        return None
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(images):
        image.thumbnail((max_side, max_side))
        paths.append(out / f"{stem}-{index}.png")
        image.save(paths[-1])
    return paths


def write_image_pair(
    original: bytes, edited: bytes, out: Path, stem: str, *, max_side: int = MAX_SIDE
) -> tuple[Path, Path] | None:
    """An image and its edit scaled identically (lossless, so unedited pixels stay equal)."""
    paths = write_image_chain((original, edited), out, stem, max_side=max_side)
    if paths is None or len(paths) != 2:
        return None
    named = (out / f"{stem}-original.png", out / f"{stem}-edited.png")
    for path, target in zip(paths, named, strict=True):
        path.replace(target)
    return named


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
    fps: int = VIDEO_FPS,
) -> None:
    """A clip as small video (scaled into 640x640 at ``fps`` frames/s, no sound) and, optionally,
    its soundtrack at 24 kHz mono (the speech realizer's rate). The video realizer takes at most
    193 frames, so ``seconds * fps`` stays below that."""
    cut = ["-ss", str(start), *(["-t", str(seconds)] if seconds else [])]
    _ffmpeg(
        *cut,
        "-i",
        str(source),
        "-an",
        "-vf",
        "scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-r",
        str(fps),
        str(video),
    )
    if audio is not None:
        _ffmpeg(*cut, "-i", str(source), "-vn", "-ac", "1", "-ar", "24000", str(audio))


def _fps(source: OpenSource) -> int:
    return source.options.get("fps", VIDEO_FPS)


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


def _without_image_tags(text: str) -> str | None:
    """A prompt as written for a model that was shown images: without the ``<image>`` markers."""
    return _IMAGE_TAG.sub(" ", text).strip() or None


TEXT_CLEANERS: dict[str, Callable[[str], str | None]] = {
    "gigaspeech": _gigaspeech,
    "image_tags": _without_image_tags,
}


def _label(source: OpenSource, record: Mapping[str, Any], role: str) -> Any:
    return field_of(record, source.columns.get(role))


def _text_value(source: OpenSource, value: Any) -> str | None:
    """A non-empty text label, shaped by the source's options (picking one of a list, cleaning)."""
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


def _text_label(source: OpenSource, record: Mapping[str, Any], role: str) -> str | None:
    return _text_value(source, _label(source, record, role))


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
    payload = audio_blob(_label(source, record, "audio"))
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
    if source.options.get("sustained_translation"):
        partition = record.get("__key__", "").split("/", 1)[0]
        language = source.options.get("language_pairs", {}).get(partition)
        if not language:
            raise ValueError(
                f"{source.source_id}: streaming source language is not declared for {partition!r}"
            )
        clip["language"] = language
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


def audio_caption_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    caption = _text_label(source, record, "caption")
    if caption is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    audio = _speech_file(source, record, out, _stem(clip_id))
    return [{"id": clip_id, "family": family, "audio": audio, "caption": caption}] if audio else []


_USER_ROLES = frozenset(("human", "user"))
_ASSISTANT_ROLES = frozenset(("gpt", "assistant", "model"))
_IMAGE_TAG = re.compile(r"\s*<image(?:_\d+)?>\s*")


def _clean_turn(text: Any) -> str | None:
    """A turn's text without ``<image>`` placeholders; None if nothing is left."""
    if not isinstance(text, str):
        return None
    text = _IMAGE_TAG.sub(" ", text).strip()
    return text or None


def conversation(source: OpenSource, record: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The ``(question, answer)`` turns of a record, whichever way the publisher stored them.

    ``options.format`` is ``choices`` (``question``, ``choices`` and ``answer`` fields: a
    multiple-choice question), ``pairs`` (a list of ``{user, assistant}`` mappings, the keys named by
    ``options.user_key`` / ``options.assistant_key``), ``messages`` (a list of role/text
    mappings: ``from``/``value`` or ``role``/``content``; a turn that does not alternate user,
    assistant ends the conversation) or ``parallel`` (the ``question`` and ``answer`` fields
    are equally long lists, or single strings). ``options.max_turns`` cuts a longer one and
    ``options.min_answer_chars`` drops a turn with a shorter answer (and every later one).
    """
    layout = source.options.get("format", "pairs")
    turns: list[tuple[str, str]] = []
    if layout == "parallel":
        questions, answers = (_list_label(source, record, role) for role in ("question", "answer"))
        pairs = list(zip(questions, answers, strict=False))
    elif layout == "choices":
        # A multiple-choice question: the options are listed in the question and the answer
        # is the right option's letter and text (the answer field is its index or its text).
        pairs = []
        questions, option_lists, answers = (
            _list_label(source, record, role) for role in ("question", "choices", "answer")
        )
        if option_lists and not isinstance(option_lists[0], (list, tuple)):
            option_lists = [option_lists]  # one question's options
        for question, options, answer in zip(questions, option_lists, answers, strict=False):
            options = [str(option) for option in options or ()]
            right = (
                answer
                if isinstance(answer, int)
                else (options.index(answer) if answer in options else None)
            )
            if not options or right is None or not 0 <= right < len(options):
                continue
            letters = [f"{chr(65 + i)}. {option}" for i, option in enumerate(options)]
            pairs.append((f"{question}\n" + "\n".join(letters), letters[right]))
    elif layout == "pairs":
        user, assistant = (
            source.options.get("user_key", "user"),
            source.options.get("assistant_key", "assistant"),
        )
        pairs = [
            (turn.get(user), turn.get(assistant))
            for turn in _list_label(source, record, "turns")
            if isinstance(turn, Mapping)
        ]
    elif layout == "messages":
        messages = [m for m in _list_label(source, record, "turns") if isinstance(m, Mapping)]
        role = lambda m: str(m.get("from") or m.get("role") or "").lower()  # noqa: E731
        text = lambda m: m.get("value") if "value" in m else m.get("content")  # noqa: E731
        if messages and role(messages[0]) == "system":
            messages = messages[1:]
        pairs = [
            (text(a), text(b))
            for a, b in takewhile(
                lambda pair: role(pair[0]) in _USER_ROLES and role(pair[1]) in _ASSISTANT_ROLES,
                zip(messages[0::2], messages[1::2], strict=False),
            )
        ]
    else:
        raise ValueError(f"{source.source_id}: unknown conversation format {layout!r}")
    for question, answer in pairs:
        question, answer = _clean_turn(question), _clean_turn(answer)
        if (
            question is None
            or answer is None
            or len(answer) < source.options.get("min_answer_chars", 1)
        ):
            break
        turns.append((question, answer))
    return turns[: source.options.get("max_turns", 8)]


def audio_question_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """A conversation about one recording."""
    turns = conversation(source, record)
    if not turns or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    audio = _speech_file(source, record, out, _stem(clip_id))
    return [{"id": clip_id, "family": family, "audio": audio, "turns": turns}] if audio else []


def _cut_video(source: OpenSource, payload: bytes, clip_id: str, out: Path) -> Path | None:
    """A clip of at most ``max_seconds`` as small video (the source's own sound dropped);
    ``options.whole_only`` skips a longer video instead of cutting it."""
    original = _video_source(payload, out, _stem(clip_id))
    try:
        seconds, kinds = probe(original)
        longest = source.options.get("max_seconds", 8.0)
        if "video" not in kinds or seconds < source.options.get("min_seconds", 1.5):
            return None
        if source.options.get("whole_only") and seconds > longest:
            return None  # a question about the whole video cannot be asked of a cut of it
        video = out / f"{_stem(clip_id)}.mp4"
        transcode(
            original,
            video,
            seconds=min(seconds, source.options.get("max_seconds", 8.0)),
            fps=_fps(source),
        )
    finally:
        original.unlink()
    return video


def video_question_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """A conversation about one video (the first ``max_seconds`` of it)."""
    turns = conversation(source, record)
    payload = blob(_label(source, record, "video"))
    if not turns or payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    video = _cut_video(source, payload, clip_id, out)
    return [{"id": clip_id, "family": family, "video": video, "turns": turns}] if video else []


def video_caption_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    caption = _text_label(source, record, "caption")
    payload = blob(_label(source, record, "video"))
    if caption is None or payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    video = _cut_video(source, payload, clip_id, out)
    return [{"id": clip_id, "family": family, "video": video, "caption": caption}] if video else []


def question_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """A conversation about one image (a record with several images is skipped)."""
    images = _list_label(source, record, "image")
    turns = conversation(source, record)
    payload = blob(images[0]) if len(images) == 1 else None
    if not turns or payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    image = write_image(
        payload, out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    return [{"id": clip_id, "family": family, "image": image, "turns": turns}] if image else []


def picture_clips(source: OpenSource, record: Mapping[str, Any], out: Path) -> list[dict[str, Any]]:
    payload = blob(_label(source, record, "image"))
    if payload is None or not _accepts(source, record):
        return []
    clip_id, family = _identity(source, record)
    image = write_image(
        payload, out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    return [{"id": clip_id, "family": family, "image": image}] if image else []


def _chosen(source: OpenSource, record: Mapping[str, Any]) -> Any:
    """The best of several candidate edits: ``options.choose`` pairs ``candidates`` (columns)
    with their ``scores`` (columns of ratings); the highest rated is taken, None if unrated."""
    choice = source.options["choose"]
    scores = [field_of(record, name) for name in choice["scores"]]
    if any(not isinstance(score, (int, float)) for score in scores):
        return None
    return field_of(record, choice["candidates"][scores.index(max(scores))])


def image_edit_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    instruction = _text_label(source, record, "instruction")
    original = blob(_label(source, record, "original"))
    edited = blob(
        _chosen(source, record) if "choose" in source.options else _label(source, record, "edited")
    )
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
        transcode(original, video, audio, seconds=min(seconds, high), fps=_fps(source))
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
            transcode(original, video, audio, start=start, seconds=seconds, fps=_fps(source))
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
        transcode(original, video, seconds=min(seconds, high), fps=_fps(source))
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
            transcode(path, paths[name], seconds=seconds, fps=_fps(source))
    finally:
        for path in sources.values():
            path.unlink()
    return [{"id": clip_id, "family": family, **paths, "instruction": instruction}]


def _list_label(source: OpenSource, record: Mapping[str, Any], role: str) -> list[Any]:
    """A labelled field as a list: a session's per-turn values, or one value."""
    value = _label(source, record, role)
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _original_and_edits(source: OpenSource, record: Mapping[str, Any]) -> list[Any]:
    """The first version (None if it is missing: the edits then have nothing to follow) and
    each result."""
    original = _label(source, record, "original")
    if isinstance(original, (list, tuple)):
        original = original[0] if original else None
    return [original, *_list_label(source, record, "edited")]


def _chain_inputs(
    source: OpenSource, record: Mapping[str, Any]
) -> tuple[list[bytes], list[str]] | None:
    """The versions (original, then each result) and requests of one editing session.

    Fields are ``original`` (the first version, or every turn's input), ``edited`` (each turn's
    result) and ``instruction`` (each turn's request), or ``images`` (all versions in order)
    in place of the first two. ``options.max_steps`` cuts a longer session.
    """
    if not _accepts(source, record):
        return None
    if "images" in source.columns:
        versions = [blob(item) for item in _list_label(source, record, "images")]
    else:
        versions = [blob(item) for item in _original_and_edits(source, record)]
    instructions = [
        _text_value(source, item) for item in _list_label(source, record, "instruction")
    ]
    steps = min(len(versions) - 1, len(instructions), source.options.get("max_steps", 8))
    # A session continues only while every version and request is usable.
    usable = 0
    while usable < steps and versions[usable + 1] and instructions[usable]:
        usable += 1
    if usable < 1 or not versions[0]:
        return None
    return versions[: usable + 1], instructions[:usable]


def image_chain_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """A multi-turn image editing session: the original image, each result and each request."""
    inputs = _chain_inputs(source, record)
    if inputs is None:
        return []
    clip_id, family = _identity(source, record)
    paths = write_image_chain(
        inputs[0], out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    if paths is None:
        return []
    return [
        {
            "id": clip_id,
            "family": family,
            "frames": paths,
            "instructions": inputs[1][: len(paths) - 1],
        }
    ]


def video_chain_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """A multi-turn video editing session; every version is trimmed and scaled identically."""
    inputs = _chain_inputs(source, record)
    if inputs is None:
        return []
    clip_id, family = _identity(source, record)
    sources = [
        _video_source(data, out, f"{_stem(clip_id)}-{index}")
        for index, data in enumerate(inputs[0])
    ]
    try:
        durations = [probe(path)[0] for path in sources]
        kept = 1
        while (
            kept < len(sources)
            and min(durations[0], durations[kept]) >= 1.0
            and abs(durations[0] - durations[kept]) <= 0.5
        ):
            kept += 1
        if kept < 2:
            return []
        seconds = min(min(durations[:kept]), source.options.get("max_seconds", 6.0))
        paths = [out / f"{_stem(clip_id)}-{index}.mp4" for index in range(kept)]
        for path, video in zip(sources, paths, strict=False):
            transcode(path, video, seconds=seconds, fps=_fps(source))
    finally:
        for path in sources:
            path.unlink(missing_ok=True)
    return [
        {"id": clip_id, "family": family, "frames": paths, "instructions": inputs[1][: kept - 1]}
    ]


def _plan_text(source: OpenSource, value: Any) -> str | None:
    """A plan or tool call as text: written text as it is, a structure (tool calls, settings) as
    compact JSON."""
    if isinstance(value, (Mapping, list, tuple)):
        value = _without_nulls(value)
        value = json.dumps(value, ensure_ascii=False, separators=(", ", ": ")) if value else None
    return _text_value(source, value)


def _without_nulls(value: Any) -> Any:
    """A structure without the null fields that a table's common schema adds to every row."""
    if isinstance(value, Mapping):
        return {k: _without_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_without_nulls(item) for item in value]
    return value


def _plan_inputs(
    source: OpenSource, record: Mapping[str, Any]
) -> tuple[list[bytes], str, list[str]] | None:
    """The versions (original, then each result), the request and each step's plan of one record.

    ``plan`` is one step's plan (a list is the tool calls of that one step) unless
    ``options.plan_per_step``, where each item is the plan of one result in ``edited``."""
    if not _accepts(source, record):
        return None
    instruction = _text_label(source, record, "instruction")
    value = _label(source, record, "plan")
    items = (
        list(value) if source.options.get("plan_per_step") and isinstance(value, list) else [value]
    )
    plans = [_plan_text(source, item) for item in items]
    versions = [blob(item) for item in _original_and_edits(source, record)]
    steps = min(len(versions) - 1, len(plans), source.options.get("max_steps", 8))
    usable = 0
    while usable < steps and versions[usable + 1] and plans[usable]:
        usable += 1
    if instruction is None or usable < 1 or not versions[0]:
        return None
    return versions[: usable + 1], instruction, plans[:usable]


def image_plan_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """An image edit with the plan (or tool call) written before its result."""
    inputs = _plan_inputs(source, record)
    if inputs is None:
        return []
    clip_id, family = _identity(source, record)
    paths = write_image_chain(
        inputs[0], out, _stem(clip_id), max_side=source.options.get("max_side", MAX_SIDE)
    )
    if paths is None:
        return []
    return [
        {
            "id": clip_id,
            "family": family,
            "frames": paths,
            "instruction": inputs[1],
            "plans": inputs[2][: len(paths) - 1],
        }
    ]


def _verdict(source: OpenSource, record: Mapping[str, Any], order: Sequence[int]) -> str | None:
    """The judgement of the candidates shown in ``order`` (indexes into the source's candidates).

    ``columns.verdict`` is a judgement written out by the publisher. Otherwise the text is built
    from ``options.judge``:

    - ``scores``: aspect -> one score column per candidate (``outof``: the scale's top);
    - ``overall``: one column per candidate (``overall_outof``: its scale's top);
    - which candidate is best, taken from the first of ``vote`` (a column and the value meaning
      each candidate, or a tie), ``winner`` (the index of the best candidate in the source's
      list, when the publisher lists the preferred edit first) or the highest ``overall`` (else
      the sum of the scores);
    - ``reason``: a column of text added at the end.
    """
    written = _text_label(source, record, "verdict")
    judge = source.options.get("judge", {})
    if written is not None:
        return written
    letters = [chr(65 + position) for position in range(len(order))]
    scores = {
        aspect: [field_of(record, column) for column in columns]
        for aspect, columns in judge.get("scores", {}).items()
    }
    overall = [field_of(record, column) for column in judge.get("overall", ())]
    for values in [*scores.values(), overall]:
        if any(not isinstance(value, (int, float)) for value in values):
            return None
    lines = []
    for position, index in enumerate(order):
        parts = [
            f"{aspect} {_number(values[index])}"
            + (f" of {judge['outof']}" if "outof" in judge else "")
            for aspect, values in scores.items()
        ]
        if overall:
            top = judge.get("overall_outof", judge.get("outof"))
            parts.append(f"overall {_number(overall[index])}" + (f" of {top}" if top else ""))
        if parts:
            lines.append(f"Candidate {letters[position]}: {', '.join(parts)}.")
    if "vote" in judge:
        value = field_of(record, judge["vote"]["column"])
        meaning = judge["vote"]["values"]
        if value not in meaning:
            return None
        best = [] if meaning[value] is None else [meaning[value]]
    elif "winner" in judge:
        best = [judge["winner"]]
    elif scores or overall:
        basis = overall or [sum(values[i] for values in scores.values()) for i in range(len(order))]
        top = max(basis)
        best = [index for index, total in enumerate(basis) if total == top]
        best = [] if len(best) == len(order) else best
    else:
        return None
    winners = [letters[list(order).index(index)] for index in best]
    if not winners:
        lines.append("Verdict: the candidates are equally good.")
    elif len(winners) > 1:
        lines.append(f"Verdict: Candidates {' and '.join(sorted(winners))} are the best edits.")
    elif len(order) == 2:
        lines.append(f"Verdict: Candidate {winners[0]} is the better edit.")
    else:
        lines.append(f"Verdict: Candidate {winners[0]} is the best edit; the others are worse.")
    reason = _text_value(source, field_of(record, judge["reason"])) if "reason" in judge else None
    if reason:
        lines.append(reason)
    return "\n".join(lines)


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def edit_judging_clips(
    source: OpenSource, record: Mapping[str, Any], out: Path
) -> list[dict[str, Any]]:
    """An original, a request and two or more candidate edits with the verdict on them.

    The candidates are shown in a fixed pseudo-random order (by clip id), so the best one is
    not always first."""
    if not _accepts(source, record):
        return []
    instruction = _text_label(source, record, "instruction")
    spec = source.columns["candidates"]
    values = (
        [locate(record, name) for name in spec]
        if isinstance(spec, (list, tuple))
        else _list_label(source, record, "candidates")
    )
    images = [blob(_label(source, record, "original")), *(blob(item) for item in values)]
    if instruction is None or len(images) < 2 or any(image is None for image in images):
        return []
    clip_id, family = _identity(source, record)
    order = sorted(range(len(values)), key=lambda index: _sha(clip_id, "order", str(index)))
    verdict = _verdict(source, record, order)
    if verdict is None:
        return []
    paths = write_image_chain(
        [images[0], *(images[1 + index] for index in order)],
        out,
        _stem(clip_id),
        max_side=source.options.get("max_side", MAX_SIDE),
    )
    if paths is None or len(paths) != len(images):
        return []
    return [
        {
            "id": clip_id,
            "family": family,
            "original": paths[0],
            "candidates": paths[1:],
            "instruction": instruction,
            "verdict": verdict,
        }
    ]


SPEECH_PLAN_EVERY = 8  # one speech edit in eight also gives a plan episode
SPEECH_EDITS = (
    ("Make the speech 6 decibels louder.", ("-af", "volume=6dB,alimiter=limit=0.97")),
    ("Speak 25 percent faster without changing the pitch.", ("-af", "atempo=1.25")),
    ("Play the recording backwards.", ("-af", "areverse")),
    ("Add a short echo.", ("-af", "aecho=0.8:0.88:60:0.4")),
    ("Fade the recording in over the first half second.", ("-af", "afade=t=in:d=0.5")),
    ("Cut out the low rumble below 150 hertz.", ("-af", "highpass=f=150")),
)
VIDEO_EDITS = (
    ("Mirror the video horizontally.", ("-vf", "hflip")),
    ("Flip the video upside down.", ("-vf", "vflip")),
    ("Convert the video to black and white.", ("-vf", "hue=s=0")),
    ("Play the video at double speed.", ("-vf", "setpts=0.5*PTS")),
    ("Play the video in reverse.", ("-vf", "reverse")),
    ("Make the video brighter.", ("-vf", "eq=brightness=0.15")),
    ("Blur the video.", ("-vf", "boxblur=6:1")),
    ("Rotate the video a quarter turn clockwise.", ("-vf", "transpose=1")),
)
# Picture edits are small programs on a PIL ``image``; the plan is the program itself, which is
# run to make the edited picture, so the plan and its result always agree.
IMAGE_EDITS = (
    (
        "Convert the picture to black and white.",
        'edited = ImageOps.grayscale(image).convert("RGB")',
    ),
    ("Mirror the picture horizontally.", "edited = ImageOps.mirror(image)"),
    ("Flip the picture upside down.", "edited = ImageOps.flip(image)"),
    ("Rotate the picture a quarter turn clockwise.", "edited = image.rotate(-90, expand=True)"),
    ("Make the picture brighter.", "edited = ImageEnhance.Brightness(image).enhance(1.4)"),
    ("Increase the contrast.", "edited = ImageEnhance.Contrast(image).enhance(1.5)"),
    ("Make the colors more vivid.", "edited = ImageEnhance.Color(image).enhance(1.8)"),
    ("Blur the picture.", "edited = image.filter(ImageFilter.GaussianBlur(radius=4))"),
    ("Sharpen the picture.", "edited = image.filter(ImageFilter.SHARPEN)"),
    ("Invert the colors.", "edited = ImageOps.invert(image)"),
    ("Reduce the colors to eight levels per channel.", "edited = ImageOps.posterize(image, 3)"),
    (
        "Crop to the middle half of the picture.",
        "width, height = image.size\n"
        "edited = image.crop((width // 4, height // 4, width - width // 4, height - height // 4))",
    ),
)


def _procedural(
    options: Sequence[tuple[str, tuple[str, ...]]],
    clip: Mapping[str, Any],
    original: Path,
    out: Path,
) -> dict[str, Any]:
    """Apply the one edit this clip id selects (deterministically) and return its clip, with
    the ffmpeg command that made it as the plan."""
    out.mkdir(parents=True, exist_ok=True)
    instruction, arguments = options[int(_sha(clip["id"]), 16) % len(options)]
    edited = out / f"{_stem(clip['id'])}.edited{original.suffix}"
    _ffmpeg("-i", str(original), *arguments, str(edited))
    command = " ".join(
        ["ffmpeg", "-i", f"input{original.suffix}", *(shlex.quote(a) for a in arguments)]
        + [f"output{original.suffix}"]
    )
    return {
        "id": clip["id"],
        "family": clip["family"],
        "original": original,
        "edited": edited,
        "instruction": instruction,
        "plan": f"```sh\n{command}\n```",
    }


def speech_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    return _procedural(SPEECH_EDITS, clip, Path(clip["audio"]), out)


def video_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    return _procedural(VIDEO_EDITS, clip, Path(clip["video"]), out)


def image_edit(clip: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """A deterministic edit of a picture by a short PIL program, which is also its plan."""
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps

    out.mkdir(parents=True, exist_ok=True)
    instruction, code = IMAGE_EDITS[int(_sha(clip["id"]), 16) % len(IMAGE_EDITS)]
    with Image.open(clip["image"]) as opened:
        namespace: dict[str, Any] = {
            "image": opened.convert("RGB"),
            "ImageEnhance": ImageEnhance,
            "ImageFilter": ImageFilter,
            "ImageOps": ImageOps,
        }
        exec(code, {"__builtins__": {}}, namespace)  # our own constant programs, no input is code
    edited = out / f"{_stem(clip['id'])}.edited.png"
    namespace["edited"].save(edited)
    return {
        "id": clip["id"],
        "family": clip["family"],
        "original": Path(clip["image"]),
        "edited": edited,
        "instruction": instruction,
        "plan": f"```python\n{code}\n```",
    }


@dataclass(frozen=True, slots=True)
class Task:
    """What a source's labels mean: how a record becomes a clip, and a clip becomes episodes."""

    streams: frozenset[str]
    episodes: Callable[
        [Preparer, OpenSource, Mapping[str, Any], Mapping[str, Any] | None], list[Episode]
    ]
    materialize: Callable[[OpenSource, Mapping[str, Any], Path], list[dict[str, Any]]] | None = None
    # Which procedural edit applies to this task's clips: "speech", "video" or "image".
    procedural: str | None = None
    # Whether the previous clip of the same family is passed as a voice reference.
    reference: bool = False

    @property
    def edit_stream(self) -> str | None:
        """The edit stream a procedural edit feeds (a picture's is only read as a plan)."""
        return {"speech": "speech_edit", "video": "av_edit"}.get(self.procedural or "")

    @property
    def plan_stream(self) -> str | None:
        """Every procedural edit is also an executed plan: the program that made it."""
        return "edit_plan" if self.procedural else None


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
        procedural="image",
    ),
    "picture": Task(
        frozenset({"image_identity"}), _plain(picture_episodes), picture_clips, procedural="image"
    ),
    "image_questions": Task(
        frozenset({"image_description", "image_identity"}),
        _plain(question_episodes),
        question_clips,
    ),
    "audio_questions": Task(
        frozenset({"audio_description", "audio_identity"}),
        _plain(audio_question_episodes),
        audio_question_clips,
    ),
    "captioned_video": Task(
        frozenset({"video_description", "video_identity"}),
        _plain(video_caption_episodes),
        video_caption_clips,
        procedural="video",
    ),
    "video_questions": Task(
        frozenset({"video_description", "video_identity"}),
        _plain(video_question_episodes),
        video_question_clips,
        procedural="video",
    ),
    "captioned_audio": Task(
        frozenset({"audio_description", "audio_identity"}),
        _plain(audio_caption_episodes),
        audio_caption_clips,
    ),
    "image_edit": Task(
        frozenset({"image_edit"}),
        lambda media, source, clip, reference: edit_episodes(
            media, source, "image", clip, provenance=_provenance(source, "media")
        ),
        image_edit_clips,
    ),
    "image_edit_chain": Task(
        frozenset({"image_edit"}),
        lambda media, source, clip, reference: chain_episodes(
            media, source, "image", clip, provenance=_provenance(source, "media")
        ),
        image_chain_clips,
    ),
    "edit_judging": Task(
        frozenset({"edit_judging"}),
        lambda media, source, clip, reference: judging_episodes(
            media, source, clip, tier=source.options.get("tier")
        ),
        edit_judging_clips,
    ),
    "image_edit_plan": Task(
        frozenset({"edit_plan"}),
        lambda media, source, clip, reference: plan_episodes(
            media, source, "image", clip, provenance=_provenance(source, "media")
        ),
        image_plan_clips,
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
    "video_edit_chain": Task(
        frozenset({"av_edit"}),
        lambda media, source, clip, reference: chain_episodes(
            media, source, "av", clip, provenance=_provenance(source, "media")
        ),
        video_chain_clips,
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
    "linked": linked_records,
    "table": table_records,
    "folders": folders_records,
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


def sessions(source: OpenSource, records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Rows that are the turns of one session -> one record whose fields are the turns' lists.

    ``options.session`` is ``{"by": column, "order": column}``: consecutive rows with the same
    ``by`` value form a session (its column keeps that one value), its turns sorted by ``order``.
    (A session split across shards
    is read as two shorter sessions, each valid.)
    """
    by, order = source.options["session"]["by"], source.options["session"].get("order")
    turns: list[Mapping[str, Any]] = []

    def merged() -> dict[str, Any]:
        rows = sorted(turns, key=lambda row: field_of(row, order)) if order else turns
        fields = {name: [row.get(name) for row in rows] for name in rows[0]}
        return {**fields, by: rows[0][by], "__key__": str(rows[0][by])}

    for record in records:
        if turns and field_of(record, by) != field_of(turns[0], by):
            yield merged()
            turns = []
        turns.append(record)
    if turns:
        yield merged()


def _materialized(source: OpenSource, shards: Shards, out: Path) -> Iterator[dict[str, Any]]:
    task, unique, seen = TASKS[source.task], source.options.get("unique"), set()
    records = CONTAINERS[source.reader](source, shards)
    if "session" in source.options:
        records = sessions(source, records)
    if shards.stripe is not None:
        # Every n-th record of all the files: the share of a source too big to divide by file.
        index, count = shards.stripe
        records = (record for number, record in enumerate(records) if number % count == index)
    if "transform" in source.options:
        transform = ROW_TRANSFORMS[source.options["transform"]]
        records = (
            {"__key__": record.get("__key__"), **reshaped}
            for record in records
            if (reshaped := transform(source, record)) is not None
        )
    if "fetch" in source.options:
        # The table names its pictures by URL: this sandbox downloads them (see ``fetch.py``),
        # only for the rows it keeps, and counts every dead link.
        fetch = source.options["fetch"]
        fetcher = Fetcher(
            out.parent / ".fetch-cache",
            stats=FETCH_STATS.setdefault(source.source_id, Counter()),
            **fetch.get("limits", {}),
        )
        records = fetched_records(
            records, fetch["fields"], fetcher, workers=fetch.get("workers", 8)
        )
    for record in records:
        if unique:
            key = field_of(record, unique)
            if key in seen:
                continue
            seen.add(key)
        yield from task.materialize(source, record, out)


def discard(clip: Mapping[str, Any]) -> None:
    """Delete a clip's local media (it is prepared once its episodes exist)."""
    for value in clip.values():
        for path in value if isinstance(value, list) else [value]:
            if isinstance(path, Path):
                path.unlink(missing_ok=True)


def build_episodes(
    media: Preparer,
    source: OpenSource,
    clips: Iterable[Mapping[str, Any]],
    *,
    work: Path,
    procedural: int = 0,
    procedural_every: int = 0,
) -> Iterator[Episode]:
    """The episodes of each clip, then one procedural edit for each of the first ``procedural``
    and for every ``procedural_every``-th clip (chosen by clip id, so shards agree)."""
    task = TASKS[source.task]
    references: dict[str, Mapping[str, Any]] = {}
    for index, clip in enumerate(clips):
        reference = references.get(clip["family"]) if task.reference else None
        yield from task.episodes(media, source, clip, reference)
        if task.procedural and (
            index < procedural
            or (procedural_every and int(_sha(clip["id"]), 16) % procedural_every == 0)
        ):
            kind = {"speech": "speech", "video": "av", "image": "image"}[task.procedural]
            edit = {"speech": speech_edit, "video": video_edit, "image": image_edit}[
                task.procedural
            ](clip, work / "edits")
            if kind != "image":  # real picture edits are plentiful: only their plans are made
                yield from edit_episodes(
                    media, source, kind, edit, provenance="executed", tier="procedural"
                )
            # Speech edits are as many as the speech edit stream asks for; their plans (a filter
            # command) are only a sample, so that the plan stream is not mostly audio.
            if kind != "speech" or int(_sha(clip["id"], "plan"), 16) % SPEECH_PLAN_EVERY == 0:
                plan = {
                    "id": edit["id"],
                    "family": edit["family"],
                    "frames": [edit["original"], edit["edited"]],
                    "plans": [edit["plan"]],
                    "instruction": edit["instruction"],
                }
                yield from plan_episodes(
                    media, source, kind, plan, provenance="executed", tier="procedural"
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
    procedural_every: int = 0,
    inputs: Path | None = None,
    shard_slice: tuple[int, int] = (0, 1),
    files: Sequence[str] | None = None,
    stripe: tuple[int, int] | None = None,
    sizes: Mapping[str, int] | None = None,
) -> Iterator[Episode]:
    """Every episode of ``count`` clips of one approved source (``files``, ``stripe`` and
    their stored ``sizes``: the part of it that one shard of ``data/media_sharding.py`` reads)."""
    source.terms.require_approval()
    shards = Shards(source, inputs, shard_slice, files, stripe, sizes, cache=work / ".web-cache")
    out = work / source.source_id
    yield from build_episodes(
        media,
        source,
        read_clips(source, shards, out, count),
        work=out,
        procedural=procedural,
        procedural_every=procedural_every,
    )
