"""Deterministic aligned-bundle -> ordinary causal Episode/meridian compiler.

A bundle is immutable storage and supervision; neither its hidden annotations
nor target payloads are prompts. Compiled views use the existing learner.
No modality is invented when an approved asset/annotation is missing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from megatoken.data.media_assets import (
    DESCRIPTOR_MODALITY,
    FEATURE_MODALITIES,
    MEDIA_RECORD_TYPE,
    MediaRecord,
)
from megatoken.data.schema import (
    Dependency,
    Episode,
    Lineage,
    MediaSpan,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackSpec,
    canonical_json,
    stable_id,
)

COMPILER_VERSION = "concordance-v3"
RELATION_KINDS = frozenset(
    {
        "same_spoken_content",
        "translation_of",
        "describes",
        "synchronized_with",
        "first_frame_of",
        "last_frame_of",
        "edit_of",
        "preserves",
    }
)
TASK_FAMILIES = frozenset(
    {
        "transcription",
        "verbalization",
        "speech_translation",
        "image_description",
        "image_creation",
        "audio_description",
        "video_description",
        "av_understanding",
        "av_creation",
        "endpoints",
        "endpoint_conditioned_video",
        "speech_counterfactual",
        "image_edit",
        "speech_edit",
        "av_edit",
        "edit_judging",
        "edit_plan",
        "cross_track_evidence",
        "timed_conversation",
        "streaming_translation",
    }
)


@dataclass(frozen=True, slots=True)
class BundleArtifact:
    """One versioned source or derived target, with an explicit semantic role.

    A text artifact's bytes are distinguishable from its labels; media stores
    only immutable references to original/frozen features and codec targets.
    A version parent must be present in this same source family.
    """

    artifact_id: str
    label: str
    modality: str
    payload: bytes = b""
    media: MediaRecord | None = None
    version: str = "v0"
    parent_id: str | None = None
    source_end_ms: int | None = None
    verification: str = "unverified"
    provenance: str = "natural"
    media_span: MediaSpan | None = None
    available_at_ns: int | None = None
    speaker: str = ""
    transform_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(json.loads(self.transform_json), dict):
            raise ValueError("Media transformation provenance must be a JSON object")
        if self.available_at_ns is not None and (
            type(self.available_at_ns) is not int or self.available_at_ns < 0
        ):
            raise ValueError("Artifact availability must be an external nonnegative time")
        if not self.artifact_id or not self.label or not self.version:
            raise ValueError("Bundle artifact needs stable ID, label and version")
        if self.modality not in {"text", *FEATURE_MODALITIES}:
            raise ValueError("Unsupported bundle artifact modality")
        if self.modality in FEATURE_MODALITIES:
            if (
                self.media is None
                or self.media.modality != DESCRIPTOR_MODALITY[self.modality]
                or self.payload
            ):
                raise ValueError("Feature bundle artifact must have a matching MediaRecord")
        elif self.media is not None or not isinstance(self.payload, bytes):
            raise ValueError("Text artifacts must carry raw bytes only")
        if self.source_end_ms is not None and self.source_end_ms < 0:
            raise ValueError("Source cutoff cannot be negative")
        if self.provenance not in {"natural", "teacher", "student", "executed", "simulated"}:
            raise ValueError("Unknown artifact provenance")

    def record_bytes(self, *, generated: bool) -> tuple[bytes, str]:
        if self.media is None:
            return self.payload, "text/plain; charset=utf-8"
        if generated and self.media.target is None:
            raise ValueError(f"Generated media artifact {self.artifact_id} lacks native target")
        # An observed input keeps its own codec latents: they are the input
        # itself, never a future answer, and the encode-and-reconstruct
        # objective trains on inputs too. Training reads only their source.
        return self.media.to_bytes(), MEDIA_RECORD_TYPE


@dataclass(frozen=True, slots=True)
class AlignedBundle:
    bundle_id: str
    lineage: Lineage
    artifacts: tuple[BundleArtifact, ...]
    source_license: str
    preprocessing_id: str
    verification_tier: str
    teacher_id: str = ""
    codec_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.bundle_id or not self.source_license or not self.preprocessing_id:
            raise ValueError("Bundle requires identity, terms and preprocessing revision")
        if self.lineage.frozen_split not in {"train", "validation", "test", "quarantine"}:
            raise ValueError("Bundles must freeze a family split before task compilation")
        by_id = {a.artifact_id: a for a in self.artifacts}
        if len(by_id) != len(self.artifacts) or not self.artifacts:
            raise ValueError("Bundle artifact identities must be nonempty and unique")
        for artifact in self.artifacts:
            if artifact.parent_id:
                parent = by_id.get(artifact.parent_id)
                if parent is None or parent.modality != artifact.modality:
                    raise ValueError("Edit version has no matching causal parent artifact")
                if parent.artifact_id == artifact.artifact_id:
                    raise ValueError("Artifact cannot parent itself")
                history = {artifact.artifact_id}
                while parent is not None:
                    if parent.artifact_id in history:
                        raise ValueError("Cyclic artifact version ancestry")
                    history.add(parent.artifact_id)
                    parent = by_id.get(parent.parent_id) if parent.parent_id else None

    def by_id(self, key: str) -> BundleArtifact:
        for artifact in self.artifacts:
            if artifact.artifact_id == key:
                return artifact
        raise ValueError(f"Unlisted bundle artifact: {key}")

    def labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        for a in self.artifacts:
            if a.label in labels:
                raise ValueError("Catalogue requires unique semantic labels; use explicit TaskView")
            labels[a.label] = a.artifact_id
        return labels


@dataclass(frozen=True, slots=True)
class EventRequest:
    """An observation or a generated event with one effective prompt revision."""

    artifact_id: str
    track_id: str
    instruction: str
    generated: bool
    revision: int = 0
    settings_json: str = "{}"
    reference_artifact_ids: tuple[str, ...] = ()
    alignment_id: str | None = None
    available_until_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.artifact_id or not self.track_id or not self.instruction.strip():
            raise ValueError("Each event needs a real task instruction, not just a track name")
        if self.revision < 0 or not isinstance(json.loads(self.settings_json), dict):
            raise ValueError("Invalid request revision or settings")
        if len(set(self.reference_artifact_ids)) != len(self.reference_artifact_ids):
            raise ValueError("Duplicate reference artifacts")


@dataclass(frozen=True, slots=True)
class TaskGroup:
    events: tuple[EventRequest, ...]
    selected_at_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.events or len({e.artifact_id for e in self.events}) != len(self.events):
            raise ValueError("Task group must have unique event artifacts")
        if len({e.generated for e in self.events}) != 1:
            raise ValueError("Observed and generated events need separate causal meridians")


@dataclass(frozen=True, slots=True)
class Relation:
    kind: str
    left: str
    right: str
    annotation_json: str = "{}"

    def __post_init__(self) -> None:
        if self.kind not in RELATION_KINDS or self.left == self.right:
            raise ValueError("Unknown/self relation")
        if not isinstance(json.loads(self.annotation_json), dict):
            raise ValueError("Relation supervision must be a JSON object")


@dataclass(frozen=True, slots=True)
class TaskView:
    view_id: str
    family: str
    groups: tuple[TaskGroup, ...]
    relations: tuple[Relation, ...] = ()
    shared_instruction: str = ""
    verification_json: str = "{}"
    compiler_version: str = COMPILER_VERSION

    def __post_init__(self) -> None:
        if self.family not in TASK_FAMILIES or not self.view_id or not self.groups:
            raise ValueError("Task requires a known family, ID and meridians")
        if self.compiler_version != COMPILER_VERSION:
            raise ValueError("Unknown task compiler version")
        if not isinstance(json.loads(self.verification_json), dict):
            raise ValueError("Task verification must be an object")


def observed(artifact: str, track: str, instruction: str, **kw) -> EventRequest:
    return EventRequest(artifact, track, instruction, False, **kw)


def generated(artifact: str, track: str, instruction: str, **kw) -> EventRequest:
    return EventRequest(artifact, track, instruction, True, **kw)


def group(*events: EventRequest) -> TaskGroup:
    return TaskGroup(tuple(events))


def compile_task(bundle: AlignedBundle, view: TaskView) -> Episode:
    """Compile literal available-input/target partitions without content leakage.

    A target record contains supervision bytes for E/R but P sees only prior
    observed/group-selected raw history and per-event requests. Reference and
    available-prefix checks operate on already committed event IDs.
    """
    seen_artifacts: dict[str, str] = {}
    records = []
    groups = []
    tracks: dict[str, TrackSpec] = {}
    output_records: dict[str, str] = {}
    deps = []
    latest_revisions: dict[str, int] = {}
    for index, item in enumerate(view.groups):
        entries = []
        current_generated = item.events[0].generated
        for event in item.events:
            artifact = bundle.by_id(event.artifact_id)
            if event.artifact_id in seen_artifacts:
                raise ValueError("One artifact cannot be selected twice within a task view")
            if event.revision < latest_revisions.get(event.track_id, 0):
                raise ValueError("A later edit request cannot have an earlier revision")
            latest_revisions[event.track_id] = event.revision
            if event.available_until_ms is not None:
                if event.generated or artifact.media is None:
                    raise ValueError("Source-prefix cutoff applies only to observed media")
                if (
                    artifact.source_end_ms is None
                    or artifact.source_end_ms > event.available_until_ms
                ):
                    raise ValueError(
                        "Frozen encoder view extends beyond the available source prefix"
                    )
            missing = set(event.reference_artifact_ids) - set(seen_artifacts)
            if missing:
                raise ValueError("Future/same-meridian references cannot be model inputs")
            if artifact.parent_id and event.generated:
                # Editing demands the actual preceding artifact version, not a
                # sibling with the same family ID.
                if artifact.parent_id not in seen_artifacts:
                    raise ValueError("Edited target's exact base version must precede the edit")
            record_id = stable_id(
                "bundle-record",
                bundle.bundle_id,
                view.view_id,
                str(index),
                event.track_id,
                event.artifact_id,
            )[:32]
            payload, media_type = artifact.record_bytes(generated=current_generated)
            record = PhysicalRecord(
                record_id,
                "assistant" if current_generated else "user",
                payload,
                turn_index=index,
                policy_mask=current_generated,
                provenance=artifact.provenance,
                media_type=media_type,
            )
            records.append(record)
            if event.track_id not in tracks:
                tracks[event.track_id] = TrackSpec(
                    event.track_id,
                    record.role,
                    artifact.modality,
                    description=event.instruction,
                )
            entries.append(
                TrackEvent(
                    record_id,
                    event.track_id,
                    artifact.modality,
                    index,
                    record.role,
                    alignment_id=event.alignment_id,
                    media_span=artifact.media_span,
                    speaker=artifact.speaker,
                    direction="outgoing" if current_generated else "incoming",
                    available_at_ns=None if current_generated else artifact.available_at_ns,
                    source_revision=artifact.version,
                    asset_sha256=artifact.media.source.sha256 if artifact.media else None,
                    instruction=event.instruction,
                    request_revision=event.revision,
                    settings_json=event.settings_json,
                    reference_event_ids=tuple(
                        seen_artifacts[key] for key in event.reference_artifact_ids
                    ),
                )
            )
            if current_generated:
                output_records[event.artifact_id] = record_id
            if current_generated and artifact.parent_id:
                deps.append(Dependency(seen_artifacts[artifact.parent_id], record_id, "edit"))
        groups.append(Meridian(index, tuple(entries), item.selected_at_ns))
        seen_artifacts.update(
            {e.artifact_id: event.record_id for e, event in zip(item.events, entries, strict=True)}
        )
    if not any(e.generated for g in view.groups for e in g.events):
        raise ValueError("A training task must contain supervised output events")
    relation_rows = []
    for relation in view.relations:
        if relation.left not in seen_artifacts or relation.right not in seen_artifacts:
            raise ValueError("A relation cannot name withheld or absent artifacts")
        if relation.left not in output_records and relation.right not in output_records:
            raise ValueError("A supervised relation must involve a generated output")
        relation_rows.append(
            {
                "kind": relation.kind,
                "left": seen_artifacts[relation.left],
                "right": seen_artifacts[relation.right],
                "annotation": json.loads(relation.annotation_json),
            }
        )
    annotations = {
        "compiler_version": view.compiler_version,
        "source_family": bundle.lineage.family_id,
        "source_bundle": bundle.bundle_id,
        "source_license": bundle.source_license,
        "preprocessing_id": bundle.preprocessing_id,
        "teacher_id": bundle.teacher_id,
        "codec_ids": list(bundle.codec_ids),
        "task_family": view.family,
        "task_view": view.view_id,
        "relations": relation_rows,
        "available_artifacts": [
            e.artifact_id for g in view.groups for e in g.events if not e.generated
        ],
        "target_artifacts": [e.artifact_id for g in view.groups for e in g.events if e.generated],
        "verification": json.loads(view.verification_json),
        "media_provenance": {a.artifact_id: json.loads(a.transform_json) for a in bundle.artifacts},
    }
    lineage = Lineage(
        source_id=bundle.lineage.source_id,
        revision=bundle.lineage.revision,
        family_id=bundle.lineage.family_id,
        ancestor_ids=tuple(
            sorted(
                set(
                    (
                        *bundle.lineage.ancestor_ids,
                        bundle.bundle_id,
                    )
                )
            )
        ),
        repository=bundle.lineage.repository,
        # Unique to this bundle's task: a task ID shared by every bundle would put all of them
        # in one split family (Lineage.family_keys), quarantining mixed frozen splits.
        task_id=f"{bundle.bundle_id}:{view.family}:{view.view_id}",
        parent_episode_id=bundle.lineage.parent_episode_id,
        source_uri=bundle.lineage.source_uri,
        created_at=bundle.lineage.created_at,
        source_row=bundle.lineage.source_row,
        raw_sha256=bundle.lineage.raw_sha256,
        frozen_split=bundle.lineage.frozen_split,
    )
    return Episode(
        episode_id=stable_id("compiled-episode", bundle.bundle_id, view.view_id, COMPILER_VERSION),
        records=tuple(records),
        lineage=lineage,
        stream=view.family,
        supervision=Supervision(
            verification=bundle.verification_tier,
            annotations_json=canonical_json(annotations),
        ),
        dependencies=tuple(deps),
        tracks=tuple(tracks.values()),
        meridians=tuple(groups),
        control_seed=view.shared_instruction.encode("utf-8"),
    )


def standard_views(bundle: AlignedBundle) -> tuple[TaskView, ...]:
    """Emit only semantically supported directions. No labels are inferred.

    Known labels are declared by the data adapter, not guessed from filenames.
    Extra pairings and actual edit chains use explicit TaskView declarations.
    """
    labels = bundle.labels()
    result: list[TaskView] = []

    def has(*roles: str) -> bool:
        return all(role in labels for role in roles)

    def add(
        family: str, name: str, events: tuple[TaskGroup, ...], relations: tuple[Relation, ...] = ()
    ) -> None:
        result.append(TaskView(name, family, events, relations))

    if has("source_speech", "verbatim_transcript"):
        add(
            "transcription",
            "direct-asr",
            (
                group(
                    observed(
                        labels["source_speech"],
                        "source_speech",
                        "Listen to the available source speech; transcribe it verbatim.",
                    )
                ),
                group(
                    generated(
                        labels["verbatim_transcript"],
                        "source_transcript",
                        "Transcribe exactly what was spoken, preserving names and numbers.",
                    )
                ),
            ),
            (
                Relation(
                    "same_spoken_content", labels["source_speech"], labels["verbatim_transcript"]
                ),
            ),
        )
    if has("written_text", "speech"):
        peers = [
            generated(
                labels["speech"],
                "speech",
                "Speak the given text using the requested language and voice.",
            )
        ]
        if has("spoken_form"):
            peers.insert(
                0,
                generated(
                    labels["spoken_form"],
                    "spoken_form",
                    "Render the normalized spoken wording of the supplied text.",
                ),
            )
        inputs = [
            observed(
                labels["written_text"],
                "written_text",
                "Read this text exactly, observing spoken normalization.",
            )
        ]
        if has("voice_reference"):
            inputs.append(
                observed(
                    labels["voice_reference"],
                    "voice_reference",
                    "Use this available speaker sample for voice identity.",
                )
            )
        add(
            "verbalization",
            "joint-tts",
            (group(*inputs), group(*peers)),
            (Relation("same_spoken_content", labels["written_text"], labels["speech"]),),
        )
    if has("source_speech", "translated_text"):
        # The translated speech is optional: many corpora give only the target-language text.
        peers = [
            generated(
                labels["translated_text"],
                "translation",
                "Translate the source speech accurately into the requested language.",
            )
        ]
        relations = [Relation("translation_of", labels["translated_text"], labels["source_speech"])]
        if has("translated_speech"):
            peers.append(
                generated(
                    labels["translated_speech"],
                    "translated_speech",
                    "Speak the translation; preserve names and numbers without reading peer text.",
                )
            )
            relations.append(
                Relation(
                    "same_spoken_content", labels["translated_text"], labels["translated_speech"]
                )
            )
        if has("verbatim_transcript"):
            peers.insert(
                0,
                generated(
                    labels["verbatim_transcript"],
                    "source_transcript",
                    "Transcribe the available source speech verbatim.",
                ),
            )
        add(
            "speech_translation",
            "direct-joint-translation" if has("translated_speech") else "direct-translation",
            (
                group(
                    observed(
                        labels["source_speech"],
                        "source_speech",
                        "Listen to the source-language speech; only the supplied prefix is available.",
                    )
                ),
                group(*peers),
            ),
            tuple(relations),
        )
        if has("translated_speech"):
            add(
                "speech_translation",
                "text-then-speech-translation",
                (
                    group(
                        observed(
                            labels["source_speech"],
                            "source_speech",
                            "Listen to the available source-language speech.",
                        )
                    ),
                    group(
                        generated(
                            labels["translated_text"],
                            "translation",
                            "Translate the available source into the requested language.",
                        )
                    ),
                    group(
                        generated(
                            labels["translated_speech"],
                            "translation",
                            "Speak the translation that was committed in the preceding step.",
                            reference_artifact_ids=(labels["translated_text"],),
                        )
                    ),
                ),
                tuple(relations),
            )
    if has("image", "display_caption"):
        add(
            "image_description",
            "grounded-image-description",
            (
                group(observed(labels["image"], "image", "Inspect the supplied visual content.")),
                group(
                    generated(
                        labels["display_caption"],
                        "caption",
                        "Describe visible objects, attributes, counts and spatial relations.",
                    )
                ),
            ),
            (Relation("describes", labels["display_caption"], labels["image"]),),
        )
    if has("audio", "audio_caption"):
        add(
            "audio_description",
            "grounded-audio-description",
            (
                group(observed(labels["audio"], "audio", "Listen to the supplied recording.")),
                group(
                    generated(
                        labels["audio_caption"],
                        "caption",
                        "Describe the audible sources, events, music and their order.",
                    )
                ),
            ),
            (Relation("describes", labels["audio_caption"], labels["audio"]),),
        )
    if has("video", "video_caption"):
        add(
            "video_description",
            "grounded-video-description",
            (
                group(observed(labels["video"], "video", "Watch the supplied clip.")),
                group(
                    generated(
                        labels["video_caption"],
                        "caption",
                        "Describe the visible subjects, motion and their order.",
                    )
                ),
            ),
            (Relation("describes", labels["video_caption"], labels["video"]),),
        )
    if has("creation_request", "image", "display_caption"):
        add(
            "image_creation",
            "joint-image-caption",
            (
                group(
                    observed(
                        labels["creation_request"],
                        "request",
                        "Choose image content from the creation request.",
                    )
                ),
                group(
                    generated(labels["image"], "image_out", "Generate the requested image."),
                    generated(
                        labels["display_caption"],
                        "caption",
                        "Describe the selected generated scene, not a hidden gold image.",
                    ),
                ),
            ),
            (Relation("describes", labels["display_caption"], labels["image"]),),
        )
    if has("video", "source_audio", "display_caption"):
        outputs = [
            generated(
                labels["display_caption"],
                "caption",
                "Describe visible actions and their actual order.",
            )
        ]
        if has("verbatim_transcript"):
            outputs.append(
                generated(
                    labels["verbatim_transcript"],
                    "transcript",
                    "Transcribe spoken words, not visual scene descriptions.",
                )
            )
        if has("audio_event_description"):
            outputs.append(
                generated(
                    labels["audio_event_description"],
                    "audio_events",
                    "Describe audible sound events, not subtitles.",
                )
            )
        add(
            "av_understanding",
            "av-understanding",
            (
                group(
                    observed(labels["video"], "source_video", "Observe the available video."),
                    observed(
                        labels["source_audio"],
                        "source_audio",
                        "Listen to the synchronized source sound.",
                    ),
                ),
                group(*outputs),
            ),
            (Relation("describes", labels["display_caption"], labels["video"]),),
        )
    if has("creation_request", "video", "soundtrack", "display_caption"):
        prefix = [
            group(
                observed(
                    labels["creation_request"],
                    "request",
                    "Create synchronized audiovisual content.",
                )
            )
        ]
        if has("content_plan"):
            prefix.append(
                group(
                    generated(
                        labels["content_plan"],
                        "content_plan",
                        "Choose a coherent scene, timing, speakers and events.",
                    )
                )
            )
        peers = [
            generated(
                labels["display_caption"], "caption", "Describe the chosen scene and event order."
            ),
            generated(labels["video"], "video_out", "Realize the selected scene and motion."),
            generated(labels["soundtrack"], "soundtrack", "Produce synchronized sound events."),
        ]
        relations = [
            Relation("describes", labels["display_caption"], labels["video"]),
            Relation("synchronized_with", labels["soundtrack"], labels["video"]),
        ]
        if has("first_frame"):
            peers.append(
                generated(
                    labels["first_frame"],
                    "first_frame",
                    "Generate the independently predicted first visible frame.",
                )
            )
            relations.append(Relation("first_frame_of", labels["first_frame"], labels["video"]))
        if has("last_frame"):
            peers.append(
                generated(
                    labels["last_frame"],
                    "last_frame",
                    "Generate the independently predicted final visible frame.",
                )
            )
            relations.append(Relation("last_frame_of", labels["last_frame"], labels["video"]))
        add("av_creation", "joint-av", (*prefix, group(*peers)), tuple(relations))
    if has("first_frame", "last_frame", "video", "creation_request"):
        add(
            "endpoint_conditioned_video",
            "endpoints-input",
            (
                group(
                    observed(
                        labels["first_frame"],
                        "start_frame",
                        "Use this genuinely supplied starting frame.",
                    ),
                    observed(
                        labels["last_frame"],
                        "end_frame",
                        "Use this genuinely supplied ending frame.",
                    ),
                    observed(
                        labels["creation_request"], "request", "Interpolate the requested motion."
                    ),
                ),
                group(
                    generated(
                        labels["video"],
                        "video_out",
                        "Produce motion constrained by both observed endpoint frames.",
                    )
                ),
            ),
            (
                Relation("first_frame_of", labels["first_frame"], labels["video"]),
                Relation("last_frame_of", labels["last_frame"], labels["video"]),
            ),
        )
    if has(
        "written_text",
        "voice_reference_a",
        "voice_reference_b",
        "speech_variant_a",
        "speech_variant_b",
    ):
        add(
            "speech_counterfactual",
            "voice-ab-variants",
            (
                group(
                    observed(
                        labels["written_text"],
                        "written_text",
                        "Read the exact shared source sentence.",
                    ),
                    observed(
                        labels["voice_reference_a"],
                        "voice_ref_a",
                        "Available sample for voice identity A.",
                    ),
                    observed(
                        labels["voice_reference_b"],
                        "voice_ref_b",
                        "Available sample for voice identity B.",
                    ),
                ),
                group(
                    generated(
                        labels["speech_variant_a"],
                        "variant_a",
                        "Speak the text in voice A; no text or source-voice changes.",
                        revision=1,
                        reference_artifact_ids=(labels["voice_reference_a"],),
                    ),
                    generated(
                        labels["speech_variant_b"],
                        "variant_b",
                        "Speak the same text in voice B; only speaker identity changes.",
                        revision=1,
                        reference_artifact_ids=(labels["voice_reference_b"],),
                    ),
                ),
            ),
            (
                Relation(
                    "same_spoken_content", labels["speech_variant_a"], labels["speech_variant_b"]
                ),
                Relation("same_spoken_content", labels["written_text"], labels["speech_variant_a"]),
                Relation("same_spoken_content", labels["written_text"], labels["speech_variant_b"]),
            ),
        )
    for family, prefix, modality in (
        ("image_edit", "image", "image"),
        ("speech_edit", "speech", "audio"),
        ("av_edit", "av", "video"),
    ):
        base, target, instruction = (
            f"original_{prefix}",
            f"edited_{prefix}",
            f"{prefix}_edit_instruction",
        )
        if has(base, target, instruction):
            original = bundle.by_id(labels[base])
            edited = bundle.by_id(labels[target])
            if original.modality != modality or edited.modality != modality:
                raise ValueError(f"{family} labels have invalid modality")
            if edited.parent_id != original.artifact_id:
                raise ValueError(f"{family} target must name its exact input version")
            add(
                family,
                f"verified-{prefix}-edit",
                (
                    group(
                        observed(labels[base], "original", "Observe the original media version."),
                        observed(
                            labels[instruction],
                            "edit_instruction",
                            "Apply only the verified requested semantic change.",
                        ),
                    ),
                    group(
                        generated(
                            labels[target],
                            "edited",
                            "Render the edited version with unchanged regions preserved.",
                            revision=1,
                            reference_artifact_ids=(labels[base],),
                        )
                    ),
                ),
                (
                    Relation("edit_of", labels[target], labels[base]),
                    Relation("preserves", labels[target], labels[base]),
                ),
            )
    if has("audio_evidence", "video_evidence", "text_evidence", "grounded_answer"):
        add(
            "cross_track_evidence",
            "joint-evidence-grounding",
            (
                group(
                    observed(
                        labels["audio_evidence"],
                        "audio_evidence",
                        "Listen to the available auditory evidence.",
                    ),
                    observed(
                        labels["video_evidence"],
                        "video_evidence",
                        "Inspect the available visual evidence.",
                    ),
                    observed(
                        labels["text_evidence"],
                        "text_evidence",
                        "Read the available written evidence.",
                    ),
                ),
                group(
                    generated(
                        labels["grounded_answer"],
                        "grounded_answer",
                        "Answer using corroborated sources; do not hallucinate missing facts.",
                    )
                ),
            ),
            (
                Relation("describes", labels["grounded_answer"], labels["video_evidence"]),
                Relation("same_spoken_content", labels["grounded_answer"], labels["text_evidence"]),
            ),
        )
    return tuple(result)


def edit_task_view(
    *,
    family: str,
    view_id: str,
    original: str,
    instruction: str,
    edited: str,
    version: int = 1,
    optional_peers: tuple[tuple[str, str], ...] = (),
    preserves_json: str = "{}",
) -> TaskView:
    """Construct an explicit verified video/image/speech edit with peer updating.

    The actual original must be available in a prior meridian; all changed
    outputs are selected as peers from the same previous history and event
    request, not chained through an unavailable same-step target. Exact parent
    ancestry is enforced by compile_task.
    """
    if family not in {"image_edit", "speech_edit", "av_edit"} or version < 1:
        raise ValueError("Unknown editing task family/revision")
    output = [
        generated(
            edited,
            "edited_media",
            "Apply the requested edit, preserving all unaffected content.",
            revision=version,
            reference_artifact_ids=(original,),
        )
    ]
    relations = [
        Relation("edit_of", edited, original),
        Relation("preserves", edited, original, preserves_json),
    ]
    for identifier, modality in optional_peers:
        if modality not in {"text", "audio", "image", "video"}:
            raise ValueError("Unknown edited peer modality")
        output.append(
            generated(
                identifier,
                "edited_" + modality,
                "Update only this related output track to agree with the requested edit.",
                revision=version,
                reference_artifact_ids=(original,),
            )
        )
        relations.append(
            Relation("describes" if modality == "text" else "synchronized_with", identifier, edited)
        )
    return TaskView(
        view_id,
        family,
        (
            group(
                observed(
                    original, "original", "Observe this exact media version as the edit baseline."
                ),
                observed(instruction, "instruction", "Read the verified edit request."),
            ),
            group(*output),
        ),
        tuple(relations),
    )


def edit_chain_view(
    *,
    family: str,
    view_id: str,
    original: str,
    steps: Sequence[tuple[str, str]],
    preserves_json: str = "{}",
) -> TaskView:
    """A multi-turn editing session: the user asks, the model renders the edit, the user asks
    for a follow-up change to *that result*, and so on.

    ``steps`` is each turn's ``(instruction, edited)`` artifact IDs, in order. Every edited
    version names the previous result as its exact parent (the first names ``original``); the
    follow-up requests are observed after the result they refer to, so the model conditions on
    its own earlier edits. Every turn is one supervised output; a one-step chain is
    ``edit_task_view``.
    """
    if family not in {"image_edit", "speech_edit", "av_edit"} or not steps:
        raise ValueError("An edit chain needs an editing family and at least one step")
    first_instruction = steps[0][0]
    groups = [
        group(
            observed(
                original, "original", "Observe this exact media version as the edit baseline."
            ),
            observed(first_instruction, "instruction", "Read the verified edit request."),
        )
    ]
    relations: list[Relation] = []
    previous = original
    for turn, (instruction, edited) in enumerate(steps, 1):
        if turn > 1:
            groups.append(
                group(
                    observed(
                        instruction,
                        "instruction",
                        "Read the follow-up request; it changes the latest result.",
                    )
                )
            )
        groups.append(
            group(
                generated(
                    edited,
                    "edited_media",
                    "Apply the requested edit to the latest result, preserving all unaffected content.",
                    revision=turn,
                    reference_artifact_ids=(previous,),
                )
            )
        )
        relations += [
            Relation("edit_of", edited, previous),
            Relation("preserves", edited, previous, preserves_json),
        ]
        previous = edited
    return TaskView(view_id, family, tuple(groups), tuple(relations))


def edit_judging_view(
    *,
    view_id: str,
    original: str,
    instruction: str,
    candidates: Sequence[str],
    verdict: str,
) -> TaskView:
    """Judging candidate edits: the model is shown the original, the request and one or more
    candidate results (all observed, none generated) and writes the verdict (which is best, with
    the scores or critique that justify it). The verdict names every candidate it covers."""
    if not candidates:
        raise ValueError("Judging needs at least one candidate edit")
    groups = [
        group(
            observed(
                original, "original", "Observe this exact media version as the edit baseline."
            ),
            observed(instruction, "instruction", "Read the edit request that is being judged."),
            *(
                observed(candidate, f"candidate_{number}", f"Inspect candidate {chr(64 + number)}.")
                for number, candidate in enumerate(candidates, 1)
            ),
        ),
        group(
            generated(
                verdict,
                "verdict",
                "Judge how well each candidate carries out the request and which is best.",
                reference_artifact_ids=(original, *candidates),
            )
        ),
    ]
    relations = [Relation("describes", verdict, candidate) for candidate in candidates]
    return TaskView(view_id, "edit_judging", tuple(groups), tuple(relations))


def edit_plan_view(
    *,
    family: str,
    view_id: str,
    original: str,
    instruction: str,
    steps: Sequence[tuple[str, str]],
    preserves_json: str = "{}",
) -> TaskView:
    """Editing by planning: the model is shown the original and the request, writes a plan or
    tool call, then renders the edited version the plan produces; further steps (a trajectory)
    plan again from the latest result.

    ``steps`` is each step's ``(plan, edited)`` artifact IDs; the plan text is supervised before
    its result and every result names the previous one as its exact parent.
    """
    if family != "edit_plan" or not steps:
        raise ValueError("An edit plan needs the edit_plan family and at least one step")
    groups = [
        group(
            observed(
                original, "original", "Observe this exact media version as the edit baseline."
            ),
            observed(instruction, "instruction", "Read the verified edit request."),
        )
    ]
    relations: list[Relation] = []
    previous = original
    for turn, (plan, edited) in enumerate(steps, 1):
        groups.append(
            group(
                generated(
                    plan,
                    "plan",
                    "Write the plan or tool call that carries out the request on the latest result.",
                    revision=turn,
                    reference_artifact_ids=(previous,),
                )
            )
        )
        groups.append(
            group(
                generated(
                    edited,
                    "edited_media",
                    "Render the edit exactly as the plan describes, preserving unaffected content.",
                    revision=turn,
                    reference_artifact_ids=(previous, plan),
                )
            )
        )
        relations += [
            Relation("describes", plan, edited),
            Relation("edit_of", edited, previous),
            Relation("preserves", edited, previous, preserves_json),
        ]
        previous = edited
    return TaskView(view_id, family, tuple(groups), tuple(relations))


def question_chain_view(
    *,
    family: str,
    view_id: str,
    media: str,
    turns: Sequence[tuple[str, str]],
    modality: str = "image",
) -> TaskView:
    """A conversation about one observed recording: the user shows the ``media`` with a first
    question, the model answers, the user asks a follow-up, and so on.

    ``turns`` is each turn's ``(question, answer)`` text artifact IDs. Every answer is one
    supervised output that may use the media and every earlier turn.
    """
    if family not in {"image_description", "audio_description", "video_description"} or not turns:
        raise ValueError("A question chain needs a describing family and at least one turn")
    groups = [
        group(
            observed(media, modality, f"Inspect the supplied {modality}."),
            observed(turns[0][0], "question", "Read the question about it."),
        )
    ]
    relations = []
    for turn, (question, answer) in enumerate(turns, 1):
        if turn > 1:
            groups.append(group(observed(question, "question", "Read the follow-up question.")))
        groups.append(
            group(
                generated(
                    answer,
                    "answer",
                    f"Answer from what the {modality} shows and what was said before.",
                    revision=turn,
                    reference_artifact_ids=(media,),
                )
            )
        )
        relations.append(Relation("describes", answer, media))
    return TaskView(view_id, family, tuple(groups), tuple(relations))


def manifest_fingerprint(bundle: AlignedBundle, views: tuple[TaskView, ...]) -> str:
    """Stable manifest digest binds exact task family, source split and versions."""
    h = hashlib.sha256()

    def field(text: str) -> None:
        # Length-prefixed, so adjacent fields cannot trade characters across a boundary.
        data = text.encode()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)

    for text in (
        COMPILER_VERSION,
        bundle.bundle_id,
        bundle.lineage.family_id,
        bundle.lineage.frozen_split,
    ):
        field(text)
    for view in views:
        episode = compile_task(bundle, view)
        for text in (
            episode.episode_id,
            episode.content_sha256,
            episode.supervision.annotations_json,
        ):
            field(text)
    return h.hexdigest()
