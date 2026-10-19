"""Deterministic, leak-resistant views of immutable aligned source bundles.

The compiler is deliberately independent of model inference and dataset downloads.
It emits existing Episode/meridian objects through the public schema; verification
and withheld targets never appear in a prospective prompt.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Sequence

RELATIONS = frozenset(("same_spoken_content", "translation_of", "describes",
    "synchronized_with", "first_frame_of", "last_frame_of", "edit_of", "preserves"))
FAMILIES = frozenset(("transcription", "verbalization", "speech_translation",
    "image_description", "image_creation", "av_understanding", "av_creation",
    "endpoint_creation", "endpoint_conditioned", "speech_edit", "image_edit",
    "video_edit", "counterfactual", "cross_track", "replay"))
MODALITIES = frozenset(("text", "audio", "image", "video"))

@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    modality: str
    payload: bytes
    version: str = "v0"
    parent_version: str = ""
    media_type: str = "text/plain"
    clock_start: float = 0.0
    clock_end: float = 0.0
    reference_origin: str = ""
    def __post_init__(self):
        if not self.artifact_id or self.modality not in MODALITIES or not isinstance(self.payload, bytes):
            raise ValueError("Invalid aligned artifact")
        if not 0 <= self.clock_start <= self.clock_end:
            raise ValueError("Invalid artifact clock")

@dataclass(frozen=True)
class Relation:
    kind: str
    left: str
    right: str
    weight: float = 1.0
    def __post_init__(self):
        if self.kind not in RELATIONS or not self.left or not self.right or not 0 < self.weight <= 10:
            raise ValueError("Unsupported or invalid relation")

@dataclass(frozen=True)
class Edit:
    base_version: str
    result_version: str
    change: str
    preserve: tuple[str, ...] = ()
    def __post_init__(self):
        if not self.base_version or not self.result_version or not self.change:
            raise ValueError("Edits require explicit ancestry and instructions")
        if self.base_version == self.result_version:
            raise ValueError("Edit must create a new artifact version")

@dataclass(frozen=True)
class AlignedBundle:
    bundle_id: str
    source_family: str
    split: str
    artifacts: tuple[Artifact, ...]
    relations: tuple[Relation, ...] = ()
    edits: tuple[Edit, ...] = ()
    compiler_version: str = "omni-bundle-v1"
    def __post_init__(self):
        if not self.bundle_id or not self.source_family or self.split not in ("train", "validation", "test"):
            raise ValueError("Bundle needs identity, family and split")
        ids = [a.artifact_id for a in self.artifacts]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate artifact identity")
        if any(r.left not in ids or r.right not in ids for r in self.relations):
            raise ValueError("Relation references an absent artifact")
        versions = {a.version for a in self.artifacts}
        if any(e.base_version not in versions or e.result_version not in versions for e in self.edits):
            raise ValueError("Edit refers to an unknown artifact version")

@dataclass(frozen=True)
class TaskView:
    family: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    requests: Mapping[str, str]
    revision: int = 0
    relations: tuple[Relation, ...] = ()
    task_version: str = "v1"
    def __post_init__(self):
        if self.family not in FAMILIES or not self.outputs or set(self.inputs) & set(self.outputs):
            raise ValueError("Invalid causal task partitions")
        if len(set(self.outputs)) != len(self.outputs) or len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Repeated task artifact")
        if any(not self.requests.get(x) for x in (*self.inputs, *self.outputs)):
            raise ValueError("Every task track needs an interpretation or output prompt")
        if self.revision < 0:
            raise ValueError("Negative request revision")

@dataclass(frozen=True)
class CompiledView:
    episode: object
    bundle_id: str
    source_family: str
    split: str
    family: str
    fingerprint: str
    relations: tuple[Relation, ...]
    verification: Mapping[str, object] = field(default_factory=dict)

def compile_views(bundle: AlignedBundle, views: Sequence[TaskView]) -> tuple[CompiledView, ...]:
    """Compile existing physical records and grouped meridians.

    Only input bytes become observations. All generated siblings occupy one
    atomic meridian; relations/verification are deliberately side metadata.
    """
    from megatoken.data.schema import Episode, TrackSpec, TrackEvent, Meridian, Lineage, canonical_json
    # No direct dependency on tokenization: records contain bytes and are encoded
    # later by the repository's normal episode readers.
    by_id = {a.artifact_id: a for a in bundle.artifacts}
    results = []
    for view in views:
        names = (*view.inputs, *view.outputs)
        if any(name not in by_id for name in names):
            raise ValueError("Task references an absent bundle artifact")
        if any(r not in bundle.relations for r in view.relations):
            raise ValueError("Task relation is not certified by source bundle")
        records = []
        meridians = []
        tracks = []
        for index, name in enumerate(names):
            a = by_id[name]
            # Preserve distinct native source records and target records. Media
            # payloads must already be canonical pinned MediaRecord descriptors.
            from megatoken.data.schema import PhysicalRecord
            record = PhysicalRecord(
                record_id=sha256((bundle.bundle_id + ":" + view.family + ":" + name + ":" + str(index)).encode()).hexdigest()[:32],
                role="user" if index < len(view.inputs) else "assistant",
                payload=a.payload, turn_index=0 if index < len(view.inputs) else 1,
                logical_id=name, part_index=0, part_count=1,
                policy_mask=index >= len(view.inputs),
                media_type=a.media_type,
            )
            records.append(record)
            tracks.append(TrackSpec(name, record.role, default_modality=a.modality, description=view.requests[name]))
        for group_id, indices in enumerate((range(len(view.inputs)), range(len(view.inputs),len(names)))):
            if not indices:
                continue
            members = tuple(TrackEvent(
                records[i].record_id, names[i], by_id[names[i]].modality, group_id,
                records[i].role, request=view.requests[names[i]], request_revision=view.revision,
            ) for i in indices)
            meridians.append(Meridian(group_id, members))
        episode = Episode(
            episode_id=sha256((bundle.bundle_id + ":" + view.family + ":" + view.task_version + ":" + ",".join(names)).encode()).hexdigest()[:32],
            lineage=Lineage(source_id=bundle.bundle_id, revision=bundle.compiler_version, family_id=bundle.source_family, frozen_split=bundle.split), records=tuple(records),
            tracks=tuple(tracks), meridians=tuple(meridians),
        )
        fingerprint = sha256(canonical_json(episode.to_dict()).encode()).hexdigest()
        results.append(CompiledView(episode, bundle.bundle_id, bundle.source_family,
                                    bundle.split, view.family, fingerprint, view.relations))
    return tuple(results)
