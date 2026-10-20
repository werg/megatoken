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
    track_id: str = ""
    observed_role: str = "user"
    annotation_type: str = ""
    def __post_init__(self):
        if not self.artifact_id or self.modality not in MODALITIES or not isinstance(self.payload, bytes):
            raise ValueError("Invalid aligned artifact")
        if not 0 <= self.clock_start <= self.clock_end:
            raise ValueError("Invalid artifact clock")
        if self.observed_role not in {"user", "tool", "environment", "document", "system"}:
            raise ValueError("Observed artifact has an invalid causal role")
        if self.annotation_type not in {
            "", "written_text", "spoken_form", "verbatim_transcript",
            "display_caption", "audio_event_description", "translation_text",
            "content_plan",
        }:
            raise ValueError("Artifact annotation meaning must be explicit")

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
    content_plan: str | None = None
    event_revisions: Mapping[str, int] = field(default_factory=dict)
    reference_event_ids: tuple[str, ...] = ()
    def __post_init__(self):
        if self.family not in FAMILIES or not self.outputs or set(self.inputs) & set(self.outputs):
            raise ValueError("Invalid causal task partitions")
        if len(set(self.outputs)) != len(self.outputs) or len(set(self.inputs)) != len(self.inputs):
            raise ValueError("Repeated task artifact")
        names = (*self.inputs, *((self.content_plan,) if self.content_plan else ()),
                 *self.outputs)
        if self.content_plan and self.content_plan in (*self.inputs, *self.outputs):
            raise ValueError("The committed content plan must be separately generated")
        if any(not self.requests.get(x) for x in names):
            raise ValueError("Every task track needs an interpretation or output prompt")
        if any(ref not in self.inputs for ref in self.reference_event_ids):
            raise ValueError(
                "Voice/style references must be available observed inputs, never peer targets"
            )
        if self.revision < 0 or any(
            track not in names or type(revision) is not int or revision < 0
            for track, revision in self.event_revisions.items()
        ):
            raise ValueError("Invalid event request revision")

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
    from megatoken.data.schema import Episode, TrackSpec, TrackEvent, Meridian, Lineage, EpisodeRelation, canonical_json
    # No direct dependency on tokenization: records contain bytes and are encoded
    # later by the repository's normal episode readers.
    by_id = {a.artifact_id: a for a in bundle.artifacts}
    results = []
    for view in views:
        names = (*view.inputs, *((view.content_plan,) if view.content_plan else ()),
                 *view.outputs)
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
                role=a.observed_role if index < len(view.inputs) else "assistant",
                payload=a.payload, turn_index=(
                    0 if index < len(view.inputs)
                    else 1 if view.content_plan and name == view.content_plan
                    else 2 if view.content_plan else 1
                ),
                logical_id=name, part_index=0, part_count=1,
                policy_mask=index >= len(view.inputs),
                media_type=a.media_type,
            )
            records.append(record)
            tracks.append(TrackSpec(name, record.role, default_modality=a.modality, description=view.requests[name]))
        partitions = []
        if view.inputs:
            partitions.append(range(len(view.inputs)))
        start = len(view.inputs)
        if view.content_plan:
            partitions.append(range(start, start + 1))
            start += 1
        partitions.append(range(start, len(names)))
        for group_id, indices in enumerate(partitions):
            members = tuple(TrackEvent(
                records[i].record_id, names[i], by_id[names[i]].modality, group_id,
                records[i].role, request=view.requests[names[i]],
                request_revision=view.event_revisions.get(names[i], view.revision),
            ) for i in indices)
            meridians.append(Meridian(group_id, members))
        episode = Episode(
            episode_id=sha256((bundle.bundle_id + ":" + view.family + ":" + view.task_version + ":" + ",".join(names)).encode()).hexdigest()[:32],
            lineage=Lineage(source_id=bundle.bundle_id, revision=bundle.compiler_version, family_id=bundle.source_family, frozen_split=bundle.split),
            stream={
                "transcription": "transcription_and_verbalization",
                "verbalization": "transcription_and_verbalization",
                "speech_translation": "speech_translation",
                "image_description": "image_description_and_creation",
                "image_creation": "image_description_and_creation",
                "av_creation": "audiovisual_and_endpoints",
                "endpoint_creation": "audiovisual_and_endpoints",
                "speech_edit": "sequential_editing",
                "image_edit": "sequential_editing",
                "video_edit": "sequential_editing",
            }.get(view.family,view.family),
            records=tuple(records),
            tracks=tuple(tracks), meridians=tuple(meridians),
            omni_relations=tuple(
                EpisodeRelation(r.kind, r.left, r.right, r.weight)
                for r in view.relations
            ),
        )
        fingerprint = sha256(canonical_json(episode.to_dict()).encode()).hexdigest()
        results.append(CompiledView(episode, bundle.bundle_id, bundle.source_family,
                                    bundle.split, view.family, fingerprint, view.relations))
    return tuple(results)

def compile_edit_trajectory(
    bundle: AlignedBundle,
    *, base_artifact_id: str,
    steps: Sequence[tuple[Edit, str, tuple[str, ...]]],
    request_revision: int = 0,
) -> CompiledView:
    """Compile a *genuine ordered* recorded edit chain into normal meridians.

    Each step = (versioned Edit, exact edited artifact ID, related-output IDs).
    A later edit's base must be the actual prior target's version. If that
    target lacks a grouped native tape, the recurrent learner explicitly uses
    detached E(recorded) rather than claiming the sampled output was edited.
    """
    from megatoken.data.schema import (
        Episode, Lineage, PhysicalRecord, TrackSpec, TrackEvent, Meridian,
        canonical_json,
    )
    by_id={a.artifact_id:a for a in bundle.artifacts}
    if base_artifact_id not in by_id or not steps:
        raise ValueError("Edit trajectory needs a valid base and at least one edit")
    base=by_id[base_artifact_id]
    records=[]
    meridians=[]
    tracks={}
    def add_group(entries, generated=False):
        gid=len(meridians)
        events=[]
        for idx,(artifact, track, instruction, is_output, old_version) in enumerate(entries):
            if track not in tracks:
                tracks[track]=TrackSpec(track, "assistant" if is_output else "user",
                                        default_modality=artifact.modality,
                                        description=instruction)
            rid=sha256((bundle.bundle_id+":"+str(gid)+":"+artifact.artifact_id+
                         ":"+str(idx)).encode()).hexdigest()
            record=PhysicalRecord(
                record_id=rid, logical_id=rid,
                payload=artifact.payload, media_type=artifact.media_type,
                role="assistant" if is_output else "user",
                turn_index=gid, policy_mask=is_output,
                artifact_version=artifact.version if artifact.modality != "text" else "",
                base_artifact_version=old_version if is_output else "",
                edit_continuation="recorded" if old_version else "",
            )
            records.append(record)
            events.append(TrackEvent(rid,track,artifact.modality,gid,record.role,
                                      request=instruction,request_revision=request_revision+gid))
        if len({event.track_id for event in events})!=len(events):
            raise ValueError("Edit peers require distinct semantic tracks")
        meridians.append(Meridian(gid,tuple(events)))
    add_group(((base,base.track_id or base.modality,
                "Inspect the available base artifact.",False,""),))
    previous=base.version
    for edit,output_id,related in steps:
        if edit not in bundle.edits or edit.base_version != previous:
            raise ValueError("Edit chain does not follow the actual previous version")
        if output_id not in by_id or by_id[output_id].version != edit.result_version:
            raise ValueError("Edited target version does not match the requested edit")
        instruction=Artifact(
            artifact_id=sha256((output_id+":"+edit.change).encode()).hexdigest(),
            modality="text",payload=edit.change.encode(),version="instruction",
        )
        add_group(((instruction,"edit_request",edit.change,False,""),))
        result = by_id[output_id]
        targets=[(result,result.track_id or result.modality,
                  "Apply the requested change; preserve: "+", ".join(edit.preserve),
                  True,previous)]
        for slot, peer in enumerate(related):
            if peer not in by_id or by_id[peer].version != edit.result_version:
                raise ValueError("Related edit targets need the same result version")
            item = by_id[peer]
            # Derived artifact IDs usually include a new version suffix, but
            # physical track identity must remain bounded across the edit chain.
            stable_track = item.track_id or f"related_{slot}_{item.modality}"
            targets.append((item,stable_track,
                            "Update this aligned output for the recorded edit",True,previous))
        add_group(targets,True)
        previous=edit.result_version
    episode=Episode(
        episode_id=sha256((bundle.bundle_id+":edits:"+base_artifact_id).encode()).hexdigest(),
        lineage=Lineage(source_id=bundle.bundle_id,revision=bundle.compiler_version,
                        family_id=bundle.source_family,frozen_split=bundle.split),
        stream="sequential_editing",records=tuple(records),
        tracks=tuple(tracks.values()),meridians=tuple(meridians),
    )
    fingerprint=sha256(canonical_json(episode.to_dict()).encode()).hexdigest()
    return CompiledView(episode,bundle.bundle_id,bundle.source_family,bundle.split,
                        "sequential_editing",fingerprint,())
