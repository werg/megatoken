"""Sustained translation with executed wait/continue/finish supervision.

Original audio and publisher translations remain ordinary corresponding assets.
The scheduling policy is explicitly authored over their arrival annotations; it
is never labeled as naturally recorded assistant behavior. No later translation
or target-ready flag is included in the model's public observations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace

from megatoken.lifecycle import TrackRegistry, lifecycle_tools

from .media_assets import MEDIA_RECORD_TYPE
from .schema import (
    Dependency,
    Episode,
    EpisodeRelation,
    LogicalAction,
    Meridian,
    PhysicalRecord,
    Supervision,
    TrackEvent,
    TrackRevision,
    TrackSpec,
    canonical_json,
    digest_bytes,
    stable_id,
)


def validate_streaming_translation(row):
    options = row.get("streaming_translation")
    if options is None:
        return
    if not isinstance(options, dict) or options.get("policy") not in {
        "segment_end",
        "aligned_prefix",
    }:
        raise ValueError(
            "Streaming translation needs a declared segment_end or aligned_prefix policy"
        )
    language = options.get("language")
    if not isinstance(language, dict) or any(
        not isinstance(language.get(k), str) or not language[k] for k in ("source", "target")
    ):
        raise ValueError("Streaming translation needs public source and target languages")
    capabilities = options.get("output_modalities", ["text"])
    if (
        not isinstance(capabilities, list)
        or any(not isinstance(item, str) for item in capabilities)
        or "text" not in capabilities
        or len(set(capabilities)) != len(capabilities)
        or not set(capabilities) <= {"text", "audio"}
    ):
        raise ValueError(
            "Streaming output capabilities must explicitly include text and optionally audio"
        )
    budget = options.get("max_decisions", 4096)
    if type(budget) is not int or budget < 1:
        raise ValueError("Streaming work needs a positive public decision budget")
    order = sorted(
        row["chunks"],
        key=lambda c: c.get(
            "available_at_ns", c["end_sample"] * 1_000_000_000 // row["sample_rate"]
        ),
    )
    if len({(c.get("speaker", c.get("channel", 0)), c.get("channel", 0)) for c in order}) != 1:
        raise ValueError(
            "Use one original speaker channel per translation stream, preserving the common source family across streams"
        )
    arrived = []
    for index, chunk in enumerate(order):
        arrived.append((chunk["start_sample"], chunk["end_sample"]))
        if chunk.get("direction", "incoming") != "incoming":
            raise ValueError("Streaming translation consumes incoming source chunks only")
        if type(chunk.get("wait_for_more", False)) is not bool:
            raise ValueError("wait_for_more is an explicit supervision boolean")
        target = chunk.get("translation") is not None
        if chunk.get("wait_for_more") and target:
            raise ValueError("A streaming decision cannot both wait and emit a translation")
        if not target and not chunk.get("wait_for_more"):
            raise ValueError(
                "Every streaming arrival needs a supplied translation or explicit wait supervision"
            )
        if target:
            if not isinstance(chunk["translation"], str) or not chunk["translation"].strip():
                raise ValueError("Supplied streaming translations must be nonempty text")
            chunk.setdefault("language", dict(language))
            if chunk["language"] != language:
                raise ValueError(
                    "A sustained translation stream cannot silently change its language pair"
                )
        if options["policy"] == "segment_end" and target and index + 1 != len(order):
            raise ValueError("A segment_end policy may emit only after the announced source end")
        if chunk.get("translated_speech") and "audio" not in capabilities:
            raise ValueError("Translated speech needs an explicitly requested audio capability")
        span = chunk.get("translation_span")
        if span is not None and (
            not isinstance(span, dict)
            or type(span.get("start_sample")) is not int
            or type(span.get("end_sample")) is not int
            or not 0 <= span["start_sample"] < span["end_sample"]
        ):
            raise ValueError(
                "Translation labels must identify only already available source samples"
            )
        if target:
            start, end = (
                (chunk["start_sample"], chunk["end_sample"])
                if span is None
                else (span["start_sample"], span["end_sample"])
            )
            covered = start
            for left, right in sorted(arrived):
                if left > covered:
                    break
                covered = max(covered, right)
            if covered < end:
                raise ValueError(
                    "Translation labels must identify only already available source samples without gaps"
                )
        if target and options["policy"] == "segment_end" and len(order) > 1:
            if (
                span is None
                or span["start_sample"] != min(c["start_sample"] for c in order)
                or span["end_sample"] != max(c["end_sample"] for c in order)
            ):
                raise ValueError("A whole-segment translation needs its full available source span")
    if not order[-1].get("translation"):
        raise ValueError("The final streaming source arrival needs a nonempty supplied translation")
    minimum = len(order) + sum(c.get("translation") is not None for c in order) + 2
    if budget < minimum:
        raise ValueError("Streaming work budget cannot cover the declared lifecycle decisions")


def compile_streaming_translation(bundle, row, availability) -> Episode:
    """One immutable history: listen, choose wait/continue, emit, listen again.

    ``segment_end`` waits for an externally announced end of the selected source
    window. ``aligned_prefix`` additionally admits supplied prefix translations.
    Earlier emitted text/speech remains in every later decision's history.
    """
    options = row["streaming_translation"]
    policy = options["policy"]
    language = options["language"]
    namespace = stable_id("streaming-translation-v2", bundle.bundle_id)
    registry = TrackRegistry(max_active=2, address_prefix=namespace[:12])
    capabilities = tuple(options.get("output_modalities", ["text"]))
    budget = options.get("max_decisions", 4096)
    root = registry.create(
        "Coordinate streaming translation using only arrived source speech",
        capabilities=capabilities,
        budget=budget,
    )
    records, groups, actions, revisions, dependencies, decisions, translations = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    tracks = {root.address: TrackSpec(root.address, "assistant", description=root.purpose)}
    ready = set()
    current_ns = 0
    tools = lifecycle_tools(capabilities)
    relations = []

    def add(items):
        index = len(groups)
        events, physical = [], []
        for role, address, payload, metadata in items:
            metadata = dict(metadata)
            record_id = metadata.pop("record_id", stable_id(namespace, str(len(records))))
            scored = metadata.pop("scored", False)
            record = PhysicalRecord(
                record_id,
                role,
                payload,
                turn_index=index,
                logical_id=record_id,
                policy_mask=scored,
                provenance=metadata.pop("provenance", "executed"),
                media_type=metadata.pop("media_type", "text/plain; charset=utf-8"),
                action_id=metadata.pop("action_id", None),
                result_of=metadata.pop("result_of", None),
                execution_id=metadata.pop("execution_id", None),
            )
            records.append(record)
            physical.append(record)
            modality = metadata.pop("modality", "text")
            tracks.setdefault(address, TrackSpec(address, role))
            events.append(TrackEvent(record_id, address, modality, index, role, **metadata))
        groups.append(Meridian(index, tuple(events), selected_at_ns=current_ns))
        ready.update(record.record_id for record in physical)
        return tuple(physical)

    def control(operation, arguments):
        registry.charge((root.address,))
        payload = canonical_json({"name": "track_" + operation, "arguments": arguments}).encode()
        action_id = stable_id(namespace, "action", str(len(actions)), payload)
        (call,) = add(
            [
                (
                    "assistant",
                    root.address,
                    payload,
                    {
                        "scored": True,
                        "action_id": action_id,
                        "settings_json": canonical_json({"allowed_modalities": ["text"]}),
                    },
                )
            ]
        )
        if operation == "create":
            args = dict(arguments)
            purpose = args.pop("purpose")
            updated = registry.create(purpose, **args)
        else:
            args = dict(arguments)
            address = args.pop("address")
            updated = registry.change(address, operation, **args)
        execution = stable_id(namespace, "lifecycle-execution", str(len(actions)))
        actions.append(
            LogicalAction(
                action_id,
                action_id,
                "track_" + operation,
                canonical_json(arguments),
                (call.record_id,),
                digest_bytes(payload),
                execution_id=execution,
            )
        )
        (receipt,) = add(
            [
                (
                    "tool",
                    root.address,
                    canonical_json({"track": asdict(updated)}).encode(),
                    {
                        "result_of": action_id,
                        "execution_id": execution,
                    },
                )
            ]
        )
        if operation == "create":
            tracks.setdefault(updated.address, TrackSpec(updated.address, "assistant"))
            revisions.append(
                TrackRevision(receipt.record_id, updated.address, updated.purpose, updated.revision)
            )
        decisions.append(
            {
                "operation": operation,
                "record_id": call.record_id,
                "receipt": receipt.record_id,
                "available_at_ns": current_ns,
            }
        )
        return updated, receipt.record_id

    instruction = (
        f"Translate arriving speech from {language['source']} into {language['target']}. "
        "Keep earlier output and continue from it without restarting. "
        + (
            "Wait for the announced end of the source window before translating it."
            if policy == "segment_end"
            else "Emit a stable translation when the available speech supports it; otherwise wait for more source."
        )
    )
    add([("user", "user", instruction.encode(), {})])
    add([("environment", root.address, canonical_json({"coordinator": asdict(root)}).encode(), {})])
    worker, _ = control(
        "create",
        {
            "purpose": f"Continue the {language['target']} translation from arrived {language['source']} speech",
            "completion": "The final announced source window has been translated",
            "parent": root.address,
            "capabilities": list(capabilities),
            "budget": budget - 1,
        },
    )
    order = sorted(range(len(row["chunks"])), key=lambda i: (availability[f"c{i}"], i))
    source_ids = {i: stable_id(namespace, "source", str(i)) for i in order}
    prefix = []
    for ordinal, index in enumerate(order):
        chunk = row["chunks"][index]
        current_ns = availability[f"c{index}"]
        artifact = bundle.by_id(f"c{index}a")
        payload, media_type = artifact.record_bytes(generated=False)
        source_id = source_ids[index]
        add(
            [
                (
                    "user",
                    f"source:{chunk.get('speaker', chunk.get('channel', 0))}",
                    payload,
                    {
                        "record_id": source_id,
                        "media_type": media_type,
                        "modality": "audio",
                        "media_span": artifact.media_span,
                        "speaker": artifact.speaker,
                        "direction": "incoming",
                        "source_revision": f"{artifact.media.source.source_family}:{artifact.version}",
                        "available_at_ns": current_ns,
                        "asset_sha256": artifact.media.source.sha256,
                        "provenance": artifact.provenance,
                    },
                )
            ]
        )
        prefix.append(source_id)
        final = ordinal + 1 == len(order)
        next_source = None if final else source_ids[order[ordinal + 1]]
        # EOF and the next packet address are public transport facts. Whether
        # an aligned target exists and its words are strictly label-only.
        add(
            [
                (
                    "environment",
                    root.address,
                    canonical_json(
                        {
                            "source_event": source_id,
                            "end_of_source": final,
                            "next_event": next_source,
                        }
                    ).encode(),
                    {},
                )
            ]
        )
        if chunk.get("translation") is not None:
            if (
                registry.tracks[worker.address].status == "waiting"
                and not set(registry.tracks[worker.address].prerequisites) <= ready
            ):
                raise ValueError("Translation resumed before its requested source arrived")
            _, decision = control("continue", {"address": worker.address})
            if not registry.selectable(worker.address, ready):
                raise ValueError("Translation output selected on an ineligible worker")
            text = bundle.by_id(f"c{index}translation")
            items = [
                (
                    "assistant",
                    worker.address,
                    text.payload,
                    {
                        "scored": True,
                        "provenance": text.provenance,
                        "media_span": text.media_span,
                        "direction": "outgoing",
                        "alignment_id": f"{namespace}:{index}",
                        "instruction": f"Continue translating the available {language['source']} source into {language['target']}.",
                        "settings_json": canonical_json({"allowed_modalities": ["text"]}),
                    },
                )
            ]
            if chunk.get("translated_speech"):
                speech = bundle.by_id(f"c{index}sa")
                items.append(
                    (
                        "assistant",
                        worker.address,
                        speech.media.to_bytes(),
                        {
                            "scored": True,
                            "provenance": speech.provenance,
                            "media_type": MEDIA_RECORD_TYPE,
                            "modality": "audio",
                            "media_span": speech.media_span,
                            "direction": "outgoing",
                            "alignment_id": f"{namespace}:{index}",
                            "instruction": "Speak the continued translation using its own speech clock.",
                            "settings_json": canonical_json({"allowed_modalities": ["audio"]}),
                        },
                    )
                )
            emitted = add(items)
            if len(emitted) == 2:
                relations.append(
                    EpisodeRelation(
                        "same_spoken_content", emitted[0].record_id, emitted[1].record_id
                    )
                )
            registry.charge((worker.address,) * len(emitted))
            for record in emitted:
                dependencies.extend(
                    Dependency(earlier, record.record_id, "available_source") for earlier in prefix
                )
                dependencies.append(Dependency(decision, record.record_id, "translation_continued"))
            translations.append(
                {
                    "records": [r.record_id for r in emitted],
                    "source_prefix": tuple(prefix),
                    "available_at_ns": current_ns,
                }
            )
        if not final:
            control("wait", {"address": worker.address, "prerequisites": [next_source]})
        else:
            if chunk.get("translation") is None:
                raise ValueError(
                    "A completed streaming source needs its supplied final translation"
                )
            control("complete", {"address": worker.address})
            control("complete", {"address": root.address})
    annotations = {
        "compiler": "sustained-streaming-translation-v2",
        "language": language,
        "scheduling_provenance": "authored_policy_over_real_recorded_source",
        "natural_assistant_timing": False,
        "policy": policy,
        "decisions": decisions,
        "translations": translations,
        "source_license": bundle.source_license,
        "source_bundle": bundle.bundle_id,
        "media_provenance": {a.artifact_id: json.loads(a.transform_json) for a in bundle.artifacts},
    }
    return Episode(
        namespace,
        tuple(records),
        replace(bundle.lineage, task_id=namespace),
        stream="streaming_translation",
        actions=tuple(actions),
        tracks=tuple(tracks.values()),
        meridians=tuple(groups),
        track_revisions=tuple(revisions),
        dependencies=tuple(dependencies),
        omni_relations=tuple(relations),
        tools_json=canonical_json(tools),
        control_seed=instruction.encode(),
        supervision=Supervision(
            verification="annotated", annotations_json=canonical_json(annotations)
        ),
    )


def compile_open_translation(media, source, clip, *, language):
    """Use the same compiler in bounded and sharded open-media preparation.

    The streaming clock is the actual decoded local file's sample clock. Any
    upstream rate conversion remains a recorded transformation rather than an
    unsupported assertion that these are the original container coordinates.
    """
    import soundfile as sf

    from .concordance import AlignedBundle, BundleArtifact
    from .media_prepare import file_sha256
    from .open_media import frozen_split
    from .schema import Lineage, MediaSpan

    incoming = clip.get("audio", clip.get("source_audio"))
    info = sf.info(incoming)
    if info.channels != 1 or info.frames < 3:
        raise ValueError("Sustained translation needs an explicit original mono stream")
    raw_hash = file_sha256(incoming)
    clock = f"decoded-audio:{raw_hash}"
    boundaries = [0, info.frames // 3, 2 * info.frames // 3, info.frames]
    chunks, artifacts, availability = [], [], {}
    translation = clip.get("translation", clip.get("target_text"))
    if not translation or not language or not all(language.get(k) for k in ("source", "target")):
        raise ValueError("Streaming conversion requires supplied translation and public languages")
    speech = clip.get("target_audio")
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        when = end * 1_000_000_000 // info.samplerate
        chunk = {
            "start_sample": start,
            "end_sample": end,
            "channel": 0,
            "speaker": "",
            "available_at_ns": when,
        }
        if index < 2:
            chunk["wait_for_more"] = True
        else:
            chunk.update(
                translation=translation,
                translation_span={"start_sample": 0, "end_sample": info.frames},
            )
            if speech:
                target_info = sf.info(speech)
                if target_info.channels != 1:
                    raise ValueError("Translated speech needs one declared channel")
                chunk["translated_speech"] = {
                    "path": str(speech),
                    "start_sample": 0,
                    "end_sample": target_info.frames,
                    "channel": 0,
                    "sample_rate": target_info.samplerate,
                }
        chunks.append(chunk)
        prepared = media.audio_interval(
            incoming, start_sample=start, end_sample=end, channel=0, target=False
        )
        artifacts.append(
            BundleArtifact(
                f"c{index}a",
                "arrived source speech",
                "audio",
                media=prepared,
                media_span=MediaSpan(
                    clock,
                    start * 1000 // info.samplerate,
                    end * 1000 // info.samplerate,
                    start,
                    end,
                    info.samplerate,
                ),
                available_at_ns=when,
                transform_json=prepared.transform_json,
            )
        )
        availability[f"c{index}"] = when
    artifacts.append(
        BundleArtifact(
            "c2translation",
            "translated text",
            "text",
            payload=translation.encode(),
            media_span=MediaSpan(
                clock, 0, info.frames * 1000 // info.samplerate, 0, info.frames, info.samplerate
            ),
        )
    )
    if speech:
        output = media.media("audio", speech)
        speech_info = sf.info(speech)
        speech_hash = file_sha256(speech)
        artifacts.append(
            BundleArtifact(
                "c2sa",
                "translated speech",
                "audio",
                media=output,
                media_span=MediaSpan(
                    f"decoded-audio:{speech_hash}",
                    0,
                    speech_info.frames * 1000 // speech_info.samplerate,
                    0,
                    speech_info.frames,
                    speech_info.samplerate,
                ),
                transform_json=output.transform_json,
                provenance=source.options.get("translated_speech_provenance", "natural"),
            )
        )
    row = {
        "sample_rate": info.samplerate,
        "channels": 1,
        "chunks": chunks,
        "streaming_translation": {
            "policy": "segment_end",
            "language": language,
            "output_modalities": ["text", "audio"] if speech else ["text"],
        },
    }
    validate_streaming_translation(row)
    bundle = AlignedBundle(
        stable_id(
            source.source_id,
            clip["id"],
            "streaming-v2",
            raw_hash,
            translation,
            "" if not speech else speech_hash,
        ),
        Lineage(
            source.source_id,
            source.revision,
            clip["family"],
            frozen_split=frozen_split(clip["family"]),
            source_uri=f"hf://datasets/{source.repository}",
            raw_sha256=raw_hash,
        ),
        tuple(artifacts),
        source.license,
        "open-media-decoded-sample-clock-v1",
        "publisher-annotated",
    )
    return compile_streaming_translation(bundle, row, availability)
