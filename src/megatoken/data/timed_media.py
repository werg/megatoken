"""Manifest-driven preparation of locally supplied, clocked media chunks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from megatoken.data.concordance import (
    AlignedBundle,
    BundleArtifact,
    Relation,
    TaskGroup,
    TaskView,
    compile_task,
    generated,
    group,
    observed,
)
from megatoken.data.media_prepare import MediaPreparer, file_sha256
from megatoken.data.schema import Lineage, MediaSpan
from megatoken.data.sources import TermsReview


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL and validate clocks, overlap, source directions, and frozen splits.

    Overlap is legal across speakers/channels. Within one speaker/channel stream,
    chunks must be ordered and may not overlap.
    """
    path = Path(path)
    rows, families, recordings = [], {}, set()
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        required = {
            "recording_id",
            "family_id",
            "split",
            "audio",
            "sample_rate",
            "channels",
            "chunks",
        }
        if required - row.keys():
            raise ValueError(
                f"{path}:{line_no}: missing {', '.join(sorted(required - row.keys()))}"
            )
        if row["split"] not in {"train", "validation", "test", "quarantine"}:
            raise ValueError(f"{path}:{line_no}: invalid frozen family split")
        if families.setdefault(row["family_id"], row["split"]) != row["split"]:
            raise ValueError(f"{path}:{line_no}: related recordings have inconsistent splits")
        if row["recording_id"] in recordings:
            raise ValueError(f"{path}:{line_no}: duplicate recording ID")
        recordings.add(row["recording_id"])
        if (
            not row["recording_id"]
            or not row["family_id"]
            or type(row["sample_rate"]) is not int
            or row["sample_rate"] < 1
            or type(row["channels"]) is not int
            or row["channels"] < 1
            or not isinstance(row["chunks"], list)
            or not row["chunks"]
        ):
            raise ValueError(f"{path}:{line_no}: invalid recording identity or audio clock")
        last_by_stream: dict[tuple[str, int], int] = {}
        for chunk in row["chunks"]:
            if not isinstance(chunk, dict):
                raise ValueError(f"{path}:{line_no}: each timed chunk must be an object")
            start, end, channel = (
                chunk.get("start_sample"),
                chunk.get("end_sample"),
                chunk.get("channel", 0),
            )
            speaker = str(chunk.get("speaker", f"channel-{channel}"))
            key = (speaker, channel)
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or start < last_by_stream.get(key, -1)
                or end <= start
            ):
                raise ValueError(
                    f"{path}:{line_no}: chunks need ordered nonempty intervals within each speaker/channel"
                )
            if type(channel) is not int or not 0 <= channel < row["channels"]:
                raise ValueError(f"{path}:{line_no}: channel outside source recording")
            last_by_stream[key] = end
            if chunk.get("direction", "incoming") not in {"incoming", "outgoing"}:
                raise ValueError(f"{path}:{line_no}: direction must be incoming or outgoing")
            avail = chunk.get("available_at_ns", end * 1_000_000_000 // row["sample_rate"])
            if type(avail) is not int or avail < 0:
                raise ValueError(
                    f"{path}:{line_no}: availability must be an external nonnegative time"
                )
            video = chunk.get("video")
            if video and (
                type(video.get("start_tick")) is not int
                or type(video.get("end_tick")) is not int
                or video["start_tick"] < 0
                or video["start_tick"] >= video["end_tick"]
                or type(video.get("ticks_per_second")) is not int
                or video["ticks_per_second"] < 1
            ):
                raise ValueError(
                    f"{path}:{line_no}: video chunk needs exact presentation ticks and clock"
                )
            if video and (
                type(video.get("available_at_ns", avail)) is not int
                or video.get("available_at_ns", avail) < 0
            ):
                raise ValueError(f"{path}:{line_no}: video availability must be an external time")
            if not any(
                k in chunk
                for k in (
                    "transcript",
                    "translation",
                    "caption",
                    "translated_speech",
                    "generation_request",
                )
            ):
                raise ValueError(
                    f"{path}:{line_no}: chunk needs a supplied target or generation request"
                )
            if chunk.get("translated_speech"):
                speech = chunk["translated_speech"]
                if (
                    not isinstance(speech, dict)
                    or not isinstance(speech.get("path"), str)
                    or not all(
                        k in speech
                        for k in ("path", "start_sample", "end_sample", "sample_rate", "channel")
                    )
                    or type(speech["start_sample"]) is not int
                    or type(speech["end_sample"]) is not int
                    or type(speech["sample_rate"]) is not int
                    or type(speech["channel"]) is not int
                    or speech["start_sample"] < 0
                    or speech["end_sample"] <= speech["start_sample"]
                    or speech["sample_rate"] < 1
                    or speech["channel"] < 0
                ):
                    raise ValueError(
                        f"{path}:{line_no}: translated speech needs a checked local interval"
                    )
                if not chunk.get("translation"):
                    raise ValueError(
                        f"{path}:{line_no}: translated speech requires its matching text"
                    )
                if chunk.get("direction", "incoming") != "incoming":
                    raise ValueError(
                        f"{path}:{line_no}: translated-speech references must be conditioned on incoming source audio"
                    )
            if chunk.get("translation") is not None:
                language = chunk.get("language")
                if not isinstance(language, dict) or not all(
                    isinstance(language.get(k), str) and language[k] for k in ("source", "target")
                ):
                    raise ValueError(
                        f"{path}:{line_no}: translation needs source and target language labels"
                    )
            request = chunk.get("generation_request")
            if request is not None and (
                not isinstance(request, dict)
                or not isinstance(request.get("outputs"), list)
                or not request["outputs"]
                or not all(isinstance(output, str) for output in request["outputs"])
                or len(set(request["outputs"])) != len(request["outputs"])
                or not set(request["outputs"]) <= {"audio", "video", "caption"}
            ):
                raise ValueError(
                    f"{path}:{line_no}: generation_request.outputs supports audio, video, and caption"
                )
            if request is not None and (
                not isinstance(request.get("instruction"), str)
                or not request["instruction"].strip()
            ):
                raise ValueError(
                    f"{path}:{line_no}: generation_request needs an explicit instruction"
                )
            targets = chunk.get("generation_targets", {})
            if not isinstance(targets, dict):
                raise ValueError(f"{path}:{line_no}: generation_targets must be an object")
            if set(targets) - {"audio", "video", "caption"}:
                raise ValueError(f"{path}:{line_no}: unknown generated target modality")
            if request:
                missing_targets = set(request["outputs"]) - targets.keys()
                if missing_targets:
                    raise ValueError(
                        f"{path}:{line_no}: missing supplied targets for requested outputs {sorted(missing_targets)}"
                    )
            if "audio" in targets:
                spec = targets["audio"]
                if (
                    not all(
                        k in spec
                        for k in ("path", "start_sample", "end_sample", "sample_rate", "channel")
                    )
                    or type(spec["start_sample"]) is not int
                    or type(spec["end_sample"]) is not int
                    or spec["start_sample"] < 0
                    or spec["end_sample"] <= spec["start_sample"]
                ):
                    raise ValueError(
                        f"{path}:{line_no}: generated audio target needs an exact local interval"
                    )
            if "video" in targets:
                spec = targets["video"]
                if (
                    not all(
                        k in spec for k in ("path", "start_tick", "end_tick", "ticks_per_second")
                    )
                    or type(spec["start_tick"]) is not int
                    or type(spec["end_tick"]) is not int
                    or type(spec["ticks_per_second"]) is not int
                    or spec["start_tick"] < 0
                    or spec["end_tick"] <= spec["start_tick"]
                    or spec["ticks_per_second"] < 1
                ):
                    raise ValueError(
                        f"{path}:{line_no}: generated video target needs exact presentation ticks"
                    )
            if "caption" in targets and (
                not isinstance(targets["caption"], str) or not targets["caption"].strip()
            ):
                raise ValueError(
                    f"{path}:{line_no}: generated caption target must be nonempty text"
                )
            if "caption" in (request or {}).get("outputs", []) and not video:
                raise ValueError(
                    f"{path}:{line_no}: audiovisual caption generation needs a video interval"
                )
            if len(set((request or {}).get("outputs", [])) & {"audio", "video"}) == 2:
                audio_alignment = targets["audio"].get("alignment")
                video_alignment = targets["video"].get("alignment")
                if not audio_alignment or not video_alignment or audio_alignment != video_alignment:
                    raise ValueError(
                        f"{path}:{line_no}: generated audio/video peers require one explicit shared alignment annotation"
                    )
        rows.append(row)
    return rows


def _audio_span(row, chunk):
    start, end, rate = chunk["start_sample"], chunk["end_sample"], row["sample_rate"]
    return MediaSpan(
        row["recording_id"], round(start * 1000 / rate), round(end * 1000 / rate), start, end, rate
    )


def _target_span(path: Path, spec: dict, modality: str) -> MediaSpan:
    """Use an explicitly mapped source span, or retain the target's own clock."""
    alignment = spec.get("alignment")
    if alignment is not None:
        required = ("clock_id", "start_tick", "end_tick", "ticks_per_second")
        if (
            not all(k in alignment for k in required)
            or type(alignment["start_tick"]) is not int
            or type(alignment["end_tick"]) is not int
            or type(alignment["ticks_per_second"]) is not int
            or alignment["start_tick"] < 0
            or alignment["end_tick"] <= alignment["start_tick"]
            or alignment["ticks_per_second"] < 1
        ):
            raise ValueError("Target alignment needs an exact named clock interval")
        start, end, rate = (
            alignment["start_tick"],
            alignment["end_tick"],
            alignment["ticks_per_second"],
        )
        return MediaSpan(
            alignment["clock_id"],
            round(start * 1000 / rate),
            round(end * 1000 / rate),
            start,
            end,
            rate,
        )
    if modality == "audio":
        start, end, rate = spec["start_sample"], spec["end_sample"], spec["sample_rate"]
    else:
        start, end, rate = spec["start_tick"], spec["end_tick"], spec["ticks_per_second"]
    clock = f"{modality}:{file_sha256(path)}"
    return MediaSpan(clock, round(start * 1000 / rate), round(end * 1000 / rate), start, end, rate)


def _timed_group(events, at_ns: int) -> TaskGroup:
    return TaskGroup(tuple(events), selected_at_ns=at_ns)


def _terms(value: TermsReview | dict[str, Any]) -> TermsReview:
    terms = value if isinstance(value, TermsReview) else TermsReview(**value)
    terms.require_approval()
    if not terms.license or terms.license == "unreviewed":
        raise ValueError("An approved, recorded source license is required")
    return terms


def prepare_rows(
    manifest: str | Path,
    *,
    media: MediaPreparer,
    output: str | Path,
    source_terms: TermsReview | dict[str, Any],
    preprocessing_id: str = "timed-media-v2",
    source_id: str | None = None,
    max_recordings: int | None = None,
    codec=None,
    shard_bytes: int = 128 << 20,
) -> dict:
    """Prepare, compile, deduplicate, and optionally export admitted timed data."""
    terms = _terms(source_terms)
    manifest_path, output = Path(manifest).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest_path)
    if max_recordings is not None:
        if max_recordings < 1:
            raise ValueError("Pilot recording count must be positive")
        rows = rows[:max_recordings]
    episodes, receipts = [], []
    for row in rows:
        audio_path = (manifest_path.parent / row["audio"]).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        import soundfile as sf

        info = sf.info(audio_path)
        if info.samplerate != row["sample_rate"] or info.channels != row["channels"]:
            raise ValueError(f"Declared audio clock/channels differ from source file: {audio_path}")
        if max(c["end_sample"] for c in row["chunks"]) > info.frames:
            raise ValueError(f"Audio chunk extends beyond source recording: {audio_path}")
        if source_id and row.get("source_id", source_id) != source_id:
            raise ValueError(
                f"Recording source ID differs from its reviewed terms record: {row['recording_id']}"
            )
        audio_hash = file_sha256(audio_path)
        if row.get("audio_sha256", audio_hash) != audio_hash:
            raise ValueError(f"Audio digest mismatch: {audio_path}")
        artifacts, avail_by_id, audio_avail_by_id, video_avail_by_id = [], {}, {}, {}
        source_files = [{"role": "audio", "path": str(audio_path), "sha256": audio_hash}]
        for i, chunk in enumerate(row["chunks"]):
            cid, start, end = f"c{i}", chunk["start_sample"], chunk["end_sample"]
            channel, speaker = (
                chunk.get("channel", 0),
                str(chunk.get("speaker", f"channel-{chunk.get('channel', 0)}")),
            )
            media_record = media.audio_interval(
                audio_path, start_sample=start, end_sample=end, channel=channel
            )
            audio_avail = chunk.get("available_at_ns", end * 1_000_000_000 // row["sample_rate"])
            audio_avail_by_id[cid] = audio_avail
            video_avail = chunk.get("video", {}).get("available_at_ns", audio_avail)
            video_avail_by_id[cid] = video_avail
            avail = max(audio_avail, video_avail)
            avail_by_id[cid] = avail
            artifacts.append(
                BundleArtifact(
                    cid + "a",
                    "audio" + cid,
                    "audio",
                    media=media_record,
                    source_end_ms=round(end * 1000 / row["sample_rate"]),
                    media_span=_audio_span(row, chunk),
                    speaker=speaker if chunk.get("direction", "incoming") == "incoming" else "",
                    available_at_ns=audio_avail,
                    transform_json=media_record.transform_json,
                )
            )
            if chunk.get("video"):
                v = chunk["video"]
                vp = (manifest_path.parent / v.get("path", row.get("video", ""))).resolve()
                if not vp.is_file():
                    raise FileNotFoundError(vp)
                source_files.append({"role": "video", "path": str(vp), "sha256": file_sha256(vp)})
                fps = Fraction(v.get("output_fps_num", 2), v.get("output_fps_den", 1))
                vm = media.video_interval(
                    vp, start_tick=v["start_tick"], end_tick=v["end_tick"], fps=fps
                )
                vspan = MediaSpan(
                    row["recording_id"],
                    round(v["start_tick"] * 1000 / v["ticks_per_second"]),
                    round(v["end_tick"] * 1000 / v["ticks_per_second"]),
                    v["start_tick"],
                    v["end_tick"],
                    v["ticks_per_second"],
                )
                if max(
                    Fraction(chunk["start_sample"], row["sample_rate"]),
                    Fraction(v["start_tick"], v["ticks_per_second"]),
                ) >= min(
                    Fraction(chunk["end_sample"], row["sample_rate"]),
                    Fraction(v["end_tick"], v["ticks_per_second"]),
                ):
                    raise ValueError(
                        "Timed audio/video chunks do not overlap on their shared recording clock"
                    )
                artifacts.append(
                    BundleArtifact(
                        cid + "v",
                        "video" + cid,
                        "video",
                        media=vm,
                        source_end_ms=vspan.end_ms,
                        media_span=vspan,
                        speaker=speaker,
                        available_at_ns=video_avail,
                        transform_json=vm.transform_json,
                    )
                )
            for kind in ("transcript", "translation", "caption"):
                if chunk.get(kind) is not None:
                    artifacts.append(
                        BundleArtifact(
                            cid + kind,
                            kind + cid,
                            "text",
                            payload=str(chunk[kind]).encode(),
                            media_span=_audio_span(row, chunk),
                            speaker=speaker,
                            provenance="natural",
                            verification="annotated",
                        )
                    )
            targets = chunk.get("generation_targets", {})
            if "audio" in targets:
                spec = targets["audio"]
                target_path = (manifest_path.parent / spec["path"]).resolve()
                info = sf.info(target_path)
                if (
                    info.samplerate != spec["sample_rate"]
                    or spec["channel"] < 0
                    or spec["channel"] >= info.channels
                    or spec["end_sample"] > info.frames
                ):
                    raise ValueError(
                        f"Generated audio target clock/channel mismatch: {target_path}"
                    )
                source_files.append(
                    {
                        "role": "generated_audio_target",
                        "path": str(target_path),
                        "sha256": file_sha256(target_path),
                    }
                )
                target_audio = media.audio_interval(
                    target_path,
                    start_sample=spec["start_sample"],
                    end_sample=spec["end_sample"],
                    channel=spec["channel"],
                )
                span = _target_span(target_path, spec, "audio")
                artifacts.append(
                    BundleArtifact(
                        cid + "ga",
                        "generated audio" + cid,
                        "audio",
                        media=target_audio,
                        media_span=span,
                        speaker="",
                        provenance="natural",
                        verification="annotated",
                        transform_json=target_audio.transform_json,
                    )
                )
            if "video" in targets:
                spec = targets["video"]
                target_path = (manifest_path.parent / spec["path"]).resolve()
                if not target_path.is_file():
                    raise FileNotFoundError(target_path)
                source_files.append(
                    {
                        "role": "generated_video_target",
                        "path": str(target_path),
                        "sha256": file_sha256(target_path),
                    }
                )
                rate = Fraction(spec.get("output_fps_num", 2), spec.get("output_fps_den", 1))
                target_video = media.video_interval(
                    target_path, start_tick=spec["start_tick"], end_tick=spec["end_tick"], fps=rate
                )
                span = _target_span(target_path, spec, "video")
                artifacts.append(
                    BundleArtifact(
                        cid + "gv",
                        "generated video" + cid,
                        "video",
                        media=target_video,
                        media_span=span,
                        provenance="natural",
                        verification="annotated",
                        transform_json=target_video.transform_json,
                    )
                )
            if "caption" in targets:
                caption_alignment = targets.get("audio", {}).get("alignment") or targets.get(
                    "video", {}
                ).get("alignment")
                caption_span = (
                    _target_span(audio_path, {"alignment": caption_alignment}, "audio")
                    if caption_alignment
                    else _audio_span(row, chunk)
                )
                artifacts.append(
                    BundleArtifact(
                        cid + "gc",
                        "generated caption" + cid,
                        "text",
                        payload=targets["caption"].encode(),
                        media_span=caption_span,
                        provenance="natural",
                        verification="annotated",
                    )
                )
            if chunk.get("translated_speech"):
                s = chunk["translated_speech"]
                sp = (manifest_path.parent / s["path"]).resolve()
                si = sf.info(sp)
                if si.samplerate != s["sample_rate"] or si.channels <= s["channel"]:
                    raise ValueError(f"Translated speech clock/channel mismatch: {sp}")
                if s.get("sha256") and file_sha256(sp) != s["sha256"]:
                    raise ValueError(f"Translated speech digest mismatch: {sp}")
                source_files.append(
                    {"role": "translated_speech", "path": str(sp), "sha256": file_sha256(sp)}
                )
                if (
                    si.samplerate != s["sample_rate"]
                    or type(s["channel"]) is not int
                    or not 0 <= s["channel"] < si.channels
                    or type(s["start_sample"]) is not int
                    or type(s["end_sample"]) is not int
                    or not 0 <= s["start_sample"] < s["end_sample"] <= si.frames
                ):
                    raise ValueError(f"Translated speech interval or clock is invalid: {sp}")
                output_audio = media.audio_interval(
                    sp,
                    start_sample=s["start_sample"],
                    end_sample=s["end_sample"],
                    channel=s["channel"],
                )
                translated_span = _target_span(sp, s, "audio")
                artifacts.append(
                    BundleArtifact(
                        cid + "sa",
                        "translated speech" + cid,
                        "audio",
                        media=output_audio,
                        media_span=translated_span,
                        speaker="",
                        provenance="natural",
                        verification="annotated",
                        transform_json=output_audio.transform_json,
                    )
                )
        lineage = Lineage(
            source_id=row["recording_id"],
            revision=str(row.get("revision", "local")),
            family_id=row["family_id"],
            source_uri=str(audio_path),
            raw_sha256=audio_hash,
            frozen_split=row["split"],
        )
        bundle_id = (
            "timed-" + hashlib.sha256((row["recording_id"] + audio_hash).encode()).hexdigest()[:20]
        )
        bundle = AlignedBundle(
            bundle_id,
            lineage,
            tuple(artifacts),
            terms.license,
            preprocessing_id,
            row.get("verification", "annotated"),
            teacher_id=str(getattr(media.teacher, "identity", "locked-teacher")),
        )
        # Coavailable source intervals become peers. Incoming audio only is usable input;
        # outgoing source material can only be a checked generation target.
        availability: dict[int, list[int]] = defaultdict(list)
        for i, c in enumerate(row["chunks"]):
            if c.get("direction", "incoming") == "incoming":
                availability[avail_by_id[f"c{i}"]].append(i)
        for target_i, chunk in enumerate(row["chunks"]):
            cutoff = avail_by_id[f"c{target_i}"]
            outgoing = chunk.get("direction", "incoming") == "outgoing"
            grouped: dict[int, list] = defaultdict(list)
            for j in range(len(row["chunks"])):
                source_chunk = row["chunks"][j]
                if (
                    source_chunk.get("direction", "incoming") != "incoming"
                    or audio_avail_by_id[f"c{j}"] > cutoff
                ):
                    continue
                audio_at = audio_avail_by_id[f"c{j}"]
                track = f"speaker-{source_chunk.get('speaker', source_chunk.get('channel', 0))}-chunk-{j}"
                grouped[audio_at].append(
                    observed(
                        f"c{j}a",
                        track,
                        "Listen to this available source channel.",
                        available_until_ms=round(
                            source_chunk["end_sample"] * 1000 / row["sample_rate"]
                        ),
                    )
                )
                if source_chunk.get("video") and video_avail_by_id[f"c{j}"] <= cutoff:
                    grouped[video_avail_by_id[f"c{j}"]].append(
                        observed(
                            f"c{j}v",
                            f"source-video-chunk-{j}",
                            "Inspect the corresponding available video interval.",
                            available_until_ms=round(
                                source_chunk["video"]["end_tick"]
                                * 1000
                                / source_chunk["video"]["ticks_per_second"]
                            ),
                        )
                    )
            source_groups = tuple(
                _timed_group(events, when) for when, events in sorted(grouped.items())
            )
            tasks = []
            if chunk.get("transcript") is not None and not outgoing:
                tasks.append(
                    ("transcription", f"c{target_i}transcript", "transcribe the available speech")
                )
            if chunk.get("translation") is not None and not outgoing:
                languages = chunk["language"]
                instruction = f"Translate this available speech chunk from {languages['source']} into {languages['target']}."
                tasks.append(("speech_translation", f"c{target_i}translation", instruction))
                tasks.append(("streaming_translation", f"c{target_i}translation", instruction))
            if chunk.get("caption") is not None and not outgoing:
                tasks.append(
                    (
                        "av_understanding",
                        f"c{target_i}caption",
                        "describe the available audiovisual event",
                    )
                )
                tasks.append(
                    (
                        "timed_conversation",
                        f"c{target_i}caption",
                        "describe the available audiovisual event",
                    )
                )
            if outgoing:
                if chunk.get("transcript") is None:
                    raise ValueError("Outgoing recorded speech needs its aligned transcript target")
                tasks.append(
                    (
                        "timed_conversation",
                        f"c{target_i}transcript",
                        "produce the recorded outgoing speech content",
                    )
                )
            for family, aid, instruction in tasks:
                target_events = (
                    []
                    if outgoing
                    else [
                        generated(
                            aid,
                            "text-output",
                            instruction,
                            alignment_id=f"{row['recording_id']}:{target_i}",
                        )
                    ]
                )
                if outgoing:
                    target_events.append(
                        generated(
                            f"c{target_i}a",
                            "speech-output",
                            instruction,
                            alignment_id=f"{row['recording_id']}:{target_i}",
                        )
                    )
                    target_events.append(
                        generated(
                            aid,
                            "text-output",
                            "Preserve the aligned outgoing transcript.",
                            alignment_id=f"{row['recording_id']}:{target_i}",
                        )
                    )
                if family == "streaming_translation" and chunk.get("translated_speech"):
                    target_events.append(
                        generated(
                            f"c{target_i}sa",
                            "speech-output",
                            "speak the translation",
                            alignment_id=f"{row['recording_id']}:{target_i}",
                        )
                    )
                relations = []
                if (
                    not outgoing
                    and family == "streaming_translation"
                    and chunk.get("translated_speech")
                ):
                    relations.append(
                        Relation(
                            "same_spoken_content",
                            aid,
                            f"c{target_i}sa",
                            json.dumps(
                                {"language": chunk.get("language", {}), "chunk_index": target_i}
                            ),
                        )
                    )
                if family == "timed_conversation" and chunk.get("video"):
                    relations.append(Relation("describes", aid, f"c{target_i}v"))
                view = TaskView(
                    f"{target_i}-{family}",
                    family,
                    (*source_groups, _timed_group(target_events, cutoff)),
                    relations=tuple(relations),
                    shared_instruction=str(
                        chunk.get("generation_request", {}).get("instruction", "")
                    ),
                    verification_json=json.dumps(
                        {"language": chunk.get("language", {}), "natural_timing_label": False}
                    ),
                )
                episodes.append(compile_task(bundle, view))
            targets = chunk.get("generation_targets", {})
            outputs = chunk.get("generation_request", {}).get("outputs", [])
            if outputs:
                output_artifacts = {
                    "audio": f"c{target_i}ga",
                    "video": f"c{target_i}gv",
                    "caption": f"c{target_i}gc",
                }
                prompt = str(chunk["generation_request"]["instruction"])
                requests = [
                    generated(
                        output_artifacts[modality],
                        f"generated-{modality}-{target_i}",
                        f"{prompt} Produce the {modality} output.",
                        alignment_id=f"{row['recording_id']}:{target_i}",
                    )
                    for modality in outputs
                ]
                relations = []
                if "audio" in outputs and "video" in outputs:
                    relations.append(
                        Relation(
                            "synchronized_with",
                            output_artifacts["audio"],
                            output_artifacts["video"],
                            json.dumps(
                                {
                                    "timing_supervision": "annotation",
                                    "alignment": targets["audio"]["alignment"],
                                }
                            ),
                        )
                    )
                if "caption" in outputs and "video" in outputs:
                    relations.append(
                        Relation(
                            "describes", output_artifacts["caption"], output_artifacts["video"]
                        )
                    )
                episodes.append(
                    compile_task(
                        bundle,
                        TaskView(
                            f"{target_i}-av-creation",
                            "av_creation",
                            (*source_groups, _timed_group(requests, cutoff)),
                            relations=tuple(relations),
                            shared_instruction=str(
                                chunk["generation_request"].get("instruction", "")
                            ),
                            verification_json=json.dumps(
                                {
                                    "timing_supervision": "annotation",
                                    "language": chunk.get("language", {}),
                                }
                            ),
                        ),
                    )
                )
        # Explicit simultaneous natural speech overlap examples use source events and
        # supplied transcript targets, grouped by external arrival time.
        overlap_pairs = []
        for ai, left in enumerate(row["chunks"]):
            for right_i in range(ai + 1, len(row["chunks"])):
                right = row["chunks"][right_i]
                if (
                    left.get("direction", "incoming")
                    == right.get("direction", "incoming")
                    == "incoming"
                    and left.get("speaker", left.get("channel", 0))
                    != right.get("speaker", right.get("channel", 0))
                    and max(
                        left["start_sample"] / row["sample_rate"],
                        right["start_sample"] / row["sample_rate"],
                    )
                    < min(
                        left["end_sample"] / row["sample_rate"],
                        right["end_sample"] / row["sample_rate"],
                    )
                ):
                    overlap_pairs.append((ai, right_i))
        overlap_indices = sorted({i for pair in overlap_pairs for i in pair})
        if overlap_indices:
            when = max(avail_by_id[f"c{i}"] for i in overlap_indices)
            peers = [
                observed(
                    f"c{i}a",
                    f"speaker-{row['chunks'][i].get('speaker', row['chunks'][i].get('channel', 0))}-chunk-{i}",
                    "Listen to each simultaneously available speaker.",
                )
                for i in overlap_indices
            ]
            target_ids = [
                f"c{i}transcript"
                for i in overlap_indices
                if row["chunks"][i].get("transcript") is not None
            ]
            if target_ids:
                groups = [_timed_group(peers, when)]
                groups.append(
                    group(
                        *(
                            generated(
                                t, f"speaker-{i}", "Transcribe this speaker's overlapping speech."
                            )
                            for i, t in enumerate(target_ids)
                        )
                    )
                )
                episodes.append(
                    compile_task(
                        bundle,
                        TaskView(
                            f"overlap-{when}",
                            "timed_conversation",
                            tuple(groups),
                            shared_instruction="Preserve speaker separation.",
                        ),
                    )
                )
        receipts.append(
            {
                "source_id": row.get("source_id", source_id or "local-reviewed-source"),
                "recording_id": row["recording_id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "source_files": source_files,
                "chunk_count": len(row["chunks"]),
                "bundle_id": bundle_id,
            }
        )
    receipt = {
        "format": "timed-media-preparation/v2",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "terms": terms.__dict__
        if hasattr(terms, "__dict__")
        else {k: getattr(terms, k) for k in terms.__dataclass_fields__},
        "preprocessing_id": preprocessing_id,
        "episode_count": len(episodes),
        "sources": receipts,
    }
    (output / "episodes.jsonl").write_text(
        "".join(
            json.dumps(e.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for e in episodes
        )
    )
    (output / "source-receipts.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    if codec is not None:
        from megatoken.data.dedup import DedupConfig, DedupStore
        from megatoken.data.storage import export_prepared

        with DedupStore(
            output / "timed-media-dedup.sqlite",
            DedupConfig(train_fraction=1, validation_fraction=0),
        ) as store:
            for episode in episodes:
                store.add(episode, codec)
            store.freeze()
            export_prepared(
                store,
                output,
                codec=codec,
                source_receipts=[receipt],
                pipeline_fingerprint=hashlib.sha256(
                    (receipt["manifest_sha256"] + preprocessing_id).encode()
                ).hexdigest(),
                shard_bytes=shard_bytes,
                sidecars={"timed-media-report.json": receipt},
            )
    return receipt
