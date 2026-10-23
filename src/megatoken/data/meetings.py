"""AMI release joins over original channels, word annotations and exact prefixes.

Inspected against manual annotations 1.6.2 (2017 release) and ES2002a signals.
Natural speaker continuation is not relabeled as assistant tool execution.
Video timing is admitted only with an explicit reviewed offset; a matching
meeting name alone cannot certify lip synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from megatoken.io import sha256_file

from .media_assets import MEDIA_RECORD_TYPE
from .schema import (
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


@dataclass(frozen=True)
class Word:
    start: Fraction
    end: Fraction
    text: str
    punctuation: bool


@dataclass(frozen=True)
class MeetingSpeaker:
    agent: str
    identity: str
    channel: int
    camera: str
    words: tuple[Word, ...]


def read_ami_annotations(archive: str | Path, meeting: str):
    with ZipFile(archive) as source:
        root = ET.fromstring(source.read("corpusResources/meetings.xml"))
        selected = next((m for m in root if m.get("observation") == meeting), None)
        if selected is None:
            raise ValueError("Meeting has no released channel/speaker join")
        speakers = []
        for speaker in selected:
            agent = speaker.attrib["nxt_agent"]
            words = ET.fromstring(source.read(f"words/{meeting}.{agent}.words.xml"))
            labels = tuple(
                Word(
                    Fraction(w.attrib["starttime"]),
                    Fraction(w.attrib["endtime"]),
                    w.text or "",
                    w.get("punc") == "true",
                )
                for w in words
                if w.tag == "w" and "starttime" in w.attrib and "endtime" in w.attrib
            )
            if any(w.start < 0 or w.end < w.start for w in labels):
                raise ValueError("Released word timestamps are invalid")
            speakers.append(
                MeetingSpeaker(
                    agent,
                    speaker.attrib["global_name"],
                    int(speaker.attrib["channel"]),
                    speaker.attrib["camera"],
                    labels,
                )
            )
    return tuple(speakers)


def ami_conversation(
    preparer,
    directory: str | Path,
    archive: str | Path,
    *,
    meeting: str,
    start_second: int,
    chunks: int = 4,
    chunk_seconds: int = 2,
    agents=("A", "B"),
    outgoing="A",
    split="train",
):
    """Predict the next fixed-duration speaker chunk from strictly earlier media.

    First observe both speakers, then select outgoing speech before observing
    the corresponding incoming chunk. Silence and overlap remain in each
    natural channel. Public duration is a fixed collection policy, not inferred
    from hidden target speech. Exact word labels are loss-only annotations.
    """
    import soundfile as sf

    if chunks < 2 or not 1 <= chunk_seconds <= 20 or outgoing not in agents:
        raise ValueError("Conversation needs a bounded causal prefix and an outgoing speaker")
    directory, archive = Path(directory), Path(archive)
    speakers = {s.agent: s for s in read_ami_annotations(archive, meeting)}
    selected = [speakers[agent] for agent in agents]
    files = {s.agent: directory / f"{meeting}.Headset-{s.channel}.wav" for s in selected}
    clocks = {s.agent: sf.info(files[s.agent]) for s in selected}
    if len({v.samplerate for v in clocks.values()}) != 1 or any(
        v.channels != 1 for v in clocks.values()
    ):
        raise ValueError("AMI headset files need original separate mono channels on one clock")
    rate = next(iter(clocks.values())).samplerate
    end = (start_second + chunks * chunk_seconds) * rate
    if start_second < 0 or any(end > v.frames for v in clocks.values()):
        raise ValueError("Conversation prefix is outside an original channel")
    original_family = f"ami:{meeting[:-1]}" if meeting[-1] in "abcd" else f"ami:{meeting}"
    variant = stable_id(
        "ami-causal-prefix-v2",
        meeting,
        str(start_second),
        str(chunks),
        str(chunk_seconds),
        outgoing,
        ",".join(agents),
    )
    records, groups, dependencies, labels = [], [], [], []
    track_ids = {s.agent: f"speaker:{s.identity}" for s in selected}
    instruction = PhysicalRecord(
        "instruction",
        "user",
        b"Continue the selected participant's speech from the recorded meeting context. Keep listening to the other participant.",
    )
    records.append(instruction)
    groups.append(
        Meridian(0, (TrackEvent("instruction", "request", "text", 0, "user"),), selected_at_ns=0)
    )
    available = [instruction.record_id]

    def append_chunk(agent, number, generated):
        lo = (start_second + number * chunk_seconds) * rate
        hi = lo + chunk_seconds * rate
        media = preparer.audio_interval(
            files[agent], start_sample=lo, end_sample=hi, channel=0, target=generated
        )
        role = "assistant" if generated else "user"
        rid = stable_id("ami-chunk", variant, agent, str(number), role)
        record = PhysicalRecord(
            rid,
            role,
            media.to_bytes(),
            policy_mask=generated,
            provenance="natural",
            media_type=MEDIA_RECORD_TYPE,
        )
        records.append(record)
        span = MediaSpan(meeting, lo * 1000 // rate, hi * 1000 // rate, lo, hi, rate)
        # A selected output starts before the incoming interval is available.
        when = lo * 1_000_000_000 // rate if generated else hi * 1_000_000_000 // rate
        group_id = len(groups)
        event = TrackEvent(
            rid,
            track_ids[agent],
            "audio",
            group_id,
            role,
            media_span=span,
            speaker=speakers[agent].identity,
            direction="outgoing" if generated else "incoming",
            available_at_ns=None if generated else when,
            request=f"Continue this participant for {chunk_seconds} seconds" if generated else "",
        )
        groups.append(Meridian(group_id, (event,), selected_at_ns=when))
        if generated:
            dependencies.extend(
                Dependency(earlier, rid, "available_prefix") for earlier in available
            )
        available.append(rid)
        words = [
            w
            for w in speakers[agent].words
            if w.start < Fraction(hi, rate) and w.end > Fraction(lo, rate)
        ]
        labels.append(
            {
                "record_id": rid,
                "words": [
                    {"text": w.text, "start": str(w.start), "end": str(w.end)} for w in words
                ],
                "crosses_word_boundary": any(
                    w.start < Fraction(lo, rate) or w.end > Fraction(hi, rate) for w in words
                ),
            }
        )

    for agent in agents:
        append_chunk(agent, 0, False)
    for number in range(1, chunks):
        append_chunk(outgoing, number, True)
        for agent in agents:
            if agent != outgoing:
                append_chunk(agent, number, False)
    return Episode(
        variant,
        tuple(records),
        Lineage(
            "ami-meetings",
            "manual-1.6.2",
            original_family,
            frozen_split=split,
            source_uri="https://groups.inf.ed.ac.uk/ami/",
            raw_sha256=sha256_file(archive),
        ),
        tracks=(
            TrackSpec("request", "user"),
            *(
                TrackSpec(
                    track_ids[s.agent],
                    "assistant" if s.agent == outgoing else "user",
                    description=f"Meeting participant {s.agent}; preserve this voice and task context",
                )
                for s in selected
            ),
        ),
        meridians=tuple(groups),
        dependencies=tuple(dependencies),
        stream="timed_conversation",
        supervision=Supervision(
            annotations_json=canonical_json(
                {
                    "origin": "natural_recorded_speaker_continuation",
                    "continuation": "recorded",
                    "annotation_origin": "publisher_orthographic_words",
                    "word_labels": labels,
                    "original_files": {agent: sha256_file(path) for agent, path in files.items()},
                    "timing_quality": "original_separate_headsets; publisher_word_timing; not independently aligned",
                }
            )
        ),
    )
