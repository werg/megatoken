import json
import sys
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from megatoken.data.coverage import behaviors
from megatoken.data.media_assets import MediaRecord, OutputGeometry, TensorAssetStore
from megatoken.data.media_prepare import file_sha256
from megatoken.data.schema import Episode
from megatoken.data.sources import TermsReview
from megatoken.data.timed_media import prepare_rows, read_manifest
from megatoken.generation.codec import ByteCodec
from megatoken.teacher import PINNED_TEACHER

TERMS = TermsReview(
    approved=True,
    license="CC-BY-4.0",
    license_url="https://example.org/license",
    reviewed_by="reviewer",
    reviewed_at="2026-10-23",
    redistribute="permissive",
    redistribute_basis="Source terms permit redistribution.",
)


def write(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def row(**changes):
    value = {
        "recording_id": "rec-a",
        "family_id": "conversation-a",
        "split": "train",
        "audio": "a.wav",
        "sample_rate": 16000,
        "channels": 2,
        "chunks": [
            {
                "start_sample": 0,
                "end_sample": 1600,
                "channel": 1,
                "speaker": "spk-b",
                "transcript": "hi",
            },
            {
                "start_sample": 800,
                "end_sample": 2400,
                "channel": 0,
                "speaker": "spk-a",
                "translation": "salut",
                "language": {"source": "en", "target": "fr"},
            },
        ],
    }
    value.update(changes)
    return value


def test_reads_overlapping_distinct_speakers_but_keeps_each_stream_ordered(tmp_path):
    rows = read_manifest(write(tmp_path, [row()]))
    assert rows[0]["chunks"][0]["channel"] == 1
    bad = row(
        chunks=[row()["chunks"][0], {**row()["chunks"][0], "start_sample": 800, "end_sample": 2400}]
    )
    with pytest.raises(ValueError, match="within each speaker/channel"):
        read_manifest(write(tmp_path, [bad]))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(split="random"),
        lambda r: r["chunks"][1].update(channel=2),
        lambda r: r["chunks"][0].update(end_sample=0),
    ],
)
def test_rejects_unfrozen_split_or_invalid_clock_and_channel(tmp_path, mutate):
    value = row()
    mutate(value)
    with pytest.raises(ValueError):
        read_manifest(write(tmp_path, [value]))


def test_related_recordings_cannot_cross_frozen_family_splits(tmp_path):
    first, second = row(), row(recording_id="rec-b", split="test")
    with pytest.raises(ValueError, match="inconsistent splits"):
        read_manifest(write(tmp_path, [first, second]))


def test_video_requires_explicit_presentation_clock(tmp_path):
    value = row()
    value["chunks"][0]["video"] = {"start_tick": 0, "end_tick": 10}
    with pytest.raises(ValueError, match="presentation ticks"):
        read_manifest(write(tmp_path, [value]))


def test_preparation_requires_reviewed_source_terms():
    with pytest.raises(ValueError, match="unapproved"):
        from megatoken.data.timed_media import _terms

        _terms({"approved": False, "license": "CC-BY-4.0"})


class FakeMediaPreparer:
    """Use real immutable tensor files and real storage export, with no teacher weights."""

    teacher = type("Teacher", (), {"identity": "test-teacher"})()

    def __init__(self, root):
        self.store = TensorAssetStore(root)

    def audio_interval(self, path, *, start_sample, end_sample, channel, target=True):
        family = file_sha256(path)
        source = self.store.put(
            np.ones((2, 512), np.float32),
            modality="audio",
            kind="features",
            extractor=PINNED_TEACHER,
            source_family=family,
        )
        embedding = self.store.put(
            np.ones(768, np.float32),
            modality="audio",
            kind="embedding",
            extractor=PINNED_TEACHER,
            source_family=family,
        )
        latents = self.store.put(
            np.ones((2, 16), np.float32),
            modality="audio",
            kind="codec_latents",
            extractor="test-pocket:abc",
            source_family=family,
            sample_rate=24000,
        )
        return MediaRecord(
            "audio",
            source,
            latents if target else None,
            OutputGeometry("audio", sample_count=240, sample_rate=24000) if target else None,
            embedding=embedding,
            transform_json=json.dumps(
                {"channel": channel, "start": start_sample, "end": end_sample}
            ),
        )

    def video_interval(
        self, path, *, start_tick, end_tick, fps, target=True, ticks_per_second=None
    ):
        family = file_sha256(path)
        source = self.store.put(
            np.ones((1, 512), np.float32),
            modality="video",
            kind="features",
            extractor=PINNED_TEACHER,
            source_family=family,
        )
        embedding = self.store.put(
            np.ones(768, np.float32),
            modality="video",
            kind="embedding",
            extractor=PINNED_TEACHER,
            source_family=family,
        )
        latents = self.store.put(
            np.ones((1, 2, 2, 2), np.float32),
            modality="video",
            kind="codec_latents",
            extractor="test-video:abc",
            source_family=family,
            fps_num=fps.numerator,
            fps_den=fps.denominator,
        )
        return MediaRecord(
            "video",
            source,
            latents if target else None,
            OutputGeometry(
                "video", frames=1, height=8, width=8, fps_num=fps.numerator, fps_den=fps.denominator
            )
            if target
            else None,
            embedding=embedding,
            transform_json=json.dumps({"start": start_tick, "end": end_tick}),
        )


def test_sustained_translation_waits_and_retains_earlier_emissions(tmp_path):
    from megatoken.controls import resolve_event_control
    from megatoken.data import PreparedDataset, WindowConfig, iter_windows

    with wave.open(str(tmp_path / "a.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros(4800, dtype="<i2").tobytes())
    value = row(
        channels=1,
        streaming_translation={
            "policy": "aligned_prefix",
            "language": {"source": "en", "target": "fr"},
        },
        chunks=[
            {"start_sample": 0, "end_sample": 1600, "wait_for_more": True},
            {"start_sample": 1600, "end_sample": 3200, "translation": "bonjour"},
            {"start_sample": 3200, "end_sample": 4800, "translation": "monde"},
        ],
    )
    root = tmp_path / "prepared"
    codec = ByteCodec.byte_fallback()
    prepare_rows(
        write(tmp_path, [value]),
        media=FakeMediaPreparer(root),
        output=root,
        source_terms=TERMS,
        codec=codec,
    )
    with PreparedDataset(root, split="train", stream="streaming_translation") as data:
        assert len(data) == 1
        episode = data[0]
        assert Episode.from_dict(episode.to_dict()) == episode
        waits = [a for a in episode.actions if a.tool_name == "track_wait"]
        assert len(waits) == 2
        positions = {r.record_id: i for i, r in enumerate(episode.records)}
        events = {e.record_id: e for g in episode.meridians for e in g.events}
        for action in waits:
            for source_id in json.loads(action.arguments_json)["prerequisites"]:
                assert positions[action.record_ids[0]] < positions[source_id]
                assert events[source_id].modality == "audio"
        control = resolve_event_control(episode, events[waits[0].record_ids[0]])
        assert b"bonjour" not in control and b"monde" not in control
        last = next(r for r in episode.records if r.payload == b"monde")
        window = next(
            w
            for w in iter_windows(episode, codec, WindowConfig(max_records=2))
            if any(r.record.record_id == last.record_id for r in w.records)
        )
        assert any(r.record.payload == b"bonjour" for r in window.prefix)
        assert "lifecycle_wait" in behaviors(episode)
        assert "streaming_wait_emit" in behaviors(episode)


def test_streaming_segment_labels_cannot_cover_unarrived_audio(tmp_path):
    value = row(
        channels=1,
        streaming_translation={
            "policy": "segment_end",
            "language": {"source": "en", "target": "fr"},
        },
        chunks=[
            {"start_sample": 0, "end_sample": 1600, "wait_for_more": True},
            {
                "start_sample": 1600,
                "end_sample": 3200,
                "translation": "bonjour",
                "translation_span": {"start_sample": 0, "end_sample": 4800},
            },
        ],
    )
    with pytest.raises(ValueError, match="already available"):
        read_manifest(write(tmp_path, [value]))
    value["chunks"][1]["translation_span"]["end_sample"] = 3200
    assert read_manifest(write(tmp_path, [value]))
    value["chunks"][1]["start_sample"] = 1800
    with pytest.raises(ValueError, match="without gaps"):
        read_manifest(write(tmp_path, [value]))
    value["chunks"][1]["start_sample"] = 1600
    value["chunks"][0]["translation"] = "future target"
    with pytest.raises(ValueError, match="both wait and emit"):
        read_manifest(write(tmp_path, [value]))


def test_real_dedup_export_publishes_readable_prepared_dataset(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros((3200, 2), dtype="<i2").tobytes())
    import sys

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(
            info=lambda _path: SimpleNamespace(samplerate=16000, channels=2, frames=3200)
        ),
    )
    manifest_row = row(
        chunks=[
            {
                "start_sample": 0,
                "end_sample": 1600,
                "channel": 1,
                "speaker": "s1",
                "transcript": "hello",
            }
        ]
    )
    manifest_row["audio_sha256"] = file_sha256(audio)
    manifest = write(tmp_path, [manifest_row])
    root = tmp_path / "prepared"
    media = FakeMediaPreparer(root)
    codec = ByteCodec([None, *[bytes([i]) for i in range(256)]])
    report = prepare_rows(manifest, media=media, output=root, source_terms=TERMS, codec=codec)
    from megatoken.data.storage import PreparedDataset

    with PreparedDataset(root, split="train") as prepared:
        assert len(prepared) >= 1
        assert report["episode_count"] >= 1
        assert prepared.manifest["complete"] is True
        assert (root / "source-receipts.json").is_file()


def test_audio_interval_extracts_requested_channel_and_preserves_source_ticks(
    tmp_path, monkeypatch
):
    audio = tmp_path / "stereo.wav"
    samples = np.array([[1000, -1000], [2000, -2000], [3000, -3000], [4000, -4000]], dtype="<i2")
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())

    class SoundFileReader:
        def __init__(self, path):
            self.wav = wave.open(str(path), "rb")
            self.channels, self.samplerate = self.wav.getnchannels(), self.wav.getframerate()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.wav.close()

        def __len__(self):
            return self.wav.getnframes()

        def seek(self, sample):
            self.wav.setpos(sample)

        def read(self, frames, dtype, always_2d):
            return (
                np.frombuffer(self.wav.readframes(frames), dtype="<i2")
                .reshape(-1, self.channels)
                .astype(np.float32)
                / 32768
            )

    def write_float(path, data, rate, subtype):
        values = np.clip(np.asarray(data) * 32767, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(values.tobytes())

    monkeypatch.setitem(
        sys.modules, "soundfile", SimpleNamespace(SoundFile=SoundFileReader, write=write_float)
    )
    from megatoken.data.media_prepare import MediaPreparer

    media = object.__new__(MediaPreparer)
    fake = FakeMediaPreparer(tmp_path / "assets")
    captured = {}

    def capture(modality, path, target=True):
        with wave.open(str(path), "rb") as wav:
            captured["samples"] = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        return fake.audio_interval(
            path, start_sample=0, end_sample=len(captured["samples"]), channel=0, target=target
        )

    media.media = capture
    record = media.audio_interval(audio, start_sample=1, end_sample=4, channel=1)
    np.testing.assert_allclose(
        captured["samples"] / 32767, np.array([-2000, -3000, -4000]) / 32768, atol=1 / 32767
    )
    transform = json.loads(record.transform_json)
    assert (
        transform["channel"],
        transform["start_tick"],
        transform["end_tick"],
        transform["ticks_per_second"],
    ) == (1, 1, 4, 16000)


def test_compiles_natural_overlap_and_peer_text_speech_translation(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros((3200, 2), dtype="<i2").tobytes())
    import sys

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(
            info=lambda _path: SimpleNamespace(samplerate=16000, channels=2, frames=3200)
        ),
    )
    manifest_row = row(
        chunks=[
            {
                "start_sample": 0,
                "end_sample": 1600,
                "channel": 1,
                "speaker": "s1",
                "transcript": "hello",
            },
            {
                "start_sample": 800,
                "end_sample": 2400,
                "channel": 0,
                "speaker": "s2",
                "transcript": "there",
                "translation": "salut",
                "language": {"source": "en", "target": "fr"},
                "translated_speech": {
                    "path": "a.wav",
                    "start_sample": 0,
                    "end_sample": 1600,
                    "sample_rate": 16000,
                    "channel": 0,
                },
            },
            {
                "start_sample": 2400,
                "end_sample": 3200,
                "channel": 1,
                "speaker": "s3",
                "available_at_ns": 300_000_000,
                "transcript": "later",
            },
        ]
    )
    manifest = write(tmp_path, [manifest_row])
    root = tmp_path / "prepared"
    prepare_rows(manifest, media=FakeMediaPreparer(root), output=root, source_terms=TERMS)
    episodes = [
        Episode.from_dict(json.loads(line))
        for line in (root / "episodes.jsonl").read_text().splitlines()
    ]
    overlap = next(e for e in episodes if e.stream == "timed_conversation")
    assert len(overlap.meridians[0].events) == 2
    assert overlap.meridians[0].selected_at_ns == 150_000_000
    assert "natural_speaker_overlap" in behaviors(overlap)
    assert "versioned_source_arrival" in behaviors(overlap)
    translations = [e for e in episodes if e.stream == "streaming_translation"]
    assert len(translations) == 1
    assert len(translations[0].meridians[-1].events) == 2
    input_tracks = {
        event.track_id for group in translations[0].meridians[:-1] for event in group.events
    }
    assert "speaker-s3-chunk-2" not in input_tracks
    translated_text, translated_audio = translations[0].meridians[-1].events
    assert "en into fr" in translated_text.instruction
    assert translated_audio.media_span.clock_id.startswith("audio:")
    assert translated_audio.media_span.end_tick == 1600


def test_joint_timed_audiovisual_generation_uses_shared_annotation_clock(tmp_path, monkeypatch):
    audio = tmp_path / "a.wav"
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.zeros((16000, 2), dtype="<i2").tobytes())
    video = tmp_path / "v.mkv"
    video.write_bytes(b"fake video, decoder is mocked")
    import sys

    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(
            info=lambda _path: SimpleNamespace(samplerate=16000, channels=2, frames=16000)
        ),
    )
    alignment = {
        "clock_id": "rec-av",
        "start_tick": 0,
        "end_tick": 16000,
        "ticks_per_second": 16000,
    }
    item = {
        "recording_id": "rec-av",
        "family_id": "rec-av",
        "split": "train",
        "audio": "a.wav",
        "sample_rate": 16000,
        "channels": 2,
        "chunks": [
            {
                "start_sample": 0,
                "end_sample": 16000,
                "channel": 0,
                "speaker": "speaker-1",
                "video": {"path": "v.mkv", "start_tick": 0, "end_tick": 24, "ticks_per_second": 24},
                "generation_request": {
                    "instruction": "Create matching media.",
                    "outputs": ["audio", "video", "caption"],
                },
                "generation_targets": {
                    "audio": {
                        "path": "a.wav",
                        "start_sample": 0,
                        "end_sample": 16000,
                        "sample_rate": 16000,
                        "channel": 0,
                        "alignment": alignment,
                    },
                    "video": {
                        "path": "v.mkv",
                        "start_tick": 0,
                        "end_tick": 24,
                        "ticks_per_second": 24,
                        "alignment": alignment,
                    },
                    "caption": "A person waves.",
                },
            }
        ],
    }
    manifest = write(tmp_path, [item])
    root = tmp_path / "prepared-av"
    prepare_rows(manifest, media=FakeMediaPreparer(root), output=root, source_terms=TERMS)
    episodes = [
        Episode.from_dict(json.loads(line))
        for line in (root / "episodes.jsonl").read_text().splitlines()
    ]
    generated_av = next(e for e in episodes if e.stream == "av_creation")
    assert len(generated_av.meridians[-1].events) == 3
    audio_event, video_event, caption_event = generated_av.meridians[-1].events
    assert audio_event.media_span == video_event.media_span == caption_event.media_span
    assert "timed_audiovisual" in behaviors(generated_av)


def test_manifest_ticks_are_converted_to_the_containers_time_base():
    from fractions import Fraction

    from megatoken.data.media_timing import container_ticks

    # The shipped example declares ticks 0..24 at 24 per second for a Matroska file, whose
    # clock is 1/1000 s: that is the first second (1000 container ticks), not 24 milliseconds.
    assert container_ticks(24, 24, Fraction(1, 1000)) == 1000
    assert container_ticks(0, 24, Fraction(1, 1000)) == 0
    assert container_ticks(3, 24, Fraction(1, 90000)) == 11250  # a 90 kHz transport clock
