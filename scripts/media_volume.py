"""The media sources' volume, what it costs to prepare, and how it meets the stage plan's shares.

    python scripts/media_volume.py sources        one row per source: clips, hours, repository GB,
                                                   prepared GB and estimated GPU hours
    python scripts/media_volume.py gap            per stream: clips available against the clips the
                                                   P1 share implies, with media counted by input size

A media event enters the data meter (D) by the number of teacher feature vectors the model reads
for it (``data/schema.py: encode_record``). The teacher's processor config
(``embeddinggemma-2/processor_config.json``) fixes them: audio 40 ms per vector (25 a second),
images at most 280 soft tokens, video one frame a second (at most 32) of at most 140 soft
tokens each; ``EVENT_OVERHEAD`` stands for the begin/end markers and the chat template around
each event. Text tokens per episode are the means measured with the reference codec on real
label text (docs/data.md); sources not measured use the estimates in ``TEXT_TOKENS``. Everything
here is arithmetic over the manifests' inventories: re-run it after editing them.
"""

from __future__ import annotations

import argparse
import math

from megatoken.data.media_sharding import CLIP_SECONDS, DEFAULT_CLIP_SECONDS
from megatoken.data.open_media import SOURCES, OpenSource
from megatoken.data.recipes import stage_recipe

AUDIO_TOKENS_PER_SECOND = 25
IMAGE_TOKENS = 280
VIDEO_TOKENS_PER_FRAME = 140
VIDEO_MAX_FRAMES = 32
EVENT_OVERHEAD = 8
P1_LIMIT = 400e9

# Mean text tokens per episode (reference codec): measured for the first forty-four sources'
# label text, estimated (marked) for the others.
TEXT_TOKENS = {
    "audio_identity": 5.0,
    "image_identity": 5.0,
    "video_identity": 5.0,
    "transcription": 42.0,
    "verbalization": 19.7,
    "speech_translation": 73.5,
    "image_description": 142.2,
    "image_creation": 158.5,
    "audio_description": 30.0,
    "image_edit": 11.1,
    "speech_edit": 10.0,
    "av_creation": 54.5,
    "av_edit": 5.1,
}
EDIT_TASKS = ("image_edit", "image_edit_chain", "video_edit", "video_edit_chain")
QUESTION_TURN_TOKENS = 60.0  # one question and its answer (estimate)
EDIT_TURN_TOKENS = 11.1  # one edit request (as measured for single edits)

# Default clip lengths (seconds) when a source lists no hours: speech and sound clips are
# 0.5-20 s, video clips at most 8 s (6 for edits).
DEFAULT_SECONDS = {"audio": 10.0, "video": 6.0}

# Bytes of a prepared media event beyond its teacher features: the embedding (768 floats) and
# the codec latents (float32). Latent sizes are estimates until a real root is measured.
EMBEDDING_BYTES = 768 * 4
IMAGE_LATENT_BYTES = 32 * 32 * 32 * 4
AUDIO_LATENT_BYTES_PER_SECOND = 12.5 * 32 * 4
VIDEO_LATENT_BYTES_PER_SECOND = 150_000


def audio_seconds(source: OpenSource) -> float:
    inventory = source.inventory
    if inventory.get("hours") and inventory.get("items"):
        return min(inventory["hours"] * 3600 / inventory["items"], 20.0)
    return source.options.get("max_seconds", DEFAULT_SECONDS["audio"])


def video_seconds(source: OpenSource) -> float:
    inventory = source.inventory
    if inventory.get("hours") and inventory.get("items"):
        return min(
            inventory["hours"] * 3600 / inventory["items"], source.options.get("max_seconds", 8.0)
        )
    return source.options.get("max_seconds", DEFAULT_SECONDS["video"])


def media_tokens(modality: str, seconds: float) -> float:
    if modality == "audio":
        return AUDIO_TOKENS_PER_SECOND * seconds + EVENT_OVERHEAD
    if modality == "image":
        return IMAGE_TOKENS + EVENT_OVERHEAD
    frames = min(VIDEO_MAX_FRAMES, max(1, math.ceil(seconds)))
    return VIDEO_TOKENS_PER_FRAME * frames + EVENT_OVERHEAD


def media_bytes(modality: str, seconds: float) -> float:
    features = media_tokens(modality, seconds) * 512 * 4
    latents = {
        "audio": AUDIO_LATENT_BYTES_PER_SECOND * seconds,
        "image": IMAGE_LATENT_BYTES,
        "video": VIDEO_LATENT_BYTES_PER_SECOND * seconds,
    }[modality]
    return features + EMBEDDING_BYTES + latents


def events(source: OpenSource) -> list[tuple[str, float]]:
    """The media events one clip of the source prepares: ``(modality, seconds)``."""
    task, seconds = source.task, 0.0
    turns = source.inventory.get("mean_turns", 1.0)
    audio, video = audio_seconds(source), video_seconds(source)
    if task in ("transcription", "translation_text", "sound", "captioned_audio"):
        return [("audio", audio)]
    if task == "verbalization":
        return [("audio", audio), ("audio", audio)]  # the speech and the voice sample
    if task == "translation_pair":
        return [("audio", audio), ("audio", audio)]
    if task in ("captioned_image", "picture", "image_questions"):
        return [("image", seconds)]
    if task == "image_edit":
        return [("image", seconds)] * 2
    if task == "image_edit_chain":
        return [("image", seconds)] * round(1 + turns)
    if task in ("av_caption", "av_scenes"):
        return [("video", video), ("audio", video)]
    if task == "video":
        return [("video", video)]
    if task == "video_edit":
        return [("video", video)] * 2
    if task == "video_edit_chain":
        return [("video", video)] * round(1 + turns)
    raise ValueError(f"No event model for task {task}")


def episode_tokens(source: OpenSource) -> dict[str, tuple[float, float]]:
    """Per stream the source feeds: ``(episodes per clip, D tokens per episode)``."""
    clip_events = events(source)
    media = [media_tokens(modality, seconds) for modality, seconds in clip_events]
    turns = source.inventory.get("mean_turns", 1.0)
    first = media[0]
    text = dict(TEXT_TOKENS)
    if source.task == "image_questions":
        text["image_description"] = QUESTION_TURN_TOKENS * turns
    if source.task in ("image_edit_chain", "video_edit_chain"):
        text["image_edit"] = text["av_edit"] = EDIT_TURN_TOKENS * turns
    result: dict[str, tuple[float, float]] = {}
    for stream in source.streams:
        if stream.endswith("_identity"):
            result[stream] = (1.0, first + text[stream])
        elif stream in ("image_edit", "av_edit", "speech_edit"):
            # Real edit sets prepare both versions (all of a chain); a procedural edit of a
            # clip prepares the clip and its edited copy.
            real = source.task in EDIT_TASKS
            result[stream] = (1.0, (sum(media) if real else 2 * first) + text[stream])
        elif stream == "image_creation":
            result[stream] = (1.0, first + text[stream])
        elif stream == "av_creation":
            result[stream] = (1.0, sum(media) + text[stream])
        elif stream == "verbalization":
            result[stream] = (1.0, sum(media) + text[stream])
        else:
            result[stream] = (1.0, first + text[stream])
    return result


def prepared_gb(source: OpenSource) -> float:
    clips = source.inventory.get("items", 0)
    total = sum(media_bytes(modality, seconds) for modality, seconds in events(source))
    return clips * total / 1e9


def gpu_hours(source: OpenSource) -> float:
    seconds = source.options.get(
        "clip_seconds", CLIP_SECONDS.get(source.task, DEFAULT_CLIP_SECONDS)
    )
    return source.inventory.get("items", 0) * seconds / 3600


def sources_table() -> str:
    lines = [
        "| Source | Task | Clips | Hours | Repository GB | Prepared GB (est.) | GPU hours (est.) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    for source in SOURCES.values():
        inventory = source.inventory
        row = [
            inventory.get("items", 0),
            inventory.get("hours", 0),
            inventory.get("size_gb", 0),
            prepared_gb(source),
            gpu_hours(source),
        ]
        totals = [a + b for a, b in zip(totals, row, strict=True)]
        lines.append(
            f"| `{source.source_id}` | {source.task} | {row[0]:,.0f} | {row[1]:,.0f} | "
            f"{row[2]:,.0f} | {row[3]:,.0f} | {row[4]:,.0f} |"
        )
    lines.append(
        f"| **total** | | {totals[0]:,.0f} | {totals[1]:,.0f} | {totals[2]:,.0f} | "
        f"{totals[3]:,.0f} | {totals[4]:,.0f} |"
    )
    return "\n".join(lines)


def gap_table(only: set[str] | None = None) -> str:
    mixture = stage_recipe("P1").mixture
    streams = sorted(
        name
        for name in mixture
        if name.endswith(("_identity", "_creation", "_edit", "_translation", "_description"))
        or name in {"transcription", "verbalization"}
    )
    lines = [
        "| Stream | Sources | Episodes available | D tokens per episode | P1 share (tokens) "
        "| Episodes the share implies | Passes over the data |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for stream in streams:
        episodes = tokens = 0.0
        count = 0
        for source in SOURCES.values():
            if only is not None and source.source_id not in only:
                continue
            per_clip = episode_tokens(source).get(stream)
            if per_clip is None or per_clip[0] == 0:
                continue
            clips = source.inventory.get("items", 0) * per_clip[0]
            episodes += clips
            tokens += clips * per_clip[1]
            count += 1
        share = mixture[stream]
        mean = tokens / episodes if episodes else 0.0
        implied = share * P1_LIMIT / mean if mean else 0.0
        passes = implied / episodes if episodes else math.inf
        lines.append(
            f"| {stream} | {count} | {episodes:,.0f} | {mean:,.0f} | {share:.2%} "
            f"({share * P1_LIMIT / 1e9:.1f}B) | {implied:,.0f} | {passes:,.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=("sources", "gap"))
    parser.add_argument("--only", nargs="*", help="limit the gap table to these source ids")
    args = parser.parse_args()
    if args.table == "sources":
        print(sources_table())
    else:
        print(gap_table(set(args.only) if args.only else None))


if __name__ == "__main__":
    main()
