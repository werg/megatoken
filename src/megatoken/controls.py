"""One causal control protocol shared by training, serving, and tape replay.

Seed/reference bytes retain their exact spelling and framing. Only public tool
JSON is canonicalized, so transport whitespace cannot change model conditioning.
Payload construction needs no ML runtime; tensor packing is imported on demand.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from megatoken.io import strict_json_loads

if TYPE_CHECKING:
    from torch import Tensor

    from megatoken.model import FacetedCore


CONTROL_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ControlReference:
    """One accessible reference; media bytes remain in the asset store."""

    asset_id: str
    sha256: str
    media_type: str
    purpose: str
    origin: str

    def __post_init__(self) -> None:
        if self.origin not in {"external_evidence", "earlier_history"}:
            raise ValueError("Control reference must have available causal provenance")
        if not self.asset_id or not self.media_type or not self.purpose:
            raise ValueError("Control reference needs asset ID, type and purpose")
        if len(self.sha256) != 64 or any(x not in "0123456789abcdef" for x in self.sha256):
            raise ValueError("Control reference requires a lowercase SHA256")


@dataclass(frozen=True, slots=True)
class ControlBundle:
    """Single control source for P, realizers, and track-scoped instructions."""

    instructions: bytes = b""
    reference_text: bytes = b""
    reference_origin: str | None = None
    tools_json: str = "[]"
    settings_json: str = "{}"
    references: tuple[ControlReference, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.instructions, bytes) or not isinstance(self.reference_text, bytes):
            raise TypeError("Instructions and reference text must be bytes")
        if not isinstance(self.references, tuple):
            raise TypeError("References must be an immutable tuple")
        settings = strict_json_loads(self.settings_json)
        if not isinstance(settings, dict):
            raise ValueError("Control settings require one canonical JSON object")

    def payload(self, *, track_override: Mapping[str, object] | None = None) -> bytes:
        return build_control_payload(
            seed=self.instructions,
            reference=self.reference_text,
            reference_origin=self.reference_origin,
            tools_json=self.tools_json,
            settings_json=self.settings_json,
            references=self.references,
            track_override=track_override,
        )


def _canonical_object_json(value: str | Mapping[str, object], what: str) -> bytes:
    if isinstance(value, str):
        value = strict_json_loads(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be a JSON object")
    return json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def build_control_payload(
    *,
    seed: bytes = b"",
    reference: bytes = b"",
    reference_origin: str | None = None,
    tools_json: str = "[]",
    settings_json: str = "{}",
    references: Sequence[ControlReference] = (),
    track_override: Mapping[str, object] | None = None,
) -> bytes:
    """Frame only controls whose reference has explicit causal provenance.

    References may come from external evidence or earlier history. Current or
    future targets and verifier outcomes are not accepted reference origins.
    This validates declared provenance; callers must supply the actual earlier
    evidence rather than relabeling target bytes as an external source.
    """
    if not isinstance(seed, bytes) or not isinstance(reference, bytes):
        raise TypeError("control seed and reference must be bytes")
    origins = {"external_evidence", "earlier_history"}
    if reference_origin is not None and (
        not isinstance(reference_origin, str) or reference_origin not in origins
    ):
        raise ValueError("control reference requires causal external_evidence or earlier_history")
    if reference and reference_origin is None:
        raise ValueError("control reference has no causal provenance")
    if not isinstance(tools_json, str):
        raise TypeError("tools_json must be a JSON string containing a tool list")
    tools = strict_json_loads(tools_json)
    if not isinstance(tools, list):
        raise ValueError("tools_json must contain a list")
    payload = seed
    if reference:
        payload += b"\n<reference>\n" + reference
    if tools:
        canonical = json.dumps(
            tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload += b"\n<tools>\n" + canonical
    settings = _canonical_object_json(settings_json, "control settings")
    if settings != b"{}":
        payload += b"\n<settings>\n" + settings
    if references:
        if any(not isinstance(reference, ControlReference) for reference in references):
            raise TypeError("Control references require validated ControlReference values")
        from dataclasses import asdict

        canonical = json.dumps(
            [asdict(x) for x in references], sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload += b"\n<assets>\n" + canonical
    if track_override is not None:
        override = _canonical_object_json(track_override, "track override")
        if override != b"{}":
            payload += b"\n<track>\n" + override
    return payload


def encode_controls(
    core: FacetedCore,
    payloads: Sequence[bytes],
    *,
    encoder_grad: bool = True,
) -> Tensor | None:
    """Encode independent controls; absent controls are exactly the neutral zero.

    An all-empty batch skips the encoder. Empty rows in a mixed batch have zero
    output and zero encoder gradient, matching the core's ``control=None`` path
    even when its learned control encoder has nonzero biases.
    """
    if not payloads:
        raise ValueError("cannot encode an empty control batch")
    if any(not isinstance(payload, bytes) for payload in payloads):
        raise TypeError("control payloads must be bytes")
    if not any(payloads):
        return None
    from megatoken.tensors import byte_tensor

    values, lengths = byte_tensor(payloads, core.initial_facets.device)
    encoded = core.encode_control(values, lengths=lengths, encoder_grad=encoder_grad)
    return encoded.masked_fill(lengths.eq(0)[:, None], 0)

def effective_event_payloads(
    session_payload: bytes,
    tracks: Sequence[str],
    requests: Sequence[tuple[str, int]],
) -> tuple[bytes, ...]:
    """Bind an event-local revision without revealing later requests.

    A request only affects its own event. This framing is byte-exact and shared
    by the trainer and a caller preparing controls for select_group/replay.
    """
    if len(tracks) != len(requests) or len(set(tracks)) != len(tracks):
        raise ValueError("Each unique track requires one effective request")
    if not isinstance(session_payload, bytes):
        raise TypeError("Session control must be bytes")
    result = []
    for track, (instruction, revision) in zip(tracks, requests, strict=True):
        if not track or not isinstance(instruction, str) or type(revision) is not int or revision < 0:
            raise ValueError("Invalid effective event request")
        suffix = json.dumps(
            {"track_id": track, "request": instruction, "revision": revision},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")
        result.append(session_payload + b"\n<event_request>\n" + suffix)
    return tuple(result)

def causal_step_payloads(episode, meridian_id: int, base: bytes,
                         tracks: Sequence[str],
                         requests: Sequence[tuple[str, int]]) -> tuple[bytes, ...]:
    """Only preceding published interpretation/request revisions are visible.

    This is shared by the grouped trainer and the offline serving task adapter.
    A later edit request, even on the same reusable track, is not an initial
    global control. Peer output targets and verification metadata are excluded.
    """
    if not isinstance(base, bytes):
        raise TypeError("Base session control must be bytes")
    available = [
        {
            "meridian": m.meridian_id,
            "track": e.track_id,
            "modality": e.modality,
            "request": e.request,
            "revision": e.request_revision,
        }
        for m in episode.meridians if m.meridian_id < meridian_id
        for e in m.events if e.request
    ]
    if available:
        framed = json.dumps(available, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8")
        base += b"\\n<available_requests>\\n" + framed
    return effective_event_payloads(base, tracks, requests)



# The same resolver is used by dataset compilation, grouped training and
# serving. An input's descriptor is publicly available once its meridian is
# observed; labels on future output peers are never consulted.
_DEFAULT_EVENT_INSTRUCTIONS = {
    "text": "Produce the requested text using available causal evidence.",
    "audio": "Produce speech or sound according to the event request.",
    "image": "Produce the requested still image.",
    "video": "Produce the requested temporal video.",
    "tool": "Execute the requested tool action from available evidence.",
    "evidence": "Interpret the available evidence.",
}


def resolve_event_control(episode: object, event: object) -> bytes:
    """Resolve exact event-time control bytes, never supervision metadata.

    Precedence: episode seed -> track default -> current event request.
    The reference IDs designate *already committed* events in shared KV, not
    hash-string substitutes for extracted voice/image features.
    """
    tracks = {track.track_id: track for track in episode.tracks}
    if event.track_id not in tracks:
        raise ValueError("Missing event track descriptor")
    track = tracks[event.track_id]
    available = {}
    seen = set()
    for meridian in episode.meridians:
        if meridian.meridian_id >= event.meridian_id:
            break
        for prior in meridian.events:
            seen.add(prior.record_id)
            if prior.role != "assistant":
                prior_track = tracks[prior.track_id]
                available[prior.track_id] = {
                    "modality": prior.modality,
                    "instruction": prior.instruction or prior_track.description
                    or _DEFAULT_EVENT_INSTRUCTIONS.get(prior.modality, ""),
                    "revision": prior.request_revision,
                }
    if any(ref not in seen for ref in event.reference_event_ids):
        raise ValueError("Cannot condition an event on a future reference")
    settings = strict_json_loads(track.settings_json)
    settings.update(strict_json_loads(event.settings_json))
    instruction = event.instruction or track.description or _DEFAULT_EVENT_INSTRUCTIONS[event.modality]
    override = {
        "track_id": event.track_id,
        "modality": event.modality,
        "instruction": instruction,
        "revision": event.request_revision,
        "meridian_id": event.meridian_id,
        "settings": settings,
        "reference_event_ids": list(event.reference_event_ids),
        "available_input_tracks": available,
    }
    return build_control_payload(
        seed=episode.control_seed,
        reference=episode.control_reference,
        reference_origin=episode.reference_origin,
        tools_json=episode.tools_json,
        track_override=override,
    )


def resolve_meridian_controls(episode: object, meridian: object) -> tuple[bytes, ...]:
    """Physical-order-independent event requests for an atomic generated group."""
    return tuple(resolve_event_control(episode, event) for event in meridian.events)
