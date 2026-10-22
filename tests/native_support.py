"""Native action tapes for tests: the single-event group tape a rollout records for one draw."""

from __future__ import annotations

import numpy as np

from megatoken.model.multitrack import GROUP_TAPE_VERSION, GroupActionTape


def group_tape(
    codec,
    payload: bytes,
    *,
    emitted_ids: tuple[int, ...] | None = None,
    valid: bool = True,
    terminal: bool = False,
    facets=(0.0,),
    shape: tuple[int, int] = (1, 1),
    control: bytes = b"",
    spec_fingerprint: str = "fixture",
    effort: int = 6,
    track: str = "assistant",
    role: int = 2,
    log_prob: float = 0.0,
) -> GroupActionTape:
    """A one-event tape of ``payload``; ``facets`` are its raw coordinates (FP32 values or bytes)."""
    raw = facets if isinstance(facets, bytes) else np.asarray(facets, dtype="<f4").tobytes()
    return GroupActionTape(
        version=GROUP_TAPE_VERSION,
        meridian_id=0,
        tracks=(track,),
        roles=(role,),
        controls=(control,),
        spec_fingerprints=(spec_fingerprint,),
        codec_fingerprint=codec.fingerprint,
        facet_shape=(1, *shape),
        raw_f32=raw,
        behavior_log_prob=log_prob,
        payloads=(payload,),
        emitted_ids=(tuple(codec.encode(payload)) if emitted_ids is None else emitted_ids,),
        valid=(valid,),
        terminal=(terminal,),
        modalities=("text",),
        effort=effort,
        event_log_probs=(log_prob,),
        renderer_seeds=(0,),
        pack_fingerprints=("none",),
    )


def as_group(tape, codec, *, track: str = "assistant") -> GroupActionTape:
    """The group tape of a draw of the single-track sampler (for tests of the single-track state)."""
    return group_tape(
        codec,
        tape.payload,
        emitted_ids=tape.emitted_ids,
        valid=tape.valid,
        terminal=tape.terminal,
        facets=bytes(tape.facets) if isinstance(tape.facets, bytes) else tuple(tape.facets),
        shape=tape.facet_shape,
        control=tape.control_payload,
        spec_fingerprint=tape.spec_fingerprint,
        effort=tape.effort,
        track=track,
        role=tape.role_id,
        log_prob=tape.behavior_log_prob,
    )


def tiny_bundle(codec):
    """What ``ModelTeacher`` needs of a served bundle, around a tiny untrained model."""
    from types import SimpleNamespace

    from megatoken.data.chunking import ChunkingConfig
    from megatoken.generation.policy import FacetedGenerator
    from megatoken.model import FacetedCore, ModelConfig

    model = FacetedGenerator(FacetedCore(ModelConfig.tiny()), codec)
    return SimpleNamespace(
        model=model,
        codec=codec,
        chunking=ChunkingConfig().with_hard_limit(model.config.max_capacity),
        manifest={"model_sha256": "a" * 64},
    )
