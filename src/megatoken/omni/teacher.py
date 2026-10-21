"""Pocket teacher-side exact predecoder latent capture, with no native R bypass.

The teacher may consume transcript and an approved separate voice reference.
Only its produced *continuous native Mimi* latents are cached as supervision.
The resulting metadata must never be reclassified as a P prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PocketTeacherTrace:
    waveform: Tensor
    normalized_latents: Tensor
    mimi_latents: Tensor
    eos_flags: tuple[bool, ...]

    def __post_init__(self):
        if self.waveform.ndim != 2 or self.waveform.shape[0] != 1:
            raise ValueError("Teacher waveform must be [1, native mono samples]")
        if (
            self.normalized_latents.ndim != 2
            or self.mimi_latents.shape != self.normalized_latents.shape
            or self.normalized_latents.shape[0] < 1
            or len(self.eos_flags) != self.mimi_latents.shape[0]
        ):
            raise ValueError("Teacher trace must contain an aligned acoustic sequence")
        if not bool(torch.isfinite(self.waveform).all()) or not bool(
            torch.isfinite(self.mimi_latents).all()
        ):
            raise ValueError("Nonfinite teacher waveform or acoustic trajectory")


@torch.no_grad()
def capture_pocket_teacher(donor, *, voice_state, spoken_text: str) -> PocketTeacherTrace:
    """Capture every actual FlowLM decode output before Mimi state/mixing.

    A Pocket checkpoint uses normalized acoustic coordinates inside FlowLM;
    R supervision is the exact unnormalized Mimi decoder input. No forced
    alignment between unrelated stochastic teacher waveforms is fabricated.
    """
    if not isinstance(spoken_text, str) or not spoken_text.strip():
        raise ValueError("Teacher requires the explicitly supplied spoken text")
    flow = donor.flow_lm
    original = flow._sample_next_latent
    frames: list[Tensor] = []
    eos_bits: list[bool] = []

    def record(*args, **kwargs):
        value, eos = original(*args, **kwargs)
        if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] != flow.ldim:
            raise ValueError("Pocket teacher emitted unexpected acoustic geometry")
        frames.append(value[0].detach().float().clone())
        eos_bits.append(bool(eos.all()))
        return value, eos

    flow._sample_next_latent = record
    try:
        raw_audio = donor.generate_audio(
            voice_state,
            spoken_text,
            copy_state=True,
        )
    finally:
        flow._sample_next_latent = original

    if not frames:
        raise ValueError("Pocket teacher emitted no audio latents")
    if raw_audio.ndim == 1:
        waveform = raw_audio[None]
    elif raw_audio.ndim == 2 and raw_audio.shape[0] == 1:
        waveform = raw_audio
    else:
        raise ValueError("Pocket teacher must render mono waveform")
    normalized = torch.stack(frames, dim=0)
    decoder = normalized * flow.emb_std.float() + flow.emb_mean.float()
    return PocketTeacherTrace(
        waveform.detach().float().cpu().contiguous(),
        normalized.cpu().contiguous(),
        decoder.cpu().contiguous(),
        tuple(eos_bits),
    )
