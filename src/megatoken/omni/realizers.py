"""The one media realization path: frozen pretrained generators, conditioned on raw Z.

Each modality has exactly one resident realizer, loaded from the locked pack
manifest (``megatoken.omni.loader``):

- audio: Pocket TTS FlowLM + its Mimi codec (24 kHz continuous latents);
- image: Sana DiT + its DC-AE;
- video: Sana-Video DiT + its LTX-2 video VAE.

The pretrained generator and codec are frozen, inference-mode state held
*outside* the module tree: they never enter checkpoints, optimizer masters,
DiLoCo fragments or ``train()``/``eval()`` switching. The only registered
(trainable) weights condition the donor on all selected raw facets. The
donor's own text/voice conditioners are never called.

A realizer owns its codec in both directions: ``encode`` produces training
targets from decoded original media (``megatoken.data.media_prepare``),
``loss`` is the donor's own flow/diffusion objective on those latents, and
``realize`` samples and decodes new media from raw facets.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import wraps
from pathlib import Path
from threading import Lock, RLock
from typing import Any
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

_PIPELINE_LOCKS = WeakKeyDictionary()
_PIPELINE_LOCKS_OWNER = Lock()


def _serialized_pipeline(method):
    """Diffusion schedulers and VAE caches belong to one active call at a time.

    Weak ownership keeps locks out of model state and permits model collection.
    Different resident decoders remain independently executable.
    """

    @wraps(method)
    def call(self, *args, **kwargs):
        with _PIPELINE_LOCKS_OWNER:
            lock = _PIPELINE_LOCKS.setdefault(self, RLock())
        with lock:
            return method(self, *args, **kwargs)

    return call


def masked_square_error(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """One event-mean over real latent coordinates (padded frames excluded)."""
    if prediction.shape != target.shape:
        raise ValueError("Realizer prediction/target geometry mismatch")
    error = (prediction.float() - target.float()).square()
    if mask is None:
        return error.mean()
    mask = mask.to(device=error.device, dtype=torch.bool)
    while mask.ndim < error.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def _encoded(output: Any) -> Tensor:
    """The deterministic latent of a Diffusers VAE encode (posterior mode)."""
    if hasattr(output, "latent_dist"):
        return output.latent_dist.mode()
    for name in ("latent", "latents"):
        if isinstance(getattr(output, name, None), Tensor):
            return getattr(output, name)
    raise TypeError("Unrecognized VAE encoder output; pin the matching implementation")


class FacetCondition(nn.Module):
    """All raw facets -> donor conditioning tokens, each slot kept distinct."""

    def __init__(self, facet_count: int, facet_width: int, donor_width: int):
        super().__init__()
        self.facet_count, self.facet_width = facet_count, facet_width
        self.norm = nn.LayerNorm(facet_width)
        self.project = nn.Linear(facet_width, donor_width)
        self.slot = nn.Parameter(torch.zeros(facet_count, donor_width))

    def forward(self, raw: Tensor) -> Tensor:
        if raw.ndim != 3 or raw.shape[1:] != (self.facet_count, self.facet_width):
            raise ValueError("Realizer condition must be one complete raw facet block")
        if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
            raise ValueError("Raw facets must be finite floating point")
        weight = self.project.weight
        return self.project(self.norm(raw.to(weight.dtype))) + self.slot.to(weight.dtype)


class Realizer(nn.Module):
    """Common identity and frozen-donor ownership for the resident realizers."""

    name: str
    modality: str
    # Attribute holding the frozen donor root object (moved by ``to_device``).
    _donor: str
    # sha256 over the locked pack files; set by the loader.
    checkpoint_identity: str = ""

    def _hold(self, **donor: Any) -> None:
        """Keep frozen pretrained state outside the module tree, in inference mode."""
        for key, value in donor.items():
            if isinstance(value, nn.Module):
                value.eval().requires_grad_(False)
            object.__setattr__(self, key, value)

    @property
    def identity(self) -> str:
        if not self.checkpoint_identity:
            raise ValueError("Realizer was not installed from a locked pack manifest")
        return f"{self.name}:{self.checkpoint_identity}"

    def to_device(self, device: torch.device | str) -> Realizer:
        """Move the frozen donor with the trainable condition (``Module.to`` skips it)."""
        getattr(self, self._donor).to(device)
        return self.to(device)


POCKET_RESOURCE_KEYS = frozenset(
    {"weights_path", "weights_path_without_voice_cloning", "tokenizer_path"}
)


def resolve_pocket_config(config: Path) -> tuple[dict, list[Path]]:
    """A Pocket YAML with every resource as an absolute local path, and those paths.

    Relative paths are relative to the YAML's directory; remote resources are refused.
    """
    import yaml

    if not config.is_file():
        raise FileNotFoundError("Pocket YAML is not a local pinned file")
    mapping = yaml.safe_load(config.read_text("utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Pocket config must be a YAML mapping")
    resources: list[Path] = []

    def resolve(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in POCKET_RESOURCE_KEYS and isinstance(value, str):
                    if value.startswith(("hf://", "https://", "http://")):
                        raise ValueError(f"Remote Pocket resource in {key}")
                    resource = Path(value)
                    resource = (
                        resource if resource.is_absolute() else config.parent / resource
                    ).resolve()
                    resources.append(resource)
                    node[key] = str(resource)
                else:
                    resolve(value)
        elif isinstance(node, list):
            for value in node:
                resolve(value)

    resolve(mapping)
    return mapping, resources


class PocketRealizer(Realizer):
    """Pocket's pretrained FlowLM/flow net and matching Mimi, prefixed by raw Z."""

    name, modality, _donor = "pocket", "audio", "pocket"

    def __init__(self, pocket_model: Any, facet_count: int, facet_width: int):
        super().__init__()
        # Original token conditioning is teacher-only; drop its embedding table.
        pocket_model.flow_lm.conditioner = None
        self._hold(pocket=pocket_model, flow_lm=pocket_model.flow_lm, mimi=pocket_model.mimi)
        self.condition = FacetCondition(facet_count, facet_width, self.flow_lm.dim)
        self.frame_rate = float(self.mimi.frame_rate)
        self.sample_rate = int(self.mimi.sample_rate)
        self.latent_width = int(self.flow_lm.ldim)
        if self.latent_width != self.mimi.quantizer.dimension:
            raise ValueError("Pocket flow/Mimi continuous latent geometry differs")

    @classmethod
    def from_checkpoint(
        cls, *, config: str, checkpoint: str | None, facet_count: int, facet_width: int
    ):
        """``config`` must be the locked local Pocket YAML (no remote resources)."""
        import tempfile

        import yaml
        from pocket_tts import TTSModel

        resolved, _ = resolve_pocket_config(Path(config))
        # pocket_tts resolves relative resource paths against the process working
        # directory; hand it the same files the lock verified, as absolute paths.
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / "pocket.yaml"
            local.write_text(yaml.safe_dump(resolved), encoding="utf-8")
            model = TTSModel.load_model(config=str(local), checkpoint=checkpoint, quantize=False)
        return cls(model, facet_count, facet_width)

    @torch.no_grad()
    def encode(self, waveform: Tensor) -> Tensor:
        """24 kHz mono [B, samples] in [-1, 1] -> continuous Mimi latents [B, frames, D]."""
        if waveform.ndim != 2:
            raise ValueError("Pocket encodes mono [batch, samples] at its native rate")
        dtype = next(self.mimi.parameters()).dtype
        return self.mimi.encode_to_latent(waveform[:, None].to(dtype)).float()

    def loss(self, raw: Tensor, latents: Tensor, mask: Tensor | None, generator=None) -> Tensor:
        """Pocket's own flow objective under teacher-forced acoustic prefixes."""
        condition = self.condition(raw)
        if latents.ndim != 3 or latents.shape[-1] != self.latent_width:
            raise ValueError("Pocket target must contain continuous Mimi latents")
        clean = (latents - self.flow_lm.emb_mean) / self.flow_lm.emb_std.clamp_min(1e-6)
        clean = clean.to(condition.dtype)
        b, frames, channels = clean.shape
        if b != condition.shape[0] or frames < 1:
            raise ValueError("Mimi latent batch or frame count mismatch")
        bos = torch.full((b, 1, channels), float("nan"), device=clean.device, dtype=clean.dtype)
        prefix = torch.cat((bos, clean[:, :-1]), dim=1)
        embedded = torch.where(torch.isnan(prefix), self.flow_lm.bos_emb, prefix)
        hidden = self.flow_lm.transformer(
            torch.cat((condition, self.flow_lm.input_linear(embedded)), dim=1), model_state=None
        )
        if self.flow_lm.out_norm is not None:
            hidden = self.flow_lm.out_norm(hidden)
        hidden = hidden[:, -frames:].float()
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        ).float()
        clean = clean.float()
        flow_type = self.flow_lm.flow_type
        if flow_type == "flow_matching":
            t = torch.rand((b, frames, 1), device=clean.device, generator=generator)
            sig_min = 0.001
            x_t = (1 - (1 - sig_min) * t) * noise + t * clean
            prediction = self.flow_lm.flow_net(hidden, t, x_t)
            return masked_square_error(prediction, clean - (1 - sig_min) * noise, mask)
        if flow_type != "lsd":
            raise ValueError("Unsupported Pocket flow objective: use LSD or OT")
        # Pocket's LSD diagonal term plus its stop-gradient self-distillation term.
        t = torch.sigmoid(
            torch.randn((b, frames, 1), device=clean.device, generator=generator) + 0.4
        )
        x_t = t * clean + (1 - t) * noise
        diagonal = self.flow_lm.flow_net(hidden, t, t, x_t)
        logits = torch.randn((b, frames, 2), device=clean.device, generator=generator) + 0.4
        s = torch.sigmoid(logits.min(dim=-1, keepdim=True).values)
        end = torch.sigmoid(logits.max(dim=-1, keepdim=True).values)
        x_s = s * clean + (1 - s) * noise

        def field(s0, t0, x0):
            return self.flow_lm.flow_net(hidden, s0, t0, x0)

        vt, dvdt = torch.func.jvp(
            field, (s, end, x_s), (torch.zeros_like(s), torch.ones_like(end), torch.zeros_like(x_s))
        )
        with torch.no_grad():
            target_velocity = self.flow_lm.flow_net(hidden, end, end, x_s + (end - s) * vt)
        return 0.75 * masked_square_error(diagonal, clean - noise, mask) + 0.25 * (
            masked_square_error(vt + (end - s) * dvdt, target_velocity, mask)
        )

    @torch.no_grad()
    def latent(self, raw: Tensor, *, max_frames: int, steps: int, generator=None) -> Tensor:
        """Free rollout from raw Z only; no target-conditioned acoustic prefix."""
        from pocket_tts.modules.stateful_module import increment_steps, init_states

        condition = self.condition(raw)
        b = raw.shape[0]
        if not 1 <= max_frames <= 1500 or not 1 <= steps <= 128:
            raise ValueError("Unbounded speech rollout")
        states = init_states(self.flow_lm, b, condition.shape[1] + max_frames + 4)
        previous = torch.full(
            (b, 1, self.latent_width), float("nan"), device=raw.device, dtype=condition.dtype
        )
        outputs = []
        for t in range(max_frames):
            # The stateful FlowLM sees the facet prefix only on step zero.
            embeddings = condition if t == 0 else condition[:, :0]
            devices = [raw.device.index] if raw.is_cuda and raw.device.index is not None else []
            with torch.random.fork_rng(devices=devices, enabled=generator is not None):
                if generator is not None:
                    # The draw must happen on the generator's own device.
                    seed = torch.randint(
                        0, 2**31 - 1, (1,), generator=generator, device=generator.device
                    )
                    torch.manual_seed(int(seed))
                next_latent, eos = self.flow_lm._sample_next_latent(
                    previous,
                    embeddings,
                    states,
                    steps,
                    self.pocket.temp,
                    self.pocket.noise_clamp,
                    self.pocket.eos_threshold,
                )
            increment_steps(self.flow_lm, states, increment=previous.shape[1] + embeddings.shape[1])
            outputs.append(next_latent[:, None])
            previous = next_latent[:, None]
            if t >= 3 and bool(eos.all()):
                break
        return torch.cat(outputs, dim=1) * self.flow_lm.emb_std + self.flow_lm.emb_mean

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        from pocket_tts.modules.stateful_module import init_states

        ratio = self.mimi.encoder_frame_rate / self.mimi.frame_rate
        if ratio != int(ratio) or ratio < 1:
            raise ValueError("Pocket Mimi requires an integer encoder/latent frame ratio")
        # Mimi's decoder advances at encoder-frame rate, not the 12.5 Hz latent
        # rate: a 16x ratio needs 16x the decoder-state capacity.
        state = init_states(self.mimi, latents.shape[0], latents.shape[1] * int(ratio) + 16)
        waveform = self.mimi.decode_from_latent(latents, state)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("Pocket Mimi must decode mono waveform [B, 1, samples]")
        return waveform[:, 0]

    def realize(self, raw: Tensor, spec, generator=None) -> Tensor:
        """[B, spec.sample_count] at 24 kHz; renderer EOS may end speech early (zero pad)."""
        # Exact rational ceiling: float division rounds some exact multiples up a frame.
        frames = max(
            1, math.ceil(Fraction(spec.sample_count) * Fraction(self.frame_rate) / self.sample_rate)
        )
        audio = self.decode(
            self.latent(raw, max_frames=frames, steps=spec.steps, generator=generator)
        )
        if audio.shape[1] < spec.sample_count:
            audio = nn.functional.pad(audio, (0, spec.sample_count - audio.shape[1]))
        return audio[:, : spec.sample_count]


class SanaRealizer(Realizer):
    """Sana (image) or Sana-Video DiT with its matched frozen VAE, prompted by raw Z."""

    _donor = "pipe"

    @_serialized_pipeline
    def to_device(self, device):
        # Diffusers' generic pipeline mover iterates an unordered component
        # set. Keep large allocations in a reproducible order on shared RAM.
        self.transformer.to(device)
        self.vae.to(device)
        return self.to(device)

    def __init__(self, pipeline: Any, modality: str, facet_count: int, facet_width: int):
        super().__init__()
        if modality not in ("image", "video"):
            raise ValueError("Sana realizes still images or video")
        self.name = "sana" if modality == "image" else "sana_video"
        self.modality = modality
        # The donor text encoder is teacher-only; facets replace prompt embeddings.
        pipeline.text_encoder = None
        pipeline.tokenizer = None
        self._hold(pipe=pipeline, transformer=pipeline.transformer, vae=pipeline.vae)
        self.condition = FacetCondition(
            facet_count, facet_width, int(pipeline.transformer.config.caption_channels)
        )

    @classmethod
    def from_checkpoint(cls, *, snapshot: str, modality: str, facet_count: int, facet_width: int):
        from diffusers import SanaPipeline, SanaVideoPipeline

        if not Path(snapshot).is_dir():
            raise FileNotFoundError("Require a pinned local Diffusers snapshot")
        klass = SanaPipeline if modality == "image" else SanaVideoPipeline
        # Facets replace the prompt: the Gemma-2 text encoder is never loaded,
        # so packs need not contain it (scripts/acquire_packs.py).
        pipeline = klass.from_pretrained(
            snapshot,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            text_encoder=None,
            tokenizer=None,
        )
        expected = "AutoencoderDC" if modality == "image" else "AutoencoderKLLTX2Video"
        if type(pipeline.vae).__name__ != expected:
            raise ValueError(f"Sana {modality} requires its matching {expected}")
        return cls(pipeline, modality, facet_count, facet_width)

    def _prompt(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        embeds = self.condition(raw)
        return embeds, torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.bool)

    def _grid(self) -> tuple[int, int]:
        """(spatial multiple of H and W, frame step) this VAE and DiT accept."""
        spatial = getattr(self.vae, "spatial_compression_ratio", None) or getattr(
            self.pipe, "vae_scale_factor", None
        )
        patch = self.transformer.config.patch_size
        patch = (1, patch, patch) if isinstance(patch, int) else tuple(patch)
        if self.modality == "image":
            return int(spatial) * patch[-1], 1
        temporal = int(self.vae.temporal_compression_ratio)
        return int(spatial) * patch[-1], temporal * patch[0]

    def fit(self, pixels: Tensor) -> Tensor:
        """Resize (and, for video, trim frames of) RGB media to the nearest geometry
        the VAE and DiT accept: sides a multiple of the spatial grid, and video
        frame counts of the form ``1 + k * frame_step``. Image [B, 3, H, W];
        video [B, F, 3, H, W]."""
        multiple, step = self._grid()
        height, width = pixels.shape[-2:]
        size = tuple(max(multiple, round(side / multiple) * multiple) for side in (height, width))
        frames = pixels.shape[1] if self.modality == "video" else 1
        frames = 1 + (frames - 1) // step * step
        flat = pixels[:, :frames].flatten(0, 1) if self.modality == "video" else pixels
        if size != (height, width):
            flat = nn.functional.interpolate(
                flat.float(), size=size, mode="bilinear", antialias=True, align_corners=False
            ).clamp(0, 1)
        return (
            flat.view(pixels.shape[0], frames, *flat.shape[1:])
            if self.modality == "video"
            else flat
        )

    @_serialized_pipeline
    @torch.no_grad()
    def encode(self, pixels: Tensor) -> Tensor:
        """RGB in [0, 1] already ``fit``: image [B, 3, H, W] -> [B, C, h, w] (scaled);
        video [B, F, 3, H, W] -> time-major normalized [B, F', C, h, w]."""
        x = pixels.to(next(self.vae.parameters()).dtype) * 2 - 1
        if self.modality == "image":
            latent = _encoded(self.vae.encode(x, return_dict=True))
            return (latent * float(self.vae.config.scaling_factor)).float()
        latent = _encoded(self.vae.encode(x.permute(0, 2, 1, 3, 4), return_dict=True))
        channels = int(self.vae.config.latent_channels)
        mean = self.vae.latents_mean.view(1, channels, 1, 1, 1).to(latent)
        std = self.vae.latents_std.view(1, channels, 1, 1, 1).to(latent)
        return ((latent - mean) / std).permute(0, 2, 1, 3, 4).float()

    def loss(self, raw: Tensor, latents: Tensor, mask: Tensor | None, generator=None) -> Tensor:
        """The checkpoint scheduler's own convention (flow: sigma=1 is noise)."""
        if self.modality == "video":
            # Stored time-major [B, F, C, h, w]; the DiT reads [B, C, F, h, w].
            latents = latents.permute(0, 2, 1, 3, 4)
            mask = None if mask is None else mask.view(mask.shape[0], 1, -1, 1, 1)
        embeds, attention = self._prompt(raw)
        # Sana and Sana-Video ship DPMSolverMultistepScheduler in its flow mode:
        # x_sigma = (1 - sigma) x0 + sigma * noise, the model predicts noise - x0,
        # timestep = shifted sigma * num_train_timesteps (diffusers 0.41).
        config = self.pipe.scheduler.config
        if (
            config.get("prediction_type") != "flow_prediction"
            or not config.get("use_flow_sigmas")
            or config.get("use_dynamic_shifting")
            or config.num_train_timesteps <= 0
        ):
            raise ValueError("Sana realizers need their static-shift flow scheduler")
        b = latents.shape[0]
        noise = torch.randn(
            latents.shape, dtype=latents.dtype, device=latents.device, generator=generator
        )
        # Uniform continuous training time, independent of the mutable inference
        # timesteps, with the scheduler's static shift.
        sigma = torch.rand((b,), device=latents.device, generator=generator).to(latents.dtype)
        shift = float(config.flow_shift)
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError("Invalid donor scheduler time shift")
        sigma = (shift * sigma) / (1.0 + (shift - 1.0) * sigma)
        timestep = sigma * int(config.num_train_timesteps)
        sigma = sigma.view(b, *([1] * (latents.ndim - 1)))
        noisy = sigma * noise + (1.0 - sigma) * latents
        if self.modality == "image":
            timestep = timestep * self.transformer.config.timestep_scale
        output = self.transformer(
            noisy.to(self.transformer.dtype),
            encoder_hidden_states=embeds.to(self.transformer.dtype),
            encoder_attention_mask=attention,
            timestep=timestep,
            return_dict=False,
        )[0].float()
        if output.shape[1] == latents.shape[1] * 2:
            output = output.chunk(2, dim=1)[0]  # learned-sigma head
        return masked_square_error(output, noise - latents, mask)

    @_serialized_pipeline
    @torch.no_grad()
    def realize(self, raw: Tensor, spec, generator=None) -> Tensor:
        """Decoded RGB [B, frames, 3, H, W] through this pipeline's own VAE path."""
        multiple, step = self._grid()
        if spec.height % multiple or spec.width % multiple or (spec.frames - 1) % step:
            raise ValueError(
                f"Sana geometry needs sides a multiple of {multiple} and 1 + k*{step} frames"
            )
        embeds, attention = self._prompt(raw)
        kwargs = dict(
            prompt_embeds=embeds,
            prompt_attention_mask=attention,
            # Precomputed embeddings replace the pipeline's text prompts (whose
            # negative default "" would otherwise conflict with them).
            # No classifier-free guidance: the condition is never trained with a
            # dropped (null) variant, so there is no unconditional branch to
            # extrapolate from. Scale 1.0 makes the pipeline skip it entirely.
            negative_prompt=None,
            guidance_scale=1.0,
            num_inference_steps=spec.steps,
            height=spec.height,
            width=spec.width,
            generator=generator,
            output_type="pt",
            return_dict=False,
            use_resolution_binning=False,
        )
        if self.modality == "video":
            kwargs["frames"] = spec.frames
        output = self.pipe(**kwargs)[0]
        if self.modality == "image":
            if output.ndim != 4 or output.shape[1] != 3:
                raise ValueError("Sana RGB image must be [B, 3, H, W]")
            output = output[:, None]
        elif output.ndim != 5 or output.shape[2] != 3:
            raise ValueError("Sana-Video decoded clip must be [B, frames, 3, H, W]")
        # The pipeline may round the request to its own grid; the spec is a contract.
        if output.shape[0] != raw.shape[0] or output.shape[1] != spec.frames:
            raise ValueError("Sana output frame count differs from the requested geometry")
        if output.shape[-2:] != (spec.height, spec.width):
            raise ValueError("Sana output size differs from the requested geometry")
        return output
