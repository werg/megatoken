"""Pinned pretrained edge packs, directly conditioned on all selected raw facets.

Only one [B,F,D] selected Z enters any native student. Original text
conditioners, voice prompts, previous media and C/KV are not arguments.
Pack loaders require local, hash-verified weights; training/serving never
implicitly download checkpoints.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class PackSpec:
    pack_id: str
    family: str
    local_path: str
    weights_sha256: str
    revision: str
    codec_id: str
    conditioner_width: int
    steps: int = 16
    language: str = ""
    train_backbone: bool = False

    def __post_init__(self) -> None:
        if self.family not in {"pocket", "sana_image", "sana_video"}:
            raise ValueError("Only the selected pretrained native families are supported")
        if (not self.pack_id or not self.codec_id or not self.revision
                or len(self.weights_sha256) != 64 or
                any(ch not in "0123456789abcdef" for ch in self.weights_sha256)):
            raise ValueError("Pack needs pinned generator, exact codec and weight digest")
        if self.conditioner_width < 1 or not 1 <= self.steps <= 128:
            raise ValueError("Invalid native conditioner width or sampler steps")
        if self.family == "pocket" and not self.language:
            raise ValueError("Pocket speech requires an explicitly verified language profile")


def directory_sha256(folder: str | Path) -> str:
    """Hash file paths, sizes and bytes to bind exported submodules and weights."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ValueError("Native pack path must be an existing local directory")
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Native pack directory is empty")
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(folder).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


class FullFacetConditioner(nn.Module):
    """Project every one of the F facets to donor prefix width, preserving slots."""

    def __init__(self, facet_count: int, width: int, donor_width: int):
        super().__init__()
        self.facet_count = facet_count
        self.width = width
        self.to_donor = nn.Linear(width, donor_width)
        self.slot_position = nn.Parameter(torch.zeros(facet_count, donor_width))
        self.norm = nn.LayerNorm(donor_width)

    def forward(self, raw: Tensor) -> Tensor:
        if (raw.ndim != 3 or raw.shape[1:] != (self.facet_count, self.width)
                or not torch.isfinite(raw).all()):
            raise ValueError("Donor conditioning must consume exactly one complete raw Z")
        return self.norm(self.to_donor(raw.to(self.to_donor.weight.dtype))
                         + self.slot_position[None])


def masked_flow_mse(prediction: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """One event-mean over real coordinates, excluding padded codec tiles."""
    if prediction.shape != target.shape:
        raise ValueError("Donor velocity differs from its native latent coordinates")
    error = (prediction.float() - target.float()).square()
    if mask is None:
        return error.mean()
    valid = mask.to(device=error.device, dtype=torch.bool).expand_as(error)
    if not bool(valid.any()):
        raise ValueError("Media target has no valid codec coordinates")
    return error.masked_fill(~valid, 0).sum() / valid.sum()


class DiffusersDenoiser(nn.Module):
    """Narrow tensor-level donor bridge. No donor text encoder is called."""

    def __init__(self, transformer: nn.Module):
        super().__init__()
        self.transformer = transformer
        signature = inspect.signature(transformer.forward)
        self.args = set(signature.parameters)
        required = {"hidden_states", "encoder_hidden_states", "timestep"}
        if not required.issubset(self.args):
            raise ValueError("Donor transformer lacks the expected latent/conditioning API")

    def forward(self, noisy: Tensor, timestep: Tensor, conditioning: Tensor) -> Tensor:
        kwargs = dict(
            hidden_states=noisy,
            timestep=timestep,
            encoder_hidden_states=conditioning,
        )
        if "encoder_attention_mask" in self.args:
            kwargs["encoder_attention_mask"] = torch.ones(
                conditioning.shape[:2], device=noisy.device, dtype=torch.bool,
            )
        if "return_dict" in self.args:
            kwargs["return_dict"] = False
        output = self.transformer(**kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        elif hasattr(output, "sample"):
            output = output.sample
        if not isinstance(output, Tensor):
            raise ValueError("Native denoiser did not return a latent velocity tensor")
        return output


class MatchingVAEDecoder(nn.Module):
    """The checkpoint-paired DC-AE / LTX-2 VAE, never a cross-family codec."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae
        self.scaling = float(getattr(getattr(vae, "config", None), "scaling_factor", 1.0))
        if not math.isfinite(self.scaling) or self.scaling <= 0:
            raise ValueError("Codec scaling factor must be finite and positive")

    def forward(self, latents: Tensor) -> Tensor:
        output = self.vae.decode(latents.to(self._dtype()) / self.scaling,
                                 return_dict=False)
        value = output[0] if isinstance(output, (tuple, list)) else output.sample
        if not isinstance(value, Tensor):
            raise ValueError("Matched VAE decoder returned no tensor")
        return value

    def _dtype(self) -> torch.dtype:
        return next(self.vae.parameters()).dtype


class NativeLatentFlow(nn.Module):
    """Donor flow/DiT initialized from a pretrained pack, with trainable prefix only.

    This module implements velocity-targeted diffusion in the donor's own
    latent space. A pack using a different time/velocity convention must use
    a dedicated bridge rather than silently accepting the wrong objective.
    """

    def __init__(self, *, facet_count: int, facet_width: int, pack: PackSpec,
                 denoiser: nn.Module, decoder: nn.Module):
        super().__init__()
        if pack.family not in {"sana_image", "sana_video"}:
            raise ValueError("NativeLatentFlow is only for pinned Sana families")
        self.pack_id, self.codec_id = pack.pack_id, pack.codec_id
        self.conditioner = FullFacetConditioner(
            facet_count, facet_width, pack.conditioner_width
        )
        self.denoiser = denoiser
        self.decoder = decoder
        self.steps = pack.steps
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.denoiser.parameters():
            parameter.requires_grad_(pack.train_backbone)

    def predict_velocity(self, noisy: Tensor, time: Tensor, raw: Tensor) -> Tensor:
        cond = self.conditioner(raw)
        return self.denoiser(noisy, time, cond)

    def flow_loss(self, raw: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
        if target.shape[0] != raw.shape[0] or not target.is_floating_point():
            raise ValueError("Native flow target must have one codec tensor per raw event")
        # Linear interpolant convention: U_s = (1-s)eps+sU; v=U-eps.
        time = torch.rand(target.shape[0], device=target.device)
        epsilon = torch.randn_like(target.float())
        view_shape = (target.shape[0],) + (1,) * (target.ndim - 1)
        s = time.reshape(view_shape)
        mixed = (1.0 - s) * epsilon + s * target.float()
        predicted = self.predict_velocity(mixed.to(target.dtype), time, raw)
        return masked_flow_mse(predicted, target.float() - epsilon, mask)

    def sample(self, raw: Tensor, spec: Any, *, noise: Tensor | None = None,
               seed: int | None = None) -> Tensor:
        if spec.codec_id != self.codec_id or spec.pack_id != self.pack_id:
            raise ValueError("Native codec and generator must match exactly")
        if noise is None:
            if seed is None:
                raise ValueError("Native free rollout requires explicit renderer randomness")
            gen = torch.Generator(device=raw.device)
            gen.manual_seed(seed)
            noise = torch.randn((raw.shape[0], *spec.latent_shape),
                                device=raw.device, dtype=torch.float32, generator=gen)
        elif noise.shape != (raw.shape[0], *spec.latent_shape):
            raise ValueError("Renderer noise does not match declared native coordinates")
        current = noise
        # Complete free trajectory. Donor-specific conventions are versioned by pack.
        for i in range(self.steps):
            time = torch.full((raw.shape[0],), i / self.steps, device=raw.device)
            velocity = self.predict_velocity(current, time, raw)
            current = current + velocity.float() / self.steps
        return self.decoder(current)


class PocketFacetsOnly(nn.Module):
    """Pocket FlowLM + matching continuous Mimi, without a transcript/voice bypass.

    The original trained autoregressive, flow, normalization, BOS/EOS and Mimi
    modules are retained. The donor text/voice-prefix pathways are never called
    by the student; the sole semantic prefix is the projected raw 32-facet block.
    """

    def __init__(self, *, facet_count: int, facet_width: int,
                 pack: PackSpec, donor: nn.Module):
        super().__init__()
        if pack.family != "pocket":
            raise ValueError("A Pocket student must use a Pocket checkpoint")
        self.pack_id, self.codec_id = pack.pack_id, pack.codec_id
        self.flow_lm = donor.flow_lm
        self.mimi = donor.mimi
        self.conditioner = FullFacetConditioner(
            facet_count, facet_width, self.flow_lm.dim
        )
        self.steps = pack.steps
        for parameter in self.mimi.parameters():
            parameter.requires_grad_(False)
        for parameter in self.flow_lm.parameters():
            parameter.requires_grad_(pack.train_backbone)

    @staticmethod
    def _state(module: nn.Module, n: int, batch: int):
        from pocket_tts.modules.stateful_module import init_states
        return init_states(module, batch_size=batch, sequence_length=n)

    def _normalized(self, latent: Tensor) -> Tensor:
        return ((latent.float() - self.flow_lm.emb_mean.float())
                / self.flow_lm.emb_std.float().clamp_min(1e-6))

    def flow_loss(self, raw: Tensor, target: Tensor,
                  mask: Tensor | None = None) -> Tensor:
        if target.ndim != 3 or target.shape[-1] != self.flow_lm.ldim:
            raise ValueError("Pocket teacher target must be [B, acoustic_frames, 32]")
        normed = self._normalized(target)
        bos = torch.full_like(normed[:, :1], float("nan"))
        preceding = torch.cat([bos, normed[:, :-1]], 1)
        flow = self.flow_lm
        state = self._state(flow, preceding.shape[1] + raw.shape[1] + 1,
                            raw.shape[0])
        prefix = self.conditioner(raw)
        prep = torch.where(torch.isnan(preceding), flow.bos_emb, preceding)
        encoded = flow.input_linear(prep.to(flow.input_linear.weight.dtype))
        hidden = flow.backbone(encoded, prefix.to(encoded.dtype), preceding, state)
        epsilon = torch.randn_like(normed)
        t = torch.rand_like(normed[..., :1])
        mixed = (1-t) * epsilon + t * normed
        if flow.flow_type == "flow_matching":
            velocity = flow.flow_net(hidden.float(), t, mixed)
        elif flow.flow_type in {"lsd", "drifting"}:
            # Native continuous segment parameterization, using both s and t.
            start = torch.zeros_like(t)
            velocity = flow.flow_net(hidden.float(), start, t, mixed)
        else:
            raise ValueError("Unrecognized Pocket flow parameterization")
        return masked_flow_mse(velocity, normed - epsilon, mask)

    @torch.no_grad()
    def sample(self, raw: Tensor, spec: Any, *, noise: Tensor | None = None,
               seed: int | None = None) -> Tensor:
        if spec.codec_id != self.codec_id or spec.pack_id != self.pack_id:
            raise ValueError("Pocket generator and Mimi decoder versions must match")
        if seed is None or noise is not None:
            raise ValueError("Pocket autoregressive renderer requires a recorded seed")
        if raw.shape[0] != 1 or len(spec.latent_shape) != 2:
            raise ValueError("Streaming Pocket currently requires one [frames, channels] event")
        from pocket_tts.modules.stateful_module import increment_steps
        flow = self.flow_lm
        frames, channels = spec.latent_shape
        if channels != flow.ldim:
            raise ValueError("Pocket native codec latent width differs")
        prefix = self.conditioner(raw).to(next(flow.parameters()).dtype)
        state = self._state(flow, frames + prefix.shape[1] + 4, 1)
        token = torch.full((1, 1, flow.ldim), float("nan"),
                           device=raw.device, dtype=prefix.dtype)
        latents = []
        cuda_devices = [raw.device.index or 0] if raw.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for index in range(frames):
                prompt = prefix if index == 0 else prefix[:, :0]
                latent, eos = flow._sample_next_latent(
                    token, prompt, model_state=state,
                    sampler_decode_steps=self.steps, temp=1.0,
                    noise_clamp=None, eos_threshold=-4.0,
                )
                latent = latent.reshape(1, 1, channels)
                increment_steps(flow, state, increment=1 + prompt.shape[1])
                latents.append(latent)
                token = latent
        samples = torch.cat(latents, dim=1)
        native = (samples * flow.emb_std + flow.emb_mean).to(raw.device)
        state = self._state(self.mimi, frames * 8 + 32, 1)
        result = self.mimi.decode_from_latent(native, state)
        return result.reshape(1, -1)[..., :spec.sample_count]


def load_native_packs(model: nn.Module, manifest: str | Path) -> dict[str, dict[str, int | str]]:
    """Install verified packs before creating the optimizer or loading checkpoints.

    Manifest is explicit: {schema_version:1, packs:{audio:{...},image:{...},video:{...}}}.
    Only requested families reside in memory. The included trained decoder is
    always paired to the generator from the same verified directory.
    """
    data = json.loads(Path(manifest).read_text())
    if data.get("schema_version") != 1 or not isinstance(data.get("packs"), dict):
        raise ValueError("Unsupported native pack manifest")
    reports: dict[str, dict[str, int | str]] = {}
    for modality, config in data["packs"].items():
        if modality not in {"audio", "image", "video"}:
            raise ValueError("Unknown installed modality pack")
        spec = PackSpec(**config)
        expect = {"audio": "pocket", "image": "sana_image", "video": "sana_video"}[modality]
        if spec.family != expect:
            raise ValueError("Wrong pretrained family for selected modality")
        if directory_sha256(spec.local_path) != spec.weights_sha256:
            raise ValueError(f"Native pack digest mismatch for {modality}")
        if spec.family == "pocket":
            from pocket_tts import TTSModel
            donor = TTSModel.load_model(config=str(Path(spec.local_path) / "config.yaml"))
            if donor.sample_rate != 24000:
                raise ValueError("Selected Pocket checkpoint is not the 24k continuous codec")
            realizer = PocketFacetsOnly(
                facet_count=model.config.n_facets, facet_width=model.config.facet_dim,
                pack=spec, donor=donor,
            )
        else:
            if spec.family == "sana_image":
                from diffusers import SanaPipeline
                pipeline_class = SanaPipeline
            else:
                from diffusers import SanaVideoPipeline
                pipeline_class = SanaVideoPipeline
            pipeline = pipeline_class.from_pretrained(spec.local_path, local_files_only=True)
            realizer = NativeLatentFlow(
                facet_count=model.config.n_facets, facet_width=model.config.facet_dim,
                pack=spec, denoiser=DiffusersDenoiser(pipeline.transformer),
                decoder=MatchingVAEDecoder(pipeline.vae),
            )
        realizer = realizer.to(model.core.initial_facets.device)
        model.media_realizers.install_native(modality, realizer)
        reports[modality] = {
            "pack_id": spec.pack_id, "codec_id": spec.codec_id,
            "frozen_parameters": sum(p.numel() for p in realizer.parameters()
                                     if not p.requires_grad),
            "trainable_parameters": sum(p.numel() for p in realizer.parameters()
                                        if p.requires_grad),
        }
    return reports
