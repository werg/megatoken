"""Pretrained, **raw-facets-only** Pocket and Sana realizers.

Donor weights and decoders are loaded from explicitly pinned local snapshots.
They are not downloaded by model construction. Only the donor's trained
generator/backbone and matching codec are used; original text/voice encoders
are never called to realize a Megatoken event. Other prompting APIs are
intentionally absent. This module is importable without optional packages.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .flow import flow_matching_loss


def _raw(raw: Tensor, width: int) -> None:
    if raw.ndim != 3 or raw.shape[1:] != (32, width):
        raise ValueError("Renderer content condition must be exactly raw [B,32,D]")
    if not raw.is_floating_point() or not bool(torch.isfinite(raw).all()):
        raise ValueError("Raw facets must be finite floating point")


class PocketFacetsStudent(nn.Module):
    """Use Pocket's pretrained FlowLM/flow net and its matching frozen Mimi.

    The 32 facets are the *only* prefix for a new utterance. The donor's
    LUTConditioner, transcript, voice prompt, and earlier donor state are never
    accessed in the student forward path. This is a conditioning adaptation;
    quality must be established by actual training and free rollouts.
    """

    def __init__(self, pocket_model: Any, facet_width: int = 512):
        super().__init__()
        self.pocket = pocket_model
        self.flow_lm = pocket_model.flow_lm
        self.mimi = pocket_model.mimi
        self.facet_width = facet_width
        self.prefix = nn.Sequential(nn.LayerNorm(facet_width),
                                    nn.Linear(facet_width, self.flow_lm.dim))
        for parameter in self.mimi.parameters():
            parameter.requires_grad_(False)
        for parameter in self.flow_lm.parameters():
            parameter.requires_grad_(False)
        self.frame_rate = float(self.mimi.frame_rate)
        self.sample_rate = int(self.mimi.sample_rate)
        self.latent_width = int(self.flow_lm.ldim)
        if self.latent_width != self.mimi.quantizer.dimension:
            raise ValueError("Pocket flow/mimi continuous latent geometry differs")

    @classmethod
    def from_checkpoint(cls, *, config: str, facet_width: int = 512,
                        checkpoint: str | None = None):
        """config must refer to an inspected, pinned *local* Pocket YAML."""
        from pocket_tts import TTSModel
        if not Path(config).is_file():
            raise FileNotFoundError("Pocket YAML is not a local pinned file")
        model = TTSModel.load_model(config=config, checkpoint=checkpoint, quantize=False)
        return cls(model, facet_width=facet_width)

    def train_adapter(self, *, train_backbone: bool = False,
                      train_flow_head: bool = False):
        for parameter in self.flow_lm.transformer.parameters():
            parameter.requires_grad_(train_backbone)
        for parameter in self.flow_lm.flow_net.parameters():
            parameter.requires_grad_(train_flow_head)
        return self

    def _cond(self, raw: Tensor):
        _raw(raw, self.facet_width)
        return self.prefix(raw.to(self.prefix[1].weight.dtype))

    def encode_waveform(self, waveform: Tensor, *, sample_rate: int):
        """Frozen native Mimi codec: [B, samples] -> [B, frames, acoustic_dim]."""
        if sample_rate != self.sample_rate or waveform.ndim != 2:
            raise ValueError("Pocket Mimi requires declared native-rate mono audio")
        with torch.no_grad():
            return self.mimi.encode_to_latent(waveform[:, None].to(self.mimi.encoder.layers[0].conv.conv.weight.dtype)
                                              if False else waveform[:, None])

    def _normalized(self, target: Tensor):
        if target.ndim != 3 or target.shape[-1] != self.latent_width:
            raise ValueError("Pocket target must contain continuous Mimi latents")
        return (target - self.flow_lm.emb_mean) / self.flow_lm.emb_std.clamp_min(1e-6)

    def native_loss(self, raw: Tensor, mimi_latents: Tensor, *, valid_frames=None,
                    generator: torch.Generator | None = None):
        """Donor flow-head loss with scheduled, matched teacher acoustic prefixes.

        This trains conditioner/flow; the input sequence is teacher forcing local
        to R. It does not provide P, C, or current sibling R with target bytes.
        Supported flow parameterization is checked rather than guessed.
        """
        from pocket_tts.modules.stateful_module import init_states, increment_steps
        condition = self._cond(raw)
        clean = self._normalized(mimi_latents).to(condition.dtype)
        b, frames, channels = clean.shape
        if b != condition.shape[0] or frames < 1:
            raise ValueError("Mimi latent batch or frame count mismatch")
        bos = torch.full((b, 1, channels), float("nan"),
                         device=clean.device, dtype=clean.dtype)
        teacher_prefix = torch.cat((bos, clean[:, :-1]), dim=1)
        states = init_states(self.flow_lm, b, condition.shape[1] + frames + 4)
        embedded = torch.where(torch.isnan(teacher_prefix), self.flow_lm.bos_emb,
                               teacher_prefix)
        embedded = self.flow_lm.input_linear(embedded)
        hidden = self.flow_lm.backbone(
            embedded, condition, teacher_prefix, model_state=states
        )
        flow_type = self.flow_lm.flow_type
        if flow_type == "flow_matching":
            times = torch.rand((b, frames, 1), device=clean.device, generator=generator)
            noise = torch.randn(clean.shape, device=clean.device,
                                dtype=clean.dtype, generator=generator)
            noisy = (1 - times) * noise + times * clean
            predicted = self.flow_lm.flow_net(hidden.float(), times.float(), noisy.float())
            target = clean.float() - noise.float()
        elif flow_type == "lsd":
            # For LSD the pair (s,t) is part of the donor's training
            # convention; do not mislabel it as an OT flow objective.
            times = torch.rand((b, frames, 1), device=clean.device, generator=generator)
            noise = torch.randn(clean.shape, device=clean.device,
                                dtype=clean.dtype, generator=generator)
            noisy = (1 - times) * noise + times * clean
            predicted = self.flow_lm.flow_net(
                hidden.float(), times.float(), torch.ones_like(times),
                noisy.float()
            )
            target = clean.float() - noise.float()
        else:
            raise ValueError("Unsupported Pocket donor flow convention for native training")
        return flow_matching_loss(predicted, target, valid_frames)

    def latent(self, raw: Tensor, *, max_frames: int, steps: int = 16,
               generator: torch.Generator | None = None):
        """Full free rollout; no target-conditioned acoustic prefix is supplied."""
        from pocket_tts.modules.stateful_module import init_states, increment_steps
        condition = self._cond(raw)
        b = raw.shape[0]
        if not 1 <= max_frames <= 1200 or not 1 <= steps <= 128:
            raise ValueError("Unbounded speech rollout")
        states = init_states(self.flow_lm, b, 32 + max_frames + 4)
        previous = torch.full((b, 1, self.latent_width), float("nan"),
                              device=raw.device, dtype=condition.dtype)
        outputs = []
        for t in range(max_frames):
            # The pretrained FlowLM sample is stateful, with correct prefix
            # only on step zero. Optional RNG is isolated via fork_rng.
            embeddings = condition if t == 0 else condition[:, :0]
            if generator is not None:
                noise_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator))
                devices = [raw.device.index] if raw.is_cuda and raw.device.index is not None else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(noise_seed)
                    next_latent, eos = self.flow_lm._sample_next_latent(
                        previous, embeddings, states, steps, self.pocket.temp,
                        self.pocket.noise_clamp, self.pocket.eos_threshold,
                    )
            else:
                next_latent, eos = self.flow_lm._sample_next_latent(
                    previous, embeddings, states, steps, self.pocket.temp,
                    self.pocket.noise_clamp, self.pocket.eos_threshold,
                )
            increment_steps(self.flow_lm, states, increment=previous.shape[1] + embeddings.shape[1])
            outputs.append(next_latent[:, None])
            previous = next_latent[:, None]
            if t >= 3 and bool(eos.all()):
                break
        acoustic = torch.cat(outputs, dim=1)
        return acoustic * self.flow_lm.emb_std + self.flow_lm.emb_mean

    def decode(self, native_latents: Tensor):
        from pocket_tts.modules.stateful_module import init_states
        if native_latents.ndim != 3 or native_latents.shape[-1] != self.latent_width:
            raise ValueError("Pocket latent/decoder mismatch")
        state = init_states(self.mimi, native_latents.shape[0],
                            native_latents.shape[1] + 16)
        with torch.no_grad():
            return self.mimi.decode_from_latent(native_latents, state)

    def realize(self, raw: Tensor, *, max_frames: int, steps=16, generator=None):
        return self.decode(self.latent(raw, max_frames=max_frames, steps=steps, generator=generator))


class SanaFacetsStudent(nn.Module):
    """Diffusers Sana/SanaVideo trained transformer with matched frozen VAE.

    Both pipelines accept prompt_embeds; the donor text encoder is not called
    and may be unloaded. All 32 facets map to donor caption embeddings.
    """

    def __init__(self, pipeline: Any, modality: str, facet_width: int = 512):
        super().__init__()
        if modality not in ("image", "video"):
            raise ValueError("Sana supports still image or video")
        self.modality = modality
        self.facet_width = facet_width
        self.pipe = pipeline
        channels = int(pipeline.transformer.config.caption_channels)
        self.condition = nn.Sequential(nn.LayerNorm(facet_width),
                                       nn.Linear(facet_width, channels))
        # Pipeline holds the exact matching checkpoint components, including
        # the video LTX2 VAE; the frozen codec never receives source media.
        self.transformer = pipeline.transformer
        self.vae = pipeline.vae
        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        for parameter in self.transformer.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, *, snapshot: str, modality: str,
                        facet_width: int = 512, torch_dtype: torch.dtype = torch.bfloat16):
        from diffusers import SanaPipeline, SanaVideoPipeline
        if not Path(snapshot).is_dir():
            raise FileNotFoundError("Require a pinned downloaded local Diffusers snapshot")
        klass = SanaPipeline if modality == "image" else SanaVideoPipeline
        pipeline = klass.from_pretrained(snapshot, torch_dtype=torch_dtype, local_files_only=True)
        if modality == "video":
            if type(pipeline.vae).__name__ != "AutoencoderKLLTX2Video":
                raise ValueError("720p SANA-Video requires its matching LTX-2 video VAE")
        else:
            if type(pipeline.vae).__name__ != "AutoencoderDC":
                raise ValueError("Sana 600M requires its matching DC-AE")
        return cls(pipeline, modality, facet_width)

    def train_adapter(self, *, train_backbone=False):
        for p in self.transformer.parameters():
            p.requires_grad_(train_backbone)
        return self

    def _embeddings(self, raw):
        _raw(raw, self.facet_width)
        embeds = self.condition(raw.to(self.condition[1].weight.dtype))
        mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.bool)
        return embeds, mask

    def latent(self, raw: Tensor, *, height: int, width: int, frames=1,
               steps: int = 30, guidance: float = 4.0, generator=None):
        embeds, mask = self._embeddings(raw)
        if self.modality == "image" and frames != 1:
            raise ValueError("Image renderer only accepts one frame")
        if not 1 <= steps <= 200:
            raise ValueError("Invalid Sana scheduler steps")
        kwargs = dict(prompt_embeds=embeds, prompt_attention_mask=mask,
                      negative_prompt_embeds=torch.zeros_like(embeds),
                      negative_prompt_attention_mask=mask, height=height,
                      width=width, guidance_scale=guidance,
                      num_inference_steps=steps, generator=generator,
                      output_type="latent", return_dict=False,
                      use_resolution_binning=False)
        if self.modality == "video":
            kwargs["frames"] = frames
        # Never pass text prompt, original source latents or donor text state.
        output = self.pipe(**kwargs)[0]
        return output

    def native_loss(self, raw: Tensor, target_latents: Tensor, *, valid_mask=None,
                    generator=None):
        """Use the checkpoint scheduler's declared prediction convention.

        The diffusion training input may read noisy U (teacher target); the
        student condition reads Z only, and the target cannot enter prior/C.
        """
        embeddings, mask = self._embeddings(raw)
        config = self.pipe.scheduler.config
        prediction_type = getattr(config, "prediction_type", "epsilon")
        if prediction_type not in ("epsilon", "v_prediction", "sample"):
            raise ValueError("Unknown donor diffusion target parameterization")
        if getattr(config, "num_train_timesteps", 0) <= 0:
            raise ValueError("Sana scheduler has no declared training clock")
        b = target_latents.shape[0]
        timestep = torch.randint(
            0, int(config.num_train_timesteps), (b,),
            device=target_latents.device, generator=generator
        )
        noise = torch.randn(target_latents.shape, dtype=target_latents.dtype,
                            device=target_latents.device, generator=generator)
        noisy = self.pipe.scheduler.add_noise(target_latents, noise, timestep)
        argument_time = timestep
        if self.modality == "image":
            argument_time = timestep * self.transformer.config.timestep_scale
        output = self.transformer(
            noisy.to(self.transformer.dtype),
            encoder_hidden_states=embeddings.to(self.transformer.dtype),
            encoder_attention_mask=mask,
            timestep=argument_time,
            return_dict=False,
        )[0].float()
        if output.shape[1] == target_latents.shape[1] * 2:
            output = output.chunk(2, dim=1)[0]
        if prediction_type == "epsilon":
            target = noise.float()
        elif prediction_type == "sample":
            target = target_latents.float()
        else:
            target = self.pipe.scheduler.get_velocity(target_latents, noise, timestep).float()
        return flow_matching_loss(output, target, valid_mask)

    def realize(self, raw: Tensor, **spec):
        # Decode via *this pipeline's* matched VAE and scaling conventions;
        # don't reimplement video VAE normalization from another checkpoint.
        embeds, mask = self._embeddings(raw)
        kwargs = dict(prompt_embeds=embeds, prompt_attention_mask=mask,
                      negative_prompt_embeds=torch.zeros_like(embeds),
                      negative_prompt_attention_mask=mask,
                      guidance_scale=spec.get("guidance",4.0),
                      num_inference_steps=spec.get("steps",30),
                      height=spec["height"], width=spec["width"],
                      generator=spec.get("generator"),
                      output_type="pt", return_dict=False,
                      use_resolution_binning=False)
        if self.modality == "video":
            kwargs["frames"] = spec["frames"]
        return self.pipe(**kwargs)[0]
