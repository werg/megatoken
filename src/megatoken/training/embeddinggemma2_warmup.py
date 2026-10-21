"""Projection-only frozen EmbeddingGemma2 -> raw facets -> R autoencoding.

This is a *separate* E/R objective, not a sampled P continuation. The teacher
does not enter C, P, hidden KV, or the renderers; only its cached hidden states
are read by the trainable bridge. Train one media event at a time so the native
audio and image/video loss units remain per-event.
"""

from __future__ import annotations

import torch
from torch import Tensor

from megatoken.model.media import MediaTrainTarget


def reconstruction_loss(model, target: MediaTrainTarget) -> Tensor:
    """Autoencode one prepared observed source with frozen C/P/text E.

    The source must be teacher hidden features, *not* a pooled embedding or
    codec tokens. This function is usable standalone and by DiLoCo workers.
    """
    core = model.core
    if not core.config.embeddinggemma2_hidden_dim:
        raise ValueError("Enable the pinned EmbeddingGemma2 internal hidden-state geometry")
    if target.modality not in {"audio", "image", "video"}:
        raise ValueError("Only physical media have autoencoder reconstruction heads")
    if (
        target.source_kind != "features"
        or target.extractor != core.config.embeddinggemma2_hidden_id
    ):
        raise ValueError("Autoencoder source must be the pinned EmbeddingGemma2 hidden-state cache")
    if target.views or target.source_masks or len(target.source_views) > 1:
        raise ValueError("Autoencoder warmup expects exactly one unified teacher hidden-state view")
    if target.source_views:
        identity, source, valid = target.source_views[0]
        if identity != target.extractor or source.shape != target.source.shape:
            raise ValueError("Primary teacher view does not match its source")
        raw = core.encode_media_views(target.modality, ((identity, source, valid),))
    else:
        raw = core.encode_media_features(target.modality, target.source, extractor=target.extractor)
    context = core.prepare_realization(raw)
    renderer = model.media_realizers
    if target.target_kind == "codec_latents" and target.spec.codec_id:
        return renderer.loss_native(context, target)
    if target.target_kind == "codec_latents":
        loss, count = renderer.training_loss(
            context.facets, target.spec, target.target, valid_mask=target.target_valid_mask
        )
        if count != 1:
            raise ValueError("A codec reconstruction must account for exactly one event")
        return loss
    rendered = renderer.realize(context, target.spec)
    if rendered.shape != target.target.shape:
        raise ValueError("Reconstruction geometry mismatch")
    mask = target.target_valid_mask if target.target_valid_mask is not None else target.target_mask
    error = (rendered.float() - target.target.float()).square()
    if mask is None:
        return error.mean()
    expanded = torch.broadcast_to(mask, error.shape)
    if not bool(expanded.any()):
        raise ValueError("Autoencoder output mask cannot be empty")
    return error.masked_select(expanded).mean()


def prepared_media_events(batch, store, device):
    """Resolve only identity-verified, media-output training pairs."""
    from megatoken.training.meridians import MediaPlan

    for step in batch.steps:
        for plan in step.plans:
            if not isinstance(plan, MediaPlan):
                continue
            asset = plan.reference.source
            target = plan.reference.target
            if target is None or asset.kind != "features" or asset.source_family != target.sha256:
                raise ValueError(
                    "Autoencoder teacher features must attest to the exact original target "
                    "asset SHA256 in source_family; edit pairs cannot be used for warmup"
                )
            yield plan.resolve(store, device)


def media_warmup_batch(model, batch, store, device):
    """Sum native reconstruction losses and return the event count for global scaling."""
    values = [
        reconstruction_loss(model, target) for target in prepared_media_events(batch, store, device)
    ]
    if not values:
        return None, 0
    return torch.stack(values).sum(), len(values)


def text_hidden_alignment(
    model,
    *,
    text_bytes: bytes,
    hidden: Tensor,
    identity: str,
    valid: Tensor | None = None,
) -> Tensor:
    """Anchor native byte E to the very same text represented by frozen teacher.

    The native E is the student; teacher-projector coordinates are a detached
    alignment target. This never substitutes for the byte-exact lexical loss.
    """
    from megatoken.tensors import byte_tensor

    if not text_bytes:
        raise ValueError("Alignment must use the complete nonempty original text chunk")
    device = model.core.initial_facets.device
    ids, lengths = byte_tensor([text_bytes], device)
    student = model.core.encode_chunk(ids, lengths)
    teacher = model.core.encode_document_hidden(hidden, extractor=identity, valid=valid).detach()
    if student.shape != teacher.shape:
        raise ValueError("Teacher alignment must preserve all raw facet coordinates")
    return (student.float() - teacher.float()).square().mean()


def warmup_parameter_names(model, lane: str) -> tuple[str, ...]:
    """All DiLoCo workers keep the same masters; only gradients are lane-local."""
    if lane not in {"text", "media", "joint"}:
        raise ValueError("Warmup lane must be text, media, or joint")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    if lane == "joint":
        return tuple(names)
    media = tuple(
        name
        for name in names
        if name.startswith("core.media_stems.unified_projection.")
        or name.startswith("media_realizers.")
        and (
            name.startswith("media_realizers.pretrained.")
            and (".prefix." in name or ".condition." in name)
            or name.startswith("media_realizers.native.")
            and (
                ".conditioner." in name
                or ".condition." in name
                or ".projector." in name
                or ".adapter." in name
            )
            or name.startswith("media_realizers.")
            and name.split(".")[1]
            in {
                "fuse",
                "audio_expand",
                "audio_filter",
                "visual_features",
                "visual_positions",
                "visual_filter",
            }
        )
    )
    if lane == "media" and not media:
        raise ValueError("No trainable media input/output projections are loaded")
    return (
        media
        if lane == "media"
        else tuple(
            name for name in names if name not in media and not name.startswith("core.media_stems.")
        )
    )


def freeze_other_owners(optimizer, model, lane: str) -> tuple[str, ...]:
    """Freeze gradients without deleting masters or changing DiLoCo fragments."""
    active = set(warmup_parameter_names(model, lane))
    if not active:
        raise ValueError("Warmup lane has no trainable parameter owners")
    freeze = [name for name in optimizer.model_parameters if name not in active]
    optimizer.freeze(freeze)
    return tuple(sorted(active))
