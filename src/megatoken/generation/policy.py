"""Canonical-facet prediction, shared decoder training and exact latent replay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any

import torch
from torch import Tensor, nn

from megatoken.controls import encode_controls
from megatoken.tensors import byte_tensor

from .codec import ByteCodec
from .decoder import DecodedAction, DeterministicDecoder
from .latents import FacetGaussian
from .structure import TAPE_VERSION, ActionSpec, ActionTape, Observation, TapeMismatch
from .targets import LOSS_NAMES, PreparedAction, PreparedTargets, prepare_action, prepare_targets


@dataclass
class SupervisedResult:
    losses: dict[str, tuple[Tensor, int]]
    state: Any
    metrics: dict[str, int | float | Tensor]


@dataclass(frozen=True)
class PolicyResult:
    tape: ActionTape
    payload: bytes
    emitted_ids: tuple[int, ...]
    state: Any
    log_prob: Tensor
    scored_positions: int
    latent_dimensions: int


class FacetedGenerator(nn.Module):
    """The same raw chunk space joins observations, predictions and realization.

    Observations install E(bytes); generated actions recur through their sampled raw z.
    There is deliberately no E(D(z)) reconciliation, posterior, or private carry.
    All stochasticity is confined to one Gaussian before the deterministic decoder.
    """

    def __init__(self, core, codec: ByteCodec | None = None, *, training_heads: bool = True):
        super().__init__()
        self.core = core
        self.config = core.config
        self.codec = codec or ByteCodec.byte_fallback()
        if self.codec.vocab_size != self.config.vocab_size:
            raise ValueError("codec vocabulary rows must match the model configuration")
        self.training_heads = training_heads
        self.facet_distribution = FacetGaussian(self.config.facet_dim)
        self.decoder = DeterministicDecoder(self.config, self.codec)

    @property
    def latent_dimensions(self) -> int:
        return self.config.n_facets * self.config.facet_dim

    def _spec(self, spec):
        return spec or ActionSpec(max_tokens=min(512, self.config.max_capacity))

    def prepare_target(self, target_ids=None, **kwargs) -> PreparedTargets:
        return prepare_targets(self.codec, self.config, target_ids, **kwargs)

    def prepare_action_target(self, payload: bytes, **kwargs) -> PreparedAction:
        return prepare_action(self.codec, self.config, payload, **kwargs)

    @staticmethod
    def _validate_mode(plan, representation_mode):
        if representation_mode is not None and representation_mode != plan.mode:
            raise ValueError("forward representation_mode differs from the prepared target plan")
        if plan.mode not in {"bootstrap", "recurrent"}:
            raise ValueError("representation_mode must be bootstrap or recurrent")

    def _training_inputs(self, state, payloads, control, effort, generator, roles, mode):
        if not self.training_heads:
            raise RuntimeError("supervised training is disabled for this served model")
        if state.batch_size != len(payloads):
            raise ValueError("target batch and independent input-facet state batch differ")
        self.core.routing_aux_loss(clear=True)
        # Complete the six body stages, nonresidual remix and ten predictive
        # residual stages before exposing any current target bytes.
        processed = self.core.process(state, control=control, effort=effort, next_role_ids=roles)
        prior = self.facet_distribution(processed.facets)
        sampled = prior.sample(generator)
        values, lengths = byte_tensor(payloads, state.device)
        target = self.core.encode_chunk(values, lengths=lengths)
        zero = prior.mean.sum() * 0
        if mode == "bootstrap":
            facet_loss = -prior.log_prob(target.detach()).sum()
            encoder_loss = zero
            next_input = target  # explicit early teacher forcing, no learned writer
        else:
            facet_loss = zero
            encoder_loss = (target.float() - prior.mean.detach()).square().sum()
            next_input = sampled  # the tower stays live across predicted chunks
        next_state = self.core.replace_facets(processed, next_input, role_ids=roles)
        return next_state, sampled, target, prior.mean, facet_loss, encoder_loss

    def _finish_supervision(
        self,
        next_state,
        target,
        mean,
        prior_losses,
        reconstruction_losses,
        facet_loss,
        encoder_loss,
        plan,
        payloads,
    ):
        zero = mean.sum() * 0
        sums = dict.fromkeys(LOSS_NAMES, zero)
        counts = dict.fromkeys(LOSS_NAMES, 0)
        bootstrap = plan.mode == "bootstrap"
        sums["lexical"], counts["lexical"] = prior_losses["lexical"]
        if reconstruction_losses is not None:
            sums["reconstruction"], counts["reconstruction"] = reconstruction_losses["lexical"]
        for key in ("structure", "binding", "placement", "terminal"):
            first, first_count = prior_losses[key]
            second, second_count = (
                (zero, 0) if reconstruction_losses is None else reconstruction_losses[key]
            )
            sums[key], counts[key] = first + second, first_count + second_count
        sums["facet_prediction"] = facet_loss
        sums["encoder_alignment"] = encoder_loss
        counts["facet_prediction"] = target.numel() if bootstrap else 0
        counts["encoder_alignment"] = 0 if bootstrap else target.numel()
        balance, stabilization = self.core.routing_aux_loss(clear=True)
        batch = len(payloads)
        sums["router_balance"], sums["router_z"] = balance * batch, stabilization * batch
        counts["router_balance"] = counts["router_z"] = batch
        for name in LOSS_NAMES:
            if counts[name] != plan.denominators[name]:
                raise RuntimeError(
                    f"prepared {name} count {plan.denominators[name]} differs from actual {counts[name]}"
                )
        detached_target, detached_mean = target.detach().float(), mean.detach().float()
        metrics = {
            "scored_positions": plan.scored_positions,
            "capacity": plan.capacity,
            "prior_examples": batch,
            "reconstruction_examples": batch if bootstrap else 0,
            "prior_lexical_positions": counts["lexical"],
            "reconstruction_lexical_positions": counts["reconstruction"],
            "prior_lexical_loss_sum": sums["lexical"].detach(),
            "reconstruction_lexical_loss_sum": sums["reconstruction"].detach(),
            "facet_prediction_loss_sum": facet_loss.detach(),
            "facet_prediction_coordinates": counts["facet_prediction"],
            "encoder_alignment_loss_sum": encoder_loss.detach(),
            "encoder_alignment_coordinates": counts["encoder_alignment"],
            "facet_examples": batch,
            "prediction_facet_norm_sum": detached_mean.flatten(1).norm(dim=1).sum(),
            "encoded_facet_norm_sum": detached_target.flatten(1).norm(dim=1).sum(),
            "facet_alignment_squared_error_sum": (detached_target - detached_mean).square().sum(),
            "facet_alignment_coordinates": target.numel(),
            "facet_encodes": batch,
            "facet_bytes_encoded": sum(map(len, payloads)),
            "capacity_positions": (1 + int(bootstrap)) * batch * plan.capacity,
            "lexical_passes": 1 + int(bootstrap),
            "structured_actions": int(isinstance(plan, PreparedAction)),
        }
        return SupervisedResult(
            {name: (sums[name], counts[name]) for name in LOSS_NAMES}, next_state, metrics
        )

    def supervised_step(
        self,
        state,
        target_ids=None,
        *,
        target_bytes=None,
        free_mask=None,
        plan: PreparedTargets | None = None,
        effort=6,
        control=None,
        generator=None,
        role_ids=None,
        representation_mode=None,
    ) -> SupervisedResult:
        plan = plan or self.prepare_target(
            target_ids,
            target_bytes=target_bytes,
            free_mask=free_mask,
            role_ids=role_ids,
            representation_mode=representation_mode or "bootstrap",
        )
        self._validate_mode(plan, representation_mode)
        roles = (
            torch.tensor(plan.roles, device=state.device)
            if role_ids is None
            else role_ids.to(state.device)
        )
        next_state, sampled, target, mean, facet_loss, encoder_loss = self._training_inputs(
            state, plan.payloads, control, effort, generator, roles, plan.mode
        )
        prior_losses = self.decoder.teacher_raw(self.core, sampled, plan)
        reconstruction_losses = (
            self.decoder.teacher_raw(self.core, target, plan) if plan.mode == "bootstrap" else None
        )
        return self._finish_supervision(
            next_state,
            target,
            mean,
            prior_losses,
            reconstruction_losses,
            facet_loss,
            encoder_loss,
            plan,
            plan.payloads,
        )

    def supervised_action(
        self,
        state,
        *,
        plan: PreparedAction,
        effort=6,
        control=None,
        generator=None,
        representation_mode=None,
    ) -> SupervisedResult:
        self._validate_mode(plan, representation_mode)
        roles = torch.tensor([plan.role_id], device=state.device)
        payloads = [plan.payload]
        next_state, sampled, target, mean, facet_loss, encoder_loss = self._training_inputs(
            state, payloads, control, effort, generator, roles, plan.mode
        )
        prior = self.decoder(self.core, sampled, plan.spec, teacher=plan)
        reconstruction = (
            self.decoder(self.core, target, plan.spec, teacher=plan)
            if plan.mode == "bootstrap"
            else None
        )
        return self._finish_supervision(
            next_state,
            target,
            mean,
            prior.losses,
            None if reconstruction is None else reconstruction.losses,
            facet_loss,
            encoder_loss,
            plan,
            payloads,
        )

    @torch.no_grad()
    def decode_facets(self, raw_facets: Tensor, *, spec: ActionSpec | None = None) -> DecodedAction:
        """Pure deterministic realization; accepts E(target) or a prior sample."""
        expected = (1, self.config.n_facets, self.config.facet_dim)
        if tuple(raw_facets.shape) != expected or not bool(torch.isfinite(raw_facets).all()):
            raise ValueError(f"decode_facets requires one finite raw tensor with shape {expected}")
        return self.decoder(self.core, raw_facets, self._spec(spec))

    @torch.no_grad()
    def decode_mean(
        self, state, *, spec=None, control_payload: bytes = b"", effort=6
    ) -> DecodedAction:
        """Inspect the predictor mean without sampling or committing anything."""
        control = encode_controls(self.core, [control_payload], encoder_grad=False)
        processed = self.core.process(state, control=control, effort=effort)
        return self.decode_facets(processed.facets, spec=spec)

    @torch.no_grad()
    def sample(
        self,
        state,
        *,
        spec=None,
        control_payload: bytes = b"",
        effort=6,
        generator=None,
        role_id=2,
        rl=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
    ) -> PolicyResult:
        if state.batch_size != 1:
            raise ValueError("sample and score_tape require one independent episode")
        if temperature != 1.0 or top_k is not None or top_p is not None:
            raise ValueError(
                "one Gaussian draw and deterministic argmax do not support token sampling controls"
            )
        if rl and effort != 6:
            raise ValueError("outcome RL requires fixed effort 6")
        if not 0 <= role_id <= 5:
            raise ValueError("action role ID is unsupported")
        spec = self._spec(spec)
        self.core.routing_aux_loss(clear=True)
        roles = torch.tensor([role_id], device=state.device)
        control = encode_controls(self.core, [control_payload], encoder_grad=False)
        processed = self.core.process(state, control=control, effort=effort, next_role_ids=roles)
        distribution = self.facet_distribution(processed.facets)
        raw = distribution.sample(generator)
        decoded = self.decode_facets(raw, spec=spec)
        log_prob = distribution.log_prob(raw).sum()
        tape = ActionTape(
            TAPE_VERSION,
            self.codec.fingerprint,
            spec.fingerprint,
            (self.config.n_facets, self.config.facet_dim),
            tuple(raw[0].flatten().tolist()),
            effort,
            role_id,
            decoded.payload,
            decoded.emitted_ids,
            decoded.valid,
            decoded.terminal,
            float(log_prob),
            decoded.trace,
            decoded.error,
            control_payload,
        )
        next_state = self.core.replace_facets(processed, raw, role_ids=roles)
        self.core.routing_aux_loss(clear=True)
        return PolicyResult(
            tape,
            decoded.payload,
            decoded.emitted_ids,
            next_state,
            log_prob,
            0,
            self.latent_dimensions,
        )

    def score_tape(
        self, state, tape: ActionTape, *, spec=None, control_payload: bytes | None = None
    ) -> PolicyResult:
        if state.batch_size != 1:
            raise ValueError("sample and score_tape require one independent episode")
        spec = self._spec(spec)
        if tape.version != TAPE_VERSION or tape.codec_fingerprint != self.codec.fingerprint:
            raise TapeMismatch("tape policy or codec version differs")
        if tape.spec_fingerprint != spec.fingerprint:
            raise TapeMismatch("tape public constraints or copy bytes differ")
        if tape.facet_shape != (self.config.n_facets, self.config.facet_dim):
            raise TapeMismatch("tape raw facet geometry differs")
        if control_payload is not None and (
            not isinstance(control_payload, bytes) or control_payload != tape.control_payload
        ):
            raise TapeMismatch("control override differs from the tape's original causal bytes")
        self.core.routing_aux_loss(clear=True)
        roles = torch.tensor([tape.role_id], device=state.device)
        control = encode_controls(self.core, [tape.control_payload], encoder_grad=False)
        processed = self.core.process(
            state, control=control, effort=tape.effort, next_role_ids=roles
        )
        distribution = self.facet_distribution(processed.facets)
        raw = torch.tensor(tape.facets, device=state.device, dtype=torch.float32).reshape(
            1, *tape.facet_shape
        )
        decoded = self.decode_facets(raw, spec=spec)
        actual = (
            decoded.payload,
            decoded.emitted_ids,
            decoded.valid,
            decoded.terminal,
            decoded.trace,
            decoded.error,
        )
        recorded = (
            tape.payload,
            tape.emitted_ids,
            tape.valid,
            tape.terminal,
            tape.trace,
            tape.error,
        )
        if actual != recorded:
            raise TapeMismatch(
                "deterministic action map changed or tape materialization was tampered with"
            )
        log_prob = distribution.log_prob(raw).sum()
        next_state = self.core.replace_facets(processed, raw, role_ids=roles)
        self.core.routing_aux_loss(clear=True)
        return PolicyResult(
            tape,
            decoded.payload,
            decoded.emitted_ids,
            next_state,
            log_prob,
            0,
            self.latent_dimensions,
        )

    def replay(
        self,
        initial_state,
        records: Sequence[bytes | Observation | ActionTape],
        *,
        specs=None,
    ):
        """Rebuild a complete differentiable episode from observation chunks and raw tapes."""
        state, log_probs, decision = initial_state, [], 0
        for record in records:
            if isinstance(record, (bytes, Observation)):
                observation = Observation(record) if isinstance(record, bytes) else record
                values, lengths = byte_tensor([observation.payload], state.device)
                roles = torch.tensor([observation.role_id], device=state.device)
                state = self.core.observe(values, state, lengths=lengths, role_ids=roles)
            else:
                result = self.score_tape(
                    state,
                    record,
                    spec=None if specs is None else specs[decision],
                )
                state = result.state
                log_probs.append(result.log_prob)
                decision += 1
        return state, tuple(log_probs)

    def deterministic_decoder_parameters(self):
        """Owners of E and z→action, including the shared U bank, exactly once."""
        seen = set()
        for parameter in chain(
            self.decoder.parameters(),
            self.core.encoder_parameters(),
            self.core.realizer_parameters(),
        ):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter

    def freeze_decoder_for_rl(self) -> tuple[str, ...]:
        frozen = {id(parameter) for parameter in self.deterministic_decoder_parameters()}
        names = []
        for name, parameter in self.named_parameters():
            if id(parameter) in frozen:
                parameter.requires_grad_(False)
                parameter.grad = None
                names.append(name)
        return tuple(names)

    def served_state_dict(self) -> dict[str, Tensor]:
        """All encoder, decoder and predictor owners are shared with serving."""
        if self.core.frontend.ngrams.import_map is not None:
            raise RuntimeError("bake the engram adapter before served export")
        return self.state_dict()
