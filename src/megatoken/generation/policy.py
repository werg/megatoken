"""Integrated prior/parallel policy, exact replay, and supervised objectives.

The sample and score methods execute the same ordered program. Only the source
of choices differs. No behavior-policy routes or hidden-state caches are replayed.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.tensors import byte_tensor

from .codec import ByteCodec
from .distributions import TiedVocabulary
from .latents import DiagonalGaussian, GaussianHead, TargetPosterior
from .realizer import ParallelRealizer, positional_features
from .structure import (
    ActionSpec,
    ActionTape,
    Choice,
    CopySource,
    DraftAttempt,
    FieldSpec,
    LexicalDraw,
    Observation,
    SchemaSpec,
    TapeMismatch,
    strict_json_loads,
)

LOSS_NAMES = (
    "lexical",
    "structure",
    "binding",
    "placement",
    "terminal",
    "prior_fit",
    "posterior_pressure",
    "content_alignment",
    "router_balance",
    "router_z",
)
TAPE_VERSION = "fmt-augmented-policy-v1"


@dataclass(frozen=True)
class PreparedTargets:
    payloads: tuple[bytes, ...]
    ids: tuple[tuple[int, ...], ...]
    free_masks: tuple[tuple[bool, ...], ...]
    terminals: tuple[bool, ...]
    capacity: int
    repair: bool
    denominators: Mapping[str, int]
    scored_positions: int
    roles: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreparedAction:
    payload: bytes
    spec: ActionSpec
    schema_index: int
    layout: int
    capacity: int
    field_ids: Mapping[str, tuple[int, ...]]
    bindings: Mapping[str, int]
    placements: Mapping[str, int]
    repair: bool
    denominators: Mapping[str, int]
    scored_positions: int


@dataclass
class SupervisedResult:
    losses: dict[str, tuple[Tensor, int]]
    state: Any
    metrics: dict[str, int | float | Tensor]


@dataclass
class PolicyResult:
    tape: ActionTape
    payload: bytes
    emitted_ids: tuple[int, ...]
    state: Any
    log_prob: Tensor
    scored_positions: int
    losses: dict[str, tuple[Tensor, int]] = field(default_factory=dict)
    prediction: Any = None
    root_distribution: DiagonalGaussian | None = None
    regional_distribution: DiagonalGaussian | None = None
    root_value: Tensor | None = None
    regional_context: Tensor | None = None
    lexical_rows: Tensor | None = None
    header: Tensor | None = None
    repair_targets: tuple[tuple[tuple[int, ...], tuple[int, ...], str], ...] = ()
    draft_embeddings: Tensor | None = None
    used_prior: bool | None = None


class _Events:
    """The shared probability accumulator and ordered tape reader/writer."""

    def __init__(self, reference: Tensor, *, tape=None, teacher=None, generator=None):
        self.tape = tape
        self.teacher = teacher
        self.generator = generator
        self.choice_cursor = 0
        self.choices: list[Choice] = []
        self.log_prob = reference.float().sum() * 0
        self.losses = {name: [self.log_prob, 0] for name in LOSS_NAMES}
        self.scored_positions = 0

    def add(self, key: str, loss: Tensor, denominator: int) -> None:
        self.losses[key][0] = self.losses[key][0] + loss
        self.losses[key][1] += denominator

    def choose(
        self, name: str, logits: Tensor, allowed: Sequence[int], *, key="structure", forced=None
    ) -> int:
        if not allowed:
            raise ValueError(f"no feasible support for {name}")
        allowed = tuple(int(index) for index in allowed)
        support = torch.tensor(allowed, dtype=torch.long, device=logits.device)
        log_probs = logits.float().reshape(-1)[support].log_softmax(-1)
        if self.tape is not None:
            if self.choice_cursor >= len(self.tape.choices):
                raise TapeMismatch(f"missing categorical event {name}")
            choice = self.tape.choices[self.choice_cursor]
            if choice.name != name:
                raise TapeMismatch(f"expected event {name}, found {choice.name}")
            value = choice.value
        elif self.teacher is not None:
            if forced is None:
                raise ValueError(f"missing supervised target for {name}")
            value = int(forced)
        else:
            local = torch.multinomial(log_probs.exp(), 1, generator=self.generator).item()
            value = allowed[local]
        if value not in allowed:
            raise TapeMismatch(f"event {name}={value} is outside feasible support {allowed[:16]}")
        selected = log_probs[allowed.index(value)]
        self.log_prob = self.log_prob + selected
        self.add(key, -selected, 1)
        self.choices.append(Choice(name, value))
        self.choice_cursor += 1
        return value

    def lexical(
        self, name: str, model, states: Tensor, valid: Tensor, *, supplied=None, forced=None
    ) -> LexicalDraw:
        if self.tape is not None:
            if supplied is None or supplied.name != name:
                raise TapeMismatch(f"missing parallel lexical draw {name}")
            ids = supplied.ids
        elif self.teacher is not None:
            if forced is None:
                raise ValueError(f"missing lexical supervision for {name}")
            ids = tuple(forced)
        else:
            ids = tuple(model.vocabulary.sample(states, valid, generator=self.generator).tolist())
        if len(ids) != states.shape[0]:
            raise TapeMismatch(f"draw {name} has {len(ids)} IDs, expected {states.shape[0]}")
        target = torch.tensor(ids, device=states.device, dtype=torch.long)
        loss = model.vocabulary.loss(states, target, valid).sum()
        self.log_prob = self.log_prob - loss
        self.add("lexical", loss, len(ids))
        self.scored_positions += len(ids)
        return LexicalDraw(name, tuple(ids))

    def finish(self) -> None:
        if self.tape is not None and self.choice_cursor != len(self.tape.choices):
            raise TapeMismatch("tape contains unused categorical events")


@dataclass
class _Field:
    spec: FieldSpec
    fixed_ids: tuple[int, ...] | None = None
    copy_ids: tuple[int, ...] = ()
    reserved_minimum: int = 0
    length: int = 0
    placement: int = 0
    offset: int = 0

    @property
    def minimum(self) -> int:
        return (
            len(self.fixed_ids)
            if self.fixed_ids is not None
            else max(self.spec.min_tokens, self.reserved_minimum, len(self.copy_ids))
        )

    @property
    def free_positions(self) -> tuple[int, ...]:
        if self.fixed_ids is not None:
            return ()
        return tuple(
            self.offset + index
            for index in range(self.length)
            if not self.placement <= index < self.placement + len(self.copy_ids)
        )

    def assemble(self, free_ids: tuple[int, ...]) -> tuple[int, ...]:
        if self.fixed_ids is not None:
            return self.fixed_ids
        return free_ids[: self.placement] + self.copy_ids + free_ids[self.placement :]


class FacetedGenerator(nn.Module):
    def __init__(
        self, core, codec: ByteCodec | None = None, *, training_heads: bool = True
    ) -> None:
        super().__init__()
        self.core = core
        self.config = core.config
        self.codec = codec or ByteCodec.byte_fallback()
        if self.codec.vocab_size != self.config.vocab_size:
            raise ValueError(
                "codec vocabulary rows must match model config; production requires the pinned raw tokenizer"
            )
        cfg = self.config
        width = cfg.facet_dim
        regional = cfg.n_regions * cfg.regional_dim
        self.root_prior = GaussianHead(width, cfg.root_dim, cfg.source_dim)
        self.regional_prior = GaussianHead(width + cfg.root_dim, regional, cfg.source_dim)
        self.posterior = (
            TargetPosterior(width, cfg.source_dim, cfg.root_dim, regional)
            if training_heads
            else None
        )
        self.control_projection = nn.Linear(cfg.control_dim, width, bias=False)
        self.regional_projection = nn.Linear(regional, width, bias=False)
        self.header_start = nn.Linear(width, width, bias=False)
        self.header_cell = nn.GRUCell(width, width)
        self.mode_codes = nn.Embedding(2, width)
        self.schema_codes = nn.Embedding(128, width)
        self.layout_codes = nn.Embedding(16, width)
        self.length_codes = nn.Embedding(cfg.max_capacity + 1, width)
        self.terminal_codes = nn.Embedding(2, width)
        self.repair_codes = nn.Embedding(2, width)
        self.identifier_slots = nn.Embedding(32, width)
        self.binding_query = nn.Linear(width, width, bias=False)
        self.free_binding = nn.Parameter(torch.zeros(width))
        self.realizer = ParallelRealizer(cfg)
        self.vocabulary = TiedVocabulary(width, cfg.vocab_size, cfg.output_rank)
        buckets = []
        value = 1
        while value < cfg.max_capacity:
            buckets.append(value)
            value *= 2
        buckets.append(cfg.max_capacity)
        self.capacities = tuple(buckets)
        self.bucket_codes = nn.Embedding(len(self.capacities), width)
        for module in (
            self.mode_codes,
            self.schema_codes,
            self.layout_codes,
            self.length_codes,
            self.terminal_codes,
            self.repair_codes,
            self.identifier_slots,
            self.bucket_codes,
        ):
            nn.init.normal_(module.weight, std=width**-0.5)
        self._allowed_ids = {}
        for kind in ("bytes", "utf8", "ascii", "identifier"):
            allowed = self.codec.allowed_ids(kind)
            self._allowed_ids[kind] = frozenset(allowed)
            mask = torch.zeros(cfg.vocab_size, dtype=torch.bool)
            mask[list(allowed)] = True
            self.register_buffer("valid_" + kind, mask, persistent=False)

    def _context(self, state, control=None) -> Tensor:
        context = state.facets.mean(1) + state.carry.mean(1)
        if control is not None:
            context = context + self.control_projection(control.to(context.dtype))
        return context

    @staticmethod
    def _logits(header: Tensor, codes: nn.Embedding) -> Tensor:
        return F.linear(header, codes.weight) / math.sqrt(header.shape[-1])

    def _condition(self, header: Tensor, codes: nn.Embedding, value: int | Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            value = torch.full((header.shape[0],), value, device=header.device, dtype=torch.long)
        return self.header_cell(codes(value), header)

    def _capacity_bucket(self, capacity: int) -> int:
        if not 1 <= capacity <= self.config.max_capacity:
            raise ValueError("requested capacity outside configured range")
        return next(index for index, upper in enumerate(self.capacities) if capacity <= upper)

    def _bucket_range(self, index: int, maximum: int, minimum: int = 1) -> range:
        lower = 1 if index == 0 else self.capacities[index - 1] + 1
        return range(max(lower, minimum), min(self.capacities[index], maximum) + 1)

    def _planned_fields(self, specs: Sequence[FieldSpec]) -> dict[str, _Field] | None:
        """Reserve causal minima before any mode, capacity, or binding choice.

        Every occurrence of a shared identifier needs the same eventual length.
        Reserving their intersected minimum together prevents a later identifier
        or copy choice from exhausting capacity already promised to another one.
        """
        fields = {
            spec.name: _Field(spec, reserved_minimum=int(spec.kind == "json")) for spec in specs
        }
        for value in fields.values():
            if value.spec.fixed_json is not None:
                value.fixed_ids = tuple(self.codec.encode(value.spec.fixed_json.encode()))
            elif value.minimum > value.spec.max_tokens:
                return None
        for definition in (field_.identifier for field_ in specs if field_.identifier):
            consumers = [
                value
                for value in fields.values()
                if value.spec.identifier == definition or value.spec.bound_to == definition
            ]
            lower = max(1, *(value.minimum for value in consumers))
            upper = min(value.spec.max_tokens for value in consumers)
            if lower > upper:
                return None
            for value in consumers:
                value.reserved_minimum = lower
        return fields

    @staticmethod
    def _byte_batch(payloads: Sequence[bytes], device) -> tuple[Tensor, Tensor, Tensor]:
        values, lengths = byte_tensor(payloads, device)
        mask = torch.arange(values.shape[1], device=values.device)[None] < lengths[:, None]
        return values, mask, lengths

    def _commit(self, state, payloads, content, root, role_ids=None):
        # Reconcile after validation and ingest the actual bytes exactly once.
        state = self.core.reconcile_carry(state, content, root.to(content.dtype))
        bytes_, _, lengths = self._byte_batch(payloads, content.device)
        if role_ids is None:
            role_ids = torch.full((len(payloads),), 2, device=content.device, dtype=torch.long)
        return self.core.ingest(
            bytes_, state, lengths=lengths, role_ids=role_ids, deterministic=True
        )

    def prepare_target(
        self,
        target_ids=None,
        *,
        target_bytes=None,
        free_mask=None,
        terminal=True,
        capacity=None,
        repair=False,
        role_ids=None,
    ) -> PreparedTargets:
        if (target_ids is None) == (target_bytes is None):
            raise ValueError("supply exactly one of target_ids or target_bytes")
        if target_bytes is not None:
            payloads = (target_bytes,) if isinstance(target_bytes, bytes) else tuple(target_bytes)
            ids = tuple(tuple(self.codec.encode(payload)) for payload in payloads)
        else:
            raw_ids = target_ids.tolist() if isinstance(target_ids, Tensor) else target_ids
            if not raw_ids or isinstance(raw_ids[0], int):
                raw_ids = [raw_ids]
            ids = tuple(tuple(int(value) for value in row) for row in raw_ids)
            payloads = tuple(self.codec.decode(row) for row in ids)
        if not payloads:
            raise ValueError("empty target batch")
        if free_mask is None:
            masks = tuple(tuple(True for _ in row) for row in ids)
        else:
            raw_masks = free_mask.tolist() if isinstance(free_mask, Tensor) else free_mask
            if not raw_masks or isinstance(raw_masks[0], (bool, int)):
                raw_masks = [raw_masks]
            masks = tuple(tuple(bool(value) for value in row) for row in raw_masks)
            if len(masks) != len(ids) or any(
                len(mask) != len(row) for mask, row in zip(masks, ids, strict=True)
            ):
                raise ValueError("free masks must match the actual target ID sequences")
        maximum = max(map(len, ids), default=0)
        capacity = capacity or next(
            (upper for upper in self.capacities if upper >= max(1, maximum)), None
        )
        if capacity is None or maximum > capacity:
            raise ValueError("target exceeds maximum capacity; split at causal semantic boundaries")
        self._capacity_bucket(capacity)
        batch = len(ids)
        terminals = (
            (bool(terminal),) * batch
            if isinstance(terminal, bool)
            else tuple(bool(value) for value in terminal)
        )
        if len(terminals) != batch:
            raise ValueError("one terminal target is required per episode")
        lexical = sum(sum(mask) for mask in masks) * (2 if repair else 1)
        gaussian = batch * (self.config.root_dim + self.config.n_regions * self.config.regional_dim)
        denominators = dict.fromkeys(LOSS_NAMES, 0)
        denominators.update(
            lexical=lexical,
            structure=batch * (5 + int(repair)),
            terminal=batch,
            prior_fit=gaussian,
            posterior_pressure=gaussian,
            content_alignment=batch * self.config.facet_dim,
            router_balance=batch,
            router_z=batch,
        )
        roles = (2,) * batch if role_ids is None else tuple(int(value) for value in role_ids)
        if len(roles) != batch:
            raise ValueError("one role ID is required per target")
        return PreparedTargets(
            tuple(payloads),
            ids,
            masks,
            terminals,
            capacity,
            bool(repair),
            denominators,
            lexical,
            roles,
        )

    def supervised_step(
        self,
        state,
        target_ids=None,
        *,
        target_bytes=None,
        free_mask=None,
        plan: PreparedTargets | None = None,
        prior_probability=0.5,
        effort=6,
        control=None,
        generator=None,
        role_ids=None,
    ) -> SupervisedResult:
        if self.posterior is None:
            raise RuntimeError(
                "supervised training requires training_heads=True; this is a served-only model"
            )
        plan = plan or self.prepare_target(
            target_ids, target_bytes=target_bytes, free_mask=free_mask
        )
        batch = len(plan.ids)
        if state.facets.shape[0] != batch:
            raise ValueError("target batch and independent recurrent state batch differ")
        if not 0 <= prior_probability <= 1:
            raise ValueError("prior_probability must be in [0,1]")
        context = self._context(state, control)
        target_bytes_, target_mask, _ = self._byte_batch(plan.payloads, context.device)
        target_content = self.posterior.encode(target_bytes_, target_mask)
        p_root = self.root_prior(context)
        q_root = self.posterior.root(torch.cat((context.detach(), target_content), -1))
        from_prior = (
            torch.rand((batch, 1), device=context.device, generator=generator) < prior_probability
        )
        root = torch.where(from_prior, p_root.sample(generator), q_root.sample(generator))
        prediction = self.core.begin_prediction(
            state, root.to(context.dtype), control=control, effort=effort, deterministic=True
        )
        prediction = self.core.advance_prediction(prediction, 7)
        header = self.header_start(prediction.facets.mean(1))
        zero = context.float().sum() * 0
        sums = dict.fromkeys(LOSS_NAMES, zero)

        def supervised_choice(codes, targets, key="structure", allowed=None):
            nonlocal header
            logits = self._logits(header, codes).float()
            if allowed is not None:
                mask = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
                mask[list(allowed)] = True
                logits = logits.masked_fill(~mask[None], -torch.inf)
            targets = (
                torch.full((batch,), targets, device=context.device, dtype=torch.long)
                if isinstance(targets, int)
                else targets
            )
            sums[key] = sums[key] + F.cross_entropy(logits, targets, reduction="sum")
            header = self._condition(header, codes, targets)

        # Raw records have no externally supplied schema. These two constrained
        # heads have probability one; typed action supervision learns alternatives.
        supervised_choice(self.mode_codes, 0, allowed=[0])
        supervised_choice(self.schema_codes, 0, allowed=[0])
        bucket = self._capacity_bucket(plan.capacity)
        supervised_choice(self.bucket_codes, bucket)
        prediction = self.core.advance_prediction(prediction, 8, conditioning=header)
        supervised_choice(
            self.length_codes,
            plan.capacity,
            allowed=self._bucket_range(bucket, self.config.max_capacity),
        )
        rows = self.realizer.begin(self.core, prediction, plan.capacity)
        prediction = self.core.advance_prediction(prediction, 10, conditioning=header)
        regional_context = prediction.facets.mean(1) + header
        p_regional = self.regional_prior(torch.cat((regional_context, root.to(context.dtype)), -1))
        q_regional = self.posterior.regional(
            torch.cat(
                (regional_context.detach(), target_content, root.detach().to(context.dtype)), -1
            )
        )
        regional = torch.where(
            from_prior, p_regional.sample(generator), q_regional.sample(generator)
        )
        regional_condition = header + self.regional_projection(regional.to(context.dtype))
        prediction = self.core.refresh_prediction(prediction, regional_condition)
        rows = self.realizer.finish_growth(
            self.core, prediction, rows, plan.capacity, regional_condition
        )
        # All capacity work is complete before the exact-length target exists.
        prediction = self.core.advance_prediction(prediction, 15)
        lengths = torch.tensor(
            [len(ids) for ids in plan.ids], device=context.device, dtype=torch.long
        )
        supervised_choice(self.length_codes, lengths, allowed=range(plan.capacity + 1))
        prediction = self.core.advance_prediction(prediction, 16, conditioning=header)
        terminals = torch.tensor(plan.terminals, device=context.device, dtype=torch.long)
        supervised_choice(self.terminal_codes, terminals, key="terminal")
        rows = self.realizer.refine(self.core, prediction, rows + header[:, None])
        all_ids = torch.zeros((batch, plan.capacity), device=context.device, dtype=torch.long)
        for index, ids in enumerate(plan.ids):
            all_ids[index, : len(ids)] = torch.tensor(ids, device=context.device, dtype=torch.long)
        # The target plan already knows this sparse support on the host. A fixed
        # index_select avoids the CUDA nonzero/synchronization implicit in rows[mask].
        free_indices = torch.tensor(
            [
                example * plan.capacity + position
                for example, masks in enumerate(plan.free_masks)
                for position, scored in enumerate(masks)
                if scored
            ],
            device=context.device,
            dtype=torch.long,
        )
        free_targets = all_ids.reshape(-1).index_select(0, free_indices)
        free_rows = rows.reshape(-1, rows.shape[-1]).index_select(0, free_indices)
        losses = self.vocabulary.loss(free_rows, free_targets, self.valid_bytes)
        sums["lexical"] = losses.sum()
        combined_lexical = losses
        if plan.repair:
            supervised_choice(self.repair_codes, 1)
            # A real sampled draft is exposed to the repair network. It never sees
            # the replacement target as input. Both lexical passes are charged.
            with torch.no_grad():
                sampled = self.vocabulary.sample(
                    rows.reshape(-1, rows.shape[-1]), self.valid_bytes, generator=generator
                )
            draft = self.vocabulary.embed(sampled.reshape(batch, plan.capacity))
            repaired = self.realizer.repair(self.core, prediction, rows, draft)
            repair_rows = repaired.reshape(-1, repaired.shape[-1]).index_select(0, free_indices)
            repair_losses = self.vocabulary.loss(repair_rows, free_targets, self.valid_bytes)
            sums["lexical"] = sums["lexical"] + repair_losses.sum()
            combined_lexical = combined_lexical + repair_losses
            rows = repaired
        # KL directions must not acquire a hidden path through reparameterized
        # ancestor innovations. Recompute the tiny heads on detached ancestors;
        # the lexical/served likelihood paths retain their ordinary gradients.
        fit_root = self.root_prior(context.detach())
        fit_regional = self.regional_prior(
            torch.cat((regional_context.detach(), root.detach().to(context.dtype)), -1)
        )
        sums["prior_fit"] = (
            q_root.detached().kl(fit_root).sum() + q_regional.detached().kl(fit_regional).sum()
        )
        sums["posterior_pressure"] = (
            q_root.kl(p_root.detached()).sum() + q_regional.kl(p_regional.detached()).sum()
        )
        # The target encoder is detached; the training-only alignment map and the
        # served content representation learn. Private carry is absent.
        content = prediction.facets.mean(1)
        sums["content_alignment"] = (
            (content.float() - self.posterior.alignment(target_content.detach()).float())
            .square()
            .sum()
        )
        sums["router_balance"], sums["router_z"] = self._routing_losses(prediction, batch)
        role_ids = (
            torch.tensor(plan.roles or (2,) * batch, device=context.device)
            if role_ids is None
            else role_ids
        )
        actual_mask = torch.arange(plan.capacity, device=context.device)[None] < lengths[:, None]
        actual_embeddings = self.vocabulary.embed(all_ids) * actual_mask[..., None]
        actual_summary = actual_embeddings.sum(1) / lengths[:, None].clamp_min(1)
        next_state = self._commit(
            state,
            plan.payloads,
            content + rows.mean(1) + actual_summary + header,
            root,
            role_ids=role_ids,
        )
        prior_positions = from_prior[:, 0].index_select(0, free_indices // plan.capacity)
        posterior_positions = ~prior_positions
        detached_lexical = combined_lexical.detach()
        metrics = {
            "scored_positions": plan.scored_positions,
            "capacity": plan.capacity,
            "prior_examples": from_prior.sum().detach(),
            "posterior_examples": (~from_prior).sum().detach(),
            "prior_lexical_positions": prior_positions.sum().detach() * (1 + int(plan.repair)),
            "posterior_lexical_positions": posterior_positions.sum().detach()
            * (1 + int(plan.repair)),
            "prior_lexical_loss_sum": (detached_lexical * prior_positions).sum(),
            "posterior_lexical_loss_sum": (detached_lexical * posterior_positions).sum(),
            "posterior_bytes_encoded": sum(map(len, plan.payloads)),
            "capacity_positions": batch * plan.capacity,
            "lexical_passes": 1 + int(plan.repair),
        }
        return SupervisedResult(
            {name: (sums[name], plan.denominators[name]) for name in LOSS_NAMES},
            next_state,
            metrics,
        )

    def _routing_losses(self, prediction, batch=1):
        programs = (*prediction.previous_programs, prediction.program)
        balance = torch.stack([program.balance_loss for program in programs]).mean()
        stabilization = torch.stack([program.stabilization_loss for program in programs]).mean()
        u_balance, u_stabilization = self.core.u_routing_losses()
        factual = prediction.factual.last_program
        if factual is not None:
            balance = balance + factual.balance_loss
            stabilization = stabilization + factual.stabilization_loss
        return (balance + u_balance) * batch, (stabilization + u_stabilization) * batch

    def _source_ids(self, source: CopySource) -> tuple[int, ...]:
        ids = source.emitted_ids or tuple(self.codec.encode(source.payload))
        if self.codec.decode(ids) != source.payload:
            raise ValueError(f"copy source {source.name} IDs do not decode to its actual bytes")
        return ids

    def _field_mask(self, field: FieldSpec) -> Tensor:
        return getattr(self, "valid_" + ("utf8" if field.kind == "json" else field.kind))

    @staticmethod
    def _escaped_piece(piece: bytes) -> bytes:
        escapes = {
            8: b"\\b",
            9: b"\\t",
            10: b"\\n",
            12: b"\\f",
            13: b"\\r",
            34: b'\\"',
            92: b"\\\\",
        }
        return b"".join(
            escapes.get(value, (f"\\u{value:04x}".encode() if value < 32 else bytes([value])))
            for value in piece
        )

    def _serialize_ids(
        self,
        schema: SchemaSpec | None,
        fields: dict[str, _Field],
        ids: dict[str, tuple[int, ...]],
        layout: int,
    ) -> tuple[bytes, tuple[int, ...]]:
        values = {name: self.codec.decode(value) for name, value in ids.items()}
        for name, field_ in fields.items():
            if field_.spec.kind in {"utf8", "json"}:
                values[name].decode("utf-8", errors="strict")
        if schema is None:
            return values["text"], ids["text"]
        payload = schema.serialize(values, layout)
        emitted: list[int] = self.codec.encode(b"{")
        for index, name in enumerate(schema.orders[layout]):
            field_ = fields[name]
            if index:
                emitted.extend(self.codec.encode(b","))
            emitted.extend(self.codec.encode(json.dumps(name, ensure_ascii=False).encode() + b":"))
            if field_.spec.kind == "json" or field_.spec.fixed_json is not None:
                emitted.extend(ids[name])
            else:
                emitted.extend(self.codec.encode(b'"'))
                for token in ids[name]:
                    piece = self.codec.token_bytes(token)
                    escaped = self._escaped_piece(piece)
                    # Preserve the emitted segmentation wherever syntax does not
                    # transform a byte. Escapes are deterministic serializer IDs.
                    emitted.extend((token,) if piece == escaped else self.codec.encode(escaped))
                emitted.extend(self.codec.encode(b'"'))
        emitted.extend(self.codec.encode(b"}"))
        if self.codec.decode(emitted) != payload:
            raise RuntimeError("serializer byte/ID paths disagree")
        return payload, tuple(emitted)

    def sample(
        self,
        state,
        *,
        spec: ActionSpec | None = None,
        control=None,
        effort=6,
        generator=None,
        temperature=1.0,
        top_k=None,
        top_p=None,
        rl=False,
    ) -> PolicyResult:
        if temperature != 1.0 or top_k is not None or top_p is not None:
            raise ValueError("augmented policy sampling uses temperature 1 and no truncation")
        if rl and effort != 6:
            raise ValueError("initial outcome RL requires fixed effort 6")
        with torch.no_grad():
            return self._execute(
                state, spec=spec, control=control, effort=effort, generator=generator
            )

    def score_tape(
        self, state, tape: ActionTape, *, spec: ActionSpec | None = None, control=None
    ) -> PolicyResult:
        return self._execute(state, spec=spec, control=control, effort=tape.effort, tape=tape)

    def _execute(
        self,
        state,
        *,
        spec=None,
        control=None,
        effort=6,
        generator=None,
        tape=None,
        teacher: PreparedAction | None = None,
        prior_probability=0.5,
        commit=True,
    ) -> PolicyResult:
        if state.facets.shape[0] != 1:
            raise ValueError(
                "sample/score/typed supervision consume one episode; batch raw supervision uses supervised_step"
            )
        spec = spec or ActionSpec(max_tokens=self.config.max_capacity)
        if len(spec.schemas) > self.schema_codes.num_embeddings:
            raise ValueError("too many simultaneously available schemas")
        if tape is not None:
            if tape.version != TAPE_VERSION or tape.codec_fingerprint != self.codec.fingerprint:
                raise TapeMismatch("tape policy/codec version differs")
            if tape.spec_fingerprint != spec.fingerprint:
                raise TapeMismatch("tape public constraints or causal copy sources differ")
            if (
                len(tape.root) != self.config.root_dim
                or len(tape.regional) != self.config.n_regions * self.config.regional_dim
            ):
                raise TapeMismatch("tape innovation dimensions differ")
        context = self._context(state, control)
        events = _Events(context, tape=tape, teacher=teacher, generator=generator)
        p_root = self.root_prior(context)
        target_content = None
        q_root = None
        from_prior = True
        if teacher is not None:
            raw, mask, _ = self._byte_batch([teacher.payload], context.device)
            target_content = self.posterior.encode(raw, mask)
            q_root = self.posterior.root(torch.cat((context.detach(), target_content), -1))
            from_prior = bool(
                torch.rand((), device=context.device, generator=generator) < prior_probability
            )
            root = p_root.sample(generator) if from_prior else q_root.sample(generator)
            events.add(
                "prior_fit",
                q_root.detached().kl(self.root_prior(context.detach())).sum(),
                self.config.root_dim,
            )
            events.add(
                "posterior_pressure", q_root.kl(p_root.detached()).sum(), self.config.root_dim
            )
        elif tape is not None:
            root = torch.tensor(tape.root, device=context.device, dtype=torch.float32)[None]
        else:
            root = p_root.sample(generator).detach()
        if not bool(torch.isfinite(root).all()):
            raise TapeMismatch("nonfinite root innovation")
        events.log_prob = events.log_prob + p_root.log_prob(root).sum()
        prediction = self.core.begin_prediction(
            state, root.to(context.dtype), control=control, effort=effort, deterministic=True
        )
        prediction = self.core.advance_prediction(prediction, 7)
        header = self.header_start(prediction.facets.mean(1))

        def choose(name, codes, allowed, forced=None, key="structure"):
            nonlocal header
            value = events.choose(
                name, self._logits(header, codes)[0], allowed, forced=forced, key=key
            )
            header = self._condition(header, codes, value)
            return value

        token_limit = min(spec.max_tokens, self.config.max_capacity)
        schema_fields = {
            index: self._planned_fields(schema.fields) for index, schema in enumerate(spec.schemas)
        }
        feasible_schemas = [
            index
            for index, values in schema_fields.items()
            if values is not None and sum(value.minimum for value in values.values()) <= token_limit
        ]
        text_fields = self._planned_fields(
            (
                FieldSpec(
                    "text",
                    kind=spec.text_kind,
                    min_tokens=spec.min_tokens,
                    max_tokens=spec.max_tokens,
                ),
            )
        )
        text_feasible = spec.allow_text and spec.min_tokens <= token_limit
        modes = ([0] if text_feasible else []) + ([1] if feasible_schemas else [])
        if not modes:
            raise ValueError(
                "no output mode or schema fits the configured token capacity and field constraints"
            )
        mode = choose("mode", self.mode_codes, modes, forced=1 if teacher else None)
        schema_index = choose(
            "schema",
            self.schema_codes,
            feasible_schemas if mode else [0],
            forced=teacher.schema_index if teacher else None,
        )
        schema = spec.schemas[schema_index] if mode else None
        if schema is not None and len(schema.orders) > self.layout_codes.num_embeddings:
            raise ValueError("too many schema layouts")
        fields = schema_fields[schema_index] if mode else text_fields
        minimum = max(1, sum(value.minimum for value in fields.values()))
        maximum = max(1, token_limit)
        bucket_options = [
            index
            for index in range(len(self.capacities))
            if self._bucket_range(index, maximum, minimum)
        ]
        bucket = choose(
            "size_bucket",
            self.bucket_codes,
            bucket_options,
            forced=self._capacity_bucket(teacher.capacity) if teacher else None,
        )
        prediction = self.core.advance_prediction(prediction, 8, conditioning=header)
        capacity = choose(
            "capacity",
            self.length_codes,
            self._bucket_range(bucket, maximum, minimum),
            forced=teacher.capacity if teacher else None,
        )
        rows = self.realizer.begin(self.core, prediction, capacity)
        prediction = self.core.advance_prediction(prediction, 10, conditioning=header)
        source_ids = tuple(self._source_ids(source) for source in spec.copy_sources)
        source_embeddings = [self.free_binding]
        for ids in source_ids:
            if ids:
                tensor = torch.tensor(ids, device=context.device, dtype=torch.long)
                source_embeddings.append(self.vocabulary.embed(tensor).mean(0))
            else:
                source_embeddings.append(torch.zeros_like(self.free_binding))
        source_keys = torch.stack(source_embeddings)
        identifiers: dict[str, tuple[int, ...]] = {}
        identifier_draws: list[LexicalDraw] = []

        def minimum_total():
            return sum(value.minimum for value in fields.values())

        def identifier_constraints(definition):
            consumers = [
                value.spec
                for value in fields.values()
                if value.spec.identifier == definition or value.spec.bound_to == definition
            ]
            return (
                max(field_.min_tokens for field_ in consumers),
                min(field_.max_tokens for field_ in consumers),
                len(consumers),
            )

        def identifier_reserved(definition):
            return sum(
                value.minimum
                for value in fields.values()
                if value.spec.identifier == definition or value.spec.bound_to == definition
            )

        def identifier_kind(definition):
            kinds = {
                value.spec.kind
                for value in fields.values()
                if value.spec.identifier == definition or value.spec.bound_to == definition
            }
            return (
                "identifier" if "identifier" in kinds else "ascii" if "ascii" in kinds else "bytes"
            )

        for slot, value in enumerate(fields.values()):
            field_ = value.spec
            if value.fixed_ids is not None:
                continue
            if field_.bound_to:
                value.fixed_ids = identifiers[field_.bound_to]
                continue
            if field_.allow_copy and source_ids:
                allowed = [0]
                for index, ids in enumerate(source_ids, 1):
                    kind = (
                        identifier_kind(field_.identifier)
                        if field_.identifier
                        else ("utf8" if field_.kind == "json" else field_.kind)
                    )
                    if len(ids) > field_.max_tokens or not all(
                        token in self._allowed_ids[kind] for token in ids
                    ):
                        continue
                    previous = value.copy_ids
                    value.copy_ids = ids
                    feasible = minimum_total() <= capacity
                    if field_.identifier:
                        lower, upper, multiplicity = identifier_constraints(field_.identifier)
                        lower = max(lower, 1)
                        feasible = feasible and lower <= len(ids) <= upper
                        feasible = (
                            feasible
                            and minimum_total()
                            - identifier_reserved(field_.identifier)
                            + multiplicity * len(ids)
                            <= capacity
                        )
                    value.copy_ids = previous
                    if feasible:
                        allowed.append(index)
                logits = (source_keys @ self.binding_query(header)[0]) / math.sqrt(
                    context.shape[-1]
                )
                selected = events.choose(
                    "binding/" + field_.name,
                    logits,
                    allowed,
                    key="binding",
                    forced=teacher.bindings.get(field_.name, 0) if teacher else None,
                )
                header = self.header_cell(source_keys[selected][None], header)
                value.copy_ids = source_ids[selected - 1] if selected else ()
                if field_.identifier and selected:
                    value.fixed_ids = value.copy_ids
            if field_.identifier:
                if value.fixed_ids is None:
                    lower, upper, multiplicity = identifier_constraints(field_.identifier)
                    lower = max(lower, 1)
                    available = capacity - minimum_total() + identifier_reserved(field_.identifier)
                    upper = min(upper, available // multiplicity)
                    length = choose(
                        "identifier_length/" + field_.name,
                        self.length_codes,
                        range(lower, upper + 1),
                        forced=len(teacher.field_ids[field_.name]) if teacher else None,
                    )
                    id_rows = header[:, None] + self.identifier_slots.weight[slot % 32][None, None]
                    id_rows = (
                        id_rows
                        + positional_features(
                            length, context.shape[-1], device=context.device, dtype=context.dtype
                        )[None]
                    )
                    supplied = None
                    if tape is not None:
                        if len(identifier_draws) >= len(tape.identifiers):
                            raise TapeMismatch("missing early identifier draw")
                        supplied = tape.identifiers[len(identifier_draws)]
                    draw = events.lexical(
                        "identifier/" + field_.name,
                        self,
                        id_rows[0],
                        getattr(self, "valid_" + identifier_kind(field_.identifier)),
                        supplied=supplied,
                        forced=teacher.field_ids[field_.name] if teacher else None,
                    )
                    identifier_draws.append(draw)
                    value.fixed_ids = draw.ids
                    header = self.header_cell(
                        self.vocabulary.embed(torch.tensor(draw.ids, device=context.device)).mean(
                            0
                        )[None],
                        header,
                    )
                identifiers[field_.identifier] = value.fixed_ids
                # Reserve exact subsequent occurrences before other decisions.
                for other in fields.values():
                    if other.spec.bound_to == field_.identifier:
                        other.fixed_ids = value.fixed_ids
        if tape is not None and len(identifier_draws) != len(tape.identifiers):
            raise TapeMismatch("tape contains unused identifier draws")
        regional_context = prediction.facets.mean(1) + header
        p_regional = self.regional_prior(torch.cat((regional_context, root.to(context.dtype)), -1))
        if teacher is not None:
            q_regional = self.posterior.regional(
                torch.cat(
                    (regional_context.detach(), target_content, root.detach().to(context.dtype)), -1
                )
            )
            regional = p_regional.sample(generator) if from_prior else q_regional.sample(generator)
            dimensions = self.config.n_regions * self.config.regional_dim
            fit_regional = self.regional_prior(
                torch.cat((regional_context.detach(), root.detach().to(context.dtype)), -1)
            )
            events.add("prior_fit", q_regional.detached().kl(fit_regional).sum(), dimensions)
            events.add("posterior_pressure", q_regional.kl(p_regional.detached()).sum(), dimensions)
        elif tape is not None:
            regional = torch.tensor(tape.regional, device=context.device, dtype=torch.float32)[None]
        else:
            regional = p_regional.sample(generator).detach()
        if not bool(torch.isfinite(regional).all()):
            raise TapeMismatch("nonfinite regional innovation")
        events.log_prob = events.log_prob + p_regional.log_prob(regional).sum()
        conditioning = header + self.regional_projection(regional.to(context.dtype))
        prediction = self.core.refresh_prediction(prediction, conditioning)
        rows = self.realizer.finish_growth(self.core, prediction, rows, capacity, conditioning)
        prediction = self.core.advance_prediction(prediction, 15)
        layout = (
            choose(
                "layout",
                self.layout_codes,
                range(len(schema.orders)),
                forced=teacher.layout if teacher else None,
            )
            if schema is not None
            else 0
        )
        assigned = 0
        field_list = list(fields.values())
        for index, value in enumerate(field_list):
            if value.fixed_ids is not None:
                value.length = len(value.fixed_ids)
            else:
                future_minimum = sum(other.minimum for other in field_list[index + 1 :])
                upper = min(value.spec.max_tokens, capacity - assigned - future_minimum)
                value.length = choose(
                    "length/" + value.spec.name,
                    self.length_codes,
                    range(value.minimum, upper + 1),
                    forced=len(teacher.field_ids[value.spec.name]) if teacher else None,
                )
                if value.copy_ids:
                    value.placement = choose(
                        "placement/" + value.spec.name,
                        self.length_codes,
                        range(value.length - len(value.copy_ids) + 1),
                        key="placement",
                        forced=teacher.placements[value.spec.name] if teacher else None,
                    )
            assigned += value.length
        offset = 0
        for name in schema.orders[layout] if schema else ("text",):
            fields[name].offset = offset
            offset += fields[name].length
        if offset > capacity:
            raise TapeMismatch("layout exceeds chosen capacity")
        prediction = self.core.advance_prediction(prediction, 16, conditioning=header)
        terminal_options = [0, 1] if spec.require_terminal is None else [int(spec.require_terminal)]
        terminal = bool(
            choose(
                "terminal",
                self.terminal_codes,
                terminal_options,
                key="terminal",
                forced=(1 if spec.require_terminal is None else int(spec.require_terminal))
                if teacher
                else None,
            )
        )
        rows = self.realizer.refine(self.core, prediction, rows + header[:, None])
        attempts: list[DraftAttempt] = []
        payload, emitted_ids = b"", ()
        all_ids: dict[str, tuple[int, ...]] = {}
        draft_embeddings = torch.zeros_like(rows)
        valid = False
        repair_targets = []
        max_repairs = 0 if teacher is not None else spec.max_repairs
        for attempt in range(max_repairs + 1):
            if attempt:
                rows = self.realizer.repair(self.core, prediction, rows, draft_embeddings)
            draws: list[LexicalDraw] = []
            supplied_attempt = None
            if tape is not None:
                if attempt >= len(tape.attempts):
                    raise TapeMismatch("missing draft or repair attempt")
                supplied_attempt = tape.attempts[attempt]
            for value in fields.values():
                if value.fixed_ids is not None:
                    all_ids[value.spec.name] = value.fixed_ids
                    continue
                positions = torch.tensor(
                    value.free_positions, device=context.device, dtype=torch.long
                )
                supplied = None
                if supplied_attempt is not None:
                    if len(draws) >= len(supplied_attempt.draws):
                        raise TapeMismatch("missing field draw")
                    supplied = supplied_attempt.draws[len(draws)]
                forced = None
                if teacher is not None:
                    target_ids = teacher.field_ids[value.spec.name]
                    forced = (
                        target_ids[: value.placement]
                        + target_ids[value.placement + len(value.copy_ids) :]
                    )
                    repair_targets.append((value.free_positions, forced, value.spec.kind))
                draw = events.lexical(
                    f"draft/{attempt}/{value.spec.name}",
                    self,
                    rows[0, positions],
                    self._field_mask(value.spec),
                    supplied=supplied,
                    forced=forced,
                )
                draws.append(draw)
                all_ids[value.spec.name] = value.assemble(draw.ids)
            error = None
            try:
                payload, emitted_ids = self._serialize_ids(schema, fields, all_ids, layout)
                valid = True
            except (ValueError, UnicodeError) as exc:
                valid = False
                error = type(exc).__name__
                payload, emitted_ids = b"", ()
            except Exception as exc:
                # jsonschema validation errors are third-party exceptions; avoid
                # swallowing implementation errors behind a generic failed draft.
                if type(exc).__module__.startswith("jsonschema"):
                    valid, error, payload, emitted_ids = False, "SchemaValidationError", b"", ()
                else:
                    raise
            actual_attempt = DraftAttempt(tuple(draws), valid, error)
            if supplied_attempt is not None:
                if actual_attempt != supplied_attempt:
                    raise TapeMismatch(
                        "draft validation or lexical event list differs during replay"
                    )
            attempts.append(actual_attempt)
            if attempt == max_repairs:
                break
            if not valid:
                repair_options = [1]
            elif spec.repair_probability == 0:
                repair_options = [0]
            elif spec.repair_probability == 1:
                repair_options = [1]
            else:
                repair_options = [0, 1]
            repair_logits = self._logits(header, self.repair_codes)[0].float()
            if 0 < spec.repair_probability < 1:
                repair_logits = repair_logits + repair_logits.new_tensor(
                    [0, math.log(spec.repair_probability / (1 - spec.repair_probability))]
                )
            repair = events.choose(f"repair/{attempt}", repair_logits, repair_options)
            header = self._condition(header, self.repair_codes, repair)
            if not repair:
                break
            draft_embeddings = torch.zeros_like(rows)
            for value in fields.values():
                ids = all_ids[value.spec.name]
                if ids:
                    token_ids = torch.tensor(ids, device=context.device, dtype=torch.long)
                    draft_embeddings[0, value.offset : value.offset + value.length] = (
                        self.vocabulary.embed(token_ids)
                    )
        if tape is not None and len(attempts) != len(tape.attempts):
            raise TapeMismatch("tape contains unused failed/repair attempts")
        if teacher is not None and (not valid or payload != teacher.payload):
            raise ValueError(
                "structured target must match deterministic schema serialization exactly"
            )
        content = prediction.facets.mean(1)
        if teacher is not None:
            alignment = (
                (content.float() - self.posterior.alignment(target_content.detach()).float())
                .square()
                .sum()
            )
            events.add("content_alignment", alignment, self.config.facet_dim)
        balance, stabilization = self._routing_losses(prediction)
        events.add("router_balance", balance, 1)
        events.add("router_z", stabilization, 1)
        action_ids = [token for draw in identifier_draws for token in draw.ids]
        action_ids.extend(
            token for attempt_ in attempts for draw in attempt_.draws for token in draw.ids
        )
        action_summary = torch.zeros_like(content)
        if action_ids:
            sampled_ids = torch.tensor(action_ids, device=context.device, dtype=torch.long)
            action_summary = self.vocabulary.embed(sampled_ids).mean(0)[None]
        carry_content = content + rows.mean(1) + action_summary + header
        # Failed drafts never enter factual history and are never executed. Their
        # complete attempts still update private carry for subsequent recovery.
        if not commit:
            next_state = state
        else:
            next_state = (
                self._commit(state, [payload], carry_content, root)
                if valid
                else self.core.reconcile_carry(state, carry_content, root.to(context.dtype))
            )
        final_draft = torch.zeros_like(rows)
        for value in fields.values():
            ids = all_ids[value.spec.name]
            if ids:
                token_ids = torch.tensor(ids, device=context.device, dtype=torch.long)
                final_draft[0, value.offset : value.offset + value.length] = self.vocabulary.embed(
                    token_ids
                )
        events.finish()
        result_tape = ActionTape(
            TAPE_VERSION,
            self.codec.fingerprint,
            spec.fingerprint,
            effort,
            tuple(root.detach()[0].tolist()),
            tuple(regional.detach()[0].tolist()),
            tuple(events.choices),
            tuple(identifier_draws),
            tuple(attempts),
            payload,
            emitted_ids,
            valid,
            terminal,
            float(events.log_prob.detach().item()),
            events.scored_positions,
        )
        if tape is not None:
            if (
                tape.payload,
                tape.emitted_ids,
                tape.committed,
                tape.terminal,
                tape.scored_positions,
            ) != (payload, emitted_ids, valid, terminal, events.scored_positions):
                raise TapeMismatch("committed bytes/IDs or action accounting differ")
        return PolicyResult(
            result_tape,
            payload,
            emitted_ids,
            next_state,
            events.log_prob,
            events.scored_positions,
            {name: (value[0], value[1]) for name, value in events.losses.items()},
            prediction,
            p_root,
            p_regional,
            root,
            regional_context,
            rows,
            header,
            tuple(repair_targets),
            final_draft,
            from_prior if teacher is not None else None,
        )

    def prepare_action_target(
        self,
        payload: bytes,
        *,
        spec: ActionSpec,
        schema_index=0,
        bindings: Mapping[str, int] | None = None,
        capacity=None,
        repair=False,
        layout=0,
    ) -> PreparedAction:
        """Prepare typed supervision from a schema that was available causally.

        Fixed JSON syntax is canonicalized explicitly. The returned ``payload``
        is the teacher-forced action to commit; callers retain original record
        bytes/provenance separately. JSON values keep their actual API types.
        Bindings index ``copy_sources`` starting at one; zero means generate.
        """
        if not 0 <= schema_index < len(spec.schemas):
            raise ValueError("selected schema is not available in the public action spec")
        schema = spec.schemas[schema_index]
        parsed = strict_json_loads(payload.decode("utf-8", errors="strict"))
        if not isinstance(parsed, dict) or set(parsed) != {field_.name for field_ in schema.fields}:
            raise ValueError("target JSON object does not match schema fields")
        values: dict[str, bytes] = {}
        ids: dict[str, tuple[int, ...]] = {}
        identifier_values: dict[str, tuple[int, ...]] = {}
        declared_bindings = dict(bindings or {})
        placements: dict[str, int] = {}
        lexical = 0
        late_lexical = 0
        structure = 5  # mode, schema, bucket, capacity, layout
        binding_count = 0
        placement_count = 0
        sources = tuple(self._source_ids(source) for source in spec.copy_sources)
        for field_ in schema.fields:
            value = parsed[field_.name]
            if field_.fixed_json is not None:
                if strict_json_loads(field_.fixed_json) != value:
                    raise ValueError(
                        f"target fixed field {field_.name} differs from declared syntax"
                    )
                raw = field_.fixed_json.encode()
            elif field_.kind == "json":
                raw = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode()
            else:
                if not isinstance(value, str):
                    raise ValueError(f"string field {field_.name} received {type(value).__name__}")
                raw = value.encode("utf-8")
            values[field_.name] = raw
            ids[field_.name] = tuple(self.codec.encode(raw))
            if field_.fixed_json is not None:
                continue
            if not field_.min_tokens <= len(ids[field_.name]) <= field_.max_tokens:
                raise ValueError(f"target field {field_.name} exceeds its token limits")
            kind = "utf8" if field_.kind == "json" else field_.kind
            if not all(token in self._allowed_ids[kind] for token in ids[field_.name]):
                raise ValueError(f"target field {field_.name} violates its lexical alphabet")
            if field_.bound_to:
                if self.codec.decode(identifier_values[field_.bound_to]) != raw:
                    raise ValueError(f"bound field {field_.name} differs from its identifier")
                ids[field_.name] = identifier_values[field_.bound_to]
                continue
            selected = declared_bindings.get(field_.name, 0)
            if selected and (not field_.allow_copy or not 1 <= selected <= len(sources)):
                raise ValueError(f"invalid copy binding for {field_.name}")
            if field_.allow_copy and sources:
                binding_count += 1
            copied = sources[selected - 1] if selected else ()
            if copied:
                copied_bytes = self.codec.decode(copied)
                position_bytes = raw.find(copied_bytes)
                if position_bytes < 0:
                    raise ValueError(f"declared copy is absent from target field {field_.name}")
                before = tuple(self.codec.encode(raw[:position_bytes]))
                after = tuple(self.codec.encode(raw[position_bytes + len(copied_bytes) :]))
                if field_.identifier and (before or after):
                    raise ValueError("an early identifier copy must bind the complete value")
                ids[field_.name] = before + copied + after
                placements[field_.name] = len(before)
            if field_.identifier:
                if not ids[field_.name]:
                    raise ValueError("early identifiers must contain at least one token")
                if not copied:
                    structure += 1  # early identifier length
                    lexical += len(ids[field_.name])
                identifier_values[field_.identifier] = ids[field_.name]
            else:
                structure += 1  # feasible field length
                free_count = len(ids[field_.name]) - len(copied)
                lexical += free_count
                late_lexical += free_count
                if copied:
                    placement_count += 1
        # Canonical fixed syntax is a deliberate, visible preparation transform.
        payload = schema.serialize(values, layout)
        maximum = sum(len(row) for row in ids.values())
        capacity = capacity or next(
            (upper for upper in self.capacities if upper >= max(1, maximum)), None
        )
        if (
            capacity is None
            or maximum > capacity
            or capacity > min(spec.max_tokens, self.config.max_capacity)
        ):
            raise ValueError("typed action exceeds allowed generated capacity")
        self._capacity_bucket(capacity)
        gaussian = self.config.root_dim + self.config.n_regions * self.config.regional_dim
        denominators = dict.fromkeys(LOSS_NAMES, 0)
        denominators.update(
            lexical=lexical + int(repair) * late_lexical,
            structure=structure + int(repair),
            binding=binding_count,
            placement=placement_count,
            terminal=1,
            prior_fit=gaussian,
            posterior_pressure=gaussian,
            content_alignment=self.config.facet_dim,
            router_balance=1,
            router_z=1,
        )
        return PreparedAction(
            payload,
            spec,
            schema_index,
            layout,
            capacity,
            ids,
            declared_bindings,
            placements,
            bool(repair),
            denominators,
            denominators["lexical"],
        )

    def supervised_action(
        self,
        state,
        *,
        plan: PreparedAction,
        prior_probability=0.5,
        effort=6,
        control=None,
        generator=None,
    ) -> SupervisedResult:
        if self.posterior is None:
            raise RuntimeError(
                "supervised training requires training_heads=True; this is a served-only model"
            )
        if not 0 <= prior_probability <= 1:
            raise ValueError("prior_probability must be in [0,1]")
        result = self._execute(
            state,
            spec=plan.spec,
            teacher=plan,
            prior_probability=prior_probability,
            effort=effort,
            control=control,
            generator=generator,
            commit=not plan.repair,
        )
        losses = result.losses
        if plan.repair:
            logits = self._logits(result.header, self.repair_codes)
            repair_loss = F.cross_entropy(
                logits.float(),
                torch.ones(1, device=logits.device, dtype=torch.long),
                reduction="sum",
            )
            previous, count = losses["structure"]
            losses["structure"] = previous + repair_loss, count + 1
            # Genuine sampled corruption, not target embeddings, is the repair
            # input. Identifiers and copies remain bound and are never rescored.
            draft = result.draft_embeddings.clone()
            for positions, _, kind in result.repair_targets:
                selected = torch.tensor(positions, device=logits.device, dtype=torch.long)
                valid = getattr(self, "valid_" + ("utf8" if kind == "json" else kind))
                with torch.no_grad():
                    sampled = self.vocabulary.sample(
                        result.lexical_rows[0, selected], valid, generator=generator
                    )
                draft[0, selected] = self.vocabulary.embed(sampled)
            repaired = self.realizer.repair(
                self.core, result.prediction, result.lexical_rows, draft
            )
            previous, count = losses["lexical"]
            for positions, target_ids, kind in result.repair_targets:
                selected = torch.tensor(positions, device=logits.device, dtype=torch.long)
                target = torch.tensor(target_ids, device=logits.device, dtype=torch.long)
                valid = getattr(self, "valid_" + ("utf8" if kind == "json" else kind))
                previous = (
                    previous + self.vocabulary.loss(repaired[0, selected], target, valid).sum()
                )
                count += len(target_ids)
            losses["lexical"] = previous, count
            carry_content = (
                result.prediction.facets.mean(1) + repaired.mean(1) + draft.mean(1) + result.header
            )
            result.state = self._commit(state, [plan.payload], carry_content, result.root_value)
        for name in LOSS_NAMES:
            if losses[name][1] != plan.denominators[name]:
                raise RuntimeError(
                    f"prepared {name} denominator {plan.denominators[name]} differs from actual {losses[name][1]}"
                )
        lexical_sum = losses["lexical"][0].detach()
        return SupervisedResult(
            losses,
            result.state,
            {
                "scored_positions": float(plan.scored_positions),
                "capacity": float(plan.capacity),
                "structured_actions": 1.0,
                "prior_examples": float(result.used_prior),
                "posterior_examples": float(not result.used_prior),
                "prior_lexical_positions": float(plan.scored_positions if result.used_prior else 0),
                "posterior_lexical_positions": float(
                    0 if result.used_prior else plan.scored_positions
                ),
                "prior_lexical_loss_sum": lexical_sum if result.used_prior else 0.0,
                "posterior_lexical_loss_sum": 0.0 if result.used_prior else lexical_sum,
                "posterior_bytes_encoded": float(len(plan.payload)),
                "capacity_positions": float(plan.capacity),
                "lexical_passes": float(1 + int(plan.repair)),
            },
        )

    def replay(
        self,
        initial_state,
        records: Sequence[bytes | Observation | ActionTape],
        *,
        specs=None,
        controls=None,
    ):
        """Rebuild every candidate state from actual observations and full tapes.

        This function retains the full differentiable short-episode graph. A
        caller adding detach boundaries must label TBPTT's biased approximation.
        """
        state = initial_state
        log_probs = []
        decision = 0
        for record in records:
            if isinstance(record, (bytes, Observation)):
                observation = Observation(record) if isinstance(record, bytes) else record
                raw, _, lengths = self._byte_batch([observation.payload], state.facets.device)
                role = torch.full((1,), observation.role_id, device=raw.device, dtype=torch.long)
                state = self.core.ingest(
                    raw, state, lengths=lengths, role_ids=role, deterministic=True
                )
            else:
                result = self.score_tape(
                    state,
                    record,
                    spec=None if specs is None else specs[decision],
                    control=None if controls is None else controls[decision],
                )
                state = result.state
                log_probs.append(result.log_prob)
                decision += 1
        return state, tuple(log_probs)

    def served_state_dict(self) -> dict[str, Tensor]:
        """Export without posterior/target assets; ordinary strict loading applies.

        Engram adapters must be baked by the core before this method is called.
        No donor extraction or training-only fallback runs implicitly at serve.
        """
        if self.core.frontend.ngrams.import_map is not None:
            raise RuntimeError("bake the engram adapter before served export")
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if not name.startswith("posterior.")
        }
