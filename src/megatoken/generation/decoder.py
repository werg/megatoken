"""A single deterministic action program for serving and teacher-forced learning."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.model.layers import position_features

from .distributions import TiedVocabulary
from .feasibility import wire_overhead_tokens, wire_witness_tokens
from .realizer import ParallelRealizer
from .structure import ActionSpec, Decision, FieldSpec
from .targets import PreparedAction, PreparedTargets, planned_fields, source_ids, workspace_capacity

DECODER_LOSSES = ("lexical", "structure", "binding", "placement", "terminal")


@dataclass(frozen=True)
class DecodedAction:
    payload: bytes
    emitted_ids: tuple[int, ...]
    terminal: bool
    valid: bool
    error: str | None
    trace: tuple[Decision, ...]
    capacity: int
    field_ids: tuple[tuple[str, tuple[int, ...]], ...] = ()
    losses: dict[str, tuple[Tensor, int]] = field(default_factory=dict)


class _Program:
    """Argmax at serving; the same masked logits with labels during learning."""

    def __init__(self, reference: Tensor, teacher: PreparedAction | None):
        self.teacher = teacher
        zero = reference.float().sum() * 0
        self.losses = {name: (zero, 0) for name in DECODER_LOSSES}
        self.trace: list[Decision] = []

    def add(self, key: str, loss: Tensor, count: int) -> None:
        previous, denominator = self.losses[key]
        self.losses[key] = previous + loss, denominator + count

    def choose(self, name, logits, allowed, *, forced=None, key="structure") -> int:
        allowed = tuple(map(int, allowed))
        if not allowed:
            raise ValueError(f"no feasible support for {name}")
        support = torch.tensor(allowed, device=logits.device, dtype=torch.long)
        selected_logits = logits.float().reshape(-1).index_select(0, support)
        if self.teacher is None:
            # Public singleton decisions have no uncertainty. Do not insert a
            # GPU->host synchronization just to rediscover a forced schema,
            # tool mode, fixed length, or required terminal.
            value = allowed[0] if len(allowed) == 1 else allowed[int(selected_logits.argmax())]
        else:
            if forced is None or int(forced) not in allowed:
                raise ValueError(f"teacher decision {name}={forced} is outside public constraints")
            value = int(forced)
            target = torch.tensor([allowed.index(value)], device=logits.device)
            self.add(key, F.cross_entropy(selected_logits[None], target, reduction="sum"), 1)
        self.trace.append(Decision(name, value))
        return value

    def lexical(self, decoder, states, kind, *, forced=None) -> tuple[int, ...]:
        valid = decoder.alphabet(kind)
        if self.teacher is None:
            return tuple(decoder.vocabulary.argmax(states, valid).tolist())
        ids = tuple(forced or ())
        if len(ids) != states.shape[0]:
            raise ValueError("teacher free token count differs from the planned span")
        target = torch.tensor(ids, device=states.device, dtype=torch.long)
        self.add("lexical", decoder.vocabulary.loss(states, target, valid).sum(), len(ids))
        return ids


class DeterministicDecoder(nn.Module):
    def __init__(self, config, codec) -> None:
        super().__init__()
        self.config, self.codec = config, codec
        width = config.facet_dim
        self.header_start = nn.Linear(width, width, bias=False)
        self.header_early = nn.Linear(width, width, bias=False)
        self.header_final = nn.Linear(width, width, bias=False)
        self.header_cell = nn.GRUCell(width, width)
        self.mode_codes = nn.Embedding(2, width)
        self.schema_codes = nn.Embedding(128, width)
        self.layout_codes = nn.Embedding(16, width)
        self.length_codes = nn.Embedding(config.max_capacity + 1, width)
        self.terminal_codes = nn.Embedding(2, width)
        self.identifier_slots = nn.Embedding(32, width)
        self.binding_query = nn.Linear(width, width, bias=False)
        self.public_projection = nn.Linear(width, width, bias=False)
        self.free_binding = nn.Parameter(torch.zeros(width))
        self.realizer = ParallelRealizer(config)
        self.vocabulary = TiedVocabulary(width, config.vocab_size, config.output_rank)
        for codes in (
            self.mode_codes,
            self.schema_codes,
            self.layout_codes,
            self.length_codes,
            self.terminal_codes,
            self.identifier_slots,
        ):
            nn.init.normal_(codes.weight, std=width**-0.5)
        self.allowed_ids = {}
        self._public_tokens: dict[str, tuple[int, ...]] = {}
        for kind in ("bytes", "utf8", "ascii", "identifier"):
            allowed = self.codec.allowed_ids(kind)
            self.allowed_ids[kind] = frozenset(allowed)
            mask = torch.zeros(config.vocab_size, dtype=torch.bool)
            mask[list(allowed)] = True
            self.register_buffer("valid_" + kind, mask, persistent=False)

    def alphabet(self, kind: str) -> Tensor:
        return getattr(self, "valid_" + ("utf8" if kind == "json" else kind))

    @staticmethod
    def logits(header: Tensor, codes: nn.Embedding) -> Tensor:
        return F.linear(header, codes.weight) / math.sqrt(header.shape[-1])

    def condition(self, header, codes, value):
        if not isinstance(value, Tensor):
            value = torch.full((header.shape[0],), value, device=header.device, dtype=torch.long)
        return self.header_cell(codes(value), header)

    def public_embedding(self, ids: Sequence[int], reference: Tensor) -> Tensor:
        if not ids:
            return torch.zeros_like(reference)
        tokens = torch.tensor(ids, device=reference.device, dtype=torch.long)
        return self.public_projection(self.vocabulary.embed(tokens).mean(0))

    def _description(self, value, reference):
        text = json.dumps(asdict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if text not in self._public_tokens:
            self._public_tokens[text] = tuple(self.codec.encode(text.encode()))
        return self.public_embedding(self._public_tokens[text], reference)

    def header(self, context) -> Tensor:
        raw = context if isinstance(context, Tensor) else context.facets
        early = None if isinstance(context, Tensor) else context.early
        final = None if isinstance(context, Tensor) else context.final
        dtype = self.header_start.weight.dtype
        return (
            self.header_start(raw.to(dtype).mean(1))
            + self.header_early((raw if early is None else early).to(dtype).mean(1))
            + self.header_final((raw if final is None else final).to(dtype).mean(1))
        )

    def teacher_raw(self, core, context, plan: PreparedTargets):
        """Vectorized independent examples; only the sparse CE positions are gathered."""
        context = core.prepare_realization(context) if isinstance(context, Tensor) else context
        batch = len(plan.ids)
        header = self.condition(self.header(context), self.mode_codes, 0)
        zero = header.float().sum() * 0
        losses = {name: (zero, 0) for name in DECODER_LOSSES}
        lengths = torch.tensor([len(row) for row in plan.ids], device=header.device)
        losses["structure"] = (
            F.cross_entropy(
                self.logits(header, self.length_codes).float(), lengths, reduction="sum"
            ),
            batch,
        )
        header = self.condition(header, self.length_codes, lengths)
        terminals = torch.tensor(plan.terminals, device=header.device, dtype=torch.long)
        losses["terminal"] = (
            F.cross_entropy(
                self.logits(header, self.terminal_codes).float(), terminals, reduction="sum"
            ),
            batch,
        )
        header = self.condition(header, self.terminal_codes, terminals)
        rows = self.realizer(core, context, plan.capacity, header)
        positions, targets = [], []
        for example, (ids, mask) in enumerate(zip(plan.ids, plan.free_masks, strict=True)):
            for position, (token, scored) in enumerate(zip(ids, mask, strict=True)):
                if scored:
                    positions.append(example * plan.capacity + position)
                    targets.append(token)
        indices = torch.tensor(positions, device=header.device, dtype=torch.long)
        targets = torch.tensor(targets, device=header.device, dtype=torch.long)
        selected = rows.flatten(0, 1).index_select(0, indices)
        losses["lexical"] = (
            self.vocabulary.loss(selected, targets, self.valid_bytes).sum(),
            len(positions),
        )
        return losses

    def forward(self, core, context, spec: ActionSpec, teacher: PreparedAction | None = None):
        context = core.prepare_realization(context) if isinstance(context, Tensor) else context
        if context.facets.shape[0] != 1:
            raise ValueError("action realization takes one example; raw supervision is batched")
        if len(spec.schemas) > self.schema_codes.num_embeddings:
            raise ValueError("too many public schemas for the configured schema head")
        header = self.header(context)
        program = _Program(header, teacher)
        limit = min(spec.max_tokens, self.config.max_capacity)
        schema_fields = {
            index: planned_fields(self.codec, schema.fields)
            for index, schema in enumerate(spec.schemas)
        }
        # The model's field lengths are not the wire length. A tool may
        # require more tokens for its fixed JSON envelope than for all its
        # semantic values combined. Require a schema-valid complete wire
        # witness at this limit before making it a candidate.
        # Field lengths live in the model's workspace, but a complete tool
        # action must also fit its literal wire syntax (keys, separators, quotes).
        # Reserve those fixed tokens before deciding which schemas or value
        # lengths the model may choose.
        schema_limits = {
            index: limit - wire_overhead_tokens(schema, self.codec)
            for index, schema in enumerate(spec.schemas)
        }
        feasible = []
        for index, fields in schema_fields.items():
            if fields is None or sum(value.minimum for value in fields.values()) > schema_limits[index]:
                continue
            witness = wire_witness_tokens(spec.schemas[index], self.codec)
            if witness is not None and witness > limit:
                continue
            feasible.append(index)
        modes = ([0] if spec.allow_text and spec.min_tokens <= limit else []) + (
            [1] if feasible else []
        )
        if not modes:
            raise ValueError("no public output mode fits the configured capacity")

        def choose(name, codes, allowed, forced=None, key="structure"):
            nonlocal header
            value = program.choose(
                name, self.logits(header, codes)[0], allowed, forced=forced, key=key
            )
            header = self.condition(header, codes, value)
            return value

        mode = choose("mode", self.mode_codes, modes, 1 if teacher is not None else None)
        schema = None
        if mode:
            descriptions = torch.stack(
                [self._description(schema_, header[0]) for schema_ in spec.schemas]
            )
            schema_keys = self.schema_codes.weight[: len(spec.schemas)] + descriptions
            logits = (schema_keys @ header[0]) / math.sqrt(header.shape[-1])
            selected = program.choose(
                "schema", logits, feasible, forced=teacher.schema_index if teacher else None
            )
            header = self.header_cell(schema_keys[selected][None], header)
            schema = spec.schemas[selected]
            if len(schema.orders) > self.layout_codes.num_embeddings:
                raise ValueError("too many declared layouts")
            fields = schema_fields[selected]
            field_limit = schema_limits[selected]
        else:
            field_limit = limit
            fields = planned_fields(
                self.codec,
                (
                    FieldSpec(
                        "text",
                        kind=spec.text_kind,
                        min_tokens=spec.min_tokens,
                        max_tokens=limit,
                        allow_copy=False,
                    ),
                ),
            )
        if fields is None:
            raise ValueError("infeasible field constraints")
        sources = tuple(source_ids(self.codec, source) for source in spec.copy_sources)
        source_keys = torch.stack(
            [self.free_binding] + [self.public_embedding(ids, header[0]) for ids in sources]
        )
        identifiers: dict[str, tuple[int, ...]] = {}

        def consumers(definition):
            return [
                value
                for value in fields.values()
                if value.spec.identifier == definition or value.spec.bound_to == definition
            ]

        def identifier_kind(definition):
            kinds = {value.spec.kind for value in consumers(definition)}
            return (
                "identifier" if "identifier" in kinds else "ascii" if "ascii" in kinds else "bytes"
            )

        def minimum_total():
            return sum(value.minimum for value in fields.values())

        for slot, value in enumerate(fields.values()):
            field_ = value.spec
            if value.fixed_ids is not None:
                continue
            if field_.bound_to:
                value.fixed_ids = identifiers[field_.bound_to]
                continue
            if schema is not None:
                header = self.header_cell(self._description(field_, header[0])[None], header)
            if field_.allow_copy and sources:
                allowed = [0]
                for index, ids in enumerate(sources, 1):
                    kind = (
                        identifier_kind(field_.identifier)
                        if field_.identifier
                        else ("utf8" if field_.kind == "json" else field_.kind)
                    )
                    if (
                        not ids
                        or len(ids) > field_.max_tokens
                        or not all(token in self.allowed_ids[kind] for token in ids)
                    ):
                        continue
                    if field_.identifier:
                        shared = consumers(field_.identifier)
                        lower = max(1, *(other.minimum for other in shared))
                        upper = min(other.spec.max_tokens for other in shared)
                        fits = (
                            lower <= len(ids) <= upper
                            and minimum_total()
                            - sum(other.minimum for other in shared)
                            + len(shared) * len(ids)
                            <= field_limit
                        )
                    else:
                        fits = (
                            minimum_total() - value.minimum + max(value.minimum, len(ids)) <= field_limit
                        )
                    if fits:
                        allowed.append(index)
                logits = source_keys @ self.binding_query(header)[0] / math.sqrt(header.shape[-1])
                selected = program.choose(
                    "binding/" + field_.name,
                    logits,
                    allowed,
                    key="binding",
                    forced=teacher.bindings.get(field_.name, 0) if teacher else None,
                )
                header = self.header_cell(source_keys[selected][None], header)
                value.copy_ids = sources[selected - 1] if selected else ()
                if field_.identifier and selected:
                    value.fixed_ids = value.copy_ids
            if field_.identifier:
                if value.fixed_ids is None:
                    shared = consumers(field_.identifier)
                    lower = max(1, *(other.minimum for other in shared))
                    available = field_limit - minimum_total() + sum(other.minimum for other in shared)
                    upper = min(
                        min(other.spec.max_tokens for other in shared), available // len(shared)
                    )
                    length = choose(
                        "identifier_length/" + field_.name,
                        self.length_codes,
                        range(lower, upper + 1),
                        len(teacher.field_ids[field_.name]) if teacher else None,
                    )
                    id_rows = (
                        header[:, None]
                        + self.identifier_slots.weight[slot % self.identifier_slots.num_embeddings][
                            None, None
                        ]
                    )
                    id_rows = (
                        id_rows
                        + position_features(length, header.shape[-1], header.device).to(
                            header.dtype
                        )[None]
                    )
                    value.fixed_ids = program.lexical(
                        self,
                        id_rows[0],
                        identifier_kind(field_.identifier),
                        forced=teacher.field_ids[field_.name] if teacher else None,
                    )
                    tensor = torch.tensor(value.fixed_ids, device=header.device, dtype=torch.long)
                    header = self.header_cell(self.vocabulary.embed(tensor).mean(0)[None], header)
                identifiers[field_.identifier] = value.fixed_ids
                for other in fields.values():
                    if other.spec.bound_to == field_.identifier:
                        other.fixed_ids = value.fixed_ids
        layout = (
            choose(
                "layout",
                self.layout_codes,
                range(len(schema.orders)),
                teacher.layout if teacher else None,
            )
            if schema
            else 0
        )
        assigned = 0
        values = list(fields.values())
        for index, value in enumerate(values):
            if value.fixed_ids is not None:
                value.length = len(value.fixed_ids)
            else:
                remaining = sum(other.minimum for other in values[index + 1 :])
                upper = min(value.spec.max_tokens, field_limit - assigned - remaining)
                value.length = choose(
                    "length/" + value.spec.name,
                    self.length_codes,
                    range(value.minimum, upper + 1),
                    len(teacher.field_ids[value.spec.name]) if teacher else None,
                )
                if value.copy_ids:
                    value.placement = choose(
                        "placement/" + value.spec.name,
                        self.length_codes,
                        range(value.length - len(value.copy_ids) + 1),
                        teacher.placements[value.spec.name] if teacher else None,
                        key="placement",
                    )
            value.offset = assigned
            assigned += value.length
        terminal = bool(
            choose(
                "terminal",
                self.terminal_codes,
                [int(spec.require_terminal)] if spec.require_terminal is not None else [0, 1],
                int(teacher.terminal) if teacher else None,
                key="terminal",
            )
        )
        capacity = workspace_capacity(
            assigned, teacher.capacity if teacher else None, self.config.max_capacity
        )
        rows = self.realizer(core, context, capacity, header)
        outputs = {}
        for name, value in fields.items():
            if value.fixed_ids is not None:
                outputs[name] = value.fixed_ids
                continue
            positions = torch.tensor(value.free_positions, device=rows.device, dtype=torch.long)
            forced = None
            if teacher is not None:
                labels = teacher.field_ids[name]
                forced = tuple(
                    token
                    for index, token in enumerate(labels)
                    if not value.placement <= index < value.placement + len(value.copy_ids)
                )
            free_ids = program.lexical(
                self, rows[0].index_select(0, positions), value.spec.kind, forced=forced
            )
            outputs[name] = value.assemble(free_ids)
        payload, emitted, error = b"", (), None
        try:
            raw_values = {name: self.codec.decode(ids) for name, ids in outputs.items()}
            if schema is None:
                payload, emitted = raw_values["text"], outputs["text"]
                if spec.text_kind == "utf8":
                    payload.decode("utf-8", errors="strict")
            else:
                payload = schema.serialize(raw_values, layout)
                # Serialization itself is deterministic. Its complete tokenization
                # is recorded verbatim; only free value IDs have a training CE.
                emitted = tuple(self.codec.encode(payload))
            if not spec.min_tokens <= len(emitted) <= limit:
                raise ValueError("serialized output including fixed syntax exceeds token limits")
        except (ValueError, UnicodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            # jsonschema's ValidationError is optional until structured data is used.
            from jsonschema import ValidationError

            if not isinstance(exc, ValidationError):
                raise
            error = f"ValidationError: {exc.message}"
        if teacher is not None and (error is not None or payload != teacher.payload):
            raise ValueError(f"teacher action did not reproduce its exact payload: {error}")
        return DecodedAction(
            payload,
            emitted,
            terminal,
            error is None,
            error,
            tuple(program.trace),
            capacity,
            tuple(outputs.items()),
            program.losses,
        )
