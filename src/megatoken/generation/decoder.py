"""A single deterministic action program for serving and teacher-forced learning."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from megatoken.contracts import DECODER_LOSSES
from megatoken.model.layers import position_features
from megatoken.model.state import RealizationContext

from .distributions import TiedVocabulary
from .feasibility import wire_overhead_tokens, wire_witness_tokens
from .realizer import ParallelRealizer
from .structure import ActionSpec, Decision, FieldSpec, SchemaSpec
from .targets import PreparedAction, PreparedTargets, planned_fields, source_ids, workspace_capacity


@lru_cache(maxsize=4096)
def _description_tokens(codec, text: str) -> tuple[int, ...]:
    return tuple(codec.encode(text.encode()))


@lru_cache(maxsize=1024)
def _wire_costs(schema: SchemaSpec, codec) -> tuple[int, int | None]:
    """(fixed wire overhead, smallest valid wire witness or None), per public schema."""
    return wire_overhead_tokens(schema, codec), wire_witness_tokens(schema, codec)


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

    def lexical_batch(self, decoder, states, requests):
        """Sample independent free spans by alphabet with one host readback.

        Header choices, identifiers, and bindings retain their serial semantic
        dependencies. Once all output states exist, ordinary free fields do
        not: gather those positions by alphabet, execute one vocabulary call
        per alphabet, then materialize the chosen IDs with one .tolist().
        """
        if self.teacher is not None:
            raise RuntimeError("Batch sampling is only used for unforced realization")
        offsets = {}
        kinds = {}
        count = 0
        for name, positions, kind in requests:
            offsets[name] = (count, len(positions))
            if positions:
                # Both JSON and UTF-8 use the same lexical token alphabet.
                alphabet = "utf8" if kind == "json" else kind
                group = kinds.setdefault(alphabet, ([], []))
                group[0].extend(positions)
                group[1].extend(range(count, count + len(positions)))
            count += len(positions)
        if not count:
            return {name: () for name in offsets}

        chosen = torch.empty(count, dtype=torch.long, device=states.device)
        for kind, (positions, destinations) in kinds.items():
            indices = torch.tensor(positions, dtype=torch.long, device=states.device)
            target = torch.tensor(destinations, dtype=torch.long, device=states.device)
            sampled = decoder.vocabulary.argmax(
                states.index_select(0, indices), decoder.alphabet(kind)
            )
            chosen.index_copy_(0, target, sampled)
        ids = chosen.tolist()  # Exactly one GPU -> host readback for all free fields.
        result = {}
        for name, (start, length) in offsets.items():
            result[name] = tuple(ids[start : start + length])
        return result


class DeterministicDecoder(nn.Module):
    def __init__(self, config, codec) -> None:
        super().__init__()
        self.config, self.codec = config, codec
        width = config.facet_dim
        self.header_start = nn.Linear(width, width, bias=False)
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
        # Header states are GRU-bounded and codes start near unit norm, so logits
        # already begin small. An extra 1/sqrt(width) temperature capped them near
        # one and left structure/terminal heads stuck at chance in real training.
        return F.linear(header, codes.weight)

    def condition(self, header, codes, value):
        if not isinstance(value, Tensor):
            value = torch.full((header.shape[0],), value, device=header.device, dtype=torch.long)
        return self.header_cell(codes(value), header)

    def public_embedding(self, ids: Sequence[int], reference: Tensor) -> Tensor:
        if not ids:
            return torch.zeros_like(reference)
        tokens = torch.tensor(ids, device=reference.device, dtype=torch.long)
        # Public descriptions are queried one at a time from an unbatched
        # header vector. Keep a singleton leading axis while traversing routed
        # projections so a live [1, adapters] scope remains well formed.
        return self.public_projection(self.vocabulary.embed(tokens).mean(0, keepdim=True))[0]

    def _description(self, value, reference):
        text = json.dumps(asdict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self.public_embedding(_description_tokens(self.codec, text), reference)

    def header(self, context) -> Tensor:
        raw = context if isinstance(context, Tensor) else context.facets
        dtype = self.header_start.weight.dtype
        return self.header_start(raw.to(dtype).mean(1))

    def teacher_raw(self, core, context, plan: PreparedTargets):
        context = core.prepare_realization(context) if isinstance(context, Tensor) else context
        with core.expert_pools.routing_epoch_if_absent(context.facets.mean(1), role="R", stage=0):
            return self._teacher_raw_impl(core, context, plan)

    def _public_support(self, spec):
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
        costs = [_wire_costs(schema, self.codec) for schema in spec.schemas]
        schema_limits = {index: limit - overhead for index, (overhead, _) in enumerate(costs)}
        feasible = []
        for index, fields in schema_fields.items():
            if fields is None or (
                sum(value.minimum for value in fields.values()) > schema_limits[index]
            ):
                continue
            # No constructible minimal valid example means no proof that the schema fits.
            witness = costs[index][1]
            if witness is None or witness > limit:
                continue
            feasible.append(index)
        modes = ([0] if spec.allow_text and spec.min_tokens <= limit else []) + (
            [1] if feasible else []
        )
        if not modes:
            raise ValueError("no public output mode fits the configured capacity")

        return schema_fields, schema_limits, feasible, modes

    def _teacher_raw_impl(self, core, context, plan: PreparedTargets):
        """Vectorized independent examples; only the sparse CE positions are gathered."""
        batch = len(plan.ids)
        public_header = self.header(context)
        header = self.condition(public_header, self.mode_codes, 0)
        zero = header.float().sum() * 0
        losses = {name: (zero, 0) for name in DECODER_LOSSES}
        lengths = torch.tensor([len(row) for row in plan.ids], device=header.device)
        losses["structure"] = (
            F.cross_entropy(
                self.logits(header, self.length_codes).float(), lengths, reduction="sum"
            ),
            batch,
        )
        # Score the format against the same public alternatives as serving.
        # The target never decides which alternatives are legal.
        for index, spec in enumerate(plan.mode_specs):
            if spec is None:
                continue
            modes = self._public_support(spec)[3]
            if 0 not in modes:
                raise ValueError("Text target is outside its public output support")
            support = torch.tensor(modes, device=header.device, dtype=torch.long)
            logits = self.logits(public_header[index : index + 1], self.mode_codes)
            loss = F.cross_entropy(
                logits.float().index_select(1, support),
                torch.tensor([modes.index(0)], device=header.device),
                reduction="sum",
            )
            previous, count = losses["structure"]
            losses["structure"] = previous + loss, count + 1
        header = self.condition(header, self.length_codes, lengths)
        terminals = torch.tensor(plan.terminals, device=header.device, dtype=torch.long)
        losses["terminal"] = (
            F.cross_entropy(
                self.logits(header, self.terminal_codes).float(), terminals, reduction="sum"
            ),
            batch,
        )
        header = self.condition(header, self.terminal_codes, terminals)
        # Each example is realized at its own workspace size, as serving does.
        groups: dict[int, list[int]] = {}
        for example, capacity in enumerate(plan.capacities):
            groups.setdefault(capacity, []).append(example)
        states = []
        targets: list[int] = []
        for capacity, members in groups.items():
            if len(members) == batch:
                group_context, group_header = context, header
                rows = self.realizer(core, group_context, capacity, group_header).flatten(0, 1)
            else:
                member_index = torch.tensor(members, device=header.device, dtype=torch.long)
                group_context = RealizationContext(context.facets.index_select(0, member_index))
                group_header = header.index_select(0, member_index)
                # Each workspace group is a true batch subset. Keep the
                # retrieval-conditioned router features, cached palettes, and
                # adapter rows aligned to those examples while realizing it.
                with core.expert_pools.routing_subset(member_index):
                    rows = self.realizer(core, group_context, capacity, group_header).flatten(0, 1)
            positions = []
            for local, example in enumerate(members):
                for position, (token, scored) in enumerate(
                    zip(plan.ids[example], plan.free_masks[example], strict=True)
                ):
                    if scored:
                        positions.append(local * capacity + position)
                        targets.append(token)
            states.append(
                rows.index_select(0, torch.tensor(positions, device=rows.device, dtype=torch.long))
            )
        selected = torch.cat(states)
        targets = torch.tensor(targets, device=header.device, dtype=torch.long)
        losses["lexical"] = (
            self.vocabulary.loss(selected, targets, self.valid_bytes).sum(),
            selected.shape[0],
        )
        return losses

    def forward(self, core, context, spec: ActionSpec, teacher: PreparedAction | None = None):
        context = core.prepare_realization(context) if isinstance(context, Tensor) else context
        with core.expert_pools.routing_epoch_if_absent(context.facets.mean(1), role="R", stage=0):
            return self._forward_impl(core, context, spec, teacher)

    def _forward_impl(self, core, context, spec: ActionSpec, teacher: PreparedAction | None = None):
        context = core.prepare_realization(context) if isinstance(context, Tensor) else context
        if context.facets.shape[0] != 1:
            raise ValueError("action realization takes one example; raw supervision is batched")
        if len(spec.schemas) > self.schema_codes.num_embeddings:
            raise ValueError("too many public schemas for the configured schema head")
        header = self.header(context)
        program = _Program(header, teacher)
        limit = min(spec.max_tokens, self.config.max_capacity)
        schema_fields, schema_limits, feasible, modes = self._public_support(spec)

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
                            minimum_total() - value.minimum + max(value.minimum, len(ids))
                            <= field_limit
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
                    available = (
                        field_limit - minimum_total() + sum(other.minimum for other in shared)
                    )
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
        # Unlike the earlier header and identifier decisions, free fields are
        # conditionally independent once the final realizer states exist. Batch
        # their lexical selections and transfer all chosen IDs together.
        sampled_fields = (
            program.lexical_batch(
                self,
                rows[0],
                [
                    (name, value.free_positions, value.spec.kind)
                    for name, value in fields.items()
                    if value.fixed_ids is None
                ],
            )
            if teacher is None
            else {}
        )
        outputs = {}
        for name, value in fields.items():
            if value.fixed_ids is not None:
                outputs[name] = value.fixed_ids
                continue
            if teacher is None:
                free_ids = sampled_fields[name]
            else:
                positions = torch.tensor(value.free_positions, device=rows.device, dtype=torch.long)
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
