"""Host-side target plans and exact loss denominators, shared by both decoders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from megatoken.contracts import LOSS_NAMES, ROLE_TO_ID

from .codec import ByteCodec
from .structure import (
    ActionSpec,
    CopySource,
    FieldSpec,
    strict_json_loads,
)

if TYPE_CHECKING:
    from megatoken.model.multitrack import GroupActionTape


@dataclass(frozen=True)
class PreparedTargets:
    payloads: tuple[bytes, ...]
    ids: tuple[tuple[int, ...], ...]
    free_masks: tuple[tuple[bool, ...], ...]
    terminals: tuple[bool, ...]
    # One output workspace size per example: the power of two above its own length,
    # exactly the rule serving applies to a free realization of that length.
    capacities: tuple[int, ...]
    denominators: Mapping[str, int]
    scored_positions: int
    roles: tuple[int, ...]
    representation_mode: str = "bootstrap"
    native_tapes: tuple[GroupActionTape | None, ...] = ()

    @property
    def mode(self) -> str:
        return self.representation_mode


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
    terminal: bool
    denominators: Mapping[str, int]
    scored_positions: int
    role_id: int = 2
    representation_mode: str = "bootstrap"
    native_tape: GroupActionTape | None = None

    @property
    def mode(self) -> str:
        return self.representation_mode

    @property
    def capacities(self) -> tuple[int, ...]:
        return (self.capacity,)


@dataclass
class FieldPlan:
    spec: FieldSpec
    fixed_ids: tuple[int, ...] | None = None
    copy_ids: tuple[int, ...] = ()
    reserved_minimum: int = 0
    length: int = 0
    placement: int = 0
    offset: int = 0

    @property
    def minimum(self) -> int:
        if self.fixed_ids is not None:
            return len(self.fixed_ids)
        return max(self.spec.min_tokens, self.reserved_minimum, len(self.copy_ids))

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


def planned_fields(codec: ByteCodec, specs: Sequence[FieldSpec]) -> dict[str, FieldPlan] | None:
    fields = {
        spec.name: FieldPlan(spec, reserved_minimum=int(spec.kind == "json")) for spec in specs
    }
    for value in fields.values():
        if value.spec.fixed_json is not None:
            value.fixed_ids = tuple(codec.encode(value.spec.fixed_json.encode()))
        elif value.minimum > value.spec.max_tokens:
            return None
    for definition in (spec.identifier for spec in specs if spec.identifier):
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


def source_ids(codec: ByteCodec, source: CopySource) -> tuple[int, ...]:
    ids = source.emitted_ids or tuple(codec.encode(source.payload))
    if codec.decode(ids) != source.payload:
        raise ValueError(f"copy source {source.name} IDs do not decode to its bytes")
    return ids


def workspace_capacity(length: int, requested: int | None, hard_limit: int) -> int:
    needed = max(1, length)
    capacity = min(1 << (needed - 1).bit_length(), hard_limit) if requested is None else requested
    if not needed <= capacity <= hard_limit:
        raise ValueError("target exceeds output capacity; split it at a causal chunk boundary")
    return capacity


def _denominators(
    config, batch, lexical, structure, binding=0, placement=0, *, mode="bootstrap", native=0
):
    if mode not in {"bootstrap", "recurrent"}:
        raise ValueError("representation_mode must be bootstrap or recurrent")
    bootstrap = mode == "bootstrap"
    passes = 2
    result = dict.fromkeys(LOSS_NAMES, 0)
    result.update(
        lexical=lexical,
        reconstruction=lexical,
        structure=passes * structure,
        binding=passes * binding,
        placement=passes * placement,
        terminal=passes * batch,
        facet_prediction=batch * config.n_facets * config.facet_dim if bootstrap else 0,
        encoder_alignment=0 if bootstrap else native * config.n_facets * config.facet_dim,
        router_balance=batch,
        router_z=batch,
    )
    return MappingProxyType(result)


def _native_pairs(codec, config, payloads, ids, roles, tapes):
    """Accept only saved actions that actually materialized the supervised bytes.

    This is a target label, never a prior input: a one-event group tape from a rollout.
    Chronological linkage to the episode's record cursor is additionally checked by
    NativeHistory upstream.
    """
    from megatoken.model.multitrack import GroupActionMember, GroupActionTape

    tapes = (None,) * len(payloads) if tapes is None else tuple(tapes)
    if len(tapes) != len(payloads):
        raise ValueError("native_tapes must identify every target batch row")
    for payload, tokens, role, tape in zip(payloads, ids, roles, tapes, strict=True):
        if tape is None:
            continue
        if not isinstance(tape, (GroupActionTape, GroupActionMember)) or len(tape.tracks) != 1:
            raise ValueError("native alignment requires a current single-event group tape")
        if tape.codec_fingerprint != codec.fingerprint or tape.facet_shape != (
            1,
            config.n_facets,
            config.facet_dim,
        ):
            raise ValueError("native alignment codec or facet geometry differs")
        if (
            not tape.valid[0]
            or tape.payloads[0] != payload
            or tape.emitted_ids[0] != tuple(tokens)
            or tape.roles[0] != role
        ):
            raise ValueError(
                "native alignment must match the actual valid output bytes, IDs and role"
            )
    return tapes


def prepare_targets(
    codec,
    config,
    target_ids=None,
    *,
    target_bytes=None,
    free_mask=None,
    terminal=True,
    capacity=None,
    role_ids=None,
    representation_mode="bootstrap",
    native_tapes=None,
) -> PreparedTargets:
    if (target_ids is None) == (target_bytes is None):
        raise ValueError("supply exactly one of target_ids or target_bytes")
    if target_bytes is not None:
        payloads = (target_bytes,) if isinstance(target_bytes, bytes) else tuple(target_bytes)
        payloads = tuple(bytes(payload) for payload in payloads)
        ids = tuple(tuple(codec.encode(payload)) for payload in payloads)
    else:
        raw_ids = target_ids.tolist() if hasattr(target_ids, "tolist") else target_ids
        if not raw_ids or isinstance(raw_ids[0], int):
            raw_ids = [raw_ids]
        ids = tuple(tuple(int(value) for value in row) for row in raw_ids)
        payloads = tuple(codec.decode(row) for row in ids)
    if not payloads:
        raise ValueError("empty target batch")
    if free_mask is None:
        masks = tuple((True,) * len(row) for row in ids)
    else:
        masks = free_mask.tolist() if hasattr(free_mask, "tolist") else free_mask
        if not masks or isinstance(masks[0], (bool, int)):
            masks = [masks]
        masks = tuple(tuple(bool(value) for value in row) for row in masks)
        if len(masks) != len(ids) or any(
            len(mask) != len(row) for mask, row in zip(masks, ids, strict=True)
        ):
            raise ValueError("free masks must match actual target ID sequences")
    batch = len(ids)
    terminals = (terminal,) * batch if isinstance(terminal, bool) else tuple(map(bool, terminal))
    roles = (2,) * batch if role_ids is None else tuple(map(int, role_ids))
    if (
        len(terminals) != batch
        or len(roles) != batch
        or any(not 0 <= role < len(ROLE_TO_ID) for role in roles)
    ):
        raise ValueError("one supported role and terminal label are required per target")
    capacities = tuple(workspace_capacity(len(row), capacity, config.max_capacity) for row in ids)
    lexical = sum(map(sum, masks))
    pairs = _native_pairs(codec, config, payloads, ids, roles, native_tapes)
    denominators = _denominators(
        config,
        batch,
        lexical,
        batch,
        mode=representation_mode,
        native=sum(tape is not None for tape in pairs),
    )
    return PreparedTargets(
        payloads,
        ids,
        masks,
        terminals,
        capacities,
        denominators,
        denominators["lexical"] + denominators["reconstruction"],
        roles,
        representation_mode,
        pairs,
    )


def prepare_action(
    codec,
    config,
    payload: bytes,
    *,
    spec: ActionSpec,
    schema_index=0,
    bindings: Mapping[str, int] | None = None,
    capacity=None,
    layout=0,
    terminal=True,
    role_id=2,
    representation_mode="bootstrap",
    native_tape=None,
) -> PreparedAction:
    """Use only a public causal schema; canonicalization is visible in .payload.

    Teacher labels train exactly the same deterministic decisions used at
    inference. Fixed syntax, copies, and repeated identifiers incur no lexical
    CE. Both modes have a prior and a reconstruction pass; the recurrent
    reconstruction branch updates only E through frozen contextual processing.
    """
    if not 0 <= schema_index < len(spec.schemas):
        raise ValueError("selected schema is absent from the public action spec")
    schema = spec.schemas[schema_index]
    if not 0 <= layout < len(schema.orders):
        raise ValueError("selected layout is not declared by the schema")
    parsed = strict_json_loads(payload.decode("utf-8", errors="strict"))
    if not isinstance(parsed, dict) or set(parsed) != {field.name for field in schema.fields}:
        raise ValueError("target JSON object does not match schema fields")
    values, ids, identifiers, placements = {}, {}, {}, {}
    declared_bindings = dict(bindings or {})
    if set(declared_bindings) - {field.name for field in schema.fields}:
        raise ValueError("copy labels contain unknown fields")
    lexical, structure, binding_count, placement_count = 0, 3, 0, 0
    sources = tuple(source_ids(codec, source) for source in spec.copy_sources)
    alphabets = {
        kind: frozenset(codec.allowed_ids(kind))
        for kind in ("bytes", "utf8", "ascii", "identifier")
    }
    for field in schema.fields:
        value = parsed[field.name]
        if field.fixed_json is not None:
            if strict_json_loads(field.fixed_json) != value:
                raise ValueError(f"target fixed field {field.name} differs from declared syntax")
            raw = field.fixed_json.encode()
        elif field.kind == "json":
            raw = json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        else:
            if not isinstance(value, str):
                raise ValueError(f"string field {field.name} received {type(value).__name__}")
            raw = value.encode("utf-8")
        values[field.name] = raw
        ids[field.name] = tuple(codec.encode(raw))
        if field.fixed_json is not None:
            if declared_bindings.get(field.name, 0):
                raise ValueError("fixed syntax cannot have a copy label")
            continue
        kind = "utf8" if field.kind == "json" else field.kind
        if not all(token in alphabets[kind] for token in ids[field.name]):
            raise ValueError(f"target field {field.name} violates its lexical alphabet")
        if field.bound_to:
            if codec.decode(identifiers[field.bound_to]) != raw:
                raise ValueError(f"bound field {field.name} differs from its identifier")
            ids[field.name] = identifiers[field.bound_to]
            if declared_bindings.get(field.name, 0):
                raise ValueError("bound references cannot have an independent copy label")
        else:
            selected = declared_bindings.get(field.name, 0)
            if selected and (not field.allow_copy or not 1 <= selected <= len(sources)):
                raise ValueError(f"invalid copy binding for {field.name}")
            if field.allow_copy and sources:
                binding_count += 1
            copied = sources[selected - 1] if selected else ()
            if selected and not copied:
                raise ValueError("an empty copy source cannot bind a field")
            if copied:
                copied_bytes = codec.decode(copied)
                position = raw.find(copied_bytes)
                if position < 0:
                    raise ValueError(f"declared copy is absent from target field {field.name}")
                before = tuple(codec.encode(raw[:position]))
                after = tuple(codec.encode(raw[position + len(copied_bytes) :]))
                if field.identifier and (before or after):
                    raise ValueError("identifier copies must bind the entire value")
                ids[field.name] = before + copied + after
                placements[field.name] = len(before)
            if field.identifier:
                if not ids[field.name]:
                    raise ValueError("identifiers must contain at least one token")
                if not copied:
                    structure += 1
                    lexical += len(ids[field.name])
                identifiers[field.identifier] = ids[field.name]
            else:
                structure += 1
                lexical += len(ids[field.name]) - len(copied)
                placement_count += bool(copied)
        if not field.min_tokens <= len(ids[field.name]) <= field.max_tokens:
            raise ValueError(f"target field {field.name} exceeds its token limits")
    payload = schema.serialize(values, layout)
    output_length = len(codec.encode(payload))
    value_length = sum(map(len, ids.values()))
    if max(output_length, value_length) > min(spec.max_tokens, config.max_capacity):
        raise ValueError("typed action including fixed syntax exceeds the public output limit")
    if spec.require_terminal is not None:
        terminal = spec.require_terminal
    if not 0 <= role_id < len(ROLE_TO_ID):
        raise ValueError("target role ID is unsupported")
    pairs = _native_pairs(
        codec, config, (payload,), (tuple(codec.encode(payload)),), (role_id,), (native_tape,)
    )
    denominators = _denominators(
        config,
        1,
        lexical,
        structure,
        binding_count,
        placement_count,
        mode=representation_mode,
        native=int(pairs[0] is not None),
    )
    return PreparedAction(
        payload,
        spec,
        schema_index,
        layout,
        # The realizer emits value tokens only; the fixed wire syntax is serialized
        # afterwards and needs no workspace (serving sizes it from the value length).
        workspace_capacity(value_length, capacity, config.max_capacity),
        MappingProxyType(ids),
        MappingProxyType(declared_bindings),
        MappingProxyType(placements),
        bool(terminal),
        denominators,
        denominators["lexical"] + denominators["reconstruction"],
        role_id,
        representation_mode,
        pairs[0],
    )
