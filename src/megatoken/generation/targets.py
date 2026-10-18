"""Host-side target plans and exact loss denominators, shared by both decoders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from megatoken.contracts import LOSS_NAMES

from .codec import ByteCodec
from .structure import ActionSpec, CopySource, FieldSpec, strict_json_loads


@dataclass(frozen=True)
class PreparedTargets:
    payloads: tuple[bytes, ...]
    ids: tuple[tuple[int, ...], ...]
    free_masks: tuple[tuple[bool, ...], ...]
    terminals: tuple[bool, ...]
    capacity: int
    denominators: Mapping[str, int]
    scored_positions: int
    roles: tuple[int, ...]
    representation_mode: str = "bootstrap"

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

    @property
    def mode(self) -> str:
        return self.representation_mode


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


def _denominators(config, batch, lexical, structure, binding=0, placement=0, *, mode="bootstrap"):
    if mode not in {"bootstrap", "recurrent"}:
        raise ValueError("representation_mode must be bootstrap or recurrent")
    bootstrap = mode == "bootstrap"
    passes = 1 + int(bootstrap)
    result = dict.fromkeys(LOSS_NAMES, 0)
    result.update(
        lexical=lexical,
        reconstruction=lexical if bootstrap else 0,
        structure=passes * structure,
        binding=passes * binding,
        placement=passes * placement,
        terminal=passes * batch,
        facet_prediction=batch * config.n_facets * config.facet_dim if bootstrap else 0,
        encoder_alignment=0 if bootstrap else batch * config.n_facets * config.facet_dim,
        router_balance=batch,
        router_z=batch,
    )
    return MappingProxyType(result)


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
    if len(terminals) != batch or len(roles) != batch or any(not 0 <= role <= 5 for role in roles):
        raise ValueError("one supported role and terminal label are required per target")
    capacity = workspace_capacity(max(map(len, ids)), capacity, config.max_capacity)
    lexical = sum(map(sum, masks))
    denominators = _denominators(config, batch, lexical, batch, mode=representation_mode)
    return PreparedTargets(
        payloads,
        ids,
        masks,
        terminals,
        capacity,
        denominators,
        denominators["lexical"] + denominators["reconstruction"],
        roles,
        representation_mode,
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
) -> PreparedAction:
    """Use only a public causal schema; canonicalization is visible in .payload.

    Teacher labels train exactly the same deterministic decisions used at
    inference. Fixed syntax, copies, and repeated identifiers incur no lexical
    CE. Bootstrap has a prior and a reconstruction pass; recurrent training
    evaluates only the live predictive decoder.
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
    if not 0 <= role_id <= 5:
        raise ValueError("target role ID is unsupported")
    denominators = _denominators(
        config, 1, lexical, structure, binding_count, placement_count, mode=representation_mode
    )
    return PreparedAction(
        payload,
        spec,
        schema_index,
        layout,
        workspace_capacity(max(output_length, value_length), capacity, config.max_capacity),
        MappingProxyType(ids),
        MappingProxyType(declared_bindings),
        MappingProxyType(placements),
        bool(terminal),
        denominators,
        denominators["lexical"] + denominators["reconstruction"],
        role_id,
        representation_mode,
    )
