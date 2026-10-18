"""Immutable action data, causal schema constraints, and deterministic serializers."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from functools import cached_property, lru_cache
from typing import Any


def strict_json_loads(value: str | bytes) -> Any:
    def invalid_constant(name):
        raise ValueError(f"nonstandard JSON constant {name}")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key}")
            value[key] = item
        return value

    return json.loads(value, parse_constant=invalid_constant, object_pairs_hook=unique_object)


@lru_cache(maxsize=256)
def _validator(schema_text: str):
    import jsonschema

    schema = strict_json_loads(schema_text)

    def check_refs(value):
        if isinstance(value, dict):
            reference = value.get("$ref")
            if reference is not None and not reference.startswith("#"):
                raise ValueError(
                    "tool schemas must bundle references locally; network schema resolution is disabled"
                )
            for child in value.values():
                check_refs(child)
        elif isinstance(value, list):
            for child in value:
                check_refs(child)

    check_refs(schema)
    validator_type = jsonschema.validators.validator_for(schema)

    def validate_json_content(validator, content_schema, instance, subschema):
        # JSON Schema normally treats contentSchema as an annotation. FMT's
        # explicit tool-envelope contract validates decoded JSON strings here so
        # OpenAI-style arguments cannot evade the inner tool parameter schema.
        if not isinstance(instance, str) or subschema.get("contentMediaType") != "application/json":
            return
        try:
            decoded = strict_json_loads(instance)
        except (ValueError, UnicodeError) as exc:
            yield jsonschema.ValidationError(
                f"encoded tool arguments are not JSON: {type(exc).__name__}"
            )
            return
        yield from validator.descend(decoded, content_schema)

    validator_type = jsonschema.validators.extend(
        validator_type, {"contentSchema": validate_json_content}
    )
    validator_type.check_schema(schema)
    return validator_type(schema)


@dataclass(frozen=True)
class CopySource:
    name: str
    payload: bytes
    emitted_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        object.__setattr__(self, "emitted_ids", tuple(self.emitted_ids))


@dataclass(frozen=True)
class Observation:
    payload: bytes
    role_id: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", bytes(self.payload))
        if not 0 <= self.role_id <= 5:
            raise ValueError("observation role ID must identify a supported record role")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str = "utf8"
    min_tokens: int = 0
    max_tokens: int = 256
    identifier: str | None = None
    bound_to: str | None = None
    allow_copy: bool = True
    json_schema: str | None = None
    fixed_json: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"bytes", "utf8", "ascii", "identifier", "json"}:
            raise ValueError(f"unsupported field alphabet {self.kind}")
        if not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError("field token limits must be ordered and nonnegative")
        if self.identifier and self.bound_to:
            raise ValueError("a field cannot both define and reference an identifier")
        if self.json_schema is not None:
            _validator(self.json_schema)
        if self.fixed_json is not None:
            fixed = strict_json_loads(self.fixed_json)
            if self.json_schema is not None:
                _validator(self.json_schema).validate(fixed)
            if self.identifier or self.bound_to:
                raise ValueError("fixed syntax cannot define a sampled identifier")


@dataclass(frozen=True)
class SchemaSpec:
    """A JSON object with fixed keys and typed sampled value spans.

    More complex tool schemas should be compiled into fixed syntax plus typed
    fields upstream. A single complete validated object is the execution unit.
    ``layouts`` are explicitly permitted key orders, never arbitrary SQL/code.
    """

    name: str
    fields: tuple[FieldSpec, ...]
    layouts: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "layouts", tuple(tuple(order) for order in self.layouts))
        names = [field.name for field in self.fields]
        if not names or len(set(names)) != len(names):
            raise ValueError("schema needs unique field names")
        identifiers: set[str] = set()
        for field in self.fields:
            if field.bound_to and field.bound_to not in identifiers:
                raise ValueError("bound fields must refer to an earlier identifier definition")
            if field.identifier:
                if field.identifier in identifiers:
                    raise ValueError("identifier is defined twice")
                identifiers.add(field.identifier)
        for order in self.layouts:
            if sorted(order) != sorted(names):
                raise ValueError("every layout must contain every field once")

    @property
    def orders(self) -> tuple[tuple[str, ...], ...]:
        return self.layouts or (tuple(field.name for field in self.fields),)

    def serialize(self, values: dict[str, bytes], layout: int = 0) -> bytes:
        order = self.orders[layout]
        if set(values) != {field.name for field in self.fields}:
            raise ValueError("serialized field set does not match schema")
        # JSON escaping is deterministic syntax, with no extra policy factors.
        fields = {field.name: field for field in self.fields}
        fragments = []
        for name in order:
            field = fields[name]
            if field.fixed_json is not None:
                encoded = field.fixed_json.encode("utf-8")
            elif field.kind == "json":
                encoded = values[name]
                value = strict_json_loads(encoded.decode("utf-8", errors="strict"))
                if field.json_schema is not None:
                    _validator(field.json_schema).validate(value)
            else:
                decoded = values[name].decode("utf-8", errors="strict")
                encoded = json.dumps(decoded, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            key = json.dumps(name, ensure_ascii=False).encode("utf-8")
            fragments.append(key + b":" + encoded)
        return b"{" + b",".join(fragments) + b"}"


@dataclass(frozen=True)
class ActionSpec:
    """Public constraints selected from currently available task information.

    Callers are responsible for supplying schemas/copy sources from actual causal
    observations. Their fingerprint is recorded to reject mismatched replay.
    """

    allow_text: bool = True
    text_kind: str = "bytes"
    schemas: tuple[SchemaSpec, ...] = ()
    copy_sources: tuple[CopySource, ...] = ()
    min_tokens: int = 0
    max_tokens: int = 4096
    max_repairs: int = 1
    repair_probability: float = 0.0
    require_terminal: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "schemas", tuple(self.schemas))
        object.__setattr__(self, "copy_sources", tuple(self.copy_sources))
        if not self.allow_text and not self.schemas:
            raise ValueError("action spec permits no output mode")
        if self.text_kind not in {"bytes", "utf8", "ascii", "identifier"}:
            raise ValueError("invalid text alphabet")
        if not 0 <= self.min_tokens <= self.max_tokens:
            raise ValueError("invalid action token limits")
        if not 0 <= self.max_repairs <= 8:
            raise ValueError("repair count must be bounded between zero and eight")
        if not 0 <= self.repair_probability <= 1:
            raise ValueError("invalid repair probability")
        if len({source.name for source in self.copy_sources}) != len(self.copy_sources):
            raise ValueError("copy source names must be unique")

    @cached_property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json_bytes(asdict(self))).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


@dataclass(frozen=True)
class Choice:
    """One sampled categorical event, in causal execution order."""

    name: str
    value: int


@dataclass(frozen=True)
class LexicalDraw:
    """A parallel draw; IDs include noncanonical segmentations verbatim."""

    name: str
    ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ids", tuple(self.ids))


@dataclass(frozen=True)
class DraftAttempt:
    draws: tuple[LexicalDraw, ...]
    valid: bool
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "draws", tuple(self.draws))


@dataclass(frozen=True)
class ActionTape:
    """Complete augmented policy sample, including rejected draft/repair attempts.

    No cached hidden state or behavior-policy expert routes are authoritative.
    Candidate scoring reconstructs state and chooses its own deterministic routes.
    """

    version: str
    codec_fingerprint: str
    spec_fingerprint: str
    effort: int
    root: tuple[float, ...]
    regional: tuple[float, ...]
    choices: tuple[Choice, ...]
    identifiers: tuple[LexicalDraw, ...]
    attempts: tuple[DraftAttempt, ...]
    payload: bytes
    emitted_ids: tuple[int, ...]
    committed: bool
    terminal: bool
    behavior_log_prob: float
    scored_positions: int

    def __post_init__(self) -> None:
        for name in ("root", "regional", "choices", "identifiers", "attempts", "emitted_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "payload", bytes(self.payload))

    def to_json(self) -> str:
        return _json_bytes(asdict(self)).decode()

    @classmethod
    def from_json(cls, text: str) -> ActionTape:
        value = json.loads(text)
        value["root"] = tuple(value["root"])
        value["regional"] = tuple(value["regional"])
        value["choices"] = tuple(Choice(**choice) for choice in value["choices"])
        value["identifiers"] = tuple(
            LexicalDraw(draw["name"], tuple(draw["ids"])) for draw in value["identifiers"]
        )
        value["attempts"] = tuple(
            DraftAttempt(
                tuple(LexicalDraw(draw["name"], tuple(draw["ids"])) for draw in attempt["draws"]),
                attempt["valid"],
                attempt.get("error"),
            )
            for attempt in value["attempts"]
        )
        value["payload"] = base64.b64decode(value["payload"]["$bytes"], validate=True)
        value["emitted_ids"] = tuple(value["emitted_ids"])
        return cls(**value)


class TapeMismatch(ValueError):
    """A tape is incomplete, tampered with, or infeasible under public constraints."""
