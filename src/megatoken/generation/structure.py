"""Immutable action data, causal schema constraints, and deterministic serializers."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import cache, cached_property, lru_cache
from typing import Any

from megatoken.contracts import ROLE_TO_ID
from megatoken.io import strict_json_loads


def _parse_bytes(value: Any, field: str) -> bytes:
    if (
        not isinstance(value, dict)
        or set(value) != {"$bytes"}
        or not isinstance(value["$bytes"], str)
    ):
        raise ValueError(f"{field} must use the serialized base64 byte representation")
    decoded = base64.b64decode(value["$bytes"], validate=True)
    if base64.b64encode(decoded).decode("ascii") != value["$bytes"]:
        raise ValueError(f"{field} has a noncanonical base64 encoding")
    return decoded


def _validate_json_content(validator, content_schema, instance, subschema):
    # JSON Schema normally treats contentSchema as an annotation. FMT's
    # explicit tool-envelope contract validates decoded JSON strings here so
    # OpenAI-style arguments cannot evade the inner tool parameter schema.
    import jsonschema

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


def _check_refs(value) -> None:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None and not reference.startswith("#"):
            raise ValueError(
                "tool schemas must bundle references locally; network schema resolution is disabled"
            )
        for child in value.values():
            _check_refs(child)
    elif isinstance(value, list):
        for child in value:
            _check_refs(child)


@cache
def _validator_type(base):
    """Each JSON Schema draft's validator class with the tool-envelope content rule (built once)."""
    import jsonschema

    return jsonschema.validators.extend(base, {"contentSchema": _validate_json_content})


_JSON_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})
_NUMBER_KEYWORDS = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})
_COUNT_KEYWORDS = frozenset({"minLength", "maxLength", "minItems", "maxItems"})
_SCHEMA_KEYWORDS = frozenset({"items", "additionalProperties", "not", "contentSchema"})
_SCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_TEXT_KEYWORDS = frozenset(
    {"title", "description", "$comment", "format", "contentMediaType", "contentEncoding"}
)
# "example" is no keyword of any vocabulary, so the metaschema leaves its value free.
_FREE_KEYWORDS = frozenset({"default", "const", "example"})
_FLAG_KEYWORDS = frozenset({"deprecated", "readOnly", "writeOnly"})


def _is_plain_schema(schema: Any) -> bool:
    """Whether ``schema`` is certainly valid under the 2020-12 metaschema.

    Tool parameter schemas are made of a few common keywords. For those, the metaschema
    check reduces to the type of each keyword's value, which takes microseconds instead of
    the milliseconds that resolving the metaschema's dynamic references costs. Any keyword
    not listed here, or any value that does not plainly fit, answers False and the caller
    runs the full check, so this never accepts what the metaschema would refuse.
    """
    if isinstance(schema, bool):
        return True
    if type(schema) is not dict:
        return False
    for key, value in schema.items():
        if key in _TEXT_KEYWORDS:
            ok = isinstance(value, str)
        elif key in _FREE_KEYWORDS:
            ok = True
        elif key == "type":
            ok = (isinstance(value, str) and value in _JSON_TYPES) or (
                type(value) is list
                and 0 < len(value) == len(set(map(str, value)))
                and all(isinstance(item, str) and item in _JSON_TYPES for item in value)
            )
        elif key == "properties":
            ok = type(value) is dict and all(_is_plain_schema(item) for item in value.values())
        elif key == "required":
            ok = (
                type(value) is list
                and all(isinstance(item, str) for item in value)
                and len(value) == len(set(value))
            )
        elif key == "enum":
            ok = type(value) is list
        elif key == "examples":
            ok = type(value) is list
        elif key in _SCHEMA_KEYWORDS:
            ok = _is_plain_schema(value)
        elif key in _SCHEMA_LIST_KEYWORDS:
            ok = type(value) is list and bool(value) and all(map(_is_plain_schema, value))
        elif key in _NUMBER_KEYWORDS:
            ok = type(value) in (int, float)
        elif key in _COUNT_KEYWORDS:
            ok = type(value) is int and value >= 0
        elif key in _FLAG_KEYWORDS:
            ok = isinstance(value, bool)
        else:
            ok = False
        if not ok:
            return False
    return True


# Real corpora reuse the same function definitions across many rows.
@lru_cache(maxsize=4096)
def _validator(schema_text: str):
    import jsonschema

    schema = strict_json_loads(schema_text)
    _check_refs(schema)
    base = jsonschema.validators.validator_for(schema)
    validator_type = _validator_type(base)
    if base is not jsonschema.Draft202012Validator or not _is_plain_schema(schema):
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
        if not 0 <= self.role_id < len(ROLE_TO_ID):
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
                raise ValueError("fixed syntax cannot define a generated identifier")


@dataclass(frozen=True)
class SchemaSpec:
    """A JSON object with fixed keys and typed decoded value spans.

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
    max_tokens: int = 512
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
        if self.require_terminal is not None and not isinstance(self.require_terminal, bool):
            raise ValueError("require_terminal must be true, false or null")
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


TAPE_VERSION = "fmt-canonical-facets-v8"


@dataclass(frozen=True)
class Decision:
    """An auditable deterministic decoder choice, never a policy factor."""

    name: str
    value: int


@dataclass(frozen=True)
class ActionTape:
    """The single Gaussian action and its deterministic materialization.

    The full FP32 raw facet tensor is authoritative. Replay recomputes the
    candidate Gaussian density, centrally processes these same coordinates
    against the same causal history, and verifies conditional realization.
    Encoding the emitted bytes would describe a different state trajectory.
    """

    version: str
    codec_fingerprint: str
    spec_fingerprint: str
    facet_shape: tuple[int, int]
    facets: tuple[float, ...] | bytes
    effort: int
    role_id: int
    payload: bytes
    emitted_ids: tuple[int, ...]
    valid: bool
    terminal: bool
    behavior_log_prob: float
    trace: tuple[Decision, ...] = ()
    error: str | None = None
    control_payload: bytes = b""

    def __post_init__(self) -> None:
        if self.version != TAPE_VERSION:
            raise ValueError("legacy action tapes cannot be replayed by the canonical-facet policy")
        for name in ("facet_shape", "emitted_ids", "trace"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if isinstance(self.facets, (bytes, bytearray)):
            object.__setattr__(self, "facets", bytes(self.facets))
        else:
            object.__setattr__(self, "facets", tuple(self.facets))
        object.__setattr__(self, "payload", bytes(self.payload))
        if not isinstance(self.control_payload, bytes):
            raise TypeError("control_payload must contain exact immutable bytes")
        if len(self.facet_shape) != 2 or any(value <= 0 for value in self.facet_shape):
            raise ValueError("facet_shape must contain a positive facet count and width")
        import numpy as np

        if isinstance(self.facets, bytes):
            if len(self.facets) != 4 * math.prod(self.facet_shape):
                raise ValueError(
                    "tape does not contain its complete raw facet tensor (packed FP32 length)"
                )
            values = np.frombuffer(self.facets, dtype="<f4")
            if not bool(np.isfinite(values).all()):
                raise ValueError("raw facet actions must be finite")
        else:
            if len(self.facets) != math.prod(self.facet_shape):
                raise ValueError("tape does not contain its complete raw facet tensor")
            if not all(math.isfinite(value) for value in self.facets):
                raise ValueError("raw facet actions must be finite")
        if not math.isfinite(self.behavior_log_prob):
            raise ValueError("behavior Gaussian density must be finite")
        if not 0 <= self.role_id < len(ROLE_TO_ID):
            raise ValueError("tape role ID is unsupported")

    def facet_array(self):
        """Raw FP32 array without allocating thousands of Python float objects."""
        import numpy as np

        if isinstance(self.facets, bytes):
            return np.frombuffer(self.facets, dtype="<f4")
        return np.asarray(self.facets, dtype=np.float32)

    @property
    def latent_dimensions(self) -> int:
        return math.prod(self.facet_shape)

    @property
    def scored_positions(self) -> int:
        """Deterministic lexical outputs contribute no policy likelihood."""
        return 0

    def to_json(self) -> str:
        return _json_bytes(asdict(self)).decode()

    @classmethod
    def from_json(cls, text: str) -> ActionTape:
        value = strict_json_loads(text)
        if not isinstance(value, dict):
            raise ValueError("action tape must be a JSON object")
        if value.get("version") != TAPE_VERSION:
            raise ValueError("legacy action tape version is incompatible with raw-facet replay")
        if "control_payload" not in value:
            raise ValueError("action tapes require their exact control_payload bytes")
        value["trace"] = tuple(Decision(**decision) for decision in value.get("trace", ()))
        value["payload"] = _parse_bytes(value["payload"], "payload")
        value["control_payload"] = _parse_bytes(value["control_payload"], "control_payload")
        if isinstance(value.get("facets"), dict):
            value["facets"] = _parse_bytes(value["facets"], "facets")
        return cls(**value)


class TapeMismatch(ValueError):
    """A tape is incomplete, tampered with, or infeasible under public constraints."""
