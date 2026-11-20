"""Historical V7 detail wire client of the neutral typed operation schema."""

from __future__ import annotations

import json
from typing import Any

from .semantic_json import stable_algorithm_json
from .semantic_records import (
    MAX_RECORD_BYTES,
    _fail,
    _pairs,
)
from .semantic_records import (
    SemanticRecordError as DetailedV7Error,
)
from .semantic_records import (
    graph as graph,
)
from .semantic_records import (
    validate_detailed_record as validate_detailed_record,
)
from .semantic_records import (
    validate_record_role as _role,
)
from .semantic_records import (
    value as value,
)


def encode_record(kind: str, fields: dict, *, role: str = "any") -> str:
    validate_detailed_record(kind, fields)
    _role(kind, role)
    result = "detail " + stable_algorithm_json({"kind": kind, "fields": fields})
    if len(result.encode("utf-8")) > MAX_RECORD_BYTES:
        _fail("Detailed record exceeds its declared byte bound")
    return result


def parse_record(line: Any, *, role: str = "any") -> tuple[str, dict]:
    if not isinstance(line, str) or not line.startswith("detail ") or "\n" in line or "\r" in line:
        _fail("Expected one explicit detailed v7 record")
    if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
        _fail("Detailed record exceeds its declared byte bound")
    try:
        record = json.loads(
            line[7:],
            object_pairs_hook=_pairs,
            parse_constant=lambda _: _fail("Nonfinite JSON must use tagged numbers"),
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise DetailedV7Error("Invalid detailed v7 JSON") from error
    if type(record) is not dict or set(record) != {"kind", "fields"}:
        _fail("Detailed wrapper requires exactly kind and fields")
    kind, fields = record["kind"], record["fields"]
    if encode_record(kind, fields, role=role) != line:
        _fail("Noncanonical detailed v7 bytes")
    return kind, fields
