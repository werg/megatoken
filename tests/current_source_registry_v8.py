"""Current producer-generated registries for tests that inject audit boundaries.

Each fixture retains the exact original source, lexical fingerprint, normalized
core and complete graph. Tests select a training seed only to exercise training
plumbing; this fixture selection is not a corpus or model-quality observation.
"""

import copy
import hashlib
import json
from pathlib import Path


def current_source_registry(source, identity="source-1", family="family-1"):
    key = hashlib.sha256(
        json.dumps([source, identity, family], ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    values = json.loads(
        (Path(__file__).parent / "fixtures" / "current-preparation-registries-v8.json").read_text()
    )
    if key not in values:
        raise ValueError("Regenerate the current producer registry for this exact fixture source")
    return copy.deepcopy(values[key])
