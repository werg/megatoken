"""Identity of the one frozen teacher, EmbeddingGemma2 (``google/embeddinggemma-2``).

It plays every teacher role: media (and text read in compressed form) enter E
as the ordered ``[T, 512]`` input of its ``language_model.embedding_projection``,
and the embedding head's targets are its pooled 768-d embeddings, both
from one forward pass (``megatoken.omni.embeddinggemma2``). The teacher never
joins a Megatoken checkpoint; its identity is pinned by the locked pack
manifest, so a feature cache, the transducer bank, a model and a serving boundary
agree on exactly the same weights and preprocessing.

Torch-free (outside the omni package): imported by model configuration and data validation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PACK = "embeddinggemma2"
HIDDEN_WIDTH = 512
EMBEDDING_WIDTH = 768
HIDDEN_TAP = "language_model.embedding_projection:input"
# Chat-template prompt, bf16 teacher weights; any change is a new identity.
PREPROCESSING = "chat-bf16-v1"
MAX_TOKENS = 8192
# google/embeddinggemma-2 @ 914f7f89, every snapshot file locked (pack manifest).
PINNED_TEACHER = (
    "embeddinggemma2:914f7f89142e33e77833254d9c9b90c3cef7303b:"
    "dcc342ef4bf5d120eeba035c685d9ced3b435d75f23d95a9a7c2ab77ad1383c9:"
    f"{HIDDEN_TAP}:{PREPROCESSING}"
)
_EXTRACTOR = re.compile(
    rf"embeddinggemma2:[0-9a-f]{{40}}:[0-9a-f]{{64}}:{re.escape(HIDDEN_TAP)}:{PREPROCESSING}"
)


def extractor_id(revision: str, files: dict[str, str], snapshot: str) -> str:
    """Bind the hub revision AND every locked snapshot file checksum into one identity.

    Paths are taken relative to the snapshot, so the identity does not depend on
    where a machine keeps its pack root.
    """
    prefix = Path(snapshot).as_posix().rstrip("/") + "/"
    if (
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not files
        or not all(path.startswith(prefix) for path in files)
        or any(".cache" in Path(path).parts for path in files)
    ):
        raise ValueError(
            "Teacher identity needs a 40-hex revision and its snapshot's model file hashes"
        )
    relative = {path.removeprefix(prefix): sha for path, sha in files.items()}
    digest = hashlib.sha256(
        json.dumps(relative, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"embeddinggemma2:{revision}:{digest}:{HIDDEN_TAP}:{PREPROCESSING}"


def validate_extractor(identity: str) -> str:
    if not isinstance(identity, str) or not _EXTRACTOR.fullmatch(identity):
        raise ValueError("Media features must come from the pinned EmbeddingGemma2 hidden tap")
    return identity


def locked_teacher(base: Path, packs: dict) -> tuple[Path, str]:
    """The teacher snapshot and identity from a verified pack manifest
    (``megatoken.omni.loader.read_locked_manifest``)."""
    item = packs.get(PACK)
    if item is None:
        raise ValueError(f"Pack manifest does not lock the {PACK} teacher")
    return base / item["snapshot"], extractor_id(item["revision"], item["files"], item["snapshot"])
