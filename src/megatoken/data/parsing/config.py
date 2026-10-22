"""Versioned parsing and planning policy, independent of reference tokenization."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from megatoken.io import content_hash

from .registry import parser_provider_specs
from .types import ParseLimits

if TYPE_CHECKING:
    from ..schema import Codec

CHUNKING_VERSION = "recursive-parser-v2"


@lru_cache(maxsize=1)
def implementation_fingerprint() -> str:
    """Parser amendments invalidate identities even before a package release."""
    root = Path(__file__).parent
    files = [
        *root.glob("*.py"),
        *root.glob("*.json"),
        *(root.parent / name for name in ("chunking.py", "adapters.py", "schema.py")),
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root.parent).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=64)
def _identity(
    config: ChunkingConfig, codec_fingerprint: str, providers: tuple[tuple[str, str], ...]
) -> dict[str, Any]:
    """Computed once per policy, codec and registered parser set: it hashes the runtime."""
    value = {
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "codec_fingerprint": codec_fingerprint,
        "parsers": config.parser_identity(),
    }
    return value | {"fingerprint": content_hash(value)}


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 512
    soft_max_tokens: int = 768
    hard_max_tokens: int = 1024
    max_bytes: int = 16384
    version: str = CHUNKING_VERSION
    parser_backend: str = "tree_sitter"
    chat_template: str = "auto"
    use_icu: bool = False
    limits: ParseLimits = field(default_factory=ParseLimits)
    planner_max_edges: int = 64
    tree_distance_budget: float = 4.0
    tree_distance_weight: float = 0.15
    code_tree_scale: float = 1.5

    def __post_init__(self) -> None:
        if isinstance(self.limits, dict):
            object.__setattr__(self, "limits", ParseLimits(**self.limits))
        if not isinstance(self.limits, ParseLimits):
            raise TypeError("limits must be ParseLimits or its serialized mapping")
        counts = (self.target_tokens, self.soft_max_tokens, self.hard_max_tokens, self.max_bytes)
        if any(type(value) is not int for value in counts):
            raise ValueError("Chunk sizes must be integer counts")
        if not 1 <= self.target_tokens <= self.soft_max_tokens <= self.hard_max_tokens:
            raise ValueError("Chunk sizes must satisfy 1 <= target <= soft <= hard")
        if self.max_bytes < 1:
            raise ValueError("Chunk max_bytes must be positive")
        if self.version != CHUNKING_VERSION:
            raise ValueError(f"Unsupported chunking version: {self.version}")
        if self.parser_backend not in {"tree_sitter", "fallback"}:
            raise ValueError("parser_backend must be tree_sitter or fallback")
        if not isinstance(self.chat_template, str) or not self.chat_template:
            raise ValueError("chat_template must name a template, auto, or none")
        if type(self.use_icu) is not bool:
            raise ValueError("use_icu must be boolean")
        if type(self.planner_max_edges) is not int or self.planner_max_edges < 4:
            raise ValueError("planner_max_edges must be an integer of at least four")
        for name in ("tree_distance_budget", "tree_distance_weight", "code_tree_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")

    def with_hard_limit(self, limit: int, *, max_bytes: int | None = None) -> ChunkingConfig:
        if type(limit) is not int or limit < 1:
            raise ValueError("The stage hard limit must be a positive integer")
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 1):
            raise ValueError("max_bytes must be a positive integer")
        hard = min(self.hard_max_tokens, limit)
        return replace(
            self,
            target_tokens=min(self.target_tokens, hard),
            soft_max_tokens=min(self.soft_max_tokens, hard),
            hard_max_tokens=hard,
            max_bytes=self.max_bytes if max_bytes is None else min(self.max_bytes, max_bytes),
        )

    def with_record_limits(
        self, *, max_tokens: int | None = None, max_bytes: int | None = None
    ) -> ChunkingConfig:
        """Set the hard physical-record limits, whether that lowers or raises them.

        Targets stay where they are unless the new hard limit is below them.
        """
        hard = self.hard_max_tokens if max_tokens is None else max_tokens
        return replace(
            self,
            hard_max_tokens=hard,
            soft_max_tokens=min(self.soft_max_tokens, hard),
            target_tokens=min(self.target_tokens, hard),
            max_bytes=self.max_bytes if max_bytes is None else max_bytes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def parser_identity(self) -> dict[str, Any]:
        from .chat import template_identity
        from .prose import prose_runtime_identity
        from .registry import parser_runtime_identity

        return {
            "implementation": implementation_fingerprint(),
            "syntax": parser_runtime_identity(self.parser_backend),
            "prose": prose_runtime_identity(self.use_icu),
            "chat": template_identity(),
        }

    @property
    def fingerprint(self) -> str:
        return content_hash({"config": self.to_dict(), "parsers": self.parser_identity()})

    def _identity(self, codec: Codec) -> dict[str, Any]:
        providers = tuple((item.name, item.version) for item in parser_provider_specs())
        return _identity(self, codec.fingerprint, providers)

    def identity(self, codec: Codec) -> dict[str, Any]:
        return copy.deepcopy(self._identity(codec))

    def identity_fingerprint(self, codec: Codec) -> str:
        """``identity(codec)["fingerprint"]`` without copying the identity."""
        return self._identity(codec)["fingerprint"]

    @classmethod
    def from_identity(cls, identity: dict[str, Any], codec: Codec) -> ChunkingConfig:
        try:
            config = cls(**identity["config"])
            if identity != config.identity(codec):
                raise ValueError("identity differs")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Incompatible chunker, codec, or parser runtime; regenerate/replay with the declared preparation policy"
            ) from error
        return config
