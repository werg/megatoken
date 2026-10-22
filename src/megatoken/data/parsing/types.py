"""Shared byte-coordinate contracts for independently composable parsers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Boundary:
    offset: int
    kind: str
    depth: int = 0
    source: str = "structural"
    mandatory: bool = False


@dataclass(frozen=True, slots=True)
class Region:
    start: int
    end: int
    kind: str
    language: str | None = None
    depth: int = 0
    protect: bool = True
    source: str = "structural"
    path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Diagnostic:
    code: str
    start: int
    end: int
    parser: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ParseLimits:
    max_depth: int = 12
    max_nodes: int = 100_000
    max_parse_bytes: int = 1_048_576
    max_regions: int = 20_000
    max_injections: int = 2_048
    max_tree_depth: int = 128
    max_parse_milliseconds: int = 0
    isolate_native: bool = True
    worker_timeout_ms: int = 5_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name == "isolate_native":
                if type(value) is not bool:
                    raise ValueError("isolate_native must be boolean")
                continue
            minimum = 0 if name == "max_parse_milliseconds" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class ParseResult:
    boundaries: list[Boundary] = field(default_factory=list)
    regions: list[Region] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    parsers: set[str] = field(default_factory=set)
    languages: set[str] = field(default_factory=set)

    def extend(self, other: ParseResult, *, offset: int = 0, depth: int = 0) -> None:
        """Merge independently parsed content without changing its source bytes."""
        if offset == 0 and depth == 0:
            # Evidence is immutable, so unshifted items are shared, not copied.
            self.boundaries.extend(other.boundaries)
            self.regions.extend(other.regions)
            self.diagnostics.extend(other.diagnostics)
        else:
            self.boundaries.extend(
                Boundary(
                    item.offset + offset, item.kind, item.depth + depth, item.source, item.mandatory
                )
                for item in other.boundaries
            )
            self.regions.extend(
                Region(
                    item.start + offset,
                    item.end + offset,
                    item.kind,
                    item.language,
                    item.depth + depth,
                    item.protect,
                    item.source,
                    item.path,
                )
                for item in other.regions
            )
            self.diagnostics.extend(
                Diagnostic(
                    item.code, item.start + offset, item.end + offset, item.parser, item.detail
                )
                for item in other.diagnostics
            )
        self.parsers.update(other.parsers)
        self.languages.update(other.languages)

    def validate(self, byte_length: int) -> None:
        """Reject invalid plugin evidence before it can affect physical records."""
        if any(not 0 <= item.offset <= byte_length for item in self.boundaries):
            raise ValueError("Parser boundary is outside the original source")
        if any(not 0 <= item.start <= item.end <= byte_length for item in self.regions):
            raise ValueError("Parser region is outside the original source")
        if any(not 0 <= item.start <= item.end <= byte_length for item in self.diagnostics):
            raise ValueError("Parser diagnostic is outside the original source")
