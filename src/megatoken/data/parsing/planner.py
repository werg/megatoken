"""Bounded dynamic programming over byte-exact, structurally legal token cuts.

The objective considers the complete candidate graph, including the final tail.
Candidate pruning bounds work; no decision re-tokenizes or rewrites the source.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .config import ChunkingConfig
from .types import Boundary, Diagnostic, ParseResult, Region

if TYPE_CHECKING:
    from ..schema import Codec

# Cost of ending a chunk at each kind of boundary. Structural cohesion wins over
# filling a few spare positions. Unknown extension kinds get a conservative cost.
BOUNDARY_COST = {
    "end": 0.0,
    "chat": 0.0,
    "json": 0.0,
    "code_block": 0.01,
    "function": 0.02,
    "definition": 0.02,
    "section": 0.02,
    "paragraph": 0.03,
    "list_item": 0.03,
    "element": 0.05,
    "block": 0.08,
    "row": 0.10,
    "statement": 0.12,
    "sentence": 0.18,
    "comment": 0.25,
    "string": 0.30,
    "line": 0.70,
    "word": 1.20,
    "heading": 3.0,
    "grapheme": 2.5,
    "token": 8.0,
}
_CONTAINERS = frozenset(
    {
        "function",
        "definition",
        "section",
        "code_block",
        "element",
        "list_item",
        "chat_message",
        "module",
        "document",
        "string",
        "comment",
    }
)
_ATTACHMENTS = frozenset({"heading_attachment", "signature_attachment"})
_PROSE = frozenset({"prose", "text", "markdown", "html", "xml", "rst", "org", "asciidoc"})


@dataclass(frozen=True)
class TokenIndex:
    payload: bytes
    ids: tuple[int, ...]
    offsets: tuple[int, ...]

    @classmethod
    def encode(cls, payload: bytes, codec: Codec) -> TokenIndex:
        ids = tuple(codec.encode(payload))
        pieces: dict[int, bytes] = {}
        offsets = [0]
        for token in ids:
            if token not in pieces:
                piece = codec.decode([token])
                if not isinstance(piece, bytes) or not piece:
                    raise ValueError("The raw codec must have nonempty byte-exact tokens")
                pieces[token] = piece
            piece = pieces[token]
            start = offsets[-1]
            if payload[start : start + len(piece)] != piece:
                raise ValueError("The raw codec must have byte-exact compositional tokens")
            offsets.append(start + len(piece))
        if offsets[-1] != len(payload):
            raise ValueError("The raw codec must preserve every source byte")
        return cls(payload, ids, tuple(offsets))

    def count(self, start: int, end: int) -> int:
        if end <= start:
            return 0
        return bisect.bisect_left(self.offsets, end) - max(
            0, bisect.bisect_right(self.offsets, start) - 1
        )

    def hard_end(self, start: int, config: ChunkingConfig) -> int:
        end = min(
            len(self.ids),
            start + config.hard_max_tokens,
            bisect.bisect_right(self.offsets, self.offsets[start] + config.max_bytes) - 1,
        )
        if end <= start:
            raise ValueError("One indivisible codec token exceeds max_record_bytes")
        return end


@dataclass(frozen=True, slots=True)
class PlannedChunk:
    start: int
    end: int
    boundary: str
    forced: bool
    tree_distance: float = 0.0


@dataclass(frozen=True, slots=True)
class ChunkPlan:
    chunks: tuple[PlannedChunk, ...]
    cost: float
    candidate_count: int
    evaluated_edges: int


def _cost(boundary: Boundary) -> float:
    return BOUNDARY_COST.get(boundary.kind, 0.4) + 0.002 * min(boundary.depth, 32)


def _safe_indices(index: TokenIndex) -> list[int]:
    from .prose import grapheme_cut_mask

    mask = grapheme_cut_mask(index.payload)
    offsets = np.fromiter(index.offsets, dtype=np.int64, count=len(index.offsets))
    return np.flatnonzero(mask[offsets]).tolist()


class _SafeCuts:
    """Project source coordinates onto canonical token edges outside graphemes."""

    def __init__(self, index: TokenIndex, safe: list[int]):
        self.index = index
        self.safe = safe

    def floor(self, offset: int) -> int:
        token = max(0, bisect.bisect_right(self.index.offsets, offset) - 1)
        return self.safe[max(0, bisect.bisect_right(self.safe, token) - 1)]

    def ceil(self, offset: int) -> int:
        token = bisect.bisect_left(self.index.offsets, offset)
        return self.safe[bisect.bisect_left(self.safe, token)]

    def fits(self, left: int, right: int, cap: int, max_bytes: int) -> bool:
        return (
            0 < right - left <= cap
            and self.index.offsets[right] - self.index.offsets[left] <= max_bytes
        )


class _AttachmentPrefixes:
    """Find useful bounded prefixes without repeatedly scanning parser evidence."""

    def __init__(self, index: TokenIndex, parsed: ParseResult, cuts: _SafeCuts):
        self.index, self.cuts = index, cuts
        self.heading_ends: dict[int, int] = {}
        body_starts, unit_ends = set(), set()
        for region in parsed.regions:
            if region.kind == "heading":
                self.heading_ends[region.start] = max(
                    region.end, self.heading_ends.get(region.start, 0)
                )
            if region.kind in {"code_block", "block", "statement"}:
                body_starts.add(region.start)
            if region.kind in {"sentence", "statement"}:
                unit_ends.add(region.end)
        self.body_starts = sorted(body_starts)
        self.unit_ends = sorted(unit_ends)
        self.word_ends = sorted({b.offset for b in parsed.boundaries if b.kind == "word"})
        self.cache: dict[tuple[int, int, str], int | None] = {}

    def end(self, region: Region, config: ChunkingConfig) -> int | None:
        key = region.start, region.end, region.kind
        if key not in self.cache:
            self.cache[key] = self._end(region, config)
        return self.cache[key]

    def _end(self, region: Region, config: ChunkingConfig) -> int | None:
        index, cuts = self.index, self.cuts
        left = cuts.floor(region.start)
        hard = index.hard_end(left, config)
        limit = min(region.end, index.offsets[hard])
        right = cuts.floor(limit)
        if right <= left:
            return None
        limit = index.offsets[right]
        anchor = (
            self.heading_ends.get(region.start) if region.kind == "heading_attachment" else None
        )
        if anchor is None and region.kind == "signature_attachment":
            position = bisect.bisect_right(self.body_starts, region.start)
            if position < len(self.body_starts) and self.body_starts[position] < region.end:
                anchor = self.body_starts[position]
        if anchor is None:
            # A provider may report just an attachment. Its first line is a
            # conservative header fallback; searches are bounded by capacity.
            endings = [
                end
                for end in (
                    index.payload.find(b"\n", region.start, limit),
                    index.payload.find(b"\r", region.start, limit),
                )
                if end >= 0
            ]
            anchor = min(endings) + 1 if endings else region.start
        if anchor >= limit:
            return None
        text = index.payload[anchor:limit].decode("utf-8", "surrogateescape")
        useful = text.lstrip()
        if not useful:
            return None
        first = anchor + len(text[: len(text) - len(useful)].encode("utf-8", "surrogateescape"))
        # Prefer a whole first sentence/statement, even when it uses more than
        # the target. Never call indentation or an empty separator a body.
        position = bisect.bisect_right(self.unit_ends, first)
        if position < len(self.unit_ends):
            end = self.unit_ends[position]
            if end <= region.end and cuts.fits(
                left, cuts.ceil(end), config.hard_max_tokens, config.max_bytes
            ):
                return end
        # The first meaningful unit may itself exceed capacity. Attach a useful
        # word/grapheme prefix near the target, leaving its remainder splittable.
        minimum = cuts.ceil(first + len(useful[0].encode("utf-8", "surrogateescape")))
        desired = cuts.floor(index.offsets[min(right, max(minimum, left + config.target_tokens))])
        if desired < minimum or desired <= left:
            return None
        position = bisect.bisect_right(self.word_ends, index.offsets[desired]) - 1
        if position < 0 or self.word_ends[position] <= first:
            # A long header or indentation may consume the size target. Keep
            # the first whole body word when capacity still permits it.
            position = bisect.bisect_right(self.word_ends, first)
        if position < len(self.word_ends):
            end = self.word_ends[position]
            if (
                end <= region.end
                and cuts.ceil(end) >= minimum
                and cuts.fits(left, cuts.ceil(end), config.hard_max_tokens, config.max_bytes)
            ):
                return end
        return index.offsets[desired]


@dataclass(frozen=True, slots=True)
class _SourceProtection:
    start: int
    end: int
    left: int
    right: int


def _protected_ranges(
    index: TokenIndex,
    parsed: ParseResult,
    config: ChunkingConfig,
    safe: list[int],
) -> list[tuple[int, int]]:
    cuts = _SafeCuts(index, safe)
    prefixes: _AttachmentPrefixes | None = None
    intervals = set()
    atomic = _ATTACHMENTS | {"json", "escape", "url", "email"}
    for region in parsed.regions:
        if not region.protect or region.end <= region.start:
            continue
        cap = (
            config.hard_max_tokens
            if region.kind == "json" or region.kind in _ATTACHMENTS
            else config.soft_max_tokens
        )
        end = region.end
        if region.kind in atomic:
            # Actual attachments and atomic lexemes retain their entire source,
            # including an outward edge that would otherwise fall inside CRLF.
            left, right = cuts.floor(region.start), cuts.ceil(end)
            if not cuts.fits(left, right, cap, config.max_bytes) and region.kind in _ATTACHMENTS:
                if prefixes is None:
                    prefixes = _AttachmentPrefixes(index, parsed, cuts)
                end = prefixes.end(region, config)
                if end is None:
                    continue
                right = cuts.ceil(end)
        else:
            # Adjacent sentences often share one BPE token (". The", "。これ").
            # Their canonical interiors remain cohesive, but their indivisible
            # edge token may belong to either neighbor. Outward expansion here
            # would incorrectly fuse an entire paragraph into a single atom.
            if index.count(region.start, end) > cap or end - region.start > config.max_bytes:
                continue
            left, right = cuts.ceil(region.start), cuts.floor(end)
        if cuts.fits(left, right, cap, config.max_bytes):
            intervals.add(_SourceProtection(region.start, end, left, right))

    # Diagnose contradictions in the parser's original geometry, before any
    # token/grapheme rounding. Projection overlap is not crossing source syntax.
    sources: list[_SourceProtection] = []
    for item in sorted(intervals, key=lambda r: (r.start, -r.end, r.left, r.right)):
        if sources and item.start < sources[-1].end:
            previous = sources[-1]
            left, right = min(previous.left, item.left), max(previous.right, item.right)
            if cuts.fits(left, right, config.hard_max_tokens, config.max_bytes):
                sources[-1] = _SourceProtection(
                    previous.start, max(previous.end, item.end), left, right
                )
            elif item.end > previous.end:
                parsed.diagnostics.append(
                    Diagnostic(
                        "crossing_protected_regions",
                        item.start,
                        item.end,
                        "planner",
                        "Overlapping source constraints exceed physical capacity",
                    )
                )
            elif left != previous.left or right != previous.right:
                parsed.diagnostics.append(
                    Diagnostic(
                        "token_overlapping_protection",
                        item.start,
                        item.end,
                        "planner",
                        "Contained atomic edge cannot expand an already full canonical span",
                    )
                )
        else:
            sources.append(item)

    merged: list[tuple[int, int]] = []
    for item in sources:
        left, right = item.left, item.right
        if merged and left < merged[-1][1]:
            old_left, old_right = merged[-1]
            combined = max(old_right, right)
            if cuts.fits(old_left, combined, config.hard_max_tokens, config.max_bytes):
                merged[-1] = old_left, combined
            elif right > old_right:
                # Two truly atomic neighbors can demand the same codec token.
                # Keep canonical IDs and the physical limit, and report this
                # benign ownership relaxation separately from parser conflicts.
                parsed.diagnostics.append(
                    Diagnostic(
                        "token_overlapping_protection",
                        index.offsets[left],
                        index.offsets[min(old_right, right)],
                        "planner",
                        "Adjacent source units share an indivisible codec token; the earlier unit owns it",
                    )
                )
                merged.append((old_right, right))
        else:
            merged.append((left, right))
    return merged


class _TreeDistances:
    """Weighted container movement with an event sweep and persistent ancestry.

    Native paths are authoritative. For lexical/document fallbacks, interval
    containment supplies the missing hierarchy; a singleton source-view prefix
    is context, not a complete semantic path. The sweep never rescans all active
    regions or materializes every enclosing fallback path.
    """

    class _Hierarchy:
        """Interned path prefixes with binary lifting for lowest common ancestors."""

        def __init__(self):
            self.parents = [0]
            self.depths = [0]
            self.jumps: list[list[int]] = [[0]]
            self.children: dict[tuple[int, object], int] = {}

        def child(self, parent: int, component: object) -> int:
            key = parent, component
            existing = self.children.get(key)
            if existing is not None:
                return existing
            node = len(self.parents)
            depth = self.depths[parent] + 1
            jumps = [parent]
            while 1 << len(jumps) <= depth:
                jumps.append(self.jumps[jumps[-1]][len(jumps) - 1])
            self.parents.append(parent)
            self.depths.append(depth)
            self.jumps.append(jumps)
            self.children[key] = node
            return node

        def path(self, components: tuple[str, ...]) -> int:
            node = 0
            for component in components:
                node = self.child(node, component)
            return node

        def distance(self, left: int, right: int) -> int:
            total = self.depths[left] + self.depths[right]
            if self.depths[left] < self.depths[right]:
                left, right = right, left
            difference = self.depths[left] - self.depths[right]
            power = 0
            while difference:
                if difference & 1:
                    left = self.jumps[left][power]
                difference >>= 1
                power += 1
            if left != right:
                for power in range(self.depths[left].bit_length() - 1, -1, -1):
                    a = self.jumps[left][power] if power < len(self.jumps[left]) else 0
                    b = self.jumps[right][power] if power < len(self.jumps[right]) else 0
                    if a != b:
                        left, right = a, b
                left = self.parents[left]
            return total - 2 * self.depths[left]

    @staticmethod
    def _view_only(path: tuple[str, ...]) -> bool:
        # SourceView.merge prefixes paths with start:end:depth. Generic parsers
        # have no AST path after that prefix; treating it as a full path made all
        # containers in a generic view indistinguishable (zero tree movement).
        if len(path) != 1:
            return False
        parts = path[0].split(":")
        return len(parts) == 3 and all(part.isascii() and part.isdecimal() for part in parts)

    @classmethod
    def _paths(cls, selected: list[Region], hierarchy: _Hierarchy) -> list[int]:
        nodes = [0] * len(selected)
        ordered = sorted(
            range(len(selected)),
            key=lambda i: (selected[i].start, -selected[i].end, selected[i].depth, i),
        )
        containers: list[int] = []
        cursor = 0
        while cursor < len(ordered):
            first = selected[ordered[cursor]]
            stop = cursor + 1
            while stop < len(ordered):
                other = selected[ordered[stop]]
                if (other.start, other.end) != (first.start, first.end):
                    break
                stop += 1
            # A later-starting interval whose end extends beyond an earlier one
            # dominates it for all future containment queries. Popping crossings
            # also prevents overlapping siblings from becoming fake ancestors.
            while containers and selected[containers[-1]].end < first.end:
                containers.pop()
            parent = containers[-1] if containers else None
            group = ordered[cursor:stop]
            for number in group:
                region = selected[number]
                if region.path and not cls._view_only(region.path):
                    nodes[number] = hierarchy.path(region.path)
                    continue
                base = nodes[parent] if parent is not None else 0
                if region.path:
                    context = region.path[0]
                    parent_path = selected[parent].path if parent is not None else ()
                    if not parent_path or parent_path[0] != context:
                        base = hierarchy.child(base, context)
                # Synthetic components cannot collide with native string keys.
                nodes[number] = hierarchy.child(base, ("interval", number))
            # Coextensive annotations are alternatives, not nested blocks. Pick
            # the same representative as the event heap when descendants need a
            # fallback parent; duplicate evidence cannot inflate generic depth.
            representative = min(
                group, key=lambda i: (-hierarchy.depths[nodes[i]], selected[i].depth, i)
            )
            containers.append(representative)
            cursor = stop
        return nodes

    def __init__(self, regions: list[Region], code_scale: float):
        import heapq

        selected = [item for item in regions if item.kind in _CONTAINERS and item.end > item.start]
        hierarchy = self._Hierarchy()
        nodes = self._paths(selected, hierarchy)
        events = sorted(
            event
            for number, region in enumerate(selected)
            for event in ((region.start, True, number), (region.end, False, number))
        )
        is_code = [bool(item.language and item.language not in _PROSE) for item in selected]
        active = [False] * len(selected)
        deepest: list[tuple[int, int, int, int, int]] = []
        code_count = previous = cursor = 0
        self.offsets: list[int] = []
        self.cumulative = [0.0]
        while cursor < len(events):
            offset = events[cursor][0]
            entered = False
            while cursor < len(events) and events[cursor][0] == offset:
                _, entering, number = events[cursor]
                active[number] = entering
                code_count += is_code[number] if entering else -is_code[number]
                if entering:
                    entered = True
                    region = selected[number]
                    heapq.heappush(
                        deepest,
                        (
                            -hierarchy.depths[nodes[number]],
                            region.start,
                            -region.end,
                            region.depth,
                            number,
                        ),
                    )
                cursor += 1
            while deepest and not active[deepest[0][-1]]:
                heapq.heappop(deepest)
            # Charge ascent when the next container is entered, not while
            # leaving the final block into its trailing newline/whitespace.
            if not entered or not deepest:
                continue
            current = nodes[deepest[0][-1]]
            distance = hierarchy.distance(previous, current)
            scale = code_scale if code_count else 1.0
            self.offsets.append(offset)
            self.cumulative.append(self.cumulative[-1] + distance * scale)
            previous = current

    def between(self, start: int, end: int) -> float:
        # Entering at the beginning and leaving at the end are not movement
        # *inside* a chunk. Whitespace between containers is still accounted for.
        left = bisect.bisect_right(self.offsets, start)
        right = bisect.bisect_left(self.offsets, end)
        return max(0.0, self.cumulative[right] - self.cumulative[left])


def _cheapest_cuts(
    index: TokenIndex,
    config: ChunkingConfig,
    candidates: dict[int, Boundary],
    points: list[int],
    mandatory_points: list[int],
    tree: _TreeDistances,
    whole_json: bool,
) -> tuple[list[int], float, int]:
    """The dynamic program over candidate cuts: each point's best next cut, and the total cost.

    Every point scores at most ``planner_max_edges`` outgoing edges, chosen by closeness to
    the size target and by cheapest boundary. Scores are evaluated for all of a point's
    edges at once with float64 arrays, in the same order of operations as a scalar loop.
    """
    n = len(index.ids)
    count = len(points)
    target = config.target_tokens
    limit = config.planner_max_edges
    positions = np.array(points, dtype=np.int64)
    boundary_cost = np.array([_cost(candidates[point]) for point in points], dtype=np.float64)
    offsets = np.fromiter(index.offsets, dtype=np.int64, count=len(index.offsets))[positions]
    tree_offsets = np.array(tree.offsets, dtype=np.int64)
    cumulative = np.array(tree.cumulative, dtype=np.float64)
    first_after = cumulative[np.searchsorted(tree_offsets, offsets, side="right")]
    last_before = cumulative[np.searchsorted(tree_offsets, offsets, side="left")]
    movement_scale = max(1.0, 2.0 * config.tree_distance_budget)
    total = np.full(count, math.inf)
    next_point = [-1] * count
    total[-1] = 0.0
    edge_count = 0
    arange = np.arange(count)
    for i in range(count - 2, -1, -1):
        start = points[i]
        hard = min(
            index.hard_end(start, config),
            mandatory_points[bisect.bisect_right(mandatory_points, start)],
        )
        stop = bisect.bisect_right(points, hard)
        if stop - i - 1 > limit:
            span = arange[i + 1 : stop]
            gap = np.abs(positions[i + 1 : stop] - start - target)
            by_target = np.lexsort((span, gap))
            by_structure = np.lexsort((span, gap, boundary_cost[i + 1 : stop]))
            chosen = {i + 1, stop - 1}
            chosen.update((by_target[: max(1, limit // 2 - 1)] + i + 1).tolist())
            for j in (by_structure[:limit] + i + 1).tolist():
                if len(chosen) >= limit:
                    break
                chosen.add(j)
            options = np.array(sorted(chosen), dtype=np.int64)
        else:
            options = arange[i + 1 : stop]
        ends = positions[options]
        length = ends - start
        ratio = length / target
        distance = np.maximum(0.0, last_before[options] - first_after[i])
        movement = np.maximum(0.0, distance - config.tree_distance_budget) / movement_scale
        cost = (ratio - 1.0) ** 2 + boundary_cost[options]
        cost += config.tree_distance_weight * movement * movement
        cost += 8.0 * (np.maximum(0, length - config.soft_max_tokens) / target) ** 2
        cost += 4.0 * np.maximum(0.0, 0.25 - ratio) ** 2
        # Complete JSON actions may intentionally use the hard capacity.
        if start == 0 and whole_json:
            cost[ends == n] = 0.0
        score = cost + total[options]
        edges = len(options)
        edge_count += edges
        best = np.lexsort((-ends, np.abs(length - target), score))[0]
        total[i] = score[best]
        next_point[i] = int(options[best])
        if not np.isfinite(total[i]):
            raise RuntimeError("Parser planner failed to make forward progress")
    return next_point, float(total[0]), edge_count


def plan_chunks(index: TokenIndex, parsed: ParseResult, config: ChunkingConfig) -> ChunkPlan:
    n = len(index.ids)
    if not n:
        return ChunkPlan((PlannedChunk(0, 0, "end", False),), 0.0, 1, 0)
    safe = _safe_indices(index)
    safe_set = set(safe)
    cuts = _SafeCuts(index, safe)
    protected = _protected_ranges(index, parsed, config, safe)
    protected_starts = [left for left, _ in protected]

    def blocked(position: int) -> bool:
        which = bisect.bisect_left(protected_starts, position) - 1
        return which >= 0 and position < protected[which][1]

    candidates: dict[int, Boundary] = {
        0: Boundary(0, "end"),
        n: Boundary(len(index.payload), "end"),
    }
    mandatory = {0, n}

    def add(position: int, boundary: Boundary, *, enforce: bool = False) -> None:
        if not 0 <= position <= n or (
            not enforce and (blocked(position) or position not in safe_set)
        ):
            return
        existing = candidates.get(position)
        if existing is None or boundary.mandatory or _cost(boundary) < _cost(existing):
            candidates[position] = boundary
        if boundary.mandatory:
            mandatory.add(position)

    for boundary in parsed.boundaries:
        if not 0 <= boundary.offset <= len(index.payload):
            continue
        position = bisect.bisect_left(index.offsets, boundary.offset)
        exact = index.offsets[position] == boundary.offset
        if boundary.mandatory:
            if not exact:
                raise ValueError(
                    "A mandatory event boundary crosses a codec token; frame events before tokenization"
                )
            add(position, boundary, enforce=True)
            continue
        if exact and position in safe_set:
            add(position, boundary)
            continue
        # Native syntax is expressed in source bytes, whereas canonical tokens
        # may straddle that boundary. Keep the nearest legal safe edge as useful
        # evidence instead of discarding the sentence/function boundary.
        choices = [
            candidate
            for candidate in {cuts.floor(boundary.offset), cuts.ceil(boundary.offset)}
            if not blocked(candidate)
        ]
        if choices:
            position = min(
                choices,
                key=lambda candidate: (abs(index.offsets[candidate] - boundary.offset), candidate),
            )
            add(position, boundary)

    # A large following section/function starts a new chunk. Otherwise its
    # signature/heading could be borrowed by the previous body to fill space.
    # Smaller containers can still be packed together, but only whole.
    for region in parsed.regions:
        if region.kind not in {"function", "definition", "section"}:
            continue
        if index.count(region.start, region.end) <= config.soft_max_tokens:
            continue
        position = cuts.floor(region.start)
        # A grammar may expose both a decorated definition and the enclosed
        # function. The inner start must not detach its decorators, comments,
        # or multiline signature from the first body statement.
        if not blocked(position):
            add(
                position,
                Boundary(region.start, region.kind, region.depth, region.source, True),
                enforce=True,
            )

    for left, right in protected:
        for position in (left, right):
            add(position, Boundary(index.offsets[position], "grapheme", source="protected_edge"))

    # Sparse fallback lattice bounds the candidate graph even for a giant atom.
    # Semantic boundaries and protected-unit edges are retained independently.
    stride = max(1, config.target_tokens // 8)
    for middle in range(stride, n, stride):
        j = bisect.bisect_left(safe, middle)
        for candidate in safe[max(0, j - 1) : j + 1]:
            add(candidate, Boundary(index.offsets[candidate], "grapheme", source="fallback"))

    # Any gap in the sparse graph must be traversable under both physical caps.
    # If a grapheme itself cannot fit, canonical token cuts are explicit damage.
    ordered = sorted(candidates)
    for left, right in zip(ordered, ordered[1:], strict=False):
        cursor = left
        while right > index.hard_end(cursor, config):
            hard = index.hard_end(cursor, config)
            j = bisect.bisect_right(safe, hard) - 1
            cut = safe[j] if j >= 0 and safe[j] > cursor else hard
            kind = "grapheme" if cut in safe_set else "token"
            add(cut, Boundary(index.offsets[cut], kind, source="physical_limit"), enforce=True)
            cursor = cut

    points = sorted(candidates)
    tree = _TreeDistances(parsed.regions, config.code_tree_scale)
    mandatory_points = sorted(mandatory)
    whole_json = any(
        r.kind == "json" and r.start == 0 and r.end == len(index.payload) for r in parsed.regions
    )
    next_point, total_cost, edge_count = _cheapest_cuts(
        index, config, candidates, points, mandatory_points, tree, whole_json
    )

    chunks = []
    cursor = 0
    while cursor < len(points) - 1:
        following = next_point[cursor]
        start, end = points[cursor], points[following]
        boundary = candidates[end]
        forced = boundary.kind in {"grapheme", "token"}
        if boundary.offset != index.offsets[end]:
            parsed.diagnostics.append(
                Diagnostic(
                    "semantic_boundary_projected",
                    min(boundary.offset, index.offsets[end]),
                    max(boundary.offset, index.offsets[end]),
                    "planner",
                    f"{boundary.kind} source boundary {boundary.offset} uses canonical safe edge {index.offsets[end]}",
                )
            )
        if forced:
            parsed.diagnostics.append(
                Diagnostic(
                    "forced_token_cut" if end not in safe_set else "forced_grapheme_cut",
                    index.offsets[start],
                    index.offsets[end],
                    "planner",
                    "No preferred structural cut selected within the physical policy",
                )
            )
        chunks.append(
            PlannedChunk(
                start,
                end,
                "end" if end == n else boundary.kind,
                forced,
                tree.between(index.offsets[start], index.offsets[end]),
            )
        )
        cursor = following
    return ChunkPlan(tuple(chunks), total_cost, len(points), edge_count)
