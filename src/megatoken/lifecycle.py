"""Session-owned track addresses, lifecycle transitions and bounded readiness.

No model parameter is indexed by these addresses. Completed work keeps a small
record so that late results cannot be redirected to a newly created task.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

LIFECYCLE_VERSION = 1
OPERATIONS = ("create", "assign", "progress", "wait", "continue", "complete", "cancel", "resume")


@dataclass(frozen=True)
class TrackDescription:
    address: str
    purpose: str
    label: str = ""
    parent: str | None = None
    completion: str = ""
    capabilities: tuple[str, ...] = ("text",)
    budget: int = 128
    revision: int = 0
    status: str = "active"
    prerequisites: tuple[str, ...] = ()
    progress: str = ""

    def __post_init__(self):
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        if type(self.budget) is not int or type(self.revision) is not int:
            raise ValueError("Track budget and revision must be integers")
        if not self.address or not self.purpose.strip() or self.budget < 0 or self.revision < 0:
            raise ValueError("Track needs a stable address, purpose and nonnegative work budget")
        if self.status not in {"active", "waiting", "completed", "cancelled"}:
            raise ValueError("Unknown track lifecycle state")
        if not self.capabilities or len(set(self.capabilities)) != len(self.capabilities):
            raise ValueError("Track capabilities must be distinct and nonempty")


@dataclass
class TrackRegistry:
    max_active: int = 8
    tracks: dict[str, TrackDescription] = field(default_factory=dict)
    sequence: int = 0
    address_prefix: str = "track"
    # Runtime choices and model actions are recorded separately by the caller.
    transitions: list[tuple[str, TrackDescription]] = field(default_factory=list)

    def create(self, purpose: str, **options) -> TrackDescription:
        if sum(t.status in {"active", "waiting"} for t in self.tracks.values()) >= self.max_active:
            raise ValueError("Active work capacity exhausted; queue work or finish an active track")
        if options.get("budget", 128) < 1:
            raise ValueError("New work requires a positive budget")
        parent = options.get("parent")
        if parent is not None and parent not in self.tracks:
            raise ValueError("Parent task does not exist")
        if parent is not None and not set(options.get("capabilities", ("text",))) <= set(
            self.tracks[parent].capabilities
        ):
            raise ValueError("Child work cannot acquire capabilities its parent does not have")
        address = f"{self.address_prefix}:{self.sequence}"
        description = TrackDescription(address, purpose, **options)
        self.sequence += 1
        self.tracks[address] = description
        self.transitions.append(("create", description))
        return description

    def change(self, address: str, operation: str, **changes) -> TrackDescription:
        current = self.tracks[address]
        if operation not in OPERATIONS[1:] + ("rename",):
            raise ValueError("Unknown lifecycle operation")
        if current.status in {"completed", "cancelled"} and operation not in {"resume", "rename"}:
            raise ValueError("Closed work requires explicit resumption")
        allowed = {
            "assign": {"purpose", "completion", "prerequisites"},
            "progress": {"progress"},
            "wait": {"prerequisites"},
            "rename": {"label"},
            "continue": set(),
            "complete": set(),
            "cancel": set(),
            "resume": set(),
        }[operation]
        if set(changes) - allowed:
            raise ValueError("Lifecycle action contains unsupported changes")
        if operation == "resume" and current.status in {"completed", "cancelled"}:
            if (
                sum(t.status in {"active", "waiting"} for t in self.tracks.values())
                >= self.max_active
            ):
                raise ValueError("Active work capacity exhausted; queue resumption")
        status = {
            "wait": "waiting",
            "complete": "completed",
            "cancel": "cancelled",
            "resume": "active",
            "continue": "active",
        }.get(operation, current.status)
        updated = replace(
            current,
            **changes,
            status=status,
            revision=current.revision + int(operation == "assign"),
        )
        self.tracks[address] = updated
        self.transitions.append((operation, updated))
        return updated

    def legal_tools(self, capabilities=("text",)):
        """Mask address selection with current public lifecycle state."""
        tools = lifecycle_tools(capabilities)
        ready = [t.address for t in self.tracks.values() if t.status == "active" and t.budget > 0]
        live = [t.address for t in self.tracks.values() if t.status in {"active", "waiting"}]
        closed = [t.address for t in self.tracks.values() if t.status in {"completed", "cancelled"}]
        available = (
            sum(t.status in {"active", "waiting"} for t in self.tracks.values()) < self.max_active
        )
        result = []
        for tool in tools:
            name = tool["name"]
            properties = tool["parameters"]["properties"]
            if name == "track_create":
                if not available:
                    continue
                if live:
                    properties["parent"] = {"enum": live}
                else:
                    properties.pop("parent")
            elif name == "track_resume":
                choices = closed if available else []
                if not choices:
                    continue
                properties["address"] = {"enum": choices}
            else:
                choices = ready if name == "track_output" else live
                if not choices:
                    continue
                properties["address"] = {"enum": choices}
            result.append(tool)
        return result

    def selectable(self, address: str, completed: set[str]) -> bool:
        track = self.tracks[address]
        return (
            track.status == "active" and track.budget > 0 and set(track.prerequisites) <= completed
        )

    def charge(self, addresses: tuple[str, ...]):
        counts = {a: addresses.count(a) for a in set(addresses)}
        for address, count in counts.items():
            if address in self.tracks and self.tracks[address].budget < count:
                raise ValueError("Track work budget exhausted before selection")
        for address, count in counts.items():
            if address in self.tracks:
                self.tracks[address] = replace(
                    self.tracks[address], budget=self.tracks[address].budget - count
                )


def lifecycle_tools(capabilities=("text",)) -> list[dict]:
    """Typed lifecycle actions use the existing constrained tool/content policy.

    These are genuine control messages, not dummy user-facing text. Their raw
    selection and policy density are retained in the ordinary grouped tape.
    """
    return [
        {
            "name": "track_create",
            "parameters": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "completion": {"type": "string"},
                    "budget": {"type": "integer", "minimum": 1},
                    "parent": {"type": "string"},
                    "capabilities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
                "required": ["purpose", "completion", "budget"],
                "additionalProperties": False,
            },
        },
        {
            "name": "track_output",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "request": {"type": "string"},
                    "modality": {"enum": ["auto", *capabilities]},
                },
                "required": ["address", "request", "modality"],
                "additionalProperties": False,
            },
        },
        *[
            {
                "name": "track_" + operation,
                "parameters": {
                    "type": "object",
                    "properties": {"address": {"type": "string"}, **properties},
                    "required": ["address", *properties],
                    "additionalProperties": False,
                },
            }
            for operation, properties in (
                ("assign", {"purpose": {"type": "string"}, "completion": {"type": "string"}}),
                ("progress", {"progress": {"type": "string"}}),
                (
                    "wait",
                    {
                        "prerequisites": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                        }
                    },
                ),
            )
        ],
        *[
            {
                "name": "track_" + operation,
                "parameters": {
                    "type": "object",
                    "properties": {"address": {"type": "string"}},
                    "required": ["address"],
                    "additionalProperties": False,
                },
            }
            for operation in ("continue", "complete", "cancel", "resume")
        ],
    ]
