"""Session-owned track addresses, lifecycle transitions and bounded readiness.

No model parameter is indexed by these addresses. Completed work keeps a small
record so that late results cannot be redirected to a newly created task.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

LIFECYCLE_VERSION = 2
OPERATIONS = ("create", "assign", "progress", "wait", "continue", "complete", "cancel", "resume")


@dataclass(frozen=True)
class ExecutionBinding:
    """Runtime-validated invocation identity attached to a created work track.

    This is a typed ingress intent, not a generated JS source string or a
    completed invocation. Code and input must be published *after* creating
    the track and *before* activate_execution marks it runnable.
    """

    program_id: str
    source_digest: str
    entry: str
    invocation_id: str
    realm_id: str
    trigger_id: str
    input_ref: str
    realm_policy: str = "fresh"
    completion_mode: str = "await-returned-promise"
    capability_grants: tuple[str, ...] = ()
    completion_target: str = ""
    schema: str = "semantic-execution-binding/4"

    def __post_init__(self):
        object.__setattr__(self, "capability_grants", tuple(self.capability_grants))
        if self.schema != "semantic-execution-binding/4":
            raise ValueError("Unrecognized selected execution-binding schema")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (
                self.program_id,
                self.source_digest,
                self.entry,
                self.invocation_id,
                self.realm_id,
                self.trigger_id,
                self.input_ref,
            )
        ):
            raise ValueError(
                "Execution binding needs source, invocation, input and trigger identities"
            )
        if len(self.source_digest) != 64 or any(
            char not in "0123456789abcdef" for char in self.source_digest
        ):
            raise ValueError("Execution binding requires pinned lowercase SHA-256 source")
        if self.realm_policy not in {"fresh", "retained"}:
            raise ValueError("Execution binding needs explicit fresh or retained realm policy")
        if self.completion_mode not in {
            "call-return",
            "await-returned-promise",
            "owned-quiescence",
        }:
            raise ValueError("Unknown selected execution completion policy")
        if len(self.capability_grants) != len(set(self.capability_grants)) or any(
            not isinstance(name, str) or not name for name in self.capability_grants
        ):
            raise ValueError("Execution capability grants must be distinct")


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
    execution: ExecutionBinding | None = None
    bootstrap_ready: bool = False

    def __post_init__(self):
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "prerequisites", tuple(self.prerequisites))
        if isinstance(self.execution, dict):
            object.__setattr__(self, "execution", ExecutionBinding(**self.execution))
        elif self.execution is not None and not isinstance(self.execution, ExecutionBinding):
            raise ValueError("Execution binding must be typed or a validated mapping")
        if type(self.bootstrap_ready) is not bool or (
            self.bootstrap_ready and self.execution is None
        ):
            raise ValueError("Only execution tracks can be marked bootstrapped")
        if (
            self.execution is not None
            and not self.bootstrap_ready
            and self.status not in {"waiting", "cancelled"}
        ):
            raise ValueError("Unbootstrapped execution tracks cannot be runnable")

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
    max_waiting: int = 64
    tracks: dict[str, TrackDescription] = field(default_factory=dict)
    sequence: int = 0
    address_prefix: str = "track"
    # Runtime choices and model actions are recorded separately by the caller.
    transitions: list[tuple[str, TrackDescription]] = field(default_factory=list)
    admissions: dict[str, dict] = field(default_factory=dict)
    dependencies: dict[str, dict] = field(default_factory=dict)
    completions: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        if (
            type(self.max_active) is not int
            or self.max_active < 1
            or type(self.max_waiting) is not int
            or self.max_waiting < 0
        ):
            raise ValueError(
                "Track capacities need a positive active and nonnegative waiting limit"
            )

    def create(self, purpose: str, **options) -> TrackDescription:
        if (
            options.get("execution") is None
            and options.get("status", "active") == "active"
            and sum(t.status == "active" for t in self.tracks.values()) >= self.max_active
        ):
            raise ValueError("Active work capacity exhausted; queue work or finish an active track")
        if options.get("budget", 128) < 1:
            raise ValueError("New work requires a positive budget")
        if (
            options.get("execution") is None
            and options.get("status") == "waiting"
            and sum(track.status == "waiting" for track in self.tracks.values()) >= self.max_waiting
        ):
            raise ValueError("Waiting work capacity exhausted; finish or cancel pending work")
        execution = options.get("execution")
        if execution is not None:
            if isinstance(execution, dict):
                execution = ExecutionBinding(**execution)
            if not isinstance(execution, ExecutionBinding):
                raise ValueError("Execution work requires a typed invocation binding")
            if sum(t.status == "waiting" for t in self.tracks.values()) >= self.max_waiting:
                raise ValueError("Execution bootstrap waiting capacity exhausted")
            if any(
                t.execution and t.execution.invocation_id == execution.invocation_id
                for t in self.tracks.values()
            ):
                raise ValueError("An execution invocation cannot create a second track")
            if options.get("status", "waiting") != "waiting" or options.get("bootstrap_ready"):
                raise ValueError(
                    "New execution work must await observed program and invocation bootstrap"
                )
            options["execution"] = execution
            options["status"] = "waiting"
        parent = options.get("parent")
        if parent is not None and parent not in self.tracks:
            raise ValueError("Parent task does not exist")
        if parent is not None and self.tracks[parent].status in {"completed", "cancelled"}:
            raise ValueError("Closed parent tasks cannot acquire new children")
        if parent is not None and not set(options.get("capabilities", ("text",))) <= set(
            self.tracks[parent].capabilities
        ):
            raise ValueError("Child work cannot acquire capabilities its parent does not have")
        if execution is not None and parent is not None:
            inherited = self.tracks[parent].execution
            grants = inherited.capability_grants if inherited else self.tracks[parent].capabilities
            if not set(execution.capability_grants) <= set(grants):
                raise ValueError("Child execution cannot expand parent external grants")
            if execution.completion_target not in {"", parent}:
                raise ValueError("Child execution completion must return to its owning parent")
        address = options.pop("address", None) or f"{self.address_prefix}:{self.sequence}"
        if not isinstance(address, str) or not address or address in self.tracks:
            raise ValueError("Track creation requires a fresh address")
        description = TrackDescription(address, purpose, **options)
        self.sequence += 1
        self.tracks[address] = description
        self.transitions.append(("create", description))
        return description

    def admit_execution(self, ingress_key: str, *, execution, purpose: str, **options):
        """Reserve one address per admitted outside/selected activation.

        Duplicate delivery returns the original allocation; changing a retry's
        invocation, arguments, owner or grants is an error. Failed bootstrap is
        retained as cancelled work and cannot silently allocate another child.
        """
        if not isinstance(ingress_key, str) or not ingress_key:
            raise ValueError("Execution ingress requires a stable delivery key")
        execution = ExecutionBinding(**execution) if isinstance(execution, dict) else execution
        if not isinstance(execution, ExecutionBinding):
            raise ValueError("Execution admission requires a typed binding")
        payload = json.dumps(
            {"execution": asdict(execution), "purpose": purpose, "options": options},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        previous = self.admissions.get(ingress_key)
        if previous is not None:
            if previous["digest"] != digest:
                raise ValueError("Execution ingress retry changed its immutable binding")
            return self.tracks[previous["address"]]
        description = self.create(purpose, execution=execution, **options)
        self.admissions[ingress_key] = {"digest": digest, "address": description.address}
        return description

    def completion_dependency(self, address: str) -> dict:
        """Issue a typed dependency for this exact invocation and task revision."""
        track = self.tracks[address]
        if track.execution is None:
            raise ValueError("Only execution work has an invocation completion dependency")
        key = f"completion:{address}:{track.revision}"
        reference = {
            "schema": "semantic-completion-dependency/4",
            "id": key,
            "track": address,
            "revision": track.revision,
            "invocation": track.execution.invocation_id,
        }
        if key in self.dependencies and self.dependencies[key] != reference:
            raise ValueError("Completion dependency identity cannot be reused")
        self.dependencies[key] = reference
        return dict(reference)

    def _dependency(self, reference: dict) -> dict:
        if (
            not isinstance(reference, dict)
            or self.dependencies.get(reference.get("id")) != reference
        ):
            raise ValueError("Dependency was not issued for this runtime invocation")
        return reference

    def wait_for_execution(
        self, address: str, reference: dict, *, delivered: set[str] | None = None
    ) -> TrackDescription:
        reference = self._dependency(reference)
        child = self.tracks[reference["track"]]
        if child.parent != address:
            raise ValueError("Child completion can only suspend its owning parent")
        current = self.tracks[address]
        dependencies = tuple(dict.fromkeys((*current.prerequisites, reference["id"])))
        waiting = self.change(address, "wait", prerequisites=dependencies)
        # Joining frees an active slot for an already published child.
        self.wake_ready_executions(delivered=delivered or set())
        return waiting

    def wake_execution(self, address: str, *, delivered: set[str]) -> TrackDescription:
        """Readiness follows observed delivery and capacity admission, never arrival alone."""
        current = self.tracks[address]
        coordinator_join = current.execution is None and any(
            key in self.dependencies for key in current.prerequisites
        )
        if current.status != "waiting" or not (current.bootstrap_ready or coordinator_join):
            raise ValueError("Only bootstrapped execution or a typed parent join can be woken")
        if not set(current.prerequisites) <= delivered:
            raise ValueError("Execution dependencies have not all been delivered")
        return self.change(address, "continue")

    def wake_ready_executions(self, *, delivered: set[str]) -> tuple[TrackDescription, ...]:
        """Admit ready waiting work in stable creation order under the active bound."""
        ready = []
        for track in tuple(self.tracks.values()):
            if sum(t.status == "active" for t in self.tracks.values()) >= self.max_active:
                break
            if (
                track.status == "waiting"
                and (
                    track.bootstrap_ready
                    or (
                        track.execution is None
                        and any(key in self.dependencies for key in track.prerequisites)
                    )
                )
                and set(track.prerequisites) <= delivered
            ):
                ready.append(self.wake_execution(track.address, delivered=delivered))
        return tuple(ready)

    def complete_execution(self, address: str, *, kind: str, value) -> dict:
        """Produce a scoped receipt distinct from the earlier creation acknowledgement."""
        track = self.tracks[address]
        if track.execution is None or kind not in {"return", "throw", "cancelled", "failed"}:
            raise ValueError("Execution completion requires a bound invocation and disposition")
        reference = self.completion_dependency(address)
        receipt = {
            "schema": "semantic-completion/4",
            "dependency": reference,
            "kind": kind,
            "value": value,
            "parent": track.parent,
            "target": track.execution.completion_target or track.parent,
        }
        # Verify transport representability before any lifecycle mutation.
        json.dumps(receipt, allow_nan=False)
        previous = self.completions.get(reference["id"])
        if previous is not None:
            if previous != receipt:
                raise ValueError("Invocation completion cannot change after publication")
            return dict(previous)
        if kind in {"return", "throw"} and any(
            child.parent == address and child.status in {"active", "waiting"}
            for child in self.tracks.values()
        ):
            raise ValueError("Required child work blocks parent completion")
        if track.status == "cancelled" and kind not in {"cancelled", "failed"}:
            raise ValueError("Cancelled invocation cannot publish successful completion")
        if track.status not in {"completed", "cancelled"}:
            self.change(address, "cancel" if kind in {"cancelled", "failed"} else "complete")
        self.completions[reference["id"]] = receipt
        return dict(receipt)

    def checkpoint(self) -> dict:
        return {
            "schema": "semantic-lifecycle-checkpoint/4",
            "version": LIFECYCLE_VERSION,
            "max_active": self.max_active,
            "max_waiting": self.max_waiting,
            "sequence": self.sequence,
            "address_prefix": self.address_prefix,
            "tracks": [asdict(track) for track in self.tracks.values()],
            "admissions": self.admissions,
            "dependencies": self.dependencies,
            "completions": self.completions,
        }

    @classmethod
    def restore(cls, checkpoint: dict):
        required = {
            "schema",
            "version",
            "max_active",
            "max_waiting",
            "sequence",
            "address_prefix",
            "tracks",
            "admissions",
            "dependencies",
            "completions",
        }
        if (
            not isinstance(checkpoint, dict)
            or set(checkpoint) != required
            or checkpoint["schema"] != "semantic-lifecycle-checkpoint/4"
            or checkpoint["version"] != LIFECYCLE_VERSION
        ):
            raise ValueError("Incompatible lifecycle checkpoint")
        saved = json.loads(json.dumps(checkpoint, allow_nan=False))
        result = cls(
            max_active=saved["max_active"],
            max_waiting=saved["max_waiting"],
            sequence=saved["sequence"],
            address_prefix=saved["address_prefix"],
        )
        if (
            type(result.sequence) is not int
            or result.sequence < 0
            or not isinstance(result.address_prefix, str)
            or not result.address_prefix
        ):
            raise ValueError("Invalid persisted lifecycle allocation sequence")
        for value in saved["tracks"]:
            track = TrackDescription(**value)
            if track.address in result.tracks:
                raise ValueError("Duplicate checkpoint track")
            result.tracks[track.address] = track
        if (
            sum(t.status == "active" for t in result.tracks.values()) > result.max_active
            or sum(t.status == "waiting" for t in result.tracks.values()) > result.max_waiting
            or any(
                t.parent is not None and t.parent not in result.tracks
                for t in result.tracks.values()
            )
        ):
            raise ValueError("Checkpoint ownership or capacity is invalid")
        result.admissions = saved["admissions"]
        result.dependencies = saved["dependencies"]
        result.completions = saved["completions"]
        invocations = [
            track.execution.invocation_id for track in result.tracks.values() if track.execution
        ]
        if len(invocations) != len(set(invocations)):
            raise ValueError("A restored invocation cannot own two tracks")
        for track in result.tracks.values():
            seen, owner = {track.address}, track.parent
            while owner is not None:
                if owner in seen:
                    raise ValueError("Checkpoint parent ownership contains a cycle")
                seen.add(owner)
                owner = result.tracks[owner].parent
        for admission in result.admissions.values():
            if (
                admission.get("address") not in result.tracks
                or len(admission.get("digest", "")) != 64
            ):
                raise ValueError("Invalid persisted execution admission")
        for key, reference in result.dependencies.items():
            track = result.tracks.get(reference.get("track"))
            if (
                not track
                or not track.execution
                or reference.get("id") != key
                or reference.get("schema") != "semantic-completion-dependency/4"
                or type(reference.get("revision")) is not int
                or reference["revision"] > track.revision
                or reference["revision"] < 0
                or (
                    reference["revision"] == track.revision
                    and reference.get("invocation") != track.execution.invocation_id
                )
            ):
                raise ValueError("Invalid persisted completion dependency")
        for key, receipt in result.completions.items():
            if receipt.get("schema") != "semantic-completion/4" or result.dependencies.get(
                key
            ) != receipt.get("dependency"):
                raise ValueError("Invalid persisted completion receipt")
        return result

    def activate_execution(
        self, address: str, *, source_digest: str, input_ref: str
    ) -> TrackDescription:
        """Host-only, idempotent post-publication bootstrap. Never a model tool."""
        current = self.tracks[address]
        binding = current.execution
        if (
            binding is None
            or binding.source_digest != source_digest
            or binding.input_ref != input_ref
        ):
            raise ValueError("Execution bootstrap must match previously observed code and input")
        if current.bootstrap_ready:
            if current.status in {"completed", "cancelled"}:
                raise ValueError("Closed execution bootstrap cannot be resumed")
            return current
        if current.status != "waiting":
            raise ValueError("Closed or mutated execution bootstrap cannot be resumed")
        available = sum(t.status == "active" for t in self.tracks.values()) < self.max_active
        # Publishing source consumes no active execution slot. Capacity queues
        # a fully bootstrapped child instead of turning successful publication
        # into a failed invocation that callers might retry with duplicate work.
        updated = replace(
            current, bootstrap_ready=True, status="active" if available else "waiting"
        )
        self.tracks[address] = updated
        self.transitions.append(("bootstrap", updated))
        return updated

    def change(self, address: str, operation: str, **changes) -> TrackDescription:
        current = self.tracks[address]
        if (
            current.execution is not None
            and not current.bootstrap_ready
            and operation
            not in {
                "cancel",
                "wait",
                "progress",
                "rename",
            }
        ):
            raise ValueError("Execution bootstrap must publish source and invocation before work")
        if operation not in OPERATIONS[1:] + ("rename",):
            raise ValueError("Unknown lifecycle operation")
        if current.status in {"completed", "cancelled"} and operation not in {"resume", "rename"}:
            raise ValueError("Closed work requires explicit resumption")
        allowed = {
            "assign": {"purpose", "completion", "prerequisites", "execution"},
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
        if "execution" in changes:
            if operation != "assign" or current.status != "waiting":
                raise ValueError("A new execution binding needs an eligible idle waiting track")
            if (
                current.execution
                and f"completion:{address}:{current.revision}" not in self.completions
            ):
                raise ValueError("Assign cannot replace an unfinished owned invocation")
            binding = changes["execution"]
            binding = ExecutionBinding(**binding) if isinstance(binding, dict) else binding
            if not isinstance(binding, ExecutionBinding) or any(
                t.execution and t.execution.invocation_id == binding.invocation_id
                for t in self.tracks.values()
            ):
                raise ValueError("Assigned invocation must have a fresh typed identity")
            if not set(binding.capability_grants) <= set(current.capabilities):
                raise ValueError("Assigned execution cannot expand the track's grants")
            changes["execution"] = binding
            changes["bootstrap_ready"] = False
        if operation in {"resume", "continue"} and current.status != "active":
            if sum(t.status == "active" for t in self.tracks.values()) >= self.max_active:
                raise ValueError("Active work capacity exhausted; queue resumption")
        if (
            operation == "wait"
            and current.status != "waiting"
            and sum(t.status == "waiting" for t in self.tracks.values()) >= self.max_waiting
        ):
            raise ValueError("Waiting work capacity exhausted; finish or cancel pending work")
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
        if operation == "cancel":
            for child in tuple(self.tracks.values()):
                if child.parent == address and child.status in {"active", "waiting"}:
                    self.change(child.address, "cancel")
        return updated

    def legal_tools(self, capabilities=("text",)):
        """Mask address selection with current public lifecycle state."""
        tools = lifecycle_tools(capabilities)
        ready = [t.address for t in self.tracks.values() if t.status == "active" and t.budget > 0]
        live = [t.address for t in self.tracks.values() if t.status in {"active", "waiting"}]
        closed = [t.address for t in self.tracks.values() if t.status in {"completed", "cancelled"}]
        available = sum(t.status == "active" for t in self.tracks.values()) < self.max_active
        result = []
        for tool in tools:
            name = tool["name"]
            properties = tool["parameters"]["properties"]
            if name == "track_create":
                if not available:
                    if sum(t.status == "waiting" for t in self.tracks.values()) >= self.max_waiting:
                        continue
                    tool["parameters"]["required"].append("execution")
                if live:
                    properties["parent"] = {"enum": live}
                else:
                    properties.pop("parent")
            elif name == "track_resume":
                choices = closed if available else []
                if not choices:
                    continue
                properties["address"] = {"enum": choices}
            elif name in {"track_continue", "track_wait"}:
                choices = [
                    track.address
                    for track in self.tracks.values()
                    if track.address in live
                    and (
                        (name == "track_continue" and (track.status == "active" or available))
                        or (
                            name == "track_wait"
                            and (
                                track.status == "waiting"
                                or sum(t.status == "waiting" for t in self.tracks.values())
                                < self.max_waiting
                            )
                        )
                    )
                ]
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
            track.status == "active"
            and track.budget > 0
            and (track.execution is None or track.bootstrap_ready)
            and set(track.prerequisites) <= completed
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
    tools = [
        {
            "name": "track_join",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "completion_dependency": {
                        "type": "object",
                        "properties": {
                            "schema": {"const": "semantic-completion-dependency/4"},
                            "id": {"type": "string"},
                            "track": {"type": "string"},
                            "revision": {"type": "integer", "minimum": 0},
                            "invocation": {"type": "string"},
                        },
                        "required": ["schema", "id", "track", "revision", "invocation"],
                        "additionalProperties": False,
                    },
                },
                "required": ["address", "completion_dependency"],
                "additionalProperties": False,
            },
        },
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
                    "execution": {
                        "type": "object",
                        "properties": {
                            **{
                                key: {"type": "string", "minLength": 1}
                                for key in (
                                    "program_id",
                                    "source_digest",
                                    "entry",
                                    "invocation_id",
                                    "realm_id",
                                    "trigger_id",
                                    "input_ref",
                                )
                            },
                            "realm_policy": {"enum": ["fresh", "retained"]},
                            "completion_mode": {
                                "enum": [
                                    "call-return",
                                    "await-returned-promise",
                                    "owned-quiescence",
                                ]
                            },
                            "capability_grants": {
                                "type": "array",
                                "items": {"type": "string"},
                                "uniqueItems": True,
                            },
                            "completion_target": {"type": "string"},
                            "schema": {"const": "semantic-execution-binding/4"},
                        },
                        "required": [
                            "program_id",
                            "source_digest",
                            "entry",
                            "invocation_id",
                            "realm_id",
                            "trigger_id",
                            "input_ref",
                            "realm_policy",
                            "completion_mode",
                            "capability_grants",
                        ],
                        "additionalProperties": False,
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
        {
            "name": "track_context",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "local_events": {"type": "integer", "minimum": 0},
                    "cross_events": {"type": "integer", "minimum": 0},
                    "pinned_events": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 8,
                        "uniqueItems": True,
                    },
                },
                "required": ["address", "local_events", "cross_events", "pinned_events"],
                "additionalProperties": False,
            },
        },
        {
            "name": "track_fetch",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string"},
                    "source_event": {"type": "string"},
                },
                "required": ["address", "source_event"],
                "additionalProperties": False,
            },
        },
        {
            "name": "track_context_status",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"],
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

    create = next(tool for tool in tools if tool["name"] == "track_create")
    assign = next(tool for tool in tools if tool["name"] == "track_assign")
    assign["parameters"]["properties"]["execution"] = create["parameters"]["properties"][
        "execution"
    ]
    return tools
