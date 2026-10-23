"""Nonblocking model-driven work loop over the ordinary serving runtime.

The host supplies capabilities and bounds. The model selects typed lifecycle
operations, tools, content and modalities; the scheduler only chooses among
already requested ready events. Every effect follows a committed selection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace

import torch

from megatoken.controls import build_control_payload
from megatoken.generation.structure import ActionSpec
from megatoken.multitrack_runtime import ConcurrentToolRunner
from megatoken.tool_schema import compile_tool_spec, parse_tool_action


@dataclass
class SessionExecutor:
    runtime: object
    session: object
    tools: tuple[dict, ...] = ()
    adapters: dict = field(default_factory=dict)
    max_selections: int = 128
    runner: ConcurrentToolRunner = field(default_factory=ConcurrentToolRunner)
    processed: set[int] = field(default_factory=set)
    selections: int = 0
    root: str | None = None
    queue_controls: dict[str, bytes] = field(default_factory=dict)

    def __post_init__(self):
        if self.max_selections < 1 or {tool["name"] for tool in self.tools} != set(self.adapters):
            raise ValueError(
                "Execution needs positive bounds and exactly the declared tool adapters"
            )

    def close(self):
        self.runner.close()

    def tick(self, *, effort=6, seed=0):
        with self.session.lock:
            return self._tick(effort=effort, seed=seed)

    def _tick(self, *, effort=6, seed=0):
        runtime, session = self.runtime, self.session
        self.processed.intersection_update(session.rendering)
        queued = {work.event_id for work in session.queue}
        self.queue_controls = {
            key: value for key, value in self.queue_controls.items() if key in queued
        }
        results, failures = runtime.collect_tools(session, self.runner, effort=effort)
        outputs, outcomes = [], []
        for revision in tuple(session.rendering):
            if revision in self.processed:
                continue
            work = runtime.poll_generation(session, revision)
            if work.status == "rendering":
                continue
            self.processed.add(revision)
            if work.result is None:
                outcomes.append(
                    {"selection_revision": revision, "status": work.status, "error": work.error}
                )
                continue
            group = work.result
            has_lifecycle = has_external = False
            for payload, valid, modality in zip(
                group.tape.payloads, group.tape.valid, group.tape.modalities, strict=True
            ):
                if not valid or not payload or modality != "text":
                    continue
                try:
                    name, _, _, _ = parse_tool_action(payload)
                except (ValueError, KeyError, TypeError):
                    continue
                has_lifecycle |= name.startswith("track_")
                has_external |= name in self.adapters
            # Dispatch methods filter their own capabilities; mixed tool and
            # user-facing peers remain one committed policy decision.
            if has_lifecycle:
                outcomes.extend(runtime.execute_lifecycle(session, group))
            if has_external:
                runtime.dispatch_tools(
                    session,
                    group,
                    self.runner,
                    self.adapters,
                    schemas={tool["name"]: tool["parameters"] for tool in self.tools},
                )
            outputs.append(
                {"selection_revision": revision, "tape": json.loads(group.tape.to_json())}
            )
        if self.selections >= self.max_selections:
            return {"status": "budget_exhausted", "outputs": outputs, "outcomes": outcomes}
        if (
            sum(w.status == "rendering" for w in session.rendering.values())
            >= runtime.max_rendering
        ):
            return {"status": "rendering", "outputs": outputs, "outcomes": outcomes}
        if (
            len(session.selected_groups.keys() | session.rendering.keys())
            >= runtime.max_retained_outputs
        ):
            return {"status": "awaiting_delivery", "outputs": outputs, "outcomes": outcomes}
        for index, work in enumerate(session.queue):
            permitted = session.tracks.tracks[work.track].capabilities
            tools = [
                *session.tracks.legal_tools(tuple(runtime.output_specs)),
                *(tool for tool in self.tools if tool["name"] in permitted),
            ]
            maximum = min(512, runtime.generator.generator.config.max_capacity)
            spec = replace(
                compile_tool_spec(json.dumps(tools), max_tokens=maximum), allow_text=True
            )
            options = tuple(spec if isinstance(p, ActionSpec) else p for p in work.modality_options)
            session.queue[index] = replace(
                work,
                spec=spec if isinstance(work.spec, ActionSpec) else work.spec,
                modality_options=options,
                control=build_control_payload(
                    seed=self.queue_controls.setdefault(work.event_id, work.control),
                    tools_json=json.dumps(tools),
                ),
            )
        revision = runtime.schedule_ready(session, effort=effort)
        if revision is None and not outputs and not self.runner.pending:
            if self.root is None:
                description = runtime.track_action(
                    session,
                    "create",
                    purpose="Decide and coordinate the next useful action for the user",
                    completion="The user's request has been answered or explicitly deferred",
                    capabilities=tuple(dict.fromkeys((*runtime.output_specs, *self.adapters))),
                    budget=self.max_selections,
                )
                self.root = description.address
            root = session.tracks.tracks[self.root]
            if root.status == "active" and root.budget:
                tools = [*session.tracks.legal_tools(tuple(runtime.output_specs)), *self.tools]
                maximum = min(512, runtime.generator.generator.config.max_capacity)
                spec = replace(
                    compile_tool_spec(json.dumps(tools), max_tokens=maximum), allow_text=True
                )
                controls = build_control_payload(
                    seed=b"Use only completed observations. Create work only when useful. Wait for needed results. Finish or cancel work explicitly.",
                    tools_json=json.dumps(tools),
                )
                revision = runtime.start_generation(
                    session,
                    tracks=(self.root,),
                    event_ids=(f"decision:{self.selections}",),
                    specs=(spec,),
                    controls=(controls,),
                    effort=effort,
                    generator=torch.Generator(device=session.state.device).manual_seed(
                        seed + self.selections
                    ),
                )
        if revision is not None:
            self.selections += 1
        return {
            "status": "selected"
            if revision is not None
            else "waiting"
            if self.runner.pending
            else "idle",
            "selection_revision": revision,
            "outputs": outputs,
            "outcomes": outcomes,
            "completed_tools": tuple(results),
            "failed_tools": tuple(failures),
        }
