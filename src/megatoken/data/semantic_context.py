"""Causal bindings for public committed selected-state observations."""

import hashlib

from .host_operations import canonical_json
from .semantic_contract import DEFINITION
from .witness_v5 import parse_record, value


def realm_prefix(timeline, invocation_id):
    invocation = next(
        (
            item
            for item in timeline
            if item["kind"] == "invocation" and item["invocationId"] == invocation_id
        ),
        None,
    )
    if invocation is None:
        raise ValueError("Selected-state observation requires its already admitted invocation")
    realm = invocation["realmId"]
    calls = [item for item in timeline if item["kind"] == "invocation" and item["realmId"] == realm]
    identities, programs, tracks = (
        {item[key] for item in calls} for key in ("invocationId", "programId", "track")
    )
    subscriptions = {
        item["subscriptionId"]
        for item in timeline
        if item["kind"] == "handler" and item["realmId"] == realm
    }
    return [
        item
        for item in timeline
        if (
            item["kind"] == "program"
            and item.get("programId") in programs
            or item.get("invocationId") in identities
            or item.get("subscriptionId") in subscriptions
            or item["kind"] in {"source-read", "source-evict"}
            and item["track"] in tracks
            and item["programId"] in programs
        )
    ]


def committed_random_state(timeline, invocation_id):
    """Replay the realm's public seeded draws through its exact observed prefix."""
    prefix = realm_prefix(timeline, invocation_id)
    seeds = {event.get("randomSeed") for event in prefix if event["kind"] == "program"}
    if len(seeds) != 1:
        raise ValueError("Retained realm programs changed their deterministic RNG seed")
    state = next(iter(seeds))
    for event in prefix:
        if event["kind"] != "semantic":
            continue
        kind, fields = parse_record(event["line"])
        if kind != "compute" or fields["expr"] != {"op": "random"}:
            continue
        if type(state) is not int or not 0 <= state <= 0xFFFFFFFF:
            raise ValueError("Random draw has no observed deterministic seed")
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        if fields["result"] != ["num", state / 4294967296]:
            raise ValueError("Random draw differs from its observed deterministic state")
    return state


def validate_selected_context(item, prior):
    """Validate exact public ancestry; the independent source replay proves values."""
    fields = {"selectedState", "realmPrefixDigest", "visibleThrough"}
    if not fields & set(item):
        return False
    if not fields <= set(item) or item["visibleThrough"] != item["sequence"] - 1:
        raise ValueError("Selected-state observation changed its causal cutoff")
    state = item["selectedState"]
    declaration = DEFINITION["modelContext"]
    if (
        not isinstance(state, dict)
        or set(state) != set(declaration["stateFields"])
        or state["schema"] != declaration["stateSchema"]
    ):
        raise ValueError("Selected-state observation requires the exact declared continuation")
    for key in ("heap", "locals", "captureAliases", "stack", "continuations", "ready"):
        if not isinstance(state[key], list):
            raise ValueError("Selected-state continuation contains malformed public state")
    for key in ("heap", "locals", "captureAliases", "continuations"):
        rows = state[key]
        if any(
            not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
            for row in rows
        ) or len({row[0] for row in rows}) != len(rows):
            raise ValueError("Selected-state table contains malformed or duplicate identities")
    random_state = state.get("randomState")
    if random_state is not None and (
        type(random_state) is not int or not 0 <= random_state <= 0xFFFFFFFF
    ):
        raise ValueError("Selected-state deterministic RNG value must be null or uint32")
    for _, selected in state["locals"]:
        value(selected)
    if any(not isinstance(identity, str) for identity in (*state["stack"], *state["ready"])):
        raise ValueError("Selected-state scheduling identity is malformed")
    prefix = realm_prefix(prior, item["invocationId"])
    if not prior or prior[-1]["sequence"] != item["visibleThrough"]:
        raise ValueError("Selected-state observation refers to future or missing public evidence")
    invocation = next(
        row
        for row in prefix
        if row["kind"] == "invocation" and row["invocationId"] == item["invocationId"]
    )
    reread = prior[-1]
    if (
        reread["kind"] != "source-read"
        or reread["track"] != invocation["track"]
        or reread["programId"] != invocation["programId"]
        or reread["digest"] != invocation["sourceDigest"]
    ):
        raise ValueError(
            "Selected-state observation requires its immediately preceding source reread"
        )
    if hashlib.sha256(canonical_json(prefix).encode()).hexdigest() != item["realmPrefixDigest"]:
        raise ValueError("Selected-state observation changed its committed realm prefix")
    if state["randomState"] != committed_random_state(prior, item["invocationId"]):
        raise ValueError("Selected-state random value differs from the committed realm draws")
    return True
