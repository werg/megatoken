#!/usr/bin/env python3
"""Restore the seed prompts NVIDIA withheld from Nemotron chat v3's chat split.

Rows seeded from LMSYS-Chat-1M or WildChat-1M ship with their first user turn
(and a leading system turn) set to null, plus the SHA-256 of the withheld
prompt. The publisher's own script matches only each source conversation's
first user message and aborts if any prompt is missing. This one matches every
user message of every source conversation, restores what it finds, and drops
the rows it cannot restore (counted in the summary), so one gap does not cost
the whole split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    "lmsys": ("lmsys/lmsys-chat-1m", "200748d9d3cddcc9d782887541057aca0b18c5da"),
    "wildchat": ("allenai/WildChat-1M", "7d6490e462285cf85d91eabea0f9a954fbddcd1f"),
}


def family_of(metadata: dict) -> str | None:
    for value in (metadata.get("seed_dataset"), metadata.get("seed_source")):
        if isinstance(value, str):
            name = value.lower()
            for family, (repository, _) in SOURCES.items():
                if name == repository.lower() or name.startswith(family):
                    return family
    return None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def needed_hashes(path: Path) -> dict[str, set[str]]:
    needed: dict[str, set[str]] = {family: set() for family in SOURCES}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            metadata = json.loads(line).get("metadata") or {}
            family = family_of(metadata)
            if family and metadata.get("seed_prompt_sha256"):
                needed[family].add(metadata["seed_prompt_sha256"])
    return needed


def find_prompts(needed: dict[str, set[str]], token: str) -> dict[str, tuple[str | None, str]]:
    """SHA-256 -> (preceding system turn, user turn) over every user turn of every source row."""
    from datasets import load_dataset

    found: dict[str, tuple[str | None, str]] = {}
    for family, hashes in needed.items():
        if not hashes:
            continue
        repository, revision = SOURCES[family]
        remaining = set(hashes)
        rows = load_dataset(
            repository, split="train", revision=revision, streaming=True, token=token
        )
        for row in rows:
            system = None
            for message in row.get("conversation") or ():
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    continue
                if message.get("role") == "system" and system is None:
                    system = content
                elif message.get("role") == "user":
                    key = digest(content)
                    if key in remaining:
                        found[key] = (system, content)
                        remaining.discard(key)
            if not remaining:
                break
        print(json.dumps({family: {"needed": len(hashes), "missing": len(remaining)}}), flush=True)
    return found


def restore(source: Path, output: Path, found: dict[str, tuple[str | None, str]]) -> dict:
    counts = {"rows": 0, "restored": 0, "dropped": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, output.open("w", encoding="utf-8") as writer:
        for line in reader:
            row = json.loads(line)
            counts["rows"] += 1
            metadata = row.get("metadata") or {}
            key = metadata.get("seed_prompt_sha256")
            if family_of(metadata) and key:
                if key not in found:
                    counts["dropped"] += 1
                    continue
                system, user = found[key]
                messages = row["messages"]
                if messages and messages[0].get("role") == "system":
                    if system is None:
                        # The source had no system turn: drop the empty one and its mask entry.
                        messages.pop(0)
                        turns = metadata.get("train_turns")
                        if isinstance(turns, list) and len(turns) == len(messages) + 1:
                            turns.pop(0)
                    else:
                        messages[0]["content"] = system
                user_index = next(i for i, m in enumerate(messages) if m.get("role") == "user")
                messages[user_index]["content"] = user
                counts["restored"] += 1
            writer.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="The released data/chat.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--token", required=True, help="Hugging Face token with both gates accepted"
    )
    args = parser.parse_args()
    found = find_prompts(needed_hashes(args.input), args.token)
    print(json.dumps(restore(args.input, args.output, found)), flush=True)


if __name__ == "__main__":
    main()
