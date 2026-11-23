#!/usr/bin/env python3
"""Resolve a current paired producer once, then check out its exact observed commit."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

PRODUCER_REPOSITORY = "https://github.com/werg/tracing-js-self.git"
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _branch_ref(value):
    branch = value.removeprefix("refs/heads/")
    ref = "refs/heads/" + branch
    checked = subprocess.run(
        ["git", "check-ref-format", ref], capture_output=True, text=True, check=False
    )
    if checked.returncode or not branch:
        raise ValueError("The paired producer branch is not a valid Git branch")
    return ref


def resolve_producer_revision(*, environment=None, run=subprocess.run):
    environment = os.environ if environment is None else environment

    def lookup(ref, *, allow_missing=False):
        result = run(
            ["git", "ls-remote", "--exit-code", PRODUCER_REPOSITORY, ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 2 and allow_missing:
            return None
        if result.returncode:
            raise ValueError("Could not resolve the paired producer reference: " + ref)
        matches = [line.split() for line in result.stdout.splitlines() if line.strip()]
        if (
            len(matches) != 1
            or len(matches[0]) != 2
            or matches[0][1] != ref
            or COMMIT.fullmatch(matches[0][0]) is None
        ):
            raise ValueError("The paired producer reference did not resolve to one exact commit")
        return matches[0][0]

    explicit = environment.get("TRACING_JS_SELF_REF", "").strip()
    if explicit:
        if COMMIT.fullmatch(explicit):
            # Checkout verifies the supplied historical commit exists. Never
            # reinterpret a failed exact checkout as permission to use HEAD.
            return {"revision": explicit, "ref": explicit, "selection": "explicit-revision"}
        ref = _branch_ref(explicit)
        return {"revision": lookup(ref), "ref": ref, "selection": "explicit-branch"}

    branch = environment.get("GITHUB_HEAD_REF") or environment.get("GITHUB_REF_NAME")
    if branch:
        ref = _branch_ref(branch)
        revision = lookup(ref, allow_missing=True)
        if revision is not None:
            return {"revision": revision, "ref": ref, "selection": "matching-branch"}
    return {"revision": lookup("HEAD"), "ref": "HEAD", "selection": "current-default"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path)
    options = parser.parse_args()
    selected = resolve_producer_revision()
    if options.github_output:
        with options.github_output.open("a") as output:
            for key, value in selected.items():
                output.write(key + "=" + value + "\n")
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    main()
