"""Paired jobs resolve moving branches once without masking lookup failures."""

from types import SimpleNamespace

import pytest

from scripts.resolve_producer_revision import resolve_producer_revision

REVISION = "a" * 40


def remote(replies):
    calls = []

    def run(args, **_options):
        ref = args[-1]
        calls.append(ref)
        result = replies[ref]
        return SimpleNamespace(
            returncode=result if isinstance(result, int) else 0,
            stdout="" if isinstance(result, int) else result + "\t" + ref + "\n",
        )

    return run, calls


def test_matching_branch_is_resolved_to_one_immutable_commit():
    run, calls = remote({"refs/heads/codex/coverage": REVISION})
    selected = resolve_producer_revision(
        environment={"GITHUB_HEAD_REF": "codex/coverage", "GITHUB_REF_NAME": "42/merge"},
        run=run,
    )
    assert selected == {
        "revision": REVISION,
        "ref": "refs/heads/codex/coverage",
        "selection": "matching-branch",
    }
    assert calls == ["refs/heads/codex/coverage"]


def test_only_a_missing_matching_branch_falls_back_to_current_default():
    run, calls = remote({"refs/heads/unpaired": 2, "HEAD": REVISION})
    assert resolve_producer_revision(environment={"GITHUB_REF_NAME": "unpaired"}, run=run) == {
        "revision": REVISION,
        "ref": "HEAD",
        "selection": "current-default",
    }
    assert calls == ["refs/heads/unpaired", "HEAD"]


@pytest.mark.parametrize("status", [1, 128])
def test_network_and_repository_failures_do_not_silently_change_producer(status):
    run, calls = remote({"refs/heads/paired": status, "HEAD": REVISION})
    with pytest.raises(ValueError, match="Could not resolve"):
        resolve_producer_revision(environment={"GITHUB_REF_NAME": "paired"}, run=run)
    assert calls == ["refs/heads/paired"]


def test_explicit_branch_or_commit_is_required_exactly():
    run, calls = remote({"refs/heads/selected": REVISION, "refs/heads/missing": 2})
    assert (
        resolve_producer_revision(environment={"TRACING_JS_SELF_REF": REVISION}, run=run)[
            "selection"
        ]
        == "explicit-revision"
    )
    assert calls == []
    selected = resolve_producer_revision(
        environment={"TRACING_JS_SELF_REF": "refs/heads/selected", "GITHUB_REF_NAME": "other"},
        run=run,
    )
    assert selected["revision"] == REVISION and selected["selection"] == "explicit-branch"
    with pytest.raises(ValueError, match="Could not resolve"):
        resolve_producer_revision(environment={"TRACING_JS_SELF_REF": "missing"}, run=run)
    assert calls == ["refs/heads/selected", "refs/heads/missing"]


@pytest.mark.parametrize("branch", ["bad\nrevision=changed", "../escape", "x..y", ""])
def test_invalid_or_empty_explicit_branch_cannot_inject_outputs(branch):
    if not branch:
        run, calls = remote({"HEAD": REVISION})
        assert (
            resolve_producer_revision(environment={"TRACING_JS_SELF_REF": branch}, run=run)[
                "selection"
            ]
            == "current-default"
        )
        assert calls == ["HEAD"]
        return
    with pytest.raises(ValueError, match="valid Git branch"):
        resolve_producer_revision(environment={"TRACING_JS_SELF_REF": branch})


@pytest.mark.parametrize("stdout", ["", "bad\tHEAD\n", REVISION + "\trefs/heads/other\n"])
def test_malformed_remote_identity_is_rejected(stdout):
    with pytest.raises(ValueError, match="one exact commit"):
        resolve_producer_revision(
            environment={},
            run=lambda *_args, **_options: SimpleNamespace(returncode=0, stdout=stdout),
        )
