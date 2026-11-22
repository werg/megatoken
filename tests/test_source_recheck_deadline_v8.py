"""A larger declared engineering budget never weakens exact source authority."""

import subprocess
from unittest.mock import patch

import pytest

from scripts import admit_semantic_v8


@pytest.mark.parametrize("value", [0, -1, 901, True, 1.5, "600", None])
def test_source_recheck_budget_is_an_explicit_finite_integer(value, tmp_path):
    with pytest.raises(ValueError, match="integer from 1 through 900"):
        admit_semantic_v8.recheck_source_binding(
            {}, tmp_path / "input", None, None, timeout_seconds=value
        )


def checkout_answers(root, revision):
    return [revision, str(root), "", "tools/recheck-semantic-native-source.mjs"] * 2


@pytest.mark.parametrize("budget", [30, 600, 900])
def test_deadline_is_passed_to_the_complete_source_process_and_timeout_refuses(budget, tmp_path):
    revision = "a" * 40
    with (
        patch.object(subprocess, "check_output", side_effect=checkout_answers(tmp_path, revision)),
        patch.object(
            subprocess, "run", side_effect=subprocess.TimeoutExpired("source", budget)
        ) as run,
    ):
        with pytest.raises(ValueError, match="declared time bound"):
            admit_semantic_v8.recheck_source_binding(
                {}, tmp_path / "input", tmp_path, revision, timeout_seconds=budget
            )
        assert run.call_args.kwargs["timeout"] == budget


@pytest.mark.parametrize("stdout,stderr", [("x" * 65537, ""), ("{}", "warning")])
def test_larger_budget_still_refuses_unbounded_or_unclean_output(stdout, stderr, tmp_path):
    revision = "a" * 40
    completed = subprocess.CompletedProcess([], 0, stdout=stdout, stderr=stderr)
    with (
        patch.object(subprocess, "check_output", side_effect=checkout_answers(tmp_path, revision)),
        patch.object(subprocess, "run", return_value=completed),
    ):
        with pytest.raises(ValueError, match="bounded and clean"):
            admit_semantic_v8.recheck_source_binding(
                {}, tmp_path / "input", tmp_path, revision, timeout_seconds=600
            )


def test_existing_callers_keep_the_thirty_second_default(tmp_path):
    revision = "a" * 40
    with (
        patch.object(subprocess, "check_output", side_effect=checkout_answers(tmp_path, revision)),
        patch.object(
            subprocess, "run", return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="")
        ) as run,
    ):
        with pytest.raises(ValueError, match="rejected semantic source correspondence"):
            admit_semantic_v8.recheck_source_binding({}, tmp_path / "input", tmp_path, revision)
        assert run.call_args.kwargs["timeout"] == 30
