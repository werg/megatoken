from copy import deepcopy
from pathlib import Path

import pytest

from megatoken import cloud, cloud_capacity


@pytest.fixture
def job():
    return {
        "schema": cloud.SCHEMA,
        "id": "test-capacity",
        "image": "test-image",
        "resources": {"cpu": "4", "memory": "16Gi", "gpu": 1},
        "resource_fallbacks": [{"cpu": "2", "memory": "16Gi", "gpu": 1}],
        "lifetime_hours": 1,
        "command": ["run-workload"],
        "outputs": ["out"],
    }


@pytest.mark.parametrize(
    "message",
    [
        "runner capacity exhausted",
        "SandboxTimeoutError: Sandbox example did not become ready within 900s",
    ],
)
def test_placement_fallback_preserves_workload_and_inputs(monkeypatch, job, message):
    before = deepcopy(job)
    calls = []

    def start(candidate, repo, **kwargs):
        calls.append((candidate, repo, kwargs))
        if len(calls) == 1:
            raise RuntimeError(message)
        return "started", "sandbox"

    monkeypatch.setattr(cloud, "start", start)
    monkeypatch.setattr(cloud, "record", lambda event: None)
    snapshot = (Path("archive"), {"commit": "fixed"})
    assert cloud_capacity.start_with_fallbacks(job, Path("repo"), snapshot=snapshot) == (
        "started",
        "sandbox",
    )
    assert [call[0]["resources"]["cpu"] for call in calls] == ["4", "2"]
    assert all(call[0]["command"] == before["command"] for call in calls)
    assert all(call[2]["snapshot"] == snapshot for call in calls)
    assert job == before


@pytest.mark.parametrize(
    "message",
    [
        "concurrent sandbox quota exceeded",
        "runner capacity exhausted: account quota exceeded",
        "owning gateway could not complete CreateSandbox",
        "authentication failed",
        "upload checksum mismatch",
    ],
)
def test_other_failures_do_not_change_resource_request(monkeypatch, job, message):
    calls = []

    def start(*args, **kwargs):
        calls.append(args)
        raise RuntimeError(message)

    monkeypatch.setattr(cloud, "start", start)
    with pytest.raises(RuntimeError, match=message):
        cloud_capacity.start_with_fallbacks(job, Path("repo"))
    assert len(calls) == 1


def test_fallback_never_silently_drops_gpu(monkeypatch, job):
    job["resource_fallbacks"][0]["gpu"] = 0
    monkeypatch.setattr(cloud, "start", lambda *args, **kwargs: pytest.fail("invalid allocation"))
    with pytest.raises(cloud.CloudError, match="preserve the GPU"):
        cloud_capacity.start_with_fallbacks(job, Path("repo"))
