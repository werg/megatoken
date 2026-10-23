"""Use explicitly approved smaller requests when cloud placement is exhausted."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from megatoken import cloud


def start_with_fallbacks(job: dict[str, Any], repo: Path, **kwargs):
    """Start the same workload using its declared resource alternatives.

    Only placement exhaustion or startup timeouts advance to the next request.
    Account quotas, authentication, uploads, and workload failures never trigger
    smaller allocations. The caller declares safe sizes for this particular job.
    ``cloud.start`` owns cleanup of a failed startup before another is attempted.
    """
    original = deepcopy(job)
    alternatives = original.pop("resource_fallbacks", [])
    gpu = original["resources"].get("gpu", 0)
    requests = [original["resources"], *alternatives]
    for resources in requests:
        if not isinstance(resources, dict) or not {"cpu", "memory"} <= resources.keys():
            raise cloud.CloudError("Resource alternatives need CPU and memory requests")
        if resources.get("gpu", 0) != gpu:
            raise cloud.CloudError("Resource alternatives must preserve the GPU requirement")
    for index, resources in enumerate(requests):
        candidate = {**original, "resources": deepcopy(resources)}
        try:
            return cloud.start(cloud.validate(candidate), repo, **kwargs)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}".lower()
            exhausted = "runner capacity exhausted" in reason
            startup_timeout = "sandboxtimeouterror" in reason and "did not become ready" in reason
            quota = "quota" in reason
            if quota or not (exhausted or startup_timeout) or index == len(requests) - 1:
                raise
            cloud.record(
                {
                    "event": "resource_fallback",
                    "job": original["id"],
                    "previous_resources": resources,
                    "next_resources": requests[index + 1],
                    "reason": "runner-capacity" if exhausted else "startup-timeout",
                }
            )
    raise AssertionError("unreachable")
