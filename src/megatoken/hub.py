"""Hugging Face Hub access that waits out rate limits instead of failing.

Every cloud machine of a data build reads the Hub with one account's token, and the Hub
answers bursts with 429 (and overload with 503), naming a wait in ``Retry-After``. A build
that stops on the first refusal loses hours of work, so every Hub request made through
``huggingface_hub`` (and ``datasets``, which uses it) retries those after the named wait.
"""

from __future__ import annotations

import random
import time

import httpx

RETRY_STATUSES = frozenset({429, 503})
MAX_WAIT_SECONDS = 3600.0


def retry_after(response: httpx.Response, attempt: int) -> float:
    """The wait the server named, else exponential backoff; jittered so machines spread out."""
    try:
        named = float(response.headers.get("Retry-After", ""))
    except ValueError:
        named = min(2.0**attempt, 300.0)
    return named + random.uniform(0.0, 1.0 + named / 4)


class PatientTransport(httpx.BaseTransport):
    """Re-sends a request the Hub refused for load, until ``MAX_WAIT_SECONDS`` of waiting."""

    def __init__(self, inner: httpx.BaseTransport, max_wait: float = MAX_WAIT_SECONDS):
        self.inner = inner
        self.max_wait = max_wait

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        waited = 0.0
        attempt = 0
        while True:
            response = self.inner.handle_request(request)
            if response.status_code not in RETRY_STATUSES or waited >= self.max_wait:
                return response
            delay = min(retry_after(response, attempt), self.max_wait - waited)
            response.close()
            time.sleep(delay)
            waited += delay
            attempt += 1

    def close(self) -> None:
        self.inner.close()


def patient_client() -> httpx.Client:
    from huggingface_hub.utils._http import hf_request_event_hook

    return httpx.Client(
        transport=PatientTransport(httpx.HTTPTransport()),
        event_hooks={"request": [hf_request_event_hook]},
        follow_redirects=True,
        timeout=None,
    )


def install() -> None:
    """Route every later ``huggingface_hub`` request through :class:`PatientTransport`."""
    from huggingface_hub import set_client_factory

    set_client_factory(patient_client)
