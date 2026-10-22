import httpx
import pytest

pytest.importorskip("huggingface_hub")

from megatoken import hub


def test_a_rate_limited_request_waits_the_named_time_and_succeeds(monkeypatch):
    answers = iter(
        [
            httpx.Response(429, headers={"Retry-After": "13"}),
            httpx.Response(503),
            httpx.Response(200, text="ok"),
        ]
    )
    waits: list[float] = []
    monkeypatch.setattr(hub.time, "sleep", waits.append)
    client = httpx.Client(
        transport=hub.PatientTransport(httpx.MockTransport(lambda _: next(answers)))
    )
    assert client.get("https://huggingface.co/api/datasets/x").text == "ok"
    assert len(waits) == 2 and 13 <= waits[0] <= 13 + 1 + 13 / 4


def test_waiting_stops_at_the_limit_and_returns_the_refusal(monkeypatch):
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)
    transport = hub.PatientTransport(
        httpx.MockTransport(lambda _: httpx.Response(429, headers={"Retry-After": "40"})),
        max_wait=100,
    )
    assert httpx.Client(transport=transport).get("https://huggingface.co/").status_code == 429


def test_install_routes_hub_requests_through_the_patient_transport():
    from huggingface_hub import set_client_factory
    from huggingface_hub.utils._http import default_client_factory, get_session

    hub.install()
    try:
        assert isinstance(get_session()._transport, hub.PatientTransport)
    finally:
        set_client_factory(default_client_factory)
