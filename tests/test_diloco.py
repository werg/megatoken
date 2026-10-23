"""Crash, convergence, sharding and mixed-optimizer tests for Decoupled DiLoCo."""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path

import pytest
import torch
from torch import nn

from megatoken.data import PreparedCorpus, WindowConfig, prepare_sources, write_demo_source
from megatoken.generation.codec import ByteCodec
from megatoken.model.config import ModelConfig
from megatoken.training.diloco_protocol import (
    install_fragment,
    load_tensors,
    merge_pseudogradients,
    outer_nesterov,
    pack_fragment,
    parameter_identity,
    plan_fragments,
    save_tensors,
)
from megatoken.training.diloco_server import Conflict, Coordinator, initialize_coordinator
from megatoken.training.optimizer import MasterAdamW
from megatoken.training.stream import IndependentWindowStream

TRANSDUCER = {
    "geometry": dict(vars(ModelConfig.tiny().transducer_geometry)),
    "evidence_from": 0,
}


def _coordinator(tmp_path, *, workers=3, quorum=2, max_staleness=0):
    baseline = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    model = {"linear.weight": baseline.clone()}
    manifest = initialize_coordinator(
        tmp_path / "syncer",
        run_id="unit-run",
        model_state=model,
        masters=model,
        model_config={"test": True},
        codec_fingerprint="codec",
        train_identity="objective",
        workers=workers,
        min_quorum=quorum,
        fragment_bytes=1024,
        outer_lr=1,
        outer_momentum=0,
        grace_seconds=0,
        max_staleness=max_staleness,
        transducer=TRANSDUCER,
    )
    return Coordinator(tmp_path / "syncer"), baseline, manifest


def _submit(coordinator, tmp_path, *, worker, session, delta, sequence=1, version=0):
    path = tmp_path / f"delta-{worker}.safetensors"
    save_tensors(path, {"delta": torch.full((64,), delta, dtype=torch.float32)})
    return coordinator.submit(
        0,
        worker,
        session,
        sequence,
        version,
        steps=10,
        tokens=100,
        incoming=path,
    )


def test_fragment_plan_handles_large_packed_experts_and_exact_reinstallation():
    model = nn.Module()
    model.packed = nn.Parameter(torch.arange(400, dtype=torch.float32).reshape(10, 10, 4))
    model.bias = nn.Parameter(torch.linspace(0, 1, 11))
    optimizer = MasterAdamW(model, lr=0.01, inner_optimizer="adamw")
    try:
        fragments = plan_fragments(optimizer.masters, target_bytes=1024)
        assert len(fragments) > 1
        snapshot = {name: p.detach().clone() for name, p in optimizer.masters.items()}
        for fragment in fragments:
            original = pack_fragment(optimizer.masters, fragment)
            install_fragment(optimizer, fragment, original + 7)
        for name, value in snapshot.items():
            torch.testing.assert_close(optimizer.masters[name], value + 7, rtol=0, atol=0)
            torch.testing.assert_close(model.state_dict()[name], value + 7, rtol=0, atol=0)
        assert parameter_identity(snapshot) == parameter_identity(optimizer.masters)
    finally:
        optimizer.close()


def test_quorum_merge_is_atomic_and_crash_idempotent(tmp_path):
    coordinator, initial, manifest = _coordinator(tmp_path)
    sessions = [str(uuid.uuid4()) for _ in range(3)]
    for i in range(3):
        coordinator.lease(i, sessions[i])
    assert (
        _submit(coordinator, tmp_path, worker=0, session=sessions[0], delta=1)["status"] == "queued"
    )
    assert coordinator.status()["versions"] == [0]
    assert _submit(coordinator, tmp_path, worker=1, session=sessions[1], delta=3)["version"] == 1
    state = load_tensors(coordinator._path(0))
    torch.testing.assert_close(state["theta"], initial.reshape(-1) - 2, rtol=0, atol=0)
    assert coordinator.status()["merges"] == 1
    # Simulate a crash after atomic publication but before pending cleanup.
    queued = coordinator._pending_dir(0) / "000.safetensors"
    save_tensors(
        queued,
        {"delta": torch.ones(64)},
        metadata={
            "worker": "0",
            "sequence": "1",
            "base_version": "0",
            "steps": "10",
            "tokens": "100",
        },
    )
    restarted = Coordinator(tmp_path / "syncer")
    assert not queued.exists()
    assert restarted.status()["versions"] == [1]
    assert (
        _submit(restarted, tmp_path, worker=0, session=sessions[0], delta=1)["status"]
        == "already_merged"
    )
    with pytest.raises(Conflict, match="Stale"):
        _submit(restarted, tmp_path, worker=2, session=sessions[2], delta=1, version=0)
    assert manifest["min_quorum"] == 2


def test_weighting_and_outer_momentum_use_finite_float32():
    delta = merge_pseudogradients(
        [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])],
        tokens=[10, 20],
        steps=[10, 10],
        staleness=[0, 0],
    )
    expected = (torch.tensor([1.0, 2.0]) + 4 * torch.tensor([3.0, 4.0])) / 5
    torch.testing.assert_close(delta, expected)
    theta, momentum = outer_nesterov(
        torch.tensor([10.0, 10.0]),
        torch.zeros(2),
        delta,
        lr=0.5,
        momentum=0.8,
    )
    torch.testing.assert_close(momentum, delta)
    torch.testing.assert_close(theta, 10 - 0.9 * delta)
    with pytest.raises(ValueError, match="Invalid outer gradient"):
        merge_pseudogradients([torch.tensor([float("nan")])], [1], [1], [0])


def test_hybrid_muon_only_selects_eligible_hidden_matrices_and_resumes():
    torch.manual_seed(23)
    model = nn.ModuleDict(
        {
            "hidden": nn.Linear(8, 8),
            "embed": nn.Embedding(16, 8),
            "layernorm": nn.LayerNorm(8),
        }
    ).bfloat16()
    optimizer = MasterAdamW(model, lr=0.003, inner_optimizer="muon")
    assert optimizer.muon_names == ["hidden.weight"]
    assert optimizer.optimizer is not None
    assert optimizer.muon_optimizer is not None
    try:
        for master in optimizer.masters.values():
            master.grad = torch.randn_like(master) * 0.03
        optimizer.step()
        state = copy.deepcopy(optimizer.state_dict())
        replica = copy.deepcopy(model)
        recovered = MasterAdamW(replica, lr=0.003, inner_optimizer="muon")
        try:
            recovered.load_state_dict(state)
            for key in optimizer.masters:
                gradient = torch.randn_like(optimizer.masters[key])
                optimizer.masters[key].grad = gradient.clone()
                recovered.masters[key].grad = gradient.clone()
            optimizer.step()
            recovered.step()
            for key in optimizer.masters:
                torch.testing.assert_close(
                    optimizer.masters[key], recovered.masters[key], rtol=0, atol=0
                )
        finally:
            recovered.close()
    finally:
        optimizer.close()


def test_independent_shard_cursor_is_restartable_without_collectives(tmp_path):
    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "source.jsonl", count=24, max_record_bytes=128)
    prepared = prepare_sources([source], tmp_path / "data", codec)
    prepared.close()
    ds0 = PreparedCorpus([tmp_path / "data"])
    ds1 = PreparedCorpus([tmp_path / "data"])
    ds2 = PreparedCorpus([tmp_path / "data"])
    config = WindowConfig(max_window_tokens=512, max_records=2)
    a = IndependentWindowStream(ds0, codec, config, seed=31, rank=0, world_size=2)
    b = IndependentWindowStream(ds1, codec, config, seed=31, rank=1, world_size=2)
    clone = IndependentWindowStream(ds2, codec, config, seed=31, rank=0, world_size=2)
    try:
        first = a.next()
        other = b.next()
        assert first.episode.episode_id != other.episode.episode_id
        state = a.state_dict()
        clone.load_state_dict(state)
        for _ in range(3):
            left, right = a.next(), clone.next()
            assert left.episode.episode_id == right.episode.episode_id
            assert left.window.start == right.window.start
    finally:
        a.close()
        b.close()
        clone.close()


def test_coordinator_strict_init_never_replaces_committed_run(tmp_path):
    coordinator, initial, _ = _coordinator(tmp_path, workers=1, quorum=1)
    with pytest.raises(FileExistsError):
        initialize_coordinator(
            tmp_path / "syncer",
            run_id="unit-run",
            model_state={"linear.weight": initial},
            masters={"linear.weight": initial},
            model_config={},
            codec_fingerprint="codec",
            train_identity="objective",
            workers=1,
            min_quorum=1,
            transducer=TRANSDUCER,
        )
    session = str(uuid.uuid4())
    coordinator.lease(0, session)
    status = coordinator.heartbeat(0, session, steps=4, tokens=100)
    assert status["global_tokens"] == 100
    # Recover an older checkpoint; server never double-counts replayed windows.
    assert coordinator.heartbeat(0, session, steps=2, tokens=50)["global_tokens"] == 100
    assert coordinator.heartbeat(0, session, steps=5, tokens=120)["global_tokens"] == 120


def test_global_install_carries_local_progress_made_while_the_round_is_in_flight():
    from megatoken.training.diloco_client import DecoupledWorker

    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(6))
    optimizer = MasterAdamW(model, lr=0.01, inner_optimizer="adamw")
    try:
        (fragment,) = plan_fragments(optimizer.masters, target_bytes=1 << 20)
        worker = DecoupledWorker.__new__(DecoupledWorker)
        worker.fragments = (fragment,)
        worker.versions = [3]
        worker.awaiting_global = {0}
        # Anchor = the upload point; two more local steps then moved the weights.
        upload_point = torch.full((6,), 1.0)
        worker.baselines = [upload_point.clone()]
        install_fragment(optimizer, fragment, upload_point + 0.25)
        merged = torch.full((6,), 5.0)
        assert worker._install(0, (4, merged), optimizer)
        torch.testing.assert_close(pack_fragment(optimizer.masters, fragment), merged + 0.25)
        torch.testing.assert_close(model.weight.detach(), merged + 0.25)
        # The next pseudo-gradient is exactly the residual, against the new global.
        torch.testing.assert_close(
            worker.baselines[0] - pack_fragment(optimizer.masters, fragment),
            torch.full((6,), -0.25),
        )
        assert worker.versions == [4] and not worker.awaiting_global
        assert not worker._install(0, (4, merged), optimizer)
    finally:
        optimizer.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned fragment packing needs CUDA")
def test_device_fragment_packing_matches_host_values():
    masters = {
        "a": torch.arange(12, dtype=torch.bfloat16, device="cuda").reshape(3, 4),
        "b": torch.linspace(-1, 1, 7, device="cuda"),
    }
    (fragment,) = plan_fragments(masters, target_bytes=1 << 20)
    packed = pack_fragment(masters, fragment)
    assert packed.dtype == torch.float32 and packed.is_pinned()
    expected = torch.cat(
        [
            masters[p.name].float().reshape(-1)[p.offset : p.offset + p.length].cpu()
            for p in fragment.slices
        ]
    )
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)


def test_transport_reads_headers_case_insensitively_like_http2_gateways(tmp_path):
    """An HTTPS gateway (HTTP/2) delivers lowercase header names."""
    import http.server
    import threading

    from megatoken.training.diloco_client import Transport

    class Gateway(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("x-fragment-version", "7")
            self.send_header("content-length", "3")
            self.end_headers()
            self.wfile.write(b"abc")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        transport = Transport(f"http://127.0.0.1:{server.server_address[1]}", "t" * 32, 10)
        status, path, headers = transport.request("GET", "/v1/fragments/0", output=tmp_path / "f")
    finally:
        server.shutdown()
    assert status == 200 and path.read_bytes() == b"abc"
    assert int(headers["X-Fragment-Version"]) == 7


# ------------------------------------------------------------ leases, ledger, final merge


def test_lease_length_is_a_manifest_setting_and_an_expired_lease_is_told_apart(tmp_path):
    coordinator, _, manifest = _coordinator(tmp_path)
    assert manifest["lease_seconds"] == 600.0
    session = str(uuid.uuid4())
    coordinator.lease(0, session)
    assert coordinator.progress["leases"]["0"]["expires"] > time.time() + 500
    coordinator.progress["leases"]["0"]["expires"] = time.time() - 1
    with pytest.raises(Conflict) as expired:
        coordinator.heartbeat(0, session, steps=1, tokens=1)
    assert expired.value.detail == {"lease": True}
    # The same session can lease again; a different live session cannot take the slot.
    coordinator.lease(0, session)
    coordinator.heartbeat(0, session, steps=1, tokens=1)
    with pytest.raises(Conflict) as taken:
        coordinator.lease(0, str(uuid.uuid4()))
    assert taken.value.detail == {}


def test_lease_reports_what_the_coordinator_holds_from_the_worker(tmp_path):
    coordinator, _, _ = _coordinator(tmp_path, workers=2, quorum=2)
    sessions = [str(uuid.uuid4()) for _ in range(2)]
    first = coordinator.lease(0, sessions[0])
    assert first["sequences"] == {"fragments": [0], "transducer": 0}
    coordinator.lease(1, sessions[1])
    _submit(coordinator, tmp_path, worker=0, session=sessions[0], delta=1, sequence=1)
    _submit(coordinator, tmp_path, worker=1, session=sessions[1], delta=1, sequence=1)  # merges
    _submit(coordinator, tmp_path, worker=0, session=sessions[0], delta=1, sequence=2, version=1)
    # Sequence 1 is merged and 2 only queued: both count, so a restored learner numbers past 2.
    assert coordinator.lease(0, sessions[0])["sequences"]["fragments"] == [2]
    assert coordinator.lease(1, sessions[1])["sequences"]["fragments"] == [1]


def test_closing_merges_a_final_round_even_below_quorum(tmp_path):
    coordinator, initial, _ = _coordinator(tmp_path, workers=3, quorum=3)
    session = str(uuid.uuid4())
    coordinator.lease(0, session)
    assert _submit(coordinator, tmp_path, worker=0, session=session, delta=2)["status"] == "queued"
    coordinator.tick()
    assert coordinator.status()["versions"] == [0]  # one of three: no quorum yet
    coordinator.close()
    assert coordinator.status()["versions"] == [1]
    torch.testing.assert_close(
        load_tensors(coordinator._path(0))["theta"], initial.reshape(-1) - 2, rtol=0, atol=0
    )


# ---------------------------------------------------------------- a real server


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A coordinator on a local port (short lease), served from a thread."""
    import threading

    from megatoken.training import diloco_server

    token = "t" * 40
    servers = []

    class Capturing(diloco_server.ThreadingHTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            servers.append(self)

    monkeypatch.setattr(diloco_server, "ThreadingHTTPServer", Capturing)
    started = []

    def start(*, workers=1, quorum=1, lease_seconds=1.0, token_budget=0):
        baseline = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        initialize_coordinator(
            tmp_path / "served",
            run_id="unit-run",
            model_state={"linear.weight": baseline.clone()},
            masters={"linear.weight": baseline.clone()},
            model_config={"test": True},
            codec_fingerprint="codec",
            train_identity="objective",
            workers=workers,
            min_quorum=quorum,
            fragment_bytes=1024,
            outer_lr=1,
            outer_momentum=0,
            grace_seconds=0,
            lease_seconds=lease_seconds,
            token_budget=token_budget,
            transducer=TRANSDUCER,
        )
        thread = threading.Thread(
            target=diloco_server.serve_coordinator,
            args=(tmp_path / "served", "127.0.0.1", 0, token),
            daemon=True,
        )
        thread.start()
        while not servers:
            time.sleep(0.01)
        started.append(thread)
        return f"http://127.0.0.1:{servers[0].server_address[1]}", token

    yield start
    for server in servers:
        server.shutdown()
    for thread in started:
        thread.join(timeout=10)


def _raw(url, token, method, path, *, body=None, headers=None):
    import http.client
    from urllib.parse import urlsplit

    connection = http.client.HTTPConnection(urlsplit(url).netloc, timeout=10)
    try:
        connection.request(
            method,
            path,
            body=body,
            headers={"Authorization": f"Bearer {token}", **(headers or {})},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def test_the_server_answers_400_to_malformed_requests_and_401_to_odd_credentials(served):
    url, token = served()
    session = str(uuid.uuid4())
    body = json.dumps({"worker": 0, "session": session}).encode()
    assert _raw(url, token, "POST", "/v1/lease", body=body)[0] == 200
    round_headers = {
        "X-Worker": "0",
        "X-Session": session,
        "X-Sequence": "1",
        "X-Base-Version": "0",
    }
    update_headers = {**round_headers, "X-Steps": "1", "X-Tokens": "1"}
    # A missing header is a client error, not a dropped connection.
    for path in ("/v1/updates/0", "/v1/transducer/rounds"):
        status, _ = _raw(url, token, "POST", path, body=b"x", headers={"X-Worker": "0"})
        assert status == 400
    # Bytes that are not safetensors / a torch archive.
    assert (
        _raw(url, token, "POST", "/v1/updates/0", body=b"not safetensors", headers=update_headers)[
            0
        ]
        == 400
    )
    assert (
        _raw(url, token, "POST", "/v1/transducer/rounds", body=b"junk", headers=round_headers)[0]
        == 400
    )
    # A non-ASCII credential is refused instead of crashing the comparison.
    status, _ = _raw(url, "é" * 40, "GET", "/v1/status")
    assert status == 401
    assert _raw(url, token, "GET", "/v1/status")[0] == 200


def test_a_worker_keeps_its_lease_on_the_wall_clock_and_leases_again_after_losing_it(
    served, tmp_path, monkeypatch
):
    from megatoken.training.config import DecoupledConfig
    from megatoken.training.diloco_client import DecoupledWorker

    url, token = served(lease_seconds=1.0)
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    settings = DecoupledConfig(
        coordinator_url=url, run_id="unit-run", workers=1, worker_id=0, timeout_seconds=10
    )
    masters = {"linear.weight": torch.zeros(8, 8)}
    worker = DecoupledWorker(
        settings,
        model_config={"test": True},
        codec_fingerprint="codec",
        masters=masters,
        output=tmp_path / "worker",
        token_budget=0,
        train_identity="objective",
        dataset_fingerprint="",
    )
    try:
        assert worker.heartbeat_seconds == 0.25  # a quarter of the manifest's lease
        worker._lease()
        worker._start_heartbeat()
        worker.local_steps, worker.local_tokens = 3, 90
        time.sleep(2.5)  # more than two leases, with no training step in between
        status = worker.transport.json("GET", "/v1/status")
        assert status["live_workers"] == 1 and status["global_tokens"] == 90
        # Lose the lease (heartbeat stopped, lease lapsed): an upload leases again and lands.
        worker.stopping.set()
        worker.heartbeat_thread.join()
        time.sleep(1.2)
        assert worker.transport.json("GET", "/v1/status")["live_workers"] == 0
        entry = {
            "filename": str(tmp_path / "worker" / "outbox" / "u.safetensors"),
            "sequence": 1,
            "version": 0,
            "steps": 1,
            "tokens": 10,
            "delta": torch.ones(64),
        }
        renewals = []
        original_lease = worker._lease

        def renewed_lease():
            result = original_lease()
            renewals.append(result)
            return result

        monkeypatch.setattr(worker, "_lease", renewed_lease)
        assert worker._upload(0, entry)["status"] == "queued"
        assert renewals
        # A queued authenticated update plus a new lease proves recovery.
        # Heartbeats remain deliberately stopped: another HTTP round trip is
        # allowed to outlive this test's one-second lease under contention.
    finally:
        worker.close()


def test_a_restored_worker_numbers_new_uploads_past_the_coordinators_ledger(
    served, tmp_path, monkeypatch
):
    from megatoken.training.config import DecoupledConfig
    from megatoken.training.diloco_client import DecoupledWorker

    url, token = served(lease_seconds=30.0)
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    settings = DecoupledConfig(coordinator_url=url, run_id="unit-run", workers=1, worker_id=0)
    worker = DecoupledWorker(
        settings,
        model_config={"test": True},
        codec_fingerprint="codec",
        masters={"linear.weight": torch.zeros(8, 8)},
        output=tmp_path / "worker",
        token_budget=0,
        train_identity="objective",
        dataset_fingerprint="",
    )
    try:
        for sequence in (1, 2, 3):  # an earlier life of this learner got three rounds merged
            entry = {
                "filename": str(tmp_path / f"{sequence}.safetensors"),
                "sequence": sequence,
                "version": sequence - 1,
                "steps": 1,
                "tokens": 10,
                "delta": torch.ones(64) * 0.001,
            }
            worker._lease()
            worker._upload(0, entry)
        # Restored from a checkpoint that only knew sequence 1.
        worker.sequences = [1]
        worker._adopt_ledger(worker._lease())
        assert worker.sequences == [3]
    finally:
        worker.close()


def test_finish_hands_over_the_last_partial_window(served, tmp_path, monkeypatch):
    from megatoken.training.config import DecoupledConfig
    from megatoken.training.diloco_client import DecoupledWorker

    url, token = served(lease_seconds=30.0)
    monkeypatch.setenv("MEGATOKEN_DILOCO_TOKEN", token)
    # The learner's round length (256 steps) is far longer than this run: only the final
    # flush can ever send its progress.
    settings = DecoupledConfig(
        coordinator_url=url, run_id="unit-run", workers=1, worker_id=0, timeout_seconds=10
    )
    model = nn.Module()
    model.linear = nn.Linear(8, 8, bias=False)
    optimizer = MasterAdamW(model, lr=0.01, inner_optimizer="adamw")
    worker = DecoupledWorker(
        settings,
        model_config={"test": True},
        codec_fingerprint="codec",
        masters=optimizer.masters,
        output=tmp_path / "worker",
        token_budget=0,
        train_identity="objective",
        dataset_fingerprint="",
    )
    try:
        worker.bootstrap(model, optimizer)
        start = optimizer.masters["linear.weight"].detach().clone().reshape(-1)
        for step in range(1, 4):
            with torch.no_grad():
                optimizer.masters["linear.weight"].add_(0.5)
            worker.after_step(optimizer, steps=step, tokens=10)
        assert worker.transport.json("GET", "/v1/status")["merges"] == 0
        assert worker.finish(optimizer)
        deadline = time.time() + 10
        while worker.transport.json("GET", "/v1/status")["merges"] < 1:
            assert time.time() < deadline
            time.sleep(0.05)
        _, response, headers = worker.transport.request(
            "GET", "/v1/fragments/0", output=tmp_path / "global.safetensors"
        )
        merged = load_tensors(response)["theta"]
        # One worker, outer lr 1, no momentum: the global is the learner's own weights.
        torch.testing.assert_close(merged, start + 1.5, rtol=0, atol=1e-2)
    finally:
        worker.close()
        optimizer.close()


def test_fully_frozen_fragments_are_never_sent():
    from megatoken.training.diloco_client import DecoupledWorker

    model = nn.Module()
    model.frozen = nn.Parameter(torch.zeros(4))
    model.live = nn.Parameter(torch.zeros(4))
    optimizer = MasterAdamW(model, lr=0.01, inner_optimizer="adamw")
    try:
        # One fragment per parameter.
        fragments = plan_fragments(optimizer.masters, target_bytes=1024)
        small = plan_fragments({"frozen": optimizer.masters["frozen"]}, target_bytes=1024)
        assert len(fragments) == 1 and len(small) == 1
        optimizer.freeze(["frozen"])
        worker = DecoupledWorker.__new__(DecoupledWorker)
        worker.fragments = (small[0], fragments[0])
        worker.versions = [0, 0]
        worker.pending = {}
        worker.awaiting_global = set()
        worker.steps_since = [5, 5]
        worker.tokens_since = [50, 50]
        worker.baselines = [torch.zeros(4), torch.zeros(8)]
        worker.sequences = [0, 0]
        worker.sent = []
        worker._send_async = lambda index: worker.sent.append(index)
        worker.outbox = Path("outbox")
        worker.settings = type("S", (), {"wire_dtype": "fp32"})()
        worker._queue(0, optimizer)  # only the frozen parameter
        assert worker.sent == [] and worker.pending == {}
        worker._queue(1, optimizer)  # the frozen and the live one together
        assert worker.sent == [1]
    finally:
        optimizer.close()


def test_a_redirect_to_another_origin_never_carries_the_bearer_token(tmp_path):
    import http.server
    import threading

    from megatoken.training.diloco_client import RemoteError, Transport

    seen = []

    class Elsewhere(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    elsewhere = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Elsewhere)
    target = f"http://127.0.0.1:{elsewhere.server_address[1]}/stolen"

    class Redirecting(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", target)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    origin = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Redirecting)
    for server in (elsewhere, origin):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        transport = Transport(f"http://127.0.0.1:{origin.server_address[1]}", "t" * 32, 10)
        with pytest.raises(RemoteError) as refused:
            transport.request("GET", "/away")
        assert refused.value.status == 302 and seen == []
    finally:
        for server in (elsewhere, origin):
            server.shutdown()


def test_independent_stream_treats_an_episode_without_windows_as_an_idle_slot(
    tmp_path, monkeypatch
):
    from megatoken.training import stream as stream_module

    codec = ByteCodec.byte_fallback()
    source = write_demo_source(tmp_path / "source.jsonl", count=8, max_record_bytes=128)
    prepare_sources([source], tmp_path / "data", codec).close()
    dataset = PreparedCorpus([tmp_path / "data"])
    config = WindowConfig(max_window_tokens=512, max_records=2)
    real_plan = stream_module._WindowPlan

    class EmptyFirst:
        calls = 0

        def __init__(self, episode, codec, config):
            EmptyFirst.calls += 1
            self.real = real_plan(episode, codec, config)

        def __len__(self):
            return 0 if EmptyFirst.calls == 1 else len(self.real)

        def __getitem__(self, index):
            return self.real[index]

    monkeypatch.setattr(stream_module, "_WindowPlan", EmptyFirst)
    stream = IndependentWindowStream(dataset, codec, config, seed=5, rank=0, world_size=1)
    try:
        assert stream.next() is None  # the empty episode: one idle slot, no exception
        assert stream.next() is not None  # then it moves on to the next episode
    finally:
        stream.close()


def test_a_heartbeat_is_answered_while_a_merge_holds_the_coordinator(tmp_path):
    import threading

    coordinator, _, _ = _coordinator(tmp_path, workers=1, quorum=1)
    session = str(uuid.uuid4())
    coordinator.lease(0, session)
    entered, release = threading.Event(), threading.Event()

    def long_merge():
        with coordinator.lock:  # what a slow merge or snapshot holds
            entered.set()
            release.wait(30)

    thread = threading.Thread(target=long_merge, daemon=True)
    thread.start()
    assert entered.wait(5)
    try:
        answered = []
        beat = threading.Thread(
            target=lambda: answered.append(coordinator.heartbeat(0, session, steps=1, tokens=7)),
            daemon=True,
        )
        beat.start()
        beat.join(5)
        assert answered and answered[0]["global_tokens"] == 7
        assert len(answered[0]["versions"]) == len(coordinator.fragments)
    finally:
        release.set()
        thread.join(5)
