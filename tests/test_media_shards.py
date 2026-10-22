"""Sharded full-volume media preparation: the plan, the jobs, and the whole flow on one machine."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import pytest
import soundfile
from media_support import prepared_record

from megatoken import cloud, cloud_prepare
from megatoken.data import PreparedCorpus, reconcile
from megatoken.data.media_assets import TensorAssetStore
from megatoken.data.media_sharding import (
    plan_media_shards,
    read_listing,
    shard_seconds,
)
from megatoken.data.open_media import SOURCES, OpenSource
from megatoken.data.sources import TermsReview

ROOT = Path(__file__).resolve().parents[1]
TERMS = TermsReview(True, "CC-BY-4.0", "https://example.org", "Reviewer", "2026-10-22")


def source(source_id, task="transcription", reader="parquet", items=1000, **fields) -> OpenSource:
    return OpenSource(
        source_id,
        f"org/{source_id}",
        "0" * 40,
        reader,
        task,
        TERMS,
        include=("data/*.parquet",),
        inventory={"items": items, "basis": "test"},
        **fields,
    )


def run_spec(**changes):
    spec = {
        "schema": cloud_prepare.SCHEMA,
        "kind": "media",
        "id": "unit-media",
        "image": "nvcr.io/nvidia/pytorch:26.09-py3",
        "media_sources": ["librispeech_asr", "magicbrush"],
        "listing": "configs/data/files.media.json",
        "pip": ["Pillow"],
        "resources": {"cpu": "8", "memory": "32Gi", "gpu": 1},
        "sandboxes": 2,
        "shards_per_sandbox": 2,
        "processes": 1,
        "hours": {"ingest": 6, "reconcile": 2, "finish": 2},
    }
    return {**spec, **changes}


# ---- the plan -----------------------------------------------------------------------------


def test_every_file_of_every_source_is_in_exactly_one_shard():
    sources = {
        "many": source("many", items=100_000),
        "few": source("few", task="captioned_image", items=1_000),
    }
    listings = {
        "many": [(f"data/{i:03d}.parquet", 1000 + i) for i in range(40)],
        "few": [("data/0.parquet", 5000), ("data/1.parquet", 5000)],
    }
    plan = plan_media_shards(sources, 4, listings)
    assert len(plan) == 4
    for source_id, listing in listings.items():
        taken = [name for spec in plan for name in spec.get(source_id, {}).get("files", ())]
        assert sorted(taken) == sorted(name for name, _ in listing)
    hours = [shard_seconds(sources, spec, listings) for spec in plan]
    assert max(hours) < 1.3 * min(hours)  # balanced by estimated work, not by file count


def test_a_source_with_one_huge_file_is_striped_over_the_shards():
    sources = {"big": source("big", items=1_000_000), "small": source("small", items=10)}
    listings = {"big": [("data/all.parquet", 10**9)], "small": [("data/0.parquet", 1)]}
    plan = plan_media_shards(sources, 4, listings)
    stripes = [spec["big"]["stripe"] for spec in plan if "big" in spec]
    assert sorted(stripes) == [[i, 4] for i in range(4)]
    assert all("files" not in spec["big"] for spec in plan if "big" in spec)


def test_fleurs_is_divided_by_language():
    fleurs = source(
        "fleurs", task="translation_pair", reader="fleurs", options={"languages": list("abcdef")}
    )
    plan = plan_media_shards({"fleurs": fleurs}, 3, {})
    languages = [lang for spec in plan for lang in spec["fleurs"]["options"]["languages"]]
    assert sorted(languages) == list("abcdef")


def test_the_listing_is_read_from_its_file_when_it_exists(tmp_path):
    path = tmp_path / "files.json"
    path.write_text(json.dumps({"x": [["data/a.parquet", 7]]}))
    assert read_listing({}, path) == {"x": [("data/a.parquet", 7)]}


# ---- the run file -------------------------------------------------------------------------


def test_media_jobs_use_a_gpu_to_ingest_and_none_to_reconcile(tmp_path):
    spec = cloud_prepare.validate(run_spec())
    written = cloud_prepare.write_jobs(spec, "configs/cloud/unit-media.json", tmp_path)
    jobs = {path.stem: json.loads(path.read_text()) for path in written}
    # The finish wave runs inside the ingest sandboxes: no sandboxes of its own.
    assert set(jobs) == {"ingest-0", "ingest-1", "reconcile"}
    for job in jobs.values():
        cloud.validate(job)
        cloud.request(job)
    assert jobs["ingest-0"]["resources"]["gpu"] == 1 and not jobs["reconcile"]["resources"]["gpu"]
    assert jobs["ingest-0"]["outputs"] == ["out/prepared", "out/logs"]
    command = jobs["ingest-1"]["command"][2]
    assert "scripts/acquire_packs.py" in command and "--sandbox 1" in command
    assert "ffmpeg" in command
    # Sandboxes outlive their own work: ingest waits through reconcile and finish.
    assert jobs["ingest-0"]["lifetime_hours"] == 6 + 2 + 2 + cloud_prepare.PULL_HOURS
    digests = [item["path"] for item in jobs["reconcile"]["inputs"]]
    assert digests == [f"in/digests/{shard:05d}.sqlite" for shard in range(4)]


@pytest.mark.parametrize(
    "change",
    [
        {"resources": {"cpu": "8", "memory": "32Gi"}},
        {"media_sources": []},
        {"media_sources": "some"},
        {"kind": "audio"},
        {"sandboxes": 0},
    ],
)
def test_invalid_media_runs_are_rejected(change):
    with pytest.raises(cloud.CloudError):
        cloud_prepare.validate(run_spec(**change))


def test_run_options_and_procedural_shares_reach_every_shard(tmp_path):
    spec = run_spec(
        options={"librispeech_asr": {"max_seconds": 10.0}},
        procedural_every={"librispeech_asr": 20},
    )
    (tmp_path / "configs/data").mkdir(parents=True)
    listing = {
        s: [[f"data/{i:04d}.parquet", 100] for i in range(8)]
        for s in ("librispeech_asr", "magicbrush")
    }
    (tmp_path / spec["listing"]).write_text(json.dumps(listing))
    plan = cloud_prepare.shard_plan(spec, tmp_path)
    for shard in plan:
        entry = shard["librispeech_asr"]
        assert entry["options"] == {"max_seconds": 10.0} and entry["procedural_every"] == 20
    with pytest.raises(cloud.CloudError, match="unknown"):
        cloud_prepare.shard_plan({**spec, "procedural_every": {"nope": 3}}, tmp_path)


def test_committed_media_runs_are_valid_and_their_listings_cover_their_sources(tmp_path):
    runs = sorted((ROOT / "configs/cloud").glob("prepare-full-media-*.json"))
    assert runs
    for path in runs:
        spec = cloud_prepare.validate(json.loads(path.read_text()))
        assert spec["kind"] == "media"
        sources = cloud_prepare.media_sources(spec)
        listing = json.loads((ROOT / spec["listing"]).read_text())
        file_based = {
            s for s, v in sources.items() if v.reader in ("parquet", "webdataset", "archive")
        }
        assert file_based <= set(listing), (path, sorted(file_based - set(listing)))
        plan = cloud_prepare.shard_plan(spec, ROOT)  # no shard is empty
        assert len(plan) == cloud_prepare.shard_count(spec)
        for job in (
            json.loads(item.read_text())
            for item in cloud_prepare.write_jobs(spec, str(path.relative_to(ROOT)), tmp_path)
        ):
            cloud.validate(job)
        # Every source the run names is approved, and no two runs build the same source.
    seen: dict[str, str] = {}
    for path in runs:
        for source_id in cloud_prepare.media_sources(json.loads(path.read_text())):
            assert source_id not in seen, f"{source_id}: {seen[source_id]} and {path.name}"
            seen[source_id] = path.name
    assert set(seen) == set(SOURCES)


# ---- the whole flow on one machine --------------------------------------------------------


def wave(seed: int, seconds: float = 2.0, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    time = np.arange(int(seconds * rate)) / rate
    soundfile.write(buffer, 0.3 * np.sin(2 * np.pi * (200 + 40 * seed) * time), rate, format="FLAC")
    return buffer.getvalue()


class FakePreparer:
    def __init__(self, packs, root, device="cpu"):
        self.store = TensorAssetStore(root)

    def media(self, modality, path, *, target=True):
        seed = int(hashlib.sha256(Path(path).read_bytes()).hexdigest()[:6], 16)
        return prepared_record(self.store, modality, seed=seed, target=target)


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "prepare_open_media", ROOT / "scripts/prepare_open_media.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MediaPreparer", FakePreparer)
    return module


def test_two_shards_ingest_reconcile_and_finish_keeping_every_asset_in_place(
    tmp_path, monkeypatch, script
):
    """The same recording in both shards is found by reconcile and kept once; each root holds
    the assets of the episodes it kept."""
    folder = tmp_path / "inputs" / "librispeech_asr" / "all"
    folder.mkdir(parents=True)
    for number in range(2):
        shared = [wave(1)]  # one recording appears in both files
        own = [wave(10 * number + i + 2) for i in range(3)]
        parquet.write_table(
            pa.table(
                {
                    "audio": [{"bytes": data, "path": ""} for data in shared + own],
                    "text": [f"SHARED {number}"] + [f"TEXT {number} {i}" for i in range(3)],
                    "id": [f"{number}-{i}" for i in range(4)],
                    "speaker_id": [number * 10 + i for i in range(4)],
                }
            ),
            folder / f"{number}.parquet",
        )
    roots = [tmp_path / "out/prepared" / f"{shard:05d}" for shard in range(2)]

    def run(*arguments):
        monkeypatch.setattr("sys.argv", ["prepare_open_media", *arguments])
        script.main()

    for shard, root in enumerate(roots):
        spec_file = tmp_path / f"shard-{shard}.json"
        spec_file.write_text(
            json.dumps(
                {"librispeech_asr": {"files": [f"all/{shard}.parquet"], "procedural_every": 0}}
            )
        )
        common = ["--root", str(root), "--codec", "byte", "--shard-spec", str(spec_file)]
        run(
            *common,
            *("--packs", "unused", "--inputs", str(tmp_path / "inputs")),
            *("--work", str(tmp_path / "scratch"), "--ingest-only"),
        )
        assert (root / "digest.sqlite").is_file() and not (root / "manifest.json").exists()
        # Running the ingest again changes nothing: the shard is done.
        run(
            *common,
            *("--packs", "unused", "--inputs", str(tmp_path / "inputs")),
            *("--work", str(tmp_path / "scratch"), "--ingest-only"),
        )
    summary = reconcile([root / "digest.sqlite" for root in roots], tmp_path / "decisions")
    # The shared recording's audio identity is in both shards (its two transcripts differ, so
    # those are two episodes); the second shard's copy is a duplicate of the first's.
    assert summary["cross_shard_duplicates"] == 1, summary
    for shard, root in enumerate(roots):
        spec_file = tmp_path / f"shard-{shard}.json"
        run(
            *("--root", str(root), "--codec", "byte", "--shard-spec", str(spec_file)),
            *("--decisions", str(tmp_path / "decisions" / f"decisions-{shard:04d}.sqlite")),
        )
        assert (root / "manifest.json").is_file()
        assert any(path.is_file() for path in root.rglob("*") if "sqlite" not in path.name)
    with PreparedCorpus(roots, split=None) as corpus:
        assert len(corpus) == 2 * 4 * 2 - 1  # 8 clips x (asr, identity), the shared identity once


# ---- the launcher, against a fake fleet ---------------------------------------------------


class Result:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class Fleet:
    """Sandboxes that record what the launcher starts, writes and runs in them."""

    def __init__(self, spec):
        self.spec, self.jobs, self.stopped, self.failing = spec, {}, [], set()
        self.written: dict[str, dict[str, bytes]] = {}
        self.ran: dict[str, list[list[str]]] = {}

    def start(self, job, repo, *, secrets=None, snapshot=None):
        self.jobs[job["id"]] = job
        return {"sandbox_id": job["id"], "url": f"https://{job['id']}.invalid"}, None

    def files(self, sandbox_id):
        if sandbox_id.endswith("reconcile"):
            names = [
                f"out/decisions/{cloud_prepare._decision_name(s)}"
                for s in range(cloud_prepare.shard_count(self.spec))
            ]
        else:
            index = int(sandbox_id.rsplit("-", 1)[1])
            names = [
                f"out/prepared/{s:05d}/digest.sqlite"
                for s in cloud_prepare.shards_of(self.spec, index)
            ]
        return [{"path": name, "bytes": 1, "sha256": "a" * 64} for name in names]

    def attach(self, sandbox_id):
        fleet = self

        class Sandbox:
            def read_file(self, path):
                return Result(json.dumps({"files": fleet.files(sandbox_id)}).encode())

            def write_file(self, path, content):
                fleet.written.setdefault(sandbox_id, {})[path] = content
                return Result(None)

            def exec(self, argv):
                fleet.ran.setdefault(sandbox_id, []).append(argv)
                if argv[0] == "cat":
                    status = "1" if sandbox_id in fleet.failing else "0"
                    return Result(type("Done", (), {"stdout": status, "returncode": 0}))
                return Result(type("Done", (), {"stdout": f"log of {sandbox_id}", "returncode": 0}))

        return Sandbox()

    def stop_all(self, ids, reason, grace=None):
        self.stopped += list(ids)
        return []


def media_fleet(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "configs/data").mkdir(parents=True)
    spec = run_spec(media_sources=["librispeech_asr", "magicbrush"])
    listing = {
        s: [[f"data/{i:04d}.parquet", 1000] for i in range(8)]
        for s in ("librispeech_asr", "magicbrush")
    }
    (repo / spec["listing"]).write_text(json.dumps(listing))
    (repo / "configs/run.json").write_text(json.dumps(spec))
    token = tmp_path / "token"
    token.write_text("hf_secret\n")
    fleet = Fleet(cloud_prepare.validate(spec))
    monkeypatch.setattr(cloud_prepare, "RUNS", tmp_path / "state")
    monkeypatch.setattr(
        cloud, "code_snapshot", lambda repo, archive: {"commit": "c1", "sha256": "0"}
    )
    monkeypatch.setattr(cloud, "start", fleet.start)
    monkeypatch.setattr(cloud, "_attach", fleet.attach)
    monkeypatch.setattr(cloud, "share_token", lambda sandbox_id: f"token-of-{sandbox_id}")
    monkeypatch.setattr(cloud, "outputs_ready", lambda sandbox_id: True)
    monkeypatch.setattr(cloud, "exit_status", lambda sandbox_id: 0)
    monkeypatch.setattr(cloud, "stop_all", fleet.stop_all)
    monkeypatch.setattr(cloud, "record", lambda entry: None)
    return repo, token, fleet


def test_launch_finishes_a_media_run_inside_the_ingest_sandboxes(tmp_path, monkeypatch):
    repo, token, fleet = media_fleet(tmp_path, monkeypatch)
    run = cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    assert set(fleet.jobs) == {
        "unit-media-ingest-0",
        "unit-media-ingest-1",
        "unit-media-reconcile",
    }
    assert fleet.jobs["unit-media-reconcile"]["resources"]["gpu"] == 0
    # The ingest sandboxes got the verdicts' address and the reconcile token, then ran finish.
    for index in range(2):
        sandbox = f"unit-media-ingest-{index}"
        written = fleet.written[sandbox]
        given = json.loads(written[cloud_prepare.FINISH_INPUTS])
        assert given["peer"] == "unit-media-reconcile" and given["url"].startswith("https://")
        assert sorted(given["files"]) == [
            f"out/decisions/decisions-{shard:04d}.sqlite"
            for shard in cloud_prepare.shards_of(fleet.spec, index)
        ]
        assert written["/work/.cloud/secrets/share-unit-media-reconcile"] == (
            b"token-of-unit-media-reconcile"
        )
        start = next(argv for argv in fleet.ran[sandbox] if argv[0] == "bash")
        assert f"finish --run configs/run.json --sandbox {index}" in start[2]
    # Only the reconcile sandbox is released: the ingest sandboxes serve the finished roots.
    assert fleet.stopped == ["unit-media-reconcile"]
    assert run["waves"]["finish"] == {"0": "unit-media-ingest-0", "1": "unit-media-ingest-1"}
    assert cloud_prepare._everything(run).count("unit-media-ingest-0") == 1


def test_a_failed_media_finish_is_redone_alone(tmp_path, monkeypatch):
    repo, token, fleet = media_fleet(tmp_path, monkeypatch)
    fleet.failing = {"unit-media-ingest-1"}
    with pytest.raises(cloud.CloudError, match="finish failed in"):
        cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    assert (tmp_path / "state/unit-media.logs/unit-media-ingest-1.log").read_text() == (
        "log of unit-media-ingest-1"
    )
    assert "unit-media-ingest-1" not in fleet.stopped
    fleet.failing, fleet.jobs, fleet.ran = set(), {}, {}
    cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    assert fleet.jobs == {}  # nothing is started again
    assert list(fleet.ran) == ["unit-media-ingest-1"]  # only the failed sandbox finishes again
