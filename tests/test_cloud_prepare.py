"""Offline checks of the sharded full-volume preparation on sandboxes (no API calls)."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from megatoken import cloud, cloud_prepare
from megatoken.data import (
    Episode,
    Lineage,
    PreparedCorpus,
    TermsReview,
    make_records,
    pin_local_source,
    reconcile,
)
from megatoken.data.sources import SourceFile, write_source_manifest

ROOT = Path(__file__).resolve().parents[1]


def run_spec(**overrides):
    return {
        "schema": cloud_prepare.SCHEMA,
        "id": "unit-prep",
        "image": "nvcr.io/nvidia/pytorch:26.09-py3",
        "sources": "configs/data/sources.agentic.json",
        "listing": "configs/data/files.agentic.json",
        "review_config": "configs/data/parser-review.agentic.json",
        "pip": ["datasets>=3,<5"],
        "resources": {"cpu": "16", "memory": "32Gi"},
        "sandboxes": 3,
        "shards_per_sandbox": 4,
        "processes": 4,
        "hours": {"ingest": 10, "reconcile": 5, "finish": 3},
        **overrides,
    }


def test_every_wave_job_is_valid_and_wired_to_the_right_shards(tmp_path):
    spec = cloud_prepare.validate(run_spec())
    written = cloud_prepare.write_jobs(spec, "configs/cloud/unit-prep.json", tmp_path)
    jobs = {path.stem: json.loads(path.read_text()) for path in written}
    assert set(jobs) == {f"{wave}-{i}" for wave in ("ingest", "finish") for i in range(3)} | {
        "reconcile"
    }
    for job in jobs.values():
        cloud.validate(job)
        assert job["resources"] == {"cpu": "16", "memory": "32Gi"}
        assert job["serve_outputs"] and job["serve_port"] == cloud_prepare.PORT
        # Sandboxes outlive their own work until the next wave has fetched it.
        assert int(job["env"]["MEGATOKEN_CW_COLLECT_WAIT"]) == int(job["lifetime_hours"] * 3600)
        # The token is read from a secret file in the sandbox, never spelled out.
        assert "TOKEN=" not in json.dumps(job["env"])
    assert jobs["ingest-0"]["lifetime_hours"] == 20 and jobs["finish-0"]["lifetime_hours"] == 3
    assert jobs["reconcile"]["lifetime_hours"] == 10
    assert "megatoken.cloud_prepare ingest" in jobs["ingest-1"]["command"][2]
    assert "--sandbox 1" in jobs["ingest-1"]["command"][2]
    # Reconcile sees every shard's digest and only digests, in shard order (the glob sorts them).
    digests = [item["path"] for item in jobs["reconcile"]["inputs"]]
    assert digests == [f"in/digests/{shard:05d}.sqlite" for shard in range(12)]
    assert digests == sorted(digests)
    assert {item["peer"] for item in jobs["reconcile"]["inputs"]} == {
        "ingest-0",
        "ingest-1",
        "ingest-2",
    }
    # A finishing sandbox fetches its own shards' databases and verdicts, nothing else.
    inputs = jobs["finish-2"]["inputs"]
    assert sorted(item["path"] for item in inputs) == sorted(
        [f"out/prepared/{shard:05d}/preparation.sqlite" for shard in range(8, 12)]
        + [f"in/decisions/decisions-{shard:04d}.sqlite" for shard in range(8, 12)]
    )
    assert {item["peer"] for item in inputs} == {"ingest-2", "reconcile"}
    assert all(item["url"].startswith("https://") and len(item["sha256"]) == 64 for item in inputs)
    cloud.request(jobs["finish-2"])  # a well-formed sandbox request


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"sandboxes": 0},
        {"processes": 1.5},
        {"resources": {"cpu": "16", "memory": "32Gi", "gpu": 1}},
        {"hours": {"ingest": 10, "reconcile": 5}},
        {"hours": {"ingest": 700, "reconcile": 30, "finish": 1}},
        {"sandboxes": 1000, "shards_per_sandbox": 1000},
    ],
)
def test_invalid_runs_are_rejected(change):
    with pytest.raises(cloud.CloudError):
        cloud_prepare.validate(run_spec(**change))


def test_a_missing_published_file_is_named():
    with pytest.raises(cloud.CloudError, match="did not publish"):
        cloud_prepare.peer_inputs("x", "https://x", [], {"out/a": "in/a"})


def test_committed_run_files_are_valid_and_their_listings_exist(tmp_path):
    for path in sorted((ROOT / "configs/cloud").glob("prepare-full-*.json")):
        spec = cloud_prepare.validate(json.loads(path.read_text()))
        assert (ROOT / spec["sources"]).is_file() and (ROOT / spec["listing"]).is_file(), path
        assert (ROOT / spec["review_config"]).is_file(), path
        manifest = json.loads((ROOT / spec["sources"]).read_text())
        listing = json.loads((ROOT / spec["listing"]).read_text())
        assert {s["source_id"] for s in manifest["sources"] if s["format"] == "hf"} <= set(listing)
        cloud_prepare.shard_plan(spec, ROOT)  # no shard is empty
        for job in cloud_prepare_jobs(spec, path, tmp_path):
            cloud.validate(job)


def cloud_prepare_jobs(spec, path, tmp_path):
    return [
        json.loads(item.read_text())
        for item in cloud_prepare.write_jobs(spec, str(path.relative_to(ROOT)), Path(tmp_path))
    ]


# ---- the whole flow on this machine -------------------------------------------------------

TERMS = TermsReview(True, "fixture", reviewed_by="test", reviewed_at="2026-10-17")


def document(name, text, family):
    return Episode(
        name,
        make_records(text.encode(), logical_id=name, role="document", policy_mask=True),
        Lineage("test", "fixed", family),
        stream="natural_documents",
    )


def body(index):
    return " ".join(f"{index:x}{i:x}" for i in range(120))


def test_ingest_reconcile_and_finish_on_one_machine_remove_cross_shard_duplicates(tmp_path):
    files = []
    for number in range(4):
        path = tmp_path / f"part-{number}.jsonl"
        episodes = [
            document(f"d{number}-{i}", body(10 * number + i), f"f{number}-{i}") for i in range(3)
        ]
        # Two neighbouring rows of one text: wherever the plan cuts, both shards hold it.
        episodes += [
            document(f"copy-{number}-{side}", body(99), f"copy-{number}-{side}") for side in "ab"
        ]
        path.write_text("\n".join(json.dumps(item.to_dict()) for item in episodes) + "\n")
        files.append(path)
    base = pin_local_source(
        files[0], source_id="parts", adapter="native", stream="natural_documents", terms=TERMS
    )
    source = replace(
        base,
        files=tuple(
            SourceFile(
                str(item),
                pin_local_source(
                    item, source_id="x", adapter="native", stream="natural_documents", terms=TERMS
                )
                .files[0]
                .sha256,
                item.stat().st_size,
            )
            for item in files
        ),
    )
    manifest = tmp_path / "sources.json"
    write_source_manifest(manifest, (source,))
    spec = run_spec(
        sources=str(manifest),
        listing=str(tmp_path / "listing.json"),
        review_config=str(ROOT / "configs/data/parser-review.json"),
        sandboxes=2,
        shards_per_sandbox=1,
        processes=2,
    )
    work = tmp_path / "work"
    for sandbox in range(2):
        commands = cloud_prepare.shard_commands(
            spec, sandbox, stage="ingest", work=work, codec="byte"
        )
        assert cloud_prepare.run_shards(commands, 2, work / "logs") == []
    digests = [work / "out/shards" / f"{shard:05d}" / "digest.sqlite" for shard in range(2)]
    assert all(path.is_file() for path in digests)
    summary = reconcile(digests, work / "out/decisions")
    assert (summary["cross_shard_duplicates"], summary["accepted"]) == (1, 13), summary
    (work / "in").mkdir()
    (work / "in/decisions").symlink_to(work / "out/decisions")
    for shard in range(2):  # the finishing sandbox received the shard database as an input
        (work / "out/prepared" / f"{shard:05d}").mkdir(parents=True)
        (work / "out/prepared" / f"{shard:05d}" / "preparation.sqlite").write_bytes(
            (work / "out/shards" / f"{shard:05d}" / "preparation.sqlite").read_bytes()
        )
    for sandbox in range(2):
        commands = cloud_prepare.shard_commands(
            spec, sandbox, stage="finish", work=work, codec="byte"
        )
        assert cloud_prepare.run_shards(commands, 2, work / "logs") == []
    roots = [work / "out/prepared" / f"{shard:05d}" for shard in range(2)]
    with PreparedCorpus(roots, split=None) as corpus:
        assert len(corpus) == 13  # 20 documents; seven are copies of another


def test_the_launcher_refuses_a_run_whose_shards_are_not_pinned(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps(run_spec(listing="configs/data/none.json")))
    with pytest.raises(cloud.CloudError, match="is missing"):
        cloud_prepare.launch(tmp_path / "run.json", tmp_path)


def test_a_shard_with_nothing_to_read_is_refused(tmp_path):
    manifest = {
        "schema_version": 1,
        "sources": [
            {
                "source_id": "only",
                "uri": "hf://datasets/example/only",
                "format": "hf",
                "adapter": "chat",
                "stream": "chat",
                "revision": "a" * 40,
                "terms": {"approved": True},
                "split": "train",
            }
        ],
    }
    (tmp_path / "sources.json").write_text(json.dumps(manifest))
    (tmp_path / "listing.json").write_text(json.dumps({"only": [["hf://x/only-0.parquet", 0]]}))
    spec = run_spec(sources="sources.json", listing="listing.json", sandboxes=2)
    with pytest.raises(cloud.CloudError, match="no rows to read"):
        cloud_prepare.shard_plan(spec, tmp_path)


def test_job_ids_are_unique_and_plain():
    spec = cloud_prepare.validate(run_spec())
    ids = [cloud_prepare.ingest_job(spec, i, "x.json")["id"] for i in range(3)]
    assert len(set(ids)) == 3 and all(re.fullmatch(r"[a-z0-9-]+", name) for name in ids)


# ---- the launcher, against a fake fleet ---------------------------------------------------


class Result:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class Fleet:
    """Stands in for the sandbox service: records starts and stops, serves outputs manifests."""

    def __init__(self, spec):
        self.spec, self.jobs, self.stopped, self.failing, self.secrets = spec, {}, [], set(), {}

    def start(self, job, repo, *, secrets=None, snapshot=None):
        self.jobs[job["id"]] = job
        self.secrets[job["id"]] = sorted(secrets)
        return {"sandbox_id": job["id"], "url": f"https://{job['id']}.invalid"}, None

    def files(self, sandbox_id):
        spec = self.spec
        if sandbox_id.endswith("reconcile"):
            names = [
                f"out/decisions/{cloud_prepare._decision_name(s)}"
                for s in range(cloud_prepare.shard_count(spec))
            ]
        else:
            index = int(sandbox_id.rsplit("-", 1)[1])
            names = [
                f"out/shards/{s:05d}/{kind}.sqlite"
                for s in cloud_prepare.shards_of(spec, index)
                for kind in ("digest", "preparation")
            ]
        return [{"path": name, "bytes": 1, "sha256": "a" * 64} for name in names]

    def attach(self, sandbox_id):
        fleet = self

        class Sandbox:
            def read_file(self, path):
                return Result(json.dumps({"files": fleet.files(sandbox_id)}).encode())

            def exec(self, argv):
                return Result(type("Done", (), {"stdout": f"log of {sandbox_id}", "returncode": 0}))

        return Sandbox()

    def stop_all(self, ids, reason, grace=None):
        self.stopped += list(ids)
        return []


def fleet_env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    sources = [
        {
            "source_id": name,
            "uri": f"hf://datasets/example/{name}",
            "format": "hf",
            "adapter": "chat",
            "stream": "chat",
            "revision": "a" * 40,
            "terms": {"approved": True},
            "split": "train",
        }
        for name in ("alpha", "beta")
    ]
    (repo / "configs/sources.json").write_text(
        json.dumps({"schema_version": 1, "sources": sources})
    )
    listing = {
        s["source_id"]: [[f"hf://x/{s['source_id']}-{i}.parquet", 1000] for i in range(6)]
        for s in sources
    }
    (repo / "configs/listing.json").write_text(json.dumps(listing))
    spec = run_spec(
        sources="configs/sources.json",
        listing="configs/listing.json",
        sandboxes=2,
        shards_per_sandbox=2,
        processes=2,
    )
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
    monkeypatch.setattr(cloud, "outputs_ready", lambda sandbox_id: True)
    monkeypatch.setattr(
        cloud, "exit_status", lambda sandbox_id: 1 if sandbox_id in fleet.failing else 0
    )
    monkeypatch.setattr(cloud, "stop_all", fleet.stop_all)
    monkeypatch.setattr(cloud, "record", lambda entry: None)
    return repo, token, fleet


def test_launch_runs_the_three_waves_and_releases_the_first_two(tmp_path, monkeypatch):
    repo, token, fleet = fleet_env(tmp_path, monkeypatch)
    run = cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    # Waves run in order: reconcile's inputs name the ingest sandboxes, finish's name both.
    assert set(fleet.jobs) == {
        f"unit-prep-{name}"
        for name in ("ingest-0", "ingest-1", "reconcile", "finish-0", "finish-1")
    }
    for job in fleet.jobs.values():
        cloud.validate(job)
        assert "hf_secret" not in json.dumps(job)
    assert all(names == ["hf-token"] for names in fleet.secrets.values())
    reconcile = fleet.jobs["unit-prep-reconcile"]
    assert [item["peer"] for item in reconcile["inputs"]] == (
        ["unit-prep-ingest-0"] * 2 + ["unit-prep-ingest-1"] * 2
    )
    finish = fleet.jobs["unit-prep-finish-1"]
    assert {item["peer"] for item in finish["inputs"]} == {
        "unit-prep-ingest-1",
        "unit-prep-reconcile",
    }
    # Only the sandboxes serving the finished roots are left running.
    assert sorted(fleet.stopped) == [
        "unit-prep-ingest-0",
        "unit-prep-ingest-1",
        "unit-prep-reconcile",
    ]
    assert run["waves"]["finish"] == {"0": "unit-prep-finish-0", "1": "unit-prep-finish-1"}


def test_a_failed_finish_is_redone_alone_from_the_waves_before_it(tmp_path, monkeypatch):
    repo, token, fleet = fleet_env(tmp_path, monkeypatch)
    fleet.failing = {"unit-prep-finish-1"}
    with pytest.raises(cloud.CloudError, match="finish failed in"):
        cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    # The failed sandbox is stopped and its log kept; the earlier waves still hold the data.
    assert fleet.stopped == ["unit-prep-finish-1"]
    log = tmp_path / "state/unit-prep.logs/unit-prep-finish-1.log"
    assert log.read_text() == "log of unit-prep-finish-1"
    fleet.failing, fleet.jobs, fleet.stopped = set(), {}, []
    cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    assert list(fleet.jobs) == ["unit-prep-finish-1"]
    assert sorted(fleet.stopped) == [
        "unit-prep-ingest-0",
        "unit-prep-ingest-1",
        "unit-prep-reconcile",
    ]


def test_a_run_resumes_only_from_the_commit_it_started_at(tmp_path, monkeypatch):
    repo, token, fleet = fleet_env(tmp_path, monkeypatch)
    cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
    monkeypatch.setattr(
        cloud, "code_snapshot", lambda repo, archive: {"commit": "c2", "sha256": "0"}
    )
    with pytest.raises(cloud.CloudError, match="resume from there"):
        cloud_prepare.launch(repo / "configs/run.json", repo, hf_token=token, poll=0)
