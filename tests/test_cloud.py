"""Offline checks of the CoreWeave job contract (no API calls)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from megatoken import cloud


def _job(**overrides):
    return {
        "schema": cloud.SCHEMA,
        "id": "unit",
        "image": "nvcr.io/nvidia/pytorch:26.09-py3",
        "resources": {"cpu": "8", "memory": "64Gi", "gpu": 1},
        "lifetime_hours": 2,
        "command": ["python", "-c", "print(1)"],
        "outputs": ["out"],
        "inputs": [
            {"path": "inputs/a.bin", "url": "https://example.invalid/a", "sha256": "0" * 64}
        ],
        **overrides,
    }


def test_request_mounts_bootstrap_and_never_carries_credentials(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret-value")
    request = cloud.request(cloud.validate(_job(env={"HF_HOME": "/work/hf"}, pip=["x==1"])))
    assert request["resources"] == {"cpu": "8", "memory": "64Gi", "gpu": {"count": 1}}
    assert request["max_lifetime_seconds"] == 7200
    assert cloud.BOOTSTRAP in request["mounted_files"]
    assert request["environment_variables"]["MEGATOKEN_CW_OUTPUTS"] == "out"
    assert request["environment_variables"]["MEGATOKEN_CW_PIP"] == "x==1"
    assert "secret-value" not in json.dumps(request)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "other"},
        {"lifetime_hours": 0},
        {"resources": {"cpu": "1", "memory": "1Gi", "gpu": 2}},
        {"outputs": ["/abs"]},
        {"inputs": [{"path": "../x", "url": "u", "sha256": "0" * 64}]},
        {"inputs": [{"path": "x", "sha256": "0" * 64}]},
        {"inputs": [{"path": "x", "url": "u", "source": "s", "sha256": "0" * 64}]},
    ],
)
def test_invalid_jobs_are_rejected(change):
    with pytest.raises(cloud.CloudError):
        cloud.validate(_job(**change))


def test_code_snapshot_requires_a_clean_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run([*git, "add", "a.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    snapshot = cloud.code_snapshot(repo, tmp_path / "code.tar.gz")
    assert len(snapshot["commit"]) == 40
    assert snapshot["sha256"] == cloud.sha256_file(tmp_path / "code.tar.gz")
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(cloud.CloudError, match="Commit"):
        cloud.code_snapshot(repo, tmp_path / "again.tar.gz")


def test_shared_credential_files_are_refused(tmp_path):
    secrets = tmp_path / "coreweave.env"
    secrets.write_text("WANDB_API_KEY=x\n")
    secrets.chmod(0o644)
    with pytest.raises(cloud.CloudError, match="chmod 600"):
        cloud.load_credentials(secrets)


def test_push_dirs_expand_to_pinned_inputs(tmp_path):
    (tmp_path / "corpus" / "sub").mkdir(parents=True)
    (tmp_path / "corpus" / "sub" / "a.txt").write_text("a")
    job = cloud.validate(
        _job(inputs=[], push_dirs=[{"source": str(tmp_path / "corpus"), "path": "out/corpus"}])
    )
    (item,) = cloud.expand_push_dirs(job)
    assert item["path"] == "out/corpus/sub/a.txt"
    assert item["sha256"] == cloud.sha256_file(tmp_path / "corpus" / "sub" / "a.txt")
    with pytest.raises(cloud.CloudError):
        cloud.validate(_job(push_dirs=[{"source": "x", "path": "/abs"}]))


def test_scratch_volume_only_when_requested():
    assert "volumes" not in cloud.request(cloud.validate(_job()))
    (volume,) = cloud.request(cloud.validate(_job(scratch="50Gi")))["volumes"]
    assert volume == {"name": "work", "mount_path": cloud.WORK, "size": "50Gi"}


def test_pull_splits_large_outputs_and_retries_dropped_parts(tmp_path, monkeypatch):
    SandboxError = _stub_cwsandbox(monkeypatch)
    monkeypatch.setattr(cloud, "PART_BYTES", 400)
    payload = bytes(range(256)) * 4  # 1024 bytes -> parts of 400, 400, 224
    source = tmp_path / "payload"
    source.write_bytes(payload)
    files, reads, removed = {}, [], []

    class Done:
        def __init__(self, returncode=0):
            self.returncode, self.stderr = returncode, ""

        def result(self):
            return self

    class Sandbox:
        def exec(self, argv):
            if argv[0] == "rm":
                removed.extend(argv[2:])
                return Done()
            if argv[0] == "test":
                return Done(0 if argv[2] in files else 1)
            remote, prefix, part = argv[-3:]
            for index, offset in enumerate(range(0, len(payload), int(part))):
                files[f"{prefix}{index:04d}"] = payload[offset : offset + int(part)]
            files[f"{prefix}done"] = b""
            return Done()

        def read_file_streaming(self, path, timeout_seconds):
            reads.append(path)
            data = files[path]

            def blocks():
                yield data[:100]
                if reads.count(path) == 1 and path.endswith("0001"):
                    raise SandboxError("ended without exit status")
                yield data[100:]

            return blocks()

    target = tmp_path / "out" / "file.bin"
    target.parent.mkdir()
    item = {"path": "out/file.bin", "bytes": len(payload), "sha256": cloud.sha256_file(source)}
    cloud._pull(Sandbox(), item, target)
    assert target.read_bytes() == payload
    assert [path[-4:] for path in reads] == ["0000", "0001", "0001", "0002"]
    assert sorted(removed) == sorted(files)
    assert not target.with_name(".file.bin.partial").exists()


def _stub_cwsandbox(monkeypatch):
    import sys
    import types

    class SandboxError(Exception):
        pass

    exceptions = types.ModuleType("cwsandbox.exceptions")
    exceptions.SandboxError = SandboxError
    monkeypatch.setitem(sys.modules, "cwsandbox", types.ModuleType("cwsandbox"))
    monkeypatch.setitem(sys.modules, "cwsandbox.exceptions", exceptions)
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    return SandboxError


def test_push_uploads_retried_parts_that_the_bootstrap_joins(tmp_path, monkeypatch):
    import hashlib
    import os
    import re
    import tarfile

    error = _stub_cwsandbox(monkeypatch)
    monkeypatch.setattr(cloud, "PART_BYTES", 300)
    work = tmp_path / "work"
    inbox = work / ".cloud" / "inbox"
    inbox.mkdir(parents=True)
    writes = []

    class Done:
        def result(self):
            return None

    class Dropped:
        def result(self):
            raise error("dropped")

    class Sandbox:
        def write_file(self, path, data):
            # Writes complete asynchronously; a dropped stream fails in result().
            writes.append(path)
            if len(writes) == 2:
                return Dropped()
            (inbox / path.rsplit("/", 1)[1]).write_bytes(data)
            return Done()

    monkeypatch.setattr(cloud, "WORK", str(work))
    payload = os.urandom(1000)
    big = tmp_path / "big.bin"
    big.write_bytes(payload)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "a.py").write_text("x = 1\n")
    archive = tmp_path / "code.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(code_dir / "a.py", arcname="a.py")
    digests = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (big, empty, archive)
    }
    cloud._push(Sandbox(), {digest: path for path, digest in digests.items()}, in_flight=3)
    assert len(writes) == 4 + 1 + 1 + 1  # every part once, plus one retried part
    assert sorted(p.name for p in inbox.iterdir() if p.name.startswith(digests[big])) == [
        f"{digests[big]}.{i:04d}" for i in range(4)
    ]
    spec = {
        "code": {"sha256": digests[archive]},
        "inputs": [
            {"path": "inputs/big.bin", "sha256": digests[big]},
            {"path": "inputs/copy.bin", "sha256": digests[big]},
            {"path": "inputs/empty.bin", "sha256": digests[empty]},
        ],
    }
    (work / ".cloud" / "inputs.json").write_text(json.dumps(spec))
    script = Path(cloud.__file__).with_name("cloud_bootstrap.sh").read_text()
    materialize = re.search(r"python3 - <<'PY' \|\| die inputs\n(.*?)\nPY\n", script, re.S)[1]
    subprocess.run(
        [sys.executable, "-c", materialize],
        env={**os.environ, "MEGATOKEN_CW_WORK": str(work)},
        check=True,
    )
    assert (work / "inputs" / "big.bin").read_bytes() == payload
    assert (work / "inputs" / "copy.bin").read_bytes() == payload
    assert (work / "inputs" / "empty.bin").read_bytes() == b""
    assert (work / "megatoken" / "a.py").read_text() == "x = 1\n"
    assert not inbox.exists()
