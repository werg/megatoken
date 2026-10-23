"""A stand-in for the Hugging Face Hub, shared by the cache tests (no network)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

LFS_FROM = (
    500 << 10
)  # above this size the fake Hub records a sha256 (the manifest; pieces are read)


class RepositoryNotFoundError(Exception):
    pass


class RevisionNotFoundError(Exception):
    pass


class EntryNotFoundError(Exception):
    pass


class FakeHub:
    """Enough of ``HfApi``: dataset repos holding commits, each a snapshot of named files."""

    def __init__(self):
        self.commits: dict[str, dict[str, dict[str, bytes]]] = {}
        self.head: dict[str, str] = {}
        self.downloads: list[str] = []
        self.created: list[tuple[str, bool]] = []

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        assert repo_type == "dataset" and exist_ok
        self.created.append((repo_id, private))
        if repo_id not in self.head:
            self.commits[repo_id] = {"0": {}}
            self.head[repo_id] = "0"

    def upload_large_folder(self, *, repo_id, folder_path, repo_type):
        assert repo_type == "dataset"
        files = dict(self.commits[repo_id][self.head[repo_id]])
        for path in Path(folder_path).iterdir():
            if path.is_file():
                files[path.name] = path.read_bytes()
        sha = hashlib.sha1(json.dumps(sorted((k, len(v)) for k, v in files.items())).encode())
        sha = sha.hexdigest() + str(len(self.commits[repo_id]))
        self.commits[repo_id][sha] = files
        self.head[repo_id] = sha

    def _files(self, repo_id, revision):
        if repo_id not in self.head:
            raise RepositoryNotFoundError(repo_id)
        revision = revision if revision not in (None, "main") else self.head[repo_id]
        if revision not in self.commits[repo_id]:
            raise RevisionNotFoundError(revision)
        return revision, self.commits[repo_id][revision]

    def dataset_info(self, repo_id, revision=None):
        return SimpleNamespace(sha=self._files(repo_id, revision)[0])

    def list_repo_tree(self, repo_id, *, repo_type, revision, recursive, expand):
        assert repo_type == "dataset" and recursive and expand
        _, files = self._files(repo_id, revision)
        return [
            SimpleNamespace(
                path=name,
                size=len(data),
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
                if len(data) > LFS_FROM
                else None,
            )
            for name, data in files.items()
        ]

    def hf_hub_download(self, *, repo_id, filename, repo_type, revision, local_dir):
        assert repo_type == "dataset"
        _, files = self._files(repo_id, revision)
        if filename not in files:
            raise EntryNotFoundError(filename)
        self.downloads.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[filename])
        return str(target)
