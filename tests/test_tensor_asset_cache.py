"""TensorAssetStore reads and checksums a file once, and still fails closed on change."""

import hashlib
import os

import numpy as np
import pytest

from megatoken.data.media_assets import TensorAssetStore
from megatoken.teacher import PINNED_TEACHER
from tests.media_support import features


@pytest.fixture
def store(tmp_path):
    return TensorAssetStore(tmp_path)


def put(store, seed=0):
    return store.put(
        features(6, seed=seed)[0].numpy(),
        kind="features",
        modality="text",
        extractor=PINNED_TEACHER,
        source_family="a" * 64,
    )


def test_a_file_is_checksummed_once_and_each_load_is_a_private_copy(store, monkeypatch):
    asset = put(store)
    reads = []
    original = hashlib.sha256

    def counting(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr("megatoken.data.media_assets.hashlib.sha256", counting)
    first = store.load(asset)
    second = store.load(asset)
    third = store.load(asset)
    assert len(reads) == 1
    np.testing.assert_array_equal(first, second)
    first[:] = 0  # mutating one caller's copy cannot corrupt the next
    np.testing.assert_array_equal(store.load(asset), second)
    assert third is not second


def test_a_changed_file_is_never_served_from_the_cache(store):
    asset = put(store)
    store.load(asset)
    path = store.root / asset.relative_path
    stat = path.stat()
    path.write_bytes(bytes(stat.st_size))  # same size, different bytes
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="differs"):
        store.load(asset)


def test_the_cache_is_bounded_by_bytes(store, monkeypatch):
    monkeypatch.setattr(TensorAssetStore, "CACHE_BYTES", 2 * 6 * 512 * 4)
    assets = [put(store, seed) for seed in range(4)]
    for asset in assets:
        store.load(asset)
    assert len(store._cache) == 2 and store._cached_bytes == 2 * 6 * 512 * 4
    for asset in assets:  # an evicted asset is simply read again
        assert store.load(asset).shape == (6, 512)
