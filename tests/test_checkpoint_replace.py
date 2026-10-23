"""Re-reaching a committed checkpoint's step replaces it without ever losing the last good one."""

import json
import shutil

import pytest
import torch

from megatoken.training.checkpoint import resolve_checkpoint, save_checkpoint


def _save(root, marker, *, pinned=False):
    return save_checkpoint(
        root, "P1-100", {"marker": torch.tensor([marker])}, {"global_step": 100}, pinned=pinned
    )


def _marker(directory):
    return int(torch.load(directory / "state.pt", weights_only=True)["marker"][0])


def test_a_second_save_of_the_same_step_replaces_the_first(tmp_path):
    _save(tmp_path, 1)
    _save(tmp_path, 2)
    assert _marker(resolve_checkpoint(tmp_path)) == 2
    assert not list(tmp_path.glob(".superseded-*"))
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_a_crash_after_setting_the_old_one_aside_restores_it(tmp_path):
    live = _save(tmp_path, 1)
    shutil.move(live, tmp_path / ".superseded-P1-100-7")  # the swap began, nothing replaced it
    assert _marker(resolve_checkpoint(tmp_path)) == 1
    assert (tmp_path / "P1-100").is_dir() and not list(tmp_path.glob(".superseded-*"))


def test_a_crash_before_the_pointer_moved_restores_the_old_one(tmp_path):
    live = _save(tmp_path, 1)
    old = tmp_path / ".superseded-P1-100-7"
    shutil.copytree(live, old)
    torch.save({"marker": torch.tensor([2])}, live / "state.pt")  # a different, newer content
    metadata = json.loads((live / "metadata.json").read_text())
    metadata["state_sha256"] = "changed"
    (live / "metadata.json").write_text(json.dumps(metadata))
    assert _marker(resolve_checkpoint(tmp_path)) == 1  # the pointer still names the old one
    assert not list(tmp_path.glob(".superseded-*"))


def test_a_pinned_checkpoint_is_not_replaced_by_an_unpinned_one(tmp_path):
    _save(tmp_path, 1, pinned=True)
    with pytest.raises(FileExistsError, match="pinned"):
        _save(tmp_path, 2)
    assert _marker(resolve_checkpoint(tmp_path)) == 1


@pytest.mark.parametrize("field", ["pinned", "format_version"])
def test_replacement_does_not_guess_missing_saved_fields(tmp_path, field):
    live = _save(tmp_path, 1, pinned=True)
    path = live / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.pop(field)
    path.write_text(json.dumps(metadata))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="complete current metadata"):
        _save(tmp_path, 2)
    assert path.read_bytes() == before and _marker(live) == 1
    assert not list(tmp_path.glob(".superseded-*"))
