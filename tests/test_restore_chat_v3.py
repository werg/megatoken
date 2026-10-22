import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/restore_chat_v3_prompts.py"
spec = importlib.util.spec_from_file_location("restore_chat_v3_prompts", SCRIPT)
restorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restorer)


def _row(prompt_hash, *, system=True):
    messages = [{"role": "user", "content": None}, {"role": "assistant", "content": "answer"}]
    turns = [False, True]
    if system:
        messages.insert(0, {"role": "system", "content": None})
        turns.insert(0, False)
    metadata = {"seed_dataset": "lmsys/lmsys-chat-1m", "seed_prompt_sha256": prompt_hash}
    return {"messages": messages, "metadata": {**metadata, "train_turns": turns}}


def test_restores_found_prompts_and_drops_the_rest(tmp_path):
    found_hash, missing_hash = restorer.digest("hello"), restorer.digest("gone")
    own = {"messages": [{"role": "user", "content": "kept"}], "metadata": {}}
    rows = [_row(found_hash), _row(missing_hash), own, _row(found_hash, system=False)]
    source = tmp_path / "chat.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert restorer.needed_hashes(source)["lmsys"] == {found_hash, missing_hash}
    out = tmp_path / "out.jsonl"
    counts = restorer.restore(source, out, {found_hash: (None, "hello")})
    assert counts == {"rows": 4, "restored": 2, "dropped": 1}
    restored = [json.loads(line) for line in out.read_text().splitlines()]
    # The source had no system turn: the empty one and its mask entry are gone.
    assert restored[0]["messages"][0] == {"role": "user", "content": "hello"}
    assert restored[0]["metadata"]["train_turns"] == [False, True]
    assert restored[1] == own
    assert restored[2]["messages"][0]["content"] == "hello"
