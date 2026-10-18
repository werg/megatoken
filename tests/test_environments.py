import json

import pytest

from megatoken.environments import FixtureEnvironment, HTTPEnvironment


def test_execution_verification_and_idempotency():
    environment = FixtureEnvironment()
    initial = environment.reset({"expected_files": {"answer.txt": "blue"}})
    assert b'"blue"' not in initial.observation  # Hidden expected files are not serialized.
    action = b'{"name":"write","arguments":{"path":"answer.txt","content":"blue"}}'
    result = environment.step(action, action_id="write-1", expected_state_id=initial.state_id)
    assert environment.files["answer.txt"] == "blue"
    assert (
        environment.step(action, action_id="write-1", expected_state_id=initial.state_id) == result
    )
    assert environment.sequence == 1
    with pytest.raises(ValueError, match="different bytes"):
        environment.step(b"{}", action_id="write-1", expected_state_id=initial.state_id)
    with pytest.raises(ValueError, match="stale"):
        environment.step(action, action_id="write-2", expected_state_id=initial.state_id)
    verified = environment.verify()
    assert verified.verified and verified.reward == 1.0
    assert b"blue" not in verified.observation


def test_invalid_actions_do_not_execute_host_commands_or_escape():
    environment = FixtureEnvironment()
    state = environment.reset({})
    invalid = json.dumps(
        {"name": "write", "arguments": {"path": "../../outside", "content": "oops"}}
    ).encode()
    result = environment.step(invalid, action_id="invalid", expected_state_id=state.state_id)
    assert b"error" in result.observation
    assert environment.files == {"answer.txt": "red"}
    state = result
    result = environment.step(
        b'{"name":"shell","arguments":{"cmd":"touch /tmp/outside"}}',
        action_id="shell",
        expected_state_id=state.state_id,
    )
    assert b"Unknown tool" in result.observation
    assert environment.verify().reward == 0.0


def test_remote_client_requires_secure_explicit_endpoint():
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPEnvironment("http://example.com")
    with pytest.raises(ValueError, match="credentials"):
        HTTPEnvironment("https://user:secret@example.com")
    HTTPEnvironment("http://127.0.0.1:8001")


@pytest.mark.parametrize("tools", ["[3]", '[{"function":3}]', "[{}]"])
def test_public_tool_objects_are_validated(tools):
    from megatoken.tool_schema import compile_tool_spec

    with pytest.raises(ValueError, match="object|definition"):
        compile_tool_spec(tools)
