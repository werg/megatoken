"""A provider's odd but real answers fail one attempt, not the collection run."""

import io
import json

import pytest

from megatoken.data import collection
from megatoken.data.collection import (
    MalformedTeacherResponse,
    OpenAICompatibleTeacher,
    TeacherRequest,
)
from megatoken.generation.codec import ByteCodec


def _answer(monkeypatch, body):
    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        collection, "open_url", lambda call, timeout: Reply(json.dumps(body).encode())
    )
    codec = ByteCodec.byte_fallback()
    teacher = OpenAICompatibleTeacher("http://localhost:1/v1/chat/completions", "m", codec)
    request = TeacherRequest("r1", "[]", "[]", 16, codec.fingerprint)
    return teacher.respond(request)


def test_null_tool_calls_and_usage_details_are_accepted(monkeypatch):
    body = {
        "id": "x",
        "choices": [{"message": {"content": "hi", "tool_calls": None}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        },
    }
    response = _answer(monkeypatch, body)
    assert response.calls == () and response.content == b"hi"


@pytest.mark.parametrize("arguments", ["", "not json", "[1, 2]"])
def test_invalid_tool_arguments_are_a_malformed_response(monkeypatch, arguments):
    call = {"id": "c", "function": {"name": "f", "arguments": arguments}}
    body = {"choices": [{"message": {"content": "", "tool_calls": [call]}}]}
    with pytest.raises(MalformedTeacherResponse):
        _answer(monkeypatch, body)
