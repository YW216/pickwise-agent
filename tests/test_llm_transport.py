"""真实 SDK 的传输契约回归，HTTP 使用 MockTransport，不访问外网。"""

import json
from unittest.mock import Mock

import httpx
import pytest
from openai import APIStatusError, OpenAI

from app.agent.context_budget import is_context_overflow, is_transient
from app.agent.rag.embedder import Embedder


@pytest.mark.parametrize("status, expected", [
    (400, False), (401, False), (403, False), (404, False), (422, False),
    (408, True), (409, True), (429, True), (500, True), (502, True), (503, True),
])
def test_transient_http_status_is_not_program_error(status, expected):
    request = httpx.Request("POST", "https://test.invalid/chat")
    exc = APIStatusError("failed", response=httpx.Response(status, request=request), body={})
    assert is_transient(exc) is expected


def test_overflow_is_separate_from_transient():
    from openai import BadRequestError
    request = httpx.Request("POST", "https://test.invalid/chat")
    exc = BadRequestError("maximum context length exceeded",
                          response=httpx.Response(400, request=request),
                          body={"code": "context_length_exceeded"})
    assert is_context_overflow(exc)
    assert not is_transient(exc)


@pytest.mark.parametrize("status, attempts", [(500, 3), (429, 3), (400, 1)])
def test_sdk_retry_count_and_request_body(status, attempts, monkeypatch):
    from openai import _base_client
    monkeypatch.setattr(_base_client.time, "sleep", lambda seconds: None)
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(status, json={"error": {"message": "simulated"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        with OpenAI(api_key="test-key", base_url="https://test.invalid/v1",
                    http_client=http, timeout=7.5, max_retries=2) as client:
            with pytest.raises(APIStatusError):
                client.chat.completions.create(
                    model="fake", messages=[{"role": "user", "content": "question"}],
                )
    assert len(bodies) == attempts
    assert all(body == bodies[0] for body in bodies)


def test_sdk_retry_does_not_execute_local_write_tool(monkeypatch):
    """HTTP 500 后重试，仅成功返回一次 tool_call；SDK 不执行应用函数。"""
    from openai import _base_client
    monkeypatch.setattr(_base_client.time, "sleep", lambda seconds: None)
    write_tool = Mock()
    attempts = []

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(500, json={"error": {"message": "simulated"}})
        return httpx.Response(200, json={
            "id": "fake-response", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-final", "type": "function", "function": {
                        "name": "write_order", "arguments": "{}",
                    },
                }],
            }}],
        })

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        with OpenAI(api_key="test-key", base_url="https://test.invalid/v1",
                    http_client=http, timeout=7.5, max_retries=2) as client:
            response = client.chat.completions.create(model="fake", messages=[])
            write_tool.assert_not_called()
            for call in response.choices[0].message.tool_calls:
                write_tool(call.function.arguments)
    assert len(attempts) == 2
    write_tool.assert_called_once_with("{}")


def test_embedding_client_honors_transport_configuration(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("app.agent.rag.embedder.OpenAI", factory)
    Embedder(api_key="test-key", base_url="https://test.invalid",
             timeout=13.5, max_retries=1)
    factory.assert_called_once_with(
        api_key="test-key", base_url="https://test.invalid", timeout=13.5, max_retries=1,
    )
