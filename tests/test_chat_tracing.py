from contextlib import contextmanager
from contextvars import ContextVar

import pytest
from fastapi.testclient import TestClient

from app.api.testing import create_test_app
from app.services.idempotency import InMemoryIdempotencyStore, hash_request
from tests.in_memory_auth_store import InMemoryAuthStore
from tests.test_session_manager import FakeAgent, FactoryRecorder


active_trace = ContextVar("pickwise_test_trace", default=None)


class TraceRecord:
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields
        self.updates = []
        self.identity = None
        self.ended = False

    @contextmanager
    def activate(self, *, user_id=None, session_id=None):
        self.identity = (user_id, session_id)
        token = active_trace.set(self)
        try:
            yield self
        finally:
            active_trace.reset(token)

    def update(self, **fields):
        self.updates.append(fields)

    def end(self):
        self.ended = True


class TraceRecorder:
    def __init__(self):
        self.records = []

    def start(self, name, **fields):
        record = TraceRecord(name, fields)
        self.records.append(record)
        return record

    async def aclose(self):
        pass


class TraceAwareAgent(FakeAgent):
    async def chat(self, message):
        self.active_trace = active_trace.get()
        return await super().chat(message)


@pytest.mark.parametrize(
    ("stream", "path"),
    [(False, "/api/chat"), (True, "/api/chat/stream")],
)
def test_chat_root_trace_covers_execution_and_idempotent_replay(
    tmp_path, monkeypatch, stream, path,
):
    tracing = TraceRecorder()
    monkeypatch.setattr("app.api.app.ProductionTracing", lambda **kwargs: tracing)
    factory = FactoryRecorder(lambda identity: TraceAwareAgent(identity))
    app = create_test_app(
        session_dir=tmp_path,
        factory=factory,
        idempotency_store=InMemoryIdempotencyStore(),
        auth_store=InMemoryAuthStore(),
    )

    with TestClient(app) as client:
        registration = client.post("/api/auth/register", json={
            "email": "trace@example.com",
            "password": "test-password-123",
            "display_name": "Trace",
        }).json()["data"]
        user = {**registration["user"], "csrf_token": registration["csrf_token"]}
        headers = {
            "X-Pickwise-CSRF": user["csrf_token"],
            "Idempotency-Key": " trace-request-001 ",
        }
        session_id = client.post(
            "/api/sessions", json={}, headers=headers,
        ).json()["data"]["session_id"]
        request_body = {
            "session_id": session_id,
            "message": "trace-private-user-message",
        }

        first = client.post(path, json=request_body, headers=headers)
        replay = client.post(path, json=request_body, headers=headers)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert len(tracing.records) == 2

    first_trace, replay_trace = tracing.records
    assert first_trace.name == "pickwise.chat"
    assert first_trace.fields["mode"] == ("stream" if stream else "sync")
    assert first_trace.fields["message_length"] == len(request_body["message"])
    assert first_trace.identity == (user["user_id"], session_id)
    assert first_trace.ended
    assert first_trace.updates[-1]["status"] == "success"
    assert first_trace.updates[-1]["answer_length"] > 0

    agent = factory.agents[f"{user['user_id']}:{session_id}"]
    assert agent.active_trace is first_trace
    assert agent.chat_total == 1

    assert replay_trace.ended
    assert replay_trace.updates[-1]["replayed"] is True
    assert replay_trace.updates[-1]["execution_status"] == "skipped"
    assert replay_trace.updates[-1]["idempotency_status"] == "replay"

    recorded_data = repr([
        (record.fields, record.updates) for record in tracing.records
    ])
    assert request_body["message"] not in recorded_data
    assert "reply:trace-private-user-message" not in recorded_data


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize(
    "scenario,expected_http,expected_status,expected_idempotency,expected_execution",
    [
        ("invalid", 422, "rejected", "invalid", "not_started"),
        ("conflict", 409, "rejected", "conflict", "not_started"),
        ("in_progress", 409, "rejected", "in_progress", "not_started"),
        ("begin_unavailable", 503, "error", "unavailable", "not_started"),
        ("finish_unavailable", 200, "success", "persistence_failed", "success"),
        ("answer_failure", 503, "error", "finished", "failed"),
        ("execution_error", 500, "error", "finished", "failed"),
    ],
)
def test_chat_errors_preserve_response_and_trace(
    tmp_path, monkeypatch, path, scenario, expected_http,
    expected_status, expected_idempotency, expected_execution,
):
    import asyncio
    import json

    tracing = TraceRecorder()
    monkeypatch.setattr("app.api.app.ProductionTracing", lambda **kwargs: tracing)
    store = InMemoryIdempotencyStore()

    def make_agent(identity):
        agent = TraceAwareAgent(identity)
        if scenario == "answer_failure":
            agent.last_failures = [{"stage": "agent", "affects_answer": True}]
        elif scenario == "execution_error":
            agent.chat_error = RuntimeError("private-execution-error")
        return agent

    factory = FactoryRecorder(make_agent)
    app = create_test_app(
        session_dir=tmp_path, factory=factory,
        idempotency_store=store, auth_store=InMemoryAuthStore(),
    )
    with TestClient(app) as client:
        registration = client.post("/api/auth/register", json={
            "email": "trace-errors@example.com",
            "password": "test-password-123",
            "display_name": "Trace",
        }).json()["data"]
        headers = {"X-Pickwise-CSRF": registration["csrf_token"]}
        session_id = client.post(
            "/api/sessions", json={}, headers=headers,
        ).json()["data"]["session_id"]
        body = {"session_id": session_id, "message": "private-message"}
        key = "chat-error-001"
        headers["Idempotency-Key"] = key

        if scenario == "invalid":
            headers["Idempotency-Key"] = " "
        elif scenario in {"conflict", "in_progress"}:
            fingerprint = hash_request({**body, "product_refs": []})
            if scenario == "conflict":
                fingerprint = "different-request"
            asyncio.run(store.begin(
                registration["user"]["user_id"], session_id,
                key, fingerprint, ttl_seconds=60,
            ))
        elif scenario in {"begin_unavailable", "finish_unavailable"}:
            async def unavailable(*args, **kwargs):
                raise RuntimeError("private-storage-error")

            method = "begin" if scenario == "begin_unavailable" else "finish"
            monkeypatch.setattr(store, method, unavailable)

        response = client.post(path, json=body, headers=headers)

        if path.endswith("/stream") and scenario in {
            "finish_unavailable", "answer_failure", "execution_error",
        }:
            assert response.status_code == 200
            frames = []
            for block in response.text.split("\n\n"):
                if block.startswith("event: done\n"):
                    frames.append(json.loads(block.split("data: ", 1)[1]))
            assert len(frames) == 1
            assert frames[0]["failed"] is (expected_http >= 400)
            if scenario == "finish_unavailable":
                assert "IDEMPOTENCY_PERSISTENCE_FAILED" in frames[0]["warnings"]
        else:
            assert response.status_code == expected_http
            if scenario == "finish_unavailable":
                assert response.headers["X-Idempotency-Persisted"] == "false"

        if scenario in {"answer_failure", "execution_error"}:
            replay = client.post(path, json=body, headers=headers)
            assert replay.status_code == expected_http
            assert replay.headers["X-Idempotent-Replay"] == "true"
            assert tracing.records[-1].updates[-1]["replayed"] is True
            assert tracing.records[-1].updates[-1]["execution_status"] == "skipped"
            assert next(iter(factory.agents.values())).chat_total == 1

    trace = tracing.records[0]
    assert trace.ended
    fields = trace.updates[-1]
    assert fields["status"] == expected_status
    assert fields["idempotency_status"] == expected_idempotency
    assert fields["execution_status"] == expected_execution
    if scenario in {"invalid", "conflict", "in_progress", "begin_unavailable"}:
        assert factory.calls == []
    if scenario == "in_progress":
        assert response.headers["Retry-After"] == "1"
    assert "private-execution-error" not in repr(trace.updates)
    assert "private-storage-error" not in repr(trace.updates)


def test_sse_disconnect_keeps_trace_open_until_background_save_finishes(tmp_path, monkeypatch):
    import asyncio
    from uuid import uuid4

    from fastapi import FastAPI, Request

    from app.api.auth import AuthenticatedUser
    from app.api.routes.chat import _handle_chat
    from app.api.schemas import ChatRequest
    from app.services.session_manager import SessionManager

    async def run():
        execution_gate = asyncio.Event()
        save_started = asyncio.Event()
        save_gate = asyncio.Event()
        tracing = TraceRecorder()
        store = InMemoryIdempotencyStore()
        original_finish = store.finish

        async def finish_after_gate(*args, **kwargs):
            save_started.set()
            await save_gate.wait()
            await original_finish(*args, **kwargs)

        monkeypatch.setattr(store, "finish", finish_after_gate)

        class StreamingAgent(TraceAwareAgent):
            async def chat(self, message, events=None):
                events.put_nowait({"event": "status", "data": {"stage": "agent"}})
                return await super().chat(message)

        factory = FactoryRecorder(lambda identity: StreamingAgent(
            identity, chat_gate=execution_gate,
        ))
        manager = SessionManager(session_dir=tmp_path, factory=factory)
        user_id = str(uuid4())
        identity = await manager.create_session(user_id)
        user = AuthenticatedUser(user_id, "disconnect@example.com", "Trace")
        body = ChatRequest(session_id=identity.session_id, message="private-message")
        app = FastAPI()
        app.state.tracing = tracing
        app.state.idempotency_store = store
        app.state.stream_tasks = set()
        request = Request({"type": "http", "app": app})
        task = None
        response = None
        try:
            response = await _handle_chat(
                body, request, user, manager, "disconnect-001", stream=True,
            )
            task = next(iter(app.state.stream_tasks))
            frame = await asyncio.wait_for(anext(response.body_iterator), timeout=3)
            assert "event: status" in frame

            # 关闭 SSE 迭代器模拟浏览器断连，业务任务仍须继续。
            await response.body_iterator.aclose()
            assert not task.done()
            assert not tracing.records[0].ended

            execution_gate.set()
            await asyncio.wait_for(save_started.wait(), timeout=3)
            assert not tracing.records[0].ended
            save_gate.set()
            await asyncio.wait_for(task, timeout=3)

            assert tracing.records[0].ended
            assert tracing.records[0].updates[-1]["idempotency_status"] == "finished"
            agent = next(iter(factory.agents.values()))
            assert agent.active_trace is tracing.records[0]
            fingerprint = hash_request({
                "session_id": body.session_id,
                "message": body.message,
                "product_refs": [],
            })
            claim = await store.begin(
                user_id, body.session_id, "disconnect-001", fingerprint, ttl_seconds=60,
            )
            assert claim.action == "replay"
        finally:
            execution_gate.set()
            save_gate.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if response is not None:
                await response.body_iterator.aclose()
            await manager.close()

    asyncio.run(run())
