from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.testing import create_test_app
from app.config.settings import settings
from app.services.idempotency import InMemoryIdempotencyStore, hash_request
from tests.in_memory_auth_store import InMemoryAuthStore
from tests.test_session_manager import FakeAgent, FactoryRecorder


active_trace = ContextVar("pickwise_test_trace", default=None)


class TraceRecord:
    def __init__(self, name, fields):
        self.name = name
        self.fields = fields
        self.trace_id = uuid4().hex
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
    assert replay_trace.updates[-1]["execution_status"] == "success"
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
        ("invalid", 422, "rejected", "invalid", "rejected"),
        ("conflict", 409, "rejected", "conflict", "rejected"),
        ("in_progress", 409, "rejected", "in_progress", "rejected"),
        ("begin_unavailable", 503, "error", "unavailable", "failed"),
        ("finish_unavailable", 200, "success", "persistence_failed", "degraded"),
        ("answer_failure", 503, "error", "execute", "failed"),
        ("execution_error", 500, "error", "execute", "failed"),
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
            replay_status = "success" if expected_http < 500 else "failed"
            assert tracing.records[-1].updates[-1]["execution_status"] == replay_status
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
            assert tracing.records[0].updates[-1]["idempotency_status"] == "execute"
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


# ============================================================
# X-Trace-Id 回传：三条响应路径 + 浏览器可读性
# ============================================================
#
# 根节点的 trace_id 只落在 Langfuse 侧没有意义——用户报障时要能把
# 一次请求对应到一条 trace。回传必须覆盖三条出口：普通响应、SSE
# （头要在首个事件之前到）、以及执行前就被拒绝的请求（那些请求恰恰
# 最需要 trace）。跨域场景另有一条隐藏前提：没有
# Access-Control-Expose-Headers，浏览器读不到自定义头。

TRACE_ORIGIN = "http://localhost:5173"


def _login_with_session(client, email):
    """注册并建一个窗口，返回写请求所需的 CSRF 头与窗口编号。"""
    registration = client.post("/api/auth/register", json={
        "email": email,
        "password": "test-password-123",
        "display_name": "Trace",
    }).json()["data"]
    headers = {"X-Pickwise-CSRF": registration["csrf_token"]}
    session_id = client.post(
        "/api/sessions", json={}, headers=headers,
    ).json()["data"]["session_id"]
    return registration["user"], headers, session_id


def _tracing_app(tmp_path, monkeypatch):
    """建一个注入假 TraceRecorder 的隔离应用，返回 (app, tracing)。"""
    tracing = TraceRecorder()
    monkeypatch.setattr("app.api.app.ProductionTracing", lambda **kwargs: tracing)
    app = create_test_app(
        session_dir=tmp_path,
        factory=FactoryRecorder(lambda identity: TraceAwareAgent(identity)),
        idempotency_store=InMemoryIdempotencyStore(),
        auth_store=InMemoryAuthStore(),
    )
    return app, tracing


@pytest.mark.parametrize(
    ("stream", "path"),
    [(False, "/api/chat"), (True, "/api/chat/stream")],
)
def test_chat_response_returns_trace_id_header(tmp_path, monkeypatch, stream, path):
    """普通响应与 SSE 响应都必须带回本次 trace 的编号。"""
    app, tracing = _tracing_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        _, headers, session_id = _login_with_session(
            client, f"trace-header-{stream}@example.com",
        )
        response = client.post(
            path,
            json={"session_id": session_id, "message": "hello"},
            headers=headers,
        )

    assert response.status_code == 200
    trace = tracing.records[0]
    assert trace.ended
    assert response.headers["X-Trace-Id"] == trace.trace_id


def test_sse_trace_id_header_arrives_before_first_event(tmp_path, monkeypatch):
    """SSE 的头必须先于首个事件到达，否则前端只能在流结束后才拿到编号。"""
    app, tracing = _tracing_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        _, headers, session_id = _login_with_session(client, "trace-sse@example.com")
        with client.stream(
            "POST", "/api/chat/stream",
            json={"session_id": session_id, "message": "hello"},
            headers=headers,
        ) as response:
            # 读任何正文之前先断言头：这正是「首字之前」的含义。
            assert response.headers["X-Trace-Id"] == tracing.records[0].trace_id
            frames = [line for line in response.iter_lines() if line.strip()]

    assert frames[0].startswith("event:")


def test_cross_origin_response_exposes_trace_id_header(tmp_path, monkeypatch):
    """跨域下自定义头必须被 expose，否则浏览器读不到（前端静默拿空）。"""
    monkeypatch.setattr(settings, "allowed_origins", [TRACE_ORIGIN])
    app, tracing = _tracing_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        _, headers, session_id = _login_with_session(client, "trace-cors@example.com")
        response = client.post(
            "/api/chat",
            json={"session_id": session_id, "message": "hello"},
            headers={**headers, "Origin": TRACE_ORIGIN},
        )

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == TRACE_ORIGIN
    exposed = response.headers["Access-Control-Expose-Headers"].lower()
    assert "x-trace-id" in exposed
    assert response.headers["X-Trace-Id"] == tracing.records[0].trace_id


def test_rejected_chat_still_returns_trace_id_header(tmp_path, monkeypatch):
    """执行前就被拒绝的请求同样要能按 trace 编号追溯。

    必须用「长度合法、但不是 UUID」的编号：schema 有 36 位长度约束，
    短字符串会在进入 handler 之前就被 FastAPI 挡下（那时还没有 trace）。
    """
    app, tracing = _tracing_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        _, headers, _ = _login_with_session(client, "trace-rejected@example.com")
        response = client.post(
            "/api/chat",
            json={"session_id": "x" * 36, "message": "hello"},
            headers=headers,
        )

    assert response.status_code == 422
    trace = tracing.records[0]
    assert trace.ended
    assert response.headers["X-Trace-Id"] == trace.trace_id
    assert trace.updates[-1]["failure_stage"] == "identity"
    assert trace.updates[-1]["execution_status"] == "rejected"
    assert trace.updates[-1]["idempotency_status"] == "not_used"
