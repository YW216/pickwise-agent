"""生产聊天的 Agent 调用树回归；只使用本地模拟模型和内存导出器。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.observability.tracing import (
    current_observation, production_scope, production_span,
)
from tests.test_production_tracing import tracing  # noqa: F401


@pytest.fixture
def model_run(tracing, tmp_path):
    """执行真实编排器，让 SDK 自动生成模型节点，禁止访问远端服务。"""
    tracer, exporter = tracing

    async def run(
        scenarios="presale,consult", *, failures=(), result_empty=False,
        production=True, requests=1, invalid_route=False, max_steps=5,
        tool_turn=False, tool_result='{"success": true, "data": {"product": {}}}',
        child_sdk_failure=False, router_unavailable=False, memory_failure=False,
    ):
        from langfuse.openai import AsyncOpenAI

        attempts = {}
        entered = {}
        roots = []
        orchs = []
        generation_count = 0

        async def response(request):
            nonlocal generation_count
            generation_count += 1
            observation = current_observation.get()
            if observation is None:
                # 非生产测试用各自的模型消息识别请求类型。
                messages = json.loads(request.content)["messages"]
                if messages[0]["role"] == "user":
                    name = "router"
                elif "各专家工作记录" in messages[-1]["content"]:
                    name = "agent.result"
                else:
                    name = "agent.presale"
                request_number = "offline"
            else:
                name = observation.sdk._otel_span.name
                request_number = observation.sdk._otel_span.attributes.get("session.id", "unknown")

            key = (request_number, name)
            attempts[key] = attempts.get(key, 0) + 1
            if name in {"agent.presale", "agent.consult"}:
                entered.setdefault(request_number, set()).add(name)
                # 两个专家必须都进入模型请求，才能继续，验证实际协程重叠。
                if scenarios == "presale,consult" and not invalid_route and not router_unavailable:
                    while len(entered[request_number]) < 2:
                        await asyncio.sleep(0)
                else:
                    await asyncio.sleep(0)
            if name == "router" and router_unavailable:
                return httpx.Response(500, json={"error": {
                    "message": "PRIVATE_ROUTER_FAILURE", "type": "server_error",
                }})
            if name in failures:
                return httpx.Response(400, json={"error": {
                    "message": "PRIVATE_MODEL_FAILURE", "type": "invalid_request_error",
                }})

            content = "PRIVATE_ANSWER"
            if name == "router":
                content = "PRIVATE_INVALID_ROUTE" if invalid_route else scenarios
            elif name == "agent.result" and result_empty:
                content = ""
            message = {"role": "assistant", "content": content}
            finish_reason = "stop"
            if tool_turn and name == "agent.presale" and attempts[key] == 1:
                finish_reason = "tool_calls"
                message["tool_calls"] = [{
                    "id": "private-tool-call", "type": "function",
                    "function": {
                        "name": "get_detail",
                        "arguments": '{"secret":"PRIVATE_TOOL_ARGUMENT"}',
                    },
                }]
            return httpx.Response(200, json={
                "id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{
                    "index": 0, "finish_reason": finish_reason, "message": message,
                }],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            })

        async with AsyncOpenAI(
            api_key="test-only", base_url="https://mock.invalid/v1", max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(response)),
        ) as client:
            original = client.chat.completions.create

            async def create(**kwargs):
                return await original(**kwargs, langfuse_public_key=tracer.public_key)

            client.chat.completions.create = create

            async def chat(number):
                orch = MultiAgentOrchestrator(
                    client=client, session_path=str(tmp_path / f"session-{number}.json"),
                    memory_dir=str(tmp_path / "memory"), memory_enabled=False,
                    skills_enabled=False, mcp_enabled=False, persistence_enabled=False,
                )
                orch.compaction_enabled = False
                orch.max_react_steps = max_steps
                if memory_failure:
                    orch.memory_manager.memory_enabled = True
                    orch.memory_manager.update_short_term = AsyncMock(
                        side_effect=RuntimeError("PRIVATE_MEMORY_FAILURE"),
                    )
                if tool_turn:
                    orch.agents["presale"].tool_manager.aexecute_call_as_message = AsyncMock(
                        return_value=tool_result,
                    )
                orchs.append(orch)
                if not production:
                    return await orch.chat("PRIVATE_PROMPT")
                with tracer.span("pickwise.chat", user_id=f"user-{number}", session_id=f"session-{number}") as root:
                    roots.append(root)
                    if child_sdk_failure:
                        original_start = tracer.client.start_observation

                        def unavailable(**kwargs):
                            raise RuntimeError("SDK span unavailable")

                        tracer.client.start_observation = unavailable
                        try:
                            return await orch.chat("PRIVATE_PROMPT")
                        finally:
                            tracer.client.start_observation = original_start
                    return await orch.chat("PRIVATE_PROMPT")

            replies = await asyncio.wait_for(
                asyncio.gather(*(chat(number) for number in range(requests))), timeout=10,
            )
        tracer.client.flush()
        return SimpleNamespace(
            replies=replies, spans=exporter.get_finished_spans(),
            roots=roots, orchs=orchs, attempts=attempts, generation_count=generation_count,
        )

    return run


@pytest.mark.parametrize("scenarios", ["presale", "presale,consult"])
def test_real_agent_tree_and_automatic_generations(model_run, scenarios):
    result = asyncio.run(model_run(scenarios))
    spans = {span.name: span for span in result.spans if span.name != "OpenAI-generation"}
    root = spans["pickwise.chat"]
    names = {"router", "agent.presale", "memory.update"}
    if "," in scenarios:
        names.update({"agent.consult", "agent.result"})
    assert set(spans) == names | {"pickwise.chat"}
    for name in names:
        assert spans[name].parent.span_id == root.context.span_id
        assert spans[name].context.trace_id == root.context.trace_id
        assert spans[name].attributes["session.id"] == "session-0"
        expected_status = "not_executed" if name == "memory.update" else "success"
        assert spans[name].attributes["langfuse.observation.metadata.status"] == expected_status
        if name != "memory.update":
            assert spans[name].attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    for key in scenarios.split(","):
        assert spans[f"agent.{key}"].attributes["langfuse.observation.metadata.react_steps"] == 1
        assert spans[f"agent.{key}"].attributes["langfuse.observation.metadata.mode"] == (
            "multi" if "," in scenarios else "single"
        )
    assert spans["router"].attributes["langfuse.observation.metadata.scenarios"] == scenarios
    generations = [s for s in result.spans if s.attributes.get("langfuse.observation.type") == "generation"]
    model_spans = names - {"memory.update"}
    assert len(generations) == result.generation_count == len(model_spans)
    assert {g.parent.span_id for g in generations} == {spans[name].context.span_id for name in model_spans}
    for generation in generations:
        assert generation.attributes["langfuse.observation.model.name"] == "test-model"
        assert "11" in generation.attributes["langfuse.observation.usage_details"]
        assert generation.end_time >= generation.start_time
    raw = "".join(s.to_json() for s in result.spans)
    assert "PRIVATE_PROMPT" not in raw and "PRIVATE_ANSWER" not in raw
    assert current_observation.get() is None and production_scope.get() is False


def test_parallel_requests_keep_experts_in_their_own_trace(model_run):
    result = asyncio.run(model_run(requests=2))
    roots = [s for s in result.spans if s.name == "pickwise.chat"]
    assert len(roots) == 2 and roots[0].context.trace_id != roots[1].context.trace_id
    for root in roots:
        children = [s for s in result.spans if s.parent and s.parent.span_id == root.context.span_id]
        assert {s.name for s in children} == {
            "router", "agent.presale", "agent.consult", "agent.result", "memory.update",
        }
        assert all(s.attributes["session.id"] == root.attributes["session.id"] for s in children)
        experts = [s for s in children if s.name in {"agent.presale", "agent.consult"}]
        assert max(s.start_time for s in experts) < min(s.end_time for s in experts)
    assert len(result.spans) == 20


@pytest.mark.parametrize("failure", ["agent.presale", "agent.result"])
def test_partial_expert_or_result_failure_keeps_answer_and_records_fallback(model_run, failure):
    result = asyncio.run(model_run(failures=(failure,)))
    spans = {s.name: s for s in result.spans}
    assert result.replies[0].strip()
    assert spans[failure].attributes["langfuse.observation.metadata.error_type"] == "BadRequestError"
    assert spans["agent.result"].attributes["langfuse.observation.metadata.status"] == "degraded"
    raw = "".join(s.to_json() for s in result.spans)
    assert "PRIVATE_MODEL_FAILURE" not in raw and "PRIVATE_ANSWER" not in raw


def test_invalid_router_output_marks_default_route_as_degraded(model_run):
    result = asyncio.run(model_run(invalid_route=True))
    router = next(s for s in result.spans if s.name == "router")
    assert router.attributes["langfuse.observation.metadata.scenarios"] == "consult"
    assert router.attributes["langfuse.observation.metadata.status"] == "degraded"
    assert "PRIVATE_INVALID_ROUTE" not in router.to_json()


@pytest.mark.parametrize("max_steps, expected_degraded", [(1, True), (5, False)])
def test_actual_react_steps_include_final_forced_answer(model_run, max_steps, expected_degraded):
    result = asyncio.run(model_run("presale", tool_turn=True, max_steps=max_steps))
    agent = next(s for s in result.spans if s.name == "agent.presale")
    assert agent.attributes["langfuse.observation.metadata.react_steps"] == 2
    assert agent.attributes["langfuse.observation.metadata.finish_reasons"] == "tool_calls,stop"
    assert agent.attributes["langfuse.observation.metadata.status"] == (
        "degraded" if expected_degraded else "success"
    )
    assert result.orchs[0].agents["presale"].tool_manager.aexecute_call_as_message.await_count == 1


def test_tool_span_records_failure_envelope_without_arguments_or_result(model_run):
    result = asyncio.run(model_run(
        "presale",
        tool_turn=True,
        tool_result=(
            '{"success": false, "error": "PRIVATE_TOOL_RESULT", '
            '"data": {"result_count": 3}}'
        ),
    ))
    spans = {span.name: span for span in result.spans}
    tool = spans["tool.get_detail"]
    agent = spans["agent.presale"]
    assert tool.parent.span_id == agent.context.span_id
    assert tool.attributes["langfuse.observation.metadata.status"] == "failed"
    assert tool.attributes["langfuse.observation.metadata.error_type"] == "ToolResultFailure"
    assert tool.attributes["langfuse.observation.metadata.tool"] == "get_detail"
    assert tool.attributes["langfuse.observation.metadata.agent"] == "presale"
    assert tool.attributes["langfuse.observation.metadata.tool_call_id"] == "private-tool-call"
    assert agent.attributes["langfuse.observation.metadata.status"] == "degraded"
    assert spans["pickwise.chat"].attributes[
        "langfuse.observation.metadata.failure_stage"
    ] == "tool.get_detail"
    assert spans["pickwise.chat"].attributes[
        "langfuse.observation.metadata.error_type"
    ] == "ToolResultFailure"
    raw = "".join(span.to_json() for span in result.spans)
    assert "PRIVATE_TOOL_ARGUMENT" not in raw
    assert "PRIVATE_TOOL_RESULT" not in raw
    assert result.orchs[0].agents["presale"].tool_manager.aexecute_call_as_message.await_count == 1


def test_memory_failure_degrades_execution_without_losing_answer(model_run):
    result = asyncio.run(model_run("presale", memory_failure=True))
    spans = {span.name: span for span in result.spans}
    assert result.replies == ["PRIVATE_ANSWER"]
    assert spans["memory.update"].attributes["langfuse.observation.metadata.status"] == "failed"
    assert spans["memory.update"].attributes[
        "langfuse.observation.metadata.error_type"
    ] == "RuntimeError"
    assert spans["pickwise.chat"].attributes["langfuse.observation.metadata.status"] == "degraded"
    assert spans["pickwise.chat"].attributes[
        "langfuse.observation.metadata.failure_stage"
    ] == "memory.update"
    assert "PRIVATE_MEMORY_FAILURE" not in "".join(span.to_json() for span in result.spans)


def test_chat_outcome_separates_degraded_execution_from_valid_answer():
    from app.services.session_manager import SessionManager

    class Agent:
        last_failures = [{
            "stage": "agent:consult",
            "error_type": "AgentExecutionError",
            "affects_answer": True,
        }]
        last_execution_status = "degraded"
        last_answer_status = "success"

        async def chat(self, message):
            return "usable partial answer"

    outcome = asyncio.run(SessionManager.__new__(SessionManager)._chat_agent(Agent(), "question"))
    assert outcome.failed is True
    assert outcome.execution_status == "degraded"
    assert outcome.answer_status == "success"
    assert outcome.failure_stage == "agent:consult"
    assert outcome.error_type == "AgentExecutionError"


@pytest.mark.parametrize("save_failure", [False, True])
def test_session_restore_and_save_are_observed_separately(tracing, tmp_path, save_failure):
    from datetime import datetime, timezone

    from app.identity import SessionIdentity
    from app.services.session_manager import SessionManager
    from app.services.session_store import SessionState

    tracer, exporter = tracing
    identity = SessionIdentity(
        user_id="22222222-2222-4222-8222-222222222222",
        session_id="11111111-1111-4111-8111-111111111111",
    )
    now = datetime.now(timezone.utc)

    class SessionService:
        def __init__(self):
            self.state = SessionState(
                user_id=identity.user_id, session_id=identity.session_id,
                title="Trace", summary=None, short_term_memory={}, products={},
                version=1, status="active", created_at=now, updated_at=now,
                last_message_at=None,
            )

        async def create_session(self, user_id):
            return self.state

        async def get_session(self, user_id, session_id):
            return self.state

        async def save_session(self, state, *, expected_version=None):
            if save_failure:
                raise RuntimeError("PRIVATE_SESSION_SAVE_FAILURE")
            self.state = state
            return state

    class Agent:
        last_failures = []
        last_execution_status = "success"
        last_answer_status = "success"
        summary = None
        products = {}
        recommendations = []
        raw_messages = []

        class Memory:
            def stm_to_dict(self):
                return {}

            def restore_stm(self, data):
                pass

        memory_manager = Memory()

        async def chat(self, message):
            self.raw_messages.extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": "saved answer"},
            ])
            return "saved answer"

    manager = SessionManager(
        session_dir=tmp_path, factory=lambda _: Agent(),
        session_service=SessionService(),
    )

    async def run():
        with tracer.span("pickwise.chat", user_id=identity.user_id, session_id=identity.session_id):
            return await manager.chat(identity, "question")

    outcome = asyncio.run(run())
    tracer.client.flush()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans["pickwise.chat"]
    restore = spans["conversation.restore"]
    save = spans["conversation.save"]
    assert restore.parent.span_id == root.context.span_id
    assert restore.attributes["langfuse.observation.metadata.status"] == "success"
    assert save.parent.span_id == root.context.span_id
    if save_failure:
        assert outcome.persisted is False
        assert outcome.answer_status == "success"
        assert outcome.execution_status == "degraded"
        assert save.attributes["langfuse.observation.metadata.status"] == "failed"
        assert save.attributes["langfuse.observation.metadata.persistence_status"] == "failed"
        assert save.attributes["langfuse.observation.metadata.error_type"] == "RuntimeError"
        assert root.attributes["langfuse.observation.metadata.status"] == "degraded"
        raw = "".join(span.to_json() for span in exporter.get_finished_spans())
        assert "PRIVATE_SESSION_SAVE_FAILURE" not in raw
    else:
        assert outcome.persisted is True and outcome.answer_status == "success"
        assert save.attributes["langfuse.observation.metadata.persistence_status"] == "success"


def test_idempotency_check_records_execute_and_replay_without_key(tracing):
    from types import SimpleNamespace
    from uuid import uuid4

    from app.api.routes.chat import _check_chat_idempotency
    from app.identity import SessionIdentity
    from app.services.idempotency import InMemoryIdempotencyStore

    tracer, exporter = tracing
    identity = SessionIdentity(user_id=str(uuid4()), session_id=str(uuid4()))
    body = SimpleNamespace(
        session_id=identity.session_id,
        message="PRIVATE_IDEMPOTENCY_MESSAGE",
        product_refs=[],
    )
    store = InMemoryIdempotencyStore()

    async def run():
        with tracer.span("pickwise.chat", user_id=identity.user_id, session_id=identity.session_id):
            with production_span(
                "idempotency.check", status="success", idempotency_status="execute",
            ) as observation:
                first = await _check_chat_idempotency(
                    store, identity, body, "PRIVATE_IDEMPOTENCY_KEY", "request-1", stream=False,
                )
                observation.update(idempotency_status=first.status)
            await store.finish(
                identity.user_id,
                identity.session_id,
                "PRIVATE_IDEMPOTENCY_KEY",
                first.request_hash,
                status="succeeded",
                status_code=200,
                response={"success": True, "data": {"reply": "private"}},
            )

            with production_span(
                "idempotency.check", status="success", idempotency_status="replay",
            ) as observation:
                replay = await _check_chat_idempotency(
                    store, identity, body, "PRIVATE_IDEMPOTENCY_KEY", "request-2", stream=False,
                )
                observation.update(idempotency_status=replay.status)

    asyncio.run(run())
    tracer.client.flush()
    spans = [span for span in exporter.get_finished_spans() if span.name == "idempotency.check"]
    assert [span.attributes["langfuse.observation.metadata.idempotency_status"] for span in spans] == [
        "execute", "replay",
    ]
    assert all(span.attributes["langfuse.observation.metadata.status"] == "success" for span in spans)
    raw = "".join(span.to_json() for span in exporter.get_finished_spans())
    assert "PRIVATE_IDEMPOTENCY_MESSAGE" not in raw
    assert "PRIVATE_IDEMPOTENCY_KEY" not in raw


def test_offline_eval_context_has_no_production_spans_or_masking(model_run, tracing):
    tracer, _ = tracing
    with tracer.client.start_as_current_observation(name="eval.existing"):
        result = asyncio.run(model_run("presale", production=False))
    tracer.client.flush()
    assert all(s.name not in {"router", "agent.presale", "agent.result"} for s in result.spans)
    generations = [s for s in result.spans if s.attributes.get("langfuse.observation.type") == "generation"]
    assert len(generations) == 2
    assert "PRIVATE_PROMPT" in "".join(s.to_json() for s in generations)
    assert current_observation.get() is None and production_scope.get() is False


def test_child_span_creation_failure_does_not_repeat_execution(model_run):
    result = asyncio.run(model_run("presale", child_sdk_failure=True))
    assert result.replies == ["PRIVATE_ANSWER"]
    assert result.generation_count == 2
    assert len(result.orchs[0].raw_messages) == 2
    assert current_observation.get() is None and production_scope.get() is False


def test_cancelled_agent_span_closes_and_keeps_cancellation(tracing):
    tracer, exporter = tracing

    async def run():
        with tracer.span("pickwise.chat"):
            with pytest.raises(asyncio.CancelledError):
                with production_span("agent.presale", agent="presale"):
                    raise asyncio.CancelledError()

    asyncio.run(run())
    tracer.client.flush()
    child = next(s for s in exporter.get_finished_spans() if s.name == "agent.presale")
    assert child.attributes["langfuse.observation.metadata.status"] == "cancelled"
    assert current_observation.get() is None


def test_transient_router_failure_degrades_without_extra_model_retry(model_run):
    result = asyncio.run(model_run(router_unavailable=True))
    router = next(s for s in result.spans if s.name == "router")
    assert router.attributes["langfuse.observation.metadata.scenarios"] == "consult"
    assert router.attributes["langfuse.observation.metadata.error_type"] == "InternalServerError"
    assert result.generation_count == 2
    assert "PRIVATE_ROUTER_FAILURE" not in "".join(s.to_json() for s in result.spans)


def test_all_experts_failed_skips_result_model_but_ends_result_span(model_run):
    result = asyncio.run(model_run(failures=("agent.presale", "agent.consult")))
    span = next(s for s in result.spans if s.name == "agent.result")
    assert span.attributes["langfuse.observation.metadata.status"] == "degraded"
    assert result.generation_count == 3
    assert result.replies[0].strip()
    assert "PRIVATE_MODEL_FAILURE" not in "".join(s.to_json() for s in result.spans)


def test_empty_result_uses_expert_conclusions_and_marks_failure(model_run):
    result = asyncio.run(model_run(result_empty=True))
    span = next(s for s in result.spans if s.name == "agent.result")
    assert span.attributes["langfuse.observation.metadata.status"] == "degraded"
    assert span.attributes["langfuse.observation.metadata.error_type"] == "ValueError"
    assert "PRIVATE_ANSWER" in result.replies[0]
    assert "PRIVATE_ANSWER" not in "".join(s.to_json() for s in result.spans)


def test_real_expert_cancellation_escapes_and_restores_context(tracing):
    from app.multi_agent.context_pack import ContextPack

    tracer, exporter = tracing
    orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    orch.skill_manager = None
    orch.disable_context_projection = False
    orch.max_react_steps = 5
    handle = AsyncMock(side_effect=asyncio.CancelledError())
    orch.agents = {"presale": SimpleNamespace(handle=handle)}
    pack = ContextPack(history=[], summary=None, memory_sections=[], skill_catalog="")

    async def run():
        with tracer.span("pickwise.chat"):
            with pytest.raises(asyncio.CancelledError):
                await orch._execute_agents(["presale"], pack, "single")

    asyncio.run(run())
    tracer.client.flush()
    child = next(s for s in exporter.get_finished_spans() if s.name == "agent.presale")
    assert child.attributes["langfuse.observation.metadata.status"] == "cancelled"
    assert handle.await_count == 1
    assert current_observation.get() is None and production_scope.get() is False


def test_model_finish_reasons_only_record_protocol_values(tracing):
    tracer, exporter = tracing
    with tracer.span("pickwise.chat"):
        with production_span("agent.presale") as observation:
            observation.update(finish_reason="tool_calls")
            observation.update(finish_reason="PRIVATE_UNEXPECTED_VALUE")
            observation.update(finish_reason="stop")
    tracer.client.flush()
    child = next(s for s in exporter.get_finished_spans() if s.name == "agent.presale")
    assert child.attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    assert child.attributes["langfuse.observation.metadata.finish_reasons"] == "tool_calls,stop"
    assert "PRIVATE_UNEXPECTED_VALUE" not in child.to_json()
