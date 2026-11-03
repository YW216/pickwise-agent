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
        tool_turn=False, child_sdk_failure=False, router_unavailable=False,
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
                    "function": {"name": "get_detail", "arguments": "{}"},
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
                if tool_turn:
                    orch.agents["presale"].tool_manager.aexecute_call_as_message = AsyncMock(
                        return_value='{"success": true, "data": {"product": {}}}',
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
    names = {"router", "agent.presale"}
    if "," in scenarios:
        names.update({"agent.consult", "agent.result"})
    assert set(spans) == names | {"pickwise.chat"}
    for name in names:
        assert spans[name].parent.span_id == root.context.span_id
        assert spans[name].context.trace_id == root.context.trace_id
        assert spans[name].attributes["session.id"] == "session-0"
        assert spans[name].attributes["langfuse.observation.metadata.status"] == "success"
        assert spans[name].attributes["langfuse.observation.metadata.finish_reason"] == "stop"
    for key in scenarios.split(","):
        assert spans[f"agent.{key}"].attributes["langfuse.observation.metadata.react_steps"] == 1
        assert spans[f"agent.{key}"].attributes["langfuse.observation.metadata.mode"] == (
            "multi" if "," in scenarios else "single"
        )
    assert spans["router"].attributes["langfuse.observation.metadata.scenarios"] == scenarios
    generations = [s for s in result.spans if s.attributes.get("langfuse.observation.type") == "generation"]
    assert len(generations) == result.generation_count == len(names)
    assert {g.parent.span_id for g in generations} == {spans[name].context.span_id for name in names}
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
        assert {s.name for s in children} == {"router", "agent.presale", "agent.consult", "agent.result"}
        assert all(s.attributes["session.id"] == root.attributes["session.id"] for s in children)
        experts = [s for s in children if s.name in {"agent.presale", "agent.consult"}]
        assert max(s.start_time for s in experts) < min(s.end_time for s in experts)
    assert len(result.spans) == 18


@pytest.mark.parametrize("failure", ["agent.presale", "agent.result"])
def test_partial_expert_or_result_failure_keeps_answer_and_records_fallback(model_run, failure):
    result = asyncio.run(model_run(failures=(failure,)))
    spans = {s.name: s for s in result.spans}
    assert result.replies[0].strip()
    assert spans[failure].attributes["langfuse.observation.metadata.error_type"] == "BadRequestError"
    assert spans["agent.result"].attributes["langfuse.observation.metadata.degraded"] is True
    assert spans["agent.result"].attributes["langfuse.observation.metadata.status"] == "degraded"
    raw = "".join(s.to_json() for s in result.spans)
    assert "PRIVATE_MODEL_FAILURE" not in raw and "PRIVATE_ANSWER" not in raw


def test_invalid_router_output_marks_default_route_as_degraded(model_run):
    result = asyncio.run(model_run(invalid_route=True))
    router = next(s for s in result.spans if s.name == "router")
    assert router.attributes["langfuse.observation.metadata.scenarios"] == "consult"
    assert router.attributes["langfuse.observation.metadata.degraded"] is True
    assert router.attributes["langfuse.observation.metadata.status"] == "degraded"
    assert "PRIVATE_INVALID_ROUTE" not in router.to_json()


@pytest.mark.parametrize("max_steps, expected_degraded", [(1, True), (5, False)])
def test_actual_react_steps_include_final_forced_answer(model_run, max_steps, expected_degraded):
    result = asyncio.run(model_run("presale", tool_turn=True, max_steps=max_steps))
    agent = next(s for s in result.spans if s.name == "agent.presale")
    assert agent.attributes["langfuse.observation.metadata.react_steps"] == 2
    assert agent.attributes["langfuse.observation.metadata.finish_reasons"] == "tool_calls,stop"
    assert agent.attributes["langfuse.observation.metadata.degraded"] is expected_degraded
    assert result.orchs[0].agents["presale"].tool_manager.aexecute_call_as_message.await_count == 1


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
    assert router.attributes["langfuse.observation.metadata.degraded"] is True
    assert router.attributes["langfuse.observation.metadata.error_type"] == "InternalServerError"
    assert result.generation_count == 2
    assert "PRIVATE_ROUTER_FAILURE" not in "".join(s.to_json() for s in result.spans)


def test_all_experts_failed_skips_result_model_but_ends_result_span(model_run):
    result = asyncio.run(model_run(failures=("agent.presale", "agent.consult")))
    span = next(s for s in result.spans if s.name == "agent.result")
    assert span.attributes["langfuse.observation.metadata.status"] == "degraded"
    assert span.attributes["langfuse.observation.metadata.degraded"] is True
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
