"""Stage 1: real SDK export/privacy, no real model or Cloud traffic."""

import asyncio
from uuid import uuid4

import httpx
import pytest

from app.observability.tracing import ProductionTracing


@pytest.fixture
def tracing(monkeypatch):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()

    def build(**kwargs):
        return Langfuse(**kwargs, span_exporter=exporter)

    monkeypatch.setattr("langfuse.Langfuse", build)
    tracer = ProductionTracing(
        enabled=True, public_key=f"pk-lf-test-{uuid4().hex}", secret_key="test-only",
        base_url="http://127.0.0.1:9", environment="development",
    )
    assert tracer.enabled
    yield tracer, exporter
    asyncio.run(tracer.aclose())


@pytest.mark.parametrize("enabled,public,secret", [(False, "pk", "sk"), (True, "", "sk"), (True, "pk", "")])
def test_disabled_or_missing_keys_does_not_construct_sdk(monkeypatch, enabled, public, secret):
    def forbidden(**kwargs):
        pytest.fail("Disabled tracing must not construct a client")

    monkeypatch.setattr("langfuse.Langfuse", forbidden)
    tracer = ProductionTracing(enabled=enabled, public_key=public, secret_key=secret,
                               base_url="", environment="development")
    assert not tracer.enabled
    with tracer.span("noop") as span:
        span.update(status="success")
        assert span.trace_id is None
    asyncio.run(tracer.aclose())


def test_sdk_failures_never_repeat_or_swallow_business(monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("SDK unavailable")

    monkeypatch.setattr("langfuse.Langfuse", unavailable)
    tracer = ProductionTracing(enabled=True, public_key="pk", secret_key="sk",
                               base_url="", environment="development")
    calls = []
    with tracer.span("noop"):
        calls.append(1)
    with pytest.raises(ValueError, match="business"):
        with tracer.span("noop"):
            raise ValueError("business")
    assert calls == [1]


def test_nested_spans_and_exception_redaction(tracing):
    tracer, exporter = tracing
    with pytest.raises(ValueError, match="private-message"):
        with tracer.span("pickwise.test", request_id="req-test", password="never-send") as root:
            with tracer.span("tool.test") as child:
                child.update(result_count=2)
                raise ValueError("private-message")
    tracer.client.flush()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert spans["tool.test"].parent.span_id == int(root.span_id, 16)
    assert spans["tool.test"].context.trace_id == int(root.trace_id, 16)
    assert spans["tool.test"].attributes["langfuse.observation.metadata.result_count"] == 2
    assert spans["pickwise.test"].attributes["langfuse.observation.metadata.error_type"] == "ValueError"
    assert "private-message" not in "".join(span.to_json() for span in spans.values())
    assert "never-send" not in "".join(span.to_json() for span in spans.values())
    root.end()  # Repeated ending is harmless.


def test_parallel_requests_and_manual_root_propagation(tracing):
    tracer, exporter = tracing

    async def request(number):
        root = tracer.start(f"request-{number}")

        async def background():
            with root.activate(user_id=f"user-{number}", session_id=f"chat-{number}"):
                async def agent(name):
                    with tracer.span(name):
                        await asyncio.sleep(0)
                await asyncio.gather(agent(f"a-{number}"), agent(f"b-{number}"))
            root.end()

        await asyncio.create_task(background())
        return root

    async def run():
        return await asyncio.gather(request(1), request(2))

    roots = asyncio.run(run())
    tracer.client.flush()
    spans = {span.name: span for span in exporter.get_finished_spans()}
    assert roots[0].trace_id != roots[1].trace_id
    for number, root in enumerate(roots, 1):
        for name in (f"a-{number}", f"b-{number}"):
            assert spans[name].parent.span_id == int(root.span_id, 16)
            assert spans[name].attributes["session.id"] == f"chat-{number}"


def test_automatic_openai_generation_is_masked_but_usage_remains(tracing):
    from langfuse.openai import AsyncOpenAI

    tracer, exporter = tracing

    async def run():
        def response(request):
            return httpx.Response(200, json={
                "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": "test-model", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "PRIVATE_ANSWER"},
                }], "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            })

        async with AsyncOpenAI(api_key="not-a-real-key", base_url="https://mock.invalid/v1",
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(response))) as model:
            with tracer.span("pickwise.model-test") as root:
                await model.chat.completions.create(
                    model="test-model", messages=[{"role": "user", "content": "PRIVATE_PROMPT"}],
                    langfuse_public_key=tracer.public_key,
                )
        return root

    root = asyncio.run(run())
    tracer.client.flush()
    spans = exporter.get_finished_spans()
    generations = [s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"]
    assert len(generations) == 1
    generation = generations[0]
    assert generation.parent.span_id == int(root.span_id, 16)
    assert generation.attributes["langfuse.observation.model.name"] == "test-model"
    assert "11" in generation.attributes["langfuse.observation.usage_details"]
    raw = "".join(span.to_json() for span in spans)
    assert "PRIVATE_PROMPT" not in raw and "PRIVATE_ANSWER" not in raw


def test_evaluation_outside_production_scope_is_not_masked(tracing):
    tracer, exporter = tracing
    with tracer.span("pickwise.production"):
        pass
    with tracer.client.start_as_current_observation(name="eval.existing", input="EVAL_INPUT"):
        pass
    tracer.client.flush()
    span = next(s for s in exporter.get_finished_spans() if s.name == "eval.existing")
    assert "EVAL_INPUT" in span.attributes["langfuse.observation.input"]


def test_automatic_model_error_cannot_export_response_body(tracing):
    from langfuse.openai import AsyncOpenAI
    from openai import APIStatusError

    tracer, exporter = tracing

    async def run():
        transport = httpx.MockTransport(lambda request: httpx.Response(
            500, json={"error": {"message": "PRIVATE_ERROR_RESPONSE", "type": "test"}},
        ))
        async with AsyncOpenAI(api_key="not-real", base_url="https://mock.invalid/v1",
                               max_retries=0, http_client=httpx.AsyncClient(transport=transport)) as model:
            with tracer.span("pickwise.model-error"):
                await model.chat.completions.create(
                    model="test-model", messages=[{"role": "user", "content": "PRIVATE_ERROR_INPUT"}],
                    langfuse_public_key=tracer.public_key,
                )

    with pytest.raises(APIStatusError):
        asyncio.run(run())
    tracer.client.flush()
    spans = exporter.get_finished_spans()
    parent = next(s for s in spans if s.name == "pickwise.model-error")
    assert parent.attributes["langfuse.observation.metadata.error_type"] == "InternalServerError"
    raw = "".join(span.to_json() for span in spans)
    assert "PRIVATE_ERROR_RESPONSE" not in raw and "PRIVATE_ERROR_INPUT" not in raw
    # SDK 4.15.3 cannot mask status.description/events: these unsafe observations
    # are excluded, while the sanitized enclosing error observation is kept.
    assert len(spans) == 1


def test_observation_failures_do_not_break_business():
    from app.observability.tracing import Observation

    class BrokenObservation:
        trace_id = "invalid"
        id = "invalid"

        def update(self, **kwargs):
            raise RuntimeError("update failed")

        def end(self):
            raise RuntimeError("end failed")

    observation = Observation(BrokenObservation())
    calls = []
    with observation.activate():
        calls.append(1)
        observation.update(status="success")
    observation.end()
    assert calls == [1]


def test_flush_failure_is_non_fatal(tracing, monkeypatch):
    tracer, _ = tracing

    def unavailable():
        raise ConnectionError("export unavailable")

    monkeypatch.setattr(tracer.client, "flush", unavailable)
    asyncio.run(tracer.aclose())


@pytest.mark.parametrize("failure_stage", ["enter", "exit"])
@pytest.mark.parametrize("business_error", [False, True])
def test_sdk_context_failures_preserve_business_and_scope(
    tracing, monkeypatch, failure_stage, business_error,
):
    from contextlib import contextmanager

    from app.observability.tracing import mask_content, production_scope

    tracer, _ = tracing
    received_exceptions = []

    @contextmanager
    def broken_propagation(**kwargs):
        if failure_stage == "enter":
            raise RuntimeError("SDK context entry failed")
        try:
            yield
        except BaseException as exc:
            received_exceptions.append(exc)
            raise
        finally:
            if failure_stage == "exit":
                raise RuntimeError("SDK context exit failed")

    monkeypatch.setattr("langfuse.propagate_attributes", broken_propagation)
    business_exception = ValueError("private business error")
    calls = []

    def run():
        with tracer.span("context.failure"):
            assert production_scope.get() is True
            assert mask_content(data="PRIVATE") == "[content omitted]"
            calls.append(1)
            if business_error:
                raise business_exception

    if business_error:
        with pytest.raises(ValueError) as caught:
            run()
        assert caught.value is business_exception
    else:
        run()

    assert calls == [1]
    assert received_exceptions == []
    assert production_scope.get() is False
    assert mask_content(data="EVAL_INPUT") == "EVAL_INPUT"


def test_nested_scope_restores_after_start_and_update_failures(tracing, monkeypatch):
    from app.observability.tracing import production_scope

    tracer, _ = tracing
    with tracer.span("scope.outer") as root:
        with tracer.span("scope.inner"):
            assert production_scope.get() is True
        assert production_scope.get() is True

        def unavailable(**kwargs):
            assert production_scope.get() is False
            raise RuntimeError("SDK metadata write failed")

        monkeypatch.setattr(root.sdk, "update", unavailable)
        root.update(status="success")
        assert production_scope.get() is True

        monkeypatch.setattr(tracer.client, "start_observation", unavailable)
        with tracer.span("scope.unavailable") as child:
            assert child.trace_id is None
            assert production_scope.get() is True
        assert production_scope.get() is True

    assert production_scope.get() is False


def test_export_hooks_use_span_attributes_in_background_thread(tracing):
    from types import SimpleNamespace

    from app.observability.tracing import (
        mask_otel_spans, production_scope, set_production_scope, should_export_span,
    )

    spans = {
        "production": SimpleNamespace(
            attributes={
                "langfuse.observation.metadata.pickwise_source": "production",
                "langfuse.observation.input": "PRIVATE",
            },
            events=("PRIVATE_EXCEPTION",),
            status=SimpleNamespace(description=None),
        ),
        "evaluation": SimpleNamespace(
            attributes={"langfuse.observation.input": "EVAL_INPUT"},
            events=(),
            status=SimpleNamespace(description=None),
        ),
    }

    def export():
        assert production_scope.get() is False
        result = mask_otel_spans(SimpleNamespace(spans=spans))
        assert "langfuse.observation.input" in result.span_patches["production"].delete_attributes
        assert "evaluation" not in result.span_patches
        assert should_export_span(spans["production"]) is False
        assert should_export_span(spans["evaluation"]) is True

    async def run():
        with set_production_scope(False):
            await asyncio.to_thread(export)

    tracer, _ = tracing
    with tracer.span("production.request"):
        asyncio.run(run())
        assert production_scope.get() is True
    assert production_scope.get() is False
