"""LLM 韧性的边界回归：假 client + 临时目录，不访问模型 API。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config.settings import settings
from app.multi_agent.blackboard import BlackboardEntry
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.router import DEFAULT_SCENARIO


@pytest.fixture
def orch(tmp_path):
    agent = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="merged answer"))])))))
    agent.model = "fake"
    agent.temperature = 0.0
    agent.session_path = str(tmp_path / "session.json")
    for key in (
        "context_window", "reserve_tokens", "keep_recent_tokens",
        "summary_max_tokens", "summary_max_chars", "tool_result_max_chars",
        "max_react_steps", "max_user_input_tokens",
    ):
        setattr(agent, key, getattr(settings, key))
    agent.compaction_enabled = False
    agent.raw_messages = []
    agent.summary = None
    agent.products = {}
    agent.router = SimpleNamespace(route=Mock(return_value=["presale"]))
    agent.memory_manager = SimpleNamespace(
        build_memory_prompt_sections=Mock(return_value=[]),
        update_short_term=Mock(), stm_to_dict=Mock(return_value={}),
        consolidate_to_long_term=Mock(),
    )
    agent.skill_manager = SimpleNamespace(enabled=False)
    agent.agents = {}
    agent._execute_agents = Mock(return_value=[_entry("presale", "valid answer")])
    return agent


def _entry(agent, content):
    return BlackboardEntry(agent=agent, status="success", error=None,
                           new_messages=[{"role": "assistant", "content": content}])


def test_memory_failure_keeps_completed_reply(orch):
    orch.memory_manager.update_short_term.side_effect = RuntimeError("memory broke")
    assert orch.chat("question") == "valid answer"
    assert [m["role"] for m in orch.raw_messages] == ["user", "assistant"]
    assert orch.raw_messages[-1]["content"] == "valid answer"


def test_save_failure_keeps_reply_without_retry_loop(orch, monkeypatch):
    save = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr("app.multi_agent.orchestrator.save_session", save)
    assert orch.chat("question") == "valid answer"
    assert save.call_count == 1
    assert [m["role"] for m in orch.raw_messages] == ["user", "assistant"]


def test_failed_request_and_failed_save_still_return_fallback(orch, monkeypatch):
    orch.router.route.side_effect = ValueError("invalid model parameter")
    save = Mock(side_effect=OSError("disk full"))
    monkeypatch.setattr("app.multi_agent.orchestrator.save_session", save)
    reply = orch.chat("question")
    assert reply and "invalid model parameter" not in reply
    assert [m["role"] for m in orch.raw_messages] == ["user", "assistant"]
    assert save.call_count == 1


def test_failure_before_user_is_admitted_does_not_add_orphan(orch, monkeypatch):
    monkeypatch.setattr("app.multi_agent.orchestrator.estimate_text_tokens",
                        Mock(side_effect=TypeError("bad input")))
    assert orch.chat("question")
    assert orch.raw_messages == []


@pytest.mark.parametrize("content", ["", "   "])
def test_empty_result_reply_falls_back_to_existing_conclusions(orch, content):
    orch.client.chat.completions.create.return_value.choices[0].message.content = content
    reply = orch._run_result_agent(
        [_entry("presale", "product conclusion"), _entry("consult", "policy conclusion")],
        "question",
    )
    assert "product conclusion" in reply and "policy conclusion" in reply


def test_router_transient_fallback_really_executes_default(orch):
    from openai import APIConnectionError
    orch.router.route.side_effect = APIConnectionError(request=None, message="network")
    assert orch.chat("question") == "valid answer"
    assert orch._execute_agents.call_args.args[0] == [DEFAULT_SCENARIO]


def test_router_overflow_then_transient_can_still_degrade(orch):
    import httpx
    from openai import APIConnectionError, BadRequestError
    request = httpx.Request("POST", "https://test.invalid/chat")
    overflow = BadRequestError(
        "maximum context length exceeded", response=httpx.Response(400, request=request),
        body={"code": "context_length_exceeded"},
    )
    orch.router.route.side_effect = [overflow, APIConnectionError(request=request)]
    orch._try_compact = Mock(side_effect=[False, True])
    assert orch.chat("question") == "valid answer"
    assert orch.router.route.call_count == 2
    assert orch._execute_agents.call_args.args[0] == [DEFAULT_SCENARIO]


def test_memory_failure_is_diagnostic_not_answer_failure(orch, capsys):
    orch.memory_manager.update_short_term.side_effect = RuntimeError("memory broke")
    assert orch.chat("question") == "valid answer"
    assert orch.last_failures == [{
        "stage": "memory", "error_type": "RuntimeError", "message": "memory broke",
        "affects_answer": False,
    }]
    assert capsys.readouterr().err == ""


def test_failures_reset_between_turns_and_save_can_recover(orch, monkeypatch):
    import app.multi_agent.orchestrator as module
    from app.agent.storage import load_session
    original = module.save_session
    monkeypatch.setattr(module, "save_session", Mock(side_effect=OSError("disk full")))
    assert orch.chat("first question") == "valid answer"
    assert orch.last_failures[0]["stage"] == "persistence"
    monkeypatch.setattr(module, "save_session", original)
    assert orch.chat("second question") == "valid answer"
    assert orch.last_failures == []
    restored = load_session(orch.session_path)
    assert [m["role"] for m in restored["messages"]] == ["user", "assistant"] * 2


def test_failure_keeps_existing_summary_and_products(orch):
    from app.agent.storage import load_session
    orch.summary = "existing summary"
    orch.products = {"LP-03": {"product_id": "LP-03", "price_at_mention": 1000}}
    orch.router.route.side_effect = ValueError("invalid parameter")
    assert orch.chat("question")
    restored = load_session(orch.session_path)
    assert restored["summary"] == "existing summary"
    assert restored["products"] == orch.products


def test_result_length_finish_reason_uses_existing_conclusions(orch):
    choice = orch.client.chat.completions.create.return_value.choices[0]
    choice.finish_reason = "length"
    choice.message.content = "truncated conclusion"
    reply = orch._run_result_agent(
        [_entry("presale", "product conclusion"), _entry("consult", "policy conclusion")],
        "question",
    )
    assert "product conclusion" in reply and "policy conclusion" in reply
    assert "truncated conclusion" not in reply
    assert orch.last_failures[-1]["stage"] == "result"


def test_all_agents_failed_skips_result_llm(orch):
    entries = [BlackboardEntry(agent=key, status="failed", error="private error",
                               new_messages=[]) for key in ("presale", "consult")]
    reply = orch._run_result_agent(entries, "question")
    assert "private error" not in reply
    assert "PickWise-售前" in reply and "PickWise-咨询" in reply
    orch.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("messages", [[], [{"role": "assistant", "content": " "}]])
def test_invalid_agent_final_reply_is_not_success(orch, monkeypatch, messages):
    orch.agents = {"presale": SimpleNamespace(
        name="fake", handle=Mock(return_value=("", messages)),
    )}
    monkeypatch.setattr(orch, "_execute_agents",
                        MultiAgentOrchestrator._execute_agents.__get__(orch))
    reply = orch.chat("question")
    assert reply and [m["role"] for m in orch.raw_messages] == ["user", "assistant"]
    failure = next(f for f in orch.last_failures if f["stage"] == "agent:presale")
    assert failure["error_type"] == "ValueError"


def test_program_bug_remains_diagnosable(orch, monkeypatch, capsys):
    orch.agents = {"presale": SimpleNamespace(
        name="fake", handle=Mock(side_effect=TypeError("program bug")),
    )}
    monkeypatch.setattr(orch, "_execute_agents",
                        MultiAgentOrchestrator._execute_agents.__get__(orch))
    reply = orch.chat("question")
    assert "program bug" not in reply
    assert any(f["error_type"] == "TypeError" for f in orch.last_failures)
    assert capsys.readouterr().err == ""


def test_close_cleans_tools_even_if_memory_or_other_cleanup_fails(orch):
    first_close = Mock(side_effect=RuntimeError("tool close broke"))
    second_close = Mock()
    orch.memory_manager.consolidate_to_long_term.side_effect = RuntimeError("memory broke")
    orch.client.close = Mock()
    orch.agents = {
        "presale": SimpleNamespace(tool_manager=SimpleNamespace(close=first_close)),
        "consult": SimpleNamespace(tool_manager=SimpleNamespace(close=second_close)),
    }
    orch.close()
    first_close.assert_called_once()
    second_close.assert_called_once()
    orch.client.close.assert_called_once()
    assert {f["stage"] for f in orch.last_failures} == {"long_term_memory", "tool_close"}


def test_stage_parser_does_not_match_arbitrary_error_text():
    assert MultiAgentOrchestrator._failed_agent_key(RuntimeError(
        "unexpected text containing [presale]")) is None


def test_sandbox_captures_failure_even_when_chat_returns_text(orch, tmp_path, monkeypatch):
    from app.evaluation.dataset import EvalCase
    from app.evaluation.sandbox import Sandbox
    orch.router.route.side_effect = ValueError("invalid parameter")
    sandbox = Sandbox(tmp_root=str(tmp_path / "sandbox"))
    monkeypatch.setattr(sandbox, "_build_agent", lambda session_path: orch)
    monkeypatch.setattr(sandbox, "_instrument", lambda *args: None)
    trace = sandbox.run(EvalCase(id="failure", category="test", description="failure",
                                 turns=["question"]))
    assert trace.error is None  # graceful fallback, not an uncaught exception
    assert trace.replies and trace.runtime_failures
    assert trace.runtime_failures[0]["stage"] == "chat"
    assert trace.runtime_failures[0]["turn_index"] == 0
    assert trace.to_dict()["runtime_failures"] == trace.runtime_failures


@pytest.mark.parametrize("stage, affects_answer", [
    ("chat", True), ("agent:presale", True), ("result", True),
    ("router", True), ("memory", False), ("persistence", False),
])
def test_evaluator_does_not_mistake_fallback_for_normal_success(stage, affects_answer):
    from app.evaluation.dataset import EvalCase
    from app.evaluation.evaluator import Evaluator
    from app.evaluation.trace import RunTrace
    trace = RunTrace(case_id="test", turns=["question"], replies=["some nonempty text"],
                     runtime_failures=[{
                         "stage": stage, "error_type": "RuntimeError", "message": "simulated",
                         "affects_answer": affects_answer, "turn_index": 0,
                     }])
    sandbox = SimpleNamespace(run=lambda case: trace)
    result = Evaluator(sandbox).run_case(EvalCase(
        id="test", category="test", description="test", turns=["question"],
    ))
    assert result.passed is not affects_answer
    check = next(c for c in result.checks if c.name == "runtime")
    assert check.passed is not affects_answer


@pytest.mark.parametrize("max_steps", [0, 1])
@pytest.mark.parametrize("content, finish_reason", [("", "stop"), (" ", "stop"), ("partial", "length")])
def test_subagent_rejects_empty_or_truncated_final_reply(orch, max_steps, content, finish_reason):
    from app.multi_agent.agents import SubAgent
    choice = orch.client.chat.completions.create.return_value.choices[0]
    choice.message.content = content
    choice.message.tool_calls = None
    choice.finish_reason = finish_reason
    agent = SubAgent(name="fake", tool_manager=SimpleNamespace(tool_definitions=[]),
                     client=orch.client, model="fake", temperature=0.0)
    with pytest.raises(ValueError, match="最终答复"):
        agent.handle([{"role": "user", "content": "question"}], max_steps=max_steps)


def test_sandbox_closes_http_client_without_consolidating_memory(orch, tmp_path, monkeypatch):
    from app.evaluation.dataset import EvalCase
    from app.evaluation.sandbox import Sandbox
    orch.client.close = Mock()
    sandbox = Sandbox(tmp_root=str(tmp_path / "sandbox"))
    monkeypatch.setattr(sandbox, "_build_agent", lambda session_path: orch)
    monkeypatch.setattr(sandbox, "_instrument", lambda *args: None)
    trace = sandbox.run(EvalCase(id="test", category="test", description="test",
                                 turns=["question"]))
    assert trace.replies == ["valid answer"]
    orch.client.close.assert_called_once()
    orch.memory_manager.consolidate_to_long_term.assert_not_called()
