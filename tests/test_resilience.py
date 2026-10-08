"""故障降级路径的回归锁（零 LLM 调用，全部用假 client 注入故障）。

覆盖 2026-10-08 新增的三条降级路径：
1. Router 瞬时故障 → 就地降级到默认场景，请求不中断
2. Result Agent 任意故障 → 拼接黑板已有结论，成果不丢
3. chat() 出口 → 补assistant 配对 + 落盘 + 按失败环节给不同话术

以及错误分类边界：400 类 / 程序 bug 不进降级（宁可少降级，不把 bug 藏起来）。
这些是「服务不可用时的行为契约」——重构时最容易 inadvertent 删掉的就是它。
"""

import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError, RateLimitError

from app.agent.context_budget import is_transient
from app.config.settings import settings
from app.multi_agent.agents import AGENT_CONFIGS
from app.multi_agent.blackboard import BlackboardEntry
from app.multi_agent.orchestrator import MultiAgentOrchestrator
from app.multi_agent.router import DEFAULT_SCENARIO


class _Resp:
    def __init__(self, content):
        self.choices = [
            type("C", (), {"message": type("M", (), {"content": content})()})()
        ]


class _Completions:
    def __init__(self, mode):
        self.mode = mode

    def create(self, **kwargs):
        if self.mode == "fail":
            raise APIConnectionError(request=None, message="simulated outage")
        return _Resp("presale,consult")


def _client(mode="ok"):
    return type("Cl", (), {"chat": type("C", (), {"completions": _Completions(mode)})()})()


def _build(tmp_path, fail_mode="ok"):
    """构造一个全部字段就位的编排器（绕开 __init__ 避免真连服务）。"""
    o = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
    o.client = _client(fail_mode)
    o.model, o.temperature = "fake", 0.0
    o.context_window = settings.context_window
    o.reserve_tokens = settings.reserve_tokens
    o.compaction_enabled = False
    o.keep_recent_tokens = settings.keep_recent_tokens
    o.summary_max_tokens = settings.summary_max_tokens
    o.summary_max_chars = settings.summary_max_chars
    o.tool_result_max_chars = settings.tool_result_max_chars
    o.max_react_steps = 3
    o.max_user_input_tokens = settings.max_user_input_tokens
    o.session_path = str(tmp_path / "session.json")
    o.raw_messages, o.summary, o.products = [], None, {}
    return o


# ---------- 错误分类边界 ----------

def test_transient_classification_boundary():
    """只有瞬时类进降级；400 类与程序 bug 不进。

    宁可少降级：把程序 bug 当成瞬时故障降级，会把 bug 藏起来更难查。
    """
    assert is_transient(APIConnectionError(request=None, message="x")) is True
    assert is_transient(RuntimeError("bug")) is False
    assert is_transient(ValueError("bug")) is False
    request = httpx.Request("POST", "https://test.invalid/chat")
    rate_limit = RateLimitError("limited", response=httpx.Response(429, request=request), body={})
    bad_request = BadRequestError("invalid", response=httpx.Response(400, request=request), body={})
    assert is_transient(rate_limit) is True
    assert is_transient(APITimeoutError(request=request)) is True
    assert is_transient(bad_request) is False


# ---------- 路径 1：Router 瞬时故障就地降级 ----------

def test_router_transient_degrades_in_place(tmp_path, monkeypatch):
    """真正验证默认场景被执行且返回专家答复，而不是泛泛检查有兜底文本。"""
    from types import SimpleNamespace
    from unittest.mock import Mock

    o = _build(tmp_path)
    o.router = SimpleNamespace(route=Mock(side_effect=APIConnectionError(
        request=None, message="router down")))
    o.memory_manager = SimpleNamespace(
        build_memory_prompt_sections=lambda: [], update_short_term=lambda *a: None,
        stm_to_dict=lambda: {},
    )
    o.skill_manager = SimpleNamespace(enabled=False)
    execution = Mock(return_value=[BlackboardEntry(
        agent=DEFAULT_SCENARIO, status="success", error=None,
        new_messages=[{"role": "assistant", "content": "expert answer"}],
    )])
    monkeypatch.setattr(o, "_execute_agents", execution)
    assert o.chat("推荐耳机") == "expert answer"
    assert o.router.route.call_count == 1
    assert execution.call_args.args[0] == [DEFAULT_SCENARIO]
    assert any(f["stage"] == "router" for f in o.last_failures)


def test_router_default_scenario_is_valid():
    """降级用的默认场景必须是合法场景名（否则下游 KeyError）。"""
    assert DEFAULT_SCENARIO in AGENT_CONFIGS


# ---------- 路径 2：Result Agent 失败拼接结论 ----------

def test_result_agent_failure_keeps_agent_output(tmp_path):
    """Result 调 LLM 失败 → 直接拼接黑板成果，不抛、不丢。"""
    o = _build(tmp_path, fail_mode="fail")
    entries = [
        BlackboardEntry(agent="presale", status="success", error=None,
                        new_messages=[{"role": "assistant", "content": "推荐 HP-01，1260 元。"}]),
        BlackboardEntry(agent="consult", status="success", error=None,
                        new_messages=[{"role": "assistant", "content": "保修一年。"}]),
    ]
    reply = o._run_result_agent(entries, "推荐耳机")
    assert "HP-01" in reply and "保修一年" in reply, f"成果被丢弃: {reply}"
    assert "无法回答" not in reply  # 两项都成功，不该出现失败提示


def test_result_agent_failure_marks_failed_entry(tmp_path):
    """部分失败时，失败项如实告知、不冒充成功。"""
    o = _build(tmp_path, fail_mode="fail")
    entries = [
        BlackboardEntry(agent="presale", status="success", error=None,
                        new_messages=[{"role": "assistant", "content": "推荐 HP-01。"}]),
        BlackboardEntry(agent="consult", status="failed", error="boom", new_messages=[]),
    ]
    reply = o._run_result_agent(entries, "推荐耳机")
    assert "HP-01" in reply
    assert "PickWise-咨询" in reply and "无法回答" in reply, \
        f"失败项未如实告知: {reply}"


# ---------- 路径 3：出口话术按环节区分 ----------

def test_fallback_message_is_stage_specific():
    """出口话术能区分是哪个 Agent 失败，且不暴露内部术语。"""
    for key in AGENT_CONFIGS:
        exc = RuntimeError(f"[{key}] 执行失败: simulated")
        got = MultiAgentOrchestrator._failed_agent_key(exc)
        assert got == key, f"[{key}] 应解析为 {key}，实际 {got}"
    # 无前缀异常 → None（走通用话术）
    assert MultiAgentOrchestrator._failed_agent_key(RuntimeError("Connection error")) is None


def test_fallback_text_hides_internal_terms(tmp_path):
    """检查产品真实生成的话术，而不是在测试里重写一份相同字符串。"""
    for key in AGENT_CONFIGS:
        o = _build(tmp_path)
        text = o._on_failure("question", RuntimeError(f"[{key}] 执行失败: simulated"))
        assert AGENT_CONFIGS[key]["name"] in text
        for term in ("presale", "consult", "router", "simulated", "RuntimeError"):
            assert term not in text


# ---------- 路径 3b：出口兜底补齐配对并落盘 ----------

def test_chat_failure_pairs_messages_and_persists(tmp_path):
    """chat() 失败 → user/assistant 成对 + 落盘，不留幽灵轮次。"""
    import os

    o = _build(tmp_path, fail_mode="fail")
    from app.agent.memory import MemoryManager
    from app.agent.skills import SkillManager
    from app.agent.tools.manager import ToolManager
    from app.multi_agent.agents import SubAgent

    o.memory_manager = MemoryManager(
        client=o.client, model="fake", user_id="t",
        memory_dir="app/sessions/_tmp_test_res_mem2", memory_enabled=False, max_ltm_facts=10)
    o.skill_manager = SkillManager(skills_dir=settings.skills_dir, enabled=False)
    o.agents = {
        k: SubAgent(
            name=cfg["name"],
            tool_manager=ToolManager(use_mcp=False, mcp_server_url="",
                                     allowed_tools=cfg["tools"]),
            client=o.client, model="fake", temperature=0.0)
        for k, cfg in AGENT_CONFIGS.items()
    }
    from app.multi_agent.router import Router

    o.router = Router(o.client, "fake")

    reply = o.chat("推荐耳机")
    assert isinstance(reply, str) and reply, "应返回兜底话术而非抛异常"
    roles = [m.get("role") for m in o.raw_messages]
    assert roles == ["user", "assistant"], f"配对不完整: {roles}"
    assert os.path.exists(o.session_path), "兜底后应落盘"
    from app.agent.storage import load_session
    restored = load_session(o.session_path)
    assert restored["messages"] == o.raw_messages
    assert restored["messages"][-1]["content"] == reply


# ---------- 配置生效 ----------

def test_transport_settings_are_explicit(tmp_path, monkeypatch):
    """检查实际构造参数；改成非默认值，避免“碰巧 SDK 默认一样”的假通过。"""
    from unittest.mock import Mock

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "openai_base_url", "https://test.invalid")
    monkeypatch.setattr(settings, "openai_timeout", 12.5)
    monkeypatch.setattr(settings, "openai_max_retries", 1)
    monkeypatch.setattr(settings, "memory_enabled", False)
    monkeypatch.setattr(settings, "skills_enabled", False)
    monkeypatch.setattr(settings, "mcp_enabled", False)
    factory = Mock(return_value=_client())
    monkeypatch.setattr("app.multi_agent.orchestrator.openai.OpenAI", factory)
    o = MultiAgentOrchestrator(session_path=str(tmp_path / "session.json"))
    factory.assert_called_once_with(
        api_key="test-key", base_url="https://test.invalid", timeout=12.5, max_retries=1,
    )
    assert o.router.client is o.client
    assert all(agent.client is o.client for agent in o.agents.values())
    assert o.memory_manager.client is o.client
