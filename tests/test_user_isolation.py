"""通过真实编排器、子 Agent、工具和文件存储验证用户隔离。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock
import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.agent.memory.long_term import LongTermMemory, MemoryFact
from app.agent.storage import load_session
from app.agent.tools.manager import ToolManager
from app.config.settings import settings
from app.identity import SessionIdentity, SessionOwnershipError
from app.multi_agent.orchestrator import MultiAgentOrchestrator


def fact(content):
    return MemoryFact(content, "preference", "2026-10-11")


@pytest.fixture
def create_orchestrator(tmp_path):
    created = []

    def create(owner=None, **options):
        owner = owner or SessionIdentity(str(uuid4()), str(uuid4()))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())), close=Mock())
        instance = MultiAgentOrchestrator(
            identity=owner, web_session_dir=str(tmp_path / "web"), client=client,
            skills_enabled=False, mcp_enabled=False, **options,
        )
        instance._mock_client = client  # GatedAsyncClient 包装后从这里设置副作用
        instance.compaction_enabled = False
        created.append(instance)
        return instance

    yield create
    for instance in created:
        instance.client.close()


def recall(instance, **arguments):
    return instance.agents["presale"].tool_manager.execute_tool("recall_user_memory", arguments)


def test_second_user_does_not_replace_first_users_tool(create_orchestrator):
    a = create_orchestrator()
    a.memory_manager.stm.facts = ["A 预算 5000"]
    a.memory_manager.ltm.add_facts([fact("A 偏好轻薄本")])
    a.memory_manager.ltm.save()
    b = create_orchestrator()
    b.memory_manager.stm.facts = ["B 预算 8000"]
    assert recall(a)["data"]["short_term_facts"] == ["A 预算 5000"]
    assert recall(b)["data"]["short_term_facts"] == ["B 预算 8000"]
    assert recall(a)["data"]["long_term_facts"][0]["content"] == "A 偏好轻薄本"
    assert recall(b)["data"]["long_term_facts"] == []
    disabled = create_orchestrator(memory_enabled=False)
    assert not recall(disabled)["success"]
    assert recall(a)["data"]["short_term_facts"] == ["A 预算 5000"]


def test_real_child_agents_keep_users_isolated_when_chatting_in_parallel(create_orchestrator, monkeypatch):
    a, b = create_orchestrator(), create_orchestrator()
    barrier = Barrier(2)
    monkeypatch.setattr("app.agent.memory.short_term.extract_short_term_facts", lambda *args: args[-1])

    def prepare(instance, marker):
        instance.memory_manager.stm.facts = [marker]
        instance.router.route = AsyncMock(return_value=["presale"])

        def complete(**kwargs):
            messages = kwargs["messages"]
            if messages[-1]["role"] == "tool":
                reply = json.loads(messages[-1]["content"])["data"]["short_term_facts"][0]
                message = SimpleNamespace(content=reply, tool_calls=None)
            else:
                barrier.wait(timeout=5)  # 两个用户的子 Agent 确实同时进入工具调用路径
                call = SimpleNamespace(id="memory-call", function=SimpleNamespace(
                    name="recall_user_memory", arguments="{}"))
                message = SimpleNamespace(content=None, tool_calls=[call])
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

        instance._mock_client.chat.completions.create.side_effect = complete

    prepare(a, "A 预算 5000")
    prepare(b, "B 预算 8000")
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(
            lambda instance: asyncio.run(instance.chat("记得我的预算吗？")), [a, b]))
    assert replies == ["A 预算 5000", "B 预算 8000"]
    for instance, marker in ((a, replies[0]), (b, replies[1])):
        state = load_session(instance.session_path, identity=instance.identity)
        assert state["messages"][-1]["content"] == marker
        assert state["short_term_memory"]["facts"] == [marker]


def test_one_users_windows_share_long_term_but_not_short_term(create_orchestrator):
    a = create_orchestrator()
    second = create_orchestrator(SessionIdentity(a.user_id, str(uuid4())))
    a.memory_manager.stm.facts = ["窗口一预算 5000"]
    second.memory_manager.stm.facts = ["窗口二预算 8000"]
    a.memory_manager.ltm.add_facts([fact("偏好轻薄本")])
    a.memory_manager.ltm.save()
    assert recall(second)["data"]["long_term_facts"][0]["content"] == "偏好轻薄本"
    assert recall(second)["data"]["short_term_facts"] == ["窗口二预算 8000"]
    a.raw_messages = [{"role": "user", "content": "窗口一"}]
    second.raw_messages = [{"role": "user", "content": "窗口二"}]
    a.save()
    second.save()
    a.reset()
    assert second.history_size == 1
    assert recall(second)["data"]["short_term_facts"] == ["窗口二预算 8000"]
    assert recall(second)["data"]["long_term_facts"][0]["content"] == "偏好轻薄本"


def test_parallel_consolidation_merges_both_windows_and_refreshes_existing_window(
    create_orchestrator, monkeypatch,
):
    a = create_orchestrator()
    second = create_orchestrator(SessionIdentity(a.user_id, str(uuid4())))
    stranger = create_orchestrator()
    barrier = Barrier(2)

    async def extract(client, model, messages, summary, existing):
        barrier.wait(timeout=5)
        return [fact(messages[0]["content"])], summary

    monkeypatch.setattr("app.agent.memory.long_term.extract_long_term_facts", extract)
    a.raw_messages = [{"role": "user", "content": "偏好轻薄本"}]
    second.raw_messages = [{"role": "user", "content": "希望续航长"}]
    a.summary, second.summary = "窗口一摘要", "窗口二摘要"
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda instance: asyncio.run(instance.close()), [a, second]))
    facts = recall(a)["data"]["long_term_facts"]
    assert {item["content"] for item in facts} == {"偏好轻薄本", "希望续航长"}
    assert set(recall(second)["data"]["recent_interactions"]) == {"窗口一摘要", "窗口二摘要"}
    assert recall(stranger)["data"]["long_term_facts"] == []
    restored = create_orchestrator(a.identity)
    assert {item["content"] for item in recall(restored)["data"]["long_term_facts"]} == {
        "偏好轻薄本", "希望续航长",
    }


def test_session_restart_restores_history_and_stm(create_orchestrator):
    a = create_orchestrator()
    a.raw_messages = [{"role": "user", "content": "我的问题"}]
    a.memory_manager.stm.facts = ["预算 5000"]
    a.save()
    restored = create_orchestrator(a.identity)
    assert restored.raw_messages == a.raw_messages
    assert recall(restored)["data"]["short_term_facts"] == ["预算 5000"]


def test_refused_reset_keeps_current_history_and_short_term(create_orchestrator):
    instance = create_orchestrator()
    instance.raw_messages = [{"role": "user", "content": "保留当前对话"}]
    instance.memory_manager.stm.facts = ["预算 5000"]
    instance.save()
    path = Path(instance.session_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["user_id"] = str(uuid4())
    path.write_text(json.dumps(data), encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(SessionOwnershipError):
        instance.reset()
    assert instance.raw_messages == [{"role": "user", "content": "保留当前对话"}]
    assert instance.memory_manager.stm.facts == ["预算 5000"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure_stage", ["memory", "tools", "restore"])
def test_failed_construction_closes_client(tmp_path, monkeypatch, failure_stage):
    owner = SessionIdentity(str(uuid4()), str(uuid4()))
    client = SimpleNamespace(close=Mock(side_effect=OSError("close failure")))
    failure = Mock(side_effect=ValueError("initialization failure"))
    if failure_stage == "memory":
        monkeypatch.setattr("app.agent.memory.MemoryManager", failure)
    elif failure_stage == "tools":
        monkeypatch.setattr("app.multi_agent.orchestrator.ToolManager", failure)
    else:
        from app.agent.memory import MemoryManager
        from app.agent.storage import save_session
        save_session(str(owner.session_path(tmp_path)), [], None,
                     short_term_memory={"facts": ["旧短期记忆"]}, identity=owner)
        monkeypatch.setattr(MemoryManager, "restore_stm", failure)
    with pytest.raises(ValueError, match="initialization failure"):
        MultiAgentOrchestrator(identity=owner, web_session_dir=str(tmp_path),
                               client=client, skills_enabled=False)
    client.close.assert_called_once()


def test_model_arguments_cannot_change_bound_user(create_orchestrator):
    a, b = create_orchestrator(), create_orchestrator()
    a.memory_manager.stm.facts = ["A 的预算"]
    result = recall(a, user_id=b.user_id, manager=b.memory_manager)
    assert result["data"]["short_term_facts"] == ["A 的预算"]
    assert "notice" in result
    limited = ToolManager(allowed_tools={"search_products"}, memory_manager=a.memory_manager)
    assert not limited.execute_tool("recall_user_memory", {})["success"]


def test_skill_dependencies_are_per_instance():
    a_skills = SimpleNamespace(load_skill=Mock(return_value={"success": True, "data": {"instructions": "A"}}))
    b_skills = SimpleNamespace(load_skill=Mock(return_value={"success": True, "data": {"instructions": "B"}}))
    a = ToolManager(allowed_tools={"load_skill"}, skill_manager=a_skills)
    b = ToolManager(allowed_tools={"load_skill"}, skill_manager=b_skills)
    assert a.execute_tool("load_skill", {"skill_name": "test"})["data"]["instructions"] == "A"
    assert b.execute_tool("load_skill", {"skill_name": "test"})["data"]["instructions"] == "B"


def test_web_identity_cannot_override_storage_paths(create_orchestrator, tmp_path):
    with pytest.raises(ValueError, match="identity"):
        create_orchestrator(session_path=str(tmp_path / "other.json"))
    with pytest.raises(ValueError, match="identity"):
        create_orchestrator(memory_dir=str(tmp_path / "other_memory"))


def test_foreign_memory_cannot_be_read_saved_or_deleted(tmp_path):
    a = LongTermMemory("a", str(tmp_path))
    a.add_facts([fact("私有事实")])
    a.save()
    data = json.loads(a.memory_path.read_text(encoding="utf-8"))
    data["user_id"] = "b"
    a.memory_path.write_text(json.dumps(data), encoding="utf-8")
    original = a.memory_path.read_bytes()
    for operation in (a.load, a.save, a.reset):
        with pytest.raises(SessionOwnershipError):
            operation()
        assert a.memory_path.read_bytes() == original


def test_refresh_does_not_duplicate_facts_and_stale_save_does_not_restore_deleted_data(tmp_path):
    first = LongTermMemory("user", str(tmp_path))
    second = LongTermMemory("user", str(tmp_path))
    first.add_facts([fact("old")])
    first.save()
    second.load()
    second.load()
    assert len(second.facts) == 1
    first.reset()
    second.save()
    assert second.facts == []


def test_failed_memory_write_keeps_pending_facts_for_retry(tmp_path, monkeypatch):
    memory = LongTermMemory("user", str(tmp_path))
    memory.add_facts([fact("保留事实")])
    with monkeypatch.context() as patch:
        patch.setattr("app.agent.memory.long_term.atomic_write_json", Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError):
            memory.save()
    memory.save()
    restored = LongTermMemory("user", str(tmp_path))
    restored.load()
    assert [item.content for item in restored.facts] == ["保留事实"]


def test_legacy_cli_paths_and_settings_are_preserved(tmp_path):
    old_settings = (settings.memory_dir, settings.memory_enabled, settings.mcp_enabled)
    instance = MultiAgentOrchestrator(
        session_path=str(tmp_path / "cli.json"), user_id="default",
        memory_dir=str(tmp_path / "cli_memory"), memory_enabled=False, skills_enabled=False,
        client=SimpleNamespace(close=Mock()),
    )
    instance.raw_messages = [{"role": "user", "content": "CLI 记录"}]
    instance.save()
    assert load_session(instance.session_path)["messages"] == instance.raw_messages
    assert instance.memory_manager.ltm.memory_path == tmp_path / "cli_memory" / "default.json"
    assert (settings.memory_dir, settings.memory_enabled, settings.mcp_enabled) == old_settings
    asyncio.run(instance.close())


def test_sandbox_passes_instance_options_without_mutating_settings(tmp_path, monkeypatch):
    from app.evaluation.sandbox import Sandbox
    before = (settings.memory_dir, settings.memory_enabled, settings.mcp_enabled)
    constructor = Mock(return_value=object())
    monkeypatch.setattr("app.multi_agent.orchestrator.MultiAgentOrchestrator", constructor)
    path = str(tmp_path / "case.json")
    Sandbox(tmp_root=str(tmp_path))._build_agent(path)
    assert (settings.memory_dir, settings.memory_enabled, settings.mcp_enabled) == before
    assert constructor.call_args.kwargs["memory_dir"] == str(tmp_path / "case_memory")
    assert constructor.call_args.kwargs["memory_enabled"] is True
    assert constructor.call_args.kwargs["mcp_enabled"] is False
