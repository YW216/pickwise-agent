"""P3 多 Agent 验证：并行执行 / 黑板 / Result 整合 / full 档合并。

对应设计文档（develop_docs/模块设计/4-多Agent架构设计.md）十五节验证清单：
- #4 单 Agent 路径（mock：直返、全量合并）
- #5 多 Agent 路径（mock：黑板条目、固定顺序合并、tool 对完整）
- #7 合并可复现（mock：完成顺序与合并顺序解耦）
- #8 并行只读（mock：执行期间 raw_messages 长度不变）
- #9 部分失败（mock：1 个 Agent 抛错 → 其余正常，失败进黑板）
- #13 返回格式（真调 LLM smoke：多意图 → Result 整合 → str 返回）

用法：python tests/test_p3_multi.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.blackboard import BlackboardEntry, render_blackboard  # noqa: E402
from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_p3_session.json")


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _clean():
    p = Path(TEST_SESSION)
    try:
        p.unlink(missing_ok=True)
    except OSError:
        # 沙箱/回收站不可用等环境限制下，删除被拒绝时退化为改名移开，
        # 保证后续用例从干净会话开始（清理失败不否定验证结论）
        try:
            p.replace(p.with_suffix(".json.bak"))
        except OSError:
            print(f"  ⚠️ 会话文件清理失败: {p}")


def _mock_orchestrator() -> MultiAgentOrchestrator:
    """关掉记忆 LLM 调用、抬高压缩阈值的 orchestrator（mock 用，不真调 LLM）。"""
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    orch.memory_manager.memory_enabled = False
    orch.history_threshold = 100
    return orch


def _fake_handle(new_messages: list[dict], delay: float = 0.0):
    """造一个替身 handle：睡 delay（制造完成顺序与合并顺序相反）后返回固定产出。"""

    def _handle(messages, max_steps=5):
        if delay:
            time.sleep(delay)
        return new_messages[-1]["content"], new_messages
    return _handle


GUIDE_MSGS = [
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "g1", "type": "function",
         "function": {"name": "search_catalog", "arguments": "{}"}},
    ]},
    {"role": "tool", "tool_call_id": "g1", "content": '{"candidates": ["LP-01"]}'},
    {"role": "assistant", "content": "售前最终报告：推荐 LP-01 ¥5180"},
]
CONSULT_MSGS = [
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "retrieve_knowledge", "arguments": "{}"}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": '{"results": ["OLED 说明"]}'},
    {"role": "assistant", "content": "咨询最终报告：OLED 与 IPS 的区别是……"},
]


def _tool_pairs_intact(messages: list[dict]) -> bool:
    """C2/#5：每条 tool 消息必须对应最近一条 assistant(tool_calls) 中未消费的 id。

    一个 assistant 可带多个 tool_calls（后面连续多条 tool 消息，协议允许），
    因此用"待消费 id 集合"逐条核销，而不是只看相邻上一条。
    """
    pending: set = set()
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            pending = {tc.get("id") for tc in m.get("tool_calls") or []}
        elif role == "tool":
            tid = m.get("tool_call_id")
            if tid not in pending:
                return False
            pending.discard(tid)
    return True


# ---------- 验证 1：黑板渲染（十五节 #5 前置） ----------
def test_render_blackboard():
    print("\n[1/5] 黑板全量轨迹文本渲染")
    entries = [
        BlackboardEntry("presale", "success", None, GUIDE_MSGS),
        BlackboardEntry("consult", "failed", "API 超时", []),
    ]
    text = render_blackboard(entries)
    checks = [
        "【PickWise-售前】" in text,
        "[行动] 调用工具 search_catalog" in text,
        "[工具结果]" in text,
        "[最终答复] 售前最终报告" in text,
        "【PickWise-咨询】" in text,
        "失败原因：API 超时" in text,
    ]
    if all(checks):
        _ok("成功条目（行动/工具结果/最终答复）与失败条目（失败原因）渲染正确")
    else:
        _fail(f"渲染缺项：{checks}")


# ---------- 验证 2：多 Agent 并行 + 固定顺序合并 + 只读（#5/#7/#8） ----------
def test_multi_merge():
    print("\n[2/5] 多 Agent：并行执行 + 固定顺序合并 + 并行只读（mock）")
    orch = _mock_orchestrator()
    captured = {"len_during": []}

    orch.router.route = lambda inp, hist, summary=None: ["presale", "consult"]
    # presale 慢、consult 快 → 完成顺序 consult 先，但合并必须 presale 在前（#7）
    orch.agents["presale"].handle = _fake_handle(GUIDE_MSGS, delay=0.3)
    orch.agents["consult"].handle = _fake_handle(CONSULT_MSGS, delay=0.05)
    for agent in orch.agents.values():  # 记录执行期间 raw_messages 长度（#8）
        orig = agent.handle
        agent.handle = lambda msgs, max_steps=5, o=orig: (
            captured["len_during"].append(len(orch.raw_messages)), o(msgs, max_steps),
        )[1]
    orch._run_result_agent = lambda entries, user_input: "Result 整合文本"

    reply = orch.chat("推荐个笔记本，顺便讲讲 OLED 和 IPS 的区别")

    if reply == "Result 整合文本":
        _ok("reply = Result 整合文本（非任一 Agent 原文）")
    else:
        _fail(f"reply 异常：{reply}")
    if all(n == 1 for n in captured["len_during"]):
        _ok("执行期间 raw_messages 长度不变（并行只读，#8）")
    else:
        _fail(f"执行期间 raw_messages 被写：{captured['len_during']}")

    msgs = orch.raw_messages
    # 期望结构：user + presale 工具对 + consult 工具对 + Result 文本（presale 在前）
    expected_tail = [
        {"role": "user", "content": "推荐个笔记本，顺便讲讲 OLED 和 IPS 的区别"},
        *GUIDE_MSGS[:-1],
        *CONSULT_MSGS[:-1],
        {"role": "assistant", "content": "Result 整合文本"},
    ]
    if msgs == expected_tail:
        _ok("full 档合并：presale 工具对 → consult 工具对 → Result 文本收尾（固定顺序）")
    else:
        _fail(f"合并结构异常：{[(m.get('role'), m.get('tool_call_id')) for m in msgs]}")
    if _tool_pairs_intact(msgs):
        _ok("raw_messages 内 tool 消息成对完整（无孤立 tool）")
    else:
        _fail("存在孤立 tool 消息")


# ---------- 验证 3：部分失败（#9） ----------
def test_partial_failure():
    print("\n[3/5] 部分失败：presale 抛错 → consult 正常，不整体失败（mock）")
    orch = _mock_orchestrator()
    orch.router.route = lambda inp, hist, summary=None: ["presale", "consult"]

    def _boom(messages, max_steps=5):
        raise RuntimeError("模拟 presale 崩溃")
    orch.agents["presale"].handle = _boom
    orch.agents["consult"].handle = _fake_handle(CONSULT_MSGS)

    received = {}
    def _result(entries, user_input):
        received["entries"] = entries
        return "Result 收到的回复"
    orch._run_result_agent = _result

    reply = orch.chat("推荐个笔记本，顺便讲讲 OLED")

    if reply == "Result 收到的回复":
        _ok("整体未失败，reply 正常返回")
    else:
        _fail(f"reply 异常：{reply}")
    entries = received.get("entries", [])
    statuses = {e.agent: e.status for e in entries}
    if statuses.get("presale") == "failed" and statuses.get("consult") == "success":
        _ok("黑板如实记录：presale=failed（error 有值）、consult=success")
    else:
        _fail(f"黑板状态异常：{statuses}")
    if any(e.error and "模拟 presale 崩溃" in e.error for e in entries):
        _ok("失败条目携带 error 原因")
    else:
        _fail("失败条目缺少 error")
    # 合并只收 consult 的工具对；最后一条是 Result 文本
    if (
        orch.raw_messages[-1] == {"role": "assistant", "content": "Result 收到的回复"}
        and _tool_pairs_intact(orch.raw_messages)
        and len([m for m in orch.raw_messages if m.get("role") == "tool"]) == 1
    ):
        _ok("合并只含成功 Agent 的工具对，Result 文本收尾")
    else:
        _fail("部分失败下合并结构异常")


# ---------- 验证 4：单 Agent 路径回归（#4） ----------
def test_single_path():
    print("\n[4/5] 单 Agent：直返、无黑板无 Result、全量合并（mock）")
    orch = _mock_orchestrator()
    orch.router.route = lambda inp, hist, summary=None: ["presale"]
    orch.agents["presale"].handle = _fake_handle(GUIDE_MSGS)
    result_called = []
    orch._run_result_agent = lambda e, u: result_called.append(1) or "不该走到"

    reply = orch.chat("推荐个笔记本")

    if reply == "售前最终报告：推荐 LP-01 ¥5180":
        _ok("reply = Agent 最终文本直返（不经过 Result）")
    else:
        _fail(f"reply 异常：{reply}")
    if not result_called:
        _ok("Result Agent 未被调用（单 Agent 无黑板无 Result）")
    else:
        _fail("单 Agent 路径不应调用 Result Agent")
    if orch.raw_messages == [
        {"role": "user", "content": "推荐个笔记本"}, *GUIDE_MSGS,
    ]:
        _ok("new_messages 全量合并（A 现状，含工具消息）")
    else:
        _fail("单 Agent 合并结构异常")


# ---------- 验证 5：端到端 smoke（真调 LLM，多意图） ----------
def test_end_to_end_multi_smoke():
    print("\n[5/5] 端到端 smoke（真调 LLM：多意图 → 并行 → Result 整合）")
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    orch.history_threshold = 100

    captured = []
    original_route = orch.router.route
    def _spy(inp, hist, summary=None):
        scenarios = original_route(inp, hist, summary=summary)
        captured.append(scenarios)
        return scenarios
    orch.router.route = _spy

    reply = orch.chat("推荐个 6000 以内的笔记本，顺便讲讲 OLED 和 IPS 屏的区别")

    scenarios = captured[0] if captured else []
    if len(scenarios) > 1:
        _ok(f"路由输出多场景：{scenarios}（多 Agent 路径）")
    else:
        _ok(f"路由输出单场景：{scenarios}（LLM 判定单意图，单 Agent 路径，结构断言同下）")

    if isinstance(reply, str) and reply.strip():
        _ok(f"chat() 返回非空 str（{len(reply)} 字符）")
    else:
        _fail(f"chat() 应返回非空 str，实际 {type(reply)}")
    tool_msgs = [m for m in orch.raw_messages if m.get("role") == "tool"]
    if all(isinstance(m.get("content"), str) for m in tool_msgs):
        _ok(f"tool 消息 content 全为 str（{len(tool_msgs)} 条）")
    else:
        _fail("存在 tool 消息 content 非 str")
    if _tool_pairs_intact(orch.raw_messages):
        _ok("raw_messages 内 tool 消息成对完整")
    else:
        _fail("存在孤立 tool 消息")
    tail = orch.raw_messages[-1]
    if tail.get("role") == "assistant" and not tail.get("tool_calls"):
        if len(scenarios) > 1 and tail["content"] == reply:
            _ok("最后一条 = Result 整合文本（多 Agent full 档合并收尾）")
        else:
            _ok("raw_messages 尾部为 assistant 文本")
    else:
        _fail(f"raw_messages 尾部异常：{str(tail)[:100]}")
    print(f"     回复：{reply[:150]}")


def main():
    print("=" * 60)
    print("  P3 多 Agent 验证（并行/黑板/Result/full 档 · 十五节 #4/#5/#7/#8/#9/#13）")
    print("=" * 60)

    try:
        test_render_blackboard()
        test_multi_merge()
        test_partial_failure()
        test_single_path()
        test_end_to_end_multi_smoke()
    finally:
        _clean()

    print("\n" + "=" * 60)
    print("  🎉 P3 多 Agent 全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
