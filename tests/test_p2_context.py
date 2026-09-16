"""P2 上下文验证：ContextPack 契约 / build_working_messages 结构 / 前向兼容证明。

对应设计文档（develop_docs/模块设计/6-上下文设计.md）第六节验证清单 #1-#6。
- #1 ContextPack 4 字段构造
- #2 build_working_messages 结构 = system(+skill)+memory+summary+history；空值分支
- #3 mode 选择（P2 恒 single，函数支持 multi）
- #4 一次构建多处消费：两个 Agent 共享同一 pack 引用（P3 并行铺垫）
- #5 前向兼容证明：模拟压缩后形态（窗口切片 + summary）零改动可跑，
     窗口内 tool 对完整（约束 C1/C2）
- #6 orchestrator._build_pack 集成：history 即 raw_messages 同引用、各字段来源正确

用法：python tests/test_p2_context.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.context_pack import ContextPack, build_working_messages  # noqa: E402
from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_p2_session.json")


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


def _fresh_orchestrator() -> MultiAgentOrchestrator:
    Path(TEST_SESSION).unlink(missing_ok=True)
    return MultiAgentOrchestrator(session_path=TEST_SESSION)


# ---------- 验证 1：ContextPack 4 字段构造 ----------
def test_pack_fields():
    print("\n[1/6] ContextPack 4 字段构造")
    history = [{"role": "user", "content": "hi"}]
    pack = ContextPack(
        history=history, summary=None,
        memory_sections=[], skill_catalog="",
    )
    if pack.history is history and pack.summary is None:
        _ok("history 只读引用、summary 可为 None")
    else:
        _fail("history 引用或 summary 默认值异常")
    if pack.memory_sections == [] and pack.skill_catalog == "":
        _ok("memory_sections / skill_catalog 空值合法")
    else:
        _fail("空值字段异常")


# ---------- 验证 2：build_working_messages 结构与空值分支 ----------
def test_working_messages_structure():
    print("\n[2/6] build_working_messages 结构 = system(+skill)+memory+summary+history")
    cfg = {"prompts": {"single": "SINGLE_PROMPT", "multi": "MULTI_PROMPT"}}
    history = [
        {"role": "user", "content": "6000 预算买笔记本"},
        {"role": "assistant", "content": "好的"},
    ]
    pack = ContextPack(
        history=history, summary="此前摘要文本",
        memory_sections=[{"role": "system", "content": "MEM"}],
        skill_catalog="SKILL_CATALOG",
        product_block="<mentioned-products>[]</mentioned-products>",
    )
    msgs = build_working_messages(pack, cfg, "single")
    expected = [
        {"role": "system", "content": "SINGLE_PROMPT"},
        {"role": "system", "content": "SKILL_CATALOG"},
        {"role": "system", "content": "MEM"},
        {"role": "system", "content": "以下是此前对话的摘要，用于延续上下文记忆：\n此前摘要文本"},
        {"role": "system", "content": (
            "以下是本会话提及过的商品记录（商品 ID、名称与提及当时的价格，"
            "价格为历史价、不代表现价）：\n<mentioned-products>[]</mentioned-products>"
        )},
        *history,
    ]
    if msgs == expected:
        _ok("满配结构顺序正确（6 段，含商品记忆独立段）")
    else:
        _fail(f"结构不符：{msgs}")

    pack_plain = ContextPack(
        history=history, summary=None, memory_sections=[], skill_catalog="",
    )
    msgs_plain = build_working_messages(pack_plain, cfg, "single")
    if msgs_plain == [{"role": "system", "content": "SINGLE_PROMPT"}, *history]:
        _ok("空值分支：无 skill / memory / summary 段")
    else:
        _fail(f"空值分支结构不符：{msgs_plain}")


# ---------- 验证 3：mode 选择 ----------
def test_mode_selection():
    print("\n[3/6] mode 选择（P2 恒 single，函数支持 multi）")
    cfg = {"prompts": {"single": "SINGLE_PROMPT", "multi": "MULTI_PROMPT"}}
    pack = ContextPack(history=[], summary=None, memory_sections=[], skill_catalog="")
    s = build_working_messages(pack, cfg, "single")[0]["content"]
    m = build_working_messages(pack, cfg, "multi")[0]["content"]
    if s == "SINGLE_PROMPT" and m == "MULTI_PROMPT":
        _ok("single/multi 取 cfg['prompts'][mode] 正确")
    else:
        _fail(f"mode 选择异常：single={s} multi={m}")


# ---------- 验证 4：一次构建多处消费（P3 并行铺垫） ----------
def test_shared_pack():
    print("\n[4/6] 一次构建多处消费：两个 Agent 共享同一 pack")
    history = [{"role": "user", "content": "hi"}]
    pack = ContextPack(
        history=history, summary=None,
        memory_sections=[], skill_catalog="",
    )
    cfgs = {
        "presale": {"prompts": {"single": "G"}},
        "consult": {"prompts": {"single": "C"}},
    }
    m1 = build_working_messages(pack, cfgs["presale"], "single")
    m2 = build_working_messages(pack, cfgs["consult"], "single")
    if m1[-1] is history[0] and m2[-1] is history[0]:
        _ok("两个 Agent 取到同一 history 引用（零拷贝）")
    else:
        _fail("history 引用被复制，违背一次构建多处消费")


# ---------- 验证 5：前向兼容证明（模拟未来压缩，约束 C1/C2） ----------
def _assistant_with_tool_call(call_id: str, name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }],
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _tool_pairs_intact(messages: list[dict]) -> bool:
    """C2：tool 消息必须紧跟含对应 tool_call_id 的 assistant。"""
    for i, m in enumerate(messages):
        if m.get("role") != "tool":
            continue
        prev = messages[i - 1] if i else None
        if not prev or prev.get("role") != "assistant":
            return False
        ids = {tc.get("id") for tc in prev.get("tool_calls", [])}
        if m.get("tool_call_id") not in ids:
            return False
    return True


def test_forward_compat_simulated_compression():
    print("\n[5/6] 前向兼容证明：模拟压缩后形态（窗口切片 + summary）零改动可跑")
    messages = [
        {"role": "user", "content": "第一轮：推荐笔记本"},          # 0 ┐ 已压缩（进摘要）
        _assistant_with_tool_call("c1", "search_catalog"),           # 1 │ 待删前缀
        _tool_result("c1", '{"candidates": []}'),                    # 2 │
        {"role": "assistant", "content": "第一轮答复"},              # 3 ┘
        {"role": "user", "content": "第二轮：比较两款"},             # 4 ┐ 保留区起点（cut=4）
        _assistant_with_tool_call("c2", "compare_products"),         # 5 │
        _tool_result("c2", '{"rows": {}}'),                          # 6 │ 工作历史（切后窗口）
        {"role": "assistant", "content": "第二轮最终答复"},          # 7 ┘
    ]
    cut = 4
    window = messages[cut:]               # 压缩即删除前缀：pack.history = messages[cut:]

    if _tool_pairs_intact(messages) and _tool_pairs_intact(window):
        _ok("压缩前后 tool 消息成对完整（C2：user turn 边界切分不拆对）")
    else:
        _fail("tool 消息对被切散（违反 C2）")

    pack = ContextPack(
        history=window, summary="结构化摘要（被删前缀的代表）",
        memory_sections=[{"role": "system", "content": "MEM"}],
        skill_catalog="",
    )
    cfg = {"prompts": {"single": "S"}}
    msgs = build_working_messages(pack, cfg, "single")
    if (
        msgs[0]["content"] == "S"
        and msgs[-1] is messages[-1]
        and any("结构化摘要" in m["content"] for m in msgs if m["role"] == "system")
        and msgs[-4:] == window
    ):
        _ok("删除前缀后的窗口 + summary 直接喂现有签名 → 结构正确（C1 零改动判据成立）")
    else:
        _fail("压缩后形态下 build_working_messages 行为异常")


# ---------- 验证 6：_build_pack 集成（窗口语义 = raw_messages 同引用） ----------
def test_build_pack_integration():
    print("\n[6/6] orchestrator._build_pack 集成")
    orch = _fresh_orchestrator()
    orch.raw_messages = [{"role": "user", "content": "测试"}]
    orch.summary = "旧摘要"

    pack = orch._build_pack()

    if pack.history is orch.raw_messages:
        _ok("pack.history 即工作历史同引用（零拷贝，一次构建多处消费）")
    else:
        _fail("pack.history 未直取 raw_messages")
    if pack.summary == "旧摘要" and pack.product_block == "":
        _ok("summary 存储纯叙述、product_block 随商品记忆为空（空店不注入）")
    else:
        _fail(f"summary/product_block 异常：{pack.summary!r} / {pack.product_block!r}")
    if pack.summary == "旧摘要":
        _ok("summary 直取 self.summary")
    else:
        _fail(f"summary 异常：{pack.summary}")
    if pack.memory_sections == orch.memory_manager.build_memory_prompt_sections():
        _ok("memory_sections 来源正确")
    else:
        _fail("memory_sections 来源异常")

    original_enabled = orch.skill_manager.enabled
    try:
        # 2026-09-03 契约更新：skill_catalog 从 pack 中移出，改为 per-agent 注入
        # （_build_pack 恒空串，_skill_catalog_for(agent_key) 按归属过滤）
        if orch._build_pack().skill_catalog == "":
            _ok("pack.skill_catalog 恒为空串（skill 目录已改为 per-agent 注入）")
        else:
            _fail("pack.skill_catalog 应为空串")
        if orch._skill_catalog_for("presale"):
            _ok("presale 有可见技能 → _skill_catalog_for 返回目录文本")
        else:
            _fail("presale 应有可见技能（product-recommend）")
        if "policy-check" in orch._skill_catalog_for("consult"):
            _ok("consult 可见 policy-check 技能 → 归属过滤生效")
        else:
            _fail("consult 应有可见技能（policy-check）")
    finally:
        orch.skill_manager.enabled = original_enabled
        Path(TEST_SESSION).unlink(missing_ok=True)


def main():
    print("=" * 60)
    print("  P2 上下文验证（ContextPack 契约 · 6-上下文设计.md 第六节）")
    print("=" * 60)

    test_pack_fields()
    test_working_messages_structure()
    test_mode_selection()
    test_shared_pack()
    test_forward_compat_simulated_compression()
    test_build_pack_integration()

    print("\n" + "=" * 60)
    print("  🎉 P2 上下文全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
