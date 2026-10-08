"""历史视图投影（_project_history）的单测。

为什么单独成文件：这是 2026-10-07 新增的上下文投影逻辑，且它守着一条
"协议层不可违反"的边界（tool 消息成对，见设计文档 C2），必须有 pytest
级别的回归锁——不能只靠 test_p2_context.py 里的自定义函数（pytest 不收集）。

覆盖点：
1. 纯越权块被折叠，且折叠文本不含任何工具名（符号泄漏的根因是工具名）
2. 含共享工具的块整块保留（共享工具本就该给两个 Agent 看）
3. 折叠后 tool 消息成对消失，不留孤儿 tool 消息（协议层硬约束）
4. 无白名单信息时不投影（防御：cfg 缺 tools 键的老调用方）
5. 幂等：投影结果再投影一次不变（纯函数，不破坏 pack 可复用性）
6. 不改原 history（入参是只读引用，orchestrator 的 raw_messages 不能被动）
"""

from app.multi_agent.agents import AGENT_CONFIGS
from app.multi_agent.context_pack import ContextPack, build_working_messages


def _foreign_history():
    """一轮 presale 检索（search_products 越权 + get_detail 共享）。"""
    return [
        {"role": "user", "content": "预算6000左右买什么手机"},
        {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {
                    "name": "search_products", "arguments": '{"query":"6000元手机"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "c1",
         "content": '{"data": [{"id": "HP-01", "name": "星海 Note 12", "price": 5999}]}'},
        {"role": "assistant", "content": "推荐星海 Note 12（5999 元）。"},
        {"role": "user", "content": "这款保修几年"},
    ]


def test_pure_foreign_block_is_collapsed():
    """纯越权块：3 条（assistant+tool+结果）折叠为 1 条中性 assistant。"""
    cfg = AGENT_CONFIGS["consult"]
    pack = ContextPack(history=_foreign_history(), summary=None,
                       memory_sections=[], skill_catalog="")
    msgs = build_working_messages(pack, cfg, "single")

    # 5 条历史 → 4 条：折叠掉 assistant(tool_calls)+tool 这一对
    # （user / 折叠说明 / 原答复 / 新 user）
    tail = msgs[1:]
    assert len(tail) == 4, f"期望 4 条，实际 {len(tail)}: {tail}"
    assert tail[0]["role"] == "user" and "6000" in tail[0]["content"]
    assert tail[1]["role"] == "assistant" and "其他专家" in tail[1]["content"]
    assert tail[2]["role"] == "assistant" and "星海 Note 12" in tail[2]["content"]
    assert tail[3]["role"] == "user" and "保修" in tail[3]["content"]


def test_collapsed_text_contains_no_tool_name():
    """关键断言：折叠文本里不能出现任何工具名——泄漏的正是这个符号。"""
    cfg = AGENT_CONFIGS["consult"]
    pack = ContextPack(history=_foreign_history(), summary=None,
                       memory_sections=[], skill_catalog="")
    msgs = build_working_messages(pack, cfg, "single")

    forbidden = set(AGENT_CONFIGS["presale"]["tools"]) - set(cfg["tools"])
    assert forbidden, "presale 应有 consult 白名单外的工具，否则本用例无意义"
    all_text = " ".join(m.get("content") or "" for m in msgs)
    for name in forbidden:
        assert name not in all_text, f"折叠后仍泄漏工具名: {name}"


def test_shared_tool_block_is_preserved():
    """含共享工具（get_detail）的块整块保留，不折叠。"""
    history = _foreign_history()
    history[1]["tool_calls"].append({
        "id": "c2", "type": "function",
        "function": {"name": "get_detail", "arguments": '{"product_id":"HP-01"}'},
    })
    history.insert(3, {"role": "tool", "tool_call_id": "c2", "content": '{"data": {}}'})

    cfg = AGENT_CONFIGS["consult"]
    pack = ContextPack(history=history, summary=None,
                       memory_sections=[], skill_catalog="")
    msgs = build_working_messages(pack, cfg, "single")

    assert any("tool_calls" in m for m in msgs), "共享工具块应被保留"
    # 原样保留 = 仍是 5 条历史 + system
    assert len(msgs) == len(history) + 1


def test_no_orphan_tool_message_after_collapse():
    """C2 硬约束：折叠后不得残留孤儿 tool 消息（协议会直接报错）。"""
    cfg = AGENT_CONFIGS["consult"]
    pack = ContextPack(history=_foreign_history(), summary=None,
                       memory_sections=[], skill_catalog="")
    msgs = build_working_messages(pack, cfg, "single")

    assert not any(m.get("role") == "tool" for m in msgs), "折叠后仍有 tool 消息"
    # tool_call_id 与 tool_calls 必须成对（此处都清空，天然满足）
    for m in msgs:
        if m.get("tool_calls"):
            ids = {tc["id"] for tc in m["tool_calls"]}
            for t in msgs:
                if t.get("role") == "tool":
                    assert t["tool_call_id"] in ids, "出现孤儿 tool 消息"


def test_no_allowlist_means_no_projection():
    """cfg 缺 tools 键时退化为原样返回（兼容老调用方，不静默丢历史）。"""
    history = _foreign_history()
    pack = ContextPack(history=history, summary=None,
                       memory_sections=[], skill_catalog="")
    msgs = build_working_messages(pack, {"prompts": {"single": "P"}}, "single")
    assert len(msgs) == len(history) + 1


def test_projection_is_idempotent_and_readonly():
    """纯函数：重复投影结果不变，且不改原 history（orchestrator 事实层只读）。"""
    cfg = AGENT_CONFIGS["consult"]
    history = _foreign_history()
    snapshot = [dict(m) for m in history]
    pack = ContextPack(history=history, summary=None,
                       memory_sections=[], skill_catalog="")

    once = build_working_messages(pack, cfg, "single")
    twice = build_working_messages(
        ContextPack(history=once[1:], summary=None, memory_sections=[], skill_catalog=""),
        cfg, "single",
    )
    assert [m.get("content") for m in once[1:]] == [m.get("content") for m in twice[1:]]
    # 原 list 及其元素未被就地改写
    assert history == snapshot
