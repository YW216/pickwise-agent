"""Multi-Agent 架构验收测试（P4 重写，取代旧客服三 Agent 版本）。

对应设计文档（develop_docs/模块设计/4-多Agent架构设计.md）十五节验证清单，
与 test_p1/p2/p3（分阶段载体）互补，本文件是**整体架构验收**：

一、架构验收（多数 mock，不真调 LLM）
  A1 白名单隔离（4/8/3）+ 6 份 prompt 一致性（#1/#2）
  A2 Router 多值解析 + max_tokens=512（#3）
  A3 单 Agent 路径：直返、无 Result、全量合并（#4）
  A4 多 Agent 路径：并行、黑板、固定顺序合并、tool 对完整（#5/#7/#8）
  A5 部分失败不整体失败（#9）
  A6 返回格式：chat() 返回 str（#13）
  A7 会话持久化与恢复（#12）

二、子系统验收（真调 LLM，标注 smoke）
  S1 记忆：STM 更新 + 落盘 + 恢复（#12）
  S2 压缩：阈值触发 + summary 生成 + 切点避开 tool 消息
  S3 RAG：retrieve_knowledge 真实召回（内容相关性弱断言——知识库当前为政策文档）
  S4 技能：skill catalog 注入 system prompt
  S5 端到端多意图 smoke：并行 + Result 整合（#5/#11 数值保真人工可核）

用法：python tests/test_multi_agent.py
（真调 LLM 的用例集中在第二部分，可用参数跳过：python tests/test_multi_agent.py --mock-only）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.context_budget import ContextOverflowError  # noqa: E402
from app.multi_agent.agents import AGENT_CONFIGS  # noqa: E402
from app.multi_agent.blackboard import BlackboardEntry, render_blackboard  # noqa: E402
from app.multi_agent.context_pack import build_working_messages  # noqa: E402
from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402
from app.multi_agent.router import Router  # noqa: E402
from app.prompts.agents import (  # noqa: E402
    CONSULT_MULTI_PROMPT,
    CONSULT_PROMPT,
    PRESALE_MULTI_PROMPT,
    PRESALE_PROMPT,
)

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_multi_agent_session.json")

MOCK_ONLY = "--mock-only" in sys.argv


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _warn(msg: str):
    print(f"  ⚠️  {msg}")


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


MOCK_SUMMARY = """## 用户目标
mock 摘要

## 约束与偏好
- （无）

## 进展
### 已完成
- [x] 已完成两轮查询
### 进行中
- [ ] 处理当前问题
### 受阻
- （无）

## 关键决策
- （无）

## 下一步
1. 处理当前问题

## 关键上下文
- mock 数据
"""


def _mock_orch(threshold: int = 100) -> MultiAgentOrchestrator:
    """mock 用 orchestrator：关记忆 LLM 调用、抬高压缩阈值。"""
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    orch.memory_manager.memory_enabled = False
    orch.compaction_enabled = False
    return orch


def _tool_pairs_intact(messages: list[dict]) -> bool:
    """每条 tool 消息对应最近一条 assistant(tool_calls) 中未消费的 id（一拖多安全）。"""
    pending: set = set()
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            pending = {tc.get("id") for tc in m.get("tool_calls") or []}
        elif role == "tool":
            if m.get("tool_call_id") not in pending:
                return False
            pending.discard(m.get("tool_call_id"))
    return True


# ============================================================
# 一、架构验收（mock）
# ============================================================

def a1_whitelist_and_prompts():
    print("\n[A1] 白名单隔离 + 4 份 prompt 一致性（#1/#2）")
    expected = {"presale": 8, "consult": 3}
    prompts = {
        "presale": (PRESALE_PROMPT, PRESALE_MULTI_PROMPT),
        "consult": (CONSULT_PROMPT, CONSULT_MULTI_PROMPT),
    }
    for key, count in expected.items():
        if len(set(AGENT_CONFIGS[key]["tools"])) != count:
            _fail(f"{key} 白名单数量 != {count}")
    _ok("白名单数量 presale=8 / consult=3")

    orch = _mock_orch()
    import re
    for key, (single, multi) in prompts.items():
        actual = {
            d["function"]["name"]
            for d in orch.agents[key].tool_manager.tool_definitions
        }
        if actual != set(AGENT_CONFIGS[key]["tools"]):
            _fail(f"{key} ToolManager 过滤与白名单不一致")
        whitelist = set(AGENT_CONFIGS[key]["tools"])
        for label, prompt in (("single", single), ("multi", multi)):
            m = re.search(r"## 可用工具\n(.*?)(?=\n## |\Z)", prompt, re.DOTALL)
            declared = set(re.findall(r"\*\*([a-z_]+)\*\*", m.group(1))) if m else set()
            if declared != whitelist:
                _fail(f"{key}-{label} prompt 声明 {sorted(declared)} != 白名单")
    _ok("4 份 prompt 声明工具 = 各自白名单（物理隔离生效）")

    result = orch.agents["presale"].tool_manager.execute_tool(
        "retrieve_warranty", {"category": "笔记本"},
    )
    if result.get("success") is False and "未知工具" in (result.get("error") or ""):
        _ok("presale 调 retrieve_warranty → 未知工具信封（双重保险）")
    else:
        _fail(f"物理隔离失效: {result}")


def a2_router_parse():
    print("\n[A2] Router 多值解析 + max_tokens=512（#3）")
    from types import SimpleNamespace

    class _FakeClient:
        def __init__(self, content):
            self._content = content
            self.last_kwargs = None

        @property
        def chat(self):
            return SimpleNamespace(completions=self)

        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
            )

    cases = [
        ("presale", ["presale"]),
        ("PRESALE", ["presale"]),
        ("presale,consult", ["presale", "consult"]),
        ("presale，consult", ["presale", "consult"]),
        ("consult,presale", ["presale", "consult"]),
        ("consult,consult", ["consult"]),
        ("", ["consult"]),
        ("乱码 xyz", ["consult"]),
    ]
    for raw, expected in cases:
        actual = Router(_FakeClient(raw), "test-model").route("测试输入")
        if actual != expected:
            _fail(f"「{raw}」期望 {expected}，实际 {actual}")
    _ok(f"{len(cases)} 条解析用例全过（大小写/中文逗号/固定顺序/去重/兜底）")

    client = _FakeClient("presale")
    Router(client, "test-model").route("x")
    if (client.last_kwargs or {}).get("max_tokens") == 512:
        _ok("max_tokens=512（推理模型思考预算，5.3 修订）")
    else:
        _fail("max_tokens != 512")


def a3_single_agent_path():
    print("\n[A3] 单 Agent 路径：直返、无 Result、全量合并（#4，mock）")
    orch = _mock_orch()
    presale_msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function",
             "function": {"name": "search_catalog", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "t1", "content": '{"candidates": []}'},
        {"role": "assistant", "content": "售前最终答复"},
    ]
    orch.router.route = lambda inp, hist, summary=None: ["presale"]
    orch.agents["presale"].handle = lambda msgs, max_steps=5: (presale_msgs[-1]["content"], presale_msgs)
    result_called = []
    orch._run_result_agent = lambda e, u: result_called.append(1) or "不该调用"

    reply = orch.chat("推荐个笔记本")

    if reply == "售前最终答复" and not result_called:
        _ok("reply = Agent 最终文本直返，Result 未被调用")
    else:
        _fail(f"单 Agent 路径异常：reply={reply!r}, result_called={result_called}")
    if orch.raw_messages == [
        {"role": "user", "content": "推荐个笔记本"}, *presale_msgs,
    ]:
        _ok("new_messages 全量合并（A 现状，含工具消息）")
    else:
        _fail("单 Agent 合并结构异常")


def a4_multi_agent_merge():
    print("\n[A4] 多 Agent：并行、黑板、固定顺序合并、tool 对完整（#5/#7/#8，mock）")
    import time

    orch = _mock_orch()
    presale_msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "g1", "type": "function",
             "function": {"name": "search_catalog", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "g1", "content": '{"candidates": ["LP-01"]}'},
        {"role": "assistant", "content": "售前报告：LP-01 ¥5180"},
    ]
    consult_msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "retrieve_knowledge", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"results": []}'},
        {"role": "assistant", "content": "咨询报告：OLED 自发光……"},
    ]

    def _fake(msgs, delay):
        def _handle(m, max_steps=5):
            time.sleep(delay)
            return msgs[-1]["content"], msgs
        return _handle

    orch.router.route = lambda inp, hist, summary=None: ["presale", "consult"]
    # presale 慢、consult 快：完成顺序 consult 先，合并仍须 presale 在前（#7）
    orch.agents["presale"].handle = _fake(presale_msgs, 0.3)
    orch.agents["consult"].handle = _fake(consult_msgs, 0.05)
    orch._run_result_agent = lambda e, u: "Result 整合文本"

    reply = orch.chat("推荐个笔记本，顺便讲讲 OLED")

    if reply != "Result 整合文本":
        _fail(f"reply 应为 Result 整合文本，实际 {reply!r}")

    expected = [
        {"role": "user", "content": "推荐个笔记本，顺便讲讲 OLED"},
        *presale_msgs[:-1],
        *consult_msgs[:-1],
        {"role": "assistant", "content": "Result 整合文本"},
    ]
    if orch.raw_messages == expected:
        _ok("full 档合并：presale 工具对 → consult 工具对 → Result 文本收尾（固定顺序可复现）")
    else:
        _fail("full 档合并顺序异常")
    if _tool_pairs_intact(orch.raw_messages):
        _ok("tool 消息成对完整（无孤立 tool）")
    else:
        _fail("存在孤立 tool 消息")

    # #8 并行只读：执行期间 raw_messages 长度不变
    orch2 = _mock_orch()
    orch2.router.route = lambda inp, hist, summary=None: ["presale", "consult"]
    lens = []
    for key, msgs, delay in (("presale", presale_msgs, 0.2), ("consult", consult_msgs, 0.05)):
        def _make(k, m, d):
            def _h(mm, max_steps=5):
                time.sleep(d)
                lens.append(len(orch2.raw_messages))
                return m[-1]["content"], m
            return _h
        orch2.agents[key].handle = _make(key, msgs, delay)
    orch2._run_result_agent = lambda e, u: "R"
    orch2.chat("再测一轮")
    if all(n == 1 for n in lens):
        _ok("执行期间 raw_messages 长度不变（并行只读，#8）")
    else:
        _fail(f"并行期间 raw_messages 被写入：{lens}")


def a5_partial_failure():
    print("\n[A5] 部分失败不整体失败（#9，mock）")
    orch = _mock_orch()
    ok_msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "retrieve_knowledge", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"results": []}'},
        {"role": "assistant", "content": "咨询报告"},
    ]
    orch.router.route = lambda inp, hist, summary=None: ["presale", "consult"]

    def _boom(m, max_steps=5):
        raise RuntimeError("模拟崩溃")
    orch.agents["presale"].handle = _boom
    orch.agents["consult"].handle = lambda m, max_steps=5: (ok_msgs[-1]["content"], ok_msgs)

    received = {}
    def _result(entries, user_input):
        received["entries"] = entries
        return "Result: 咨询部分如下……售前部分暂时无法回答"
    orch._run_result_agent = _result

    reply = orch.chat("推荐个笔记本，顺便讲讲 OLED")

    statuses = {e.agent: e.status for e in received["entries"]}
    if reply and statuses == {"presale": "failed", "consult": "success"}:
        _ok("presale=failed（error 有值）、consult=success，整体正常返回")
    else:
        _fail(f"部分失败行为异常：statuses={statuses}")
    if any(e.error and "模拟崩溃" in e.error for e in received["entries"]):
        _ok("failed 条目携带 error 原因")
    else:
        _fail("failed 条目缺 error")
    if _tool_pairs_intact(orch.raw_messages) and orch.raw_messages[-1]["content"] == reply:
        _ok("合并只含成功方工具对，Result 文本收尾")
    else:
        _fail("部分失败下合并结构异常")


def a6_return_format():
    print("\n[A6] 返回格式：chat() 返回 str（#13，mock）")
    orch = _mock_orch()
    orch.router.route = lambda inp, hist, summary=None: ["presale"]
    orch.agents["presale"].handle = lambda m, max_steps=5: ("最终答复", [
        {"role": "assistant", "content": "最终答复"},
    ])
    reply = orch.chat("你好")
    if isinstance(reply, str):
        _ok(f"chat() 返回 str（{len(reply)} 字符），无 schema 无提取调用")
    else:
        _fail(f"chat() 返回 {type(reply)}")


def a6b_overflow_retry():
    """循环中途溢出 → 上抛 → 编排器压缩 → 该 Agent 整体重试一次
    （v2.1 裁定：工具全只读，丢弃轨迹从头重跑无正确性风险）。"""
    print("\n[A6b] 循环中途溢出 → 压缩 → 整体重试（mock）")
    orch = _mock_orch()
    orch.compaction_enabled = True   # 本用例需要压缩生效（其余 mock 用例关闭）
    orch.router.route = lambda inp, hist, summary=None: ["presale"]

    big_tool = {"role": "tool", "tool_call_id": "tX",
                "content": '{"data": "' + "x" * 4000 + '"}'}

    def fake_success(user_text):
        """正常轮：一次大工具调用 + 最终答复（撑起可压缩的历史）。"""
        def handle(messages, max_steps=5):
            tc = {"id": "tX", "type": "function",
                  "function": {"name": "search_catalog", "arguments": "{}"}}
            traj = [
                {"role": "assistant", "content": None, "tool_calls": [tc]},
                big_tool,
            ]
            answer = {"role": "assistant", "content": f"关于{user_text}的答复"}
            return answer["content"], traj + [answer]
        return handle

    # 前两轮正常：撑起两轮可压缩历史（单轮历史没有 user 切点，见 find_cut_point）
    orch.agents["presale"].handle = fake_success("第一轮")
    orch.chat("第一轮")
    orch.agents["presale"].handle = fake_success("第二轮")
    orch.chat("第二轮")

    # 第三轮：第一次 handle 中途溢出（模拟第 2 步超限），重入后正常
    calls = {"n": 0}

    def flaky(messages, max_steps=5):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ContextOverflowError("模拟循环中途溢出")
        return fake_success("第三轮")(messages, max_steps=max_steps)

    orch.agents["presale"].handle = flaky

    import app.multi_agent.orchestrator as om
    real_compact = om.compact
    compact_calls = {"n": 0}

    def fake_compact(messages, summary, **kwargs):
        compact_calls["n"] += 1
        cut = 4  # 删掉第一轮（u1 起始的 4 条），保留第二轮 + 当前问题
        return cut, MOCK_SUMMARY

    om.compact = fake_compact

    reply = orch.chat("第三轮")
    om.compact = real_compact

    if compact_calls["n"] >= 1:
        _ok("中途溢出触发编排器 force 压缩（主线程唯一写者）")
    else:
        _fail("未触发编排器压缩")
    if calls["n"] == 2:
        _ok("溢出后整体重试一次：handle 共被调用 2 次（丢弃轨迹，从头重跑）")
    else:
        _fail(f"重试次数异常：handle 调用 {calls['n']} 次")
    if orch.summary:
        _ok(f"压缩摘要已生成（{len(orch.summary)} 字，商品记忆 {len(orch.products)} 款）")
    else:
        _fail("压缩后 summary 为空")
    if isinstance(reply, str) and reply.strip() and "第三轮" in reply:
        _ok("重试后正常返回本轮答复")
    else:
        _fail(f"重试后回复异常: {reply!r}")
    if _tool_pairs_intact(orch.raw_messages):
        _ok("重跑轨迹 tool 对完整")
    else:
        _fail("重跑轨迹 tool 对不完整")


def a7_session_persistence():
    print("\n[A7] 会话持久化与恢复（#12，mock）")
    orch = _mock_orch()
    orch.router.route = lambda inp, hist, summary=None: ["presale"]
    msgs = [{"role": "assistant", "content": "答复一"}]
    orch.agents["presale"].handle = lambda m, max_steps=5: (msgs[0]["content"], msgs)
    orch.chat("第一轮")
    orch.save()

    # 新实例恢复
    orch2 = MultiAgentOrchestrator(session_path=TEST_SESSION)
    orch2.memory_manager.memory_enabled = False
    if len(orch2.raw_messages) == 2 and orch2.raw_messages[-1]["content"] == "答复一":
        _ok(f"新实例恢复 {len(orch2.raw_messages)} 条历史（save_session/load_session）")
    else:
        _fail(f"会话恢复异常：{orch2.raw_messages}")

    # reset 清空
    orch2.reset()
    if orch2.history_size == 0 and not Path(TEST_SESSION).exists():
        _ok("reset 清空历史并删除落盘文件")
    else:
        _fail("reset 行为异常")
    _clean()


# ============================================================
# 二、子系统验收（真调 LLM，MOCK_ONLY 时跳过）
# ============================================================

def s1_memory():
    print("\n[S1] 记忆子系统：STM 更新 + 落盘恢复（#12，真调 LLM）")
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    if not orch.memory_manager.memory_enabled:
        _warn("memory_enabled=False，跳过 S1")
        return
    orch.chat("你好，我叫测试员，只买星海品牌的电子产品")
    stm_facts = orch.memory_manager.stm.facts
    if stm_facts:
        _ok(f"STM 更新（{len(stm_facts)} 条 facts）")
    else:
        _warn("STM 未提取到 facts（模型提取差异，非致命）")
    orch.save()
    orch2 = MultiAgentOrchestrator(session_path=TEST_SESSION)
    if orch2.memory_manager.stm.facts:
        _ok("STM 随会话恢复（restore_stm）")
    else:
        _warn("STM 恢复为空")
    print(f"     STM：{orch2.memory_manager.stm.facts[:2]}")
    _clean()


def s2_compression():
    print("\n[S2] 压缩子系统：token 阈值 + 前缀删除 + summary + 切点避开 tool（真调 LLM）")
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    # 触发阈值 = context_window - reserve_tokens。真实窗口保持不动——摘要调用自身
    # 也要用真实窗口做预算检查，缩小 window 会让压缩永远不可行（小窗口模型连
    # prompt 都装不下）；改用抬高 reserve 把阈值压到基线（prompt+tool schema
    # ≈ 4200 est tokens）之上，使 should_compact 对话增长后确定性触发。
    orch.reserve_tokens = 195700
    orch.keep_recent_tokens = 300

    orch.chat("推荐个 5000 内的耳机")
    orch.chat("要降噪好的，通勤用")
    # 第 3 轮请求前才存在"可对齐的 user 切点"（第 1 个已完成 turn 的下标 > 0），
    # 压缩在轮间触发——2 轮时唯一完成 turn 位于下标 0，按 D12 宁可不压
    orch.chat("差不多就这款了，帮我看看它的口碑")
    if orch.summary:
        _ok(f"压缩已触发（删除前缀后保留 {len(orch.raw_messages)} 条），summary {len(orch.summary or '')} 字")
    else:
        _fail(f"未触发压缩：summary 为空，messages={len(orch.raw_messages)}")
    if orch.raw_messages and orch.raw_messages[0].get("role") == "user":
        _ok("工作历史起点为 user turn（前缀删除保持窗口结构）")
    else:
        _fail(f"窗口起点不是 user turn：{orch.raw_messages[0] if orch.raw_messages else '空'}")
    if orch.summary:
        _ok("summary 非空（增量压缩生效）")
    else:
        _fail("summary 为空")
    if _tool_pairs_intact(orch.raw_messages):
        _ok("压缩切点避开 tool 消息（窗口内 tool 对完整）")
    else:
        _fail("压缩切散了 tool 消息对")
    if orch.products:
        ids = sorted(orch.products)
        _ok(f"商品记忆独立存储累积 {len(ids)} 款：{ids}（含提及价 {sum(1 for v in orch.products.values() if 'price_at_mention' in v)} 条）")
    else:
        _fail("商品记忆为空（product_tracker 未随压缩事务更新）")
    # 压缩后追问（Router 靠 summary 兜底）
    reply = orch.chat("第一款多少钱")
    if isinstance(reply, str) and reply.strip():
        _ok(f"压缩后追问正常（跨压缩指代，{len(reply)} 字符）")
    else:
        _fail("压缩后追问失败")
    _clean()


def s3_rag():
    print("\n[S3] RAG 子系统：retrieve_knowledge 真实召回（真调 LLM）")
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    result = orch.agents["consult"].tool_manager.execute_tool(
        "retrieve_knowledge", {"query": "退换货政策", "top_k": 2},
    )
    if result.get("success") and result["data"].get("results"):
        docs = {r["doc"] for r in result["data"]["results"]}
        _ok(f"RAG 召回 {len(result['data']['results'])} 条（docs: {docs}）")
        _warn("已知限制：知识库当前为政策文档（无选购指南语料，P1 遗留 H5 关联）")
    else:
        _fail(f"RAG 召回失败: {result.get('error')}")
    _clean()


def s4_skills():
    print("\n[S4] 技能子系统：skill catalog 注入（真调 LLM 可选）")
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    if not orch.skill_manager.enabled:
        _warn("skills_enabled=False，跳过 S4")
        return
    catalog = orch.skill_manager.build_catalog_prompt()
    if "可用技能" in catalog or catalog == "":
        _ok(f"skill catalog 构建（{len(catalog)} 字符）")
    else:
        _fail("skill catalog 内容异常")
    # per-agent 注入契约（2026-09-03 起）：pack.skill_catalog 恒空，
    # 目录在 _execute_agents 内用 replace(pack, skill_catalog=...) 注入后再组装
    guide_catalog = orch._skill_catalog_for("presale")
    if guide_catalog:
        from dataclasses import replace

        agent_pack = replace(orch._build_pack(), skill_catalog=guide_catalog)
        msgs = build_working_messages(agent_pack, AGENT_CONFIGS["presale"], "single")
        if msgs[1]["role"] == "system" and msgs[1]["content"] == guide_catalog:
            _ok("skill catalog 独立成第二条 system 消息注入（presale 可见）")
        else:
            _fail("skill catalog 注入位置异常")
    else:
        _warn("presale skill catalog 为空，注入分支未覆盖")
    _clean()


def s5_end_to_end_smoke():
    print("\n[S5] 端到端多意图 smoke（真调 LLM：#5/#11）")
    _clean()
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    reply = orch.chat("推荐个 6000 内的笔记本，顺便讲讲 OLED 和 IPS 屏的区别")
    if not (isinstance(reply, str) and reply.strip()):
        _fail("端到端返回异常")
    _ok(f"端到端成功（reply {len(reply)} 字符）")
    if _tool_pairs_intact(orch.raw_messages):
        _ok("raw_messages 内 tool 对完整")
    else:
        _fail("存在孤立 tool 消息")
    tail = orch.raw_messages[-1]
    if tail.get("role") == "assistant" and not tail.get("tool_calls"):
        _ok("尾部为 assistant 文本")
    else:
        _fail("尾部结构异常")
    # #11 数值保真：reply 中所有商品 ID 必须 ∈ mock 目录（真值层全量 100 款，
    # 而非 search_catalog 前 20——扩容后曾因 limit=20 误报"幻觉"）
    import re

    from app.db.snapshot import PRODUCTS

    mentioned_ids = set(re.findall(r"\b(?:LP|PH|HP)-\d{2}\b", reply))
    valid_ids = set(PRODUCTS)
    bad = mentioned_ids - valid_ids
    if not bad:
        _ok(f"数值保真：reply 中商品 ID {sorted(mentioned_ids) or '（无）'} 均 ∈ 目录")
    else:
        _fail(f"幻觉！reply 中出现目录外的商品 ID：{sorted(bad)}")
    print(f"     回复：{reply[:120]}")
    _clean()


def main():
    print("=" * 60)
    print("  Multi-Agent 架构验收测试（P4 重写版）")
    print("=" * 60)

    try:
        a1_whitelist_and_prompts()
        a2_router_parse()
        a3_single_agent_path()
        a4_multi_agent_merge()
        a5_partial_failure()
        a6_return_format()
        a6b_overflow_retry()
        a7_session_persistence()

        if MOCK_ONLY:
            print("\n（--mock-only：跳过真调 LLM 的子系统用例）")
            print("\n" + "=" * 60)
            print("  🎉 架构验收（mock 部分）全部通过")
            print("=" * 60)
            return

        s1_memory()
        s2_compression()
        s3_rag()
        s4_skills()
        s5_end_to_end_smoke()
    finally:
        _clean()

    print("\n" + "=" * 60)
    print("  🎉 Multi-Agent 架构验收全部通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
