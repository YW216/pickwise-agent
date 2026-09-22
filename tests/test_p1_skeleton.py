"""P1 骨架验证：白名单隔离 / prompt-schema 一致性 / Router 多值解析 / 端到端 smoke。

对应设计文档（develop_docs/模块/4-多Agent架构设计.md）十五节验证清单：
- #1 白名单隔离（售前 9 / 咨询 3；售前调 retrieve_warranty 返回"未知工具"）
- #2 prompt 声明工具 vs 白名单一致性
- #3 Router 多值解析用例表（mock client，不真调 LLM）
- #10 result_str 类型（tool 消息 content 全程为 str，bug #4 修复验证，随端到端检查）
（2026-09-04 业务重构：guide+compare 融合为 presale；2026-09-05 双 Agent 定稿——
售后 Agent 回滚删除，retrieve_warranty 归咨询）

用法：python tests/test_p1_skeleton.py
"""

import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.agents import AGENT_CONFIGS  # noqa: E402
from app.multi_agent.orchestrator import MultiAgentOrchestrator  # noqa: E402
from app.multi_agent.router import Router  # noqa: E402
from app.prompts.agents import (  # noqa: E402
    CONSULT_MULTI_PROMPT,
    CONSULT_PROMPT,
    PRESALE_MULTI_PROMPT,
    PRESALE_PROMPT,
)

TEST_SESSION = str(ROOT / "app" / "sessions" / "test_p1_session.json")


def _fresh_orchestrator(threshold: int = 30, keep: int = 6) -> MultiAgentOrchestrator:
    orch = MultiAgentOrchestrator(session_path=TEST_SESSION)
    orch.history_threshold = threshold
    orch.history_keep_recent = keep
    return orch


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


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")
    sys.exit(1)


# ---------- 验证 1：白名单隔离（十五节 #1） ----------
def test_whitelist():
    print("\n[1/4] 工具白名单隔离（售前 8 / 咨询 3）")
    expected = {"presale": 8, "consult": 3}

    for key, count in expected.items():
        names = set(AGENT_CONFIGS[key]["tools"])
        if len(names) != count:
            _fail(f"{key} 白名单数量 {len(names)} != {count}")
    _ok("AGENT_CONFIGS 白名单数量 presale=8 / consult=3 正确")

    orch = _fresh_orchestrator()
    for key in expected:
        actual = {
            d["function"]["name"]
            for d in orch.agents[key].tool_manager.tool_definitions
        }
        if actual != set(AGENT_CONFIGS[key]["tools"]):
            _fail(f"{key} ToolManager 过滤结果与白名单不一致：{sorted(actual)}")
    _ok("ToolManager 白名单过滤正确")

    result = orch.agents["presale"].tool_manager.execute_tool(
        "retrieve_warranty", {"category": "笔记本"},
    )
    if result.get("success") is False and "未知工具" in (result.get("error") or ""):
        _ok("presale 调 retrieve_warranty → 未知工具信封")
    else:
        _fail(f"presale 调 retrieve_warranty 应返回未知工具信封，实际 {result}")


# ---------- 验证 2：prompt 声明工具 vs 白名单一致性（十五节 #2） ----------
def _extract_prompt_tools(prompt: str) -> set[str]:
    """从 prompt 的「## 可用工具」段抽取加粗工具名。"""
    m = re.search(r"## 可用工具\n(.*?)(?=\n## |\Z)", prompt, re.DOTALL)
    if not m:
        return set()
    return set(re.findall(r"\*\*([a-z_]+)\*\*", m.group(1)))


def test_prompt_schema_consistency():
    print("\n[2/4] prompt 声明工具 vs 白名单一致性（4 份：single + multi）")
    prompts = {
        "presale": PRESALE_PROMPT,
        "consult": CONSULT_PROMPT,
        "presale-multi": PRESALE_MULTI_PROMPT,
        "consult-multi": CONSULT_MULTI_PROMPT,
    }
    for key, prompt in prompts.items():
        agent_key = key.replace("-multi", "")
        declared = _extract_prompt_tools(prompt)
        whitelist = set(AGENT_CONFIGS[agent_key]["tools"])
        if declared == whitelist:
            _ok(f"{key}: 声明 {len(declared)} 个 = 白名单 {len(whitelist)} 个")
        else:
            _fail(
                f"{key} 不一致：prompt 声明 {sorted(declared)} vs "
                f"白名单 {sorted(whitelist)}"
            )


# ---------- 验证 3：Router 多值解析用例表（十五节 #3，mock） ----------
class _FakeClient:
    """支撑 Router.route 的最小假 client：返回固定文本并记录调用参数。"""

    def __init__(self, content: str):
        self._content = content
        self.last_kwargs: dict | None = None

    @property
    def chat(self):
        return SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_router_parse():
    print("\n[3/4] Router 多值解析用例表（mock，不真调 LLM）")
    cases = [
        ("presale", ["presale"]),
        ("PRESALE", ["presale"]),                           # 大写容忍
        ("presale,consult", ["presale", "consult"]),
        ("presale，consult", ["presale", "consult"]),       # 中文逗号
        ("consult,presale", ["presale", "consult"]),        # 固定顺序
        ("consult,consult", ["consult"]),                   # 去重
        ("  presale , consult ", ["presale", "consult"]),   # 空白容忍
        ("presale,,consult", ["presale", "consult"]),       # 空段容忍
        ("", ["consult"]),                                  # 空 → 兜底
        ("乱码 xyz", ["consult"]),                          # 全非法 → 兜底
    ]
    for raw, expected in cases:
        router = Router(_FakeClient(raw), "test-model")
        actual = router.route("测试输入")
        if actual == expected:
            _ok(f"「{raw}」→ {actual}")
        else:
            _fail(f"「{raw}」期望 {expected}，实际 {actual}")

    client = _FakeClient("presale")
    router = Router(client, "test-model")
    router.route("测试输入")
    mt = (client.last_kwargs or {}).get("max_tokens")
    if mt == 2048:
        _ok("max_tokens=2048 已生效（推理模型思考需预算；2026-09-16 由 512 上调）")
    else:
        _fail(f"max_tokens 应为 2048，实际 {mt}")


# ---------- 验证 4：端到端 smoke（真调 LLM，单场景） ----------
def test_end_to_end_smoke():
    print("\n[4/4] 端到端 smoke（真调 LLM，单场景 → 返回 str）")
    _clean()
    orch = _fresh_orchestrator()

    reply = orch.chat("有什么适合编程的笔记本推荐吗")

    if isinstance(reply, str) and reply.strip():
        _ok(f"chat() 返回非空 str（{len(reply)} 字符）")
    else:
        _fail(f"chat() 应返回非空 str，实际 {type(reply)}: {str(reply)[:80]}")

    tool_msgs = [m for m in orch.raw_messages if m.get("role") == "tool"]
    if tool_msgs and all(isinstance(m.get("content"), str) for m in tool_msgs):
        _ok(f"tool 消息 content 全为 str（{len(tool_msgs)} 条，bug #4 修复验证）")
    elif not tool_msgs:
        _ok("本轮未触发工具调用（跳过 result_str 类型检查）")
    else:
        _fail("存在 tool 消息 content 非 str")

    tail = orch.raw_messages[-1]
    if tail.get("role") == "assistant" and not tail.get("tool_calls"):
        _ok("raw_messages 尾部为 assistant 文本（无 JSON 尾巴）")
    else:
        _fail(f"raw_messages 尾部异常：{str(tail)[:100]}")

    print(f"     回复：{reply[:120]}")


def main():
    print("=" * 60)
    print("  P1 骨架验证（多 Agent 设计 v2.1 · 十五节 #1/#2/#3/#10）")
    print("=" * 60)

    try:
        test_whitelist()
        test_prompt_schema_consistency()
        test_router_parse()
        test_end_to_end_smoke()
    finally:
        _clean()

    print("\n" + "=" * 60)
    print("  🎉 P1 骨架全部验证通过")
    print("=" * 60)


if __name__ == "__main__":
    main()
