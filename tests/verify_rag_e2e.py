"""临时验证：真调 LLM——知识类问题是否命中选购语料（跑完可删）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.orchestrator import MultiAgentOrchestrator

TEST = ROOT / "app" / "sessions" / "test_rag_corpus.json"
TEST.unlink(missing_ok=True)

orch = MultiAgentOrchestrator(session_path=str(TEST))
orch.history_threshold = 50

reply = orch.chat("OLED 和 IPS 屏幕有什么区别？我该选哪种")

print("\n工具调用轨迹：")
for m in orch.raw_messages:
    if m.get("role") == "assistant" and m.get("tool_calls"):
        for tc in m["tool_calls"]:
            print(f"  调用: {tc['function']['name']}({tc['function']['arguments'][:50]})")
    elif m.get("role") == "tool":
        ok = "success" in m.get("content", "")[:30]
        has_guide = "选购" in m.get("content", "") or "面板" in m.get("content", "")
        print(f"  结果: {'✅' if ok else '❌'} | 含选购语料: {has_guide} | {m.get('content', '')[:80]}")

mentioned = ("屏幕面板科普" in reply) or ("选购指南" in reply) or ("根据知识库" in reply)
fallback = "通用常识" in reply
print(f"\n回复 {len(reply)} 字 | 引用语料: {mentioned} | 仍走通用常识兜底: {fallback}")
print(f"回复开头: {reply[:120]}")
TEST.unlink(missing_ok=True)
