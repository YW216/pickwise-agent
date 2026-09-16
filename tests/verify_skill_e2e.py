"""临时验证：真调 LLM——guide 是否走通 load_skill → recall_user_memory → search_catalog（跑完可删）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.multi_agent.orchestrator import MultiAgentOrchestrator

TEST = str(Path(__file__).resolve().parent.parent / "app" / "sessions" / "test_skill_e2e.json")
Path(TEST).unlink(missing_ok=True)

orch = MultiAgentOrchestrator(session_path=TEST)
orch.history_threshold = 50

reply = orch.chat("我想买个耳机，帮我推荐一下")

print("\n" + "=" * 60)
print("工具调用轨迹：")
for m in orch.raw_messages:
    if m.get("role") == "assistant" and m.get("tool_calls"):
        for tc in m["tool_calls"]:
            print(f"  调用: {tc['function']['name']}({tc['function']['arguments'][:60]})")
    elif m.get("role") == "tool":
        c = m.get("content", "")
        ok = '"success": true' in c or '"success":true' in c
        print(f"  结果: {'✅ 成功' if ok else '❌ 失败'} | {c[:80]}")

loaded = any(
    tc["function"]["name"] == "load_skill"
    for m in orch.raw_messages if m.get("role") == "assistant"
    for tc in m.get("tool_calls", [])
)
recalled = any(
    tc["function"]["name"] == "recall_user_memory"
    for m in orch.raw_messages if m.get("role") == "assistant"
    for tc in m.get("tool_calls", [])
)
print(f"\nload_skill 被调用: {loaded}")
print(f"recall_user_memory 被调用: {recalled}")
print(f"回复 {len(reply)} 字：{reply[:100]}")
Path(TEST).unlink(missing_ok=True)
