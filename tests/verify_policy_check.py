"""临时验证：consult 走 policy-check 技能（跑完可删）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.multi_agent.orchestrator import MultiAgentOrchestrator

TEST = ROOT / "app" / "sessions" / "test_policy_check.json"
TEST.unlink(missing_ok=True)

orch = MultiAgentOrchestrator(session_path=str(TEST))
orch.history_threshold = 50

reply = orch.chat("我上个月买了台曜石磐石 15，最近电池越来越不耐用，这种情况能保修吗？")

tools = [
    tc["function"]["name"]
    for m in orch.raw_messages if m.get("role") == "assistant"
    for tc in m.get("tool_calls", [])
]
loaded = "load_skill" in tools and any(
    "policy-check" in m.get("content", "")
    for m in orch.raw_messages if m.get("role") == "tool"
)
warranty = "retrieve_warranty" not in tools  # 2026-09-15 下线

print(f"\n工具轨迹: {tools}")
print(f"load_skill(policy-check): {loaded} | retrieve_warranty 已下线: {warranty}")
print(f"回复 {len(reply)} 字:\n{reply[:400]}")
TEST.unlink(missing_ok=True)
