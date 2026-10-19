"""记忆查询工具：让 Agent 在 ReAct 循环中主动查询用户记忆。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agent.tools.result import fail, ok

if TYPE_CHECKING:
    from app.agent.memory.manager import MemoryManager

def recall_user_memory(query: str = "", *, manager: MemoryManager | None = None) -> dict:
    """查询当前用户的记忆信息（长期记忆和短期记忆）。"""
    if manager is None:
        return fail("缺少用户记忆上下文")
    if not manager.memory_enabled:
        return fail("记忆系统未启用")

    manager.ltm.load()
    payload: dict = {
        "short_term_facts": list(manager.stm.facts),
        # 用户画像：四类固定槽位（constraint/reject/scene/weight），按 key 唯一
        "profile": [
            {
                "slot": f.slot,
                "key": f.key,
                "value": f.value,
                "quote": f.quote,
                "evidence": f.evidence,
                "updated_at": f.updated_at,
            }
            for f in manager.ltm.facts
        ],
    }
    if manager.ltm.interaction_summaries:
        payload["recent_interactions"] = [
            s["summary"] for s in manager.ltm.interaction_summaries[-3:]
        ]
    return ok(payload)
