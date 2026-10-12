"""技能加载工具：让 Agent 在 ReAct 循环中按需加载 Skill 指令。

由 ToolManager 绑定实例的 SkillManager，Agent 按需获取技能指令。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.agent.tools.result import fail

if TYPE_CHECKING:
    from app.agent.skills.loader import SkillManager

def load_skill(skill_name: str, *, manager: SkillManager | None = None) -> dict:
    """加载指定技能的完整指令。Agent 调用后按指令处理用户问题。"""
    if manager is None:
        return fail("技能系统未启用")
    return manager.load_skill(skill_name)
