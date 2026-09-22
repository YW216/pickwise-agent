"""黑板：多 Agent 模式的单轮回执收集器。

设计要点（develop_docs/模块设计/4-多Agent架构设计.md 第七节）：
1. 只在多 Agent 模式存在；单 Agent 无黑板（保持 A 现状）。
2. Agent 间共同写、不互读（纯并行扇出）；只有 Result Agent 读全部条目。
3. 单轮即弃——不跨轮持久化，每轮请求独立。
4. 条目由 Orchestrator 侧组装（Agent 零配合）：new_messages 原样引用进条目，
   格式风险被结构性消除（7.3）。
5. new_messages 最后一条恒为最终答复（10.5 位置约定），按位置切片消费，
   不做字符串比对。
"""

from dataclasses import dataclass

from app.agent.context_budget import tool_result_view
from app.multi_agent.agents import AGENT_CONFIGS


@dataclass
class BlackboardEntry:
    """单个 Agent 的执行回执。"""

    agent: str                # presale / consult
    status: str               # success / failed / overflow（overflow 仅编排器内部过渡态）
    error: str | None         # 失败或溢出原因（status != success 时有值）
    new_messages: list[dict]  # handle() 返回的本轮轨迹（思考/工具对/最终答复），
                              # 原样引用；其中 tool 消息是入库时生成的视图，与模型所见一致


def render_blackboard(
    entries: list[BlackboardEntry], max_result_chars: int | None = None,
) -> str:
    """把黑板全量轨迹渲染成 Result Agent 的输入文本。

    轨迹以文本形态渲染（不扁平化为消息列表，规避无 tools 定义时
    tool_calls 历史消息的 provider 容忍度问题，决策 #13/#14）；
    按 Agent 分组标注来源（【专家名】），对齐信息在渲染层解决。

    Returns:
        成功：各条目按传入顺序拼接的文本块。
        空列表：空字符串。
    """
    parts: list[str] = []
    for entry in entries:
        name = AGENT_CONFIGS[entry.agent]["name"]
        lines = [f"【{name}】（场景: {entry.agent}，状态: {entry.status}）"]
        if entry.status == "failed":
            lines.append(f"失败原因：{entry.error}")
            parts.append("\n".join(lines))
            continue
        last_idx = len(entry.new_messages) - 1
        for i, msg in enumerate(entry.new_messages):
            role = msg.get("role")
            if role == "tool":
                content = msg.get("content", "")
                if max_result_chars is not None:
                    content = tool_result_view(content, max_result_chars)
                lines.append(f"[工具结果] {content}")
            elif msg.get("tool_calls"):
                names = ", ".join(
                    tc["function"]["name"] for tc in msg["tool_calls"]
                )
                thought = msg.get("content") or ""
                prefix = f"[行动] 调用工具 {names}"
                lines.append(f"{prefix} {thought}".rstrip())
            else:  # assistant 纯文本
                tag = "最终答复" if i == last_idx else "思考"
                lines.append(f"[{tag}] {msg.get('content', '')}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
