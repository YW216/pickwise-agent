"""ContextPack：一次请求的完整上下文，构建一次、N 个 Agent 各取全量。

设计要点（develop_docs/模块设计/6-上下文设计.md）：
1. history 是"本轮进入上下文的消息窗口"，不是"会话全量"——装的是
   orchestrator 的工作历史（raw_messages：未压缩部分；已压缩段以 summary
   承载），压缩方案的替换不改本模块字段与签名（约束 C1，前向兼容判据）。
2. build_working_messages 是模块级自由函数：无状态、输入输出明确（项目风格）。
3. skill catalog 独立成第二条 system 消息（不拼进 prompt 字符串尾部），
   与 memory/summary 段顺序固定，利于 provider 前缀缓存命中。
4. product_block（v2.1）与 skill_catalog 同模式：预渲染文本字段、空串省略。
   商品记忆（product_tracker）的存储态是 dict，注入时独立成 system 消息，
   不拼进 summary 文本——叙述与结构化数据在 pack 层就是两段。
5. history 按 Agent 白名单做视图投影（_project_history，2026-10-07）：
   共享历史里含**其他 Agent** 的 tool_calls 轨迹，模型看到邻居用过某工具会
   模仿去调它，而执行层必然拒收（_TOOL_MAP 只含白名单）——白费两轮 LLM。
   本模块按"整块"判定：块内工具全部越权则折叠为一句中性文本，共享工具
   正常保留。降级只作用于**入站视图**，不改 raw_messages 事实。
"""

from dataclasses import dataclass


@dataclass
class ContextPack:
    """一次请求的完整上下文。构建一次，N 个 Agent 各取全量。"""

    history: list[dict]          # 本轮进入上下文的消息窗口（含刚追加的 user 消息；只读引用）
    summary: str | None          # 历史压缩摘要（六段叙述，被压缩段的代表；纯 LLM 叙述）
    memory_sections: list[dict]  # MemoryManager.build_memory_prompt_sections()
    skill_catalog: str           # SkillManager.build_catalog_prompt()，可为 ""
    product_block: str = ""      # 商品记忆渲染块（product_tracker.format_products_block），可为 ""


def build_working_messages(pack: ContextPack, cfg: dict, mode: str) -> list[dict]:
    """用 pack + Agent 配置组装该 Agent 的 working messages。

    Args:
        pack：本轮上下文（orchestrator._build_pack() 的产物，多 Agent 共享同一引用）
        cfg：AGENT_CONFIGS[key]，取 cfg["prompts"][mode] 作 system prompt
        mode："single" | "multi"（P2 阶段恒为 single，P3 起按场景数选择）

    Returns:
        消息列表，固定顺序：system(prompt) + system(skill) + memory 段
        + summary 段 + product_block 段 + history 全量。空字段对应段直接省略。
    """
    messages = [{"role": "system", "content": cfg["prompts"][mode]}]
    if pack.skill_catalog:
        messages.append({"role": "system", "content": pack.skill_catalog})
    messages.extend(pack.memory_sections)
    if pack.summary:
        messages.append({
            "role": "system",
            "content": f"以下是此前对话的摘要，用于延续上下文记忆：\n{pack.summary}",
        })
    if pack.product_block:
        messages.append({
            "role": "system",
            "content": (
                "以下是本会话提及过的商品记录（商品 ID、名称与提及当时的价格，"
                f"价格为历史价、不代表现价）：\n{pack.product_block}"
            ),
        })
    messages.extend(_project_history(pack.history, cfg.get("tools") or ()))
    return messages


# 其他专家的工具块折叠成的中性描述。刻意不出现任何工具名——泄漏的正是
# 「工具名」这个符号本身，出现即等于没降级。措辞给"发生过检索"的事实，
# 不给"检索到了什么"（那是 product_block / summary 的职责，避免两处重复）。
_FOREIGN_BLOCK_NOTE = "（上一轮由其他专家完成了一次信息检索，具体工具与结果已略。）"


def _project_history(history: list[dict], allowed_tools) -> list[dict]:
    """按 Agent 白名单投影历史：越权工具块折叠为中性文本，其余原样保留。

    问题是"符号泄漏"而非权限泄漏：raw_messages 是共享的，里面有其他 Agent
    的 tool_calls，模型模仿邻居调白名单外的工具，执行层必然拒收（fail
    「未知工具」），白费两轮 LLM。执行层那道硬兜底使最坏情况只是效率损失，
    但根因在上下文构造，不该指望 prompt 文字约束（软约束）去兜。

    判定粒度是**整块**（一条 assistant(tool_calls) + 其后连续 tool 消息）：
    - 块内工具名全部不在白名单 → 折叠为一条中性 assistant 文本
    - 只要有一个在白名单内 → 整块原样保留（共享工具本就该给两个 Agent 看）

    已知限制（刻意接受）：若同一次响应里混调了越权工具与共享工具
    （如 presale 同时调 search_products + get_detail），整块保留会带走
     越权的那一个。要彻底消除需按 call 粒度摘除，但会把并行调用轨迹拆成
    "只查了详情、没搜列表"的失真形态，反而更可能诱发重复查询——两轮浪费
    与轨迹失真相比不值得。当前 2 Agent / 8:3 工具配置下该组合非主路径。

    协议约束：tool 消息必须与 assistant(tool_calls) 成对（设计文档 C2），
    故折叠必然整块进出，不拆单条。
    """
    if not allowed_tools:
        return list(history)  # 无白名单信息（如测试构造的 cfg）→ 不投影

    out: list[dict] = []
    i, n = 0, len(history)
    while i < n:
        msg = history[i]
        tool_calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and tool_calls:
            # 吃掉紧随其后的连续 tool 消息，构成一个完整调用块
            end = i + 1
            while end < n and history[end].get("role") == "tool":
                end += 1
            names = [tc["function"]["name"] for tc in tool_calls]
            if all(name not in allowed_tools for name in names):
                out.append({"role": "assistant", "content": _FOREIGN_BLOCK_NOTE})
                i = end
                continue
            out.append(msg)
            out.extend(history[i + 1:end])
            i = end
            continue
        out.append(msg)
        i += 1
    return out
