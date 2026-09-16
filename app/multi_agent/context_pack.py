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
    messages.extend(pack.history)
    return messages
