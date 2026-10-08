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


# 其他专家工具块折叠成的中性描述。刻意不出现任何工具名——泄漏的正是
# 「工具名」这个符号本身，出现即等于没降级。
#
# 为什么按语义分类给不同文案：统一说"信息检索"是**错的**。5 个 presale
# 独占工具里只有 2 个真是检索，其余是读取用户偏好 / 商品对比。若把
# "查询用户收藏夹"说成"完成了一次信息检索"，下一轮 consult 会形成错误
# 认知——用户问偏好、上一轮"检索过" → 可能跳过 recall_user_memory 直接
# 编造偏好。**文案失真会诱发本不该发生的幻觉**，比泄漏工具名的代价更大。
#
# 措辞只给"发生过什么类型的事"，不给细节：
#   丢了什么：tool_calls 里的工具名与 arguments、tool 消息里的完整结果。
#   为什么必须连结果一起丢：OpenAI 协议要求 assistant(tool_calls) 与 tool
#   成对出现，拆开会让 API 直接报错——保不住单独的 tool 结果（见 docstring
#   协议约束段）。
#   丢的信息去哪了：① 最终答复原样保留（对话连贯性）；② 商品信息由
#   product_block 独立承载（ID/名称/价格）；③ 共享工具的结果本就保留。
#   即"丢工具怎么执行的，留对话说了什么"。
#
# 键是**语义类别**而非工具名——文案里出现的必须是类别，不能是标识符。
_TOOL_SEMANTIC_KIND: dict[str, str] = {
    "search_products": "检索",
    "search_catalog": "检索",
    "retrieve_knowledge": "检索",
    "get_detail": "查询",
    "compare_products": "对比",
    "get_user_favorites": "用户偏好",
    "recall_user_memory": "用户偏好",
    "load_skill": "流程加载",
}

_FOREIGN_BLOCK_NOTES: dict[str, str] = {
    "检索": "（上一轮完成了商品与知识检索，具体工具与结果已略。）",
    "查询": "（上一轮查询了单个商品信息，具体内容已略。）",
    "对比": "（上一轮完成了商品对比，具体数据已略。）",
    "用户偏好": "（上一轮读取了用户偏好信息，具体内容已略。）",
    "流程加载": "（上一轮加载了操作流程文档。）",
}

# 块内工具跨多个语义类别时用这句——宁可粗，不可给错语义
_FOREIGN_BLOCK_NOTE_GENERIC = "（上一轮由其他专家完成了一次工具调用，内容已略。）"


def _foreign_note(names: list[str]) -> str:
    """按块内工具的语义类别选折叠文案。

    类别一致用该类别的精确文案；跨类别或遇到未登记的工具则退回通用句。
    不判断工具名是否越权（调用方已确保全部越权），只关心**语义类别**。
    """
    kinds = {_TOOL_SEMANTIC_KIND.get(name) for name in names}
    if len(kinds) == 1:
        (kind,) = kinds
        if kind is not None:
            return _FOREIGN_BLOCK_NOTES[kind]
    return _FOREIGN_BLOCK_NOTE_GENERIC


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
    与轨迹失真相比不值得。

    为什么这个组合结构性罕见（不是"懒得处理"，是推导出来的）：
    白名单结构是 presale 独有 5 个 / consult 独有 **0** 个 / 共享 3 个。
    ① consult 无任何独占工具 → 它自己不可能产生含越权工具的调用块，
       故混调只可能出现在 presale 的历史里；
    ② presale 调共享工具时通常已经拿到商品 ID（走"查详情"路径），
       不会再同时调搜索工具——两个工具在同一次响应里并���的前提是
       "还没搜就查详情"，语义上不成立。
    实测四种组合（consult 视角）：纯越权→折叠无泄漏；纯共享→保留（正确）；
    混调→保留并泄漏。残留窗口存在但窄，且后果仅为"多一轮被拒的工具调用"。

    已知取舍二：折叠时原 content 被整句丢弃（换成 _foreign_note(names)）。
    当前模型下无损——推理模型思考走 reasoning_content，落盘 content
    只有面向用户的最终答复，工具调用那一轮实测恒为空字符串。故这是
    **靠实测数据成立、而非设计上保证**。若将来模型把思考写进 content，
    需重新评估：应改为"保留 content 中的用户约束、剥离工具名"，
    否则会连用户约束（如"要 15 寸以内的"）一起丢失——product_block
    只兜商品 ID/名称/价格，兜不住非商品类筛选条件。

    协议约束：tool 消息必须与 assistant(tool_calls) 成对（设计文档 C2，
    develop_docs/模块设计/6-上下文设计.md），故折叠必然整块进出，不拆单条。
    这是"连工具结果一起丢"的根本原因，不是取舍。
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
                out.append({"role": "assistant", "content": _foreign_note(names)})
                i = end
                continue
            out.append(msg)
            out.extend(history[i + 1:end])
            i = end
            continue
        out.append(msg)
        i += 1
    return out
