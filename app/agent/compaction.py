"""会话压缩：超出预算时把最老的完整 turn 摘要化，并从工作历史中删除。

压缩器是无状态模块级纯函数：只计算切点与新摘要，不修改传入的列表；
真正的前缀删除由编排器在摘要校验通过后执行（先摘要、后删除的事务顺序），
任何失败都保持原状态，零副作用。商品记忆（mentioned-products）独立于本模块，
见 product_tracker.py——压缩器只负责六段叙述摘要。
"""

from __future__ import annotations

import re
from typing import Optional

from openai import OpenAI

from app.agent.context_budget import estimate_message_tokens, estimate_text_tokens
from app.agent.summarizer import summarize


_SUMMARY_SECTIONS = (
    "## 用户目标",
    "## 约束与偏好",
    "## 进展",
    "## 关键决策",
    "## 下一步",
    "## 关键上下文",
)


def should_compact(
    context_tokens: int,
    context_window: int,
    reserve_tokens: int,
    enabled: bool = True,
) -> bool:
    """判断上下文是否超过为回复预留空间后的安全线。"""
    if not enabled:
        return False
    return context_tokens > context_window - reserve_tokens


def find_cut_point(
    messages: list[dict],
    keep_recent_tokens: int,
    end: Optional[int] = None,
) -> int:
    """寻找保留区起点（= 将被删除的前缀长度），只在已完成 user turn 边界切分。

    从最新消息往回累积 token，够数后在附近对齐最近的 user 起点——
    最近的上下文最重要，保护方向是逆向的（不是"找哪里能切"，
    而是"找哪里值得留"）。返回 0 表示没有可安全压缩的完整 turn（宁可不压）。

    ``end`` 是排他的边界。请求进行中时，编排器传入当前 user 消息索引，
    从而不会把当前尚未完成的 turn 压进摘要。
    """
    if keep_recent_tokens <= 0:
        raise ValueError("keep_recent_tokens 必须大于 0")

    end = len(messages) if end is None else end
    if type(end) is not int or not 0 <= end <= len(messages):
        raise ValueError("end 超出消息列表范围")

    # 合法切点 = user turn 起点（index 0 除外：切在 0 等于什么都不删）。
    # 只在 user 处切天然不拆开 user → assistant(tool_calls) → tool → assistant 链。
    cut_points = [
        index
        for index in range(1, end)
        if messages[index].get("role") == "user"
    ]
    if not cut_points:
        return 0

    accumulated = 0
    for index in range(end - 1, -1, -1):
        accumulated += estimate_message_tokens(messages[index])
        if accumulated >= keep_recent_tokens:
            # 向前对齐 user 起点，保留整个 turn，而不是只保留该轮结尾。
            for point in reversed(cut_points):
                if point <= index:
                    return point
            return 0
    return 0


def _validate_summary(summary: str) -> str:
    # 规范化：整体首尾与每行行尾去空白——容忍 LLM 输出的尾随空格与 CRLF 残留，
    # 避免逐字符比较把"## 用户目标 "这类正常抖动判为格式失败。
    summary = "\n".join(line.rstrip() for line in summary.strip().splitlines())
    if not summary:
        raise ValueError("压缩摘要为空")
    headings = re.findall(r"^## .+$", summary, flags=re.MULTILINE)
    if headings != list(_SUMMARY_SECTIONS):
        raise ValueError("压缩摘要缺少固定 section 或顺序不正确")
    for part in re.split(r"^## .+$", summary, flags=re.MULTILINE)[1:]:
        if not part.strip():
            raise ValueError("压缩摘要包含空 section")
    return summary


def compact(
    messages: list[dict],
    summary: str | None,
    client: OpenAI,
    model: str,
    keep_recent_tokens: int,
    end: Optional[int] = None,
    summary_max_tokens: int = 2048,
    summary_max_chars: int = 12000,
    context_window: int = 200000,
    tool_result_max_chars: int = 12000,
) -> tuple[int, str | None]:
    """计算一次压缩，返回 ``(cut, new_summary)``。

    cut 是将被删除的前缀长度（即 ``messages[:cut]``），0 表示没有可压缩内容；
    新摘要只基于本段增量（delta）+ 旧摘要增量生成，不做全量重述。
    本函数不修改 messages——调用方在摘要校验通过并确认能缩小上下文后，
    执行 ``del messages[:cut]`` 并替换 summary。任何校验失败都会抛出异常，
    由调用方保持原状态（先摘要、后删除，零副作用）。
    """
    cut = find_cut_point(messages, keep_recent_tokens, end=end)
    if cut <= 0:
        return 0, summary

    delta = messages[:cut]

    def generate() -> str:
        raw = summarize(
            client=client,
            model=model,
            old_messages=delta,
            prev_summary=summary,
            max_tokens=summary_max_tokens,
            context_window=context_window,
            tool_result_max_chars=tool_result_max_chars,
        )
        return _validate_summary(raw.strip())

    try:
        new_summary = generate()
    except ValueError:
        # 摘要格式偶发抖动（标题变体、空 section 等）重试一次再放弃；
        # 确定性失败（如输入超预算）在第二次调用中同样快速抛出，不放大成本。
        new_summary = generate()

    if len(new_summary) > summary_max_chars or estimate_text_tokens(new_summary) > summary_max_tokens:
        raise ValueError("压缩摘要超过长度上限")
    return cut, new_summary
