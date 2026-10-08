"""请求上下文预算与工具结果视图；不修改持久化的原始消息。

错误判定的两个正交维度（两者都刻意收窄，宁可漏判也不误判）：
- `is_context_overflow`：是不是**上下文超长** → 可压缩重试
- `is_transient`：是不是**瞬时故障** → 可降级回答
把二者混为一谈会出事：400 参数错误压缩无用，连接中断压缩也无用。
"""

import json
import math

from openai import (
    APIConnectionError, APIStatusError, APITimeoutError, BadRequestError, RateLimitError,
)


class ContextOverflowError(RuntimeError):
    """子 Agent 任意步超窗，由编排器压缩后重放；写工具需另有幂等保护。"""


def is_transient(exc: Exception) -> bool:
    """SDK 重试耗尽后的可降级故障；不把普通 400/鉴权错误和代码 bug 纳入。

    只认明确 SDK 异常类型及传输状态，不靠异常文案猜测；
    与上下文溢出判定正交。应用侧不重复执行 SDK 的 HTTP 重试。
    """
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and (
        exc.status_code in {408, 409, 429} or 500 <= exc.status_code < 600
    )


def is_context_overflow(exc: Exception) -> bool:
    """只识别上下文长度错误，不把鉴权、限流或普通参数错误当作超窗。"""
    if not isinstance(exc, BadRequestError):
        return False
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    code = error.get("code") if isinstance(error, dict) else None
    return code in {"context_length_exceeded", "context_window_exceeded"} or any(
        marker in str(exc).lower()
        for marker in ("maximum context length", "context length exceeded", "context window exceeded")
    )


def estimate_text_tokens(text: str) -> int:
    """按中文约 1.5 token、ASCII 约 3 字符/token 做保守估算。

    这是启发式而非 tokenizer；真实 prompt_tokens 由 SubAgent 记录以供校准。
    """
    ascii_chars = sum(ord(char) < 128 for char in text)
    unicode_bytes = len(text.encode("utf-8")) - ascii_chars
    return math.ceil(ascii_chars / 3 + unicode_bytes / 2)


def estimate_message_tokens(message: dict) -> int:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    return estimate_text_tokens(payload) + 4  # +4：role 等消息包装字段的固定开销


def estimate_messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(message) for message in messages) + 3  # +3：对话拼装的固定模板开销


def estimate_tool_definitions_tokens(tool_definitions: list[dict]) -> int:
    if not tool_definitions:
        return 0
    return estimate_text_tokens(json.dumps(tool_definitions, ensure_ascii=False)) + 16  # +16：tools 定义块的固定包装开销


def tool_result_view(content: str, max_chars: int) -> str:
    """超限结果转为明确标注的 JSON 预览，全文由调用方留在 transcript 中。

    预算分配：头尾各占 7/16，余下 1/8 留给 JSON 骨架与字符串转义膨胀——head/tail
    里的引号会被转义成 \\" ，实际比原文更长。若内容引号过于密集导致仍超限，
    逐步回缩；**硬约束是 len(view) <= max_chars**。

    字符计数按**原文口径**给出：`shown_chars = len(head) + len(tail)`，
    `omitted_chars = original_chars - shown_chars`（都不含 JSON 骨架与转义），
    所以 `shown + omitted` 恒等于 `original_chars`——模型据此就能判断缺了多少，
    不必自己从原文长度里减。
    """
    if max_chars < 256:
        raise ValueError("工具结果预览上限不能小于 256 字符")
    if len(content) <= max_chars:
        return content

    preview_chars = (max_chars - max_chars // 8) // 2
    while True:
        head = content[:preview_chars]
        tail = content[-preview_chars:] if preview_chars else ""
        view = json.dumps(
            {
                "truncated": True,
                "original_chars": len(content),
                "shown_chars": len(head) + len(tail),
                "omitted_chars": len(content) - len(head) - len(tail),
                "notice": "工具结果已截断，请缩小查询范围后重查；不可推断省略部分。",
                "head": head,
                "tail": tail,
            },
            ensure_ascii=False,
        )
        if len(view) <= max_chars or preview_chars == 0:
            return view
        preview_chars = preview_chars * 4 // 5  # 回缩 20%，避免一次砍太狠
