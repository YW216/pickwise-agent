"""请求上下文预算与工具结果视图；不修改持久化的原始消息。"""

import json
import math

from openai import BadRequestError


class ContextOverflowError(RuntimeError):
    """子 Agent 首次请求超窗，尚未执行工具，可以由编排器压缩后重试。"""


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
    """超限结果转为明确标注的 JSON 预览，全文由调用方留在 transcript 中。"""
    if max_chars < 256:
        raise ValueError("工具结果预览上限不能小于 256 字符")
    if len(content) <= max_chars:
        return content
    preview_chars = (max_chars - 200) // 12  # 200 字符预留给截断标注 JSON 骨架（函数保证 max_chars >= 256，结果恒为正）
    payload = {
        "truncated": True,
        "original_chars": len(content),
        "notice": "工具结果已截断，请缩小查询范围后重查；不可推断省略部分。",
        "head": content[:preview_chars],
        "tail": content[-preview_chars:] if preview_chars else "",
    }
    return json.dumps(payload, ensure_ascii=False)


def prepare_messages(messages: list[dict], tool_result_max_chars: int) -> list[dict]:
    """构建私有请求视图，历史与本轮工具结果使用相同的长度限制。"""
    return [
        {**message, "content": tool_result_view(message.get("content") or "", tool_result_max_chars)}
        if message.get("role") == "tool" else dict(message)
        for message in messages
    ]
