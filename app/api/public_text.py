"""对外消息文本处理，隐藏内部商品编号。"""

from __future__ import annotations

import re


_INTERNAL_PRODUCT_ID = re.compile(r"(?<![A-Za-z0-9])(?:LP|PH|HP)-\d{2}(?!\d)")


_PRODUCT_ID_PARENTHESES = re.compile(
    r"\s*[（(]\s*(?:LP|PH|HP)-\d{2}\s*[）)]"
)


_PRODUCT_ID_PAREN_PREFIX = re.compile(
    r"([（(])\s*(?:LP|PH|HP)-\d{2}\s*[，,、]?\s*"
)


def public_assistant_text(content: str) -> str:
    """隐藏内部商品主键；名称、价格和参数仍保持原样。"""
    without_parentheses = _PRODUCT_ID_PARENTHESES.sub("", content)
    without_prefix = _PRODUCT_ID_PAREN_PREFIX.sub(r"\1", without_parentheses)
    return _INTERNAL_PRODUCT_ID.sub("候选商品", without_prefix)


def public_message_content(role: str, content: str) -> str:
    return public_assistant_text(content) if role == "assistant" else content
