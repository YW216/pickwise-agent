"""HTTP 请求 schema（多租户隔离和fastapi接入.md §4.4）。

全部请求体 extra="forbid"：拒绝试图设置 user_id 等额外字段——
用户编号只能来自凭证，不接受请求正文指定。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SessionCreateRequest(BaseModel):
    """POST /api/sessions：空对象（服务端生成全部编号）。"""

    model_config = ConfigDict(extra="forbid")


class ChatRequest(BaseModel):
    """POST /api/chat：窗口编号 + 消息正文。"""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=36, max_length=36)
    message: str = Field(min_length=1, max_length=8000)

    @field_validator("message")
    @classmethod
    def strip_and_require(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("消息不能为空白")
        return stripped


class CompareRequest(BaseModel):
    """POST /api/products/compare：2~4 个不同商品 ID。"""

    model_config = ConfigDict(extra="forbid")

    product_ids: list[str] = Field(min_length=2, max_length=4)
