"""会话持久化的数据契约（第 1 步）。

本模块只定义服务层与持久化层之间的稳定数据结构和接口，不包含 SQL、HTTP
或 Agent 编排。后续可以用 PostgreSQL、文件或内存实现同一份 SessionStore。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Protocol

SessionStatus = Literal["active", "archived"]
MessageRole = Literal["user", "assistant", "tool"]


class SessionStoreError(Exception):
    """会话持久化层错误基类。"""


class SessionNotFoundError(SessionStoreError):
    """会话不存在或不属于当前用户。"""


class SessionAlreadyExistsError(SessionStoreError):
    """创建会话时主键已存在。"""


class SessionVersionConflictError(SessionStoreError):
    """乐观锁版本不匹配。"""


@dataclass(frozen=True)
class SessionMessage:
    """一条完整 Agent 消息；tool/tool_calls 也保存在 JSONB 列表中。"""

    role: MessageRole
    content: str
    message_id: int | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True)
class VisibleMessage:
    """前端可见消息；不携带 user_id、turn_id 或内部 metadata。"""

    session_id: str
    role: Literal["user", "assistant"]
    content: str
    message_id: int | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class SessionSummary:
    """会话列表摘要；message_count 只统计前端可见消息，不含工具轨迹。"""

    user_id: str
    session_id: str
    title: str
    status: SessionStatus
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None = None


@dataclass(frozen=True)
class SessionState:
    """恢复 Agent 所需的完整会话状态。"""

    user_id: str
    session_id: str
    title: str
    summary: str | None
    short_term_memory: dict[str, Any]
    products: dict[str, Any]
    version: int
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime | None
    messages: tuple[SessionMessage, ...] = ()


class SessionStore(Protocol):
    """会话持久化协议；实现层负责数据库/文件细节。"""

    async def create_session(
        self,
        user_id: str,
        session_id: str,
        *,
        title: str = "新会话",
    ) -> SessionState:
        """创建一个空会话。"""
        ...

    async def list_sessions(self, user_id: str) -> list[SessionSummary]:
        """按最近更新时间返回指定用户的会话摘要。"""
        ...

    async def load_session(
        self,
        user_id: str,
        session_id: str,
    ) -> SessionState | None:
        """加载完整上下文；不存在或不属于该用户时返回 None。"""
        ...

    async def save_session(
        self,
        state: SessionState,
        *,
        expected_version: int | None = None,
    ) -> SessionState:
        """原子保存完整会话快照，并执行乐观锁校验。"""
        ...

    async def delete_session(self, user_id: str, session_id: str) -> None:
        """删除会话及其消息。"""
        ...


def visible_messages(
    messages: list[SessionMessage] | tuple[SessionMessage, ...],
    *,
    session_id: str,
) -> list[VisibleMessage]:
    """返回前端默认展示的消息投影。

    数据库保存完整 Agent 轨迹；普通聊天界面只显示用户消息和有正文的助手最终回复，
    不直接暴露 tool 消息或 assistant 的 tool_calls metadata。
    """
    visible: list[VisibleMessage] = []
    for message in messages:
        if message.role == "user":
            visible.append(
                VisibleMessage(
                    session_id=session_id,
                    role="user",
                    content=message.content,
                    message_id=message.message_id,
                    created_at=message.created_at,
                )
            )
        elif (
            message.role == "assistant"
            and message.content.strip()
            and not message.metadata.get("tool_calls")
        ):
            visible.append(
                VisibleMessage(
                    session_id=session_id,
                    role="assistant",
                    content=message.content,
                    message_id=message.message_id,
                    created_at=message.created_at,
                )
            )
    return visible
