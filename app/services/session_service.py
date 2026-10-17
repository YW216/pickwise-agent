"""会话应用服务（第 4 步）。

负责用户归属、会话 ID 校验、用例编排和前端消息投影；不直接写 SQL。
"""

from __future__ import annotations

from uuid import UUID, uuid4

from app.identity import validate_user_id
from app.services.session_store import (
    SessionState,
    SessionStore,
    SessionSummary,
    VisibleMessage,
    visible_messages,
)


class SessionServiceError(Exception):
    """会话应用服务错误基类。"""


class SessionServiceNotFound(SessionServiceError):
    """会话不存在或不属于当前用户。"""


class SessionService:
    """面向用例的会话服务，依赖 SessionStore 而不是具体数据库。"""

    def __init__(self, store: SessionStore):
        self._store = store

    async def create_session(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        title: str = "新会话",
    ) -> SessionState:
        self._validate_user_id(user_id)
        normalized_session_id = self._normalize_session_id(session_id or str(uuid4()))
        return await self._store.create_session(
            user_id, normalized_session_id, title=title.strip() or "新会话",
        )

    async def list_sessions(self, user_id: str) -> list[SessionSummary]:
        self._validate_user_id(user_id)
        return await self._store.list_sessions(user_id)

    async def get_session(self, user_id: str, session_id: str) -> SessionState:
        self._validate_user_id(user_id)
        state = await self._store.load_session(
            user_id, self._normalize_session_id(session_id),
        )
        if state is None:
            raise SessionServiceNotFound("会话不存在")
        return state

    async def get_visible_messages(
        self,
        user_id: str,
        session_id: str,
    ) -> list[VisibleMessage]:
        state = await self.get_session(user_id, session_id)
        return visible_messages(state.messages, session_id=state.session_id)

    async def save_session(
        self,
        state: SessionState,
        *,
        expected_version: int | None = None,
    ) -> SessionState:
        """保存完整会话快照；聊天链路结束时调用。"""
        self._validate_user_id(state.user_id)
        normalized_session_id = self._normalize_session_id(state.session_id)
        if normalized_session_id != state.session_id:
            state = SessionState(
                user_id=state.user_id,
                session_id=normalized_session_id,
                title=state.title,
                summary=state.summary,
                short_term_memory=state.short_term_memory,
                products=state.products,
                version=state.version,
                status=state.status,
                created_at=state.created_at,
                updated_at=state.updated_at,
                last_message_at=state.last_message_at,
                messages=state.messages,
            )
        return await self._store.save_session(
            state,
            expected_version=expected_version,
        )

    async def delete_session(self, user_id: str, session_id: str) -> None:
        self._validate_user_id(user_id)
        try:
            await self._store.delete_session(
                user_id, self._normalize_session_id(session_id),
            )
        except Exception as exc:
            # 对外统一成应用层的“不存在”，避免暴露存储实现细节。
            from app.services.session_store import SessionNotFoundError

            if isinstance(exc, SessionNotFoundError):
                raise SessionServiceNotFound("会话不存在") from exc
            raise

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        try:
            validate_user_id(user_id)
        except ValueError as exc:
            raise SessionServiceError("用户编号不合法") from exc

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        try:
            return str(UUID(session_id))
        except (ValueError, AttributeError, TypeError) as exc:
            raise SessionServiceError("会话编号必须是 UUID") from exc
