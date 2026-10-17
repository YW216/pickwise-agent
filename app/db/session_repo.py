"""PostgreSQL 会话仓储（JSONB 快照方案）。

会话完整消息列表保存在 ``sessions.messages`` 中；本模块负责整行快照的读取、
写入、版本校验和用户归属过滤。它不处理 HTTP、Agent 编排或前端投影。
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from app.db.connection import _pool_for
from app.services.session_store import (
    SessionAlreadyExistsError,
    SessionMessage,
    SessionNotFoundError,
    SessionState,
    SessionStore,
    SessionSummary,
    SessionVersionConflictError,
)


class PostgresSessionStore:
    """基于 sessions.messages JSONB 快照的 SessionStore 实现。"""

    def __init__(self, database_url: str):
        self._database_url = database_url

    async def create_session(
        self,
        user_id: str,
        session_id: str,
        *,
        title: str = "新会话",
    ) -> SessionState:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO sessions (user_id, session_id, title, messages)
                    VALUES (%s, %s, %s, '[]'::jsonb)
                    ON CONFLICT (user_id, session_id) DO NOTHING
                    RETURNING user_id
                    """,
                    (user_id, session_id, title),
                )
                if await cur.fetchone() is None:
                    raise SessionAlreadyExistsError("会话已存在")
                state = await self._fetch_state(conn, user_id, session_id)
            await conn.commit()
        return state

    async def list_sessions(self, user_id: str) -> list[SessionSummary]:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT user_id, session_id, title, status,
                           jsonb_array_length(messages)::integer AS message_count,
                           created_at, updated_at, last_message_at
                    FROM sessions
                    WHERE user_id = %s
                    ORDER BY updated_at DESC, session_id DESC
                    """,
                    (user_id,),
                )
                rows = await cur.fetchall()
        return [
            SessionSummary(
                user_id=row[0],
                session_id=row[1],
                title=row[2],
                status=row[3],
                message_count=row[4],
                created_at=row[5],
                updated_at=row[6],
                last_message_at=row[7],
            )
            for row in rows
        ]

    async def load_session(
        self,
        user_id: str,
        session_id: str,
    ) -> SessionState | None:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            try:
                return await self._fetch_state(conn, user_id, session_id)
            except SessionNotFoundError:
                return None

    async def save_session(
        self,
        state: SessionState,
        *,
        expected_version: int | None = None,
    ) -> SessionState:
        """整份快照保存；同一会话使用 version 做乐观锁。"""
        expected = state.version if expected_version is None else expected_version
        payload = [_message_to_json(message) for message in state.messages]
        last_message_at = (
            datetime.now(timezone.utc) if state.messages else state.last_message_at
        )

        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE sessions
                    SET title = %s,
                        summary = %s,
                        messages = %s,
                        short_term_memory = %s,
                        products = %s,
                        version = version + 1,
                        status = %s,
                        updated_at = now(),
                        last_message_at = %s
                    WHERE user_id = %s
                      AND session_id = %s
                      AND version = %s
                    RETURNING version
                    """,
                    (
                        state.title,
                        state.summary,
                        Jsonb(payload),
                        Jsonb(state.short_term_memory or {}),
                        Jsonb(state.products or {}),
                        state.status,
                        last_message_at,
                        state.user_id,
                        state.session_id,
                        expected,
                    ),
                )
                updated = await cur.fetchone()
                if updated is None:
                    await cur.execute(
                        """
                        SELECT 1
                        FROM sessions
                        WHERE user_id = %s AND session_id = %s
                        """,
                        (state.user_id, state.session_id),
                    )
                    if await cur.fetchone() is None:
                        raise SessionNotFoundError("会话不存在")
                    raise SessionVersionConflictError(
                        f"会话版本冲突：expected={expected}"
                    )
                saved = await self._fetch_state(
                    conn, state.user_id, state.session_id,
                )
            await conn.commit()
        return saved

    async def delete_session(self, user_id: str, session_id: str) -> None:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM sessions
                    WHERE user_id = %s AND session_id = %s
                    """,
                    (user_id, session_id),
                )
                if cur.rowcount != 1:
                    raise SessionNotFoundError("会话不存在")
            await conn.commit()

    async def _fetch_state(
        self,
        conn,
        user_id: str,
        session_id: str,
    ) -> SessionState:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT user_id, session_id, title, summary, messages,
                       short_term_memory, products, version, status,
                       created_at, updated_at, last_message_at
                FROM sessions
                WHERE user_id = %s AND session_id = %s
                """,
                (user_id, session_id),
            )
            row = await cur.fetchone()
        if row is None:
            raise SessionNotFoundError("会话不存在")

        return SessionState(
            user_id=row[0],
            session_id=row[1],
            title=row[2],
            summary=row[3],
            messages=tuple(_message_from_json(item) for item in (row[4] or [])),
            short_term_memory=row[5] or {},
            products=row[6] or {},
            version=row[7],
            status=row[8],
            created_at=row[9],
            updated_at=row[10],
            last_message_at=row[11],
        )


def _message_to_json(message: SessionMessage) -> dict[str, Any]:
    payload = asdict(message)
    if message.created_at is not None:
        payload["created_at"] = message.created_at.isoformat()
    return payload


def _message_from_json(payload: dict[str, Any]) -> SessionMessage:
    created_at = payload.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    return SessionMessage(
        role=payload["role"],
        content=payload.get("content", ""),
        message_id=payload.get("message_id"),
        tool_call_id=payload.get("tool_call_id"),
        metadata=payload.get("metadata") or {},
        created_at=created_at,
    )


def session_store_for(database_url: str) -> SessionStore:
    """构造默认 PostgreSQL Store，方便应用组装和测试替换。"""
    return PostgresSessionStore(database_url)
