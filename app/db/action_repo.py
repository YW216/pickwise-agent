"""待确认动作：写操作意图的持久化与状态机。

写工具不直接落库，只在这里留下一条 pending 记录；用户确认后由一次独立的
HTTP 请求认领并执行。认领用条件更新完成「校验 + 抢占」，因此重复确认、
并发确认、跨用户确认都在一条 SQL 里被挡住。
"""

from __future__ import annotations

from secrets import token_urlsafe

from psycopg.types.json import Jsonb

from app.db.connection import _pool_for, connect
from app.identity import validate_user_id

# 待确认动作的有效期。必须短于商品 ref 的 1 小时 TTL，否则用户确认时 ref 已失效。
ACTION_TTL_SECONDS = 30 * 60

# 认领后允许执行的最长时间。超过视为进程中断，回收为 failed 供用户重试。
EXECUTING_LEASE_SECONDS = 2 * 60


def init_action_schema(database_url: str) -> None:
    """建表（幂等）。服务启动与测试初始化时各调一次。"""
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            # 与 user_repo / session_schema 同样的做法：防止两个初始化进程同时建表。
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (2026110501,))
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pending_actions (
                    action_id    TEXT PRIMARY KEY,
                    user_id      VARCHAR(128) NOT NULL,
                    session_id   TEXT NOT NULL,
                    tool         TEXT NOT NULL,
                    arguments    JSONB NOT NULL,
                    preview      JSONB NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    result       JSONB,
                    error        TEXT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at   TIMESTAMPTZ NOT NULL,
                    executing_at TIMESTAMPTZ,
                    finished_at  TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS pending_actions_user_status_idx
                    ON pending_actions (user_id, status);
            """)
        conn.commit()


async def create_action(database_url: str, *, user_id: str, session_id: str,
                        tool: str, arguments: dict, preview: dict) -> dict:
    """登记一条待确认动作，返回编号与过期时间。调用方不要在这里执行副作用。"""
    validate_user_id(user_id)
    action_id = token_urlsafe(16)
    sql = """
        INSERT INTO pending_actions
            (action_id, user_id, session_id, tool, arguments, preview, expires_at)
        VALUES (%s, %s, %s, %s, %s, %s, now() + make_interval(secs => %s))
        RETURNING action_id, expires_at
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (
                action_id, user_id, session_id, tool,
                Jsonb(arguments), Jsonb(preview), ACTION_TTL_SECONDS,
            ))
            row = await cur.fetchone()
    return {"action_id": row[0], "expires_at": row[1].isoformat()}


async def claim_action(database_url: str, *, action_id: str, user_id: str) -> dict | None:
    """认领并转入 executing；抢不到（已处理 / 已过期 / 不属于本人）返回 None。"""
    validate_user_id(user_id)
    sql = """
        UPDATE pending_actions
           SET status = 'executing', executing_at = now(), error = NULL
         WHERE action_id = %s
           AND user_id = %s
           AND status IN ('pending', 'failed')
           AND expires_at > now()
        RETURNING action_id, tool, arguments, session_id
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (action_id, user_id))
            row = await cur.fetchone()
    if row is None:
        return None
    return {"action_id": row[0], "tool": row[1], "arguments": row[2], "session_id": row[3]}


async def finish_action(database_url: str, *, action_id: str, status: str,
                        result: dict | None = None, error: str | None = None) -> None:
    """写终态。只从 executing 落地，避免覆盖已被回收或取消的记录。"""
    sql = """
        UPDATE pending_actions
           SET status = %s, result = %s, error = %s, finished_at = now()
         WHERE action_id = %s AND status = 'executing'
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (
                status, None if result is None else Jsonb(result), error, action_id,
            ))


async def cancel_action(database_url: str, *, action_id: str, user_id: str) -> str:
    """用户取消。返回记录最终状态；不存在或不属于本人返回 "missing"。"""
    validate_user_id(user_id)
    sql = """
        UPDATE pending_actions
           SET status = 'cancelled', finished_at = now()
         WHERE action_id = %s AND user_id = %s AND status = 'pending'
        RETURNING status
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (action_id, user_id))
            row = await cur.fetchone()
            if row is None:
                await cur.execute(
                    "SELECT status FROM pending_actions WHERE action_id = %s AND user_id = %s",
                    (action_id, user_id),
                )
                existing = await cur.fetchone()
    if row is not None:
        return row[0]
    return existing[0] if existing is not None else "missing"


async def get_action(database_url: str, *, action_id: str, user_id: str) -> dict | None:
    """按编号取一条记录（含已完成/已过期的），供确认接口判断该回什么。"""
    validate_user_id(user_id)
    sql = """
        SELECT action_id, tool, preview, status, result, error, expires_at
          FROM pending_actions
         WHERE action_id = %s AND user_id = %s
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (action_id, user_id))
            row = await cur.fetchone()
    if row is None:
        return None
    return {
        "action_id": row[0], "tool": row[1], "preview": row[2], "status": row[3],
        "result": row[4], "error": row[5], "expires_at": row[6].isoformat(),
    }


async def list_pending_actions(database_url: str, *, user_id: str,
                               session_id: str | None = None) -> list[dict]:
    """列出仍可确认的动作；前端刷新页面后靠它重建卡片。"""
    validate_user_id(user_id)
    sql = """
        SELECT action_id, tool, preview, expires_at
          FROM pending_actions
         WHERE user_id = %s
           AND status = 'pending'
           AND expires_at > now()
           AND (%s::text IS NULL OR session_id = %s::text)
         ORDER BY created_at
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, (user_id, session_id, session_id))
            rows = await cur.fetchall()
    return [
        {"action_id": action_id, "tool": tool, "preview": preview,
         "expires_at": expires_at.isoformat()}
        for action_id, tool, preview, expires_at in rows
    ]


async def reclaim_stale_actions(database_url: str) -> dict:
    """回收过期与卡死记录：pending 过期 → expired；executing 超租约 → failed。"""
    expired_sql = """
        UPDATE pending_actions
           SET status = 'expired', finished_at = now()
         WHERE status = 'pending' AND expires_at <= now()
    """
    stale_sql = """
        UPDATE pending_actions
           SET status = 'failed', error = '执行超时，可重试', finished_at = now()
         WHERE status = 'executing'
           AND executing_at IS NOT NULL
           AND executing_at < now() - make_interval(secs => %s)
    """
    async with (await _pool_for(database_url)).connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(expired_sql)
            expired = cur.rowcount
            await cur.execute(stale_sql, (EXECUTING_LEASE_SECONDS,))
            stale = cur.rowcount
    return {"expired": expired, "failed": stale}
