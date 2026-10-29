"""PostgreSQL storage for users and server-side login sessions."""

from __future__ import annotations

from uuid import uuid4

from app.api.auth import AuthStore, AuthenticatedUser, EmailAlreadyRegistered, LoginSession
from app.db.connection import _pool_for, connect

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(254) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name VARCHAR(80) NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'disabled')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token_hash CHAR(64) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    csrf_token TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions (expires_at);
"""


def ensure_auth_schema(database_url: str) -> None:
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(AUTH_SCHEMA)
        conn.commit()


class PostgresAuthStore(AuthStore):
    def __init__(self, database_url: str):
        self._database_url = database_url

    async def create_user(self, email: str, password_hash: str, display_name: str) -> AuthenticatedUser:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO users (user_id, email, password_hash, display_name) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (email) DO NOTHING "
                    "RETURNING user_id, email, display_name",
                    (str(uuid4()), email, password_hash, display_name),
                )
                row = await cur.fetchone()
                if row is None:
                    raise EmailAlreadyRegistered("这个邮箱已注册")
            await conn.commit()
        return AuthenticatedUser(*row)

    async def get_user_by_email(self, email: str) -> tuple[AuthenticatedUser, str] | None:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT user_id, email, display_name, password_hash "
                    "FROM users WHERE email = %s AND status = 'active'",
                    (email,),
                )
                row = await cur.fetchone()
        if row is None:
            return None
        return AuthenticatedUser(row[0], row[1], row[2]), row[3]  
        #`row` 表示「一行」，但对这一行做下标访问时，下标指的是 列 ——因为一行本身就是一个按列顺序排列的元组。

    async def create_session(self, token_hash: str, user_id: str, csrf_token: str, expires_at: float) -> None:
        from datetime import datetime, timezone

        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "INSERT INTO auth_sessions (token_hash, user_id, csrf_token, expires_at) "
                    "VALUES (%s, %s, %s, %s)",
                    (token_hash, user_id, csrf_token, datetime.fromtimestamp(expires_at, timezone.utc)),
                )
            await conn.commit()

    async def get_session(self, token_hash: str) -> LoginSession | None:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT u.user_id, u.email, u.display_name, s.csrf_token, "
                    "extract(epoch from s.expires_at) "
                    "FROM auth_sessions s JOIN users u USING (user_id) "
                    "WHERE s.token_hash = %s AND s.revoked_at IS NULL "
                    "AND s.expires_at > now() AND u.status = 'active'",
                    (token_hash,),
                )
                row = await cur.fetchone()
                if row is not None:
                    await cur.execute("UPDATE auth_sessions SET last_seen_at = now() WHERE token_hash = %s", (token_hash,))
            if row is not None:
                await conn.commit()
        return LoginSession(AuthenticatedUser(*row[:3]), row[3], float(row[4])) if row else None

    async def revoke_session(self, token_hash: str) -> None:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE auth_sessions SET revoked_at = now() "
                    "WHERE token_hash = %s AND revoked_at IS NULL",
                    (token_hash,),
                )
            await conn.commit()
