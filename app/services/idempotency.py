"""聊天请求幂等存储。

幂等键与 request_id 的职责不同：前者防止客户端重试导致 Agent 重复执行，
后者只用于日志追踪。生产默认使用 PostgreSQL；测试可以注入内存实现。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.db.connection import _pool_for, connect

ClaimAction = Literal["execute", "replay", "in_progress", "conflict"]

CHAT_IDEMPOTENCY_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_idempotency_records (
    user_id          VARCHAR(128) NOT NULL,
    session_id       VARCHAR(36) NOT NULL,
    idempotency_key  VARCHAR(128) NOT NULL,
    request_hash     CHAR(64) NOT NULL,
    status           VARCHAR(16) NOT NULL
        CHECK (status IN ('processing', 'succeeded', 'failed')),
    status_code      INTEGER NOT NULL DEFAULT 200,
    response         JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at       TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (user_id, session_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_chat_idem_expires
    ON chat_idempotency_records (expires_at);
"""


@dataclass(frozen=True)
class IdempotencyClaim:
    """幂等开始阶段的判定结果。"""

    action: ClaimAction
    response: dict[str, Any] | None = None
    status_code: int | None = None


class IdempotencyStore(Protocol):
    async def begin(
        self,
        user_id: str,
        session_id: str,
        idempotency_key: str,
        request_hash: str,
        *,
        ttl_seconds: int,
    ) -> IdempotencyClaim: ...

    async def finish(
        self,
        user_id: str,
        session_id: str,
        idempotency_key: str,
        request_hash: str,
        *,
        status: Literal["succeeded", "failed"],
        status_code: int,
        response: dict[str, Any],
    ) -> None: ...


def hash_request(payload: dict[str, Any]) -> str:
    """对请求体做稳定哈希，防止同一 Key 被复用于不同请求。"""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_idempotency_key(value: str) -> str:
    """校验 HTTP Header 中的幂等键，拒绝空白和控制字符。"""
    if not isinstance(value, str):
        raise ValueError("Idempotency-Key 必须是字符串")
    value = value.strip()
    if not 1 <= len(value) <= 128:
        raise ValueError("Idempotency-Key 长度必须在 1 到 128 个字符之间")
    if any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise ValueError("Idempotency-Key 不能包含空白或控制字符")
    return value


def ensure_schema(database_url: str) -> None:
    """初始化聊天幂等表，供 init_pg.py 和部署脚本调用。"""
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(CHAT_IDEMPOTENCY_SCHEMA)
        conn.commit()


class InMemoryIdempotencyStore:
    """测试用实现；不用于生产，生命周期与应用实例一致。"""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], dict[str, Any]] = {}
        # 测试可能在多个 TestClient 事件循环之间复用同一个 store，
        # 因此这里使用线程锁而不是绑定事件循环的 asyncio.Lock。
        self._lock = threading.Lock()

    async def begin(
        self,
        user_id: str,
        session_id: str,
        idempotency_key: str,
        request_hash: str,
        *,
        ttl_seconds: int,
    ) -> IdempotencyClaim:
        key = (user_id, session_id, idempotency_key)
        with self._lock:
            now = time.time()
            record = self._records.get(key)
            if record is not None and record["expires_at"] <= now:
                del self._records[key]
                record = None
            if record is None:
                self._records[key] = {
                    "request_hash": request_hash,
                    "status": "processing",
                    "status_code": 200,
                    "response": None,
                    "expires_at": now + ttl_seconds,
                }
                return IdempotencyClaim("execute")
            if record["request_hash"] != request_hash:
                return IdempotencyClaim("conflict")
            if record["status"] == "processing":
                return IdempotencyClaim("in_progress")
            return IdempotencyClaim(
                "replay",
                response=copy.deepcopy(record["response"]),
                status_code=record["status_code"],
            )

    async def finish(
        self,
        user_id: str,
        session_id: str,
        idempotency_key: str,
        request_hash: str,
        *,
        status: Literal["succeeded", "failed"],
        status_code: int,
        response: dict[str, Any],
    ) -> None:
        key = (user_id, session_id, idempotency_key)
        with self._lock:
            record = self._records.get(key)
            if record is None or record["request_hash"] != request_hash:
                raise RuntimeError("幂等记录不存在或请求指纹不匹配")
            record.update(
                status=status,
                status_code=status_code,
                response=copy.deepcopy(response),
            )


class PostgresIdempotencyStore:
    """生产实现：依赖 PostgreSQL 唯一约束完成跨请求原子抢占。"""

    def __init__(self, database_url: str):
        self._database_url = database_url

    async def begin(
        self,
        user_id: str,
        session_id: str,
        idempotency_key: str,
        request_hash: str,
        *,
        ttl_seconds: int,
    ) -> IdempotencyClaim:
        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM chat_idempotency_records
                    WHERE user_id = %s AND session_id = %s
                      AND idempotency_key = %s AND expires_at <= now()
                    """,
                    (user_id, session_id, idempotency_key),
                )
                await cur.execute(
                    """
                    INSERT INTO chat_idempotency_records
                        (user_id, session_id, idempotency_key, request_hash,
                         status, expires_at)
                    VALUES (%s, %s, %s, %s, 'processing',
                            now() + (%s * interval '1 second'))
                    ON CONFLICT (user_id, session_id, idempotency_key) DO NOTHING
                    RETURNING status
                    """,
                    (user_id, session_id, idempotency_key, request_hash, ttl_seconds),
                )
                inserted = await cur.fetchone()
                if inserted:
                    await conn.commit()
                    return IdempotencyClaim("execute")

                await cur.execute(
                    """
                    SELECT request_hash, status, status_code, response
                    FROM chat_idempotency_records
                    WHERE user_id = %s AND session_id = %s
                      AND idempotency_key = %s
                    FOR UPDATE
                    """,
                    (user_id, session_id, idempotency_key),
                )
                row = await cur.fetchone()
                if row is None:
                    raise RuntimeError("幂等记录在抢占后丢失")
                stored_hash, status, status_code, response = row
                await conn.commit()

        if stored_hash != request_hash:
            return IdempotencyClaim("conflict")
        if status == "processing":
            return IdempotencyClaim("in_progress")
        return IdempotencyClaim(
            "replay", response=response, status_code=int(status_code),
        )

    async def finish(
        self,
        user_id: str,
        session_id: str,
        idempotency_key: str,
        request_hash: str,
        *,
        status: Literal["succeeded", "failed"],
        status_code: int,
        response: dict[str, Any],
    ) -> None:
        from psycopg.types.json import Jsonb

        pool = await _pool_for(self._database_url)
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE chat_idempotency_records
                    SET status = %s, status_code = %s, response = %s,
                        updated_at = now()
                    WHERE user_id = %s AND session_id = %s
                      AND idempotency_key = %s AND request_hash = %s
                    """,
                    (
                        status, status_code, Jsonb(response), user_id, session_id,
                        idempotency_key, request_hash,
                    ),
                )
                if cur.rowcount != 1:
                    raise RuntimeError("幂等记录不存在或请求指纹不匹配")
            await conn.commit()
