"""会话表结构和 PostgreSQL 初始化（JSONB 快照方案）。

当前仓库的 ``sql/`` 目录属于本地忽略目录，因此把正式建表定义收口在可导入的
Python 模块中，由 ``app/scripts/init_pg.py`` 调用，保证冷启动与测试使用同一份 DDL。
"""

from __future__ import annotations

from app.db.connection import connect

SESSION_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id           VARCHAR(128) NOT NULL,
    session_id        VARCHAR(36) NOT NULL,
    title             VARCHAR(200) NOT NULL DEFAULT '新会话',
    summary           TEXT,
    short_term_memory JSONB NOT NULL DEFAULT '{}'::jsonb,
    messages          JSONB NOT NULL DEFAULT '[]'::jsonb,
    products          JSONB NOT NULL DEFAULT '{}'::jsonb,
    recommendations   JSONB NOT NULL DEFAULT '[]'::jsonb,
    version           INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    status            VARCHAR(20) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'archived')),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_message_at   TIMESTAMPTZ,
    PRIMARY KEY (user_id, session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
    ON sessions (user_id, updated_at DESC);

-- 兼容已经创建过旧版 sessions 表的本地数据库。
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS messages JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS recommendations JSONB NOT NULL DEFAULT '[]'::jsonb;
"""


def ensure_session_schema(database_url: str) -> None:
    """幂等创建/补齐 JSONB 会话快照字段。"""
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SESSION_SCHEMA)
        conn.commit()
