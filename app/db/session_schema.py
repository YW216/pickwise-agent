"""会话表结构和 PostgreSQL 初始化（第 2 步）。

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
    products          JSONB NOT NULL DEFAULT '{}'::jsonb,
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

CREATE TABLE IF NOT EXISTS session_messages (
    message_id    BIGSERIAL PRIMARY KEY,
    user_id       VARCHAR(128) NOT NULL,
    session_id    VARCHAR(36) NOT NULL,
    turn_id       VARCHAR(128) NOT NULL,
    turn_index    INTEGER NOT NULL CHECK (turn_index >= 0),
    sequence_no   INTEGER NOT NULL CHECK (sequence_no > 0),
    role          VARCHAR(20) NOT NULL
        CHECK (role IN ('user', 'assistant', 'tool')),
    content       TEXT NOT NULL,
    tool_call_id  VARCHAR(128),
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT fk_session_messages_session
        FOREIGN KEY (user_id, session_id)
        REFERENCES sessions (user_id, session_id)
        ON DELETE CASCADE,
    CONSTRAINT uq_session_messages_turn
        UNIQUE (user_id, session_id, turn_id, turn_index),
    CONSTRAINT uq_session_messages_sequence
        UNIQUE (user_id, session_id, sequence_no)
);

CREATE INDEX IF NOT EXISTS idx_session_messages_session_sequence
    ON session_messages (user_id, session_id, sequence_no);
"""


def ensure_session_schema(database_url: str) -> None:
    """幂等创建 sessions 和 session_messages。"""
    with connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SESSION_SCHEMA)
        conn.commit()
