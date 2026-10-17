"""测试专用 FastAPI 应用工厂。

生产代码只使用 ``create_app()``。测试需要替换 Agent、鉴权密钥、会话目录或
幂等存储时，从这里创建隔离应用，避免测试参数进入生产入口。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from app.api.app import _build_app


def create_test_app(*, auth_secret_key: str,
                    session_dir: str | Path,
                    factory=None,
                    idempotency_store=None,
                    session_service=None) -> FastAPI:
    """创建测试应用；不读取生产鉴权密钥，也不强制连接 PostgreSQL 会话池。"""
    if len(auth_secret_key.encode()) < 32:
        raise ValueError("测试 AUTH_SECRET_KEY 至少 32 字节")
    return _build_app(
        secret=auth_secret_key,
        session_dir=session_dir,
        factory=factory,
        idempotency_store=idempotency_store,
        session_service=session_service,
    )
