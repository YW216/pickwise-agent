"""Test FastAPI factory with isolated stores and overridable services."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from app.api.app import _build_app
from app.db.auth_repo import PostgresAuthStore
from app.config.settings import settings


def create_test_app(*, session_dir: str | Path,
                    factory=None,
                    idempotency_store=None,
                    session_service=None,
                    auth_store=None) -> FastAPI:
    """Create an isolated test app; auth_store must be injected explicitly."""
    if auth_store is None:
        if session_service is None:
            raise RuntimeError(
                "create_test_app 需要显式注入 auth_store"
                "（内存实现见 tests/in_memory_auth_store.py）"
            )
        auth_store = PostgresAuthStore(settings.database_url)
    return _build_app(
        session_dir=session_dir,
        factory=factory,
        idempotency_store=idempotency_store,
        session_service=session_service,
        auth_store=auth_store,
        production_tracing=False,
    )
