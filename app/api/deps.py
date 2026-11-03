"""FastAPI dependencies for server-side authentication and CSRF checks."""

from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request

from app.api.auth import AuthError, AuthenticatedUser, COOKIE_NAME, CSRF_HEADER
from app.api.envelope import clear_auth_cookie
from app.config.settings import settings
from app.identity import SessionIdentity
from app.services.session_manager import SessionManager
from app.services.session_service import SessionService


def get_auth_service(request: Request):
    return request.app.state.auth_service


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def get_session_service(request: Request) -> SessionService:
    service = getattr(request.app.state, "session_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="会话服务未配置")
    return service


async def get_current_user(request: Request) -> AuthenticatedUser | None:
    token = request.cookies.get(COOKIE_NAME)
    session = await get_auth_service(request).authenticate(token)
    if session is None:
        return None
    user, csrf_token, expires_at = session
    request.state.csrf_token = csrf_token
    request.state.auth_expires_at = expires_at
    return user


async def require_authenticated_user(
    request: Request,
    user: AuthenticatedUser | None = Depends(get_current_user),
) -> AuthenticatedUser:
    if user is None:
        raise HTTPException(
            status_code=401,
            detail={"code": "LOGIN_REQUIRED", "message": "请先登录"},
            headers=clear_auth_cookie(),
        )
    return user


async def require_csrf(
    request: Request,
    user: AuthenticatedUser = Depends(require_authenticated_user),
) -> AuthenticatedUser:
    expected = getattr(request.state, "csrf_token", "")
    supplied = request.headers.get(CSRF_HEADER, "")
    if not supplied or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="CSRF 校验失败")
    origin = request.headers.get("origin")
    if origin and settings.allowed_origins and origin not in settings.allowed_origins:
        raise HTTPException(status_code=403, detail="来源不在允许列表内")
    return user


def identity_for(user: AuthenticatedUser, session_id: str) -> SessionIdentity:
    """凭证 user_id + 请求方窗口编号组身份；UUID 非法 → 422。

    归属隔离由 SessionManager 的文件定位天然保证：窗口文件只会在
    当前用户目录下查找，别人的窗口对本用户一律不存在（统一 404）。
    """
    try:
        return SessionIdentity(user_id=user.user_id, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="窗口编号必须是 UUID") from exc
