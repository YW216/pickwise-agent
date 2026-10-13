"""FastAPI 依赖函数（多租户隔离和fastapi接入.md §4.1/§4.3）。

依赖从 request.app.state 读取运行对象（lifespan 创建），不在模块 import
时创建——每个 create_app() 实例拥有自己的 SessionManager 与 Auth，
测试可通过 override 或重建 app 隔离，绝不通过全局单例意外共享。
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from app.api.auth import (
    CSRF_HEADER,
    COOKIE_NAME,
    AnonymousAuth,
    AnonymousIdentity,
    AuthError,
)
from app.api.envelope import clear_auth_cookie
from app.config.settings import settings
from app.services.session_manager import SessionManager


def get_auth(request: Request) -> AnonymousAuth:
    return request.app.state.auth


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _unauthorized(message: str) -> HTTPException:
    """401 附带失效 Cookie 清除头（相同 Path，HttpOnly 也能删）。"""
    return HTTPException(status_code=401, detail=message,
                         headers=clear_auth_cookie())


def get_current_user(
    request: Request,
    auth: AnonymousAuth = Depends(get_auth),
) -> AnonymousIdentity:
    """从签名 Cookie 取得可信身份；缺失/非法/过期 → 401 并清除 Cookie。"""
    signed = request.cookies.get(COOKIE_NAME)
    if not signed:
        raise _unauthorized("缺少匿名凭证")
    try:
        return auth.verify(signed)
    except AuthError as exc:
        raise _unauthorized(str(exc)) from exc


def require_csrf(
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
) -> AnonymousIdentity:
    """写请求的 CSRF 与来源校验（/auth/anonymous 是唯一引导例外）。"""
    try:
        get_auth(request).verify_csrf(
            user,
            request.headers.get(CSRF_HEADER),
            request.headers.get("origin"),
            settings.allowed_origins,
        )
    except AuthError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return user
