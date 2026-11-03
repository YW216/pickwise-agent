"""账号路由：注册、登录、登录态查询与退出。"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.api.auth import AuthError, AuthenticatedUser, COOKIE_NAME, EmailAlreadyRegistered
from app.api.deps import get_auth_service, get_current_user, require_csrf
from app.api.envelope import clear_auth_cookie, error, new_request_id, success
from app.api.schemas import LoginRequest, RegisterRequest
from app.config.settings import settings


auth_router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth_payload(user: AuthenticatedUser, csrf_token: str, expires_at: float) -> dict:
    return {
        "authenticated": True,
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "display_name": user.display_name,
        },
        "csrf_token": csrf_token,
        "expires_at": expires_at,
    }


def _set_login_cookie(response: JSONResponse, token: str, expires_at: float) -> None:
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=max(0, int(expires_at - time.time())),
        expires=max(0, int(expires_at - time.time())),
        httponly=True,
        samesite="lax",
        secure=settings.auth_cookie_secure,
        path="/api",
    )


@auth_router.post("/register")
async def register(
    body: RegisterRequest,
    request: Request,
    auth=Depends(get_auth_service),
):
    request_id = new_request_id()
    try:
        user, token, csrf_token, expires_at = await auth.register(
            body.email, body.password, body.display_name,
        )
    except EmailAlreadyRegistered as exc:
        return error(409, "EMAIL_ALREADY_REGISTERED", str(exc), request_id)
    except AuthError as exc:
        return error(422, "AUTH_INPUT_INVALID", str(exc), request_id)
    response = success(_auth_payload(user, csrf_token, expires_at), request_id, 201)
    _set_login_cookie(response, token, expires_at)
    return response


@auth_router.post("/login")
async def login(
    body: LoginRequest,
    request: Request,
    auth=Depends(get_auth_service),
):
    request_id = new_request_id()
    try:
        user, token, csrf_token, expires_at = await auth.login(body.email, body.password)
    except AuthError as exc:
        return error(401, "INVALID_CREDENTIALS", str(exc), request_id)
    response = success(_auth_payload(user, csrf_token, expires_at), request_id)
    _set_login_cookie(response, token, expires_at)
    return response


@auth_router.get("/me")
async def current_user(
    request: Request,
    user: AuthenticatedUser | None = Depends(get_current_user),
):
    request_id = new_request_id()
    if user is None:
        response = success({"authenticated": False, "user": None}, request_id)
        if request.cookies.get(COOKIE_NAME):
            response.headers.update(clear_auth_cookie())
        return response
    expires_at = request.state.auth_expires_at
    return success({
        "authenticated": True,
        "user": {"user_id": user.user_id, "email": user.email,
                 "display_name": user.display_name},
        "csrf_token": request.state.csrf_token,
        "expires_at": expires_at,
    }, request_id)


@auth_router.post("/logout")
async def logout(
    request: Request,
    user: AuthenticatedUser = Depends(require_csrf),
    auth=Depends(get_auth_service),
):
    await auth.logout(request.cookies.get(COOKIE_NAME))
    response = success({"authenticated": False}, new_request_id())
    response.headers.update(clear_auth_cookie())
    return response
