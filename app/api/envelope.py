"""HTTP 统一信封与错误映射（多租户隔离和fastapi接入.md §4.5）。

信封形状：{"success", "error": {code, message}|null, "data", "request_id"}。
request_id 只做日志追踪，不是请求去重凭证。状态码映射原则：
- 不把所有失败映射成同一种状态码——按 code 分派（429/503/404/422/...）；
- 未预期的内部故障对外只返回通用错误，诊断细节留在服务端日志。
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi.responses import JSONResponse

from app.api.auth import COOKIE_NAME
from app.services.session_manager import (
    SessionBusy,
    SessionCapacityFull,
    SessionBusy,
    SessionClosing,
    SessionManagerClosed,
    SessionManagerError,
    SessionNotFound,
)

MAX_BODY_BYTES = 64 * 1024  # 请求体大小上限（初值 64 KiB，按实际字节校验）


# SessionManager 异常 → (HTTP 状态码, 稳定错误码)
_SESSION_ERROR_MAP: list[tuple[type[Exception], int, str]] = [
    (SessionNotFound, 404, "SESSION_NOT_FOUND"),
    (SessionCapacityFull, 503, "SESSION_CAPACITY_FULL"),
    (SessionBusy, 429, "SESSION_BUSY"),
    (SessionBusy, 429, "SESSION_BUSY"),
    (SessionClosing, 503, "SESSION_CLOSING"),
    (SessionManagerClosed, 503, "SERVICE_STOPPING"),
]


def new_request_id() -> str:
    return uuid4().hex


def success(data: dict[str, Any], request_id: str, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": True, "error": None, "data": data,
                 "request_id": request_id},
    )


def error(status_code: int, code: str, message: str,
          request_id: str, data: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"success": False,
                 "error": {"code": code, "message": message},
                 "data": data or {},
                 "request_id": request_id},
    )


def session_error(exc: SessionManagerError, request_id: str) -> JSONResponse:
    """SessionManager 异常按映射表转 HTTP；未知子类兜底 500。"""
    for exc_type, status, code in _SESSION_ERROR_MAP:
        if isinstance(exc, exc_type):
            return error(status, code, str(exc), request_id)
    return error(500, "SESSION_INTERNAL_ERROR", "服务内部错误", request_id)


def clear_auth_cookie() -> dict[str, str]:
    """失效凭证的清除头：用相同 Path 让浏览器删除 Cookie。"""
    return {"Set-Cookie": f"{COOKIE_NAME}=; Path=/api; Max-Age=0; HttpOnly"}
