"""HTTP 路由：参数解析、身份验证、窗口归属、调用后端、映射响应（§4.1）。

路由不写业务逻辑：聊天走 SessionManager 队列，商品/收藏走有界执行器里的
工具函数（与 Agent 共用同一注册表路径）。所有响应统一信封（§4.5）。
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.agent.tools.catalog import search_catalog
from app.agent.tools.detail import compare_products, get_detail
from app.agent.tools.favorites import get_user_favorites
from app.agent.tools.search_products import search_products
from app.api.auth import (
    COOKIE_NAME,
    AnonymousIdentity,
    AuthError,
)
from app.api.deps import get_auth, get_current_user, get_session_manager, require_csrf
from app.api.envelope import (
    clear_auth_cookie,
    error,
    new_request_id,
    session_error,
    success,
)
from app.api.schemas import ChatRequest, CompareRequest, SessionCreateRequest
from app.config.settings import settings
from app.identity import SessionIdentity
from app.services.idempotency import hash_request, validate_idempotency_key
from app.services.session_manager import (
    ChatOutcome,
    SessionManager,
    SessionManagerError,
)

sessions_router = APIRouter(prefix="/api/sessions", tags=["sessions"])
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
products_router = APIRouter(prefix="/api/products", tags=["products"])
favorites_router = APIRouter(prefix="/api/favorites", tags=["favorites"])
health_router = APIRouter(prefix="/api", tags=["health"])


# ---------- 匿名身份（CSRF 引导例外：不要求先有 CSRF 值，§4.3） ----------

@auth_router.post("/anonymous")
async def anonymous_auth(
    request: Request,
    auth=Depends(get_auth),
):
    """无有效 Cookie 时签发新身份；已有有效凭证则复用（不延长有效期）。"""
    request_id = new_request_id()
    signed = request.cookies.get(COOKIE_NAME)
    if signed:
        try:
            identity = auth.verify(signed)
        except AuthError:
            # 失效/被篡改：401 + 服务端清除失效 Cookie，不直接签发新身份
            resp = error(401, "AUTH_INVALID",
                         "匿名凭证已失效，请确认后重新进入", request_id)
            resp.headers.update(clear_auth_cookie())
            return resp
        resp = success(_auth_payload(identity), request_id)
        # 复用原凭证：同一个 CSRF 值与真实到期信息，不延长原期限（§4.2）
        resp.set_cookie(
            COOKIE_NAME, signed,
            max_age=max(0, int(identity.expires_at - time.time())),
            httponly=True, samesite="lax", secure=settings.auth_cookie_secure,
            path="/api",
        )
        return resp

    identity, signed_value = auth.issue()
    resp = success(_auth_payload(identity), request_id)
    resp.set_cookie(
        COOKIE_NAME, signed_value, max_age=int(auth.max_age_seconds),
        httponly=True, samesite="lax", secure=settings.auth_cookie_secure,
        path="/api",
    )
    return resp


def _auth_payload(identity: AnonymousIdentity) -> dict:
    return {
        "user_id": identity.user_id,
        "csrf_token": identity.csrf_token,
        "expires_at": identity.expires_at,
    }


# ---------- 会话与聊天 ----------

@sessions_router.post("")
async def create_session(
    body: SessionCreateRequest,
    request: Request,
    user: AnonymousIdentity = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
):
    """创建窗口：服务端生成 UUID 并持久化空会话归属；不调用模型。"""
    request_id = new_request_id()
    identity = await manager.create_session(user.user_id)
    return success(
        {"session_id": identity.session_id, "user_id": identity.user_id},
        request_id, status_code=201,
    )


@chat_router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    user: AnonymousIdentity = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """聊天：支持 Idempotency-Key，防止客户端重试重复执行 Agent。"""
    request_id = new_request_id()
    identity = _identity_for(user, body.session_id)

    request_hash = None
    if idempotency_key is not None:
        try:
            idempotency_key = validate_idempotency_key(idempotency_key)
        except ValueError as exc:
            return error(422, "IDEMPOTENCY_KEY_INVALID", str(exc), request_id)
        request_hash = hash_request(
            {"session_id": identity.session_id, "message": body.message}
        )
        try:
            claim = await request.app.state.idempotency_store.begin(
                identity.user_id,
                identity.session_id,
                idempotency_key,
                request_hash,
                ttl_seconds=settings.idempotency_ttl_seconds,
            )
        except Exception:
            return error(
                503,
                "IDEMPOTENCY_UNAVAILABLE",
                "请求去重服务暂时不可用",
                request_id,
            )
        if claim.action == "conflict":
            return error(
                409,
                "IDEMPOTENCY_KEY_REUSED",
                "同一个 Idempotency-Key 不能用于不同请求",
                request_id,
            )
        if claim.action == "in_progress":
            response = error(
                409,
                "REQUEST_IN_PROGRESS",
                "相同请求正在处理中，请稍后重试",
                request_id,
            )
            response.headers["Retry-After"] = "1"
            return response
        if claim.action == "replay":
            response = JSONResponse(
                status_code=claim.status_code or 200,
                content=claim.response or {},
            )
            response.headers["X-Idempotent-Replay"] = "true"
            return response

    try:
        outcome = await manager.chat(identity, body.message)
    except SessionManagerError as exc:
        response = session_error(exc, request_id)
    except Exception:
        response = error(
            500,
            "CHAT_INTERNAL_ERROR",
            "服务内部错误，请稍后重试",
            request_id,
        )
    else:
        response = _chat_response(identity, outcome, request_id)

    if idempotency_key is not None and request_hash is not None:
        try:
            payload = json.loads(response.body.decode("utf-8"))
            await request.app.state.idempotency_store.finish(
                identity.user_id,
                identity.session_id,
                idempotency_key,
                request_hash,
                status="succeeded" if response.status_code < 400 else "failed",
                status_code=response.status_code,
                response=payload,
            )
        except Exception:
            # Agent 已经执行完成，不能因为幂等账本写失败而再次执行；
            # 对客户端保留本次结果，并明确标注账本未写入。
            response.headers["X-Idempotency-Persisted"] = "false"
    return response


def _identity_for(user: AnonymousIdentity, session_id: str) -> SessionIdentity:
    """凭证 user_id + 请求方窗口编号组身份；UUID 非法 → 422。

    归属隔离由 SessionManager 的文件定位天然保证：窗口文件只会在
    当前用户目录下查找，别人的窗口对本用户一律不存在（统一 404）。
    """
    try:
        return SessionIdentity(user_id=user.user_id, session_id=session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="窗口编号必须是 UUID") from exc


def _chat_response(identity: SessionIdentity, outcome: ChatOutcome,
                   request_id: str):
    """ChatOutcome → HTTP（§4.5）：

    - 影响答案的失败 → 503，data 仍带已生成的兜底 reply；
    - 仅附属失败（STM/落盘）→ 200，warnings 给稳定 code（不伪装成功保存）。
    """
    data = {
        "session_id": identity.session_id,
        "reply": outcome.reply,
        "warnings": list(outcome.warnings),
    }
    if outcome.failed:
        return error(503, "CHAT_FAILED",
                     "本轮回答生成失败，请稍后重试", request_id, data)
    if not outcome.persisted:
        data["warnings"] = [*data["warnings"], "PERSISTENCE_FAILED"]
    return success(data, request_id)


# ---------- 商品与收藏（公共目录；async 工具直调，PG 异步池自带连接上限） ----------

@products_router.get("/search")
async def search_products_route(
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
    mode: str = Query("semantic", pattern="^(semantic|catalog)$"),
    query: str | None = Query(None, max_length=500),
    category: str | None = Query(None),
    brand: str | None = Query(None),
    min_price: int | None = Query(None, ge=0),
    max_price: int | None = Query(None, ge=0),
    limit: int = Query(5, ge=1, le=20),
):
    """商品搜索：mode=semantic 语义检索；mode=catalog 结构化筛选。"""
    request_id = new_request_id()
    if min_price is not None and max_price is not None and min_price > max_price:
        return error(422, "PRICE_RANGE_INVALID", "min_price 不能大于 max_price",
                     request_id)
    if mode == "semantic":
        if not query or not query.strip():
            return error(422, "QUERY_REQUIRED", "语义搜索必须提供 query", request_id)
        if brand:
            return error(422, "BRAND_NOT_SUPPORTED",
                         "语义搜索不支持品牌参数，请改用 mode=catalog", request_id)
        result = await search_products(query, category,
                                   min_price, max_price, limit)
    else:
        result = await search_catalog(query, brand,
                                  category, min_price, max_price, limit)
    return _product_response(result, request_id)


@products_router.get("/{product_id}")
async def product_detail(
    product_id: str,
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
):
    """商品完整详情；不存在 404，数据库不可用 503。"""
    request_id = new_request_id()
    if not product_id or "/" in product_id or "\\" in product_id:
        return error(422, "PRODUCT_ID_INVALID", "商品编号不合法", request_id)
    result = await get_detail(product_id)
    return _detail_response(result, request_id)


@products_router.post("/compare")
async def compare_products_route(
    body: CompareRequest,
    request: Request,
    user: AnonymousIdentity = Depends(require_csrf),
):
    """一次对比 2~4 个不同商品；缺失 ID 明确失败（422）。"""
    request_id = new_request_id()
    result = await compare_products(body.product_ids)
    if not result["success"]:
        data = result.get("data", {})
        if data.get("unavailable"):
            return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                         "商品服务暂时不可用", request_id)
        if data.get("missing_ids"):
            return error(422, "PRODUCT_MISSING_IDS",
                         "部分商品不存在", request_id, data)
        return error(422, "COMPARE_REQUEST_INVALID",
                     result.get("error", "对比请求不合法"), request_id)
    return success(result["data"], request_id)


def _detail_response(result: dict, request_id: str):
    if result["success"]:
        return success(result["data"], request_id)
    if result.get("data", {}).get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "商品服务暂时不可用", request_id)
    return error(404, "PRODUCT_NOT_FOUND", "商品不存在", request_id)


def _product_response(result: dict, request_id: str):
    """商品搜索工具信封 → HTTP（§4.5：按 data 内稳定标志判断，不匹配文案）。"""
    if result["success"]:
        return success(result["data"], request_id)
    data = result.get("data", {})
    if data.get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "商品服务暂时不可用", request_id)
    # 无匹配（含 nearest 兜底）是 200 + 空结果：matched=0 与参考标记原样透传
    return success(data, request_id)


@favorites_router.get("")
async def list_favorites(
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
    category: str | None = Query(None),
    limit: int = Query(20, ge=1, le=50),
):
    """当前用户收藏；空列表是正常的 200 响应。"""
    request_id = new_request_id()
    result = await get_user_favorites(
        user_id=user.user_id, category=category, limit=limit,
        database_url=settings.database_url,
    )
    if not result["success"] and result.get("data", {}).get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "收藏服务暂时不可用", request_id)
    # 收藏为空是 200 空列表（工具以 fail 表达空收藏），data 原样透传
    return success(result.get("data", {"favorites": [], "total": 0}), request_id)


# ---------- 健康检查 ----------

@health_router.get("/health")
async def health():
    """存活检查：进程和事件循环能响应即 200，不访问外部服务。"""
    return {"status": "ok"}


@health_router.get("/ready")
async def ready(request: Request):
    """接流量判断：核心依赖可用 200，降级/不可用 503（探测带短缓存）。"""
    status = await request.app.state.dependency_probe.snapshot()
    return JSONResponse(status_code=200 if status["ready"] else 503,
                        content=status)
