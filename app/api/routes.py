"""HTTP 路由：参数解析、身份验证、窗口归属、调用后端、映射响应（§4.1）。

路由不写业务逻辑：聊天走 SessionManager 队列，商品/收藏走有界执行器里的
工具函数（与 Agent 共用同一注册表路径）。所有响应统一信封（§4.5）。
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.agent.tools.catalog import search_catalog
from app.agent.tools.detail import compare_products, get_detail
from app.agent.tools.favorites import get_user_favorites
from app.agent.tools.search_products import search_products
from app.api.auth import (
    COOKIE_NAME,
    AnonymousIdentity,
    AuthError,
)
from app.api.deps import (
    get_auth,
    get_current_user,
    get_session_manager,
    get_session_service,
    require_csrf,
)
from app.api.envelope import (
    clear_auth_cookie,
    error,
    new_request_id,
    session_error,
    success,
)
from app.api.schemas import ChatRequest, CompareRequest, SessionCreateRequest
from app.config.settings import settings
from app.db.catalog_repo import aget_products_by_ids
from app.identity import SessionIdentity
from app.services.idempotency import hash_request, validate_idempotency_key
from app.services.product_refs import resolve_product_context, resolve_product_ref
from app.services.recommendations import public_cards, restore_cards
from app.services.session_store import visible_messages
from app.services.session_manager import (
    ChatOutcome,
    SessionManager,
    SessionManagerError,
)
from app.services.session_service import SessionService, SessionServiceNotFound

sessions_router = APIRouter(prefix="/api/sessions", tags=["sessions"])
chat_router = APIRouter(prefix="/api/chat", tags=["chat"])
auth_router = APIRouter(prefix="/api/auth", tags=["auth"])
products_router = APIRouter(prefix="/api/products", tags=["products"])
favorites_router = APIRouter(prefix="/api/favorites", tags=["favorites"])
health_router = APIRouter(prefix="/api", tags=["health"])

_INTERNAL_PRODUCT_ID = re.compile(r"(?<![A-Za-z0-9])(?:LP|PH|HP)-\d{2}(?!\d)")
_PRODUCT_ID_PARENTHESES = re.compile(
    r"\s*[（(]\s*(?:LP|PH|HP)-\d{2}\s*[）)]"
)
_PRODUCT_ID_PAREN_PREFIX = re.compile(
    r"([（(])\s*(?:LP|PH|HP)-\d{2}\s*[，,、]?\s*"
)


def _public_assistant_text(content: str) -> str:
    """隐藏内部商品主键；名称、价格和参数仍保持原样。"""
    without_parentheses = _PRODUCT_ID_PARENTHESES.sub("", content)
    without_prefix = _PRODUCT_ID_PAREN_PREFIX.sub(r"\1", without_parentheses)
    return _INTERNAL_PRODUCT_ID.sub("候选商品", without_prefix)


def _public_message_content(role: str, content: str) -> str:
    return _public_assistant_text(content) if role == "assistant" else content

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
    service = getattr(request.app.state, "session_service", None)
    if service is not None:
        state = await service.create_session(user.user_id, title=body.title)
        payload = {
            "session_id": state.session_id,
            "user_id": state.user_id,
            "title": state.title,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }
    else:
        identity = await manager.create_session(user.user_id)
        payload = {
            "session_id": identity.session_id,
            "user_id": identity.user_id,
            "title": body.title,
        }
    return success(
        payload,
        request_id, status_code=201,
    )


@sessions_router.get("")
async def list_sessions(
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
    service: SessionService = Depends(get_session_service),
):
    """返回当前用户的会话摘要列表。"""
    request_id = new_request_id()
    rows = await service.list_sessions(user.user_id)
    return success(
        {
            "sessions": [
                {
                    "session_id": row.session_id,
                    "title": row.title,
                    "status": row.status,
                    "message_count": row.message_count,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                    "last_message_at": (
                        row.last_message_at.isoformat()
                        if row.last_message_at else None
                    ),
                }
                for row in rows
            ],
        },
        request_id,
    )


async def _with_catalog_prices(cards: list[dict]) -> list[dict]:
    """历史卡片缺价时回查目录：快照记的是"当时写了什么"，价格要能补上。"""
    missing = [
        card["product_id"] for card in cards
        if card.get("product_id") and card.get("price") in (None, "", 0)
    ]
    if not missing:
        return cards
    try:
        products = await aget_products_by_ids(settings.database_url, missing)
    except Exception:
        return cards
    for card in cards:
        if card.get("price") not in (None, "", 0):
            continue
        product = products.get(card.get("product_id"))
        if product and product.get("price") is not None:
            card["price"] = product["price"]
    return cards


@sessions_router.get("/{session_id}/messages")
async def session_messages(
    session_id: str,
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
    service: SessionService = Depends(get_session_service),
):
    """返回当前用户可见的历史消息与右侧推荐卡片，不暴露 tool trace。"""
    request_id = new_request_id()
    try:
        state = await service.get_session(user.user_id, session_id)
    except SessionServiceNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    messages = visible_messages(state.messages, session_id=state.session_id)
    recommendations = await _with_catalog_prices(list(state.recommendations))
    return success(
        {
            "session_id": session_id,
            "messages": [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": _public_message_content(message.role, message.content),
                    "created_at": (
                        message.created_at.isoformat()
                        if message.created_at else None
                    ),
                }
                for message in messages
            ],
            "recommendations": public_cards(restore_cards(
                recommendations,
                user_id=user.user_id,
                session_id=state.session_id,
            )),
        },
        request_id,
    )


@sessions_router.get("/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    user: AnonymousIdentity = Depends(get_current_user),
    service: SessionService = Depends(get_session_service),
):
    """返回当前用户的会话元数据和可见消息。"""
    request_id = new_request_id()
    try:
        state = await service.get_session(user.user_id, session_id)
    except SessionServiceNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    return success(
        {
            "session_id": state.session_id,
            "title": state.title,
            "summary": state.summary,
            "version": state.version,
            "status": state.status,
            "messages": [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": _public_message_content(message.role, message.content),
                    "created_at": (
                        message.created_at.isoformat()
                        if message.created_at else None
                    ),
                }
                for message in await service.get_visible_messages(
                    user.user_id, session_id,
                )
            ],
        },
        request_id,
    )


@sessions_router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    user: AnonymousIdentity = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
):
    """删除当前用户的会话快照。"""
    request_id = new_request_id()
    try:
        identity = _identity_for(user, session_id)
        await manager.delete_session(identity)
    except SessionServiceNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    except SessionNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    except SessionBusy:
        return error(409, "SESSION_BUSY", "会话正在处理中，请稍后重试", request_id)
    return success({"session_id": session_id, "deleted": True}, request_id)


@chat_router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    user: AnonymousIdentity = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """聊天：支持 Idempotency-Key，防止客户端重试重复执行 Agent。"""
    return await _handle_chat(body, request, user, manager, idempotency_key)


@chat_router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    user: AnonymousIdentity = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """过程实时 SSE；答案生成并保存后分段发送，不是模型 Token 流式。"""
    return await _handle_chat(body, request, user, manager, idempotency_key, stream=True)


async def _handle_chat(body, request, user, manager, idempotency_key, *, stream=False):
    """两个 HTTP 入口共用鉴权后的幂等检查、执行与结果映射。"""
    request_id = new_request_id()
    identity = _identity_for(user, body.session_id)
    product_context = []
    for ref in body.product_refs:
        item = resolve_product_context(identity.user_id, identity.session_id, ref)
        if item is None:
            return error(422, "PRODUCT_REF_INVALID", "推荐商品已失效，请重新发起推荐", request_id)
        product_context.append(item)

    request_hash = None
    if idempotency_key is not None:
        try:
            idempotency_key = validate_idempotency_key(idempotency_key)
        except ValueError as exc:
            return error(422, "IDEMPOTENCY_KEY_INVALID", str(exc), request_id)
        request_hash = hash_request(
            {
                "session_id": identity.session_id,
                "message": body.message,
                "product_refs": body.product_refs,
            }
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
            if stream and response.status_code < 400:
                return StreamingResponse(
                    _answer_events(claim.response, replayed=True),
                    media_type="text/event-stream", headers=_SSE_HEADERS,
                )
            return response

    events = asyncio.Queue(maxsize=256) if stream else None
    execution = None
    if stream:
        try:
            execution = await manager.chat_stream(
                identity, body.message, events, product_context=product_context,
            )
        except SessionManagerError as exc:
            response = session_error(exc, request_id)
            return await _finish_chat(request, identity, idempotency_key, request_hash, response)
        except Exception:
            response = error(503, "CHAT_UNAVAILABLE", "会话服务暂时不可用", request_id)
            return await _finish_chat(request, identity, idempotency_key, request_hash, response)

    async def run():
        try:
            outcome = await execution if execution is not None else await manager.chat(
                identity, body.message, product_context=product_context,
            )
        except SessionManagerError as exc:
            response = session_error(exc, request_id)
        except Exception:
            response = error(500, "CHAT_INTERNAL_ERROR", "服务内部错误，请稍后重试", request_id)
        else:
            response = _chat_response(identity, outcome, request_id)
        response = await _finish_chat(
            request, identity, idempotency_key, request_hash, response,
        )
        if stream:
            # 明确告诉 SSE 生成器：执行和幂等记账都结束了。
            while events.full():
                events.get_nowait()
            events.put_nowait(None)
        return response

    if not stream:
        return await run()

    # 独立任务保证浏览器断连只停止传输，生成、保存和幂等记账仍继续。
    task = asyncio.create_task(run())
    request.app.state.stream_tasks.add(task)
    task.add_done_callback(request.app.state.stream_tasks.discard)
    return StreamingResponse(
        _stream_events(events, task), media_type="text/event-stream", headers=_SSE_HEADERS,
    )


async def _finish_chat(request, identity, idempotency_key, request_hash, response):
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


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_HEARTBEAT_SECONDS = 15.0


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _answer_events(payload, *, replayed=False):
    data = payload.get("data") or {}
    reply = _public_assistant_text(data.get("reply", ""))
    data = {**data, "reply": reply}
    # 保留标点和换行；超长段落继续按 80 字切分，拼接必须与原文完全一致。
    chunks = re.findall(r"[^。！？\n]{1,80}[。！？\n]?|[。！？\n]", reply)
    for seq, text in enumerate(chunks):
        yield _sse("answer.delta", {"text": text, "seq": seq})
        await asyncio.sleep(0)
    if payload.get("success") is not True:
        failure = payload.get("error") or {}
        yield _sse("error", {**failure, "request_id": payload.get("request_id")})
    yield _sse("done", {**data, "request_id": payload.get("request_id"),
                        "failed": payload.get("success") is not True,
                        "replayed": replayed})


async def _stream_events(events, task):
    pending = asyncio.create_task(events.get())
    try:
        while True:
            ready, _ = await asyncio.wait(
                {pending, task}, timeout=_HEARTBEAT_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not ready:
                yield ": ping\n\n"
            elif pending in ready:
                event = pending.result()
                if event is None:
                    break
                yield _sse(event["event"], event["data"])
                pending = asyncio.create_task(events.get())
        response = task.result()
        payload = json.loads(response.body)
        if response.headers.get("X-Idempotency-Persisted") == "false":
            payload.setdefault("data", {}).setdefault("warnings", []).append("IDEMPOTENCY_PERSISTENCE_FAILED")
        async for frame in _answer_events(payload):
            yield frame
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        # 不取消 task：它还需要保存结果；事件队列有界，断连不会无限堆积。


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
        "reply": _public_assistant_text(outcome.reply),
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


@products_router.get("/ref/{product_ref}")
async def product_ref_detail(
    product_ref: str,
    session_id: str = Query(..., min_length=36, max_length=36),
    user: AnonymousIdentity = Depends(get_current_user),
):
    """使用当前会话推荐引用查询详情，不把内部 product_id 暴露给前端。"""
    request_id = new_request_id()
    try:
        identity = SessionIdentity(user_id=user.user_id, session_id=session_id)
    except ValueError:
        return error(422, "SESSION_ID_INVALID", "会话编号不合法", request_id)
    product_id = resolve_product_ref(identity.user_id, identity.session_id, product_ref)
    if product_id is None:
        return error(404, "PRODUCT_REF_NOT_FOUND", "推荐商品已失效，请重新发起推荐", request_id)
    return _detail_response(await get_detail(product_id), request_id)


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
    return success(_public_compare_data(result["data"]), request_id)


def _detail_response(result: dict, request_id: str):
    if result["success"]:
        data = dict(result["data"])
        product = data.get("product")
        if isinstance(product, dict):
            data["product"] = {
                key: value for key, value in product.items()
                if key != "product_id"
            }
        return success(data, request_id)
    if result.get("data", {}).get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "商品服务暂时不可用", request_id)
    return error(404, "PRODUCT_NOT_FOUND", "商品不存在", request_id)


def _product_response(result: dict, request_id: str):
    """商品搜索工具信封 → HTTP（§4.5：按 data 内稳定标志判断，不匹配文案）。"""
    if result["success"]:
        return success(_public_product_data(result["data"]), request_id)
    data = result.get("data", {})
    if data.get("unavailable"):
        return error(503, "PRODUCT_SERVICE_UNAVAILABLE",
                     "商品服务暂时不可用", request_id)
    # 无匹配（含 nearest 兜底）是 200 + 空结果：matched=0 与参考标记原样透传
    return success(_public_product_data(data), request_id)


def _public_product_data(data: dict) -> dict:
    """将内部商品卡片转换为用户展示卡片。

    product_id 只在 Agent/数据库内部流转；HTTP 返回候选序号，避免把
    内部目录主键暴露给浏览器。工具层仍使用原始数据，不影响详情查询。
    """
    payload = dict(data or {})
    for key in ("results", "candidates", "nearest"):
        items = payload.get(key)
        if not isinstance(items, list):
            continue
        public_items = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                public_items.append(item)
                continue
            public_item = dict(item)
            public_item.pop("product_id", None)
            public_item["candidate_no"] = index
            public_items.append(public_item)
        payload[key] = public_items
    return payload


def _public_compare_data(data: dict) -> dict:
    """隐藏 compare 响应中的主键，并用候选序号对齐字段值。"""
    payload = dict(data or {})
    products = payload.get("products")
    if not isinstance(products, list):
        return payload
    refs: dict[str, str] = {}
    public_products = []
    for index, item in enumerate(products, start=1):
        if not isinstance(item, dict):
            public_products.append(item)
            continue
        product_id = item.get("product_id")
        if isinstance(product_id, str):
            refs[product_id] = f"候选 {index}"
        public_item = {
            key: value for key, value in item.items()
            if key != "product_id"
        }
        public_item["candidate_no"] = index
        public_products.append(public_item)
    payload["products"] = public_products
    fields = payload.get("fields")
    if isinstance(fields, list):
        public_fields = []
        for field in fields:
            if not isinstance(field, dict):
                public_fields.append(field)
                continue
            public_field = dict(field)
            values = field.get("values")
            if isinstance(values, dict):
                public_field["values"] = {
                    refs.get(key, "候选商品"): value
                    for key, value in values.items()
                }
            public_fields.append(public_field)
        payload["fields"] = public_fields
    return payload


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
