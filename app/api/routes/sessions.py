"""会话路由：创建、列表、历史消息与删除。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.auth import AuthenticatedUser
from app.api.deps import (
    get_session_manager,
    get_session_service,
    identity_for,
    require_authenticated_user,
    require_csrf,
)
from app.api.envelope import error, new_request_id, success
from app.api.public_text import public_message_content
from app.api.schemas import SessionCreateRequest
from app.config.settings import settings
from app.db.catalog_repo import aget_products_by_ids
from app.services.product_images import category_image
from app.services.recommendations import (
    build_recommendations,
    merge_recommendations,
    public_cards,
    restore_cards,
)
from app.services.session_manager import SessionBusy, SessionManager, SessionNotFound
from app.services.session_service import SessionService, SessionServiceNotFound
from app.services.session_store import visible_messages


sessions_router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@sessions_router.post("")
async def create_session(
    body: SessionCreateRequest,
    request: Request,
    user: AuthenticatedUser = Depends(require_csrf),
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
    user: AuthenticatedUser = Depends(require_authenticated_user),
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


async def _refresh_recommendations(state, user_id: str, session_id: str) -> list[dict]:
    """用最后一轮回复重算卡片，与快照合并后返回。

    快照可能落后于展示口径（例如早期版本截断到 8 款）；读取时按当前规则
    重算一次，既补齐漏掉的商品，也保留历史轮次出现过的卡片。
    编号通过 restore_cards 预先注册，重算不会打乱用户已经记住的 #01/#02。
    """
    cards = restore_cards(state.recommendations, user_id=user_id, session_id=session_id)
    reply = ""
    for message in reversed(state.messages):
        if message.role == "assistant" and str(message.content or "").strip():
            reply = str(message.content)
            break
    if not reply:
        return cards
    tools = [
        {"role": "tool", "content": message.content}
        for message in state.messages if message.role == "tool"
    ]
    fresh = await build_recommendations(
        reply, tools,
        user_id=user_id,
        session_id=session_id,
        database_url=settings.database_url,
    )
    return merge_recommendations(fresh, cards) if fresh else cards


async def _with_catalog_prices(cards: list[dict]) -> list[dict]:
    """历史卡片补齐：缺图按品类补，缺价回查目录。

    快照记的是"当时写了什么"，图与价都是可推导的展示字段，
    读取时补上，历史会话也能和当前 UI 一致。
    """
    for card in cards:
        if not card.get("image"):
            card["image"] = category_image(str(card.get("category") or ""))
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
    user: AuthenticatedUser = Depends(require_authenticated_user),
    service: SessionService = Depends(get_session_service),
):
    """返回当前用户可见的历史消息与右侧推荐卡片，不暴露 tool trace。"""
    request_id = new_request_id()
    try:
        state = await service.get_session(user.user_id, session_id)
    except SessionServiceNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    messages = visible_messages(state.messages, session_id=state.session_id)
    recommendations = await _with_catalog_prices(
        await _refresh_recommendations(state, user.user_id, state.session_id)
    )
    return success(
        {
            "session_id": session_id,
            "messages": [
                {
                    "message_id": message.message_id,
                    "role": message.role,
                    "content": public_message_content(message.role, message.content),
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
    user: AuthenticatedUser = Depends(require_authenticated_user),
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
                    "content": public_message_content(message.role, message.content),
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
    user: AuthenticatedUser = Depends(require_csrf),
    manager: SessionManager = Depends(get_session_manager),
):
    """删除当前用户的会话快照。"""
    request_id = new_request_id()
    try:
        identity = identity_for(user, session_id)
        await manager.delete_session(identity)
    except SessionServiceNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    except SessionNotFound:
        return error(404, "SESSION_NOT_FOUND", "会话不存在", request_id)
    except SessionBusy:
        return error(409, "SESSION_BUSY", "会话正在处理中，请稍后重试", request_id)
    return success({"session_id": session_id, "deleted": True}, request_id)
