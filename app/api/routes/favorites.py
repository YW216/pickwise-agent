"""收藏路由：列表、详情、添加与删除。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.agent.tools.detail import get_detail
from app.agent.tools.favorites import get_user_favorites
from app.api.auth import AuthenticatedUser
from app.api.deps import require_authenticated_user, require_csrf
from app.api.envelope import error, new_request_id, success
from app.api.product_responses import detail_response
from app.api.schemas import SessionIdRequest
from app.config.settings import settings
from app.db.user_repo import add_favorite, get_favorite_product_id, remove_favorite
from app.identity import SessionIdentity
from app.services.product_refs import resolve_product_ref


favorites_router = APIRouter(prefix="/api/favorites", tags=["favorites"])


@favorites_router.get("/item/{product_id}/detail")
async def favorite_detail(
    product_id: str,
    request: Request,
    user: AuthenticatedUser = Depends(require_authenticated_user),
):
    """查询当前用户收藏商品详情；未收藏的商品不能通过此入口查看。"""
    request_id = new_request_id()
    try:
        owned_product_id = await get_favorite_product_id(
            settings.database_url, user.user_id, product_id,
        )
    except Exception:
        return error(503, "FAVORITE_SERVICE_UNAVAILABLE", "收藏服务暂时不可用", request_id)
    if owned_product_id is None:
        return error(404, "FAVORITE_NOT_FOUND", "收藏商品不存在", request_id)
    return detail_response(await get_detail(owned_product_id, settings.database_url), request_id)


@favorites_router.get("")
async def list_favorites(
    request: Request,
    user: AuthenticatedUser = Depends(require_authenticated_user),
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
    # 收藏为空是 200 空列表；product_id 只作为前端操作参数，不在卡片上展示。
    data = result.get("data", {"favorites": [], "total": 0})
    return success(data, request_id)


@favorites_router.post("/ref/{product_ref}")
async def add_favorite_route(
    product_ref: str,
    body: SessionIdRequest,
    user: AuthenticatedUser = Depends(require_csrf),
):
    request_id = new_request_id()
    try:
        identity = SessionIdentity(user_id=user.user_id, session_id=body.session_id)
    except ValueError:
        return error(422, "SESSION_ID_INVALID", "会话编号不合法", request_id)
    product_id = resolve_product_ref(identity.user_id, identity.session_id, product_ref)
    if product_id is None:
        return error(404, "PRODUCT_REF_NOT_FOUND", "推荐商品已失效，请重新发起推荐", request_id)
    try:
        await add_favorite(settings.database_url, user.user_id, product_id)
    except Exception:
        return error(503, "FAVORITE_SERVICE_UNAVAILABLE", "收藏服务暂时不可用", request_id)
    return success({"favorited": True}, request_id)


@favorites_router.delete("/ref/{product_ref}")
async def remove_favorite_route(
    product_ref: str,
    body: SessionIdRequest,
    user: AuthenticatedUser = Depends(require_csrf),
):
    request_id = new_request_id()
    try:
        identity = SessionIdentity(user_id=user.user_id, session_id=body.session_id)
    except ValueError:
        return error(422, "SESSION_ID_INVALID", "会话编号不合法", request_id)
    product_id = resolve_product_ref(identity.user_id, identity.session_id, product_ref)
    if product_id is None:
        return error(404, "PRODUCT_REF_NOT_FOUND", "推荐商品已失效，请重新发起推荐", request_id)
    try:
        await remove_favorite(settings.database_url, user.user_id, product_id)
    except Exception:
        return error(503, "FAVORITE_SERVICE_UNAVAILABLE", "收藏服务暂时不可用", request_id)
    return success({"favorited": False}, request_id)


@favorites_router.delete("/item/{product_id}")
async def remove_favorite_item_route(
    product_id: str,
    user: AuthenticatedUser = Depends(require_csrf),
):
    request_id = new_request_id()
    try:
        removed = await remove_favorite(
            settings.database_url, user.user_id, product_id,
        )
    except Exception:
        return error(503, "FAVORITE_SERVICE_UNAVAILABLE", "收藏服务暂时不可用", request_id)
    if not removed:
        return error(404, "FAVORITE_NOT_FOUND", "收藏商品不存在", request_id)
    return success({"favorited": False}, request_id)
