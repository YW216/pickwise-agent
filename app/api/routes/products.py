"""商品路由：搜索、详情与参数对比。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from app.agent.tools.catalog import search_catalog
from app.agent.tools.detail import compare_products, get_detail
from app.agent.tools.search_products import search_products
from app.api.auth import AuthenticatedUser
from app.api.deps import require_authenticated_user, require_csrf
from app.api.envelope import error, new_request_id, success
from app.api.product_responses import detail_response, product_response, public_compare_data
from app.api.schemas import CompareRequest
from app.identity import SessionIdentity
from app.services.product_refs import resolve_product_ref


products_router = APIRouter(prefix="/api/products", tags=["products"])


@products_router.get("/search")
async def search_products_route(
    request: Request,
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
    return product_response(result, request_id)


@products_router.get("/{product_id}")
async def product_detail(
    product_id: str,
    request: Request,
):
    """商品完整详情；不存在 404，数据库不可用 503。"""
    request_id = new_request_id()
    if not product_id or "/" in product_id or "\\" in product_id:
        return error(422, "PRODUCT_ID_INVALID", "商品编号不合法", request_id)
    result = await get_detail(product_id)
    return detail_response(result, request_id)


@products_router.get("/ref/{product_ref}")
async def product_ref_detail(
    product_ref: str,
    session_id: str = Query(..., min_length=36, max_length=36),
    user: AuthenticatedUser = Depends(require_authenticated_user),
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
    return detail_response(await get_detail(product_id), request_id)


@products_router.post("/compare")
async def compare_products_route(
    body: CompareRequest,
    request: Request,
    user: AuthenticatedUser = Depends(require_csrf),
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
    return success(public_compare_data(result["data"]), request_id)
