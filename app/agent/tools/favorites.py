"""get_user_favorites：用户收藏候选池（导购 / 比选）。

单用户假设：全局一张收藏表，存 product_id 列表。
返回结构与 search_catalog 一致，便于下游（推荐/对比）复用同一套处理。
"""

from app.agent.tools.catalog import brief_view
from app.db.snapshot import FAVORITES, PRODUCTS
from app.agent.tools.result import fail, ok


def get_user_favorites(category: str | None = None, limit: int = 20) -> dict:
    """查询当前用户收藏的商品。

    Args:
        category：可选品类过滤（笔记本 / 手机 / 耳机）
        limit：返回条数

    Returns:
        成功：data = {"favorites": [...], "total": N}
        收藏为空：fail("您的收藏夹还没有商品")
    """
    try:
        favorite_ids = list(FAVORITES)
    except Exception:
        return fail("收藏数据加载失败")

    items = []
    for product_id in favorite_ids:
        product = PRODUCTS.get(product_id)
        if not product:
            continue
        if category and product["category"] != category:
            continue
        items.append(brief_view(product))

    if not items:
        return fail("您的收藏夹还没有商品", {"favorites": [], "total": 0})

    picked = items[:limit]
    return ok({"favorites": picked, "total": len(picked)})
