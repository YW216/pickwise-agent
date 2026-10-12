"""get_user_favorites：用户收藏候选池（导购 / 比选）。

身份由 ToolManager 绑定，收藏和商品卡片从 PostgreSQL 查询。
返回结构与 search_catalog 一致，便于下游（推荐/对比）复用同一套处理。
"""

from app.db.user_repo import get_favorites
from app.agent.tools.result import fail, ok


def get_user_favorites(category: str | None = None, limit: int = 20, *,
                       user_id: str | None = None, database_url: str | None = None) -> dict:
    """查询当前用户收藏的商品。

    Args:
        category：可选品类过滤（笔记本 / 手机 / 耳机）
        limit：返回条数

    Returns:
        成功：data = {"favorites": [...], "total": N}
        收藏为空：fail("您的收藏夹还没有商品")
    """
    if user_id is None:
        return fail("缺少用户收藏上下文")
    if not database_url:
        return fail("收藏服务暂未配置")
    try:
        items = get_favorites(database_url, user_id, category=category, limit=limit)
    except Exception:
        return fail("收藏服务暂时不可用，请稍后再试")
    if not items:
        return fail("您的收藏夹还没有商品", {"favorites": [], "total": 0})

    return ok({"favorites": items, "total": len(items)})
