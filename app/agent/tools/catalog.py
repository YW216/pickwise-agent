"""search_catalog：商品目录结构化检索（导购 / 比选）。

**SQL 直查版（2026-09-15）**：参数与 PG products 表列一一对应
（query=商品名 ILIKE 字面、brand=品牌精确、category 精确、price 数值上限），
确定性过滤在数据库执行——PG 数据变更即时可见，无需重启进程。PG 不可达时
回退本地内存过滤（同一套条件的 Python 实现，行为一致）。

可搜索字段边界（确定且可枚举）：
- 商品名：字面子串（query，ILIKE 大小写不敏感）
- 品牌：精确匹配（brand 参数）
- 品类：精确；预算：数值上限
specs 参数值与介绍文案**不参与**本工具匹配——参数条件（「16GB」「独显」）
与语义/模糊需求由 search_products（Milvus hybrid）承担。

防幻觉设计（2026-09-01）：严格条件查无结果时，返回「条件外最接近的候选」
（nearest，预算外最接近预算优先取 3 款）——给模型一个可引用的数据兜底，
消除编造动机。
"""

from app.agent.tools.result import fail, ok
from app.config.settings import settings

# 候选卡片的规模提示：模型看到候选清单时，据此判断"是否值得逐款拉详情"。
# 刻意不叫 notice——notice 是执行器层（registry）在参数被归一化/夹紧时的专用字段，
# 业务层占用会与那层语义冲突。此处描述"这批卡片够用/不够用"，属业务信息。
#
# 两条 hint 按「结果是否被 limit 截断」二选一：
# - 未截断（matched <= returned）：卡片即全部，提醒别逐款拉详情
# - 已截断（matched > returned）：告知真实总量，请用户补充条件缩小范围
_CARDS_HINT = (
    "以上候选卡片已含名称/品牌/价格/定位，足以向用户展示可选范围。"
    "不要为罗列清单而逐款查详情；仅当用户指名少数几款需要比对具体参数时才逐一查询。"
)
_TRUNCATED_HINT = (
    "符合条件的结果超过单次返回上限，以上仅为其中一部分。"
    "请如实告知用户符合条件共 {matched} 款、此处只列出前 {returned} 款，"
    "并请其补充品牌/预算/品类条件以缩小范围；不要逐款查询详情。"
)


def brief_view(product: dict) -> dict:
    """候选精简视图（不含完整 specs）。

    公开函数：search_products 等语义工具命中商品后复用同一张卡片形状，
    保证两个搜索工具返回结构一致，Agent 合并结果零成本。
    summary 取 introduction 首段（定位句，截断展示）。
    """
    return {
        "product_id": product["product_id"],
        "name": product["name"],
        "brand": product["brand"],
        "category": product["category"],
        "price": product["price"],
        "summary": product["introduction"][0][:64],
    }



def search_catalog(
    query: str | None = None,
    brand: str | None = None,
    category: str | None = None,
    budget_max: float | None = None,
    limit: int = 10,
) -> dict:
    """按字段对齐的确定性条件检索商品目录（SQL 直查，PG 不可达回退本地）。

    Args:
        query：品牌/型号字面关键词（如「墨白」「凌霄」）——仅匹配商品名与品牌
        brand：品牌精确匹配（封闭集合：星海/曜石/云章/墨白/极光）
        category：品类（笔记本 / 手机 / 耳机）
        budget_max：预算上限（价格低于等于该值）
        limit：返回条数

    Returns:
        成功：data = {"candidates": [...], "total": N}
        查无结果但有关键条件外的接近候选：
              fail("没有完全符合条件的产品，以下是条件外最接近的几款（供参考）",
                   data = {"candidates": [], "total": 0, "nearest": [...]})
        彻底无匹配：fail("没有符合条件的商品，请调整预算或条件")
    """
    # ---------- PG 直查（首选：数据即时可见、确定性由 SQL 保证） ----------
    if settings.database_url:
        try:
            from app.db.catalog_repo import search_candidates, search_nearest

            res = search_candidates(settings.database_url, query, brand,
                                    category, budget_max, limit)
            if res["total"]:
                returned, matched = res["total"], res.get("matched", res["total"])
                hint = (
                    _TRUNCATED_HINT.format(matched=matched, returned=returned)
                    if matched > returned else _CARDS_HINT
                )
                return ok({**res, "hint": hint})
            near = search_nearest(settings.database_url, query, brand,
                                  category, budget_max, 3)
            if near["candidates"]:
                return fail(
                    "没有完全符合条件的产品，以下是条件外最接近的几款（供参考）",
                    {"candidates": [], "total": 0, "nearest": near["candidates"]},
                )
            return fail(
                "没有符合条件的商品，请调整预算或条件",
                {"candidates": [], "total": 0},
            )
        except Exception as exc:  # noqa: BLE001 —— PG 唯一数据源：失败显式 fail（不静默回退旧快照）
            print(f"[search_catalog] 商品服务不可用: {type(exc).__name__}: {exc}")
            return fail("商品服务暂时不可用，请稍后再试", {"candidates": [], "total": 0})

    # 未配置 database_url 时无数据源可用：显式失败，不引用未定义的 res
    return fail("商品服务暂未配置，请稍后再试", {"candidates": [], "total": 0})
