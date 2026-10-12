"""详情与对比工具（比选 / 咨询）：

- get_detail：单商品完整详情（含价格与全量参数）
- compare_products：多品同参数对比（2-4 款，按"字段 → 各商品值"组织）

数据源（P2-1，2026-10-12）：按 ID 直查 PG（get_products_by_ids），价格与
参数永远当前值。旧实现读进程启动时加载的快照 PRODUCTS——PG 改价即过期，
且 import 本模块会连带触发快照全量加载；在线路径禁止读快照（快照仅供
离线任务：评测指标、索引构建、脚本）。PG 不可达时显式失败，不回退旧数据。
"""

from app.agent.tools.result import fail, ok
from app.config.settings import settings
from app.db.catalog_repo import get_products_by_ids


def get_detail(product_id: str, database_url: str | None = None) -> dict:
    """查询单个商品的完整详情（含价格与全部参数）。

    Args:
        product_id：商品 ID
        database_url：仅程序侧可传的数据库地址（模型 schema 不含此参数）。
            显式传入优先；缺省回退应用配置——服务于「直接函数调用」路径
            （CLI 脚本 / 评测），ToolManager 路径由 bind_tool_handlers
            焊死实例地址，永远显式传值。

    Returns:
        成功：data = {"product": {...}}（含 specs 全量参数，价格为 PG 当前值）
        商品不存在：fail("商品不存在")
        服务不可用：fail("商品服务暂时不可用，请稍后再试")
    """
    url = database_url or settings.database_url
    if not url:
        return fail("商品服务暂未配置，请稍后再试")
    try:
        products = get_products_by_ids(url, [product_id])
    except Exception:  # noqa: BLE001 —— PG 唯一数据源：失败显式 fail（不静默回退旧快照）
        return fail("商品服务暂时不可用，请稍后再试")

    product = products.get(product_id)
    if not product:
        return fail("商品不存在")

    return ok({"product": product})


def compare_products(product_ids: list[str], database_url: str | None = None) -> dict:
    """对比多个商品的同维度参数（2-4 款，一次传入全部 ID）。

    Args:
        product_ids：要对比的商品 ID 列表（重复 ID 自动去重，去重后须有 2-4 个）
        database_url：仅程序侧可传的数据库地址（模型 schema 不含此参数）

    Returns:
        成功：data = {"products": [...], "fields": [{field, values: {pid: value}}]}
              各商品按输入顺序排列；批量返回的 dict 不决定展示顺序
        缺商品：fail("部分商品不存在，请核对商品名称", data={"missing_ids": [...]})
              ——缺任何一款即失败并明确列出缺失 ID，不静默少比一款
        全是重复 ID（去重后不足 2 款）：fail("请提供至少两款不同商品进行对比")
    """
    # 去重保序：重复 ID 只比一次。schema 闸门（minItems/maxItems）拦的是
    # 原始数量，去重后可能跌破 2——这里必须再兜底一次
    ordered: list[str] = []
    for pid in product_ids or []:
        if pid not in ordered:
            ordered.append(pid)
    if len(ordered) < 2:
        return fail("请提供至少两款不同商品进行对比")

    url = database_url or settings.database_url
    if not url:
        return fail("商品服务暂未配置，请稍后再试")
    try:
        products_by_id = get_products_by_ids(url, ordered)
    except Exception:  # noqa: BLE001 —— 同上：显式 fail，不回退
        return fail("商品服务暂时不可用，请稍后再试")

    # 缺任何一款即失败：比选必须对齐全部请求项，给模型缺失明细供其重试或告知
    missing = [pid for pid in ordered if pid not in products_by_id]
    if missing:
        return fail("部分商品不存在，请核对商品名称", {"missing_ids": missing})
    products = [products_by_id[pid] for pid in ordered]

    # 价格作为首个对比字段
    fields: list[dict] = [
        {"field": "价格", "values": {p["product_id"]: p["price"] for p in products}}
    ]

    # 汇总所有商品出现过的参数名（保序去重）
    spec_keys: list[str] = []
    for product in products:
        for key in product["specs"]:
            if key not in spec_keys:
                spec_keys.append(key)

    for key in spec_keys:
        values = {p["product_id"]: p["specs"].get(key, "-") for p in products}
        if any(v != "-" for v in values.values()):
            fields.append({"field": key, "values": values})

    return ok(
        {
            "products": [
                {"product_id": p["product_id"], "name": p["name"]} for p in products
            ],
            "fields": fields,
        }
    )
