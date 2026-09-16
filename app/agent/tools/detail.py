"""详情与对比工具（比选 / 咨询）：

- get_detail：单商品完整详情（含价格与全量参数）
- compare_products：多品同参数对比（2-4 款，按"字段 → 各商品值"组织）
"""

from app.db.snapshot import PRODUCTS
from app.agent.tools.result import fail, ok


def get_detail(product_id: str) -> dict:
    """查询单个商品的完整详情（含价格与全部参数）。

    Args:
        product_id：商品 ID

    Returns:
        成功：data = {"product": {...}}（含 specs 全量参数）
        商品不存在：fail("商品不存在")
    """
    try:
        product = PRODUCTS.get(product_id)
    except Exception:
        return fail("商品数据加载失败")

    if not product:
        return fail("商品不存在")

    return ok({"product": product})


def compare_products(product_ids: list[str]) -> dict:
    """对比多个商品的同维度参数。

    Args:
        product_ids：要对比的商品 ID 列表（2-4 个）

    Returns:
        成功：data = {"products": [...], "fields": [{field, values: {pid: value}}]}
        商品不足：fail("部分商品不存在，请核对商品名称")
    """
    try:
        products = [PRODUCTS.get(pid) for pid in (product_ids or [])]
    except Exception:
        return fail("商品数据加载失败")

    found = [p for p in products if p]
    if len(found) < 2:
        return fail("部分商品不存在，请核对商品名称")

    # 价格作为首个对比字段
    fields: list[dict] = [
        {"field": "价格", "values": {p["product_id"]: p["price"] for p in found}}
    ]

    # 汇总所有商品出现过的参数名（保序去重）
    spec_keys: list[str] = []
    for product in found:
        for key in product["specs"]:
            if key not in spec_keys:
                spec_keys.append(key)

    for key in spec_keys:
        values = {p["product_id"]: p["specs"].get(key, "-") for p in found}
        if any(v != "-" for v in values.values()):
            fields.append({"field": key, "values": values})

    return ok(
        {
            "products": [
                {"product_id": p["product_id"], "name": p["name"]} for p in found
            ],
            "fields": fields,
        }
    )
